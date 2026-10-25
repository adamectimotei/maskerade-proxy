"""GLiNER detection stage — CPU-friendly alternative to Flair NER + LLM.

Runs in the pipeline's detector slot (maskerade/pipeline.py, _run_detectors),
in the full tier only; needs the [ner] extra. Switched by stages.gliner.

Why this exists
---------------
The original GPU implementation needs an RTX 4090: Flair ner-english-large on GPU
plus Qwen3:14b via Ollama. GLiNER is a bi-encoder NER model designed for
CPU inference with runtime-specified labels — you tell it what entity
types to look for at call time, no fine-tuning. That means it can cover
both the Flair NER stage (PER/LOC/ORG) AND part of what the LLM stage
does contextually (tenant IDs, internal hostnames, credentials), at
roughly 200ms on a CPU instead of tens of seconds on a GPU.

Two library backends are supported because the model families differ:

  backend="gliner2"  → `pip install gliner2`
                       fastino/gliner2-privacy-filter-PII-multi (205M)
                       API: GLiNER2.from_pretrained(...).extract_entities()

  backend="gliner"   → `pip install gliner`
                       knowledgator/gliner-pii-base-v1.0
                       urchade/gliner_multi_pii-v1
                       API: GLiNER.from_pretrained(...).predict_entities()

Output contract
---------------
Returns the exact same entity dict shape the rest of the pipeline uses:

    {"text", "entity_type", "start", "end", "score", "stage"}

`stage` is set to "ner" deliberately, so masker.py, validator.py and
orchestrator._remove_overlaps() treat these entities identically to Flair
output — same 3-char length floor, same overlap semantics, same leak
checking. An extra "detector": "gliner" key is added for benchmark
reporting only; nothing in the pipeline reads it.
"""
import re

# Module-level caches — lazily initialised on first use
_model = None
_model_key = None


# ──────────────────────────────────────────────────────────────
# Default labels
#
# GLiNER takes labels at inference time. Keep this list tight: more
# labels means more latency and more false positives, and any label
# that duplicates the regex stage's job loses to it.
#
# Dropped after the first 30-prompt run (~30 of 111 false positives
# between them, near-zero unique recall):
#   domain_name       → "global", "Node.js/Express", ".NET"
#   api_key           → "xxxxxx", "dGhlIHNhbXBsZSBub25jZQ"
#   customer_id       → "demo", "userId", "customerId"
#   account_number    → plain amounts: "124000", "55600", "62800"
#   internal_hostname → regex already owns this
#   project_codename  → "VC-LTIP-2024", "ENYAQ_CONNECT_2026"
#
# Override via config["gliner"]["labels"].
# ──────────────────────────────────────────────────────────────

DEFAULT_LABELS = [
    "person",
    "organization",
    "location",
    "email",
    "credential",
]

# `organization` is enabled and deliberately NOT word-shape gated.
#
# It is by far the largest single source of over-masks on heterogeneous
# public web traffic (4,716 on the HTTP benchmark corpus), and the model
# is not wrong about them: `GitHub`, `OWASP`, `Django REST framework`
# really are organisations. They are just not the client.
#
# It was disabled for one round. Measured against a gold set that
# actually contains client organisations, that cost 11 of 24 client
# identities — `<Client A> a.s.`, `<Bank B>`, `<Client C> Finance
# a.s.`, `<Bank D> a.s.`, plus tenant slugs such as `client-c-cz-prod`
# and `acme-medical-cz`, which identify both the client and their
# environment. A leaked client name is an incident; a masked `OWASP` is
# an annoyance. The label stays on.
#
# The word-shape gate is off for ORGANIZATION for the same reason: no
# cheap text-shape signal separates a vendor name from a client name.
# `<Client A> a.s.` and `Django REST framework` are both word-shaped,
# and the gate suppressed 14 of the 21 lost client names on its own, so
# enabling the label while leaving the gate on would recover only a
# third of them.
#
# The durable fix is a POSITION gate rather than a shape gate: suppress
# ORGANIZATION inside JSON keys, HTML attribute names, URL paths and JS
# identifiers, and keep it in value positions and prose. Vendor names
# overwhelmingly appear in structural positions, client names in value
# positions. That needs the span offsets this stage already computes.
#
# `phone_number` stays removed. It produced 202 over-masks and no
# catches: the model reads bare JSON floats (0.50302766) and
# epoch-millisecond timestamps as phone numbers, which corrupts numeric
# payloads. Real phone numbers are a fixed-shape identifier already
# owned by the regex stage and by Presidio's PhoneRecognizer.

# GLiNER label (lowercase) → pipeline entity_type (uppercase).
# Anything not listed falls through to .upper().replace(" ", "_").
LABEL_MAP = {
    "person": "PERSON",
    "per": "PERSON",
    "first_name": "PERSON",
    "last_name": "PERSON",
    "organization": "ORGANIZATION",
    "org": "ORGANIZATION",
    "company": "ORGANIZATION",
    "location": "LOCATION",
    "loc": "LOCATION",
    "address": "LOCATION",
    "email": "EMAIL_ADDRESS",
    "email_address": "EMAIL_ADDRESS",
    "phone_number": "PHONE_NUMBER",
    "phone": "PHONE_NUMBER",
    "credential": "CREDENTIAL",
    "password": "CREDENTIAL",
    "api_key": "API_KEY",
    "account_number": "ACCOUNT_NUMBER",
    "card_number": "CREDIT_CARD",
    "payment_card": "CREDIT_CARD",
    "iban": "IBAN_CODE",
}

# Types that must survive the generic short-string filters below.
# A 3-char hostname fragment is noise; a 3-char credential is not.
_LENGTH_EXEMPT = {"CREDENTIAL", "API_KEY", "ACCOUNT_NUMBER", "CREDIT_CARD"}


# ──────────────────────────────────────────────────────────────
# Word-shape suppression
#
# GLiNER is not wrong about most of its over-masks — it is asked "is this
# an organisation" and answers correctly for `GitHub`, `OWASP` and
# `Django REST framework`. The masks are still worthless, because a
# public vendor is not the client. The same happens for `credential`,
# where the model confidently tags i18n keys such as
# `invalidPasswordMinLengthMessage`.
#
# What separates those from a real secret is that they are made of
# English word segments and a secret is not. Two cheap language-level
# signals capture it without a dictionary, and they are deliberately the
# same ones stages/keyval.py uses for its bare-`key` value gate:
#
#   segment coverage  share of the string made of runs of >=2 same-case
#                     letters — `eventTypes` is ~100%, an opaque token
#                     much lower
#   phonotactics      English interleaves vowels; random tokens pile up
#                     consonants (max run <=4, vowel ratio >=0.25)
#   identifier marks  identifiers join segments with "." "_" "-" or a
#                     camelCase boundary. Without one the string is a
#                     single opaque run — AKIA..., base32 TOTP secrets —
#                     and must keep masking.
#
# Two gate strengths, chosen per label from what each label can lose:
#
#   strict (identifier)  requires all three. Used for CREDENTIAL, where
#                        the loose gate would swallow real secrets that
#                        happen to read as words (`pass1secret`,
#                        `M6F5HYKDS2LOI3OH6VNPAQLNVYYMMHPT`).
#   loose (word-shaped)  coverage only. Used for LOCATION, whose targets
#                        are proper nouns with no separators, so the
#                        strict gate barely fires. NOT used for
#                        ORGANIZATION — see DEFAULT_LABELS.
#
# LOCATION additionally keeps anything containing a digit: a street
# address carries a house number (`Masarykova 12`, `1 High Street`),
# which is what tells it apart from a bare place name. Without that
# exemption the loose gate removes real addresses.
#
# PERSON and USERNAME are never gated. Human names ARE word-shaped —
# suppressing them would delete exactly what those labels exist to find.
# ──────────────────────────────────────────────────────────────

_VOWELS = frozenset("aeiouyAEIOUY")
_WORD_SEGMENT = re.compile(r'[A-Z]?[a-z]{2,}|[A-Z]{2,}')
_IDENT_SEPARATOR = re.compile(r'[._\-]')
_CAMEL_BOUNDARY = re.compile(r'[a-z][A-Z]')

_SEGMENT_COVERAGE_MIN = 0.75
# English permits 5-consonant clusters across syllable boundaries —
# "length", "strength" — and i18n keys are full of them
# (invalidPasswordMinLengthMessage). stages/keyval.py uses 4 for its
# bare-`key` value gate, where the tighter bound protects two more real
# secrets; here 5 costs nothing measurable and catches the compounds.
_CONSONANT_RUN_MAX = 5
_VOWEL_RATIO_MIN = 0.25

# Longest span we bother analysing. Beyond this the shape tests say
# nothing useful and the regex work is wasted.
_SHAPE_MAX_CHARS = 200

# entity_type -> gate strength. Absent means no suppression.
#
# ORGANIZATION is deliberately absent: see the note on DEFAULT_LABELS.
# The loose gate cannot tell a client company name from a vendor name,
# and measured against a gold set containing client organisations
# it suppressed 14 of 21 of them.
WORD_SHAPE_GATES = {
    "CREDENTIAL": "strict",
    "LOCATION": "loose",
}


# ──────────────────────────────────────────────────────────────
# Identifier-shape gate  (separate from the word-shape gate above)
#
# GLiNER answers "is this string the name of a person/organization?".
# For `onLoginSuccess` the honest answer is yes-ish, and for
# `apple-mobile-web-app-title` Apple really is in there. The model is
# not wrong; it is answering a different question from the one that
# matters, which is "is this CLIENT DATA?". This gate answers the second
# question using a property of the string the model does not weigh:
# programmers write identifiers in camelCase, and nobody writes a
# person's or a company's name that way.
#
# Why this is generic rather than corpus-tuned: the camelCase convention
# is a property of how source code, JSON keys, config keys and DOM
# properties are written in every web stack, not of any one app's
# vocabulary. A prefix wordlist (`is`, `on`, `invalid`, ...) was
# considered and rejected — 69% of camelCase ORGANIZATION over-masks
# begin with an uppercase letter and so have no prefix to match, and the
# lowercase remainder has a long tail where an 80-entry list still
# covers under a third. The shape holds on an app nobody has seen; a
# vocabulary list does not.
#
# Per-label rules, and why they differ:
#
#   PERSON        camelCase, any number of capitals.
#                 Zero gold PERSON entities are camelCase on either
#                 dataset. Separators (_ -) are deliberately NOT part of
#                 this rule: dropped by decision, since `jan_novak` is a
#                 plausible username convention even though no measured
#                 entity uses it.
#                 The dot is excluded for a measured reason — `.` costs
#                 8 gold entities (seven of the form a.surname or
#                 first.surname -- names withheld), which
#                 is the dominant internal AD/LDAP username convention.
#
#   ORGANIZATION  camelCase AND a lowercase first character.
#                 Company and client names are capitalised; technical
#                 identifiers are lowerCamelCase. Requiring a lowercase
#                 start protects the whole PascalCase space
#                 (`BankBName`, `ClientCFinance`) automatically,
#                 which is what a prefix wordlist was really buying.
#                 Note ORGANIZATION must NOT get the separator rule:
#                 tenant slugs are hyphen-joined (client-c-cz-prod,
#                 acme-medical-cz, pharmacy-city-9, city-dept-schools),
#                 and a separator rule deletes 10 client identities.
#
# All-caps spans are exempt everywhere: SCREAMING_SNAKE constants are a
# different shape, and the exemption preserves CONFIG_VALUE catches like
# GEORGE_ONLINE that PERSON picks up by cross-label accident.
#
# Never applied to keyval, form or regex. Those layers infer sensitivity
# from the carrier (key name, checksum, pattern), not from the value's
# shape, and random secrets are camelCase by chance: applied to
# keyval|CREDENTIAL this rule matches live JWTs, session cookies and
# passwords (43 gold + 473 corpus true positives across all layers).
# ──────────────────────────────────────────────────────────────

IDENTIFIER_GATES = {
    "PERSON": {"min_upper": 1, "require_lower_start": False},
    "ORGANIZATION": {"min_upper": 1, "require_lower_start": True},
}


def _identifier_suppressed(entity_type, text, gates):
    """True when a span is a program identifier rather than client data."""
    rule = gates.get(entity_type)
    if not rule:
        return False

    s = text.strip()
    if not s or len(s) > _SHAPE_MAX_CHARS:
        return False
    # Multi-word spans are never identifiers. This is what keeps real
    # names ("Sascha Ißbrücker") and multi-word company names out.
    if any(ch.isspace() for ch in s):
        return False
    # SCREAMING_SNAKE / all-caps: different shape, exempt.
    if s == s.upper():
        return False
    if rule.get("require_lower_start") and not s[:1].islower():
        return False
    if not _CAMEL_BOUNDARY.search(s):
        return False
    return sum(1 for ch in s if ch.isupper()) >= int(rule.get("min_upper", 1))


def _segment_coverage(v):
    """Share of the string made of runs of >=2 same-case letters."""
    segments = _WORD_SEGMENT.findall(v)
    if not segments or not v:
        return 0.0
    return sum(len(s) for s in segments) / len(v)


def _max_consonant_run(v):
    best = run = 0
    for ch in v:
        if ch.isalpha() and ch not in _VOWELS:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def _vowel_ratio(v):
    letters = [c for c in v if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in _VOWELS for c in letters) / len(letters)


def _is_word_shaped(v, strict):
    """True when the span reads as language rather than as opaque data."""
    v = v[:_SHAPE_MAX_CHARS]
    if _segment_coverage(v) < _SEGMENT_COVERAGE_MIN:
        return False
    if not strict:
        return True
    return (_max_consonant_run(v) <= _CONSONANT_RUN_MAX
            and _vowel_ratio(v) >= _VOWEL_RATIO_MIN
            and bool(_IDENT_SEPARATOR.search(v) or _CAMEL_BOUNDARY.search(v)))


def _word_shape_suppressed(entity_type, text, gates):
    """True when this detection should be dropped as language, not data."""
    gate = gates.get(entity_type)
    if gate is None:
        return False
    # A house number is what makes a street address an address rather
    # than a place name, so never suppress a LOCATION carrying digits.
    if entity_type == "LOCATION" and any(c.isdigit() for c in text):
        return False
    return _is_word_shaped(text.strip(), strict=(gate == "strict"))

# Placeholder body, WITHOUT the surrounding brackets: JWT_TOKEN_1,
# PHONE_NUMBER_2, EMAIL_ADDRESS_5. Earlier stages have already replaced
# their hits with [LIKE_THIS_1], and GLiNER happily re-detects the inner
# text as a fresh entity — masking a placeholder produces exactly the
# nested-placeholder corruption the masker's four layers exist to stop.
# The span the model returns excludes the brackets, so a
# startswith("[") test never fires; match the body directly instead.
#
# The suffix is capped at 3 digits on purpose. The masker's counter is a
# small sequence number, so \d+ would also swallow real client data that
# ends in a year — ENYAQ_CONNECT_2026, PROJECT_ALPHA_2024. A mask map
# with 1000+ entries in a single prompt is implausible, and the
# bracket-context check below catches that case anyway.
_PLACEHOLDER_BODY = re.compile(r'^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_\d{1,3}$')


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────

def _get_model(config):
    """Lazy-load and cache the GLiNER model.

    Config keys (all under `gliner:`):
      backend:  "gliner2" (default) or "gliner"
      model:    HuggingFace model id
      device:   "cpu" (default) or "cuda"
      use_onnx: bool, gliner backend only
    """
    global _model, _model_key

    gc = config.get("gliner", {})
    backend = gc.get("backend", "gliner2")
    model_name = gc.get("model", "fastino/gliner2-privacy-filter-PII-multi")
    device = gc.get("device", "cpu")

    key = (backend, model_name, device)
    if _model is not None and _model_key == key:
        return _model, backend

    print(f"  [GLiNER] Loading {model_name} (backend={backend}, device={device}) ...")

    if backend == "gliner2":
        from gliner2 import GLiNER2
        _model = GLiNER2.from_pretrained(model_name)
    else:
        from gliner import GLiNER
        if gc.get("use_onnx"):
            _model = GLiNER.from_pretrained(
                model_name,
                load_onnx_model=True,
                onnx_model_file=gc.get("onnx_model_file", "model.onnx"),
            )
        else:
            _model = GLiNER.from_pretrained(model_name)

    # Move to GPU only if explicitly asked. The whole point is CPU.
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                _model = _model.to(torch.device("cuda"))
                print("  [GLiNER] Moved to GPU (CUDA)")
        except Exception as e:
            print(f"  [GLiNER] GPU move failed, staying on CPU: {e}")

    _model_key = key
    print("  [GLiNER] Model ready")
    return _model, backend


# ──────────────────────────────────────────────────────────────
# Chunking
#
# GLiNER encoders have a token limit (typically ~384). Long pentest
# prompts must be split. Same offset-accumulator approach as the
# original implementation's Flair chunker, but bounded by char count
# rather than paragraph breaks alone, since HTTP dumps often have
# no blank lines at all.
# ──────────────────────────────────────────────────────────────

def _chunk(text, max_chars):
    """Split text into (chunk, base_offset) pairs, preferring line breaks."""
    if len(text) <= max_chars:
        return [(text, 0)]

    chunks = []
    pos = 0
    n = len(text)

    while pos < n:
        end = min(pos + max_chars, n)
        if end < n:
            # Back off to the last newline in the final 20% of the window
            window_start = pos + int(max_chars * 0.8)
            brk = text.rfind("\n", window_start, end)
            if brk > pos:
                end = brk + 1
        chunks.append((text[pos:end], pos))
        pos = end

    return chunks


# ──────────────────────────────────────────────────────────────
# Output normalization
#
# The two libraries return different shapes, and gliner2's exact
# structure is version-dependent. Rather than hard-code one shape,
# normalize defensively into a flat list of dicts. If a new gliner2
# version stops producing entities, print `raw` here first to see
# what your installed version actually emits.
# ──────────────────────────────────────────────────────────────

def _normalize(raw):
    """Flatten library output into [{text, label, start, end, score}, ...]."""
    items = []

    # gliner2 wraps its result: {"entities": {label: [...]}}. Older/other
    # versions return the inner dict directly, so unwrap defensively.
    if isinstance(raw, dict) and "entities" in raw and isinstance(raw["entities"], dict):
        raw = raw["entities"]

    if isinstance(raw, dict):
        # {label: [entity, entity, ...]}
        for label, vals in raw.items():
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            for v in vals:
                if isinstance(v, dict):
                    d = dict(v)
                    d.setdefault("label", label)
                    items.append(d)
                elif isinstance(v, str):
                    items.append({"text": v, "label": label})
    elif isinstance(raw, (list, tuple)):
        # gliner returns [{"text","label","start","end","score"}, ...]
        for v in raw:
            if isinstance(v, dict):
                items.append(dict(v))

    return items


def _field(d, *names, default=None):
    """First present key among `names`."""
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def gliner_detect(text, config, debug=False, notes=None):
    """Run GLiNER detection over `text`.

    Args:
        text:   text to scan (already regex/custom-masked by the
                orchestrator, same as what Flair receives)
        config: pipeline config dict
        debug:  print raw model output before filtering
        notes:  optional list; a declined input appends one record to it, so
                the caller can report that this layer did not run. Passing
                None does not disable the gate, only the reporting.

    Returns:
        List of entity dicts in pipeline shape.
    """
    gc = config.get("gliner", {})

    # ── Size gate ──────────────────────────────────────────────
    # Above `max_input_chars`, this stage declines the input and the other
    # detectors handle it alone. Measured in CHARACTERS of the text this
    # function receives, which is what `max_chars` chunks on — not bytes of
    # the original file, which differ under UTF-8 and after earlier stages
    # have substituted placeholders.
    #
    # Why a size gate is the right shape of rule, independent of any corpus:
    # gliner's cost is linear in input length and independent of what it
    # finds (a 4.5 KB message with zero detections still costs seconds), so
    # unbounded input is unbounded cost by construction. Its value is
    # semantic spans in prose, which is a prompt-shaped thing; large HTTP
    # bodies are bundles, dumps and generated markup in any application.
    #
    # Measured cost of the shipped 16384 on the 41 corpus messages above it:
    # ZERO true positives lost (391 tp with and without this stage, every
    # message identical), 378 over-masks removed, wall time 1005s -> 23s.
    # Accepted residual risk, stated: a very large body containing a client
    # name in prose will not be seen by this layer. The other four still see
    # it, and on HTTP traffic regex catches more names than gliner does.
    #
    # Set to 0 to disable the gate — required for any span-for-span
    # comparison against an ungated implementation.
    max_input_chars = int(gc.get("max_input_chars", 0) or 0)
    if max_input_chars and len(text) > max_input_chars:
        msg = (f"gliner declined input: {len(text)} chars > "
               f"max_input_chars {max_input_chars}")
        print(f"  [GLiNER] {msg}")
        if notes is not None:
            notes.append({
                "stage": "gliner",
                "reason": "max_input_chars",
                "chars": len(text),
                "limit": max_input_chars,
            })
        return []

    labels = gc.get("labels", DEFAULT_LABELS)
    threshold = gc.get("threshold", 0.5)
    max_chars = gc.get("max_chars", 1200)

    # Per-label score thresholds, applied ON TOP of the global threshold, keyed
    # by pipeline entity_type (uppercase). ORGANIZATION and PERSON are GLiNER's
    # lowest-precision labels; this lets you raise their bar (e.g. 0.85-0.95)
    # without penalising high-precision labels like EMAIL_ADDRESS or IBAN. A
    # per-label value only has effect when it exceeds the global threshold.
    #   config: gliner.label_thresholds = {ORGANIZATION: 0.9, PERSON: 0.85}
    label_thresholds = {
        str(k).upper(): float(v)
        for k, v in (gc.get("label_thresholds", {}) or {}).items()
    }

    # Word-shape suppression, overridable per label. Set a label to
    # "strict", "loose", or null to disable:
    #   config: gliner.word_shape_gates = {ORGANIZATION: null}
    if "word_shape_gates" in gc:
        word_shape_gates = {
            str(k).upper(): (str(v).lower() if v else None)
            for k, v in (gc.get("word_shape_gates") or {}).items()
        }
        word_shape_gates = {k: v for k, v in word_shape_gates.items() if v}
    else:
        word_shape_gates = dict(WORD_SHAPE_GATES)

    # Identifier-shape suppression, overridable per label. Set a label to
    # null to disable it, or tune its two knobs:
    #   config: gliner.identifier_gates =
    #       {PERSON: {min_upper: 2}, ORGANIZATION: null}
    if "identifier_gates" in gc:
        identifier_gates = {}
        for k, v in (gc.get("identifier_gates") or {}).items():
            if not v:
                continue
            identifier_gates[str(k).upper()] = {
                "min_upper": int(v.get("min_upper", 1)),
                "require_lower_start": bool(v.get("require_lower_start", False)),
            }
    else:
        identifier_gates = {k: dict(v) for k, v in IDENTIFIER_GATES.items()}

    model, backend = _get_model(config)

    entities = []
    seen = set()  # (start, end) dedupe across chunk boundaries

    for chunk_text, base_offset in _chunk(text, max_chars):
        if not chunk_text.strip():
            continue

        try:
            if backend == "gliner2":
                raw = model.extract_entities(
                    chunk_text,
                    labels,
                    threshold=threshold,
                    include_confidence=True,
                    include_spans=True,
                )
            else:
                raw = model.predict_entities(
                    chunk_text, labels, threshold=threshold,
                )
        except Exception as e:
            print(f"  [GLiNER] Inference error on chunk at {base_offset}: {e}")
            continue

        if debug:
            print(f"\n  [GLiNER RAW] offset={base_offset} type={type(raw).__name__}")
            print(f"  {raw!r}\n")

        for item in _normalize(raw):
            ent_text = _field(item, "text", "span", "value", "mention")
            if not ent_text or not isinstance(ent_text, str):
                continue

            label = str(_field(item, "label", "type", "entity", "entity_type",
                               default="UNKNOWN"))
            score = _field(item, "score", "confidence", "prob", default=0.5)

            start = _field(item, "start", "start_index", "char_start")
            end = _field(item, "end", "end_index", "char_end")

            # Some versions return spans as a (start, end) tuple
            span = _field(item, "span")
            if start is None and isinstance(span, (list, tuple)) and len(span) == 2:
                start, end = span

            # No offsets returned — locate the string ourselves. Same
            # fallback the original implementation's LLM stage uses.
            if start is None:
                start = chunk_text.find(ent_text)
                if start == -1:
                    continue
                end = start + len(ent_text)

            try:
                start, end = int(start), int(end)
            except (TypeError, ValueError):
                continue

            if not (0 <= start < end <= len(chunk_text)):
                continue

            # The model's `text` field is NOT reliable — gliner2 has been
            # observed returning a reformatted value, e.g. a bare hostname
            # comes back markdown-linkified as "[host](https://host)", a
            # 48-char string for an 18-char span. That string does not
            # exist in the input, so masking it would silently no-op.
            # The offsets are correct, so slice the source instead.
            span_text = chunk_text[start:end]
            if span_text != ent_text:
                ent_text = span_text
            if not ent_text.strip():
                continue

            abs_start = base_offset + start
            abs_end = base_offset + end

            if (abs_start, abs_end) in seen:
                continue
            seen.add((abs_start, abs_end))

            entity_type = LABEL_MAP.get(
                label.lower(), label.upper().replace(" ", "_")
            )

            # Per-label score threshold (see label_thresholds above). Drop a
            # detection whose confidence is below its label's bar — used to tame
            # the noisy ORGANIZATION / PERSON labels without touching others.
            _lt = label_thresholds.get(entity_type)
            if _lt is not None and float(score) < _lt:
                continue
            # ── Filters — mirror the original implementation's NER filters exactly ──
            stripped = ent_text.strip()
            if "\n" in ent_text or "|" in ent_text:
                continue

            # ── Placeholder rejection (three ways it shows up) ──
            # 1. Span includes the brackets: "[JWT_TOKEN_1]"
            if "[" in ent_text or "]" in ent_text:
                continue
            # 2. Span is the bare body and the source has brackets
            #    immediately around it — the common case.
            if (start > 0 and end < len(chunk_text)
                    and chunk_text[start - 1] == "[" and chunk_text[end] == "]"):
                continue
            # 3. Span looks like a placeholder body on its own, e.g. the
            #    model returned "PHONE_NUMBER_1" from inside brackets it
            #    trimmed. Real client data almost never has this shape.
            if _PLACEHOLDER_BODY.match(stripped):
                continue

            if entity_type not in _LENGTH_EXEMPT and len(stripped) <= 2:
                continue
            if entity_type == "LOCATION":
                if len(stripped) < 4 or stripped.isdigit():
                    continue
            if entity_type == "ORGANIZATION" and len(stripped) < 3:
                continue
            if entity_type == "PERSON" and len(stripped) < 3:
                continue

            # Language, not client data — see WORD_SHAPE_GATES above.
            # Never applies to PERSON / USERNAME.
            if _word_shape_suppressed(entity_type, stripped, word_shape_gates):
                continue

            # Program identifier, not client data — see IDENTIFIER_GATES.
            if _identifier_suppressed(entity_type, stripped, identifier_gates):
                continue

            entities.append({
                "text": ent_text,
                "entity_type": entity_type,
                "start": abs_start,
                "end": abs_end,
                "score": round(float(score), 3),
                "stage": "ner",       # identical downstream semantics
                "detector": "gliner",  # benchmark reporting only
            })

    return entities


def warmup(config):
    """Pre-load the model so the first benchmark prompt isn't skewed."""
    try:
        _get_model(config)
        # One tiny inference to trigger any lazy graph construction
        gliner_detect("Contact John Smith at john@example.com", config)
    except Exception as e:
        print(f"  [GLiNER] Warmup failed: {e}")
        raise
