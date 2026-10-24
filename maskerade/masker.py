"""Masker — replace detected entities with placeholders and reverse it.

Two functions:
  apply_masks()    → takes text + entities, returns masked text + mask_map
  unmask_response() → takes text + mask_map, restores original values

The mask_map is the critical security artifact:
  - It maps placeholders back to original values
  - It is stored LOCALLY and NEVER sent to cloud AI

In chat mode, the mask_map is CUMULATIVE across turns:
  - Turn 1: [EMAIL_ADDRESS_1] = "jan@acme.cz"
  - Turn 2: [EMAIL_ADDRESS_2] = "petr@acme.cz" (numbering continues)
  - If "jan@acme.cz" appears again in turn 2, it reuses [EMAIL_ADDRESS_1]

Substring-corruption defense (the prompt 041 bug):
  When the LLM mis-extracts a single character like "1" from a field like
  `employeeNumber: 1`, naive str.replace would corrupt every digit `1` in
  the document — including digits inside placeholders, timestamps, IPs.
  Defense in depth across four layers:
    1. Minimum length floor for non-regex entities (don't accept "1")
    2. Word-boundary anchored substitution (\b1\b vs naive "1")
    3. LLM output sanity filter in llm_stage.py (rejects "1", "Q4", "14:22")
    4. Longest-first replacement order (mask "20241015" before "1")
  Layers 1+3 are upstream guards. Layers 2+4 ensure that even if a short
  value enters the map, it can't corrupt the surrounding text.

Credential-pair splitting (the "test:test" bug):
  A single CREDENTIAL/AZURE_CREDENTIAL/USERNAME entity that looks like
  "identifier:secret" (e.g. "test:test" pulled from an HTML comment) is
  split into two independently-masked entities BEFORE Layer 1 runs. See
  _split_credential_pairs(). This lets the agent reference just the
  username half ([CREDENTIAL_1]) instead of getting the whole pair
  ("test:test") crammed into one placeholder that breaks when used in
  a path segment or form field expecting a bare username.
"""
import re


# Layer 1: minimum length for entities allowed into the mask_map.
# Regex-stage entities bypass this — their patterns encode validation
# (IBAN length, JWT structure, email shape) so trusting them is safe.
#
# That reasoning holds only for patterns that match a WHOLE structured token.
# It does not hold for Tier-2 capture-group patterns (password_value,
# secret_value, username_value, ...), which match whatever follows a key and
# validate nothing about it. Those produced real corruption: a username
# pattern captured "==" from a JavaScript comparison and a user_id pattern
# captured "=", after which every "==" and "=" in the document was replaced by
# a placeholder. regex_stage now tags such entities capture_group=True and they
# are held to the same floor as NER and LLM output.
class UnmaskableEntityError(Exception):
    """A detected entity could not be masked and the message was refused.

    Raised only under masking.on_unmaskable: raise. The default is to widen.
    """


MIN_ENTITY_LENGTH = 3

# Layer 1 (extension): edge separators don't count toward the length floor.
# A value that is mostly leading/trailing separators — e.g. "21:" clipped
# from a MAC "…:21:8A" by an over-eager detector — has too little actual
# content to be a real entity, and masking such a fragment splits the larger
# structured token it came from (wedging a placeholder mid-value). We measure
# the value with edge non-word chars removed; if that core falls below the
# floor we drop it (never mask a sub-floor fragment). We still MASK the
# ORIGINAL value (padding, dots, slashes intact) for everything that passes —
# the strip is used ONLY for the length decision, so base64 "=" padding,
# "s.r.o.", "/var/log/x" etc. are unaffected. Unicode-aware (\W).
_EDGE_NONWORD_RE = re.compile(r'^\W+|\W+$', re.UNICODE)

# Layer 1b: reject entities whose text already CONTAINS a placeholder token
# like [USERNAME_1]. This happens when a later stage (NER/LLM) scans text that
# an earlier stage already partially masked, and detects a span that swallows
# the placeholder — e.g. regex masks "<strong>" -> [USERNAME_1], leaving
# "[USERNAME_1] demo", which the LLM then tags as one entity. Masking that
# produces a placeholder pointing at a value that contains ANOTHER placeholder
# (unmask then yields corrupted, un-restorable text). Detection firing on
# already-masked text is never a real entity: drop it, don't try to repair the
# offsets. Matches the [TYPE_N] shape emitted by placeholder_format.
_CONTAINS_PLACEHOLDER_RE = re.compile(r'\[[A-Z][A-Z0-9_]*_\d+\]')
# The same shape WITHOUT its brackets. Widening expands over word characters
# and stops at '[' and ']', so it can only ever produce a placeholder's
# innards — which _CONTAINS_PLACEHOLDER_RE does not match. Masking that would
# nest a placeholder inside a placeholder and corrupt the mask map, so
# widening tests against this instead.
_PLACEHOLDER_BODY_RE = re.compile(r'^[A-Z][A-Z0-9_]*_\d+$')

# Entity types eligible for credential-pair splitting (see
# _split_credential_pairs). Deliberately narrow: CONNECTION_STRING,
# AZURE_CREDENTIAL's cousins with multiple colons, IPv6, and timestamps
# all use different entity_types or fail the "exactly one colon" check,
# so they're never affected.
CREDENTIAL_SPLIT_TYPES = frozenset({"CREDENTIAL", "AZURE_CREDENTIAL", "USERNAME"})

# Minimum length for EACH side of a split "identifier:secret" pair.
MIN_SPLIT_SIDE_LENGTH = 2

# Entity types eligible for path-segment decomposition (see
# _split_path_entities). SENSITIVE_FIELD is the LLM stage's catch-all
# for "this looks sensitive" — it sometimes spans an entire URL path
# instead of just the variable segment.
PATH_SPLIT_TYPES = frozenset({"SENSITIVE_FIELD", "URL", "PATH"})

# Matches "/segment/segment[/...]" — at least two non-empty segments,
# no query string or fragment. Deliberately excludes '?' and '#' so a
# full URL with a query string doesn't get chopped mid-query.
_PATH_SHAPE_RE = re.compile(r'^/[^\s?#]+/[^\s?#]*[^\s?#/]$')

# Entity types where a PURELY NUMERIC value is treated as low-risk and
# intentionally left UNMASKED. Sequential/fuzzable IDs (user_id=7,
# customerId=7840219) leak little on their own — an attacker can
# usually enumerate them anyway — and masking them costs the AI
# context it needs to reason about paths and parameters.
#
# SENSITIVE_FIELD is included (not just USER_ID/EMPLOYEE_ID) because
# the LLM stage uses it as its catch-all for "this looks like an
# identifier" and often applies it to plain numeric object/customer
# IDs (see system_prompt.txt's own "7840219" example). Scope is
# intentionally still narrow: non-numeric values of these types
# (UUIDs, opaque alphanumeric tokens) are NEVER exempted — only
# strings that are 100% digits are treated as low-risk.
NUMERIC_ID_SKIP_TYPES = frozenset({"USER_ID", "EMPLOYEE_ID", "SENSITIVE_FIELD"})
_PURE_DIGITS_RE = re.compile(r'^\d+$')


def is_low_risk_numeric_id(entity, config):
    """True if this entity is a purely-numeric ID that should stay unmasked.

    IMPORTANT — this is NOT applied inside apply_masks(). apply_masks()
    only sees whatever entity list it's given; it has no way to also
    reach into the validator. If this filter lived only here, an
    entity we chose not to mask would still be in orchestrator.py's
    `all_entities`, and validate() would flag it as a "leak" — a false
    positive, since we intentionally never intended to mask it.

    Callers (orchestrator.py) must apply this at BOTH the final
    apply_masks() call and the validate() call, the same way
    `stage == "custom"` entities are already excluded from both.
    """
    masking_config = config.get("masking", {})
    skip_types = set(masking_config.get("numeric_id_skip_types", NUMERIC_ID_SKIP_TYPES))
    if not skip_types:
        return False
    etype = entity.get("entity_type", "")
    text = entity.get("text", "")
    return etype in skip_types and bool(_PURE_DIGITS_RE.match(text))


def word_boundary_pattern(value):
    """Build a regex that matches `value` only at non-word-character boundaries.

    Uses (?<!\\w) / (?!\\w) which are Unicode-aware in Python 3, so Czech
    letters (á, ě, ž, ů) are treated as word chars and "Polák" inside
    "Poláková" is correctly NOT matched.

    Anchors are added only when the value's edge char is a word char.
    Bracketed/punctuated values like "[ADMIN]" or ".env" get plain literal
    matching — \\w-anchoring on a non-word edge char would invert the
    semantics and block legitimate matches.

    Public so validator.py can use the same boundary semantics as the
    masker, otherwise a 3-char value masked at word boundaries would
    falsely appear "leaked" as a substring of a longer unmasked token.
    """
    pattern = re.escape(value)
    if value and re.match(r'\w', value[0]):
        pattern = r'(?<!\w)' + pattern
    if value and re.match(r'\w', value[-1]):
        pattern = pattern + r'(?!\w)'
    return pattern


def align_to_token_boundary(text, value):
    """Shrink `value` until it can actually be substituted in `text`.

    Layer 2 only replaces a value where both edges sit on a
    non-word-character boundary. A detection whose span ends in the
    middle of a token therefore matches NOTHING and is silently left in
    the output — detected, recorded in the mask map, never masked.

    That happens whenever a whitespace-tolerant pattern runs off the end
    of its field. In a fixed-width table:

        jan.doe01@example.cz       +420 597 482 111  52000
                                   └── detected span ──┘

    the span ends on "520" inside "52000", so `(?!\\w)` refuses to match
    and the phone number survives masking. Column-padded output — SQL
    clients, nmap, CSV exports, log lines — makes this common, and it
    fails silently, which is the dangerous part.

    This returns the longest prefix of `value` that IS substitutable,
    trimming whole trailing characters until both edges land on a
    boundary. Shrinking only ever masks less text than was detected, so
    it cannot corrupt a neighbouring token. Returns None when no prefix
    works, in which case the caller must treat the entity as unmaskable
    rather than pretend it was handled.
    """
    if not value:
        return None
    # Already fine — the common case, costs one search.
    if re.search(word_boundary_pattern(value), text):
        return value
    candidate = value
    while len(candidate) > 1:
        candidate = candidate[:-1]
        # A trailing separator would be trimmed by the edge-stripped
        # floor anyway; skip straight to the next real character.
        if not candidate or candidate[-1].isspace():
            continue
        if re.search(word_boundary_pattern(candidate), text):
            return candidate
    return None


def _split_credential_pairs(entities, config):
    """Split "identifier:secret"-shaped entities into two independent entities.

    A single regex/NER/LLM match can capture a structured pair as ONE
    entity — e.g. the LLM stage pulling "test:test" whole out of an HTML
    comment like `(test:test)`. Masking that as one opaque placeholder
    is correct for confidentiality but destroys structure the agent
    needs downstream: `GET /password/{username}` becomes unusable once
    [CREDENTIAL_1] unmasks to "test:test", colon and all, instead of
    just the username.

    Only entities whose entity_type is in CREDENTIAL_SPLIT_TYPES are
    eligible, and only when there is EXACTLY one colon with both sides
    at least MIN_SPLIT_SIDE_LENGTH chars. This intentionally excludes
    connection strings, IPv6 addresses, host:port, and timestamps —
    those use other entity_types or contain more than one colon.

    Each half keeps the original entity_type, so "admin:S3cretPass"
    (CREDENTIAL) becomes two CREDENTIAL entities. apply_masks then gives
    them independent placeholders ([CREDENTIAL_1], [CREDENTIAL_2]) while
    the literal ":" between them stays untouched in the masked text,
    since substitution only ever touches the matched spans themselves.
    """
    masking_config = config.get("masking", {})
    split_types = set(masking_config.get("credential_split_types", CREDENTIAL_SPLIT_TYPES))
    min_side = masking_config.get("credential_split_min_side_length", MIN_SPLIT_SIDE_LENGTH)

    result = []
    for entity in entities:
        text = entity.get("text", "")
        etype = entity.get("entity_type", "")
        if etype in split_types and text.count(":") == 1:
            left, right = text.split(":", 1)
            if len(left) >= min_side and len(right) >= min_side:
                left_entity = dict(entity)
                left_entity["text"] = left
                right_entity = dict(entity)
                right_entity["text"] = right
                result.append(left_entity)
                result.append(right_entity)
                continue
        result.append(entity)
    return result


def _split_path_entities(entities, config):
    """Mask only the trailing segment of a full-URL-path entity.

    The LLM stage can tag an entire path as one SENSITIVE_FIELD span
    (e.g. "/password/test" as a whole) instead of just the variable
    part. Masking the full path collapses it into one opaque
    placeholder — the agent can no longer reason about path structure,
    sibling endpoints, or traversal variants, and PoCs end up as
    `POST [SENSITIVE_FIELD_2] HTTP/1.1` instead of
    `POST /password/[SOMETHING] HTTP/1.1`.

    This is a heuristic, not a router: REST APIs overwhelmingly put the
    variable/identifier in the LAST path segment, so that's what gets
    masked; everything before it is treated as the static route and
    left readable. Multi-level dynamic paths (e.g. "/orgs/42/users/99")
    will only get the trailing segment masked — the real fix for that
    belongs in the LLM system prompt (extract the segment, not the
    path). This is the defense-in-depth backstop, not a replacement
    for that.

    Only entities in PATH_SPLIT_TYPES are eligible, and only when the
    text matches the "/segment/segment" shape (no query string —
    those are left alone so a full URL with `?token=...` isn't chopped
    mid-query).
    """
    masking_config = config.get("masking", {})
    split_types = set(masking_config.get("path_split_types", PATH_SPLIT_TYPES))

    result = []
    for entity in entities:
        text = entity.get("text", "")
        etype = entity.get("entity_type", "")
        if etype in split_types and _PATH_SHAPE_RE.match(text):
            last_segment = text.rstrip('/').rsplit('/', 1)[-1]
            if last_segment:
                new_entity = dict(entity)
                new_entity["text"] = last_segment
                result.append(new_entity)
                continue
        result.append(entity)
    return result


# Widened spans are capped: a word-character run can be a 50 KB base64 blob,
# and masking one because a 4-char fragment was detected inside it is a worse
# outcome than the fragment. Over this cap the entity stays unmaskable.
MAX_WIDENED_LENGTH = 256


def widen_to_token_boundary(text, value):
    """Grow `value` outward to the token(s) containing it.

    The fail-closed counterpart to align_to_token_boundary. Alignment only
    ever SHRINKS, and shrinking cannot rescue a span whose START is mid-token:
    every prefix of 'alified' inside 'qualified' still begins after the 'u',
    so the left-boundary assertion refuses them all and the entity is left in
    the output.

    Widening goes the other way — expand each occurrence until both edges sit
    on non-word boundaries, and mask that. 'alified' inside 'qualified' masks
    'qualified'. That is an over-mask, and over-masking is the acceptable
    failure: the alternative is emitting a detected entity to a third party.

    Returns the distinct widened literals, in order of first occurrence. A
    value can occur in different hosts ('alified' in both 'qualified' and
    'disqualified'), and each host must be masked under its own placeholder
    so unmasking restores the right text — hence a list, not one string.
    Returns [] when the value does not occur literally at all (detector
    detokenisation, NFC/NFD mismatch), which stays unmaskable: there is
    nothing in the text to widen.
    """
    if not value:
        return []
    out, seen, start = [], set(), 0
    while True:
        i = text.find(value, start)
        if i < 0:
            break
        start = i + 1
        a, b = i, i + len(value)
        while a > 0 and re.match(r'\w', text[a - 1], re.UNICODE):
            a -= 1
        while b < len(text) and re.match(r'\w', text[b], re.UNICODE):
            b += 1
        if b - a > MAX_WIDENED_LENGTH:
            continue
        widened = text[a:b]
        # Never widen into one of our own placeholders. Masking the body of
        # '[PERSON_NAME_1]' produces '[[PERSON_2]]' and breaks unmasking.
        if _PLACEHOLDER_BODY_RE.match(widened) and (
                a > 0 and text[a - 1] == "[" and b < len(text)
                and text[b] == "]"):
            continue
        if widened not in seen:
            seen.add(widened)
            out.append(widened)
    return out


def apply_masks(text, entities, config, existing_mask_map=None, report=None):
    """Replace all detected entities in text with placeholders.

    Args:
        text:              original full prompt text
        entities:          list of dicts from detection stages
        config:            pipeline config dict
        existing_mask_map: (optional) cumulative mask map from previous
                           chat turns. Used to:
                           1. Continue numbering ([EMAIL_2] not [EMAIL_1])
                           2. Reuse placeholders for the same text
        report:            (optional) dict that receives masking
                           diagnostics. Populated with:
                             "unmaskable" — detected entities that could
                               NOT be substituted at all. Each one is a
                               value the pipeline found and then left in
                               the output, so a non-empty list is a leak
                               and the caller should treat it as one.
                             "realigned"  — spans shrunk to a token
                               boundary so they could be substituted.
                               Usually means an over-greedy pattern ran
                               past the end of its field.
                           Pass a dict to collect these; omit to ignore.

    Returns:
        (masked_text, mask_map)
        - masked_text: the prompt with placeholders
        - mask_map: FULL map including existing + new entries
    """
    unmaskable = []
    widened = []
    realigned = []
    masking_config = config.get("masking", {})
    fmt = masking_config.get("placeholder_format", "[{type}_{index}]")
    consistent = masking_config.get("consistent_mapping", True)
    min_length = masking_config.get("min_entity_length", MIN_ENTITY_LENGTH)

    # Initialize from existing map (for chat continuity)
    type_counters = {}
    text_to_placeholder = {}
    mask_map = {}

    if existing_mask_map:
        # Copy existing mappings — preserve continuity even if some values
        # would be rejected by today's rules. Don't break old chats on
        # upgrade. Layer 2's word-boundary substitution makes legacy
        # short-value entries safe to keep.
        mask_map = dict(existing_mask_map)

        # Build reverse lookup: original_text → placeholder
        for placeholder, original in existing_mask_map.items():
            # Extract type and index from placeholder like [EMAIL_ADDRESS_3]
            match = re.match(r'\[(.+)_(\d+)\]', placeholder)
            if match:
                etype = match.group(1)
                idx = int(match.group(2))
                # Track highest index per type
                type_counters[etype] = max(type_counters.get(etype, 0), idx)
                # Build text→placeholder for consistent mapping
                text_to_placeholder[(original, etype)] = placeholder

    # ─── Split "identifier:secret" pairs and full-path spans before ───
    # anything else. Must run before Layer 1 so each fragment is
    # independently subject to the length floor, and before the sort
    # so all fragments compete for placeholder numbering on equal
    # footing.
    entities = _split_credential_pairs(entities, config)
    entities = _split_path_entities(entities, config)

    # ─── Layer 1: minimum length filter for non-regex entities ───
    # Whole-token regex matches already validated themselves with structural
    # patterns (Luhn for credit cards, IBAN length, JWT shape, etc.) so they
    # bypass this floor. Capture-group regex matches, NER and LLM all get the
    # 3-char minimum to block trivial values like "1", "=" and "==" that cause
    # substring corruption.
    filtered_entities = []
    for entity in entities:
        e_text = entity.get("text", "")
        stage = entity.get("stage", "")
        # Layer 1b: never mask a value that already contains a placeholder —
        # it's a detection artifact from scanning already-masked text. Applies
        # to every stage (a regex value-capture can swallow a prior placeholder
        # too). See _CONTAINS_PLACEHOLDER_RE.
        if _CONTAINS_PLACEHOLDER_RE.search(e_text):
            continue
        # A capture-group regex match is an unvalidated value, not a
        # structurally-verified token, so it does NOT inherit the regex bypass.
        trusted_regex = stage == "regex" and not entity.get("capture_group")
        if not trusted_regex:
            # Edge-stripped floor: sub-floor once separators are removed → drop
            # (fragment/junk span). Subsumes the old raw len<min check because
            # the stripped core is never longer than e_text.
            if len(_EDGE_NONWORD_RE.sub("", e_text)) < min_length:
                continue
        filtered_entities.append(entity)

    # ─── Layer 5: token-boundary alignment ───
    # Layer 2 can only substitute a value whose edges land on
    # non-word-character boundaries. A span that ends mid-token matches
    # nothing, so it would be recorded in the mask map and then left in
    # the output — a silent leak. Shrink such spans to the longest
    # substitutable prefix, and drop the ones that cannot be salvaged so
    # they are never recorded as masked when they are not.
    # on_unmaskable decides what happens when alignment cannot salvage a span.
    # Owner decision 2026-10-05: this tool does not fail open. A detected
    # entity must never reach the output, so the default WIDENS to the
    # containing token and masks that, accepting an over-mask. "raise" refuses
    # the whole message; "skip" is the pre-decision behaviour and is retained
    # only so the old result can be reproduced deliberately.
    on_unmaskable = masking_config.get("on_unmaskable", "widen")

    def _rescue(entity, e_text, reason):
        """Return entities to mask instead of e_text, or [] if unrescuable."""
        if on_unmaskable == "skip":
            unmaskable.append({
                "text": e_text, "entity_type": entity.get("entity_type"),
                "stage": entity.get("stage"), "reason": reason,
            })
            return []
        wide = widen_to_token_boundary(text, e_text)
        wide = [w for w in wide
                if len(_EDGE_NONWORD_RE.sub("", w)) >= min_length]
        if not wide:
            # Nothing in the text to widen — the value does not occur
            # literally. Record it, and refuse the message if configured to.
            unmaskable.append({
                "text": e_text, "entity_type": entity.get("entity_type"),
                "stage": entity.get("stage"),
                "reason": reason + " (not present literally; cannot widen)",
            })
            if on_unmaskable == "raise":
                raise UnmaskableEntityError(
                    "entity %r (%s, stage %s) was detected but cannot be "
                    "masked: %s. It would be sent unmasked." % (
                        e_text, entity.get("entity_type"),
                        entity.get("stage"), reason))
            return []
        for w in wide:
            widened.append({
                "from": e_text, "to": w,
                "entity_type": entity.get("entity_type"),
                "stage": entity.get("stage"), "reason": reason,
            })
        return [{**entity, "text": w} for w in wide]

    aligned_entities = []
    for entity in filtered_entities:
        e_text = entity["text"]
        fixed = align_to_token_boundary(text, e_text)
        if fixed is None:
            aligned_entities.extend(
                _rescue(entity, e_text, "no substitutable span"))
            continue
        if fixed != e_text:
            # Re-apply the length floor: trimming can take a span below it.
            if len(_EDGE_NONWORD_RE.sub("", fixed)) < min_length:
                aligned_entities.extend(
                    _rescue(entity, e_text, "trimmed span below length floor"))
                continue
            realigned.append({
                "from": e_text, "to": fixed,
                "entity_type": entity.get("entity_type"),
                "stage": entity.get("stage"),
            })
            entity = {**entity, "text": fixed}
        aligned_entities.append(entity)

    # ─── Layer 4: sort by text length DESC (longest first) ───
    sorted_entities = sorted(aligned_entities, key=lambda e: -len(e["text"]))

    for entity in sorted_entities:
        e_text = entity["text"]
        e_type = entity["entity_type"]

        key = (e_text, e_type)
        if consistent and key in text_to_placeholder:
            continue

        if e_type not in type_counters:
            type_counters[e_type] = 0
        type_counters[e_type] += 1

        placeholder = fmt.format(type=e_type, index=type_counters[e_type])
        text_to_placeholder[key] = placeholder
        mask_map[placeholder] = e_text

    # ─── Layers 2 + 4: word-boundary regex, longest-first ───
    # Critical: longest first prevents short values from masking
    # before longer values that contain them. Word-boundary regex
    # prevents matched values from substring-corrupting unrelated tokens.
    masked_text = text
    for key, placeholder in sorted(
        text_to_placeholder.items(),
        key=lambda x: -len(x[0][0])
    ):
        original_text = key[0]
        if not original_text:
            continue
        pattern = word_boundary_pattern(original_text)
        masked_text = re.sub(pattern, placeholder, masked_text)

    # ─── Post-substitution audit ───
    # Anything still present at its own word boundaries was recorded as
    # masked but never replaced. Alignment should have prevented this;
    # if it still happens the value is leaking, so say so loudly rather
    # than returning a mask map that claims otherwise.
    for key in text_to_placeholder:
        original_text = key[0]
        if not original_text:
            continue
        if re.search(word_boundary_pattern(original_text), masked_text):
            unmaskable.append({
                "text": original_text,
                "entity_type": key[1],
                "stage": "post-substitution",
                "reason": "still present after substitution",
            })

    if unmaskable:
        print(f"  [MASKER] WARNING: {len(unmaskable)} detected entit"
              f"{'y' if len(unmaskable) == 1 else 'ies'} could not be masked "
              f"and remain in the output:")
        for u in unmaskable[:5]:
            print(f"  [MASKER]   {u['entity_type']} {u['text'][:60]!r} "
                  f"({u['reason']})")

    if report is not None:
        report["unmaskable"] = unmaskable
        report["widened"] = widened
        report["realigned"] = realigned

    return masked_text, mask_map


def unmask_response(text, mask_map):
    """Reverse masking — restore original values from placeholders.

    Placeholders always start with `[` and end with `]` (non-word chars),
    so plain string replacement is safe — there's no substring-corruption
    risk because placeholder boundaries are non-word punctuation.

    Length-sort prevents `[EMAIL_1]` from matching inside e.g. `[EMAIL_10]`
    (it can't anyway — the closing `]` blocks the substring — but the
    sort is cheap insurance against future placeholder format changes).
    """
    result = text
    for placeholder, original in sorted(mask_map.items(), key=lambda x: -len(x[0])):
        result = result.replace(placeholder, original)
    return result
