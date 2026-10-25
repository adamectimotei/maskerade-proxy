"""Structured business-ID detection — issuer prefix + sequence counter.

Runs in the pipeline's detector slot (maskerade/pipeline.py, _run_detectors),
in both tiers; switched by stages.id.

What it detects, and why that generalises
-----------------------------------------
On the development gold set, 16 of 47 false negatives shared one shape —

    DRV-004578   TRIP-2026-0218-7742   KS-00142   AGT-0482   EP-2019-0847
    CC-EMEA-OPS-4421   SPZ-4AC-8821   TX-20260414-0918273   CZPE2604150082148

These are internal business keys: an alphabetic issuer code followed by a
sequence counter. Every organisation mints them, the prefixes differ per
organisation, and no enumerated regex list can know them in advance. So the
detector keys on the STRUCTURE of the key, never on a prefix vocabulary.

Why BPE is not used here (measured, not assumed)
------------------------------------------------
The obvious idea is to gate this with stages/form.py's BPE token efficiency —
low efficiency means "random", so it should mean "identifier". Measured on
the 16 gold IDs against 38 published-standard / CSS-framework / version
strings (a one-off probe, cl100k_base):

    AUC 0.517            (0.500 is a coin flip; inverted it is 0.483)
    IDs       min 1.50  mean 2.00  max 3.00
    negatives min 1.00  mean 2.00  max 3.33

The two class-conditional distributions are identical, so BPE contributes
nothing as a gate AND nothing as a weight — a feature whose distributions
coincide adds variance to a linear combination without adding evidence. It
is actively inverted at the low end: v1.2.3 (1.00), qwen2.5:7b (1.25) and
mt-4 (1.33) are less efficient than every business ID in the set.

The reason is mechanical. Token efficiency measures decomposition into
learned merges; at 8-25 characters dominated by punctuation and small
numbers everything fragments. form_stage's 2.2 threshold is calibrated on
LONG RANDOM SECRETS (keyhog-scanner's CredData F1 peak) and is correct for
that job. It has no signal at this length. Do not re-add it without
re-running the probe.

What does separate (same 16 vs 38, AUC)
----------------------------------------
    longest digit run    0.945
    total digits         0.914
    zero-padded run      0.862
    uppercase prefix     0.758
    segment count        0.500     <- no signal
    BPE efficiency       0.517     <- no signal

Four principles, each a property of how organisations mint keys rather than
a pattern copied off a benchmark prompt:

  P1 PREFIX   an internal key opens with an alphabetic issuer code,
              optionally followed by further alphabetic segments, before
              its digits. This is what separates a key from a bare number:
              3700277 and 1779299915 are sizes, timestamps and row ids.

  P2 COUNTER  a zero-padded run is a fixed-width sequence counter, so it is
              a WHOLE segment. `00` inside graphw00f is letters that happen
              to look like digits, not a counter. Zero-padding is the
              single most useful signal here because published standards
              never do it: CWE-639, AES-256, ISO-8601, RFC-3986, SHA-512,
              col-sm-2, v1.2.3 — not one leading zero among them.

  P3 TAIL     a business key is prefix + counter. Digits interleaved with
              letters (46ec0bcc021129d4, sha512-6v6nzO8RiyfE) is a hash or
              a session token, which regex and form already own.

  P4 REGISTRY published standards number from registries and their numbers
              are short. The veto for these comes from the project
              allowlist as it ships — no vocabulary is authored here.

Validation before a line of the detector was written
-----------------------------------------------------
    gold IDs                    12/12 caught
    probe negatives              0/37 fired
    corpus over-mask strings     0/5823 fired   (overmasks_base_corpus.csv)

The corpus figure is necessary but not sufficient: those 5,823 strings are
values some layer ALREADY masked. The real test is a corpus run, where this
stage sees text nothing flagged. Read PER-LABEL YIELD and the over-mask CSV.

Entity type
-----------
SENSITIVE_FIELD, deliberately. The evidence here is "this value is shaped
like an internal key", which says the value is sensitive, not what it
means — the same IDs span employees, trips, agreements, claims and vehicles.
SENSITIVE_FIELD is in orchestrator.GENERIC_ENTITY_TYPES, so when a more
authoritative layer types the same span specifically, reconciliation
promotes that label instead of this one.

Ordering
--------
Runs in the NER slot, so regex hits are already placeholders in the text by
the time this stage sees it: UUIDs, JWTs, hashes and every other enumerated
format never reach it. Placeholders are rejected explicitly as well.

Pipeline contract
-----------------
    {"text", "entity_type", "start", "end", "score",
     "stage": "ner",       # identical masker/validator semantics
     "detector": "id"}     # benchmark attribution only

Offsets index into the text passed in: text[start:end] == entity["text"].
"""
import os
import re

ENTITY_TYPE = "SENSITIVE_FIELD"
DETECTOR = "id"

# ── Candidate shape (P1) ─────────────────────────────────────
# An alphabetic issuer code, optional further segments, then digits. The
# lookarounds make the match a whole token: an ID is never part of a longer
# run of value characters.
#
# '%' is deliberately absent from every character class, so a percent-encoded
# path (%2Fadmin%2Fusers%2F1) cannot form a candidate — `admin` is followed
# by '%', which is neither a separator nor a digit.
_CANDIDATE = re.compile(
    # '/' and ':' are NOT in the lookbehind: an ID in a URL path
    # (/api/v2/drivers/DRV-004578) or after a key (ref:DRV-004578) is the
    # normal case, and neither character is in the body class, so a match
    # can never span one anyway.
    r'(?<![A-Za-z0-9._%+-])'
    r'[A-Za-z]{2,6}(?:[-_][A-Za-z0-9]{1,8})*[-_]?\d[A-Za-z0-9_-]*'
    r'(?![A-Za-z0-9_%-])'
)

_SEG = re.compile(r'[-_.:/]')
_ZERO_PADDED_SEGMENT = re.compile(r'0\d+')          # P2: whole segment only
_CONTIGUOUS_TAIL = re.compile(                      # P3: letters, then digits
    r'[A-Za-z]{2,6}(?:[-_][A-Za-z0-9]{1,8})*?[-_]?\d{4,}'
)
_VERSION = re.compile(r'[Vv]?\d+(?:\.\d+)+')
_HEX_ONLY = re.compile(r'[0-9a-fA-F]{8,}')
_UPPER_PREFIX = re.compile(r'[A-Z]{2,6}')
_PLACEHOLDER = re.compile(r'\[[A-Z][A-Z0-9_]*\]')

DEFAULTS = {
    "min_total_digits": 6,      # P3 companion: see _rule() for provenance
    "min_tail_digits": 4,       # P1 companion, multi-segment composite keys
    "min_len": 6,
    "max_len": 64,
    "allowlist_dir": None,      # default: the allowlist beside this file
    "score": 1.0,
}

_STATS = {
    "calls": 0,
    "candidates": 0,
    "rejected_placeholder": 0,
    "rejected_claimed": 0,
    "rejected_length": 0,
    "rejected_version": 0,
    "rejected_allowlist": 0,
    "rejected_shape": 0,
    "emitted": 0,
}


def id_stats():
    return dict(_STATS)


def reset_stats():
    for k in _STATS:
        _STATS[k] = 0


# ── Allowlist (P4) ───────────────────────────────────────────
# The project allowlist AS IT SHIPS. No group is added, removed or reordered
# for this stage: a vocabulary tuned to make this detector look good would be
# exactly the corpus-fitting this project rules out.
#
# THE PREFIX VETO IS AN EXCEPTION TO THE ALLOWLIST'S WHOLE-TOKEN RULE.
# Owner decision, 2026-10-14. Read this before changing it in either
# direction.
#
# maskerade/allowlist/__init__.py states a safety property in writing:
#
#     "whole-token, case-insensitive: an entity is suppressed only when its
#      ENTIRE text equals an allow-list term ... so allow-listing 'oracle'
#      cannot un-mask 'oracle-client-db-prod-4471'"
#
# That holds for the allowlist. It does NOT hold here, because _allowlisted()
# below tests the candidate's PREFIX as well as the whole value. Measured,
# with `oracle` and `sap` in the shipped list:
#
#     oracle-client-db-prod-4471   ->  NOT MASKED by this stage
#     sap-payroll-0047821          ->  NOT MASKED by this stage
#     acme-client-db-prod-4471     ->  masked
#
# The first is the exact string the allowlist docstring uses to say this
# cannot happen.
#
# It is kept deliberately. The veto is what makes P3 usable at all: without
# it `sha512-6v6nzO8RiyfE` reads as issuer code + counter, and `CZPE2604...`
# vetoes on CZPE only because the alphabetic prefix is tested separately. A
# whole-token-only veto would hand those back as business keys, and P3 is the
# branch that catches 16 of the 47 gold false negatives this stage exists for.
#
# THE ACCEPTED COST, stated so it is not rediscovered as a bug: a client
# whose internal keys open with a prefix that collides with technical
# vocabulary -- SAP-, ORACLE-, JIRA-, SRV- -- gets no id-stage masking for
# those keys, silently, for the whole engagement. The other layers still see
# them, and the engagement answer is a custom rule (now reachable through the
# proxy via --custom-rules), which is the same answer the allowlist gives for
# a client genuinely named after a technology.
#
# Terms are matched whole and case-folded, which is the allowlist's own
# matching mode; the prefix is extracted first and then matched whole.

# (key, value). Unkeyed, the FIRST config to load decided the veto set for
# every later call in the process, whatever allowlist_dir it passed — so a
# benchmark sweeping allowlists measured the first one under every name.
# gliner._get_model() already keys its cache; this did not.
_allowlist = None
_allowlist_key = None


def _load_allowlist(cfg):
    global _allowlist, _allowlist_key

    d = cfg.get("allowlist_dir") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "allowlist")
    if _allowlist is not None and _allowlist_key == d:
        return _allowlist
    terms, keep = set(), set()
    try:
        import yaml
    except ImportError:
        print("  [ID] PyYAML missing — allowlist veto disabled.")
        _allowlist = (terms, keep)
        _allowlist_key = d
        return _allowlist

    if not os.path.isdir(d):
        print(f"  [ID] No allowlist at {d} — prefix veto disabled.")
        _allowlist = (terms, keep)
        _allowlist_key = d
        return _allowlist

    for name in sorted(os.listdir(d)):
        if not name.endswith((".yaml", ".yml")):
            continue
        try:
            # encoding="utf-8" is not optional. Without it Python uses the
            # platform locale codepage, which on a Czech Windows box is cp1250:
            # elf_forms.yaml (Czech legal-form suffixes — "akciová
            # společnost") then raised UnicodeDecodeError and the whole file
            # was skipped, so the veto set differed from Linux.
            #
            # The raise was the lucky case. cp1250 decodes most byte sequences
            # WITHOUT error, so a term can arrive as mojibake instead: same
            # term count, different term, silently different masking. Compare
            # the term-set hash across platforms, never the count.
            with open(os.path.join(d, name), encoding="utf-8") as fh:
                doc = yaml.safe_load(fh) or {}
        except Exception as e:
            print(f"  [ID] Skipping {name}: {type(e).__name__}")
            continue
        for grp in (doc.get("groups") or {}).values():
            for t in (grp.get("terms") or []):
                terms.add(str(t).lower())
        # `exclude` is the hand-maintained force-KEEP list: a client that is
        # legitimately named after a technology must still be masked, so
        # those terms must not veto anything here either.
        for t in (doc.get("exclude") or []):
            keep.add(str(t).lower())

    print(f"  [ID] Allowlist: {len(terms)} terms, {len(keep)} force-keep")
    _allowlist = (terms, keep)
    _allowlist_key = d
    return _allowlist


def _allowlisted(value, prefix, cfg):
    terms, keep = _load_allowlist(cfg)
    if not terms:
        return False
    v, p = value.lower(), prefix.lower()
    if v in keep or p in keep:
        return False
    return v in terms or p in terms


# ── The rule ─────────────────────────────────────────────────

def _segments(value):
    return [s for s in _SEG.split(value) if s]


def _rule(value, cfg):
    """True when `value` is shaped like an internally minted business key."""
    segs = _segments(value)
    if not segs:
        return False

    # P2 — a fixed-width sequence counter, as a whole segment.
    # Strongest single signal: an issuer that zero-pads has allocated a
    # fixed-width field, which registries and version schemes never do.
    if any(_ZERO_PADDED_SEGMENT.fullmatch(s) for s in segs):
        return True

    # P3 — prefix then one unbroken digit run, long enough that it is not a
    # registry number. Registry numbering is short by construction: CWE-639,
    # AES-256, ISO-8601, RFC-3986, NIST-800-53, RSA-2048, X-509 all sit at
    # four digits or fewer, so the floor is set above them rather than at a
    # value read off the benchmark.
    digits = sum(c.isdigit() for c in value)
    if digits >= cfg["min_total_digits"] and _CONTIGUOUS_TAIL.fullmatch(value):
        # Two guards on this branch, because "letters then digits" alone also
        # describes a hex fragment: ba484314 is 'ba' + 6 digits and would
        # otherwise read as an issuer code plus a counter.
        #   - an all-hex run of 8+ is a hash or an object id, never a key
        #   - an issuer code is written in caps, or the key carries a
        #     separator; the target shape is "uppercase prefix + separator +
        #     digits" and a lowercase unseparated run is neither
        if _HEX_ONLY.fullmatch(value):
            return False
        if not (_UPPER_PREFIX.match(value) or len(segs) > 1):
            return False
        return True

    # P1 — a composite key: several alphabetic segments and a numeric tail.
    # One separator is a standard (AES-256); three or more segments ending in
    # a counter is an organisation's own key space.
    if (len(segs) >= 3 and segs[-1].isdigit()
            and len(segs[-1]) >= cfg["min_tail_digits"]):
        return True

    return False


# ── Public API ───────────────────────────────────────────────

def id_detect(text, config, already=None, debug=False):
    """Detect structured business IDs.

    Args:
        text:    text to scan, already custom/regex-masked by the
                 orchestrator — so UUIDs, JWTs and hashes are placeholders
        config:  pipeline config dict; reads the `id:` block
        already: entities from detectors that ran earlier in the same NER
                 slot; their spans are skipped
        debug:   print each candidate and the gate that rejected it
    """
    if not text or not text.strip():
        return []

    cfg = dict(DEFAULTS)
    cfg.update(config.get("id", {}) or {})
    _STATS["calls"] += 1

    claimed = []
    for e in (already or []):
        s, t = e.get("start"), e.get("end")
        if isinstance(s, int) and isinstance(t, int) and t > s:
            claimed.append((s, t))
    claimed += [(m.start(), m.end()) for m in _PLACEHOLDER.finditer(text)]

    out, seen = [], set()
    for m in _CANDIDATE.finditer(text):
        value, start, end = m.group(), m.start(), m.end()
        _STATS["candidates"] += 1

        if any(start < b and a < end for a, b in claimed):
            _STATS["rejected_claimed"] += 1
            continue
        if not (cfg["min_len"] <= len(value) <= cfg["max_len"]):
            _STATS["rejected_length"] += 1
            continue
        if _PLACEHOLDER.fullmatch(value):
            _STATS["rejected_placeholder"] += 1
            continue
        if _VERSION.fullmatch(value):
            _STATS["rejected_version"] += 1
            continue

        prefix = _segments(value)[0]
        # Strip a trailing digit run from the prefix so CZPE2604150082148
        # is vetoed on CZPE, and sha512-... on sha512 as the allowlist
        # spells it.
        prefix_alpha = re.match(r'[A-Za-z]+', prefix)
        prefix_alpha = prefix_alpha.group() if prefix_alpha else prefix
        if (_allowlisted(value, prefix, cfg)
                or _allowlisted(value, prefix_alpha, cfg)):
            _STATS["rejected_allowlist"] += 1
            if debug:
                print(f"    [ID] allowlisted {value!r} (prefix {prefix_alpha!r})")
            continue

        if not _rule(value, cfg):
            _STATS["rejected_shape"] += 1
            continue

        if (start, end) in seen:
            continue
        seen.add((start, end))
        _STATS["emitted"] += 1
        if debug:
            print(f"    [ID] {ENTITY_TYPE} {value!r}")

        out.append({
            "text": value,
            "entity_type": ENTITY_TYPE,
            "start": start,
            "end": end,
            "score": cfg["score"],
            "stage": "ner",
            "detector": DETECTOR,
        })

    return out


def warmup(config):
    """Load the allowlist once, before timing starts."""
    id_detect("ref DRV-004578 ok", config)
    reset_stats()
