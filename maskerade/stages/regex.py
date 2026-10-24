"""Regex detection stage — deterministic pattern matching.

This is the first and fastest detection layer. It catches structured
sensitive data with known formats: emails, tokens, Czech IDs, IPs, etc.

Why regex first?
  - 100% recall on matching patterns (never misses a valid match)
  - Sub-millisecond execution (no model loading, no API calls)
  - Deterministic (same input = same output, always)
  - Catches things LLMs sometimes miss (long tokens, exact formats)

What regex CAN'T do (that's why NER and LLM stages exist):
  - Detect names ("Jan Novák" doesn't have a regex pattern)
  - Understand context ("admin" as CSS class vs user role)
  - Handle unstructured secrets (arbitrary passwords in JSON)
"""
import re


# Placeholders emitted by earlier stages, e.g. [CLIENT_DOMAIN], [EMAIL_ADDRESS_1].
_PLACEHOLDER_RE = re.compile(r'\[[A-Z][A-Z0-9_]*\]')


# ──────────────────────────────────────────────────────────────
# Built-in patterns — always run, regardless of config
# These are universal patterns that apply to any pentest engagement
# ──────────────────────────────────────────────────────────────

BUILTIN_PATTERNS = {
    "email": {
        "pattern": r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
        "entity_type": "EMAIL_ADDRESS",
    },
    "phone_intl": {
        # Explicit "+" international format: +420 777 123 456, +1-555-123-4567.
        # A literal + is unambiguous, so this needs no validator.
        #
        # The ITU "00" exit-code form is deliberately NOT accepted here. It
        # is indistinguishable from ordinary zero-padded data — metric
        # values, zero-filled IDs — so it is handled by phone_bare_e164
        # below, which validates the country code and demands context.
        "pattern": r'\+\d{1,3}[\s-]?\d{2,4}[\s-]?\d{2,4}[\s-]?\d{2,4}',
        "entity_type": "PHONE_NUMBER",
    },
    "phone_bare_e164": {
        # Country code written WITHOUT a + or 00, which is how phone numbers
        # most often appear in JSON bodies: {"number": "420741598696"}.
        #
        # A bare digit run is far too ambiguous on its own — epoch
        # timestamps, order numbers and record IDs all share the shape — so
        # this pattern only fires when BOTH hold:
        #   1. the digits parse as a possible E.164 number (see
        #      _validate_phone_possible), which checks the country calling
        #      code and the numbering-plan length for that country;
        #   2. a phone-ish word appears within 50 chars, matched on word
        #      boundaries (context_mode: word) so "tel" cannot match inside
        #      "hotel" or "telemetry".
        #
        # Requires the `phonenumbers` package. Without it the pattern is
        # skipped rather than run unvalidated — see _validate_phone_possible.
        "pattern": r'(?<![\d+])(?:00)?[1-9]\d{8,14}(?![\d])',
        "entity_type": "PHONE_NUMBER",
        "validator": "_validate_phone_possible",
        "context": ["phone", "telephone", "tel", "mobile", "cell", "msisdn",
                    "whatsapp", "whatsap", "sms", "fax", "number", "num",
                    "contact", "telefon"],
        "context_mode": "word",
    },
    # NOTE ON ORDER: _deduplicate() keeps the longest match and breaks an
    # exact-span tie by insertion order. A 13-19 digit phone number can
    # satisfy Luhn by chance, so the phone patterns are declared BEFORE
    # credit_card to make the tie resolve to PHONE_NUMBER. Both mask the
    # value either way — this only decides which label it carries.
    "credit_card": {
        # Matches 13-19 digits with optional spaces/dashes
        "pattern": r'\b(?:\d[ -]*?){13,19}\b',
        "entity_type": "CREDIT_CARD",
        "validator": "_validate_luhn",
    },

}


# ──────────────────────────────────────────────────────────────
# Null/empty value markers — never mask these
#
# Why: Tier 2 capture-group patterns (password_value, secret_value,
# username_value, etc.) match the literal value after key=. When the
# value is "null" / "" / "undefined", masking it produces nonsense like
# `originalCreator: [CREDENTIAL_1]` for what was just an empty field.
#
# This is a global filter applied to every regex match — adding it
# centrally means new patterns get the protection automatically,
# instead of every pattern needing its own (?!null|...) lookahead.
#
# Extend at runtime via config.yaml:
#     regex:
#       null_value_markers: [TBD, REDACTED, XXX]
# ──────────────────────────────────────────────────────────────

NULL_VALUE_MARKERS = frozenset({
    # Programming-language nulls
    "null", "nil", "none", "undefined", "void", "missing",
    # Booleans — not sensitive, just pollute the mask map
    "true", "false",
    # Human "no value" markers
    "n/a", "na", "n.a.", "nan", "empty", "blank", "unset", "unknown",
    # Wrapped null variants from logs/UIs
    "(null)", "<null>", "[null]", "{null}",
    "(none)", "<none>", "[none]",
    # Placeholder dashes / dots / asterisks
    "-", "--", "---", "...", "***",
})


def _is_null_value(text, extra_markers=frozenset()):
    """Return True if matched text is a null/empty marker that shouldn't be masked.

    Handles:
      - Empty / whitespace-only strings
      - Surrounding quotes: "null", 'null', `null` (any depth)
      - Case-insensitive marker comparison
      - Extra markers from config (e.g., project-specific TBD, REDACTED)
    """
    if not text or not text.strip():
        return True

    stripped = text.strip()

    # Peel off matching surrounding quotes — "null" should be treated as null.
    # Loop handles nested quoting like '"null"' if it ever appears.
    while len(stripped) >= 2 and stripped[0] in '"\'`' and stripped[0] == stripped[-1]:
        stripped = stripped[1:-1].strip()
        if not stripped:
            return True

    return stripped.lower() in (NULL_VALUE_MARKERS | extra_markers)


def _validate_luhn(number_str):
    """Validate credit card number using the Luhn algorithm.

    Without this, the credit card regex would match any 13-19 digit
    string — order IDs, timestamps, random numbers. Luhn cuts
    false positives by ~99%.
    """
    digits = [int(d) for d in re.sub(r'[\s-]', '', number_str) if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _validate_phone_possible(number_str):
    """Validate that a bare digit run is a plausible E.164 phone number.

    Uses libphonenumber's numbering-plan metadata, which encodes the
    country calling codes and permitted national-number lengths for every
    country. That is what separates a phone number from an epoch
    timestamp or a record ID of the same length, and it stays correct for
    targets in any country without a hand-written table.

    is_possible_number, not is_valid_number: possibility checks the
    country code and length, validity additionally requires the number to
    fall in a currently assigned range. Test accounts, staging data and
    freshly allocated ranges routinely fail validity while still being
    real phone numbers in the target's data — and those must be masked.

    Returns False when `phonenumbers` is not installed, so the pattern is
    skipped rather than run unvalidated. A bare digit run without this
    check would mask timestamps and IDs across the whole corpus.
    """
    try:
        import phonenumbers
    except ImportError:
        if not _PHONE_LIB_WARNED:
            print("  [REGEX] phonenumbers not installed — bare phone-number "
                  "detection disabled (pip install phonenumbers). Numbers "
                  "written with + or 00 are unaffected.")
            _warn_phone_lib_once()
        return False

    digits = re.sub(r'[\s()\-.]', '', number_str)
    if digits.startswith("00"):
        digits = digits[2:]
    if not digits.isdigit():
        return False
    try:
        parsed = phonenumbers.parse("+" + digits, None)
    except Exception:
        return False
    return phonenumbers.is_possible_number(parsed)


_PHONE_LIB_WARNED = False


def _warn_phone_lib_once():
    global _PHONE_LIB_WARNED
    _PHONE_LIB_WARNED = True


# Validator name -> function. Patterns reference validators by name so
# config.yaml can use them too.
VALIDATORS = {
    "_validate_luhn": _validate_luhn,
    "_validate_phone_possible": _validate_phone_possible,
}


def regex_detect(text, config):
    """Run all regex patterns against text.

    Args:
        text: the full prompt text to scan
        config: pipeline config dict (contains custom patterns)

    Returns:
        List of detected entities, each a dict:
        {
            "text": "jan.novak@acme.cz",     # the matched text
            "entity_type": "EMAIL_ADDRESS",   # category label
            "start": 142,                     # position in text
            "end": 160,
            "score": 1.0,                     # always 1.0 for regex
            "stage": "regex",                 # which stage found it
        }
    """
    entities = []

    # Merge built-in + config patterns
    all_patterns = dict(BUILTIN_PATTERNS)
    regex_config = config.get("regex", {})
    config_patterns = regex_config.get("patterns", {})
    for name, pdef in config_patterns.items():
        all_patterns[name] = pdef

    # Build the runtime null-marker set: built-in + project-specific from config.
    # Lowercase everything for case-insensitive comparison in _is_null_value().
    extra_markers = frozenset(
        m.lower() for m in regex_config.get("null_value_markers", []) if m
    )

    for name, pdef in all_patterns.items():
        pattern = pdef["pattern"]
        entity_type = pdef["entity_type"]
        validator_name = pdef.get("validator")
        context_words = pdef.get("context", [])
        context_mode = pdef.get("context_mode", "substring")

        try:
            for match in re.finditer(pattern, text):
                # If the pattern has a capture group, mask only the captured
                # portion (the value). The surrounding context (key name,
                # header, cookie name) stays untouched.
                # e.g., "password":"PenTest130." → group(0) is the full match,
                #        group(1) is just "PenTest130." → only that gets masked.
                #
                # Use match.lastindex rather than hardcoding group(1) so
                # patterns can have MULTIPLE alternative capture groups —
                # e.g. a quoted-value branch vs an unquoted-value branch,
                # where only one of them participates in any given match.
                # lastindex tells us which one actually fired. For
                # single-group patterns this is always 1, so existing
                # patterns behave exactly as before.
                if match.lastindex:
                    matched_text = match.group(match.lastindex).strip()
                    start = match.start(match.lastindex)
                    end = match.end(match.lastindex)
                    # Unvalidated: the pattern guarantees the KEY, not the
                    # value. Tagged so masker.apply_masks() holds it to the
                    # 3-char floor instead of granting the regex bypass.
                    is_capture = True
                else:
                    matched_text = match.group(0).strip()
                    start = match.start()
                    end = match.end()
                    is_capture = False

                # Skip empty matches
                if not matched_text:
                    continue

                # Skip anything overlapping a placeholder left by an earlier
                # stage. The custom stage runs first, so a Host header can
                # already read "www.[CLIENT_DOMAIN]"; without this the
                # host_header pattern re-masks the whole thing, the same host
                # ends up under two different placeholders in one message, and
                # the mask_map gains a nested entry that unmasking resolves
                # only by luck of ordering. ner_stage has always had this
                # guard; regex_stage did not.
                if _PLACEHOLDER_RE.search(matched_text):
                    continue

                # Skip null/empty value markers — applies to every pattern.
                # Catches: password:null, secret:"", token:undefined,
                #          username:N/A, originalCreator:null, etc.
                if _is_null_value(matched_text, extra_markers):
                    continue

                # Run validator function if specified (Luhn for credit
                # cards, E.164 plausibility for bare phone numbers).
                validator = VALIDATORS.get(validator_name) if validator_name else None
                if validator and not validator(matched_text):
                    continue

                # Context check: if context words are defined, require at least
                # one to appear within 50 chars of the match. Tight window
                # prevents cross-field matching (e.g., "ico" in one JSON field
                # causing a different 8-digit number to match as IČO).
                #
                # context_mode: "word" matches on word boundaries instead of
                # substrings. Substring matching is right for markers that
                # carry punctuation ("s.r.o.", "IČO"), but wrong for short
                # words — plain "tel" hits inside "hotel" and "telemetry",
                # which silently turns a context gate into no gate at all.
                # Default stays "substring" so existing patterns are unchanged.
                if context_words:
                    window_start = max(0, start - 50)
                    window_end = min(len(text), end + 50)
                    window = text[window_start:window_end]
                    if context_mode == "word":
                        matched_ctx = any(
                            re.search(r'\b' + re.escape(ctx) + r'\b', window, re.IGNORECASE)
                            for ctx in context_words
                        )
                    else:
                        matched_ctx = any(
                            ctx.lower() in window.lower() for ctx in context_words
                        )
                    if not matched_ctx:
                        continue

                entities.append({
                    "text": matched_text,
                    "entity_type": entity_type,
                    "start": start,
                    "end": end,
                    "score": 1.0,
                    "stage": "regex",
                    "capture_group": is_capture,
                })

        except re.error as e:
            # Bad regex pattern in config — warn but don't crash
            print(f"  [REGEX] Warning: invalid pattern '{name}': {e}")
            continue

    # Remove overlapping detections (keep longest match)
    entities = _deduplicate(entities)
    return entities


def _deduplicate(entities):
    """Remove overlapping detections, keeping the longest match.

    Example: if both "email" and "api_key_header" match overlapping
    regions, keep whichever matched more text.
    """
    if len(entities) <= 1:
        return entities

    # Sort by start position, then by length descending (longest first)
    entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))

    result = []
    last_end = -1
    for e in entities:
        if e["start"] >= last_end:
            result.append(e)
            last_end = e["end"]

    return result
