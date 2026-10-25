"""Validator — verify that masking was complete.

Runs AFTER masking. Scans the masked output to check whether any
detected entity text is still visible. If it is, something went
wrong in the masking step and we must NOT send to the provider.

This is the last line of defense before data leaves the local
environment. If the validator fails, the pipeline blocks
the request from going to the provider.

Word-boundary semantics:
  The masker (Layer 2) only replaces values at non-word-character
  boundaries. So a 3-char value like "abc" masked at boundaries
  may still appear as a substring of a longer unmasked token
  ("abcdef") elsewhere — that's NOT a leak. We use the same
  boundary rules here so the check matches the masker's contract.
"""
import re

from .masker import word_boundary_pattern


def validate(masked_text, entities, mask_map, strict_warnings=False):
    """Check masked output for leaked sensitive data.

    Args:
        masked_text:     the text after masking was applied
        entities:        all detected entities (from every stage)
        mask_map:        the placeholder → original mapping
        strict_warnings: when True, Check 2's findings also fail the result.
                         Default False — see WHAT `passed` DOES NOT COVER.

    Returns:
        {
            "passed": bool,           # True = safe to send to the provider
            "leaked": [...],          # entities still visible in output
            "warnings": [...],        # non-critical issues
            "warning_items": [...],   # the same, structured
            "stats": {
                "total_entities": int,
                "total_checked": int,
                "total_masked": int,
                "coverage_pct": float,
                "skipped_short": int,
            }
        }

    WHAT `passed` DOES NOT COVER
    ----------------------------
    Two blind spots, both deliberate, both easy to misread as coverage:

    1. `passed` is decided by Check 1 ALONE — "was a DETECTED entity left in
       the output". Check 2 scans for emails and internal IPs that NO stage
       detected, and by default those are warnings. A caller that gates on
       `passed` therefore forwards an email the pipeline never found. That is
       the accepted reading of the recall number, not an oversight: at
       gold recall 0.919 roughly one prose entity in twelve is missed, and
       failing on every missed email would refuse a large share of real
       traffic. `strict_warnings=True` inverts that choice for a caller who
       would rather refuse than under-mask; it is a product decision with a
       measurable refusal rate, so it is OFF until somebody measures it.

    2. The masker's fail-open set (`masker_unmasked`) is invisible here BY
       CONSTRUCTION: validate() tests with the same word_boundary_pattern()
       the masker failed to match on, so every one of those passes at 100%
       coverage. A caller deciding whether it is safe to send must check
       `masker_unmasked` AND `validation`.
    """
    leaked = []
    warnings = []
    warning_items = []

    # ── Check 1: Are any detected entity texts still in the output? ──
    # This is the critical check. If we detected "PenTest130." as a
    # CREDENTIAL and it's still visible after masking, we have a leak.
    #
    # Use the same word-boundary semantics as the masker — otherwise a
    # 3-char value masked at boundaries would falsely show up as
    # "leaked" when it merely appears as a substring of a longer
    # unmasked token elsewhere in the document.
    skipped_short = 0
    for entity in entities:
        e_text = entity["text"]

        # Skip very short matches (2 chars or less) — they create
        # noise. The word "CZ" from a DIČ pattern appearing in
        # "Czech" shouldn't count as a leak.
        #
        # Measure with edge separators removed, mirroring the masker's
        # Layer-1 edge-stripped floor: a value like "21:" (core "21") is
        # intentionally NOT masked, so it must not be reported as a leak
        # here either. Same word-boundary semantics, same drop rule.
        if len(re.sub(r'^\W+|\W+$', '', e_text, flags=re.UNICODE)) <= 2:
            # Counted, not silently discarded. These used to vanish from the
            # denominator's numerator only: `total_masked` was
            # total - len(leaked), so every entity skipped here was reported
            # as MASKED. coverage_pct read higher than coverage, and the gap
            # grew with exactly the short fragments most likely to be wrong.
            skipped_short += 1
            continue

        # Word-boundary search instead of plain `in` substring check.
        # This matches what the masker actually replaced.
        try:
            if re.search(word_boundary_pattern(e_text), masked_text):
                leaked.append({
                    "text": e_text,
                    "entity_type": entity["entity_type"],
                    "stage": entity.get("stage", "unknown"),
                })
        except re.error:
            # Defensive fallback — if the value somehow produces an
            # invalid regex despite re.escape, fall back to substring
            # check so we still flag obvious leaks.
            if e_text in masked_text:
                leaked.append({
                    "text": e_text,
                    "entity_type": entity["entity_type"],
                    "stage": entity.get("stage", "unknown"),
                })

    # ── Check 2: Quick regex scan for obvious PII patterns ──
    # Even if our detection stages missed something, these patterns
    # catch common PII that should never appear in masked output.

    # Emails (but not placeholders like [EMAIL_ADDRESS_1])
    remaining_emails = re.findall(
        r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
        masked_text
    )
    remaining_emails = [e for e in remaining_emails if "[" not in e]
    if remaining_emails:
        warnings.append(f"Unmasked email(s) in output: {remaining_emails[:3]}")
        warning_items.append({"kind": "email", "values": remaining_emails[:3],
                              "count": len(remaining_emails)})

    # Internal IPs — RFC 1918 in full.
    #
    # 172.16/12 was missing while the `internal_ip` regex pattern covers it,
    # so a 172.x address that no stage detected produced no warning at all,
    # purely because two lists of private ranges had drifted apart. The
    # second octet range is 16-31, NOT 0-255: 172.15.x and 172.32.x are
    # public space and must not be reported.
    remaining_ips = re.findall(
        r'\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        r'|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}'
        r'|192\.168\.\d{1,3}\.\d{1,3})\b',
        masked_text
    )
    if remaining_ips:
        warnings.append(f"Unmasked internal IP(s) in output: {remaining_ips[:3]}")
        warning_items.append({"kind": "internal_ip", "values": remaining_ips[:3],
                              "count": len(remaining_ips)})

    # ── Stats ──
    # `checked` is the denominator the coverage number actually describes:
    # entities Check 1 was able to test. Entities skipped above are reported
    # separately rather than folded in as successes.
    total = len(entities)
    checked = total - skipped_short
    masked_count = checked - len(leaked)
    coverage = round(masked_count / checked * 100, 1) if checked > 0 else 100.0

    return {
        "passed": len(leaked) == 0 and not (strict_warnings and warning_items),
        "leaked": leaked,
        "warnings": warnings,
        "warning_items": warning_items,
        "stats": {
            "total_entities": total,
            "total_checked": checked,
            "total_masked": masked_count,
            "coverage_pct": coverage,
            "skipped_short": skipped_short,
        },
    }
