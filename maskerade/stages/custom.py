"""Custom stage — user-defined per-chat masking rules.

Runs FIRST in the pipeline, before regex/NER/LLM. Pentester intent
is the highest-priority signal — when a user says "mask this exact
string as [INTERNAL_HOST]", that wins over whatever regex would have
matched for the same value.

Each rule is a dict:
    {
        "id":                str,  # stable rule ID for edit/delete
        "value":             str,  # literal string to match
        "placeholder":       str,  # what to substitute (e.g. "[CLIENT_NAME]")
        "placeholder_type":  str,  # ALTERNATIVE to `placeholder` -- see below
        "word_boundary":     bool, # default True; \\b...\\b vs raw substring
        "case_sensitive":    bool, # default False; re.IGNORECASE when False
        "created_at":        float,
    }

TWO PLACEHOLDER MODES, AND WHY THE SECOND ONE EXISTS
----------------------------------------------------
`placeholder` is the original mode: one rule, one fixed placeholder, and the
mask map maps it to the rule's `value`. Unchanged, because library callers
depend on it.

That mode does not round-trip when matching is case-insensitive, which is
the default. A rule for `AcmeBank` matches `acmebank`, `ACMEBANK` and
`AcmeBank`, and all three unmask to the RULE's spelling:

    original  'Host: acmebank.example  X-Vendor: ACMEBANK'
    unmasked  'Host: AcmeBank.example  X-Vendor: AcmeBank'

For the library that was an accepted cosmetic difference. For the PROXY it
is not: the operator's rules file is now the main way a client's own name
reaches the masker, the model's tool calls are unmasked before they execute,
and a replayed request whose path segment came back in the wrong case is a
request that does not reproduce.

`placeholder_type` is therefore the mode the rules file uses: the stage
allocates ONE PLACEHOLDER PER DISTINCT MATCHED STRING, numbered per type.

    rule   value="acmebank.cz"  placeholder_type="CLIENT"
    text   "acmebank.cz ... ACMEBANK.CZ"
    map    [CLIENT_1] -> "acmebank.cz"      [CLIENT_2] -> "ACMEBANK.CZ"

Unmasking is then byte-exact. The cost, stated: two spellings of one host
become two placeholders, so the model sees two values where a human sees
one. That is the honest reading -- they ARE two different byte strings --
but it does spend the "same entity, same placeholder" property that makes
cross-turn reasoning work, on a case difference. If that trade ever looks
wrong, the alternative is `case_sensitive: true` in the rules file plus one
rule per observed spelling, which costs recall on the spelling nobody
anticipated. Over-masking versus a wrong-cased replay: this picks the
replay.

Entities from this mode carry the MATCHED text, so the pipeline-wide
contract `text[start:end] == entity["text"]` holds. Under `placeholder` it
does not, and never did.

How it integrates:
  The orchestrator runs apply_custom_masks() on raw_text BEFORE any
  detection stage. This is symmetric with the existing pattern of
  "regex first, then mask, then run NER/LLM on intermediate text" —
  we just add one more pre-stage so subsequent stages don't re-detect
  the values the user already named.

Why we don't go through masker.apply_masks():
  Custom rules carry word_boundary and case_sensitive flags that the
  standard masker doesn't model. Standard masker is always word-boundary
  and always case-sensitive (it uses re.escape + boundary anchors).
  Putting flag-aware substitution here keeps the standard masker's
  invariants intact and makes the validator's leak-check honest:
  values masked with case_sensitive=False, word_boundary=False might
  still appear in masked_text as substrings, and that's the user's
  explicit choice — we don't flag it as a leak.
"""
import re


def _build_pattern(rule):
    """Compile a regex pattern for a single rule, honoring its flags.

    Returns: (compiled_pattern, replacement_placeholder)
    """
    value = rule["value"]
    if not value:
        return None, None

    pattern_str = re.escape(value)

    # Word-boundary: only anchor when the edge char is a word char.
    # Same logic as masker.word_boundary_pattern() — keeps semantics
    # consistent across the pipeline.
    if rule.get("word_boundary", True):
        if value and re.match(r'\w', value[0]):
            pattern_str = r'(?<!\w)' + pattern_str
        if value and re.match(r'\w', value[-1]):
            pattern_str = pattern_str + r'(?!\w)'

    flags = 0 if rule.get("case_sensitive", False) else re.IGNORECASE

    try:
        # .get(), not ["placeholder"]: a placeholder_type rule has no fixed
        # placeholder and this would KeyError on every rule the proxy's
        # rules file produces. The second element is unused by the caller.
        return re.compile(pattern_str, flags), rule.get("placeholder")
    except re.error:
        # Should be unreachable since re.escape is used, but be defensive.
        return None, None


def _dedupe_placeholders(sorted_rules):
    """Give rules that share a placeholder distinct ones.

    Two rules pointing at the same placeholder used to collapse in the
    mask_map — the second write won, so unmasking restored BOTH values as the
    second one. A pentester adding acme.com and acme.cz, both as
    [CLIENT_DOMAIN], silently got the wrong domain back in the AI's answer with
    no warning anywhere.

    The first rule keeps the bare placeholder; later ones are suffixed
    [CLIENT_DOMAIN_2], [CLIENT_DOMAIN_3], ... Suffixes are chosen so they
    cannot collide with a placeholder the user typed themselves.

    Returns: (rules_with_effective_placeholder, collisions)
    """
    used = {r.get("placeholder") for r in sorted_rules}
    seen = {}
    out, collisions = [], []

    for rule in sorted_rules:
        ph = rule.get("placeholder")
        if not ph:
            # placeholder_type rules allocate per matched string instead, so
            # there is nothing to de-duplicate. They used to be dropped here
            # by this `continue`, which would have silently discarded every
            # rule the proxy's rules file produces.
            if rule.get("placeholder_type"):
                out.append((rule, None))
            continue
        if ph not in seen:
            seen[ph] = 1
            out.append((rule, ph))
            continue
        seen[ph] += 1
        base = ph[1:-1] if ph.startswith("[") and ph.endswith("]") else ph
        n = seen[ph]
        candidate = f"[{base}_{n}]"
        while candidate in used:
            n += 1
            candidate = f"[{base}_{n}]"
        used.add(candidate)
        collisions.append((rule.get("value"), ph, candidate))
        out.append((rule, candidate))

    return out, collisions


_PLACEHOLDER_PARTS_RE = re.compile(r"\[([A-Z][A-Z0-9_]*)_(\d+)\]")


def _seed_from_map(existing_mask_map):
    """Numbering state from the conversation's map, for placeholder_type.

    Returns (type_counters, issued, by_value): the highest index already
    issued per type; the placeholder already issued for each (type, matched
    spelling); and, for a spelling the map holds only under ANOTHER type --
    a detector found "Leo Express" before the rule was added -- that
    placeholder, so one value keeps one placeholder (masker.apply_masks
    keeps the same rule). Duplicates, which the bug described in
    apply_custom_masks left behind, resolve to the lowest (type, index), so
    the choice never depends on the order the map is iterated in.
    """
    counters, issued, best, by_value, vbest = {}, {}, {}, {}, {}
    for ph, value in (existing_mask_map or {}).items():
        m = _PLACEHOLDER_PARTS_RE.fullmatch(ph)
        if not m or not isinstance(value, str):
            continue
        etype, idx = m.group(1), int(m.group(2))
        counters[etype] = max(counters.get(etype, 0), idx)
        key = (etype, value)
        if key not in best or idx < best[key]:
            best[key] = idx
            issued[key] = ph
        if value not in vbest or (etype, idx) < vbest[value]:
            vbest[value] = (etype, idx)
            by_value[value] = ph
    return counters, issued, by_value


def apply_custom_masks(text, custom_masks, existing_mask_map=None):

    """Apply user-defined custom masks to text BEFORE other detection stages.

    Args:
        text:          raw input text
        custom_masks:  list of rule dicts (see module docstring)
        existing_mask_map: the conversation's map so far. placeholder_type
                       rules continue ITS numbering and reuse ITS
                       placeholders. Without it (the library's one-off
                       callers: `rules test`, the wizard's self-check) the
                       numbering starts at 1, as before.

    Returns:
        (masked_text, entities, partial_mask_map)
        - masked_text: text with custom-rule values substituted
        - entities: list of detected entities in standard pipeline shape
                    (one per rule that fired at least once), tagged
                    stage="custom" so masker.apply_masks() bypasses
                    the 3-char minimum length floor (same as "regex")
        - partial_mask_map: {placeholder: original_value} for rules that
                            fired. Merged into existing_mask_map by the
                            orchestrator so unmasker sees them.

    Order: rules are applied longest-value first to prevent shorter
    substrings from masking before longer values that contain them.
    Same Layer-4 protection the standard masker uses.

    Two properties worth knowing when writing rules:

      An apex rule covers the whole estate. The word-boundary anchors only
      bite on word characters, and '.' is not one, so a single rule for
      "acme.com" also masks www.acme.com, internal.acme.com and
      Domain=.acme.com (leaving the subdomain label intact), while correctly
      leaving notacme.com and acme.community alone. Do not add per-subdomain
      rules; they are redundant and multiply placeholders for one host.

      Matching runs against the ORIGINAL text. Every rule sees the same input
      and substitutions are applied in one merged pass at the end, so a rule
      can neither match inside a placeholder an earlier rule produced nor
      report an offset that has already shifted.
    """
    if not custom_masks:
        return text, [], {}

    # Longest value first — "acme-corp" must mask before "acme" if both exist.
    sorted_rules = sorted(custom_masks, key=lambda r: -len(r.get("value", "")))
    resolved, collisions = _dedupe_placeholders(sorted_rules)

    for value, wanted, assigned in collisions:
        print(f"  [CUSTOM] Rule {value!r} wanted {wanted} which is already "
              f"taken; using {assigned} so unmasking stays reversible.")

    entities = []
    mask_map = {}
    spans = []          # (start, end, placeholder) against the ORIGINAL text
    claimed = []        # spans already taken by a longer rule

    # placeholder_type bookkeeping: TYPE -> highest index issued, and
    # (TYPE, matched_text) -> the placeholder already issued for it, so the
    # same spelling in twenty places still gets one placeholder.
    #
    # SEEDED FROM THE CONVERSATION'S MAP since 2026-10-30. Before, both
    # started empty on every call, and the pipeline calls this once per
    # TEXT. So the first spelling in each message became [CLIENT_NAME_1]
    # whatever [CLIENT_NAME_1] already meant, and the pipeline's
    # merged_existing.update() then OVERWROTE the earlier value. Measured on
    # a dogfood session: "Leo Express" was [CLIENT_NAME_1] in one message
    # and "leo" in the next, the map ended holding "leo" twice, replies
    # were unmasked to the wrong spelling, and once history was re-masked
    # from the map, "Leo Express" went out in clear.
    type_counters, matched_placeholder, by_value = _seed_from_map(
        existing_mask_map)

    def overlaps(a, b):
        return any(a < e and b > s for s, e in claimed)

    for rule, placeholder in resolved:
        compiled, _ = _build_pattern(rule)
        if compiled is None:
            continue
        original_value = rule["value"]
        etype = rule.get("placeholder_type")
        fired = []

        # Match against `text`, never a partially-substituted copy: offsets
        # stay valid for the validator and stats, and a short rule cannot eat
        # the inside of a placeholder produced by a longer one.
        for match in compiled.finditer(text):
            start, end = match.start(), match.end()
            if overlaps(start, end):
                continue

            if etype:
                # Per-distinct-match placeholder. `matched` is what was
                # actually in the text, so unmasking restores those bytes.
                matched = text[start:end]
                key = (etype, matched)
                ph = matched_placeholder.get(key) or by_value.get(matched)
                if ph is None:
                    type_counters[etype] = type_counters.get(etype, 0) + 1
                    ph = "[%s_%d]" % (etype, type_counters[etype])
                    matched_placeholder[key] = ph
                    by_value[matched] = ph
                # Returned whether new or reused: the pipeline re-applies
                # the rules only when this map is non-empty, and a rule's
                # case-insensitive, substring-capable match must not be
                # replaced by the masker's exact, word-bounded one just
                # because every spelling was already known.
                mask_map[ph] = matched
                entity_text = matched
            else:
                ph = placeholder
                entity_text = original_value

            spans.append((start, end, ph))
            # One entity PER OCCURRENCE. Emitting a single record per rule
            # under-reported every stat whenever a value appeared more than
            # once.
            entities.append({
                "text": entity_text,
                "entity_type": "CUSTOM",
                "start": start,
                "end": end,
                "score": 1.0,
                "stage": "custom",
            })
            fired.append((start, end))

        if fired:
            # Only THIS rule's spans, not every span collected so far. The
            # previous version re-scanned the whole `spans` list on each
            # firing rule, which was quadratic and re-added spans already
            # present; the result was the same, the work was not.
            claimed.extend(fired)
            if not etype:
                mask_map[placeholder] = original_value

    # Single merged pass, right to left so earlier offsets stay valid.
    masked_text = text
    for start, end, placeholder in sorted(spans, key=lambda x: -x[0]):
        masked_text = masked_text[:start] + placeholder + masked_text[end:]

    return masked_text, entities, mask_map


# ─────────────────────────────────────────────────────────── self-test ──

def _self_test():
    failures = []

    def check(cond, label):
        print(("ok   " if cond else "FAIL ") + label)
        if not cond:
            failures.append(label)

    rules = [{"value": "Leo Express", "placeholder_type": "CLIENT_NAME"},
             {"value": "leo", "placeholder_type": "CLIENT_NAME"}]

    # The 2026-10-30 bug: numbering restarted at 1 on every call.
    m1 = {"[CLIENT_NAME_1]": "Leo Express", "[CLIENT_NAME_2]": "leo"}
    out, _e, mm = apply_custom_masks("host graph.test.leo.com", rules, m1)
    check(out == "host graph.test.[CLIENT_NAME_2].com",
          "a known spelling reuses its placeholder (got %r)" % out)
    check(mm == {"[CLIENT_NAME_2]": "leo"},
          "and returns it, so the pipeline re-applies the rule")
    out, _e, mm = apply_custom_masks("LEO and leo", rules, m1)
    check(mm.get("[CLIENT_NAME_3]") == "LEO"
          and mm.get("[CLIENT_NAME_2]") == "leo",
          "a new spelling continues the numbering (got %r)" % mm)
    check(all(m1.get(k, v) == v for k, v in mm.items()),
          "no existing placeholder is given a new value")

    # Without a map the numbering starts at 1, as for the one-off callers.
    out, _e, mm = apply_custom_masks("leo", rules)
    check(mm == {"[CLIENT_NAME_1]": "leo"}, "no map: numbering starts at 1")

    # A spelling the map holds under another type keeps that placeholder.
    out, _e, mm = apply_custom_masks(
        "Leo Express", rules, {"[ORGANIZATION_4]": "Leo Express"})
    check(out == "[ORGANIZATION_4]",
          "one value keeps one placeholder across types (got %r)" % out)

    # A map with one spelling twice resolves to the lowest index, in any
    # iteration order.
    dup_a = {"[CLIENT_NAME_1]": "leo", "[CLIENT_NAME_2]": "leo"}
    dup_b = dict(reversed(list(dup_a.items())))
    got = {apply_custom_masks("leo", rules, d)[0] for d in (dup_a, dup_b)}
    check(got == {"[CLIENT_NAME_1]"},
          "duplicates resolve the same way in any map order (got %r)" % got)

    print("\n%s" % ("PASSED" if not failures
                    else "%d FAILED: %s" % (len(failures), failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
