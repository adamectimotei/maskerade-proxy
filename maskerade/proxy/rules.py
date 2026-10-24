# -*- coding: utf-8 -*-
"""Operator-supplied custom masking rules, as a plain text file.

    maskerade proxy --custom-rules engagement.txt

WHY A FILE, AND WHY A PLAIN ONE
-------------------------------
The custom stage is the only layer that carries the client's OWN identity --
their name, their domains, their tenant slugs -- because no detector can know
those in advance. Through the library an operator passes `custom_masks` to
Pipeline.run(). Through the proxy there was no way to supply them at all, so
the stage was unreachable and the client's name was left to org_evidence's
E2 corroboration, whose own design note says the opposite:

    "the custom stage carries the client's own name, entered per engagement.
     That frees ORGANIZATION to be precision-first"

This file closes that gap. The format is a plain line-per-record text file
rather than YAML or JSON for one reason: the operators are pentesters, not
developers, and a mid-engagement edit must not be able to fail on
indentation. A YAML parse error at the moment someone adds a hostname is a
worse outcome than any expressiveness it buys.

THE FORMAT
----------
    # acme bank -- engagement 2026-10
    # one value per line. blank lines and # comments are ignored.

    acmebank.cz
    portal.acmebank.internal
    Acme Bank a.s.        = [CLIENT_NAME]
    ACME-PAYROLL          = CLIENT_SYSTEM

A bare line gets an auto-numbered placeholder, [CLIENT_1], [CLIENT_2], ...
`value = [NAME]` or `value = NAME` sets the type explicitly. Everything
after the first `=` that is surrounded by spaces is the placeholder; a value
containing `=` is therefore still writable as long as it is not spaced like
an assignment (see _split_record).

EVERY PLACEHOLDER ENDS IN _<DIGITS>, AND THAT IS LOAD-BEARING
-------------------------------------------------------------
The stages do not agree on what a placeholder looks like. Three of them
accept any `[UPPERCASE]` token; masker.py, keyval.py and stream.py require a
trailing `_<digits>`:

    masker._CONTAINS_PLACEHOLDER_RE   \\[[A-Z][A-Z0-9_]*_\\d+\\]
    keyval._PLACEHOLDER               \\[[A-Z][A-Z0-9_]*_\\d{1,3}\\]
    stream.PLACEHOLDER_RE             \\[[A-Z][A-Z0-9_]*_\\d+\\]
    regex._PLACEHOLDER_RE             \\[[A-Z][A-Z0-9_]*\\]
    id / checksum._PLACEHOLDER        \\[[A-Z][A-Z0-9_]*\\]

An operator who writes `= [CLIENT_NAME]` would therefore produce a
placeholder that regex and id skip but keyval and the masker do NOT
recognise -- so a later stage can detect a value CONTAINING it and mask the
whole thing, nesting one placeholder inside another and breaking unmasking.
Verified: keyval_detect('{"password":"[CLIENT_NAME]2026!"}') returns
'[CLIENT_NAME]2026!' as a CREDENTIAL.

Rather than widen five regexes -- which is a detection change, because a
value legitimately shaped `[ADMIN]` exists in real logs -- this parser
guarantees the narrow shape: a type is normalised to `[TYPE_n]` with an
index appended, so every placeholder this file can produce is recognised by
every guard in the pipeline. The disagreement still exists in the library
path; it is unreachable from here.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No regular expressions. A rule is a literal string. An operator debugging
their own regex against live client traffic, under time pressure, with the
failure mode "the value was not masked", is not a thing this tool should
make possible. The library's `custom_masks` still accepts the full rule
dict for callers who want the flags.

No CIDR ranges or wildcards. `acmebank.cz` already covers every subdomain:
the word-boundary anchors only bite on word characters and `.` is not one,
so one apex rule masks www.acmebank.cz and internal.acmebank.cz while
leaving notacme.com and acmebank.community alone. That is custom.py's
documented behaviour and it is why per-subdomain rules are redundant.
"""
import os
import re


# A value shorter than this cannot be masked anyway: masker.apply_masks()
# drops entities whose edge-stripped core is under its length floor, so
# accepting one here would produce a rule that silently never fires.
MIN_VALUE_LEN = 3

# A rules file is an operator's own list, not a vocabulary. Past a few
# hundred entries it is a detector being written by hand, the mask map
# becomes unreadable, and the per-request substitution cost stops being
# free. Refuse rather than degrade quietly.
MAX_RULES = 500

# The type part of a placeholder, before the index is appended.
_TYPE_RE = re.compile(r'^[A-Z][A-Z0-9_]*$')

# Any bracketed token. A VALUE containing one of these would nest.
_ANY_PLACEHOLDER_RE = re.compile(r'\[[A-Z][A-Z0-9_]*\]')

DEFAULT_TYPE = "CLIENT"


class RulesError(ValueError):
    """The rules file could not be used. Names the line and the reason.

    Raised rather than skipping the bad line. A rules file is the operator's
    statement of what must never reach the provider, so a line they believe
    is in force and is not is the one failure this file must not have.
    """


def _split_record(line):
    """Return (value, type_or_None) for one non-comment line.

    The separator is ` = ` with surrounding whitespace, so a value that
    itself contains `=` -- a base64 tail, a query fragment, `k=v` -- stays
    intact as long as it is not written with spaces around an `=`. On a line
    with several, the LAST one wins, because the placeholder is at the end.
    """
    parts = re.split(r'\s+=\s+', line)
    if len(parts) == 1:
        return line.strip(), None
    return " = ".join(parts[:-1]).strip(), parts[-1].strip()


def _normalise_type(raw, lineno):
    """`[CLIENT_NAME]`, `CLIENT_NAME`, `client name` -> `CLIENT_NAME`."""
    t = raw.strip()
    if t.startswith("[") and t.endswith("]"):
        t = t[1:-1].strip()
    t = re.sub(r'[\s\-]+', "_", t).upper()
    t = re.sub(r'[^A-Z0-9_]', "", t)
    t = re.sub(r'_+', "_", t).strip("_")
    # An operator who wrote `= [CLIENT_NAME_1]` gets CLIENT_NAME: the index
    # is allocated here, so carrying theirs through would collide with it.
    t = re.sub(r'_\d+$', "", t)
    if not t or not _TYPE_RE.match(t):
        raise RulesError(
            "line %d: %r is not usable as a placeholder name. Use letters, "
            "digits and underscores, for example [CLIENT_NAME]." % (lineno, raw))
    return t


def parse_rules_text(text, source="<string>"):
    """Parse the file's contents into custom-stage rule dicts.

    Returns a list of dicts shaped for stages/custom.py, each carrying
    `placeholder_type` so that stage allocates one placeholder per DISTINCT
    MATCHED TEXT rather than one per rule -- which is what makes unmasking
    return the bytes that were actually in the traffic. See custom.py.
    """
    rules = []
    seen_values = {}
    type_counts = {}

    # Strip a UTF-8 BOM. PowerShell 5.1's `Set-Content -Encoding utf8` writes
    # one, and that is the command an operator on this project will reach for,
    # so this is the common case rather than an edge case.
    #
    # Without this the BOM becomes part of the FIRST LINE: a leading `#` no
    # longer starts the line so the comment is parsed as a rule, and -- the
    # dangerous half -- a first line that is a real value becomes the rule
    # '﻿acmebank.cz', which matches nothing. The operator's first rule
    # silently never fires, which is the one failure this file exists to
    # prevent. Verified against a file written by that exact command.
    if text.startswith("﻿"):
        text = text[1:]

    for lineno, raw in enumerate(text.splitlines(), 1):
        # Strip the BOM per line as well: a file concatenated from two
        # BOM-prefixed files carries one mid-stream, and U+FEFF is also a
        # zero-width no-break space that survives a copy-paste invisibly.
        line = raw.replace("﻿", "").strip()
        if not line or line.startswith("#"):
            continue

        value, type_raw = _split_record(line)

        if not value:
            raise RulesError(
                "line %d: a placeholder with no value (%r)." % (lineno, raw))
        if len(value.strip()) < MIN_VALUE_LEN:
            raise RulesError(
                "line %d: %r is shorter than %d characters. The masker's "
                "length floor would drop it, so the rule would silently "
                "never fire." % (lineno, value, MIN_VALUE_LEN))
        if _ANY_PLACEHOLDER_RE.search(value):
            raise RulesError(
                "line %d: the value %r contains something shaped like a "
                "placeholder. Masking it would nest one placeholder inside "
                "another and the reply could not be unmasked." % (lineno, value))
        if "[" in value or "]" in value:
            raise RulesError(
                "line %d: the value %r contains a square bracket. Brackets "
                "delimit placeholders, so a value carrying one cannot be "
                "substituted safely." % (lineno, value))

        lower = value.casefold()
        if lower in seen_values:
            raise RulesError(
                "line %d: %r repeats line %d (matching ignores case). Two "
                "rules for one value would compete for the same text."
                % (lineno, value, seen_values[lower]))
        seen_values[lower] = lineno

        etype = _normalise_type(type_raw, lineno) if type_raw else DEFAULT_TYPE
        type_counts[etype] = type_counts.get(etype, 0) + 1

        rules.append({
            "id": "rules:%s:%d" % (os.path.basename(source), lineno),
            "value": value,
            # placeholder_type, not placeholder: the index is allocated per
            # distinct matched text by the custom stage.
            "placeholder_type": etype,
            # Defaults, stated rather than implied. Case-insensitive because
            # DNS is, and a host appears in mixed case across one capture;
            # word_boundary because an apex rule must cover subdomains
            # without matching inside unrelated words.
            "word_boundary": True,
            "case_sensitive": False,
            "source": source,
            "lineno": lineno,
        })

        if len(rules) > MAX_RULES:
            raise RulesError(
                "more than %d rules in %s. That is a detector written by "
                "hand: the mask map stops being readable and every request "
                "pays for the whole list. Narrow it to the client's own "
                "identifiers." % (MAX_RULES, source))

    return rules


def load_rules(path):
    """Read and parse a rules file. Raises RulesError with the reason."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise RulesError(
            "cannot read the custom rules file %s (%s). Fix the path, or "
            "drop --custom-rules to run without operator rules and accept "
            "that the client's own name is left to the detectors." % (path, e))
    except UnicodeDecodeError as e:
        raise RulesError(
            "%s is not valid UTF-8 (%s). Save it as UTF-8; a mis-decoded "
            "hostname would be a rule that never matches." % (path, e))
    return parse_rules_text(text, source=path)


class RulesFile:
    """A rules file plus its mtime, reloaded when it changes.

    Reload matters operationally: an engagement turns up a new hostname at
    11pm and the operator must not have to restart a daemon that holds the
    session state. A reload that FAILS keeps the previous rules and says so
    -- running on the last known-good list is strictly safer than dropping
    to none, which is what an exception here would do.
    """

    def __init__(self, path=None):
        self.path = path
        self.rules = []
        self.error = None
        self._stamp = None
        if path:
            self.rules = load_rules(path)        # first load is allowed to raise
            self._stamp = self._mtime()

    def _mtime(self):
        try:
            st = os.stat(self.path)
            return (st.st_mtime, st.st_size)
        except OSError:
            return None

    def maybe_reload(self):
        """Returns (changed, error_or_None)."""
        if not self.path:
            return False, None
        stamp = self._mtime()
        if stamp == self._stamp:
            return False, None
        self._stamp = stamp
        try:
            self.rules = load_rules(self.path)
            self.error = None
            return True, None
        except RulesError as e:
            self.error = str(e)
            return False, str(e)

    def summary(self):
        if not self.path:
            return "none"
        types = {}
        for r in self.rules:
            types[r["placeholder_type"]] = types.get(r["placeholder_type"], 0) + 1
        detail = ", ".join("%s=%d" % kv for kv in sorted(types.items()))
        return "%s (%d rule%s%s)" % (self.path, len(self.rules),
                                     "" if len(self.rules) == 1 else "s",
                                     "; " + detail if detail else "")

    def __len__(self):
        return len(self.rules)


# ─────────────────────────────────────────────────────────── self-test ──

def _self_test():
    import sys
    failures = []

    def check(cond, label):
        print(("ok   " if cond else "FAIL ") + label)
        if not cond:
            failures.append(label)

    def refuses(text, label):
        try:
            parse_rules_text(text)
        except RulesError as e:
            print("ok   %s  (%s)" % (label, str(e)[:60]))
            return
        print("FAIL " + label)
        failures.append(label)

    # A BOM-prefixed file, which is what PowerShell 5.1's
    # `Set-Content -Encoding utf8` produces.
    bom = parse_rules_text("﻿# a comment\nacmebank.cz\n")
    check(len(bom) == 1, "a BOM does not turn the first comment into a rule")
    check(bom[0]["value"] == "acmebank.cz",
          "a BOM does not corrupt the first value")
    bom2 = parse_rules_text("﻿acmebank.cz\n")
    check(bom2[0]["value"] == "acmebank.cz",
          "a BOM on a bare first value is stripped")

    good = """
# acme bank
acmebank.cz
portal.acmebank.internal
Acme Bank a.s.      = [CLIENT_NAME]
ACME-PAYROLL        = CLIENT_SYSTEM
tenant-slug-x       = client name
"""
    rules = parse_rules_text(good, source="engagement.txt")
    check(len(rules) == 5, "five rules parsed, comments and blanks ignored")
    check(rules[0]["value"] == "acmebank.cz", "bare value kept verbatim")
    check(rules[0]["placeholder_type"] == "CLIENT", "bare line gets CLIENT")
    check(rules[2]["placeholder_type"] == "CLIENT_NAME",
          "bracketed type is accepted")
    check(rules[3]["placeholder_type"] == "CLIENT_SYSTEM",
          "unbracketed type is accepted")
    check(rules[4]["placeholder_type"] == "CLIENT_NAME",
          "lowercase spaced type is normalised")
    check(all(r["word_boundary"] and not r["case_sensitive"] for r in rules),
          "defaults are word-boundary, case-insensitive")

    # A value containing '=' survives when not spaced as an assignment.
    r = parse_rules_text("sig=abc123def456\n")
    check(r[0]["value"] == "sig=abc123def456", "unspaced '=' stays in the value")
    r = parse_rules_text("a=b=c  = [CLIENT_NAME]\n")
    check(r[0]["value"] == "a=b=c" and r[0]["placeholder_type"] == "CLIENT_NAME",
          "last spaced '=' separates the placeholder")

    # An operator-supplied index is stripped; this parser allocates it.
    r = parse_rules_text("acme.cz = [CLIENT_NAME_7]\n")
    check(r[0]["placeholder_type"] == "CLIENT_NAME",
          "an operator-written index is dropped, not carried through")

    print()
    refuses("ab\n", "a value under the masker's length floor")
    refuses("= [CLIENT_NAME]\n", "a placeholder with no value")
    refuses("acme [CLIENT_1] cz\n", "a value containing a placeholder")
    refuses("acme]cz\n", "a value containing a bracket")
    refuses("acme.cz\nACME.CZ\n", "the same value twice, ignoring case")
    refuses("acme.cz = [123]\n", "a placeholder name that is not usable")
    refuses("acme.cz = ***\n", "a placeholder of punctuation only")
    refuses("\n".join("host%d.acme.cz" % i for i in range(MAX_RULES + 2)),
            "more rules than the cap")

    print("\n%s" % ("PASSED" if not failures
                    else "%d FAILED: %s" % (len(failures), failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
