"""Checksum-validated national identifiers — bank accounts and birth numbers.

STANDALONE / EXPERIMENTAL — nothing in pipeline/ imports this file. Wired in
at runtime by benchmark.py (--checksum) through the same NER-slot
monkey-patch as gliner, opf, keyval, proximity and id. Delete this file and
the --checksum flag to revert; no production file is touched.

Why a separate stage rather than config.yaml patterns
------------------------------------------------------
pipeline/regex_stage.py resolves a pattern's `validator` name against its own
module-level VALIDATORS dict, which currently holds _validate_luhn and
_validate_phone_possible. A new checksum therefore needs a new function IN
that file — a config.yaml pattern alone cannot carry one. Until this is
promoted, it lives here.

What a checksum buys, and what it costs
----------------------------------------
Every detector here fires on a SHAPE and then proves the value with published
arithmetic. That inverts the usual precision/recall trade: a bare
`\\d{2,10}/\\d{4}` also matches 12/2026, 1024/4096, 1920/1080, 100/1000 and
1609459200/1000 — dates, ratios, resolutions, fractions, epoch divisions. The
checksum rejects every one of those while keeping real accounts, so this is
one of the rare places where precision is free.

The cost is that a checksum cannot be argued with. If a value does not
satisfy the algorithm, it is not a real identifier of that type, and this
stage will not mask it. See the note on synthetic test data at the bottom.

Algorithms, all verified against published examples before use
---------------------------------------------------------------
IBAN — ISO 13616 structure, ISO 7064 MOD-97-10 check. Move the first four
characters to the end, map letters A-Z to 10-35, read as a decimal integer,
and require it ≡ 1 (mod 97). One implementation covers every IBAN country,
which is why Germany, Italy, Slovakia and the UK need no domestic detector
here. Verified: CZ65 0800 0000 1920 0014 5399, GB82 WEST 1234 5698 7654 32,
DE89 3704 0044 0532 0130 00, IT60 X054 2811 1010 0000 0123 456,
SK31 1200 0000 1987 4263 7541 — and the same five with a mutated check digit
all rejected.

CZ/SK domestic account — vyhláška ČNB č. 169/2011 Sb. `[prefix-]base/bank`,
prefix 0-6 digits, base 2-10 digits, bank code exactly 4 digits. Each digit
gets a weight by position, right to left: 1, 2, 4, 8, 5, 10, 9, 7, 3, 6; the
prefix uses the first six. Prefix and base are checked INDEPENDENTLY and both
must be divisible by 11. Verified against the published worked example
19-2000145399/0800 (prefix sum 11, base sum 121 = 11x11), and the two
published formulations of the weights (right-to-left above, and 6,3,7,9,10,
5,8,4,2,1 left-to-right on a zero-padded string) agree on 20,000 random
numbers. Slovakia inherited the Czechoslovak account structure, so the same
check applies there.

The bank code is NOT covered by the checksum — the algorithm cannot tell a
wrong bank code from a right one. The only guard is the ČNB číselník BA0030
registry, which is why `bank_codes` exists below. It defaults to OFF: an
incomplete or stale code list produces FALSE NEGATIVES at smaller banks, and
per §0 that is the dangerous direction. Populate it from ČNB directly if you
want the tightening.

US ABA routing transit number — 9 digits, weights 3,7,1 repeating, sum
divisible by 10. Verified: 021000021, 011000015, 121000248, 026009593 accept;
021000022 and 123456789 reject.

CZ/SK rodné číslo — YYMMDD/XXX[X]. Numbers issued from 1954 carry a fourth
serial digit and the whole ten-digit number must be divisible by 11. Month is
+50 for women and +20/+70 for numbers issued out of the ordinary sequence.
Nine-digit pre-1954 numbers have no check digit and are accepted on date
plausibility alone.

What is deliberately NOT here
------------------------------
German domestic Kontonummer+BLZ: the Bundesbank publishes roughly a hundred
different check-digit methods (00 through E4), selected per bank by the BLZ.
That is a lookup table against a registry, not an algorithm, and IBAN covers
German accounts anyway.

Italian and UK domestic forms: the Italian BBAN CIN character and the UK
VocaLink modulus-checking tables are likewise registry-driven. IBAN covers
both.

US SSN: has no check digit at all. Any "SSN validator" only tests structural
plausibility (no 000/666/900+ area, no 00 group, no 0000 serial), so it
belongs in a regex pattern, not here.

Pipeline contract
-----------------
    {"text", "entity_type", "start", "end", "score",
     "stage": "ner", "detector": "checksum"}

Offsets index into the text passed in: text[start:end] == entity["text"].
Runs in the NER slot, so regex hits are already placeholders and anything
regex owns never reaches this stage.

A note on synthetic test data
------------------------------
Checksums are an unusually sharp instrument for auditing a benchmark. Run
`checksum_audit()` over a gold set and every value that merely LOOKS like a
national identifier will separate from every value that is one.
"""
import re

DETECTOR = "checksum"

# ── ISO 7064 MOD-97-10, used by IBAN (ISO 13616) ─────────────
# Published BBAN lengths per country. A country absent from the table is
# checked on MOD-97 alone rather than rejected, so a new or missing entry
# costs precision, never recall.
IBAN_LENGTHS = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16,
    "BG": 22, "BH": 22, "BR": 29, "BY": 28, "CH": 21, "CR": 22, "CY": 28,
    "CZ": 24, "DE": 22, "DK": 18, "DO": 28, "EE": 20, "EG": 29, "ES": 24,
    "FI": 18, "FO": 18, "FR": 27, "GB": 22, "GE": 22, "GI": 23, "GL": 18,
    "GR": 27, "GT": 28, "HR": 21, "HU": 28, "IE": 22, "IL": 23, "IS": 26,
    "IT": 27, "JO": 30, "KW": 30, "KZ": 20, "LB": 28, "LC": 32, "LI": 21,
    "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MD": 24, "ME": 22, "MK": 19,
    "MR": 27, "MT": 31, "MU": 30, "NL": 18, "NO": 15, "PK": 24, "PL": 28,
    "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "SA": 24, "SE": 24,
    "SI": 19, "SK": 24, "SM": 27, "TN": 24, "TR": 26, "UA": 29, "VA": 22,
    "VG": 24, "XK": 20,
}

_IBAN_RE = re.compile(r'(?<![A-Za-z0-9])([A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}[ ]?[A-Z0-9]{0,4})(?![A-Za-z0-9])')

# ČNB vyhláška 169/2011 Sb., weights right to left; prefix uses the first six.
_CZ_WEIGHTS = [1, 2, 4, 8, 5, 10, 9, 7, 3, 6]
_CZ_ACCOUNT_RE = re.compile(r'(?<![\d/-])(?:(\d{1,6})-)?(\d{2,10})/(\d{4})(?![\d/-])')

# Rodné číslo. The 9-digit form predates 1954 and has no check digit.
_RC_RE = re.compile(r'(?<![\d/])(\d{6})/(\d{3,4})(?![\d/])')

# ABA routing transit number. Bare 9-digit runs are far too common on their
# own, so this one requires a routing-ish word nearby — the same 50-char
# word-boundary context gate regex_stage already uses for phone_bare_e164.
_ABA_RE = re.compile(r'(?<!\d)(\d{9})(?!\d)')
_ABA_WEIGHTS = [3, 7, 1, 3, 7, 1, 3, 7, 1]
_ABA_CONTEXT = ("routing", "aba", "rtn", "abarouting", "transit", "bank",
                "account", "ach", "wire", "swift", "bic")

_PLACEHOLDER = re.compile(r'\[[A-Z][A-Z0-9_]*\]')
_WORD = re.compile(r'[A-Za-z]+')

DEFAULTS = {
    "iban": True,
    "cz_account": True,
    "rodne_cislo": True,
    "aba_routing": True,
    "bank_codes": None,          # ČNB BA0030; None = do not check (see docstring)
    "context_window": 50,        # chars, matching regex_stage's existing gate
    "score": 1.0,
}

ENTITY_TYPES = {
    "iban": "BANK_ACCOUNT",
    "cz_account": "BANK_ACCOUNT",
    "rodne_cislo": "CZ_RODNE_CISLO",
    "aba_routing": "BANK_ACCOUNT",
}

_STATS = {"calls": 0, "candidates": 0, "rejected_checksum": 0,
          "rejected_claimed": 0, "rejected_context": 0,
          "rejected_bank_code": 0, "emitted": 0}


def checksum_stats():
    return dict(_STATS)


def reset_stats():
    for k in _STATS:
        _STATS[k] = 0


# ── The algorithms ───────────────────────────────────────────

def iban_valid(value):
    """ISO 7064 MOD-97-10 over the rotated, letter-expanded IBAN."""
    s = re.sub(r'\s+', '', value).upper()
    if not re.fullmatch(r'[A-Z]{2}\d{2}[A-Z0-9]{10,30}', s):
        return False
    expected = IBAN_LENGTHS.get(s[:2])
    if expected is not None and len(s) != expected:
        return False
    rotated = s[4:] + s[:4]
    return int(''.join(str(int(c, 36)) for c in rotated)) % 97 == 1


def cz_part_valid(part):
    """One segment of a Czech/Slovak account number, mod 11."""
    if not part:
        return True                      # an absent prefix is not a failure
    if len(part) > 10:
        return False
    total = sum(int(d) * _CZ_WEIGHTS[i] for i, d in enumerate(reversed(part)))
    return total % 11 == 0


def cz_account_valid(prefix, base):
    return cz_part_valid(prefix or "") and cz_part_valid(base)


def aba_valid(digits):
    return (len(digits) == 9
            and sum(int(d) * _ABA_WEIGHTS[i] for i, d in enumerate(digits)) % 10 == 0)


def rodne_cislo_valid(ddmmyy, serial):
    """Date plausibility, plus mod-11 on the ten-digit post-1954 form."""
    yy, mm, dd = int(ddmmyy[0:2]), int(ddmmyy[2:4]), int(ddmmyy[4:6])
    month = mm
    if month > 70:
        month -= 70          # out-of-sequence, women
    elif month > 50:
        month -= 50          # women
    elif month > 20:
        month -= 20          # out-of-sequence, men
    if not (1 <= month <= 12 and 1 <= dd <= 31):
        return False
    if len(serial) == 3:
        return True          # pre-1954, no check digit exists
    return int(ddmmyy + serial) % 11 == 0


# ── Detection ────────────────────────────────────────────────

def _has_context(text, start, end, words, window):
    lo, hi = max(0, start - window), min(len(text), end + window)
    near = {w.lower() for w in _WORD.findall(text[lo:hi])}
    return bool(near & set(words))


def checksum_detect(text, config, already=None, debug=False):
    """Detect national identifiers that pass their published checksum."""
    if not text or not text.strip():
        return []

    cfg = dict(DEFAULTS)
    cfg.update(config.get("checksum", {}) or {})
    _STATS["calls"] += 1

    claimed = []
    for e in (already or []):
        s, t = e.get("start"), e.get("end")
        if isinstance(s, int) and isinstance(t, int) and t > s:
            claimed.append((s, t))
    claimed += [(m.start(), m.end()) for m in _PLACEHOLDER.finditer(text)]

    out, taken = [], []

    def _free(a, b):
        return not any(a < d and c < b for c, d in claimed + taken)

    def _emit(kind, value, start, end):
        taken.append((start, end))
        _STATS["emitted"] += 1
        if debug:
            print(f"    [CHK] {kind:<12} {value!r}")
        out.append({
            "text": value,
            "entity_type": ENTITY_TYPES[kind],
            "start": start,
            "end": end,
            "score": cfg["score"],
            "stage": "ner",
            "detector": DETECTOR,
            "checksum": kind,
        })

    # IBAN first: it is the longest form and subsumes the domestic ones.
    if cfg["iban"]:
        for m in _IBAN_RE.finditer(text):
            value = m.group(1).rstrip()
            start, end = m.start(1), m.start(1) + len(value)
            _STATS["candidates"] += 1
            if not _free(start, end):
                _STATS["rejected_claimed"] += 1
                continue
            if not iban_valid(value):
                _STATS["rejected_checksum"] += 1
                continue
            _emit("iban", value, start, end)

    # Rodné číslo BEFORE the account form. A six-digit account base with a
    # four-digit bank code is textually identical to YYMMDD/XXXX, and the two
    # checksums are independent, so a birth number can satisfy the account
    # rule as well: 880714/3290 passes both. Running the account form first
    # labelled it BANK_ACCOUNT. The birth number is the more specific
    # pattern — it additionally requires a plausible date — so it claims the
    # span first and _free() keeps the account form off it.
    if cfg["rodne_cislo"]:
        for m in _RC_RE.finditer(text):
            _STATS["candidates"] += 1
            if not _free(m.start(), m.end()):
                _STATS["rejected_claimed"] += 1
                continue
            if not rodne_cislo_valid(m.group(1), m.group(2)):
                _STATS["rejected_checksum"] += 1
                continue
            _emit("rodne_cislo", m.group(), m.start(), m.end())

    if cfg["cz_account"]:
        codes = cfg.get("bank_codes")
        for m in _CZ_ACCOUNT_RE.finditer(text):
            _STATS["candidates"] += 1
            if not _free(m.start(), m.end()):
                _STATS["rejected_claimed"] += 1
                continue
            if not cz_account_valid(m.group(1), m.group(2)):
                _STATS["rejected_checksum"] += 1
                continue
            if codes and m.group(3) not in set(codes):
                _STATS["rejected_bank_code"] += 1
                continue
            _emit("cz_account", m.group(), m.start(), m.end())

    if cfg["aba_routing"]:
        for m in _ABA_RE.finditer(text):
            _STATS["candidates"] += 1
            if not _free(m.start(), m.end()):
                _STATS["rejected_claimed"] += 1
                continue
            if not aba_valid(m.group(1)):
                _STATS["rejected_checksum"] += 1
                continue
            # A 9-digit run that happens to satisfy 3-7-1 mod 10 arrives about
            # once every ten numbers, so the arithmetic alone is not evidence.
            if not _has_context(text, m.start(), m.end(), _ABA_CONTEXT,
                                cfg["context_window"]):
                _STATS["rejected_context"] += 1
                continue
            _emit("aba_routing", m.group(1), m.start(1), m.end(1))

    out.sort(key=lambda e: e["start"])
    return out


def warmup(config):
    checksum_detect("iban CZ6508000000192000145399 ok", config)
    reset_stats()


# ── Benchmark-data audit ─────────────────────────────────────

def checksum_audit(text, config=None):
    """Report every candidate and whether it passes, without masking anything.

    A checksum is a sharp instrument for auditing a gold set: a value that
    merely looks like a national identifier separates cleanly from one that
    is. Use it on test data before trusting a recall number for these types.
    """
    rows = []
    for m in _IBAN_RE.finditer(text):
        v = m.group(1).rstrip()
        rows.append(("IBAN", v, iban_valid(v)))

    # Rodné číslo first, and its spans suppress the account form over the
    # same text — the detector resolves the YYMMDD/XXXX ambiguity that way,
    # so the audit must too. Without this a valid birth number is reported
    # as a failing CZ_ACCOUNT and reads like data still to fix.
    rc_spans = []
    for m in _RC_RE.finditer(text):
        ok = rodne_cislo_valid(m.group(1), m.group(2))
        rows.append(("RODNE_CISLO", m.group(), ok))
        if ok:
            rc_spans.append((m.start(), m.end()))
    for m in _CZ_ACCOUNT_RE.finditer(text):
        if any(m.start() < b and a < m.end() for a, b in rc_spans):
            continue
        rows.append(("CZ_ACCOUNT", m.group(),
                     cz_account_valid(m.group(1), m.group(2))))
    window = (config or {}).get("checksum", {}).get(
        "context_window", DEFAULTS["context_window"])
    for m in _ABA_RE.finditer(text):
        # Report the context gate too, or the ABA rows overstate: a random
        # 9-digit run satisfies 3-7-1 mod 10 about one time in ten, so the
        # arithmetic alone is not what the detector acts on.
        ok = (aba_valid(m.group(1))
              and _has_context(text, m.start(1), m.end(1), _ABA_CONTEXT, window))
        rows.append(("ABA", m.group(1), ok))
    return rows
