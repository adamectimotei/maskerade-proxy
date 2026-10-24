"""Intrinsic-form detection — checksum validation, entropy scoring, format standards.

Pipeline stage. Runs after regex (so regex hits are already placeholders) and
before NER/LLM, on the same intermediate text, toggled by stages.form.

The idea
--------
Sensitive data has three orthogonal properties. A detector generalises only
if it keys on one of them universally:

    structural position   where it sits         -> keyval_stage.py
    semantic meaning      what it means         -> NER / LLM
    intrinsic form        what it *is*, math    -> THIS FILE

The existing regex stage nominally covers form, but it is a finite enumerated
list — it knows the formats that existed when it was written. This stage keys
on properties that hold regardless of who issued the value:

    checksum validity   a mod-97 or Luhn check covers an infinite set of
                        values under a global standard (every IBAN that will
                        ever be issued validates the same way)
    randomness          secrets are random by construction; entropy detects
                        them without knowing the issuer
    encoding structure  base64/base58check/PEM/JWT are format standards, not
                        app conventions

Nothing here is tuned to the test corpus. Every validator, threshold and
allow-list entry is traceable to a published standard or a universal property
of the format, cited inline. See the self-test at the bottom: it constructs
synthetic valid values (an IBAN from a country absent from the repo, a
Luhn-valid card, an EIP-55 address) purely from the standards, and asserts the
validators fire — if a validator only works on corpus values it is not a
validator.

Pipeline contract
-----------------
form_detect(text, config, debug=False) -> list of entity dicts:

    {"text", "entity_type", "start", "end", "score",
     "stage": "ner",        # keeps masker/validator semantics identical
     "detector": "form"}    # benchmark attribution only; pipeline ignores it

Offsets index into the text passed in: text[start:end] == entity["text"].

Detectors run high-precision first (checksums, then named formats), each
claiming its spans, and the probabilistic entropy net runs last over the
unclaimed remainder. Placeholders from earlier stages ([JWT_TOKEN_1]) are
pre-claimed so nothing masks a placeholder — this matters most for entropy,
which finds placeholders attractive.

Length floor / stage tag
------------------------
masker.py bypasses its 3-char minimum only for stage=="regex". We emit
stage=="ner", so we are subject to the floor — but the shortest scheme here
(ABA routing, 9 digits) sits far above 3, so the floor never binds and we do
NOT need to claim stage=="regex" (which would also change validator/dedup
semantics). Entity types are specific (CREDIT_CARD, IBAN, ...) and never
USER_ID / EMPLOYEE_ID / SENSITIVE_FIELD, so masker.is_low_risk_numeric_id()
never strips an all-numeric checksummed value.

Interaction with regex / Presidio
---------------------------------
regex runs first and its hits are already placeholders when we see the text,
so we never re-report them. When --ner (Presidio) runs alongside --form, both
may emit IBAN / CREDIT_CARD / CRYPTO for the same span; those collapse to one
mask_map entry, so no false-positive inflation. Measure this stage by recall
delta, not attribution count.

Config (config["form"], all optional; benchmark.py pushes --form-* here)
------------------------------------------------------------------------
    entropy_method       "bpe" | "shannon" | "both" | "off"   default "bpe"
                         bpe     -> BPE token efficiency only (PRODUCTION
                                    DEFAULT; 98.6% vs 70.4% recall on
                                    CredData, see BPE_THRESHOLD_DEFAULT)
                         shannon -> character-level randomness only
                         both    -> Shannon AND BPE (BPE gates Shannon's
                                    word-like false positives; it is a gate,
                                    not a union, so "both" is never noisier
                                    than "shannon" alone)
                         shannon/both are kept for benchmark comparison —
                         pass --form-entropy-method to select them per run.

    Every mode that needs the BPE encoder degrades to Shannon with a loud
    error when cl100k_base is unavailable (see EncoderUnavailableError).
    Silently detecting nothing is the one outcome this stage must not have.
    shannon_base64       float   default 4.5   (detect-secrets --base64-limit)
    shannon_hex          float   default 3.0   (detect-secrets --hex-limit)
    bpe_threshold        float   default 2.2   chars-per-token; values BELOW
                         this fire. See BPE_THRESHOLD_DEFAULT for provenance.
    min_entropy_len      int     default 20    (noise floor, flag-exposed)
"""
import base64
import hashlib
import json
import os
import math
import re


# ══════════════════════════════════════════════════════════════════════
# Placeholder guard — never emit an entity for, or inside, a placeholder.
# masking a placeholder produces nested-placeholder corruption.
# ══════════════════════════════════════════════════════════════════════
# Index width is unbounded: a long chat's cumulative mask_map passes 1000
# placeholders, and a guard that stops at three digits would let the entropy
# net mask the inside of [HIGH_ENTROPY_SECRET_1000]. Matches masker.py's
# _CONTAINS_PLACEHOLDER_RE.
PLACEHOLDER_RE = re.compile(r'\[[A-Z][A-Z0-9_]*_\d+\]')


# ══════════════════════════════════════════════════════════════════════
# Reusable checksum primitives
# ══════════════════════════════════════════════════════════════════════

def luhn_valid(number):
    """Luhn / mod-10 (ISO/IEC 7812-1). Double every 2nd digit from the right."""
    digits = [int(c) for c in re.sub(r'\D', '', number)]
    if len(digits) < 2:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _alnum_to_digits(s):
    """A=10 .. Z=35, digits unchanged (ISO 7064 / ISO 13616 letter conversion)."""
    return ''.join(str(int(c, 36)) if c.isalpha() else c for c in s)


def iso7064_mod97_10_valid(s):
    """ISO/IEC 7064 MOD 97-10 over an already-digit string: value mod 97 == 1."""
    return int(s) % 97 == 1


# ── Verhoeff (Verhoeff 1969) — used by Aadhaar (UIDAI) ──
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 9, 7, 6, 3, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number):
    """Verhoeff check: trailing digit is the check digit; result must be 0."""
    digits = [int(c) for c in number if c.isdigit()]
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][d]]
    return c == 0


# ── Damm (Damm 2004) — totally anti-symmetric quasigroup ──
_DAMM = [
    [0, 3, 1, 7, 5, 9, 8, 6, 4, 2],
    [7, 0, 9, 2, 1, 5, 4, 8, 6, 3],
    [4, 2, 0, 6, 8, 7, 1, 3, 5, 9],
    [1, 7, 5, 0, 9, 8, 3, 4, 2, 6],
    [6, 1, 2, 3, 0, 4, 5, 9, 7, 8],
    [3, 6, 7, 4, 2, 0, 9, 5, 8, 1],
    [5, 8, 6, 9, 7, 2, 0, 1, 3, 4],
    [8, 9, 4, 5, 3, 6, 2, 0, 1, 7],
    [9, 4, 3, 8, 6, 1, 7, 2, 0, 5],
    [2, 5, 8, 1, 4, 3, 6, 7, 9, 0],
]


def damm_valid(number):
    interim = 0
    for c in number:
        if c.isdigit():
            interim = _DAMM[interim][int(c)]
    return interim == 0


# ══════════════════════════════════════════════════════════════════════
# Keccak-256 (pure stdlib) — Ethereum EIP-55 needs original Keccak, which
# differs from hashlib.sha3_256 (FIPS-202) by one padding byte, so we cannot
# reuse hashlib here. Reference Keccak-f[1600]. Verified in the self-test
# against keccak256(b"") == c5d2460186...5d85a470.
# ══════════════════════════════════════════════════════════════════════
_KECCAK_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
    0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]
_KECCAK_ROT = [
    [0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56], [27, 20, 39, 8, 14],
]


def _keccak_f(state):
    def rol(x, n):
        return ((x << n) | (x >> (64 - n))) & 0xFFFFFFFFFFFFFFFF
    a = [[state[x][y] for y in range(5)] for x in range(5)]
    for rnd in range(24):
        c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ rol(c[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                a[x][y] ^= d[x]
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = rol(a[x][y], _KECCAK_ROT[x][y])
        for x in range(5):
            for y in range(5):
                a[x][y] = b[x][y] ^ ((~b[(x + 1) % 5][y]) & b[(x + 2) % 5][y])
        a[0][0] ^= _KECCAK_RC[rnd]
    return a


def keccak256(data):
    rate = 136  # 1088-bit rate for Keccak-256
    padded = bytearray(data)
    padded.append(0x01)                      # Keccak padding (NOT 0x06 SHA-3)
    while len(padded) % rate != 0:
        padded.append(0x00)
    padded[-1] ^= 0x80
    state = [[0] * 5 for _ in range(5)]
    for off in range(0, len(padded), rate):
        block = padded[off:off + rate]
        for i in range(rate // 8):
            lane = int.from_bytes(block[i * 8:i * 8 + 8], 'little')
            state[i % 5][i // 5] ^= lane
        state = _keccak_f(state)
    out = b''
    for i in range(4):  # 32 bytes = 4 lanes
        out += state[i % 5][i // 5].to_bytes(8, 'little')
    return out[:32]


# ══════════════════════════════════════════════════════════════════════
# Part A — checksum / check-digit validators
# ══════════════════════════════════════════════════════════════════════

# ── IBAN (ISO 13616) — per-country total length from the SWIFT IBAN Registry.
# Rearrange (move first 4 chars to the end), convert letters, mod 97 == 1.
IBAN_LENGTHS = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16,
    "BG": 22, "BH": 22, "BR": 29, "BY": 28, "CH": 21, "CR": 22, "CY": 28,
    "CZ": 24, "DE": 22, "DK": 18, "DO": 28, "EE": 20, "EG": 29, "ES": 24,
    "FI": 18, "FO": 18, "FR": 27, "GB": 22, "GE": 22, "GI": 23, "GL": 18,
    "GR": 27, "GT": 28, "HR": 21, "HU": 28, "IE": 22, "IL": 23, "IS": 26,
    "IT": 27, "JO": 30, "KW": 30, "KZ": 20, "LB": 28, "LC": 32, "LI": 21,
    "LT": 20, "LU": 20, "LV": 21, "LY": 25, "MC": 27, "MD": 24, "ME": 22,
    "MK": 19, "MR": 27, "MT": 31, "MU": 30, "NL": 18, "NO": 15, "PK": 24,
    "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "SA": 24,
    "SC": 31, "SE": 24, "SI": 19, "SK": 24, "SM": 27, "ST": 25, "SV": 28,
    "TL": 23, "TN": 24, "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20,
}


def iban_valid(s):
    s = re.sub(r'[\s]', '', s).upper()
    if not re.fullmatch(r'[A-Z]{2}\d{2}[A-Z0-9]+', s):
        return False
    if IBAN_LENGTHS.get(s[:2]) != len(s):
        return False
    return iso7064_mod97_10_valid(_alnum_to_digits(s[4:] + s[:4]))


def lei_valid(s):
    """LEI (ISO 17442): 20 chars, ISO 7064 MOD 97-10 check (last two digits)."""
    s = s.upper()
    if not re.fullmatch(r'[A-Z0-9]{18}\d{2}', s):
        return False
    return iso7064_mod97_10_valid(_alnum_to_digits(s))


def aba_valid(s):
    """ABA/Fed routing transit number: 9 digits, mod-10 with weights 3-7-1."""
    d = [int(c) for c in s if c.isdigit()]
    if len(d) != 9:
        return False
    checksum = 3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + (d[2] + d[5] + d[8])
    return checksum % 10 == 0


def isin_valid(s):
    """ISIN (ISO 6166): 2-letter country + 9 alnum + Luhn check over expansion."""
    s = s.upper()
    if not re.fullmatch(r'[A-Z]{2}[A-Z0-9]{9}\d', s):
        return False
    return luhn_valid(_alnum_to_digits(s))


_SEDOL_WEIGHTS = [1, 3, 1, 7, 3, 9]


def sedol_valid(s):
    """SEDOL (LSE): 6 significant chars (0-9, consonants) + mod-10 check digit."""
    s = s.upper()
    if not re.fullmatch(r'[0-9BCDFGHJKLMNPQRSTVWXYZ]{6}\d', s):
        return False
    total = sum((int(c) if c.isdigit() else ord(c) - 55) * w
                for c, w in zip(s[:6], _SEDOL_WEIGHTS))
    return (10 - (total % 10)) % 10 == int(s[6])


def cusip_valid(s):
    """CUSIP (CUSIP Global Services): 8 chars + mod-10 double-add-double check."""
    s = s.upper()
    if not re.fullmatch(r'[0-9A-Z*@#]{8}\d', s):
        return False
    total = 0
    for i, c in enumerate(s[:8]):
        if c.isdigit():
            v = int(c)
        elif c.isalpha():
            v = ord(c) - ord('A') + 10
        else:
            v = {'*': 36, '@': 37, '#': 38}[c]
        if i % 2 == 1:
            v *= 2
        total += v // 10 + v % 10
    return (10 - (total % 10)) % 10 == int(s[8])


def gtin_valid(s):
    """EAN-13 / GTIN-8/12/13/14 / UPC-A (GS1): mod-10, weights 3,1 from right."""
    d = [int(c) for c in s if c.isdigit()]
    if len(d) not in (8, 12, 13, 14):
        return False
    total = sum(dig * (3 if i % 2 == 0 else 1) for i, dig in enumerate(reversed(d[:-1])))
    return (10 - (total % 10)) % 10 == d[-1]


def isbn10_valid(s):
    """ISBN-10 (ISO 2108): weighted mod-11, check may be 'X'."""
    s = s.upper().replace('-', '').replace(' ', '')
    if not re.fullmatch(r'\d{9}[\dX]', s):
        return False
    total = sum((10 - i) * (10 if c == 'X' else int(c)) for i, c in enumerate(s))
    return total % 11 == 0


def vin_valid(s):
    """VIN check digit (ISO 3779 structure; check digit per US 49 CFR 565.15).
    Note: the check digit is a North-American requirement, not global ISO."""
    s = s.upper()
    if not re.fullmatch(r'[A-HJ-NPR-Z0-9]{17}', s):
        return False
    trans = {**{str(i): i for i in range(10)},
             'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
             'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'P': 7, 'R': 9,
             'S': 2, 'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9}
    weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
    total = sum(trans[c] * w for c, w in zip(s, weights))
    check = total % 11
    return s[8] == ('X' if check == 10 else str(check))


def _repdigit(d):
    return len(set(d)) == 1


def cpf_valid(s):
    """Brazilian CPF: 11 digits, two mod-11 check digits (Receita Federal)."""
    d = [int(c) for c in s if c.isdigit()]
    if len(d) != 11 or _repdigit(d):
        return False
    c1 = (sum(d[i] * (10 - i) for i in range(9)) * 10) % 11 % 10
    c2 = (sum(d[i] * (11 - i) for i in range(10)) * 10) % 11 % 10
    return c1 == d[9] and c2 == d[10]


def cnpj_valid(s):
    """Brazilian CNPJ: 14 digits, two mod-11 check digits."""
    d = [int(c) for c in s if c.isdigit()]
    if len(d) != 14 or _repdigit(d):
        return False
    w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    w2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    r1 = sum(d[i] * w1[i] for i in range(12)) % 11
    c1 = 0 if r1 < 2 else 11 - r1
    r2 = sum(d[i] * w2[i] for i in range(13)) % 11
    c2 = 0 if r2 < 2 else 11 - r2
    return c1 == d[12] and c2 == d[13]


def pesel_valid(s):
    """Polish PESEL: 11 digits, weighted mod-10 check (GUS)."""
    d = [int(c) for c in s if c.isdigit()]
    if len(d) != 11:
        return False
    w = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    return (10 - (sum(d[i] * w[i] for i in range(10)) % 10)) % 10 == d[10]


def bsn_valid(s):
    """Dutch BSN: 9 digits, 11-proef with weights 9..2 and -1 on the last."""
    d = [int(c) for c in s if c.isdigit()]
    if len(d) != 9 or _repdigit(d):
        return False
    weights = [9, 8, 7, 6, 5, 4, 3, 2, -1]
    return sum(d[i] * weights[i] for i in range(9)) % 11 == 0


_DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"


def dni_valid(s):
    """Spanish DNI/NIE: mod-23 check letter (NIE prefix X/Y/Z -> 0/1/2)."""
    s = s.upper()
    m = re.fullmatch(r'([XYZ]?)(\d{7,8})([A-Z])', s)
    if not m:
        return False
    prefix, num, letter = m.groups()
    if prefix:
        num = str("XYZ".index(prefix)) + num
    return _DNI_LETTERS[int(num) % 23] == letter


# Codice Fiscale (Italy) — odd/even position tables + mod-26 check char.
_CF_ODD = {
    '0': 1, '1': 0, '2': 5, '3': 7, '4': 9, '5': 13, '6': 15, '7': 17, '8': 19,
    '9': 21, 'A': 1, 'B': 0, 'C': 5, 'D': 7, 'E': 9, 'F': 13, 'G': 15, 'H': 17,
    'I': 19, 'J': 21, 'K': 2, 'L': 4, 'M': 18, 'N': 20, 'O': 11, 'P': 3, 'Q': 6,
    'R': 8, 'S': 12, 'T': 14, 'U': 16, 'V': 10, 'W': 22, 'X': 25, 'Y': 24, 'Z': 23,
}
_CF_EVEN = {
    **{str(i): i for i in range(10)},
    **{chr(ord('A') + i): i for i in range(26)},
}


def codice_fiscale_valid(s):
    """Italian codice fiscale: 15 chars + mod-26 check letter."""
    s = s.upper()
    if not re.fullmatch(r'[A-Z0-9]{15}[A-Z]', s):
        return False
    total = sum(_CF_ODD[c] if (i + 1) % 2 == 1 else _CF_EVEN[c]
                for i, c in enumerate(s[:15]))
    return chr(ord('A') + total % 26) == s[15]


def aadhaar_valid(s):
    """Indian Aadhaar: 12 digits, Verhoeff check; first digit is 2-9 (UIDAI)."""
    d = [c for c in s if c.isdigit()]
    if len(d) != 12 or d[0] in ('0', '1'):
        return False
    return verhoeff_valid(''.join(d))


# ── Crypto addresses ──
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def base58check_valid(s):
    """Bitcoin base58check: double-SHA256, last 4 bytes must match (BTC protocol)."""
    if not s or any(c not in _B58 for c in s):
        return False
    num = 0
    for c in s:
        num = num * 58 + _B58.index(c)
    raw = num.to_bytes((num.bit_length() + 7) // 8, 'big') if num else b''
    raw = b'\x00' * (len(s) - len(s.lstrip('1'))) + raw
    if len(raw) < 5:
        return False
    data, checksum = raw[:-4], raw[-4:]
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()[:4] == checksum


_BECH32 = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _bech32_polymod(values):
    gen = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        top = chk >> 25
        chk = ((chk & 0x1ffffff) << 5) ^ v
        for i in range(5):
            chk ^= gen[i] if ((top >> i) & 1) else 0
    return chk


def _bech32_hrp_expand(hrp):
    return [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]


def bech32_valid(s):
    """Bech32 / Bech32m (BIP-173 / BIP-350) checksum constant 1 or 0x2bc830a3."""
    if any(ord(c) < 33 or ord(c) > 126 for c in s):
        return False
    if s.lower() != s and s.upper() != s:      # mixed case is invalid
        return False
    s = s.lower()
    pos = s.rfind('1')
    if pos < 1 or pos + 7 > len(s):
        return False
    hrp, data = s[:pos], []
    for c in s[pos + 1:]:
        if c not in _BECH32:
            return False
        data.append(_BECH32.index(c))
    return _bech32_polymod(_bech32_hrp_expand(hrp) + data) in (1, 0x2bc830a3)


def eip55_valid(addr):
    """Ethereum address; EIP-55 mixed-case keccak checksum when case is mixed."""
    if not re.fullmatch(r'0x[0-9a-fA-F]{40}', addr):
        return False
    body = addr[2:]
    if body == body.lower() or body == body.upper():
        return True                            # single-case: no checksum to verify
    h = keccak256(body.lower().encode()).hex()
    for i, c in enumerate(body):
        if c.isalpha():
            upper = int(h[i], 16) >= 8
            if upper != c.isupper():
                return False
    return True


# ══════════════════════════════════════════════════════════════════════
# Checksum scheme registry — adding a scheme is one row, not new code.
# Each row: candidate regex (finds plausible spans) + validator (confirms) +
# emitted entity_type + source citation. Ordered longest/most-specific first.
# ══════════════════════════════════════════════════════════════════════
CHECKSUM_SCHEMES = [
    # name              candidate regex                              validator            entity_type        source
    ("iban",   r'\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){10,30}\b',         iban_valid,          "IBAN",            "ISO 13616 / SWIFT IBAN Registry"),
    ("lei",    r'\b[A-Z0-9]{18}\d{2}\b',                            lei_valid,           "LEI",             "ISO 17442 (ISO 7064 MOD 97-10)"),
    ("isin",   r'\b[A-Z]{2}[A-Z0-9]{9}\d\b',                        isin_valid,          "ISIN",            "ISO 6166"),
    ("vin",    r'\b[A-HJ-NPR-Z0-9]{17}\b',                          vin_valid,           "VIN",             "ISO 3779 / US 49 CFR 565.15"),
    ("cnpj",   r'\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b',          cnpj_valid,          "BR_CNPJ",         "Receita Federal (mod-11)"),
    ("aadhaar", r'\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b',                  aadhaar_valid,       "IN_AADHAAR",      "UIDAI (Verhoeff)"),
    ("cpf",    r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b',                 cpf_valid,           "BR_CPF",          "Receita Federal (mod-11)"),
    ("pesel",  r'\b\d{11}\b',                                       pesel_valid,         "PL_PESEL",        "GUS (weighted mod-10)"),
    ("cf",     r'\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b',      codice_fiscale_valid, "IT_CF",          "Agenzia delle Entrate (mod-26)"),
    ("isbn13", r'\b97[89][ -]?(?:\d[ -]?){9}\d\b',                  gtin_valid,          "ISBN",            "ISO 2108 (GS1 mod-10)"),
    ("isbn10", r'\b\d{9}[\dX]\b',                                   isbn10_valid,        "ISBN",            "ISO 2108 (mod-11)"),
    ("ean13",  r'\b\d{13}\b',                                       gtin_valid,          "GTIN",            "GS1 (mod-10)"),
    ("upca",   r'\b\d{12}\b',                                       gtin_valid,          "GTIN",            "GS1 (mod-10)"),
    ("aba",    r'\b\d{9}\b',                                        aba_valid,           "ABA_ROUTING",     "Fed/ABA routing (mod-10 3-7-1)"),
    ("bsn",    r'\b\d{9}\b',                                        bsn_valid,           "NL_BSN",          "Netherlands BSN (11-proef)"),
    ("sedol",  r'\b[0-9BCDFGHJKLMNPQRSTVWXYZ]{6}\d\b',              sedol_valid,         "SEDOL",           "LSE (weighted mod-10)"),
    ("dni",    r'\b[XYZ]?\d{7,8}[A-Z]\b',                           dni_valid,           "ES_DNI",          "Spain DNI/NIE (mod-23)"),
    # CUSIP last among short schemes: its 9-alphanumeric candidate + weak mod-10
    # check is the greediest, so more-specific schemes claim their spans first.
    ("cusip",  r'\b[0-9A-Z*@#]{8}\d\b',                             cusip_valid,         "CUSIP",           "CUSIP Global Services (mod-10)"),
    # Cards last: Luhn is the weakest check (10% random pass), so more specific
    # schemes above win the span first.
    ("imei",   r'\b\d{15}\b',                                       luhn_valid,          "IMEI",            "3GPP TS 23.003 (Luhn)"),
    ("card",   r'\b(?:\d[ -]?){12,18}\d\b',                         luhn_valid,          "CREDIT_CARD",     "ISO/IEC 7812-1 (Luhn)"),
    ("btc",    r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',              base58check_valid,   "CRYPTO_ADDRESS",  "Bitcoin base58check"),
    ("bech32", r'\b(?:bc|tb|bcrt)1[023456789acdefghjklmnpqrstuvwxyz]{6,87}\b', bech32_valid, "CRYPTO_ADDRESS", "BIP-173 / BIP-350"),
    ("eth",    r'\b0x[0-9a-fA-F]{40}\b',                            eip55_valid,         "CRYPTO_ADDRESS",  "Ethereum EIP-55"),
]
_COMPILED_CHECKSUMS = [(n, re.compile(rx), fn, et, src) for n, rx, fn, et, src in CHECKSUM_SCHEMES]


# ══════════════════════════════════════════════════════════════════════
# Part C — format-standard recognizers. Published grammars, issuer-independent.
# Only formats distinctive enough to fire safely on their own are here;
# non-distinctive ID grammars (Snowflake bare int, NanoID, KSUID base62) are
# left to the entropy net, since matching them directly would mass-over-mask.
# ══════════════════════════════════════════════════════════════════════

def _jwt_valid(s):
    """RFC 7519: three base64url segments; header+payload decode to JSON."""
    parts = s.split('.')
    if len(parts) != 3:
        return False
    try:
        for seg in parts[:2]:
            pad = '=' * (-len(seg) % 4)
            json.loads(base64.urlsafe_b64decode(seg + pad))
    except Exception:
        return False
    return True


def _uuid_valid(s):
    """RFC 9562: version nibble 1-8, variant bits 10xx (8/9/a/b)."""
    return bool(re.fullmatch(
        r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}'
        r'-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}', s))


# name, candidate regex, optional validator (None = shape is sufficient),
# entity_type, source. group(1) if the pattern captures, else group(0).
FORMAT_RECOGNIZERS = [
    ("pem",      r'-----BEGIN [A-Z0-9 ]+-----[\s\S]+?-----END [A-Z0-9 ]+-----',
     None,        "PRIVATE_KEY",        "RFC 7468 / RFC 4880 armor"),
    ("jwt",      r'\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*',
     _jwt_valid,  "JWT",                "RFC 7519"),
    ("uuid",     r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b',
     _uuid_valid, "UUID",               "RFC 9562"),
    ("ulid",     r'\b[0-7][0-9A-HJKMNP-TV-Z]{25}\b',
     None,        "OPAQUE_ID",          "ULID spec (Crockford base32)"),
    ("sid",      r'\bS-1-\d{1,2}(?:-\d+){1,15}\b',
     None,        "WINDOWS_SID",        "MS-DTYP §2.4.2"),
    ("krb_spn",  r'\b[A-Za-z0-9._-]+/[A-Za-z0-9._-]+@[A-Z0-9][A-Z0-9.-]+\b',
     None,        "KERBEROS_PRINCIPAL", "RFC 1964 (service principal)"),
    ("arn",      r'\barn:(?:aws|aws-cn|aws-us-gov):[A-Za-z0-9-]*:[a-z0-9-]*:\d{0,12}:[^\s"\']+',
     None,        "CLOUD_RESOURCE_ID",  "AWS ARN grammar"),
    ("azure_id", r'/subscriptions/[0-9a-fA-F-]{36}/resourceGroups/[^\s"\']+',
     None,        "CLOUD_RESOURCE_ID",  "Azure resource ID grammar"),
    ("gcp_id",   r'\bprojects/[a-z0-9-]+/(?:locations|zones|regions|instances|buckets|datasets|topics|subscriptions|secrets|serviceAccounts)/[^\s"\']+',
     None,        "CLOUD_RESOURCE_ID",  "Google AIP-122"),
    ("mac",      r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b',
     None,        "MAC_ADDRESS",        "IEEE 802"),
    ("ipv6",     r'\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b|\b(?:[0-9A-Fa-f]{1,4}:){1,7}:(?:[0-9A-Fa-f]{1,4}:?){0,6}\b',
     None,        "IPV6",               "RFC 4291"),
    # Connection string / URI userinfo — capture the user:pass credential only.
    ("conn_uri", r'[a-zA-Z][a-zA-Z0-9+.-]*://([^\s:/@]+:[^\s:/@]+)@',
     None,        "CONNECTION_STRING",  "RFC 3986 §3.2.1"),
]
_COMPILED_FORMATS = [(n, re.compile(rx), fn, et, src) for n, rx, fn, et, src in FORMAT_RECOGNIZERS]


# ══════════════════════════════════════════════════════════════════════
# Part B — entropy / randomness scoring
# ══════════════════════════════════════════════════════════════════════

def shannon_entropy(s):
    """Bits per character."""
    if not s:
        return 0.0
    counts = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    n = len(s)
    return -sum((k / n) * math.log2(k / n) for k in counts.values())


# ══════════════════════════════════════════════════════════════════════
# Benign-format veto — checked BEFORE the entropy net emits.
#
# An entry earns its place only if the format's own STANDARD proves the value
# cannot be a secret. Not "this compresses well", not "the corpus has one" —
# the grammar itself has to rule secrecy out. Three qualify:
#
#   multipart_boundary  RFC 2046 §5.1.1. The boundary token MUST also travel
#                       in cleartext in the Content-Type header of the same
#                       message, or the body cannot be parsed. A delimiter
#                       that is structurally required to be published
#                       alongside the data it delimits is not a secret.
#   sri_hash            W3C Subresource Integrity. sha256-/sha384-/sha512- +
#                       base64 digest of a PUBLIC subresource, published in
#                       the page source so the browser can verify it.
#   oci_digest          OCI Image Spec / distribution reference grammar
#                       `name@algorithm:encoded`. A content address for a
#                       registry blob anyone can pull.
#   asset_fingerprint   A cache-busting digest in the path of a URL that ends
#                       in a static-asset extension. The build tool emits it so
#                       the file can be served immutable, and publishes it in
#                       the page source beside the thing it names. Scoped to
#                       the path only — see the block above _asset_fingerprint.
#
# Deliberately NOT vetoed, and why — these are the entries that would look
# reasonable and cost recall:
#
#   compression / image containers (gzip H4sI, zlib, PK, /9j/, iVBOR)
#       Magic bytes prove the value is a CONTAINER, not that its contents are
#       harmless. A gzip blob can be a database dump. Vetoing it ships the
#       plaintext to the cloud LLM base64-encoded. Masking a container is the
#       conservative outcome, so we mask.
#   base64 that decodes to printable text
#       Same trap one level down: b64("SuperSecret123!") decodes to printable
#       ASCII and would be vetoed. Encoding is not evidence of non-secrecy.
#   bare hex (git SHA-1/SHA-256 object IDs)
#       A bare 32/40/64-hex run is equally an NTLM hash, an MD5/SHA password
#       hash, an HMAC, or an API secret, and the value alone cannot tell you
#       which. Masking a commit SHA costs the reader nothing — it still reads
#       "commit [HIGH_ENTROPY_SECRET_1]". Leaking an NTLM hash is a usable
#       credential. Only the labelled form (@sha256:) is vetoed, above.
#   UUID / ULID / KSUID / NanoID
#       Bare RFC 9562 UUIDs and ULIDs are already claimed by their own
#       recognizers in FORMAT_RECOGNIZERS and masked as UUID / OPAQUE_ID, so
#       there is nothing for a veto to do. NanoID has no internal structure
#       at all — no version field, no checksum, no timestamp — so a NanoID
#       recognizer is just "any N random URL-safe chars", which is the
#       entropy net itself. Anything prefixed (usr_…, sk_live_…, ing_…) is
#       masked: the prefix is application semantics we cannot evaluate, and
#       an unbounded vendor deny-list of secret prefixes fails open on every
#       prefix that ships next month.
# ══════════════════════════════════════════════════════════════════════

# RFC 2046 §5.1.1:
#   boundary      := 0*69<bchars> bcharsnospace
#   bchars        := bcharsnospace / " "
#   bcharsnospace := DIGIT / ALPHA / "'" / "(" / ")" / "+" / "_" / "," /
#                    "-" / "." / "/" / ":" / "=" / "?"
#   dash-boundary := "--" boundary
_BOUNDARY_TOKEN_RE = re.compile(r"[0-9A-Za-z'()+_,\-./:=?]{1,70}")


def _multipart_boundary(val, text, start):
    """RFC 2046 §5.1.1 dash-boundary."""
    dashes = re.match(r'--+', val)
    if not dashes:
        return None
    token = val[dashes.end():]
    if not _BOUNDARY_TOKEN_RE.fullmatch(token):
        return None
    # RFC 2046 §5.1.1: every body part is introduced by a dash-boundary and
    # the body is terminated by a close-delimiter ("--" boundary "--"), so a
    # real boundary token occurs at least twice in the message. A one-off
    # string that merely starts with dashes is not a boundary — this is what
    # stops the veto from swallowing "--" + secret.
    if text.count(token) < 2:
        return None
    return "multipart_boundary"


# W3C Subresource Integrity: hash-expression = hash-algorithm "-" base64-value,
# hash-algorithm in {sha256, sha384, sha512}. The decoded digest length is
# fixed by the algorithm, so it is a checkable invariant rather than a prefix
# guess — "sha256-" + arbitrary base64 does not qualify.
_SRI_DIGEST_BYTES = {"256": 32, "384": 48, "512": 64}


def _sri_hash(val, text, start):
    """W3C Subresource Integrity hash-expression."""
    m = re.fullmatch(r'sha(256|384|512)-([A-Za-z0-9+/]+={0,2})', val)
    if not m:
        return None
    b64 = m.group(2)
    try:
        raw = base64.b64decode(b64 + '=' * (-len(b64) % 4), validate=True)
    except Exception:
        return None
    return "sri_hash" if len(raw) == _SRI_DIGEST_BYTES[m.group(1)] else None


# OCI Image Spec descriptor digest / distribution reference grammar:
#   reference := name [":" tag] ["@" digest]
#   digest    := algorithm ":" encoded
# Anchored on the "@" so it cannot match a bare labelled hash such as
# "md5: 5f4dcc3b…" or "password_sha256=…", which are password hashes.
_OCI_DIGEST_PREFIX_RE = re.compile(r'@sha(?:256|512):$')


def _oci_digest(val, text, start):
    """OCI content-addressed image digest (name@sha256:<hex>)."""
    if not re.fullmatch(r'[0-9a-f]{64}|[0-9a-f]{128}', val):
        return None
    if not _OCI_DIGEST_PREFIX_RE.search(text[max(0, start - 12):start]):
        return None
    return "oci_digest"


# ── Cache-busting asset fingerprints ──────────────────────────────────
#
# Every build tool emits content-hashed asset paths, and the hash has no fixed
# form to match on. Measured across the documented defaults:
#
#   webpack 5         main.d587bbd6e38337f5accd.js      hex, 20 (:8 common)
#   Rollup 4 / Vite   index-Bx7fK2mQ.js                 base64url, 8
#   esbuild / Angular index-4QTUNIID.js                 base32 upper, 8
#   Next.js           page-23013a4bb4002868.js          hex, 16
#   Sprockets         application-<64 hex>.js           sha256 hex, 64
#   Django            styles.55e7cbb9ba48.css           md5 hex, 12
#   hash-as-directory /icon/<32 hex>/icon-192.png        md5 hex, 32
#
# Four alphabets, lengths 8 to 64. "index-Bx7fK2mQ" and an 8-char API key are
# the same string class, so no shape test can separate them. The discriminator
# has to be POSITION: a value in the path of a URL that terminates in a static
# asset is a cache-busting digest by construction — the build tool put it there
# so the file could be served immutable, and it is published in the page source
# next to the thing it names. Same standard-rules-out-secrecy test as the SRI
# and OCI entries above.
#
# THE CHARACTER CLASS IS THE WHOLE SAFETY ARGUMENT.
# We expand to the maximal surrounding run of PATH characters. That class
# excludes ? # & = quotes and whitespace, so expansion can never cross from a
# query string into a path. On an S3 presigned URL the path ends ".png" but the
# signature lives after the "?", so expansion from the signature stops at the
# "=", the token holds no "/", and no veto fires. Verified against presigned
# URLs, session ids in query strings, reset tokens as path segments, JWTs,
# NTLM hashes and OAuth codes — see tests/test_asset_fingerprints.py.
#
# Extensions are script/style/image/font/media only. NOT .json, .html, .xml,
# .pdf, .csv or .txt: those carry data, and a signed-CDN path such as
# /media/<hmac>/private-report.pdf must stay masked.
#
# Residual risk, stated rather than hidden: a signing scheme that puts an HMAC
# in the path of a genuinely static asset (/media/<hmac>/logo.png) is vetoed.
# The engagement answer is a custom rule. What limits the blast radius is that
# the entropy net runs LAST — checksums, JWT/PEM/UUID/SID recognizers and the
# whole regex stage have already claimed their spans, so anything with provable
# structure inside an asset path is masked before this code is reached.
ASSET_EXTENSIONS = (
    "js mjs cjs css map "
    "png jpg jpeg gif svg webp avif ico bmp cur "
    "woff woff2 ttf otf eot "
    "wasm mp3 mp4 webm ogg wav flac "
    "scss less"
).split()

_ASSET_EXT_RE = re.compile(
    r'\.(?:' + '|'.join(ASSET_EXTENSIONS) + r')$', re.IGNORECASE)

# Path characters per RFC 3986 §3.3 (unreserved + the sub-delims we actually
# see in asset paths). Deliberately excludes ? # & = " ' and whitespace.
_PATH_CHAR_RE = re.compile(r'[A-Za-z0-9._/+~-]')


def _asset_fingerprint(val, text, start):
    """Cache-busting digest in the path of a static-asset URL."""
    end = start + len(val)

    i = start
    while i > 0 and _PATH_CHAR_RE.match(text[i - 1]):
        i -= 1
    j = end
    while j < len(text) and _PATH_CHAR_RE.match(text[j]):
        j += 1

    token = text[i:j]
    if "/" not in token:
        # A bare run in a body or a header value, not a URL path.
        return None
    if not _ASSET_EXT_RE.search(token):
        return None
    return "asset_fingerprint"


BENIGN_FORMATS = [
    ("multipart_boundary", _multipart_boundary),
    ("sri_hash",           _sri_hash),
    ("oci_digest",         _oci_digest),
    ("asset_fingerprint",  _asset_fingerprint),
]

# Per-class veto tally, for the "how many over-masks did each class remove"
# report. Reset per form_detect() call; printed under debug=True.
VETO_COUNTS = {}


def benign_format(val, text, start):
    """Return the benign format-class name vetoing this candidate, else None."""
    for _name, fn in BENIGN_FORMATS:
        hit = fn(val, text, start)
        if hit:
            return hit
    return None

# ── BPE token-efficiency threshold ────────────────────────────────────
#
# Token efficiency = len(chars) / len(cl100k_base tokens). Word-like text
# compresses into few long tokens (high efficiency); random secrets have no
# learned merges and tokenise into many short pieces (low efficiency). Fire
# when efficiency falls BELOW the threshold.
#
# Published reference points:
#   - word-like text ~3-5 bytes/token, random secrets ~1.1-1.5 bytes/token,
#     with the CredData F1 peak at 2.2   (keyhog-scanner src/entropy/bpe.rs)
#   - Betterleaks reports 98.6% recall on CredData for token efficiency vs
#     70.4% for Shannon entropy (author's own benchmark)
#
# This was previously DERIVED at runtime as min(token_efficiency(sample)) over
# a list of "benign" samples — but every sample in that list was a hash or a
# UUID, i.e. maximally random, sitting at the BOTTOM of the efficiency
# distribution rather than above it. One sample (an alternating digit/letter
# hex string) tokenised at exactly 1 char/token, so the derived threshold was
# 1.000. Since every cl100k token is >= 1 byte and the entropy candidate regex
# only matches ASCII, efficiency is >= 1.0 by construction: the branch could
# never fire. Measured on the 36-app corpus, 0 of 152 gold secrets and 0 of
# 400 benign candidates fell below 1.000.
#
# The threshold is now a constant. Override per-run with --form-bpe-threshold.
BPE_THRESHOLD_DEFAULT = 2.2

# Production default. Shannon and "both" remain selectable for benchmark runs
# (--form-entropy-method); nothing in the pipeline sets them.
ENTROPY_METHOD_DEFAULT = "bpe"

# Efficiency is >= 1.0 for ASCII by construction. Any threshold at or below
# this floor makes the branch unsatisfiable, which is silent and looks exactly
# like "BPE found nothing". Refuse it loudly instead.
BPE_THRESHOLD_FLOOR = 1.05

# Where the BPE encoder lives. Resolved in code, not read from the caller's
# environment: TIKTOKEN_CACHE_DIR going missing in a fresh shell was the one
# silent failure in the whole offline story — tiktoken then looks in a temp
# directory, finds nothing, tries to download, and form quietly runs a
# different detector. A path the package owns cannot be lost.
#
# An explicit TIKTOKEN_CACHE_DIR still wins: an operator who set one meant it,
# and shared caches are how air-gapped and multi-user boxes are provisioned.
BPE_ENCODING = "cl100k_base"
BPE_BLOB_URL = ("https://openaipublic.blob.core.windows.net/encodings/"
                "cl100k_base.tiktoken")
# Published by OpenAI in tiktoken_ext/openai_public.py. Verified after download
# so a truncated or substituted file fails loudly instead of silently changing
# the entropy profile out from under every benchmarked number.
BPE_BLOB_SHA256 = ("223921b76ee99bde995b7ff738513eef100fb51d18c93597a113b"
                   "cffe865b2a7")


def bpe_cache_dir():
    """The directory tiktoken should read the encoder from."""
    env = os.environ.get("TIKTOKEN_CACHE_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_bpe_cache")


def bpe_cache_filename():
    """tiktoken keys its cache by sha1 of the blob URL — not a private API,
    just an undocumented one, so it is derived here rather than hardcoded."""
    return hashlib.sha1(BPE_BLOB_URL.encode()).hexdigest()


def bpe_is_cached():
    return os.path.exists(os.path.join(bpe_cache_dir(), bpe_cache_filename()))


_tiktoken_enc = "unset"   # sentinel: "unset" -> not tried; None -> unavailable

# Warn once per process when a BPE-requiring mode degrades to Shannon. The
# encoder is cached, so without this the warning would print on the first
# prompt only and be invisible for the rest of the server's life.
# Retained only so the self-test can save/restore process state around its
# encoder fiddling. The warn-once fallback it gated no longer exists: a
# missing encoder raises.
ENCODER_UNAVAILABLE_WARNED = False


class EncoderUnavailableError(Exception):
    """entropy_method requires BPE and cl100k_base could not be loaded.

    Owner decision 2026-10-05: this tool does not fail open. Running Shannon
    when the config said bpe is a different detector — measured at 9 of 81
    form detections lost on a 40-message sample, in the UNDER-masking
    direction, including a secret-bearing URL Shannon does not see. Reporting
    success while masking less is the failure mode the rule exists to stop.

    Escape hatch, as everywhere else: ask for Shannon explicitly.
    """


def _get_encoder():
    """Load cl100k_base, or return None. Never downgrades silently."""
    global _tiktoken_enc
    if _tiktoken_enc == "unset":
        cache = bpe_cache_dir()
        try:
            os.makedirs(cache, exist_ok=True)
        except OSError:
            pass
        # Set before the import: tiktoken reads this at call time, but an
        # operator-set value must survive, and a package-local default must
        # not leak into an unrelated tiktoken user in the same process.
        prev = os.environ.get("TIKTOKEN_CACHE_DIR")
        os.environ["TIKTOKEN_CACHE_DIR"] = cache
        try:
            import tiktoken
            _tiktoken_enc = tiktoken.get_encoding(BPE_ENCODING)
        except Exception as e:
            print(f"  [FORM] BPE unavailable ({type(e).__name__}: {e})")
            _tiktoken_enc = None
        finally:
            if prev is None:
                os.environ.pop("TIKTOKEN_CACHE_DIR", None)
            else:
                os.environ["TIKTOKEN_CACHE_DIR"] = prev
    return _tiktoken_enc


def _encoder_required_error(method):
    return EncoderUnavailableError(
        "entropy_method=%r requires the %s BPE encoder, which could not be "
        "loaded (cache: %s).\n"
        "  Fetch it once, with network:   python -m maskerade.warm\n"
        "  Or ask for Shannon explicitly: form.entropy_method: shannon\n"
        "Not falling back automatically: Shannon is a different detector and "
        "masks less than bpe, and a quietly degraded result is worse than an "
        "error for a tool whose next step is sending text to a third party."
        % (method, BPE_ENCODING, bpe_cache_dir()))


def _token_efficiency(s, enc):
    toks = enc.encode(s)
    return len(s) / len(toks) if toks else 0.0


def _bpe_threshold(cfg):
    """chars-per-token ceiling; values below this fire. See BPE_THRESHOLD_DEFAULT."""
    thr = cfg.get("bpe_threshold")
    if thr is None:
        return BPE_THRESHOLD_DEFAULT
    if thr <= BPE_THRESHOLD_FLOOR:
        print(f"  [FORM] bpe_threshold={thr} is at or below the {BPE_THRESHOLD_FLOOR} "
              f"floor — token efficiency is >= 1.0 for ASCII, so the BPE branch "
              f"would never fire. Using {BPE_THRESHOLD_DEFAULT}.")
        return BPE_THRESHOLD_DEFAULT
    return thr


# ══════════════════════════════════════════════════════════════════════
# Span bookkeeping
# ══════════════════════════════════════════════════════════════════════

def _overlaps(claimed, start, end):
    for cs, ce in claimed:
        if start < ce and cs < end:
            return True
    return False


def _emit(entities, claimed, text, start, end, etype, score):
    val = text[start:end]
    if PLACEHOLDER_RE.fullmatch(val):
        return False
    entities.append({
        "text": val, "entity_type": etype,
        "start": start, "end": end,
        "score": score, "stage": "ner", "detector": "form",
    })
    claimed.append((start, end))
    return True


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def form_detect(text, config, debug=False, notes=None):
    """Intrinsic-form detection. Returns entity dicts (see module docstring).

    notes: optional list receiving a record for any input this stage did not
           process as configured. Currently one case — a BPE entropy method
           degrading to Shannon because cl100k_base is unavailable. Same
           contract as the gliner size gate: the run is valid, but it masked
           with a different entropy profile than the mode string advertises,
           and the caller is about to send the result to a third party.

           Without this the only trace is a once-per-PROCESS line on stdout,
           so a server that starts with a cold tiktoken cache reports the
           degradation once at boot and never again — every subsequent
           message is silently scored against a profile that matches no
           benchmarked number.
    """
    global ENCODER_UNAVAILABLE_WARNED
    cfg = (config or {}).get("form", {})
    method = cfg.get("entropy_method", ENTROPY_METHOD_DEFAULT)
    thr_b64 = cfg.get("shannon_base64", 4.5)   # detect-secrets --base64-limit
    thr_hex = cfg.get("shannon_hex", 3.0)      # detect-secrets --hex-limit
    min_len = cfg.get("min_entropy_len", 20)

    entities = []
    claimed = []
    VETO_COUNTS.clear()

    # Pre-claim placeholders so nothing masks a placeholder.
    for m in PLACEHOLDER_RE.finditer(text):
        claimed.append(m.span())

    # ── Part A: checksums (high precision, claim spans) ──
    for name, rx, validator, etype, _src in _COMPILED_CHECKSUMS:
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            if _overlaps(claimed, s, e):
                continue
            if not validator(m.group(0)):
                continue
            _emit(entities, claimed, text, s, e, etype, 1.0)

    # ── Part C: named formats ──
    for name, rx, validator, etype, _src in _COMPILED_FORMATS:
        for m in rx.finditer(text):
            s = m.start(1) if m.lastindex else m.start()
            e = m.end(1) if m.lastindex else m.end()
            if _overlaps(claimed, s, e):
                continue
            if validator and not validator(m.group(0)):
                continue
            _emit(entities, claimed, text, s, e, etype, 1.0)


    # ── Part B: entropy net (probabilistic, last) ──
    if method != "off":
        enc = _get_encoder() if method in ("bpe", "both") else None
        bpe_thr = _bpe_threshold(cfg) if enc is not None else None
        bpe_usable = enc is not None and bpe_thr is not None

        # Degrade to Shannon, loudly, when a BPE mode has no encoder. "both"
        # already did this per-candidate; "bpe" did not — with no encoder it
        # evaluated `fired = False` for every candidate, so the entropy net had
        # zero recall and looked exactly like a clean run. That is the worst
        # failure mode a masking stage can have, and it is now the default mode.
        if method in ("bpe", "both") and not bpe_usable:
            # Was: warn once per process and run Shannon. That printed one
            # line at boot and then served a different entropy profile for
            # the rest of the process, with every per-message output looking
            # identical to a BPE run. Now it refuses.
            raise _encoder_required_error(method)

        # base64url-ish and long-hex candidates. '=' is base64 PADDING only, so
        # it may appear at the end but never mid-string — otherwise a candidate
        # would swallow the key in `token=<secret>` and mask the key name too.
        cand_re = re.compile(r'[A-Za-z0-9+/_-]{%d,}={0,2}|\b[0-9a-fA-F]{32,}\b' % min_len)
        for m in cand_re.finditer(text):
            s, e = m.start(), m.end()
            if _overlaps(claimed, s, e):
                continue
            val = m.group(0)
            veto = benign_format(val, text, s)
            if veto:
                VETO_COUNTS[veto] = VETO_COUNTS.get(veto, 0) + 1
                continue

            is_hex = bool(re.fullmatch(r'[0-9a-fA-F]+', val))
            fired, score = False, 0.0

            # Shannon: character-level randomness.
            ent_fired, ent_score = False, 0.0
            if method in ("shannon", "both"):
                ent = shannon_entropy(val)
                limit = thr_hex if is_hex else thr_b64
                if ent >= limit:
                    ent_fired = True
                    ent_score = round(min(1.0, ent / 6.0), 3)

            # BPE: compressibility against learned subword merges. Reaching
            # here with a bpe method guarantees a usable encoder — the
            # unusable case was rewritten to "shannon" above.
            bpe_fired = False
            if method in ("bpe", "both"):
                bpe_fired = _token_efficiency(val, enc) < bpe_thr

            if method == "shannon":
                fired, score = ent_fired, ent_score
            elif method == "bpe":
                fired, score = bpe_fired, 0.6 if bpe_fired else 0.0
            else:  # "both" — BPE is a PRECISION GATE on Shannon, not a union.
                # Shannon's dominant false positives are word-like structured
                # identifiers (dotted API paths, camelCase symbols, base64 asset
                # names): high character entropy, but they compress into a few
                # common merges. Requiring both signals suppresses exactly those.
                fired = ent_fired and bpe_fired
                score = max(ent_score, 0.6) if fired else 0.0

            if fired:
                _emit(entities, claimed, text, s, e, "HIGH_ENTROPY_SECRET", score)

    if debug:
        by_type = {}
        for ent in entities:
            by_type[ent["entity_type"]] = by_type.get(ent["entity_type"], 0) + 1
        print(f"  [FORM] {len(entities)} entities: {by_type}")
        if VETO_COUNTS:
            print(f"  [FORM] benign-format vetoes: {dict(VETO_COUNTS)}")

    # Sanity: offsets must be exact.
    for ent in entities:
        assert text[ent["start"]:ent["end"]] == ent["text"], ent

    return entities


def warmup(config):
    """Optional: pre-load the BPE encoder at startup.

    Also surfaces the threshold in the log so a misconfigured
    --form-bpe-threshold is visible before a 30-minute run, not after.
    """
    cfg = (config or {}).get("form", {})
    method = cfg.get("entropy_method", ENTROPY_METHOD_DEFAULT)
    if method in ("bpe", "both"):
        enc = _get_encoder()
        thr = _bpe_threshold(cfg)
        if enc is not None:
            print(f"  [FORM] BPE ready (cl100k_base), threshold={thr} chars/token")
        else:
            # Fail closed, and fail HERE. This used to warn that the run
            # would fall back to Shannon; it no longer falls back, so the
            # warning was advertising behaviour that does not exist. Raising
            # at warmup means the operator finds out before the first
            # prompt rather than on message 1 of a 1060-message run — and a
            # server that cannot honour its configured entropy method does
            # not start up reporting success.
            raise _encoder_required_error(method)


# ══════════════════════════════════════════════════════════════════════
# Self-test — synthetic valid values constructed from the standards, NOT the
# corpus. Run:  python3 form_stage.py
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    failures = 0

    def check(cond, label):
        global failures
        if cond:
            print(f"ok   {label}")
        else:
            failures += 1
            print(f"FAIL {label}")

    # ── primitives ──
    check(keccak256(b"").hex() ==
          "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
          "keccak256(empty) matches known vector")
    check(verhoeff_valid("2366") and not verhoeff_valid("2365"), "Verhoeff 2366")
    check(damm_valid("5724") and not damm_valid("5723"), "Damm 5724")

    # ── checksum validators (all vectors are standards examples / test data) ──
    vecs = [
        (luhn_valid, "4111111111111111", True, "Luhn test Visa"),
        (luhn_valid, "4111111111111112", False, "Luhn broken"),
        (iban_valid, "GB82WEST12345698765432", True, "IBAN GB example"),
        (iban_valid, "DE89370400440532013000", True, "IBAN DE example"),
        (iban_valid, "NO9386011117947", True, "IBAN NO (country not in repo)"),
        (iban_valid, "GB82WEST12345698765433", False, "IBAN GB corrupted"),
        (lei_valid, "5493001KJTIIGC8Y1R12", True, "LEI example"),
        (aba_valid, "021000021", True, "ABA Chase"),
        (aba_valid, "021000020", False, "ABA broken"),
        (isin_valid, "US0378331005", True, "ISIN Apple"),
        (isin_valid, "US0378331006", False, "ISIN broken"),
        (sedol_valid, "B0YBKL9", True, "SEDOL example"),
        (cusip_valid, "037833100", True, "CUSIP Apple"),
        (gtin_valid, "4006381333931", True, "EAN-13 example"),
        (gtin_valid, "036000291452", True, "UPC-A example"),
        (isbn10_valid, "0306406152", True, "ISBN-10 example"),
        (gtin_valid, "9780306406157", True, "ISBN-13 example"),
        (vin_valid, "1HGCM82633A004352", True, "VIN Honda"),
        (vin_valid, "1HGCM82633A004353", False, "VIN broken"),
        (cpf_valid, "11144477735", True, "CPF example"),
        (cnpj_valid, "11222333000181", True, "CNPJ example"),
        (pesel_valid, "44051401359", True, "PESEL example"),
        (bsn_valid, "111222333", True, "BSN example"),
        (codice_fiscale_valid, "MRTMTT25D09F205Z", True, "Codice Fiscale example"),
        (dni_valid, "12345678Z", True, "DNI example"),
        (dni_valid, "12345678A", False, "DNI wrong letter"),
        (aadhaar_valid, "234123412349", True, "Aadhaar (Verhoeff-valid)"),
        (base58check_valid, "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", True, "BTC genesis"),
        (base58check_valid, "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNb", False, "BTC corrupted"),
        (bech32_valid, "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4", True, "Bech32 BIP-173"),
        (eip55_valid, "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed", True, "EIP-55 example"),
        (eip55_valid, "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAeD", False, "EIP-55 bad case"),
    ]
    for fn, val, expected, label in vecs:
        check(fn(val) == expected, f"{label}: {fn.__name__}({val})={expected}")

    # ── end-to-end form_detect: offsets, typing, placeholder guard ──
    cfg = {"form": {"entropy_method": "shannon"}}

    t = "wire to IBAN NO9386011117947 today"
    ents = form_detect(t, cfg)
    check(any(e["entity_type"] == "IBAN" and e["text"] == "NO9386011117947" for e in ents),
          "form_detect finds IBAN with exact text")

    t2 = 'card 4111 1111 1111 1111 on file'
    ents2 = form_detect(t2, cfg)
    check(any(e["entity_type"] == "CREDIT_CARD" for e in ents2), "form_detect finds card")

    t3 = "eth 0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed sent"
    ents3 = form_detect(t3, cfg)
    check(any(e["entity_type"] == "CRYPTO_ADDRESS" for e in ents3), "form_detect finds ETH addr")

    t4 = "token=[JWT_TOKEN_1] and sid S-1-5-21-3623811015-3361044348-30300820-1013"
    ents4 = form_detect(t4, cfg)
    check(not any("[JWT_TOKEN_1]" in e["text"] for e in ents4), "placeholder not masked")
    check(any(e["entity_type"] == "WINDOWS_SID" for e in ents4), "form_detect finds SID")

    t5 = "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abc123 here"
    ents5 = form_detect(t5, cfg)
    check(any(e["entity_type"] == "JWT" for e in ents5), "form_detect finds JWT")

    # entropy: a random secret fires; an English sentence does not
    t6 = "secret=Xk7Qp2Zr9Lm4Wv8Nb3Tc6Yd1Fg5Hj0As here"
    ents6 = form_detect(t6, {"form": {"entropy_method": "shannon"}})
    check(any(e["entity_type"] == "HIGH_ENTROPY_SECRET" for e in ents6),
          "entropy fires on random blob")
    t7 = "the quick brown fox jumps over the lazy dog repeatedly today"
    ents7 = form_detect(t7, {"form": {"entropy_method": "shannon"}})
    check(not any(e["entity_type"] == "HIGH_ENTROPY_SECRET" for e in ents7),
          "entropy quiet on prose")

    # entropy must mask only the value, never swallow the "key=" prefix
    t8 = "token=Xk7Qp2Zr9Lm4Wv8Nb3Tc6Yd1Fg5Hj0As done"
    ents8 = form_detect(t8, {"form": {"entropy_method": "shannon"}})
    check(any(e["entity_type"] == "HIGH_ENTROPY_SECRET" and not e["text"].startswith("token=")
              for e in ents8), "entropy masks value, not the key= prefix")

    # ══════════════════════════════════════════════════════════════════
    # Benign-format veto — every value below is constructed here from the
    # standard, never lifted from tests/test_data/.
    # ══════════════════════════════════════════════════════════════════
    ecfg = {"form": {"entropy_method": "shannon"}}

    def _entropy_texts(t):
        return [e["text"] for e in form_detect(t, ecfg)
                if e["entity_type"] == "HIGH_ENTROPY_SECRET"]

    # RFC 2046 §5.1.1 — the token appears in the Content-Type header (4 dashes
    # in the parameter) and again as a dash-boundary in the body (6 dashes).
    mp = ("Content-Type: multipart/form-data; boundary=----BoundaryQ7rT2uV5wX8yZ1aB\r\n"
          "\r\n------BoundaryQ7rT2uV5wX8yZ1aB\r\n"
          "Content-Disposition: form-data; name=\"f\"\r\n"
          "\r\n------BoundaryQ7rT2uV5wX8yZ1aB--\r\n")
    check(not _entropy_texts(mp), "veto: multipart boundary, both dash counts")

    # A dashed one-off is NOT a boundary — it occurs once, so no close-delimiter.
    check(_entropy_texts("--Xk7Qp2Zr9Lm4Wv8Nb3Tc6Yd1Fg5Hj0As done"),
          "veto: single-occurrence '--' string still fires")

    # W3C SRI: sha384 digest of b"" is 48 bytes; length is the invariant.
    sri = "sha384-" + base64.b64encode(hashlib.sha384(b"").digest()).decode()
    check(not _entropy_texts(f'<script integrity="{sri}"></script>'),
          "veto: SRI sha384 hash-expression")
    check(_entropy_texts('integrity="sha384-Xk7Qp2Zr9Lm4Wv8Nb3Tc6Yd1Fg5Hj0AsQ7rT2uV5wX8yZ1aB4cD"'),
          "veto: sha384- prefix with wrong digest length still fires")

    # OCI reference grammar: name@algorithm:encoded.
    oci = hashlib.sha256(b"layer").hexdigest()
    check(not _entropy_texts(f"docker pull registry.io/app@sha256:{oci}"),
          "veto: OCI image digest")
    check(_entropy_texts(f"password_sha256: {oci}"),
          "veto: same hex without the @-anchored digest label still fires")

    # ── Anti-regression: real secrets must still mask ──
    keeps = [
        ("bearer token",   "Authorization: Bearer qT7vZ2mK9pX4wR8sL1nB6yH3jF5dG0aC2eU8iO4t"),
        ("stripe-style",   "key sk_" "live_51QxKpMqN7sR0vY3aB6cE8gH1 rotate it"),
        ("session token",  "session=8b3f1c9e2a7d4650b1e8c3f9a2d7460b1c8e3f9a"),
        ("bare sha-1 hex", "digest 9f2c1a7b4e8d6035af2b19c7e5d3406182bfa9c1"),
    ]
    for label, t in keeps:
        check(bool(_entropy_texts(t)), f"anti-regression: {label} still masks")

    # These mask via their own recognizer, not entropy — confirm the handoff.
    tj = ("jwt eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9."
          "eyJzdWIiOiJzdmMtYnJva2VyIiwiZXhwIjoxODAwMDAwMDAwfQ."
          "qT7vZ2mK9pX4wR8sL1nB6yH3jF5dG0aC2eU8iO4t")
    check(any(e["entity_type"] == "JWT" for e in form_detect(tj, ecfg)),
          "anti-regression: JWT masked by the JWT recognizer")
    tc = "db postgres://svc_app:Tr0ub4dor3xKq7@db.internal:5432/prod"
    check(any(e["entity_type"] == "CONNECTION_STRING" for e in form_detect(tc, ecfg)),
          "anti-regression: connection-string credential masked")
    tp = ("-----BEGIN RSA PRIVATE KEY-----\n"
          "MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Qu\n"
          "-----END RSA PRIVATE KEY-----")
    check(any(e["entity_type"] == "PRIVATE_KEY" for e in form_detect(tp, ecfg)),
          "anti-regression: PEM private key masked")

    # ══════════════════════════════════════════════════════════════════
    # BPE token efficiency — regression guards.
    #
    # The threshold shipped at 1.000 for one release, which made the branch
    # unsatisfiable (efficiency is >= 1.0 for ASCII by construction) and looked
    # identical to "BPE found nothing". These assertions fail loudly if that
    # ever recurs.
    # ══════════════════════════════════════════════════════════════════
    # The pipeline runs whatever this constant says when config.yaml is silent.
    check(ENTROPY_METHOD_DEFAULT == "bpe", "production default is BPE-only")

    # Fail-closed path (owner decision 2026-10-05): with no encoder, a
    # bpe/both run must RAISE EncoderUnavailableError -- never degrade to
    # Shannon, never return an empty, clean-looking result. Until 2026-10-21
    # this case still asserted the pre-decision degrade-to-Shannon behaviour,
    # so it raised instead of checking and every case after it never ran.
    _saved_enc, _saved_warn = _tiktoken_enc, ENCODER_UNAVAILABLE_WARNED
    _tiktoken_enc = None
    _td = "secret=qT7vZ2mK9pX4wR8sL1nB6yH3jF5dG0aC2eU8iO4t here"
    _raised = False
    try:
        form_detect(_td, {})
    except EncoderUnavailableError:
        _raised = True
    finally:
        _tiktoken_enc, ENCODER_UNAVAILABLE_WARNED = _saved_enc, _saved_warn
    check(_raised, "no encoder: default mode refuses "
                   "(EncoderUnavailableError), never degrades")

    check(_bpe_threshold({}) > BPE_THRESHOLD_FLOOR,
          f"BPE default threshold {_bpe_threshold({})} clears the "
          f"{BPE_THRESHOLD_FLOOR} unsatisfiability floor")
    check(_bpe_threshold({"bpe_threshold": 1.0}) == BPE_THRESHOLD_DEFAULT,
          "BPE threshold below the floor is rejected and replaced")
    check(_bpe_threshold({"bpe_threshold": 2.5}) == 2.5,
          "BPE threshold override above the floor is honoured")

    _enc = _get_encoder()
    if _enc is None:
        print("  [skip] cl100k_base unavailable — BPE behaviour tests skipped")
    else:
        # Efficiency floor holds for ASCII: nothing can score below 1.0.
        check(_token_efficiency("1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b", _enc) >= 1.0,
              "token efficiency is >= 1.0 for ASCII")

        # Separation: random secret compresses badly, word-like identifier well.
        _secret = "qT7vZ2mK9pX4wR8sL1nB6yH3jF5dG0aC2eU8iO4t"
        _wordy = "PInvoke.User32.WindowMessage.WM_SYSCOLORCHANGE"
        _thr = _bpe_threshold({})
        check(_token_efficiency(_secret, _enc) < _thr,
              "BPE fires on a random secret")
        check(_token_efficiency(_wordy, _enc) > _thr,
              "BPE quiet on a word-like structured identifier")

        # bpe-only mode must actually detect something.
        _tb = "secret=qT7vZ2mK9pX4wR8sL1nB6yH3jF5dG0aC2eU8iO4t here"
        check(any(e["entity_type"] == "HIGH_ENTROPY_SECRET"
                  for e in form_detect(_tb, {"form": {"entropy_method": "bpe"}})),
              "entropy_method='bpe' detects a secret (branch is reachable)")

        # Gate property: "both" is a subset of "shannon", never a superset.
        _gate_txt = (
            "secret=qT7vZ2mK9pX4wR8sL1nB6yH3jF5dG0aC2eU8iO4t "
            "sym=PInvoke.User32.WindowMessage.WM_SYSCOLORCHANGE "
            "path=getUserAuthenticationTokenFromRequestContext"
        )

        def _ent_set(m):
            return {e["text"] for e in form_detect(_gate_txt, {"form": {"entropy_method": m}})
                    if e["entity_type"] == "HIGH_ENTROPY_SECRET"}

        _sh, _bo = _ent_set("shannon"), _ent_set("both")
        check(_bo <= _sh, "'both' is a gate on shannon, never a superset")
        check(_secret in _bo, "'both' keeps the real secret through the gate")

    print(f"\n{'PASSED' if not failures else str(failures) + ' FAILED'}")
    sys.exit(1 if failures else 0)
