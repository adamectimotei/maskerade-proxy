#!/usr/bin/env python3
"""Evidence-required masking for ORGANIZATION spans.

WHAT THIS IS
------------
A FILTER, not a detector. It runs over the entity list the NER slot has
already produced and removes ORGANIZATION spans that no independent
evidence source corroborates. GLiNER keeps its job of proposing spans;
this decides which proposals survive.

WHY
---
gliner|ORGANIZATION was 63% of all corpus over-masks (3,676 occurrences,
1,828 distinct strings) at 42% measured precision on gold. The reason is
structural, not a tuning problem: "is this an organisation" and "is this
the CLIENT" are different questions, and GLiNER answers the first one.
GitHub, Vue, OWASP and Django REST framework are organisations. They are
correct detections and useless masks.

Under the agreed policy the custom stage carries the client's own name,
entered per engagement. That frees ORGANIZATION to be precision-first: a
safety net for identity the pentester did not think to enter, not a
best-effort net for every proper noun. So the decision is inverted.
Today ORG is opt-out — mask everything proposed, subtract suppressions,
which is why the work was 1,828 strings of whack-a-mole. Here it is
opt-in: mask nothing unless evidence says client.

THE TWO EVIDENCE SOURCES
------------------------
E1  LEGAL FORM
    The span carries a registered entity legal form from ISO 20275 /
    the GLEIF Entity Legal Forms code list. Measured on a probe
    fixture of published names: 212 of 232 real legal entities
    across CZ/SK/DE/AT/UK/PL/HU and 13 other jurisdictions, and ZERO of
    282 published technology terms — no tech product, security tool,
    standards body, protocol acronym, DOM/CSS token or generic noun in
    any of three languages. On the 1,060-message corpus it fires on 3
    distinct strings / 4 occurrences.

E2  CORROBORATION
    The span shares a token with something already established as client
    identity in this chat: a custom rule, a masked hostname, an email
    domain. Measured on gold: fires on 16 of 23 ORGANIZATION true
    positives AND on 9 of 33 "false positives" — where all 9 are client
    identity the gold file does not annotate (an ERP product name, a
    travel company, a fleet-app identifier, a plant code, an internal
    programme name, two Android package ids, a custom header, an
    Android/Firebase span). It fires on ZERO of
    the true noise: not CSRF, staging, ns2, Frida, ViewState, Oracle
    TNS, aws-sdk-java, CISO, PoC, dev team or diag.

    E2 needs no vocabulary, no knowledge base and no threshold. It is
    self-calibrating on what the engagement already established.

WHAT WAS DELIBERATELY NOT BUILT
-------------------------------
BPE token efficiency. Gold AUC 0.231 (inverted 0.769) looked strong and
is a language artifact: within-language on the fixture it is cs 0.587
CI [0.431, 0.740], de 0.428 CI [0.292, 0.560] — both straddling 0.5 —
and en 0.613, the OPPOSITE sign to gold. Same-label between-language
spread (cs mean 2.33 -> en mean 4.33) is ~8x the within-language signal.
The gold-derived "zero-cost" threshold of 3.78 unmasks more than half of
UK client company names and both English department names in the
fixture. Do not re-propose it. This is the second BPE rejection after
the ID-stage probe (AUC 0.517); the failure modes differ — that one
measured nothing, this one measures the wrong thing, which is worse
because the headline number looks green.

Domain and slug tiers. E2 subsumes both: all 3 gold domain true
positives and all 8 gold slug true positives are corroborated, while the
slug noise (IT-SEC, ENYAQ_CONNECT_2026, clinical-warehouse-prod,
aws-sdk-java, oracle-tns) is not. A slug tier would also have inherited
15 of 30 DOM/CSS tokens from the fixture. Domains belong to the hostname
layer regardless.

An allowlist gate on E1 for famous vendors (Apple Inc., Microsoft
Corporation). Measured and rejected: requiring "not every non-legal-form
token is in the published allowlist" catches only 13 of 37 famous
vendors in the fixture while costing 6 real legal entities — SAP SE,
Adyen N.V., Admiral Group plc, Phoenix Group Holdings plc, SSE plc,
Wirecard AG. E1 is accepted unconditionally instead, because its
measured cost is 4 over-mask occurrences in 1,060 real messages. The
inverse case (a client genuinely named after a technology) is what
allowlist_overrides.yaml's `exclude:` list is for.

ATTRIBUTION
-----------
This is a separately counted filter. It never edits gliner's
output in place and never re-labels a span; it drops or keeps, and
org_stats() reports where each proposal died. Folding it into
stages/gliner.py would destroy attribution and let layer shadowing (one
stage's effect credited to another) hide the real effect.

SCOPE
-----
Runs after the detector slot (maskerade/pipeline.py, _run_detectors), in
the full tier, and only when gliner is on. Configured by the
`gliner.org_evidence` block in default_config.yaml.
"""
import os
import re
import unicodedata
from collections import Counter

# ──────────────────────────────────────────────────────────────
# Labels this filter governs. Anything else passes through
# untouched — this must never silently police another type.
# ──────────────────────────────────────────────────────────────
ORG_LABELS = {"ORGANIZATION", "ORG", "COMPANY_NAME", "COMPANY"}

DEFAULTS = {
    "enabled": True,
    "require_evidence": True,
    "labels": sorted(ORG_LABELS),
    "min_token_len": 4,
    # Detectors whose findings may corroborate, by the `detector` field.
    # High-confidence only: never another span-NER guess, or one ORG
    # hallucination corroborates the next and the error compounds within a
    # single chat.
    #
    # NOTE what is and is not reachable here. Everything in this list runs
    # INSIDE the NER slot. The custom and regex stages run BEFORE it and
    # their hits are already placeholders in `text`, so they can never
    # appear in `entities` — they reach E2 only through the raw text handed
    # to set_corroboration_context(). They are listed anyway so the key
    # reads as the policy it is, and so a future wiring that does surface
    # them needs no config change.
    "corroborate_from": ["checksum", "keyval", "id", "opf",
                         "custom", "regex", "hostname"],
    # Do not let a public third-party domain in the traffic corroborate a
    # public org name (github.com -> GitHub).
    "exclude_public_corroborators": True,
    "elf_file": "tests/allowlist/elf_forms.yaml",
}


# ══════════════════════════════════════════════════════════════
# E1 — legal form
#
# Two matching regimes, and the split matters:
#
#   PUNCTUATED forms are matched CASE-INSENSITIVELY. `a.s.`, `s.r.o.`,
#   `s.p.a.`, `n.v.`, `d.o.o.` appear in every casing in real traffic,
#   and the dots make a false match essentially impossible. The previous
#   regex was case-sensitive and therefore missed `Adriatic Pharma
#   s.p.a.` — a gold true positive — while matching `S.p.A.`.
#
#   BARE alphabetic forms stay CASE-SENSITIVE. `SE`, `AB`, `AS`, `CO`,
#   `AD`, `KG` are ordinary English and Czech words in lowercase, and a
#   case-insensitive match on them would fire on half the corpus.
#
#   SPELLED-OUT forms are matched case-insensitively as whole phrases.
#   Six Czech entities in the fixture miss E1 without them, including
#   `Dopravní podnik hl. m. Prahy, akciová společnost`. The GLEIF ELF
#   download carries the local-language full names alongside the
#   abbreviations; a generated elf_forms.yaml is preferred when present.
#   None ships in the package today, so the seed list below is what runs.
# ══════════════════════════════════════════════════════════════

# Seed list, used when elf_forms.yaml has not been generated. Head of the
# ISO 20275 list for the jurisdictions the team works in.
_SEED_PUNCTUATED = [
    r"a\.\s?s\.", r"s\.\s?r\.\s?o\.", r"spol\.\s+s\s+r\.\s?o\.",
    r"v\.\s?o\.\s?s\.", r"k\.\s?s\.", r"o\.\s?p\.\s?s\.", r"z\.\s?s\.",
    r"z\.\s?ú\.", r"š\.\s?p\.", r"st\.\s?p\.", r"o\.\s?z\.",
    r"s\.\s?p\.\s?a\.", r"s\.\s?r\.\s?l\.", r"s\.\s?l\.", r"s\.\s?l\.\s?u\.",
    r"s\.\s?a\.", r"s\.\s?a\.\s?s\.", r"s\.\s?a\.\s?r\.\s?l\.",
    r"n\.\s?v\.", r"b\.\s?v\.", r"c\.\s?v\.", r"v\.\s?o\.\s?f\.",
    r"d\.\s?o\.\s?o\.", r"d\.\s?d\.", r"d\.\s?n\.\s?o\.",
    r"sp\.\s?z\s?o\.\s?o\.", r"sp\.\s?j\.", r"sp\.\s?k\.",
    r"e\.\s?v\.", r"e\.\s?k\.", r"g\.\s?m\.\s?b\.\s?h\.",
    r"l\.\s?l\.\s?c\.", r"p\.\s?l\.\s?c\.", r"l\.\s?p\.",
    r"a/s", r"p\.\s?s\.\s?k\.",
]

_SEED_BARE = [
    "GmbH", "gGmbH", "mbH", "AG", "KGaA", "KG", "OHG", "UG", "SE", "SCE",
    "EEIG", "Kft", "Zrt", "Nyrt", "Bt", "Kkt", "EOOD", "OOD", "AD", "EAD",
    "SA", "SARL", "SAS", "SCI", "SCA", "SL", "SLU", "SRL", "SpA", "Lda",
    "NV", "BV", "SIA", "UAB", "Ltda", "Pty", "Pte",
    "AB", "ABp", "Abp", "ASA", "ANS",
    "ApS", "Oy", "Oyj", "AS", "Ltd", "Limited", "PLC", "Plc", "plc",
    "LLP", "LLC", "Inc", "Incorporated", "Corp", "Corporation",
    "Holding", "Holdings", "Gruppe", "Groep", "Group",
]

_SEED_SPELLED = [
    # cs / sk
    "akciová společnost", "společnost s ručením omezeným",
    "veřejná obchodní společnost", "komanditní společnost",
    "státní podnik", "obecně prospěšná společnost", "zapsaný spolek",
    "akciová spoločnosť", "spoločnosť s ručením obmedzeným",
    "verejná obchodná spoločnosť", "komanditná spoločnosť",
    "štátny podnik", "družstvo", "nadace", "nadácia",
    # de
    "Aktiengesellschaft", "Gesellschaft mit beschränkter Haftung",
    "Kommanditgesellschaft", "offene Handelsgesellschaft",
    "eingetragener Verein", "eingetragene Genossenschaft",
    "Europäische Gesellschaft", "Unternehmergesellschaft",
    # pl
    "spółka akcyjna", "spółka z ograniczoną odpowiedzialnością",
    "spółka jawna", "spółka komandytowa",
    # hu
    "részvénytársaság", "korlátolt felelősségű társaság",
    "betéti társaság", "közkereseti társaság",
    # fr / es / it / pt
    "société anonyme", "société à responsabilité limitée",
    "société par actions simplifiée",
    "sociedad anónima", "sociedad limitada",
    "società per azioni", "società a responsabilità limitata",
    "sociedade anónima",
    # nl / nordics / uk
    "naamloze vennootschap", "besloten vennootschap",
    "aktiebolag", "osakeyhtiö", "aksjeselskap", "anpartsselskab",
    "aktieselskab", "public limited company", "private limited company",
    "limited liability partnership", "limited liability company",
    "community interest company",
]

_MAX_TRAIL = r"[\s,.)\]\"'»]*$"


def _compile_forms(punctuated, bare, spelled):
    """Three regexes, three matching regimes. See the block comment."""
    punct_re = re.compile(
        r"(?:^|[\s,(\[])(?:" + "|".join(punctuated) + r")" + _MAX_TRAIL,
        re.IGNORECASE)
    bare_re = re.compile(
        r"(?:^|[\s,(\[])(?:" + "|".join(re.escape(b) for b in bare)
        + r")\.?" + _MAX_TRAIL)
    spell_re = re.compile(
        r"(?:^|[\s,(\[])(?:" + "|".join(re.escape(s) for s in spelled)
        + r")" + _MAX_TRAIL,
        re.IGNORECASE)
    return punct_re, bare_re, spell_re


def _package_dir():
    """Directory of the installed package, not the current working directory.

    Everything else in this package resolves data files this way --
    Pipeline._load_allowlist() does it and explains why. These two functions
    did not, which is how `tests/allowlist/...` came to be a path that
    resolves on exactly one machine in exactly one directory.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_data_path(path):
    """Absolute path for a configured data file, or the input unchanged.

    Tried in order: as given (so an absolute path or a deliberate
    CWD-relative one still wins), then package-relative, then the package's
    allowlist directory by basename -- which is where package-data actually
    ships *.yaml. Returns the first that exists; when none do, returns the
    original so the caller's existence check fails on the path the operator
    configured and the error names something they recognise.
    """
    if not path:
        return path
    for cand in (os.path.abspath(path),
                 os.path.join(_package_dir(), path),
                 os.path.join(_package_dir(), "allowlist",
                              os.path.basename(path))):
        if os.path.exists(cand):
            return cand
    return path


# (key, value) rather than a bare value. The cache used to be a plain
# global: the FIRST config to call _load_elf() in a process decided the legal
# forms for every later call, whatever config it passed. A benchmark that
# sweeps configurations in one process therefore measured the first one's ELF
# list under every other one's name, and nothing said so. gliner._get_model()
# already keys its cache; these two did not. Same bug, same fix.
_FORMS = None
_FORMS_KEY = None


def _elf_key(config):
    cfg = (config or {}).get("gliner", {}).get("org_evidence", {}) or {}
    path = _resolve_data_path(cfg.get("elf_file", DEFAULTS["elf_file"]))
    try:
        stamp = os.path.getmtime(path) if path and os.path.exists(path) else None
    except OSError:
        stamp = None
    return (path, stamp)


def _load_elf(config):
    """Prefer the generated GLEIF list; fall back to the seed.

    A seeded run is correct but narrower — it will miss legal forms from
    jurisdictions nobody hand-listed, which is exactly the generalisation
    failure ("would this hold on an app I have never seen?"). A generated
    GLEIF list removes that limit.
    """
    global _FORMS, _FORMS_KEY
    key = _elf_key(config)
    if _FORMS is not None and _FORMS_KEY == key:
        return _FORMS
    cfg = (config or {}).get("gliner", {}).get("org_evidence", {}) or {}
    path = _resolve_data_path(cfg.get("elf_file", DEFAULTS["elf_file"]))
    punct, bare, spelled = list(_SEED_PUNCTUATED), list(_SEED_BARE), list(_SEED_SPELLED)
    src = "seed"
    if path and os.path.exists(path):
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f) or {}
            p = doc.get("punctuated") or []
            b = doc.get("bare") or []
            s = doc.get("spelled_out") or []
            if p or b or s:
                # Generated punctuated entries are literals, not patterns.
                punct = sorted(set(punct) | {re.escape(x) for x in p})
                bare = sorted(set(bare) | set(b))
                spelled = sorted(set(spelled) | set(s))
                src = f"{path} ({len(p)}/{len(b)}/{len(s)})"
        except Exception as e:
            print(f"  org_evidence: could not read {path} ({e}); using seed forms")
    _FORMS = (_compile_forms(punct, bare, spelled), src)
    _FORMS_KEY = key
    return _FORMS


def has_legal_form(text, config=None):
    """E1. True when the span ENDS in a registered entity legal form.

    Suffix-anchored by design. Nearly every jurisdiction in the register
    writes the form after the name, and a form that could also match at
    the start would fire on ordinary prose that happens to open with one
    of these tokens. The cost is that a prefix-form entity
    ("sabiedriba ar ierobezotu atbildibu Baltija") misses E1; it still
    reaches E2, which is the right place for it.
    """
    s = (text or "").strip()
    if not s:
        return False
    (punct_re, bare_re, spell_re), _src = _load_elf(config)
    return bool(punct_re.search(s) or bare_re.search(s) or spell_re.search(s))


# ══════════════════════════════════════════════════════════════
# E2 — corroboration
# ══════════════════════════════════════════════════════════════

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Tokens that appear in everyone's infrastructure and would corroborate
# anything against anything. Dropping them is what stops a shared `portal`
# or `prod` from manufacturing evidence.
#
# PROVENANCE, stated plainly: this list is HAND-AUTHORED. It is ordinary
# hostname and environment vocabulary, not a list of anything observed in
# an engagement or read off an over-mask CSV — but it has no published
# source either, and that deserves suspicion. It is the
# weakest part of this module.
#
# It is narrower in effect than a detection vocabulary: an entry here can
# only ever WITHDRAW evidence, so a wrong entry costs recall on a client
# whose name contains that word (a client literally called "Portal") and
# can never cause an over-mask. That asymmetry is why it ships, not an
# argument that it is correct.
#
# The generic replacement, when someone has the corpus pass for it, is
# per-host spread: a token seen across many UNRELATED hosts is public infrastructure
# vocabulary, and a token confined to one host is that host's identity.
# That needs no list and calibrates itself. Until then, treat every entry
# here as an admission rather than a rule, and do not extend it by reading
# benchmark output — extend it, if at all, from a published hostname or
# environment-naming convention.
_STOPTOKENS = {
    "prod", "production", "test", "testing", "stage", "staging", "dev",
    "development", "uat", "qa", "demo", "sandbox", "local", "localhost",
    "internal", "external", "intranet", "corp", "corporate", "group",
    "portal", "server", "service", "services", "client", "clients",
    "admin", "user", "users", "team", "data", "database", "http", "https",
    "json", "html", "name", "main", "true", "null", "role", "type",
    "site", "web", "app", "apps", "api", "auth", "login", "mail", "email",
    "www", "cdn", "static", "assets", "cloud", "host", "node", "srv",
    "system", "systems", "platform", "network", "office", "company",
    "index", "default", "public", "private", "secure", "example",
}


# ──────────────────────────────────────────────────────────────
# The pipeline entity contract is
#   {"text", "entity_type", "start", "end", "score", "stage", "detector"}
# `stage` is "ner" for EVERYTHING in this slot — it is the slot name, not
# the producer — and `detector` is what actually names the backend
# ("gliner", "keyval", "checksum", "id", "opf", ...).
#
# An earlier version of this module read "label"/"type" and
# "layer"/"source", which are the benchmark's REPORTING field names, not
# the pipeline's. It matched nothing, passed every span through, and
# produced a run byte-identical to the baseline with a gate report
# reading "proposals scanned: 0". Read both spellings now, and shout when
# the labels do not line up — see the guard in org_evidence_filter.
# ──────────────────────────────────────────────────────────────

def _entity_label(e):
    for k in ("entity_type", "label", "type"):
        v = e.get(k)
        if v:
            return str(v).upper()
    return ""


def _entity_detector(e):
    for k in ("detector", "layer", "source"):
        v = e.get(k)
        if v:
            return str(v).lower()
    # "stage" is deliberately last and mostly useless here: every entity
    # in this slot carries stage="ner".
    return str(e.get("stage") or "").lower()


def _fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


def tokens_of(s, min_len=4):
    """Folded tokens >= min_len, camelCase split, stopwords removed.

    Camel splitting is kept ALONGSIDE the unsplit form rather than
    replacing it: AcmeOpsDriver splits to {acme, driver} and would
    never meet `acmeops` from acmeops-platform.example, so both
    granularities are emitted and the containment pass below closes the
    rest of the gap.
    """
    out = set()
    for variant in (s or "", _CAMEL.sub(" ", s or "")):
        for t in re.split(r"[^0-9a-z]+", _fold(variant)):
            if len(t) >= min_len:
                out.add(t)
    return out - _STOPTOKENS


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})")
_FQDN_RE = re.compile(
    r"\b((?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.){1,}"
    r"[A-Za-z]{2,24})\b")

# Not a TLD — stops file names and versioned identifiers being read as
# hostnames and donating their stems as evidence.
_NOT_TLD = {
    "js", "ts", "py", "go", "rb", "php", "java", "json", "yaml", "yml",
    "xml", "html", "htm", "css", "md", "txt", "png", "jpg", "jpeg", "gif",
    "svg", "sh", "conf", "config", "log", "sql", "war", "jar", "exe",
    "dll", "so", "min", "map", "lock", "gz", "zip", "bak", "tmp",
}


def _domain_stems(host, min_len=4):
    """Labels of an FQDN that could carry an organisation name.

    The last label is the TLD and the common infrastructure labels are
    already in _STOPTOKENS. No Public Suffix List here deliberately —
    this is evidence gathering, not domain parsing, and a wrong guess at
    the registrable boundary costs at most one extra candidate token
    which still has to match an ORG span to matter.
    """
    labels = [l for l in _fold(host).split(".") if l]
    if len(labels) < 2 or labels[-1] in _NOT_TLD:
        return set()
    out = set()
    for lab in labels[:-1]:
        for part in re.split(r"[^0-9a-z]+", lab):
            if len(part) >= min_len:
                out.add(part)
        if len(lab) >= min_len:
            out.add(lab)
    return out - _STOPTOKENS


# ──────────────────────────────────────────────────────────────
# Corroboration context.
#
# The detector slot only receives (text, cfg), and by the time it runs
# the regex stage has already replaced hostnames and emails with
# placeholders, so they cannot be recovered from `text`. Pipeline.run()
# therefore sets the raw text here before the slot runs (see
# maskerade/pipeline.py).
#
# Falls back to scanning the text it is given, which is correct when the
# regex layer is off and harmless when it is on (it simply finds less).
# ──────────────────────────────────────────────────────────────
_CTX = {"raw_text": None, "values": (), "tokens": None}


def set_corroboration_context(raw_text=None, values=None):
    _CTX["raw_text"] = raw_text
    _CTX["values"] = tuple(values or ())
    _CTX["tokens"] = None


def clear_corroboration_context():
    set_corroboration_context(None, None)


_PUBLIC_TOKENS = None
_PUBLIC_TOKENS_KEY = None


def _public_tokens(min_len=4):
    """Tokens from the published allowlist groups, used to stop a public
    third-party domain corroborating a public organisation name —
    github.com in the traffic must not make `GitHub` look like client
    identity. Published sources only; this module never adds a
    term of its own."""
    # Keyed on min_token_len, which is configurable and changes the token
    # set. Unkeyed, a sweep over min_token_len silently reused the first
    # value's tokens for every later run. (The directory scan below is
    # CWD-relative and currently finds nothing, so the set is empty in
    # this build either way; the key is here so the cache stops being
    # wrong the moment a per-source fix makes it non-empty.)
    global _PUBLIC_TOKENS, _PUBLIC_TOKENS_KEY
    if _PUBLIC_TOKENS is not None and _PUBLIC_TOKENS_KEY == min_len:
        return _PUBLIC_TOKENS
    toks = set()
    for d in ("tests/allowlist", "allowlist"):
        if not os.path.isdir(d):
            continue
        try:
            import yaml
            for fn in sorted(os.listdir(d)):
                if not fn.endswith((".yaml", ".yml")):
                    continue
                with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
                    doc = yaml.safe_load(f) or {}
                for _g, v in (doc.get("groups") or {}).items():
                    for t in (v.get("terms") or []):
                        for p in re.split(r"[^0-9a-z]+", _fold(str(t))):
                            if len(p) >= min_len:
                                toks.add(p)
        except Exception:
            pass
        break
    _PUBLIC_TOKENS = toks
    _PUBLIC_TOKENS_KEY = min_len
    return toks


def corroboration_tokens(text, entities, cfg):
    """Every token that counts as established client identity right now.

    The cached raw-text set is UNFILTERED. The published-allowlist
    subtraction is applied to the finished union at the end, because
    entity-derived tokens are added after the cache is read and would
    otherwise bypass the guard entirely.
    """
    min_len = int(cfg.get("min_token_len", DEFAULTS["min_token_len"]))
    if _CTX["tokens"] is not None:
        base = set(_CTX["tokens"])
    else:
        src = _CTX["raw_text"] if _CTX["raw_text"] is not None else (text or "")
        base = set()
        for v in _CTX["values"]:
            base |= tokens_of(v, min_len)
        for m in _EMAIL_RE.finditer(src):
            base |= _domain_stems(m.group(1), min_len)
        for m in _FQDN_RE.finditer(src):
            base |= _domain_stems(m.group(1), min_len)
        _CTX["tokens"] = set(base)

    # Detections from high-confidence layers in this same slot.
    allowed = set(cfg.get("corroborate_from", DEFAULTS["corroborate_from"]))
    for e in entities or []:
        if _entity_detector(e) in allowed and _entity_label(e) not in ORG_LABELS:
            base |= tokens_of(e.get("text") or "", min_len)

    # Applied to the UNION, not to one branch of it. This subtraction used
    # to live inside the cache-miss branch above, so it only ever covered
    # tokens derived from the raw text. A `github` token arriving from a
    # checksum/keyval/id/opf finding in the same NER slot reached the
    # corroboration set unfiltered and made `GitHub` read as established
    # client identity.
    if cfg.get("exclude_public_corroborators",
               DEFAULTS["exclude_public_corroborators"]):
        base -= _public_tokens(min_len)
    return base


def is_corroborated(span_text, corro, min_len=4):
    """E2. Returns the matching tokens, or None."""
    tt = tokens_of(span_text, min_len)
    if not tt or not corro:
        return None
    ov = tt & corro
    if ov:
        return sorted(ov)
    # Containment pass: a joined form (acmeopsdriver) never token-matches
    # a separated one (acmeops). Floored at 6 chars so a short fragment
    # cannot manufacture evidence.
    folded = _fold(span_text)
    ov = sorted(c for c in corro if len(c) >= 6 and c in folded)
    return ov or None


# ══════════════════════════════════════════════════════════════
# The filter
# ══════════════════════════════════════════════════════════════

_STATS = Counter()
_DROPPED = Counter()
_KEPT = Counter()
_SEEN_KEYS = Counter()
_SEEN_LABELS = Counter()


def reset_stats():
    _STATS.clear()
    _SEEN_KEYS.clear()
    _SEEN_LABELS.clear()
    _DROPPED.clear()
    _KEPT.clear()


def org_stats():
    return {
        "calls": _STATS["calls"],
        "proposed": _STATS["proposed"],
        "kept_legal_form": _STATS["kept_legal_form"],
        "kept_corroborated": _STATS["kept_corroborated"],
        "kept_both": _STATS["kept_both"],
        "dropped": _STATS["dropped"],
        "would_drop": _STATS["would_drop"],
        "dropped_top": _DROPPED.most_common(25),
        "kept_top": _KEPT.most_common(25),
        "elf_source": (_FORMS[1] if _FORMS else "not loaded"),
    }


def org_evidence_filter(text, config, entities, debug=False):
    """Drop ORGANIZATION spans that no evidence source corroborates.

    Returns a NEW list. Non-ORG entities pass through by identity, in
    order. Nothing is re-labelled, re-scored or merged — this filter only
    decides membership, so `caught_by` and per-TP claims stay readable.
    """
    cfg = dict(DEFAULTS)
    cfg.update(((config or {}).get("gliner", {}) or {}).get("org_evidence", {}) or {})
    if not cfg.get("enabled", True):
        return list(entities or [])

    labels = {str(l).upper() for l in cfg.get("labels", DEFAULTS["labels"])}
    min_len = int(cfg.get("min_token_len", DEFAULTS["min_token_len"]))
    _STATS["calls"] += 1

    org_idx = {i for i, e in enumerate(entities or [])
               if _entity_label(e) in labels}
    if not org_idx:
        _contract_guard(entities)
        return list(entities or [])

    corro = corroboration_tokens(text, entities, cfg)
    out = []
    for i, e in enumerate(entities or []):
        if i not in org_idx:
            out.append(e)
            continue
        span = e.get("text") or ""
        _STATS["proposed"] += 1
        legal = has_legal_form(span, config)
        match = is_corroborated(span, corro, min_len)
        if not (legal or match) and not cfg.get("require_evidence", True):
            # Report mode: keep the span, but count it where it belongs.
            # Filing it under "kept: corroborated" would make report mode
            # claim evidence it does not have, and the whole point of the
            # mode is to show what enforcing would cost.
            _STATS["would_drop"] += 1
            _DROPPED[span] += 1
            out.append(e)
            continue
        if legal or match:
            if legal and match:
                _STATS["kept_both"] += 1
                why = f"legal_form+corroborated{match}"
            elif legal:
                _STATS["kept_legal_form"] += 1
                why = "legal_form"
            else:
                _STATS["kept_corroborated"] += 1
                why = f"corroborated{match}"
            _KEPT[f"{span} [{why}]"] += 1
            if debug:
                print(f"    org_evidence KEEP  {span!r}  {why}")
            out.append(e)
        else:
            _STATS["dropped"] += 1
            _DROPPED[span] += 1
            if debug:
                print(f"    org_evidence DROP  {span!r}  no evidence")
    return out


def _contract_guard(entities):
    """Remember that entities arrived without a governed label.

    A filter that silently matches nothing is indistinguishable from one
    that is working with nothing to do, and the run it produces is
    byte-identical to the baseline. That cost a session once already.

    But the test is END-OF-RUN, not per-call: most chunks legitimately
    contain entities and no ORGANIZATION, so warning after a few such
    calls cries wolf. Only "entities were scanned on many calls and NOT
    ONE governed label was ever seen" means a field-name mismatch. This
    just records the evidence; report() decides.
    """
    if not entities:
        return
    _STATS["calls_with_entities"] += 1
    if _STATS["calls_with_entities"] <= 3:
        for e in entities:
            for k in e.keys():
                _SEEN_KEYS[k] += 1
            lab = _entity_label(e)
            if lab:
                _SEEN_LABELS[lab] += 1


def report():
    """Print org_stats() for a measurement run."""
    st = org_stats()
    if not st["calls"]:
        return
    # The signature of a contract mismatch is not "no ORGANIZATION spans"
    # — a run can honestly contain none. It is "entities were scanned and
    # NO LABEL AT ALL could be read from them", i.e. the label field is
    # named something this module does not know about.
    if (not st["proposed"] and _STATS["calls_with_entities"] >= 5
            and not _SEEN_LABELS):
        print("\n  !! org_evidence: scanned entities on "
              f"{_STATS['calls_with_entities']} calls and matched NO")
        print("     governed label in the whole run.")
        print(f"     entity keys seen : {sorted(_SEEN_KEYS)}")
        print(f"     labels seen      : {sorted(_SEEN_LABELS)[:20]}")
        print("     If ORGANIZATION spans are being produced, this is a")
        print("     field-name mismatch, not an empty result: the filter")
        print("     passed everything through and this run will match the")
        print("     baseline byte for byte.")
        return
    prop = st["proposed"] or 1
    print("\n  ORG EVIDENCE GATE  (where ORGANIZATION proposals died)")
    print(f"    legal forms loaded from : {st['elf_source']}")
    print(f"    proposals scanned       : {st['proposed']}")
    for key, label in (("kept_legal_form", "kept: E1 legal form"),
                       ("kept_corroborated", "kept: E2 corroborated"),
                       ("kept_both", "kept: both")):
        print(f"    {label:<24}: {st[key]} ({100.0 * st[key] / prop:.1f}%)")
    if st.get("would_drop"):
        print(f"    REPORT ONLY, would drop : {st['would_drop']} "
              f"({100.0 * st['would_drop'] / prop:.1f}%)  -- kept, not dropped")
    print(f"    dropped: no evidence    : {st['dropped']} "
          f"({100.0 * st['dropped'] / prop:.1f}%)")
    if st["kept_top"]:
        print("    top kept:")
        for s, n in st["kept_top"][:10]:
            print(f"      {n:>5}x  {s}")
    if st["dropped_top"]:
        print("    top dropped:")
        for s, n in st["dropped_top"][:10]:
            print(f"      {n:>5}x  {s}")


# ══════════════════════════════════════════════════════════════
# Self-test — runs the fixture and the documented gold cases so a
# regression in the regexes is visible without a GPU or a corpus run.
# Needs the probe fixture, which lives in the development tree and does
# not ship in the package.
# ══════════════════════════════════════════════════════════════

def _self_test():
    """Self-test, driven ONLY by published and synthetic data.

    Deliberately contains no string taken from the gold prompts or from an
    over-mask CSV. A self-test built from benchmark output becomes the
    thing the code is tuned to pass, which is the generalisation failure wearing a
    test's clothes: it would go green on exactly the cases already seen
    and say nothing about an app nobody has run yet.

    Populations come from the probe fixture (not shipped) — real
    listed companies for the mask side, published project/standards/
    protocol vocabularies for the keep side, invented slugs where real
    ones cannot be published — plus synthetic corroboration text built
    from names invented here.
    """
    import yaml
    fx = "tests/test_data/org_probe_names.yaml"
    if not os.path.exists(fx):
        print(f"  {fx} not found; run from the repo root.")
        return
    doc = yaml.safe_load(open(fx, encoding="utf-8"))
    groups = doc.get("groups") or {}

    print("E1 — legal form, over the fixture")
    print(f"  {'group':<24}{'expect':<11}{'match E1':>12}")
    keep_hits = []
    for g, v in groups.items():
        terms = v.get("terms") or []
        hits = [t for t in terms if has_legal_form(t)]
        print(f"  {g:<24}{v.get('expect','?'):<11}{len(hits):>6}/{len(terms):<5}")
        if v.get("expect") == "keep" and hits:
            keep_hits.append((g, hits))

    print("\n  E1 SPECIFICITY — `keep` groups that matched:")
    if not keep_hits:
        print("    none")
    for g, hits in keep_hits:
        for t in hits:
            print(f"    {g}: {t!r}")
    print("""
    A non-zero here is information, not automatically a bug, and it must
    not be silenced by editing the fixture — that is fitting the
    measurement to the detector. Decide which of these it is:

      (a) The legal form is real and the entry is a real registered
          entity that simply is not the client. E1 is behaving
          correctly; the fixture's two-way label is too coarse for it.
          Leave BOTH alone and record the count as a known cost.
      (b) The form collides with ordinary text (an abbreviation that is
          also a common word). That is a generic defect: add it to
          _RISKY_BARE in tests/allowlist/build_elf.py and regenerate.

    Only (b) is a code change. (a) is the Apple/Microsoft problem, which
    a knowledge signal resolves and no legal-form rule can.""")

    print("\nE2 — corroboration, on synthetic text")
    print("  Names invented for this test; no engagement string appears.")
    set_corroboration_context(
        raw_text="anna.svoboda@vitalis-klinika.cz ops@vitalis-klinika.cz "
                 "telemetry-eu.northgate-fleet.example "
                 "github.com stackoverflow.com "
                 "api.nordledger.example portal.vitalis-klinika.cz")
    cfg = dict(DEFAULTS)
    corro = corroboration_tokens("", [], cfg)
    should_match = ["Vitalis Klinika", "VitalisKlinikaPortal",
                    "cz.northgate.fleet", "NordLedger",
                    "vitalis-klinika-prod"]
    should_not = ["GitHub", "Stack Overflow", "CSRF", "staging",
                  "Oracle TNS", "dev team", "window", "frame-ancestors"]
    bad = []
    for s_ in should_match:
        m = is_corroborated(s_, corro)
        print(f"   {'CORROBORATED' if m else 'MISSED      '}  {s_!r}"
              f"{'  <- ' + str(m) if m else ''}")
        if not m:
            bad.append(("missed", s_))
    print()
    for s_ in should_not:
        m = is_corroborated(s_, corro)
        print(f"   {'FIRED (BAD) ' if m else 'no evidence '}  {s_!r}"
              f"{'  <- ' + str(m) if m else ''}")
        if m:
            bad.append(("false", s_))
    clear_corroboration_context()
    print(f"\n   E2 result: {len(bad)} problem(s)"
          f"{'  ' + str(bad) if bad else '  (clean)'}")
    print("   GitHub and Stack Overflow must read 'no evidence': their")
    print("   domains are in the traffic above, and the published-allowlist")
    print("   guard is what stops them vouching for themselves. If they")
    print("   fire, tests/allowlist/ was not found from this directory.")

    print("\nE2 — public term arriving from an ENTITY, not from raw text")
    print("  The case above only exercises the raw-text path. Entity tokens")
    print("  are unioned in after the cache is read, so they need their own")
    print("  case: the guard once covered only the raw-text branch and this")
    print("  path bypassed it while the rest of the self-test stayed green.")
    set_corroboration_context(
        raw_text="anna.svoboda@vitalis-klinika.cz")
    ent_corro = corroboration_tokens(
        "",
        [{"text": "github.com", "entity_type": "INTERNAL_HOSTNAME",
          "detector": "checksum"}],
        dict(DEFAULTS))
    ent_bad = []
    m = is_corroborated("GitHub", ent_corro)
    print(f"   {'FIRED (BAD) ' if m else 'no evidence '}  'GitHub' via entity"
          f"{'  <- ' + str(m) if m else ''}")
    if m:
        ent_bad.append(("false", "GitHub via entity"))
    # Second assertion so the first cannot be satisfied by disabling the
    # entity path wholesale: a non-public entity token must still vouch.
    m2 = is_corroborated("Vitalis Klinika", ent_corro)
    print(f"   {'CORROBORATED' if m2 else 'MISSED (BAD)'}  'Vitalis Klinika'"
          " still corroborated from raw text"
          f"{'  <- ' + str(m2) if m2 else ''}")
    if not m2:
        ent_bad.append(("missed", "Vitalis Klinika via raw text"))
    clear_corroboration_context()
    print(f"\n   entity-path result: {len(ent_bad)} problem(s)"
          f"{'  ' + str(ent_bad) if ent_bad else '  (clean)'}")

    print("\n  HAND-PICKED PARAMETERS, not yet swept:")
    print(f"    min_token_len          = {DEFAULTS['min_token_len']}"
          "   (corroboration token floor)")
    print("    containment floor      = 6   (is_corroborated fallback)")
    print("  Both were chosen, not measured. Sweep them against the fixture")
    print("  and the corpus before treating either as settled.")


if __name__ == "__main__":
    _self_test()
