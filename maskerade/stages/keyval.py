"""Key-driven value masking — structural stage.

STANDALONE / EXPERIMENTAL — nothing in pipeline/ imports this file.
It is wired in at runtime by benchmark.py. Delete this file and every
other flag keeps working.

The idea
--------
Span-based NER classifies text by what it *looks like*. An opaque token
like a claim number or an employee reference looks like nothing, so no
NER model of any size will find it. But the surrounding structure says
everything: a value sitting under a key called "claimNumber" is a client
identifier regardless of its format.

So: parse structure, match on the KEY, mask the VALUE.

Why this generalises
--------------------
Value formats are infinitely app-specific. Key names are not — developers
draw from a small shared vocabulary, and `claimId`, `claim_id`,
`claimNumber` and `ClaimID` are the same key in every codebase on earth.
That is the opposite property from adding regex patterns, which is why
this is worth doing and per-app patterns are not.

The same argument applies to the *containers*. There are infinitely many
applications but only a few dozen ways to write "this key names this
value" in an HTTP conversation, and every one of them is defined by an
RFC or a framework convention rather than by any single app. So the stage
is split in two:

    extraction      "where does this format put keys and values"
    classification  "is this key name sensitive"

Adding support for a new container format is one function in EXTRACTORS.
The classification rules, value guards and span-claiming logic below it
do not change.

Lexicon sources
---------------
  Rails config.filter_parameters default list
      passw, email, secret, token, _key, crypt, salt, certificate,
      otp, ssn, cvv, cvc
  OpenID Connect Core 1.0 §5.1 Standard Claims
      name, family_name, given_name, middle_name, nickname,
      preferred_username, gender, birthdate, email, phone_number,
      address, ...
  RFC 4519 (LDAP schema for user applications)
      short attribute names -> canonical long names: cn = commonName,
      sn = surname, uid = userid, ...  Used to normalise DN components
      onto the lexicons above instead of adding two-letter tokens.

Container format sources
------------------------
  RFC 3986 §3.2.1 / §3.3   URL userinfo, path segments, matrix params
  RFC 7578 / RFC 6266      multipart/form-data, Content-Disposition params
  RFC 6265                 Set-Cookie attribute lists
  RFC 8446 / RFC 9113      HTTP/2 pseudo-header fields (:authority, :path)
  RFC 4514                 LDAP distinguished-name string representation
  WHATWG HTML              form control name/value attribute pairing
  GraphQL spec §2.6        field arguments (name: Value)
  ADO.NET / JDBC           connection-string keyword=value; syntax
  SQL-92                   comparison predicates, INSERT column lists
  Google AIP-122           REST collection identifiers are plural nouns,
                           resource ids are opaque — the basis for the
                           path-segment extractor and plural folding

Tokenised matching, not substring matching
------------------------------------------
Rails matches raw substrings, so `auth` hits `author` and `key` hits
`monkey`. Datadog's tracer shipped substring keyword redaction, then had
to remove "too generic redaction keywords" one release later and add an
exclusion mechanism the release after that.

Instead we split each key on camelCase / snake_case / kebab-case / dots
into a token SET and match tokens:

    customerAccountNumber -> {customer, account, number}   match
    X-Tenant-Id           -> {x, tenant, id}               match
    author                -> {author}                      no match
    orderBy               -> {order, by}                   no match
    orderId               -> {order, id}                   match

Same generality, far fewer false positives, works across every naming
convention in use.

Modes
-----
  strict  Rule 3 needs an entity noun AND an id suffix (orderId).
  loose   Rule 3 also fires when the LAST token is an id suffix and no
          token is in the technical veto list. Catches industry-specific
          entity nouns we cannot enumerate, at a precision cost.

Where the key is not enough
---------------------------
Two places need the value as well, both because a key name that is
sensitive in one application is structural in the next:

  Rule 2b   `name` is qualified before it counts. firstName is a person
            unconditionally; customerName and bare `name` are a person
            only if the value is shaped like a proper noun; fileName,
            typeName and hostName are not people at all.

  Guards    dates, times and timestamps are dropped whatever key they
            sit under — a timestamp identifies nobody and masking it
            destroys the request ordering the pentester is reading. So
            are HTTP auth scheme names and values built from two or
            fewer distinct characters.

Known limitation: the proper-noun gate counts capitalised words, so a
Chinese, Japanese or Korean name written without spaces cannot pass it.
Those are left to Flair and the LLM stage.
"""
import bisect
import re


# ──────────────────────────────────────────────────────────────
# Rule 1 — secrets. Any token match masks the value.
# ──────────────────────────────────────────────────────────────

SECRET_TOKENS = frozenset({
    "passw", "passwd", "password", "pwd", "passphrase", "passcode",
    "secret", "token", "credential", "credentials", "creds",
    "auth", "authentication", "bearer",
    "apikey", "accesskey", "privatekey", "publickey",
    "privkey", "pubkey",
    "keystore", "truststore",
    "cert", "certificate", "crypt", "cipher", "salt",
    "hash", "hmac", "signature", "sign", "nonce", "seed", "mnemonic",
    "otp", "totp", "mfa", "pin",
    "session", "sid", "cookie", "csrf", "xsrf",
    "sas", "saml", "assertion",
})

# ──────────────────────────────────────────────────────────────
# Rule 1b — `key`, disambiguated.
#
# Bare `key` is the most over-broad secret token in the field — Rails'
# `:_key` substring and Datadog's withdrawn keywords are the cautionary
# cases. Under this tokenizer it fired on partitionKey, sortKey,
# routingKey, cacheKey, idempotencyKey, objectKey (S3), messageKey
# (Kafka) — none of them credentials. So `key` is no longer a plain
# secret token; its sense is decided by what qualifies it, exactly like
# `name` in Rule 2b.
#
#   secret qualifier      apiKey, encryptionKey, signingKey  -> CREDENTIAL
#   structural qualifier  partitionKey, cacheKey, objectKey  -> not secret
#   bare / unknown        key                                -> gated on the
#                         value actually being a high-entropy secret blob
#
# Real secret keys are almost always qualified, so the recall cost of the
# gate is small; the precision gain on the *Key structural family is large.
# The value gate is _is_secret_shaped() below: a length floor, a
# contiguity check, a character-class check, and an identifier rejection.
# ──────────────────────────────────────────────────────────────

KEY_SECRET_QUALIFIERS = frozenset({
    "api", "access", "secret", "private", "public", "encryption",
    "decryption", "signing", "signature", "master", "shared", "license",
    "licence", "ssh", "gpg", "pgp", "rsa", "aes", "hmac", "jwt",
    "deploy", "webhook", "subscription", "consumer", "publishable",
    "auth", "client", "app", "host", "root", "recovery", "vault",
    "cipher", "crypto", "symmetric", "asymmetric", "secure", "signer",
})

KEY_STRUCTURAL_QUALIFIERS = frozenset({
    "partition", "sort", "routing", "route", "cache", "idempotency",
    "idempotent", "primary", "foreign", "composite", "range", "shard",
    "map", "group", "dedupe", "dedup", "object", "bucket", "file",
    "translation", "message", "locale", "i18n", "resource", "natural",
    "surrogate", "index", "lookup", "unique", "sequence", "event",
    "meta", "row", "column", "field", "property", "config", "setting",
    "hot", "cold", "warm", "join", "aggregate", "cluster", "order",
    "search", "filter", "query", "attribute", "hash",
})

_SECRET_SHAPE_MIN = 16

# ── Identifier rejection for the bare-`key` value gate ────────
#
# The character-class test below is necessary but nowhere near
# sufficient. A dotted or camelCase identifier clears it exactly like a
# real token does:
#
#   eventTypes.USER_DISABLED_BY_TEMPORARY_LOCKOUT_ERROR.description
#   -> length 63, no spaces, no "/", has lower+upper+punctuation
#
# That is how `{"key": ..., "value": ...}` map entries — i18n bundles,
# permission catalogues, feature-flag dumps, config exports — end up
# masked as credentials. The idiom is universal: any framework, any
# language, any locale file uses it, so this is not a one-app quirk.
#
# What actually separates the two is that an identifier is made of
# pronounceable English word segments and a secret is not. Two cheap,
# language-level signals capture that without a dictionary:
#
#   segment coverage   how much of the string is made of runs of >=2
#                      letters of a single case — "eventTypes" is ~100%,
#                      "jScuF9qsCvJfTrMebdWp9vg12L5ljtmk" is much lower
#   phonotactics       English words interleave vowels; random tokens
#                      pile up consonants. A max consonant run of <=4
#                      and a vowel ratio of >=0.25 is word-like.
#   identifier marks   an identifier joins its segments with "." "_" "-"
#                      or a camelCase boundary. Without one of those the
#                      string is a single opaque run, which is what
#                      AKIAIOSFODNN7EXAMPLE and other all-caps or
#                      base32 tokens look like — those must keep masking.
#
# All three conditions must hold to reject, so a high-entropy token that
# happens to contain a couple of letter runs still masks. Measured over
# the benchmark corpus this drops 3,094 of 3,465 bare-`key` false
# positives while keeping 173 of 175 real secrets — and the two it
# rejects are only reachable when a genuine secret sits under a field
# named exactly `key`, since anything under `password`, `token` or
# `apiKey` is classified by name and never reaches this gate.
# ──────────────────────────────────────────────────────────────

_VOWELS = frozenset("aeiouyAEIOUY")

# Runs of >=2 same-case letters — the building blocks of identifiers.
_WORD_SEGMENT = re.compile(r'[A-Z]?[a-z]{2,}|[A-Z]{2,}')

# How identifiers join their segments: an explicit separator, or a
# camelCase boundary. A token with neither is one opaque run.
_IDENT_SEPARATOR = re.compile(r'[._\-]')
_CAMEL_BOUNDARY = re.compile(r'[a-z][A-Z]')

_SEGMENT_COVERAGE_MIN = 0.75
_CONSONANT_RUN_MAX = 4
_VOWEL_RATIO_MIN = 0.25


def _max_consonant_run(v):
    """Longest run of consecutive consonants."""
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
    """Share of letters that are vowels. 0 when there are no letters."""
    letters = [c for c in v if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in _VOWELS for c in letters) / len(letters)


def _looks_like_identifier(v):
    """True when the value reads as a programmatic identifier rather than
    a secret: mostly word segments, pronounceable, and joined the way
    identifiers are joined."""
    segments = _WORD_SEGMENT.findall(v)
    if not segments:
        return False
    coverage = sum(len(s) for s in segments) / len(v)
    if coverage < _SEGMENT_COVERAGE_MIN:
        return False
    if not (_max_consonant_run(v) <= _CONSONANT_RUN_MAX
            and _vowel_ratio(v) >= _VOWEL_RATIO_MIN):
        return False
    # No separator and no camel boundary: a single opaque run, i.e. a
    # token shape (AKIA…, base32 secrets), not an identifier.
    return bool(_IDENT_SEPARATOR.search(v) or _CAMEL_BOUNDARY.search(v))


def _is_secret_shaped(value):
    """True when a value looks like a credential blob: long, high-entropy,
    contiguous. Used only for bare `key` with no qualifier."""
    v = value.strip().strip('"\'')
    if len(v) < _SECRET_SHAPE_MIN:
        return False
    # Secrets are contiguous — no spaces, no path separators. This rejects
    # prose and S3-style object paths that a bare `key` may also hold.
    if any(c.isspace() for c in v) or "/" in v:
        return False
    classes = (
        any(c.islower() for c in v)
        + any(c.isupper() for c in v)
        + any(c.isdigit() for c in v)
        + any((not c.isalnum()) and (not c.isspace()) for c in v)
    )
    if classes < 2:
        return False
    # Character classes alone cannot tell `eventTypes.FOO_BAR.description`
    # from a real token — both clear the test above. Reject the identifier.
    if _looks_like_identifier(v):
        return False
    return True


def _classify_key_token(tokens, value):
    """Rule 1b. Returns (entity_type, rule_name) or (None, None)."""
    if tokens & KEY_SECRET_QUALIFIERS:
        return "CREDENTIAL", "key_secret"
    if tokens & KEY_STRUCTURAL_QUALIFIERS:
        return None, None
    # Bare `key`, or a qualifier we don't recognise: mask only if the
    # value itself is secret-shaped.
    if value is not None and _is_secret_shaped(value):
        return "CREDENTIAL", "key_bare"
    return None, None




PII_TOKENS = frozenset({
    # login handles (the `*Name` spellings are handled by NAME_* below)
    "username", "login", "nickname", "initials",
    # name components that stand alone unambiguously — the OIDC / SCIM /
    # LDAP / FHIR spelling of a person's name parts. `firstName`,
    # `lastName` etc. are handled by Rule 2b's qualifier logic.
    "surname", "forename", "maiden", "given",
    # contact
    "email", "mail", "phone", "mobile", "tel", "telephone",
    "fax", "msisdn",
    # postal.  `region` deliberately removed — it is overwhelmingly a
    # cloud / locale / config field (OTel cloud.region, aws_region), and
    # real address parts are covered by address/locality/city/postal.
    "address", "street", "city", "zip", "postal", "postcode",
    "locality", "county",
    # demographic
    "birthdate", "birthday", "dob", "birth", "gender",
    "nationality", "citizenship", "marital",
    # GDPR Art. 9 special categories
    "ethnicity", "religion", "biometric",
    # government identifiers
    "ssn", "sin", "nin", "nino", "taxid", "tin", "vat",
    "passport", "license", "licence",
    # financial
    "iban", "bic", "swift", "sortcode", "routing",
    "cvv", "cvc", "cardnumber", "pan",
    "salary", "income", "wage", "compensation", "bonus", "insurance",
    # health
    "patient", "diagnosis", "medical", "health", "mrn",
    # person-linked device / vehicle identifiers
    "imei", "imsi", "iccid", "vin",
})

# ──────────────────────────────────────────────────────────────
# Rule 2b — names.
#
# `name` cannot live in the flat list above. Every object in every REST
# API has one, and the token arrives through fileName, typeName,
# hostName, tableName, eventName, productName, serviceName — none of
# which is personal data. Spring's /actuator/env alone emits hundreds of
# {"name": "<config property>"} objects.
#
# So the token is classified by what qualifies it, and the two ambiguous
# groups are gated on the value actually being shaped like a proper noun.
#
#   login      userName, loginName      -> USERNAME, ungated: a login
#                                          handle is one opaque token and
#                                          would never pass a name gate
#   definite   firstName, familyName    -> PERSON_NAME, ungated (OIDC §5.1
#                                          given_name / family_name /
#                                          middle_name; SCIM RFC 7643
#                                          name.givenName / familyName)
#   person     customerName, holderName -> PERSON_NAME, gated
#   org        companyName, vendorName  -> ORGANIZATION, gated
#   anything else, including bare `name` -> PERSON_NAME, gated
# ──────────────────────────────────────────────────────────────

# Single-token spellings the camelCase splitter cannot decompose.
NAME_CONCATENATED = frozenset({
    "firstname", "lastname", "surname", "givenname", "familyname",
    "middlename", "fullname", "maidenname", "realname", "displayname",
})

NAME_LOGIN_QUALIFIERS = frozenset({
    "user", "login", "signin", "screen", "account", "nick",
})

NAME_DEFINITE_QUALIFIERS = frozenset({
    "first", "last", "full", "given", "family", "middle", "maiden",
    "sur", "legal", "preferred", "real", "birth",
})

NAME_PERSON_CONTEXT = frozenset({
    "customer", "employee", "contact", "patient", "member", "person",
    "holder", "client", "applicant", "owner", "recipient", "sender",
    "guest", "subscriber", "author", "assignee", "beneficiary",
    "display", "student", "citizen", "staff", "manager", "operator",
})

NAME_ORG_CONTEXT = frozenset({
    "company", "organisation", "organization", "org", "merchant",
    "vendor", "supplier", "partner", "tenant", "employer",
    "institution", "bank", "school", "hospital",
})

# ──────────────────────────────────────────────────────────────
# Rule 3 — client-scoped identifiers.
# ──────────────────────────────────────────────────────────────

ENTITY_TOKENS = frozenset({
    "customer", "client", "account", "user", "member", "employee",
    "staff", "person", "patient", "student", "citizen", "subscriber",
    "tenant", "org", "organisation", "organization", "company",
    "merchant", "vendor", "supplier", "partner",
    "order", "invoice", "payment", "transaction", "claim", "policy",
    "contract", "booking", "reservation", "ticket", "case",
    "shipment", "tracking",
    "device", "vehicle", "asset", "badge", "card", "document",
    "project", "workspace", "group", "team", "role", "department",
    "branch", "store", "site", "facility", "location",

    # ── vertical expansion ──
    # These fire only alongside an ID_SUFFIX (orderId, claimNumber), so
    # the residual false-positive risk is structurally bounded. Each is a
    # party, or a record bound to a party — catalog objects, config
    # objects and physical things (product, unit, material, batch, sku)
    # are deliberately excluded because *Id on those is not client data.

    # party / relationship
    "party", "counterparty", "contact", "lead", "prospect", "applicant",
    "beneficiary", "guarantor", "borrower", "dependent", "household",
    "payee", "payer", "creditor", "debtor", "insured", "passenger",
    "guest", "candidate", "contractor", "practitioner", "landlord",
    "dealer",
    # finance / payments (ISO 20022 elements, Stripe objects)
    "charge", "payout", "refund", "chargeback", "dispute", "mandate",
    "remittance", "settlement", "loan", "mortgage", "wallet", "folio",
    "disbursement",
    # insurance (ACORD)
    "endorsement", "quote",
    # healthcare (FHIR resource types)
    "encounter", "prescription", "referral", "admission", "episode",
    "specimen",
    # logistics
    "consignment", "waybill", "awb", "parcel",
    # travel
    "itinerary",
    # gov / education / utilities
    "taxpayer", "permit", "benefit", "docket", "enrollment",
    "transcript", "meter", "premise",
    # HR / payroll
    "payroll", "payslip", "paystub", "timesheet",
    # legal / property / commerce
    "matter", "litigation", "lease", "escrow", "agreement",
    "opportunity", "subscription", "cart", "basket", "voucher",
    "coupon", "fulfillment", "registration", "warranty",
})

ID_SUFFIX_TOKENS = frozenset({
    "id", "ids", "uuid", "guid", "ref", "reference",
    "no", "num", "number", "code", "sid", "oid", "urn",
    "slug", "handle", "identifier", "serial",
})

# Loose mode only: any token here vetoes Rule 3. These are structural or
# protocol-level identifiers, never client data.
TECH_VETO_TOKENS = frozenset({
    "content", "status", "error", "http", "https", "response",
    "version", "api", "schema", "spec",
    "page", "limit", "offset", "sort", "count", "total", "size",
    "length", "max", "min", "index", "seq", "sequence",
    "time", "date", "timestamp", "epoch", "duration", "timeout",
    "type", "format", "mode", "env", "environment", "config",
    "trace", "span", "thread", "process", "node", "pod", "container",
    "build", "commit", "revision", "branchname",
    "line", "column", "row", "retry", "attempt",
    "priority", "level", "severity", "log",
    "currency", "language", "encoding", "charset", "mime",
    "port", "protocol", "scheme", "method", "verb",
    "color", "colour", "icon", "style", "theme",
})

# ──────────────────────────────────────────────────────────────
# Metadata veto — a value that describes a secret is not the secret.
#
# Every sensitive token fires regardless of what qualifies it, so
# sessionTimeout, hashAlgorithm, passwordMinLength, authMethod and
# certFormat all mask their value today — a number, "SHA-256", "PEM",
# "oauth2" — none of which is sensitive. The value there is metadata
# ABOUT the secret, named by the head noun.
#
# Two shapes, both grounded in the universal way config / schema keys
# are named rather than in any one application:
#
#   (A) the last token is a metadata noun     -> the value is a property
#       hashAlgorithm, sessionTimeout, certFormat, passwordPolicy
#
#   (B) a countable-unit secret token appears with a counting word
#       max_tokens, prompt_tokens, completion_tokens, token_count
#       — here "tokens" is a text-unit count, never a credential. This is
#       the one polysemous secret word hit by a counting *prefix*, so the
#       last-token test in (A) cannot catch it.
#
# `state` is intentionally NOT a metadata suffix: sessionState / authState
# carry opaque OAuth values, and for a masking tool over-masking those is
# safe while under-masking leaks.
# ──────────────────────────────────────────────────────────────

METADATA_SUFFIX_TOKENS = frozenset({
    # shape / category of a value
    "type", "kind", "class", "format", "encoding", "scheme", "method",
    "algorithm", "algo", "mode", "strategy", "version",
    # governance objects describing a value
    "policy", "provider", "handler", "validator", "filter", "rotation",
    # quantities / bounds
    "length", "size", "count", "total", "limit", "max", "min",
    "threshold", "attempts", "retries",
    # lifecycle / timing (dates also handled by the C3 value guard)
    "timeout", "duration", "interval", "ttl", "expiry", "expiration",
    # boolean predicates (values also caught by the stoplist)
    "enabled", "disabled", "required", "optional", "supported",
    "valid", "verified", "present", "exists", "status",
})

# (B) counting words that turn a countable-unit secret token into a count.
COUNTING_TOKENS = frozenset({
    "max", "min", "num", "total", "sum", "avg", "count",
    "prompt", "completion", "input", "output",
    "used", "remaining", "available", "per",
})
# Secret tokens that also name a countable unit. `token` is the only one
# in wide use (LLM APIs report prompt/completion/total token counts).
COUNTABLE_UNIT_TOKENS = frozenset({"token"})


def _metadata_veto(key, tokens):
    """True when the value is metadata about a secret, not the secret."""
    if _last_token(key) in METADATA_SUFFIX_TOKENS:
        return True
    if tokens & COUNTABLE_UNIT_TOKENS and tokens & COUNTING_TOKENS:
        return True
    return False




SENSITIVE_HEADERS = frozenset({
    "authorization", "proxy-authorization", "www-authenticate",
    "authentication",
    "cookie", "set-cookie",

    # Client-IP propagation — every one carries an end-user IP, which is
    # personal data under GDPR. `x-forwarded-for` / `x-real-ip` were
    # already here.
    "x-forwarded-for", "x-real-ip",
    "forwarded",                      # RFC 7239 standardised form
    "cf-connecting-ip",               # Cloudflare (all plans)
    "true-client-ip",                 # Cloudflare Enterprise / Akamai
    "fastly-client-ip",
    "x-client-ip", "x-cluster-client-ip", "x-original-forwarded-for",
    "x-envoy-external-address", "x-forwarded-client-cert",

    # Authentication / API-key / signature headers. Several of these are
    # already reachable through key-token classification (x-api-key ->
    # key, x-auth-token -> auth+token); listing them explicitly decouples
    # header coverage from the token rules so the token rules can be
    # tightened later without losing header masking.
    "x-api-key", "api-key", "apikey",
    "x-auth-token", "x-access-token", "x-session-token", "x-session-id",
    "x-csrf-token", "x-xsrf-token", "xsrf-token", "csrf-token",
    "x-amz-security-token", "x-goog-api-key",
    "ocp-apim-subscription-key", "x-functions-key",
    "x-shopify-access-token", "private-token", "job-token",
    "x-gitlab-token", "circle-token", "x-vault-token",
    "x-hub-signature", "x-hub-signature-256",
    "dpop",                           # RFC 9449 proof-of-possession
})

# ──────────────────────────────────────────────────────────────
# Value guards — never mask these regardless of key.
# ──────────────────────────────────────────────────────────────

VALUE_STOPLIST = frozenset({
    "true", "false", "null", "nil", "none", "undefined",
    "n/a", "na", "-", "", "yes", "no", "ok",
    # HTTP authentication scheme names — IANA HTTP Authentication Scheme
    # Registry / RFC 9110 §11.1. The scheme names the credential format;
    # it is not the credential. `Authorization: Bearer eyJ...` still masks
    # whole, because the guard tests the complete value, and the complete
    # value there is "Bearer eyJ...", not "Bearer".
    "basic", "bearer", "digest", "negotiate", "ntlm", "hoba", "mutual",
    "vapid", "scram-sha-1", "scram-sha-256", "aws4-hmac-sha256",
})

# ── Temporal values ──────────────────────────────────────────
# A date, a time or a timestamp is dropped whatever key it sits under.
# Both halves of this matter:
#
#   - it carries nothing on its own. Two systems logging the same second
#     are not linkable to a client, and the value is reconstructible from
#     any other event in the trace.
#   - keys that name time are everywhere and overlap the sensitive
#     lexicons: lastLogin, sessionExpires, cardExpiry, patientSince.
#     Masking their values is pure noise in the mask_map, and a masked
#     timestamp destroys the request ordering the pentester is reading.
#
# Covers ISO-8601 / RFC 3339, the three common numeric orders, IMF-fixdate
# (RFC 9110 §5.6.7) and a bare year.

_TIME_RE = r'\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,9})?'
_ZONE_RE = r'(?:Z|z|[+-]\d{2}:?\d{2})'
_TEMPORAL = re.compile(
    r'^(?:'
    r'\d{4}-\d{2}-\d{2}(?:[T ]' + _TIME_RE + _ZONE_RE + r'?)?'      # ISO-8601
    r'|' + _TIME_RE + _ZONE_RE + r'?'                                # bare time
    r'|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}'                              # d/m/y, m/d/y
    r'|\d{4}[./]\d{1,2}[./]\d{1,2}'                                  # y/m/d
    r'|(?:19|20)\d{2}'                                               # bare year
    r'|[A-Za-z]{3},\s\d{1,2}\s[A-Za-z]{3}\s\d{4}\s' + _TIME_RE + r'\s[A-Z]{1,5}'
    r')\.?$'
)

# Epoch seconds / milliseconds. Only rejected when the key names time —
# a bare 10-digit number is otherwise indistinguishable from an account
# or member number, and dropping those would leak identifiers.
_EPOCH = re.compile(r'^\d{10}(?:\d{3})?$')

TEMPORAL_KEY_TOKENS = frozenset({
    "time", "date", "at", "on", "since", "until", "expires", "expiry",
    "expiration", "stamp", "timestamp", "created", "updated", "modified",
    "issued", "seen", "epoch", "ttl", "duration", "start", "end",
    "login", "logout", "valid",
})

# An existing placeholder from an earlier stage. Masking one produces the
# nested-placeholder corruption the masker's layers exist to prevent.
_PLACEHOLDER = re.compile(r'\[[A-Z][A-Z0-9_]*_\d{1,3}\]')

MIN_VALUE_LEN = 3


# ──────────────────────────────────────────────────────────────
# Key tokenisation
# ──────────────────────────────────────────────────────────────

_NON_ALNUM = re.compile(r'[^A-Za-z0-9]+')
# Acronym-aware camelCase split: APIKey -> API, Key
_CAMEL = re.compile(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+')


def _singular(tok):
    """Fold an English plural onto its singular.

    REST collection identifiers are plural by convention (Google AIP-122:
    "Collection identifiers must be ... plural"), so `/claims/{id}` binds
    the value to the key `claims` while every lexicon entry is singular.
    Rather than duplicating every noun, fold at tokenisation time.

    Conservative: only the regular endings, and never on `-ss` / `-us`
    (address, status, focus), which are singular already.
    """
    if len(tok) <= 3:
        return tok
    if tok.endswith("ies"):
        return tok[:-3] + "y"
    if tok.endswith("sses") or tok.endswith("ses"):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith("ss") and not tok.endswith("us"):
        return tok[:-1]
    return tok


def tokenize_key(key):
    """Split a key name into a lowercase token set.

    Handles snake_case, kebab-case, camelCase, PascalCase, dotted paths,
    and bracketed form names like user[email]. Both the literal token and
    its singular form are kept, so the set is a superset of the previous
    behaviour — it can only add matches, never remove one.
    """
    tokens = set()
    for chunk in _NON_ALNUM.split(key):
        if not chunk:
            continue
        for part in _CAMEL.findall(chunk):
            low = part.lower()
            tokens.add(low)
            tokens.add(_singular(low))
    return tokens


# ──────────────────────────────────────────────────────────────
# Rule evaluation
# ──────────────────────────────────────────────────────────────

# Token -> entity_type for nicer placeholder names. Falls through to the
# rule-level default when the token isn't listed.
_PII_TYPE_HINTS = {
    "email": "EMAIL_ADDRESS", "mail": "EMAIL_ADDRESS",
    "phone": "PHONE_NUMBER", "mobile": "PHONE_NUMBER",
    "tel": "PHONE_NUMBER", "telephone": "PHONE_NUMBER",
    "msisdn": "PHONE_NUMBER",
    "address": "PHYSICAL_ADDRESS", "street": "PHYSICAL_ADDRESS",
    "city": "PHYSICAL_ADDRESS", "zip": "PHYSICAL_ADDRESS",
    "postal": "PHYSICAL_ADDRESS", "postcode": "PHYSICAL_ADDRESS",
    "name": "PERSON_NAME", "firstname": "PERSON_NAME",
    "lastname": "PERSON_NAME", "surname": "PERSON_NAME",
    "givenname": "PERSON_NAME", "familyname": "PERSON_NAME",
    "middlename": "PERSON_NAME", "fullname": "PERSON_NAME",
    "username": "USERNAME", "login": "USERNAME", "nickname": "USERNAME",
    "iban": "IBAN", "bic": "IBAN", "swift": "IBAN",
    "cvv": "CREDIT_CARD", "cvc": "CREDIT_CARD",
    "cardnumber": "CREDIT_CARD", "pan": "CREDIT_CARD",
    "ssn": "NATIONAL_ID", "sin": "NATIONAL_ID", "nin": "NATIONAL_ID",
    "nino": "NATIONAL_ID", "taxid": "NATIONAL_ID", "tin": "NATIONAL_ID",
    "passport": "NATIONAL_ID",
    "birthdate": "DATE_OF_BIRTH", "birthday": "DATE_OF_BIRTH",
    "dob": "DATE_OF_BIRTH",
    "salary": "FINANCIAL_DATA", "income": "FINANCIAL_DATA",
    "wage": "FINANCIAL_DATA", "compensation": "FINANCIAL_DATA",
}


def classify_key(key, value=None, loose=False):
    """Decide whether a key's value is sensitive.

    Returns (entity_type, rule_name) or (None, None).
    Rules are evaluated most-specific first so the placeholder name
    reflects the strongest signal. `value` is consulted only by Rule 2b,
    where the key alone is not decisive.
    """
    lowered = key.lower().strip()

    # Rule 4 — exact header names
    if lowered in SENSITIVE_HEADERS:
        return "CREDENTIAL", "header"

    tokens = tokenize_key(key)
    if not tokens:
        return None, None

    # Metadata veto — the value describes a secret rather than being one.
    # Runs after the exact-header check (a real Authorization header never
    # ends in a metadata noun) but before every token rule.
    if _metadata_veto(key, tokens):
        return None, None

    # Rule 1 — secrets
    if tokens & SECRET_TOKENS:
        return "CREDENTIAL", "secret"

    # Rule 1b — `key`, disambiguated by qualifier or value shape.
    if tokens & {"key", "keys"}:
        kt, kr = _classify_key_token(tokens, value)
        if kt:
            return kt, kr
        if tokens & KEY_STRUCTURAL_QUALIFIERS:
            # A structurally-qualified *Key (routingKey, sortKey) is never
            # a credential AND never PII — veto here so a collision like
            # `routing` (AMQP key vs bank routing number) resolves to the
            # far more common structural sense rather than masking.
            return None, None
        # bare `key` with a non-secret-shaped value: fall through and let
        # the remaining rules decide (they almost never match `*Key`).

    # Rule 2 — direct PII
    pii_hit = tokens & PII_TOKENS
    if pii_hit:
        for tok in sorted(pii_hit):
            if tok in _PII_TYPE_HINTS:
                return _PII_TYPE_HINTS[tok], "pii"
        return "PERSONAL_DATA", "pii"

    # Rule 2b — names
    name_type, name_rule = _classify_name(tokens, value)
    if name_type:
        return name_type, name_rule

    # Rule 3 — client-scoped identifiers
    has_suffix = bool(tokens & ID_SUFFIX_TOKENS)
    if has_suffix:
        if tokens & ENTITY_TOKENS:
            return "SENSITIVE_FIELD", "entity_id"
        if loose:
            # Last token must BE a suffix, so `orderBy` and `idempotent`
            # don't qualify, and nothing technical may appear.
            last = _last_token(key)
            if last in ID_SUFFIX_TOKENS and not (tokens & TECH_VETO_TOKENS):
                return "SENSITIVE_FIELD", "loose_id"

    return None, None


def _last_token(key):
    parts = []
    for chunk in _NON_ALNUM.split(key):
        if chunk:
            parts.extend(_CAMEL.findall(chunk))
    return _singular(parts[-1].lower()) if parts else ""


# ── Proper-noun value gate (Rule 2b) ─────────────────────────
# Human and organisation names are two to five whitespace-separated
# words, each opening with a capital, carrying no digits and no syntax
# characters. Identifiers, type names, enum values and config property
# names are one token (systemEnvironment, ACTIVE, order_status) or carry
# punctuation. Word count is the discriminator that does the work.

# Nobiliary and patronymic particles, which stay lowercase inside a name.
NAME_PARTICLES = frozenset({
    "van", "von", "der", "den", "de", "del", "della", "di", "da", "dos",
    "du", "la", "le", "el", "al", "bin", "ibn", "ben", "abu", "mac",
    "af", "av", "ter", "op", "y", "e",
})

_NAME_SYNTAX_CHARS = set('@/\\:=?#&<>[]{}()|;_+*"$%^~`')
_NAME_WORD = re.compile(r"^[^\W\d_][\w'’\-.]{0,39}$", re.UNICODE)
_NAME_SPLIT = re.compile(r"[,\s]+")


def _is_cased(ch):
    """True when the character belongs to a script that has case at all.

    Han, Arabic, Hebrew, Thai and Devanagari names cannot be required to
    start with a capital because no such thing exists in those scripts.
    """
    return ch.lower() != ch.upper()


def _name_word_ok(word):
    if not _NAME_WORD.match(word):
        return False
    if any(ch.isdigit() for ch in word):
        return False
    if word[0].isupper() or not _is_cased(word[0]):
        return True
    bare = word.lower().rstrip('.')
    if bare in NAME_PARTICLES:
        return True
    # Abbreviated legal-entity forms stay lowercase: s.r.o., a.s., p.l.c.
    return "." in word and len(bare) <= 6


def _looks_like_proper_noun(value):
    v = value.strip().strip('"\'').rstrip('.')
    if not (3 <= len(v) <= 64):
        return False
    if any(c in _NAME_SYNTAX_CHARS for c in v):
        return False
    words = [w for w in _NAME_SPLIT.split(v) if w]
    if not (2 <= len(words) <= 5):
        return False
    return all(_name_word_ok(w) for w in words)


def _classify_name(tokens, value):
    """Rule 2b. Returns (entity_type, rule_name) or (None, None)."""
    if tokens & NAME_CONCATENATED:
        return "PERSON_NAME", "name_definite"
    if "name" not in tokens:
        return None, None
    if tokens & NAME_LOGIN_QUALIFIERS:
        return "USERNAME", "name_login"
    if tokens & NAME_DEFINITE_QUALIFIERS:
        return "PERSON_NAME", "name_definite"

    if tokens & NAME_ORG_CONTEXT:
        entity_type, rule = "ORGANIZATION", "name_org"
    elif tokens & NAME_PERSON_CONTEXT:
        entity_type, rule = "PERSON_NAME", "name_person"
    else:
        # Bare `name`, and every qualifier we do not recognise —
        # fileName, typeName, hostName, eventName. Only a value actually
        # shaped like a proper noun gets through.
        entity_type, rule = "PERSON_NAME", "name_bare"

    if value is None or not _looks_like_proper_noun(value):
        return None, None
    return entity_type, rule


def _value_is_maskable(value, tokens=frozenset()):
    """Value guards. Returns True when the value may be masked."""
    stripped = value.strip().strip('"\'')
    if len(stripped) < MIN_VALUE_LEN:
        return False
    if stripped.lower() in VALUE_STOPLIST:
        return False
    if _PLACEHOLDER.search(stripped):
        return False
    if _TEMPORAL.match(stripped):
        return False
    if tokens & TEMPORAL_KEY_TOKENS and _EPOCH.match(stripped):
        return False
    # A value of at most two distinct characters carries at most one bit
    # per character: it is fill, a separator or an already-redacted field,
    # never an identifier, name or credential.
    if len(stripped) >= 4 and len(set(stripped)) <= 2:
        return False
    return True


# ══════════════════════════════════════════════════════════════
# EXTRACTION
#
# Every extractor is  fn(text) -> iterable of (key, start, end).
# The value is always text[start:end], computed centrally in
# _iter_pairs(), so an extractor cannot produce an offset that
# disagrees with its own value.
#
# The input is a pentester's prompt: prose mixed with HTTP dumps, so we
# cannot assume any of it parses as valid JSON/XML/YAML. Everything here
# is a scanner over raw text, tolerant of truncation and interleaving.
# ══════════════════════════════════════════════════════════════

_TRIM_LEAD = ' \t"\''
_TRIM_TRAIL = ' \t"\','


def _trim(text, s, e):
    """Strip surrounding whitespace, quotes and a trailing comma."""
    while s < e and text[s] in _TRIM_LEAD:
        s += 1
    while e > s and text[e - 1] in _TRIM_TRAIL:
        e -= 1
    return s, e


def _lines(text):
    """[(line_without_newline, start, end)] over the whole text."""
    out = []
    pos = 0
    for raw in text.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        out.append((line, pos, pos + len(line)))
        pos += len(raw)
    return out


def _split_cells(text, start, end, delim):
    """Quote-aware split of text[start:end] on delim -> [(s, e)] spans."""
    spans = []
    cell_start = start
    quote = None
    i = start
    while i < end:
        c = text[i]
        if quote:
            if c == quote:
                quote = None
        elif c in '"\'':
            quote = c
        elif c == delim:
            spans.append((cell_start, i))
            cell_start = i + 1
        i += 1
    spans.append((cell_start, end))
    return spans


# ── 1. multipart/form-data ────────────────────────────────────
# RFC 7578: the part's Content-Disposition names the field, the part body
# after the blank line is its value. Burp shows uploads exactly this way.

_CD_NAME = re.compile(
    r'Content-Disposition:[^\r\n]*?;[ \t]*name[ \t]*=[ \t]*"(?P<key>[^"\r\n]{1,64})"'
    r'[^\r\n]*',
    re.IGNORECASE,
)
_PART_HEADERS = re.compile(r'(?:\r?\n[A-Za-z][\w\-]*:[^\r\n]*)*')
_BLANK_LINE = re.compile(r'\r?\n[ \t]*\r?\n')
_PART_END = re.compile(r'\r?\n--')

# Bodies larger than this are file uploads. Masking a whole binary blob as
# one entity is useless and would blow up the mask_map, so skip the part.
MAX_PART_BODY = 4096


def extract_multipart(text):
    for m in _CD_NAME.finditer(text):
        hdr_end = _PART_HEADERS.match(text, m.end()).end()
        blank = _BLANK_LINE.match(text, hdr_end)
        if not blank:
            continue
        vs = blank.end()
        end_m = _PART_END.search(text, vs)
        ve = end_m.start() if end_m else len(text)
        if ve <= vs or ve - vs > MAX_PART_BODY:
            continue
        s, e = _trim(text, vs, ve)
        if s < e:
            yield m.group("key"), s, e


# ── 2. JSON / JSON-lines ──────────────────────────────────────
# "key": "value"  |  "key": 12345  |  "key": true

_JSON_PAIR = re.compile(
    r'"(?P<key>[^"\\\s]{1,64})"\s*:\s*'
    r'(?:"(?P<qval>(?:[^"\\]|\\.){0,1024})"'
    r'|(?P<uval>-?\d[\d.eE+-]*|true|false|null))'
)


def extract_json(text):
    for m in _JSON_PAIR.finditer(text):
        for grp in ("qval", "uval"):
            if m.group(grp) is None:
                continue
            s, e = _trim(text, *m.span(grp))
            if s < e:
                yield m.group("key"), s, e
            break


# ── 3. XML / HTML element text ────────────────────────────────

_XML_PAIR = re.compile(
    r'<(?P<key>[A-Za-z][\w.\-:]{0,63})(?:\s[^>]*)?>'
    r'(?P<val>[^<>]{1,1024})'
    r'</(?P=key)\s*>'
)


def extract_xml_text(text):
    for m in _XML_PAIR.finditer(text):
        s, e = _trim(text, *m.span("val"))
        if s < e:
            # Strip an XML namespace prefix: <ns:ssn> classifies as ssn.
            key = m.group("key").rsplit(":", 1)[-1]
            yield key, s, e


# ── 4. XML / HTML attributes ──────────────────────────────────
# Two distinct structures live in a tag:
#
#   a) form controls — WHATWG HTML defines `name` as the name used for
#      form submission and `value` as the datum. So in a tag carrying
#      both, `name` is the KEY and `value` is the VALUE. That is the
#      single most common place a sensitive value appears in an HTML
#      response body.
#   b) ordinary attributes — <user ssn="..."/>, SOAP, SAML, config XML.
#
# Attributes that markup itself defines (id/class/href/...) are skipped:
# their values are structure, not data. `name` and `value` participate
# only in (a) for the same reason — `name="ssn"` is a field label, and
# masking the literal string "ssn" would corrupt every other mention of
# it in the prompt.

_TAG = re.compile(r'<(?P<tag>[A-Za-z][\w.\-:]{0,63})(?P<attrs>\s[^<>]{0,4096}?)/?>')
_ATTR = re.compile(
    r'(?P<key>[A-Za-z_:][\w.\-:]{0,63})\s*=\s*'
    r'(?P<q>["\'])(?P<val>(?:(?!(?P=q)).){0,1024})(?P=q)'
)

MARKUP_ATTRS = frozenset({
    # WHATWG global attributes + the structural ones that carry no data
    "id", "class", "style", "title", "lang", "dir", "hidden", "tabindex",
    "role", "slot", "part", "is", "translate", "contenteditable",
    "href", "src", "srcset", "action", "formaction", "rel", "target",
    "type", "method", "enctype", "accept", "charset", "encoding",
    "width", "height", "size", "maxlength", "minlength", "pattern",
    "placeholder", "alt", "for", "colspan", "rowspan", "span",
    "checked", "selected", "disabled", "readonly", "required",
    "multiple", "autocomplete", "autofocus", "novalidate",
    "xmlns", "version", "schemalocation", "nonce",
    # name/value handled by the form-control pairing below
    "name", "value",
})


def extract_xml_attrs(text):
    for tag in _TAG.finditer(text):
        attrs = {}
        for a in _ATTR.finditer(tag.group("attrs")):
            base = tag.start("attrs")
            attrs[a.group("key").lower()] = (
                a.group("val"), base + a.start("val"), base + a.end("val"))

        # (a) form control: name/id names the value
        label = attrs.get("name") or attrs.get("id")
        if label and "value" in attrs:
            _, vs, ve = attrs["value"]
            s, e = _trim(text, vs, ve)
            if s < e:
                yield label[0], s, e

        # (b) ordinary data-bearing attributes
        for key, (_, vs, ve) in attrs.items():
            if key in MARKUP_ATTRS or key.startswith(("on", "aria-", "data-")):
                continue
            s, e = _trim(text, vs, ve)
            if s < e:
                yield key.rsplit(":", 1)[-1], s, e


# ── 5. Delimited tables ───────────────────────────────────────
# A header row names its columns for every row below it. This is how CSV
# exports, sqlmap dumps, DB console output and pasted markdown notes all
# carry client data, and Flair reliably fails on it because a table cell
# has no sentence around it.
#
# Fixed-width tables are deliberately NOT supported: there is no reliable
# generic signal separating aligned columns from ASCII art or indented
# prose, and column boundaries have to be guessed from whitespace runs.

_MD_SEPARATOR = re.compile(r'^[ \t]*\|?[ \t:|-]*-{2,}[ \t:|-]*\|?[ \t]*$')
_HEADER_CELL = re.compile(r'^["\']?[A-Za-z][A-Za-z0-9 _.\-/#()]{0,39}["\']?$')

TABLE_DELIMS = ("\t", ",", ";")
MAX_TABLE_COLS = 64


def _header_like(text, spans):
    if not (2 <= len(spans) <= MAX_TABLE_COLS):
        return False
    for s, e in spans:
        cell = text[s:e].strip()
        if not _HEADER_CELL.match(cell):
            return False
    return True


def _emit_table(text, header_spans, row_spans_list):
    keys = [text[s:e].strip().strip('"\'') for s, e in header_spans]
    for row in row_spans_list:
        for key, (s, e) in zip(keys, row):
            cs, ce = _trim(text, s, e)
            if cs < ce:
                yield key, cs, ce


def extract_tables(text):
    lines = _lines(text)
    n = len(lines)

    # (a) markdown / ASCII pipe tables — identified by the |---|---| rule
    i = 0
    while i < n:
        line, ls, le = lines[i]
        if i > 0 and "|" in line and _MD_SEPARATOR.match(line) and "|" in lines[i - 1][0]:
            hdr = _pipe_cells(text, *lines[i - 1][1:])
            rows = []
            j = i + 1
            while j < n and "|" in lines[j][0] and lines[j][0].strip():
                cells = _pipe_cells(text, *lines[j][1:])
                if len(cells) == len(hdr):
                    rows.append(cells)
                j += 1
            if len(hdr) >= 2 and rows:
                yield from _emit_table(text, hdr, rows)
            i = j
            continue
        i += 1

    # (b) delimited tables — a header row plus at least one row with the
    #     same column count under the same delimiter
    i = 0
    while i < n:
        line, ls, le = lines[i]
        if not line.strip() or "|" in line:
            i += 1
            continue
        matched = False
        for delim in TABLE_DELIMS:
            if delim not in line:
                continue
            hdr = _split_cells(text, ls, le, delim)
            if not _header_like(text, hdr):
                continue
            rows = []
            j = i + 1
            while j < n and lines[j][0].strip():
                cells = _split_cells(text, lines[j][1], lines[j][2], delim)
                if len(cells) != len(hdr):
                    break
                rows.append(cells)
                j += 1
            if rows:
                yield from _emit_table(text, hdr, rows)
                i = j
                matched = True
                break
        if not matched:
            i += 1


def _pipe_cells(text, ls, le):
    spans = _split_cells(text, ls, le, "|")
    # A markdown row starts and ends with the delimiter, producing empty
    # outer cells.
    if spans and text[spans[0][0]:spans[0][1]].strip() == "":
        spans = spans[1:]
    if spans and text[spans[-1][0]:spans[-1][1]].strip() == "":
        spans = spans[:-1]
    return spans


# ── 6. SQL ────────────────────────────────────────────────────
# Appears in error messages, ORM logs, sqlmap output and source dumps.
# Two shapes carry values: a comparison/assignment predicate, and an
# INSERT column list zipped against its VALUES tuples.

_SQL_PREDICATE = re.compile(
    r'\b(?:WHERE|AND|OR|SET|HAVING|ON)\b[^;\r\n]{0,256}?'
    r'\b(?:[\w"`\[\]]+\.)?(?P<key>[A-Za-z_]\w{0,63})\s*(?:=|<>|!=|\bLIKE\b)\s*'
    r"(?:'(?P<sval>(?:[^']|'')*)'|\"(?P<dval>[^\"]*)\"|(?P<nval>\d[\w.\-]*))",
    re.IGNORECASE,
)

_SQL_INSERT = re.compile(
    r'\bINSERT\s+INTO\s+[\w."`\[\]]+\s*\(\s*(?P<cols>[^()]{1,1024}?)\s*\)\s*VALUES\s*',
    re.IGNORECASE,
)
_SQL_TUPLE = re.compile(r'\(\s*(?P<vals>[^()]{0,4096}?)\s*\)')


def extract_sql(text):
    for m in _SQL_PREDICATE.finditer(text):
        for grp in ("sval", "dval", "nval"):
            if m.group(grp) is None:
                continue
            s, e = _trim(text, *m.span(grp))
            if s < e:
                yield m.group("key"), s, e
            break

    for m in _SQL_INSERT.finditer(text):
        cs, ce = m.span("cols")
        cols = [text[s:e].strip().strip('"`[]\' ')
                for s, e in _split_cells(text, cs, ce, ",")]
        pos = m.end()
        while pos < len(text):
            while pos < len(text) and text[pos] in ", \t\r\n":
                pos += 1
            t = _SQL_TUPLE.match(text, pos)
            if not t:
                break
            vs, ve = t.span("vals")
            cells = _split_cells(text, vs, ve, ",")
            if len(cells) == len(cols):
                for key, (s, e) in zip(cols, cells):
                    ts, te = _trim(text, s, e)
                    if ts < te:
                        yield key, ts, te
            pos = t.end()


# ── 7. LDAP distinguished names ───────────────────────────────
# RFC 4514 DN string form. Short attribute names are expanded to their
# RFC 4519 long names so they classify against the existing lexicons
# instead of adding two-letter tokens like `cn` that would collide with
# arbitrary application keys elsewhere in the prompt.
#
# ou= and dc= are intentionally not mapped: an organisational unit or
# domain component is structure shared by every DN in the directory, and
# the two-letter form appears in unrelated contexts (dc = datacenter).

LDAP_ATTRS = {
    "cn": "commonName",
    "sn": "surname",
    "givenname": "givenName",
    "displayname": "displayName",
    "uid": "userId",
    "userid": "userId",
    "mail": "mail",
    "samaccountname": "userName",
    "userprincipalname": "userName",
    "userpassword": "userPassword",
    "employeeid": "employeeId",
    "employeenumber": "employeeNumber",
    "telephonenumber": "telephoneNumber",
    "mobile": "mobile",
    "initials": "initials",
    "street": "street",
    "postalcode": "postalCode",
    "l": "city",
}

_LDAP_RDN = re.compile(
    r'(?<![\w.\-])(?P<key>' + "|".join(sorted(LDAP_ATTRS, key=len, reverse=True)) +
    r')=(?P<val>[^,;+\r\n]{1,256})',
    re.IGNORECASE,
)


def extract_ldap(text):
    for m in _LDAP_RDN.finditer(text):
        # Require DN context: an RDN is followed or preceded by another
        # comma-separated component. Without this, `l=5` in a query
        # string would be read as a city.
        tail = text[m.end():m.end() + 2]
        if not (tail.startswith(",") or "," in text[max(0, m.start() - 64):m.start()]):
            continue
        s, e = _trim(text, *m.span("val"))
        if s < e:
            yield LDAP_ATTRS[m.group("key").lower()], s, e


# ── 8. Line pairs: headers, YAML, INI, TOML, properties ───────
# `key: value` at line start. Extended over the original in three ways:
#
#   - optional leading ':' for HTTP/2 pseudo-headers (RFC 9113 §8.3)
#   - optional leading '-' for YAML sequence items
#   - the enclosing block's key is prepended, so a value under
#       credentials:
#         user: svc_acct
#     classifies as `credentials.user`, not `user`. Nesting is how every
#     config format binds a generic leaf name to a specific meaning.
#     Only the immediate parent is used — the full path would drag in
#     unrelated tokens from the root of the document.

_LINE_PAIR = re.compile(
    r'^[ \t]*(?:-[ \t]+)?(?P<key>:?[A-Za-z][\w.\-]{0,63})[ \t]*:[ \t]+'
    r'(?P<val>[^\r\n]{1,1024}?)[ \t]*$'
)
_LINE_PARENT = re.compile(r'^(?P<indent>[ \t]*)(?P<key>[A-Za-z][\w.\-]{0,63})[ \t]*:[ \t]*$')
_INI_SECTION = re.compile(r'^[ \t]*\[(?P<key>[A-Za-z][\w.\- ]{0,63})\][ \t]*$')

# A postal address, the longest legitimate field value, runs to about ten
# words. Anything longer under a `key: value` line is a sentence.
MAX_LINE_VALUE_WORDS = 10


def extract_line_pairs(text):
    section = None
    stack = []  # [(indent, key)]

    for line, ls, le in _lines(text):
        sec = _INI_SECTION.match(line)
        if sec:
            section = sec.group("key")
            stack = []
            continue

        par = _LINE_PARENT.match(line)
        if par:
            indent = len(par.group("indent").expandtabs(4))
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, par.group("key")))
            continue

        m = _LINE_PAIR.match(line)
        if not m:
            continue

        key = m.group("key").lstrip(":")
        indent = len(line[:m.start("key")].expandtabs(4))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1] if stack else section
        if parent:
            key = parent + "." + key

        s, e = _trim(text, ls + m.start("val"), ls + m.end("val"))
        if s >= e:
            continue
        # `key: value` is also the shape of an ordinary English sentence
        # with a colon in it. HTTP field values and YAML scalars that
        # carry identifiers are short; only prose runs long. The known
        # sensitive headers are exempt — a Cookie or Authorization value
        # is masked whole however long it is.
        if (key.rsplit(".", 1)[-1].lower() not in SENSITIVE_HEADERS
                and len(text[s:e].split()) > MAX_LINE_VALUE_WORDS):
            continue
        yield key, s, e


# ── 9. URL userinfo ───────────────────────────────────────────
# RFC 3986 §3.2.1. Same shape for https://, postgres://, mongodb://,
# amqp://, redis://, jdbc:mysql:// — credentials in the authority.

_URL_USERINFO = re.compile(
    r'(?P<scheme>[A-Za-z][\w+.\-]{1,31})://'
    r'(?P<user>[^\s/@:"\'<>]{1,128})'
    r'(?::(?P<pass>[^\s/@"\'<>]{0,128}))?@'
)


def extract_url_userinfo(text):
    for m in _URL_USERINFO.finditer(text):
        yield "username", m.start("user"), m.end("user")
        if m.group("pass"):
            yield "password", m.start("pass"), m.end("pass")


# ── 10. URL path segments ─────────────────────────────────────
# REST binds a resource id to the collection segment before it
# (Google AIP-122; Microsoft REST API Guidelines). /api/claims/CLM-2026-1
# names CLM-2026-1 with the key `claims` — which is exactly the value a
# pentester needs masked and no NER model can recognise.
#
# A path segment is only taken as a VALUE if it is identifier-shaped.
# Sub-resource and action segments are lowercase dictionary words drawn
# from a small vocabulary (/user/password/reset, /profile/email/verify);
# instance ids carry a digit, a separator, mixed case, or length. Without
# this guard the stage masks the literal word "reset" everywhere in the
# prompt.

_URL_IN_TEXT = re.compile(
    r'(?:[A-Za-z][\w+.\-]{1,31}://[^\s"\'<>]+)'
    r'|(?<![\w:/])(?:/[A-Za-z0-9._~%!$&\'()*+,;=:@\-]{1,128}){1,12}'
)
_STRIP_EXT = re.compile(
    r'\.(?:json|xml|html?|xhtml|php\d?|aspx?|jspx?|jsf|do|action|cgi|pl|py|rb|'
    r'js|css|txt|csv|tsv|pdf|png|jpe?g|gif|svg|zip|gz|tar|log|bak|old)$',
    re.IGNORECASE,
)
_MATRIX = re.compile(r';(?P<key>[A-Za-z_][\w.\-]{0,63})=(?P<val>[^;/?#]{0,256})')
_IDENT_SHAPE = re.compile(r'[0-9]|[-_.@%~+]|[A-Z]')


def _identifier_shaped(seg):
    return bool(_IDENT_SHAPE.search(seg)) or len(seg) >= 16


def extract_url_path(text):
    for m in _URL_IN_TEXT.finditer(text):
        s, e = m.span()
        chunk = text[s:e]

        # Trim the authority and everything from ?/# onwards.
        scheme = chunk.find("://")
        if scheme != -1:
            slash = chunk.find("/", scheme + 3)
            if slash == -1:
                continue
            s += slash
        for stop in "?#":
            cut = text.find(stop, s, e)
            if cut != -1:
                e = cut
        while e > s and text[e - 1] in '.,;:)]}>"\'':
            e -= 1
        if e <= s:
            continue

        # Split into segments, keeping absolute offsets.
        segs = []
        pos = s
        while pos < e:
            nxt = text.find("/", pos, e)
            if nxt == -1:
                nxt = e
            if nxt > pos:
                segs.append((pos, nxt))
            pos = nxt + 1

        cleaned = []
        for ss, se in segs:
            # Matrix params (RFC 3986 §3.3) are their own key/value pairs.
            semi = text.find(";", ss, se)
            body_end = semi if semi != -1 else se
            if semi != -1:
                for mm in _MATRIX.finditer(text, semi, se):
                    if mm.group("val"):
                        yield mm.group("key"), mm.start("val"), mm.end("val")
            ext = _STRIP_EXT.search(text[ss:body_end])
            if ext and ext.start() > 0:
                body_end = ss + ext.start()
            if body_end > ss:
                cleaned.append((ss, body_end))

        for (ks, ke), (vs, ve) in zip(cleaned, cleaned[1:]):
            key = text[ks:ke]
            value = text[vs:ve]
            if not key[:1].isalpha():
                continue
            if not _identifier_shaped(value):
                continue
            # Position supplies the id suffix that the segment name omits:
            # in /claims/{x} the collection segment never spells out "Id",
            # but AIP-122 defines the following segment as the resource
            # identifier. Classify `claims` as `claimsId` so Rule 3 sees
            # the entity noun and the suffix it structurally implies.
            yield key + "Id", vs, ve


# ── 11. Semicolon attribute lists ─────────────────────────────
# One syntax, three places it matters: Set-Cookie attributes (RFC 6265),
# Content-Disposition / Content-Type parameters (RFC 6266, RFC 9110), and
# ADO.NET / JDBC connection strings, whose keywords contain spaces
# (`User Id=sa;Password=...;Initial Catalog=...`) and so are invisible to
# a plain key=value scanner.
#
# Structural parameters are skipped: `name` and `filename` label a value
# rather than being one (masking the literal "ssn" out of
# `name="ssn"` would corrupt every other mention of the field).

_ATTR_LIST_PAIR = re.compile(
    r'(?:^|;)\s*(?P<key>[A-Za-z][\w.\- ]{0,63}?)\s*=\s*'
    r'(?:"(?P<qval>[^"\r\n]{0,1024})"|(?P<uval>[^;\r\n]{0,1024}))'
)

STRUCTURAL_PARAMS = frozenset({
    "name", "filename", "filename*", "boundary", "charset", "form-data",
    "path", "expires", "samesite", "secure", "httponly", "domain",
    "q", "version", "realm", "algorithm", "qop",
})


def extract_attr_lists(text):
    for line, ls, le in _lines(text):
        if ";" not in line or "=" not in line:
            continue
        for m in _ATTR_LIST_PAIR.finditer(text, ls, le):
            if m.group("key").strip().lower() in STRUCTURAL_PARAMS:
                continue
            grp = "qval" if m.group("qval") is not None else "uval"
            s, e = _trim(text, *m.span(grp))
            if s < e:
                yield m.group("key").strip(), s, e


# ── 12. Argument / object-literal pairs ───────────────────────
# GraphQL field arguments (spec §2.6) and JavaScript / JSON5 object
# literals both write `name: value` with an unquoted key, which the JSON
# scanner (quoted keys) and the line scanner (line-anchored) both miss.

_ARG_PAIR = re.compile(
    r'[({,]\s*(?P<key>[A-Za-z_]\w{0,63})\s*:\s*'
    r'(?:"(?P<qval>(?:[^"\\]|\\.){0,1024})"'
    r"|'(?P<sval>[^'\r\n]{0,1024})'"
    r'|(?P<uval>[\w.@+\-]{1,256}))'
)


def extract_args(text):
    for m in _ARG_PAIR.finditer(text):
        for grp in ("qval", "sval", "uval"):
            if m.group(grp) is None:
                continue
            s, e = _trim(text, *m.span(grp))
            if s < e:
                yield m.group("key"), s, e
            break


# ── 13. key=value ─────────────────────────────────────────────
# Query strings, form bodies, cookie pairs, env dumps, .env exports,
# Java properties, logfmt and syslog structured data. Quoted values are
# accepted so logfmt's `user="Jan Novak"` and RFC 9110 auth-params
# (`nonce="..."`) are not truncated at the first space.

_KV_PAIR = re.compile(
    r'(?P<key>[A-Za-z_][\w.\-\[\]]{0,63})\s*=\s*'
    r'(?:"(?P<qval>(?:[^"\\\r\n]|\\.){0,1024})"'
    r"|'(?P<sval>[^'\r\n]{0,1024})'"
    r'|(?P<uval>[^&;,\s"\'<>{}]{1,1024}))'
)


def extract_kv(text):
    # Sorted, non-overlapping by construction — binary search, because a
    # markup-heavy response body has thousands of both tags and pairs.
    starts = []
    ends = []
    for m in _TAG.finditer(text):
        starts.append(m.start())
        ends.append(m.end())

    def in_markup(pos):
        i = bisect.bisect_right(starts, pos) - 1
        return i >= 0 and pos < ends[i]

    for m in _KV_PAIR.finditer(text):
        # Inside a markup tag the pair is an attribute, and attributes are
        # extract_xml_attrs()'s job — it knows which attribute names are
        # structure (`name`, `id`, `class`) rather than data.
        if in_markup(m.start()):
            continue
        # Preceded by ';' on the same line, the pair is a field parameter
        # (RFC 9110 §5.6.6), not a query or form field — extract_attr_lists()
        # owns those and knows that `name=` / `filename=` label a value
        # rather than being one.
        line_start = text.rfind("\n", 0, m.start()) + 1
        if ";" in text[line_start:m.start()]:
            continue
        for grp in ("qval", "sval", "uval"):
            if m.group(grp) is None:
                continue
            s, e = _trim(text, *m.span(grp))
            if s < e:
                yield m.group("key"), s, e
            break


# ── Registry ──────────────────────────────────────────────────
# Order is load-bearing, coarsest-meaningful first. A pair only blocks a
# later one when it is actually masked, so a non-sensitive outer match
# never shadows a sensitive inner one — `X-Custom-Header: userId=4471`
# must still expose `userId`. But when the outer match IS sensitive it has
# to come first: `Cookie: a=b; c=d` is sensitive as a whole unit, so the
# line scanner must claim the entire value before the attribute-list and
# key=value scanners see the pairs inside it.

EXTRACTORS = [
    ("multipart", extract_multipart),
    ("json", extract_json),
    ("xml_text", extract_xml_text),
    ("xml_attr", extract_xml_attrs),
    ("table", extract_tables),
    ("sql", extract_sql),
    ("ldap", extract_ldap),
    ("line", extract_line_pairs),
    ("url_userinfo", extract_url_userinfo),
    ("url_path", extract_url_path),
    ("attr_list", extract_attr_lists),
    ("arg", extract_args),
    ("kv", extract_kv),
]


def _iter_pairs(text, only=None):
    """Yield (key, value, start, end, extractor_name) for every pair found.

    No span claiming happens here — the caller claims only the spans it
    actually masks.
    """
    for name, fn in EXTRACTORS:
        if only and name not in only:
            continue
        for key, start, end in fn(text):
            if 0 <= start < end <= len(text):
                yield key, text[start:end], start, end, name


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def keyval_detect(text, config, loose=False, debug=False):
    """Detect sensitive values by the key they are bound to.

    Args:
        text:   text to scan (already regex/custom-masked by the
                orchestrator, same as what the NER stage receives)
        config: pipeline config dict; reads config["keyval"] for
                extra_keys / ignore_keys / extractors overrides
        loose:  enable the looser Rule 3 variant
        debug:  print every pair considered

    Returns:
        List of entity dicts in pipeline shape.
    """
    kv_cfg = config.get("keyval", {})
    extra = {k.lower() for k in kv_cfg.get("extra_keys", [])}
    ignore = {k.lower() for k in kv_cfg.get("ignore_keys", [])}
    only = set(kv_cfg["extractors"]) if kv_cfg.get("extractors") else None
    if kv_cfg.get("loose") is not None:
        loose = bool(kv_cfg["loose"])

    entities = []
    claimed = []  # spans we actually masked; only these block later matches

    def overlaps(s, e):
        return any(s < ce and e > cs for cs, ce in claimed)

    for key, value, start, end, source in _iter_pairs(text, only):
        if key.lower() in ignore:
            continue

        entity_type, rule = classify_key(key, value, loose=loose)

        if entity_type is None and (key.lower() in extra or tokenize_key(key) & extra):
            entity_type, rule = "SENSITIVE_FIELD", "extra"

        if entity_type is None:
            continue

        if not _value_is_maskable(value, tokenize_key(key)):
            if debug:
                print(f"  [KEYVAL] skip  {key}={value[:40]!r} (value guard)")
            continue

        # A coarser sensitive match already covers this span — e.g. the
        # whole Cookie header value was masked, so don't also mask the
        # individual cookie pairs inside it.
        if overlaps(start, end):
            continue
        claimed.append((start, end))

        if debug:
            print(f"  [KEYVAL] mask  {key} -> {value[:50]!r}  ({rule}/{source})")

        entities.append({
            "text": value,
            "entity_type": entity_type,
            "start": start,
            "end": end,
            "score": 1.0,
            "stage": "ner",          # identical downstream semantics
            "detector": "keyval",    # benchmark reporting only
            "key": key,              # ignored by the pipeline, useful in debug
            "extractor": source,     # ignored by the pipeline, useful in debug
        })

    return entities


def warmup(config):
    """No model to load — present so the benchmark can treat all
    detector backends uniformly."""
    return None


# ──────────────────────────────────────────────────────────────
# Self-test — synthetic fixtures only, no corpus prompts.
#   python3 tests/keyval_stage.py
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    CFG = {}

    def run(text, loose=False):
        ents = keyval_detect(text, CFG, loose=loose)
        for e in ents:
            assert text[e["start"]:e["end"]] == e["text"], \
                f"offset mismatch: {e}"
        return {e["text"] for e in ents}

    CASES = [
        # (name, text, must_mask, must_not_mask, loose)
        ("json", '{"customerId": "CU-99", "page": 2}', {"CU-99"}, {"2"}, False),
        ("xml_text", '<ssn>123-45-6789</ssn>', {"123-45-6789"}, set(), False),
        ("xml_attr", '<user ssn="123-45-6789" id="row3"/>',
         {"123-45-6789"}, {"row3"}, False),
        ("html_form", '<input type="text" name="accountNumber" value="ACC-7781">',
         {"ACC-7781"}, {"accountNumber"}, False),
        ("md_table",
         "| userId | status |\n|--------|--------|\n| U-4471 | active |",
         {"U-4471"}, {"active"}, False),
        ("csv",
         "email,department,seats\nj.novak@corp.example,Ops,4\n",
         {"j.novak@corp.example"}, {"Ops"}, False),
        ("sql_where",
         "SELECT * FROM t WHERE customer_id = 'CU-8842' AND active = 1",
         {"CU-8842"}, set(), False),
        ("sql_insert",
         "INSERT INTO people (id, email, city) VALUES (7, 'a.b@x.example', 'Brno')",
         {"a.b@x.example", "Brno"}, set(), False),
        ("ldap",
         "cn=Jan Novak,ou=Users,dc=corp,dc=example",
         {"Jan Novak"}, {"Users"}, False),
        # Parent inheritance masks the whole block: everything under
        # `credentials:` is a credential, which is the point of the rule.
        ("yaml_parent",
         "credentials:\n  user: svc_deploy\n  host: build01\n",
         {"svc_deploy", "build01"}, set(), False),
        ("yaml_parent_scoped",
         "server:\n  host: build01\n  port: 8443\n",
         set(), {"build01"}, False),
        ("http2_pseudo",
         ":authority: api.example.test\nauthorization: Bearer abc123def\n",
         {"Bearer abc123def"}, set(), False),
        ("url_userinfo",
         "postgres://reporting:Sup3rS3cret@db01/analytics",
         {"reporting", "Sup3rS3cret"}, set(), False),
        ("url_path",
         "GET /api/v2/claims/CLM-2026-00847/documents HTTP/1.1",
         {"CLM-2026-00847"}, {"documents"}, False),
        ("url_path_action",
         "POST /user/password/reset HTTP/1.1",
         set(), {"reset"}, False),
        ("matrix_param",
         "GET /shop/items;sessionId=8fa2b1c9d4/detail",
         {"8fa2b1c9d4"}, set(), False),
        ("connstring",
         "Server=sql01;User Id=svc_rep;Password=Wint3r2026;Initial Catalog=crm",
         {"svc_rep", "Wint3r2026"}, set(), False),
        ("multipart",
         'Content-Disposition: form-data; name="ssn"\r\n\r\n'
         '760415/1234\r\n--boundary--',
         {"760415/1234"}, {"ssn"}, False),
        ("multipart_file_skipped",
         'Content-Disposition: form-data; name="password"; '
         'filename="dump.bin"\r\nContent-Type: application/octet-stream\r\n\r\n'
         + ("A" * 5000) + '\r\n--boundary--',
         set(), {"dump.bin"}, False),
        ("graphql",
         'query { employee(employeeId: "EMP-3391") { department } }',
         {"EMP-3391"}, set(), False),
        ("logfmt",
         'level=info user_name="Jan Novak" latency=42ms',
         {"Jan Novak"}, {"42ms"}, False),
        ("cookie_whole",
         "Cookie: SESSIONID=8fa2b1c9; theme=dark",
         {"SESSIONID=8fa2b1c9; theme=dark"}, set(), False),
        ("nested_header_exposed",
         "X-Custom-Header: userId=4471",
         {"4471"}, set(), False),
        ("plural_collection",
         "PUT /policies/POL-2026-77 HTTP/1.1",
         {"POL-2026-77"}, set(), False),
        ("placeholder_safe",
         '{"password": "[CREDENTIAL_1]"}',
         set(), {"[CREDENTIAL_1]"}, False),
        ("prose_untouched",
         "The author sorted the list by orderBy and checked the monkey cage.",
         set(), {"orderBy"}, True),

        # ── Part C ───────────────────────────────────────────
        # C1 — name is classified by what qualifies it
        ("c1_actuator_env",
         '{"name":"systemEnvironment","properties":{}}',
         set(), {"systemEnvironment"}, False),
        ("c1_structural_names",
         '{"fileName":"export.csv","typeName":"Order Status",'
         '"hostName":"web01.internal","eventName":"UserCreated"}',
         set(), {"export.csv", "web01.internal", "UserCreated"}, False),
        ("c1_definite_compound",
         '{"firstName":"Jan","familyName":"Novak"}',
         {"Jan", "Novak"}, set(), False),
        ("c1_login_name",
         '{"userName":"jnovak"}', {"jnovak"}, set(), False),
        ("c1_person_context_gated",
         '{"customerName":"Jan Novak","customerName2":"ACTIVE"}',
         {"Jan Novak"}, {"ACTIVE"}, False),
        ("c1_org_context",
         '{"companyName":"Acme Manufacturing"}',
         {"Acme Manufacturing"}, set(), False),
        ("c1_bare_name_person",
         '{"name":"Petr Svoboda"}', {"Petr Svoboda"}, set(), False),
        ("c1_particle_name",
         '{"name":"Ludwig van Beethoven"}', {"Ludwig van Beethoven"}, set(), False),

        # C2 — age is gone from the lexicon; HSTS survives
        ("c2_hsts",
         "Strict-Transport-Security: max-age=63072000; includeSubDomains",
         set(), {"63072000"}, False),

        # C3 — temporal values are never masked, whatever the key
        ("c3_last_login",
         '{"lastLogin":"2026-04-27T09:10:28Z","sessionId":"8fa2b1c9d4"}',
         {"8fa2b1c9d4"}, {"2026-04-27T09:10:28Z"}, False),
        ("c3_formats",
         '{"birthDate":"1984-02-19","cardExpiry":"04/2029",'
         '"passwordChangedAt":"1761554400","createdOn":"27.04.2026"}',
         set(), {"1984-02-19", "04/2029", "1761554400", "27.04.2026"}, False),
        ("c3_epoch_not_over_applied",
         '{"memberNumber":"1761554400"}', {"1761554400"}, set(), False),

        # C4 — auth schemes and low-entropy fill
        ("c4_scheme_alone",
         'authorization: Bearer\nx-api-key: ****************',
         set(), {"Bearer", "****************"}, False),
        ("c4_scheme_with_token",
         "authorization: Bearer eyJhbGciOiJIUzI1NiJ9.aa.bb",
         {"Bearer eyJhbGciOiJIUzI1NiJ9.aa.bb"}, set(), False),

        # C5a — prose that happens to look like key: value
        ("c5_prose_line",
         "Session: the tester confirmed the flow works end to end after "
         "the third attempt on staging\n",
         set(), set(), False),
        ("c5_long_header_exempt",
         "Cookie: a=1; b=2; c=3; d=4; e=5; f=6; g=7; h=8; i=9; j=10; k=11",
         {"a=1; b=2; c=3; d=4; e=5; f=6; g=7; h=8; i=9; j=10; k=11"},
         set(), False),

        # ── Lexicon expansion ────────────────────────────────
        # Metadata veto (A) — value describes a secret, not the secret
        ("meta_session_timeout",
         '{"sessionTimeout":1800,"sessionToken":"a1b2c3d4e5f6g7h8"}',
         {"a1b2c3d4e5f6g7h8"}, {"1800"}, False),
        ("meta_hash_algorithm",
         '{"hashAlgorithm":"SHA-256","passwordPolicy":"strong"}',
         set(), {"SHA-256", "strong"}, False),
        ("meta_auth_method",
         '{"authMethod":"oauth2","certFormat":"PEM"}',
         set(), {"oauth2", "PEM"}, False),
        ("meta_password_min_length",
         '{"passwordMinLength":12}', set(), {"12"}, False),
        # Metadata veto (B) — counting prefix on a countable-unit secret
        ("meta_llm_token_counts",
         '{"max_tokens":4096,"prompt_tokens":812,"completion_tokens":260}',
         set(), {"4096", "812", "260"}, False),
        # veto must NOT swallow a real credential whose key ends normally
        ("meta_real_token_survives",
         '{"refreshToken":"rt_9f8e7d6c5b4a3210"}',
         {"rt_9f8e7d6c5b4a3210"}, set(), False),

        # region relocation — no longer PII
        ("region_not_pii",
         '{"region":"eu-central-1","awsRegion":"us-east-1"}',
         set(), {"eu-central-1", "us-east-1"}, False),

        # New SENSITIVE_HEADERS — whole value masked
        ("hdr_cf_connecting_ip",
         "CF-Connecting-IP: 203.0.113.7", {"203.0.113.7"}, set(), False),
        ("hdr_x_api_key",
         "X-Api-Key: sk_" "live_9f8e7d6c5b4a", {"sk_" "live_9f8e7d6c5b4a"},
         set(), False),

        # New ENTITY_TOKENS — fire only with an id suffix
        ("entity_beneficiary_id",
         '{"beneficiaryId":"BEN-77213"}', {"BEN-77213"}, set(), False),
        ("entity_charge_ref",
         "chargeRef=ch_3Mk8vX2eZvKYlo", {"ch_3Mk8vX2eZvKYlo"},
         set(), False),
        ("entity_payslip_bare_untouched",
         '{"payslip":"generated"}', set(), {"generated"}, False),

        # New PII_TOKENS
        ("pii_surname",
         '{"surname":"Novakova"}', {"Novakova"}, set(), False),
        ("pii_imei",
         '{"imei":"356938035643809"}', {"356938035643809"}, set(), False),
        ("pii_mrn",
         "mrn=MRN00483921", {"MRN00483921"}, set(), False),
        ("pii_vin",
         '{"vin":"1HGCM82633A004352"}', {"1HGCM82633A004352"}, set(), False),

        # New SECRET_TOKENS
        ("secret_authentication",
         "authenticationToken: aB3dEf6hJk9mNp2q", {"aB3dEf6hJk9mNp2q"},
         set(), False),
        ("secret_passcode",
         '{"passcode":"918273"}', {"918273"}, set(), False),
        ("secret_mnemonic",
         '{"mnemonic":"witch collapse practice feed shame open"}',
         {"witch collapse practice feed shame open"}, set(), False),

        # ── Rule 1b — key disambiguation ─────────────────────
        ("key_secret_qualifier",
         '{"apiKey":"sk_' 'live_9f8e7d6c5b4a3210","encryptionKey":"aB3dEf6hJk9mNp2q"}',
         {"sk_" "live_9f8e7d6c5b4a3210", "aB3dEf6hJk9mNp2q"}, set(), False),
        ("key_private",
         '{"privateKey":"MIIEvQIBADANBgkqhkiG9w0"}',
         {"MIIEvQIBADANBgkqhkiG9w0"}, set(), False),
        ("key_structural_untouched",
         '{"partitionKey":"tenant-4471","sortKey":"2026#order#88",'
         '"cacheKey":"a1b2c3d4e5f6a7b8"}',
         set(), {"tenant-4471", "2026#order#88", "a1b2c3d4e5f6a7b8"}, False),
        ("key_routing_collision",
         '{"routingKey":"orders.eu.created"}',
         set(), {"orders.eu.created"}, False),
        ("key_bare_high_entropy",
         '{"key":"aB3dEf6hJk9mNp2qXyZ01"}',
         {"aB3dEf6hJk9mNp2qXyZ01"}, set(), False),
        ("key_bare_structural_word",
         '{"key":"user_email"}', set(), {"user_email"}, False),
        ("key_bare_s3_path",
         '{"key":"exports/2026/summary.json"}',
         set(), {"exports/2026/summary.json"}, False),
        # real bank routing number (no key token) must still mask
        ("routing_number_pii",
         '{"routingNumber":"021000021"}', {"021000021"}, set(), False),
    ]

    failures = 0
    for name, text, must, must_not, loose in CASES:
        got = run(text, loose=loose)
        missing = {m for m in must if not any(m == g or m in g for g in got)}
        leaked = {m for m in must_not if any(m == g for g in got)}
        if missing or leaked:
            failures += 1
            print(f"FAIL {name}")
            if missing:
                print(f"     missing: {sorted(missing)}")
            if leaked:
                print(f"     wrongly masked: {sorted(leaked)}")
            print(f"     got: {sorted(got)}")
        else:
            print(f"ok   {name}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    sys.exit(1 if failures else 0)
