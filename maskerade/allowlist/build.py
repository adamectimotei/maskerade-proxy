#!/usr/bin/env python3
"""Build the technical-vocabulary allow-list from published sources.

WHAT THIS IS
------------
A reproducible generator for tests/allowlist/allowlist.yaml — the suppression
list that stops the NER-slot detectors (GLiNER ORG, keyval) from masking
technical vocabulary that identifies nobody: technology/product names, protocol
parameter names, and standard-format acronyms.

WHY A GENERATOR (not a hand-written list)
-----------------------------------------
Every term must trace to a PUBLISHED, INDUSTRY-WIDE vocabulary, never to the
test corpus. A term earns its place because it appears across the whole software
industry (nginx runs a third of the web; client_secret is an RFC-defined
parameter name), NOT because it appears in some prompt. This script pulls whole
published registries so the provenance is auditable and the list generalises to
any application's traffic — re-run it and you get the same list from the same
sources, with zero corpus input.

SOURCES  (group -> license -> what it fixes)
--------------------------------------------
  cncf_projects        CNCF Landscape (Apache-2.0)          nginx, redis, envoy,
                       github.com/cncf/landscape             kubernetes, grpc ...
                                                             -> GLiNER ORG FPs

  oauth_oidc_params    IANA OAuth Parameters registry +     client_secret,
                       RFC 6749/7591/7662/8414, OIDC Core    refresh_token,
                       (public domain)                       grant_type, id_token
                                                             -> keyval CREDENTIAL FPs

  oauth_enum_values    RFC 6749 §4 / RFC 7591 (public        authorization_code,
                       domain) — distinctive/compound        client_credentials,
                       grant & auth-method values only       client_secret_basic
                                                             -> keyval CREDENTIAL FPs

  http_vocabulary      IANA HTTP registries + RFC 9110       content-type, accept,
                       (public domain)                       GET/POST, http/https
                                                             -> GLiNER ORG FPs

  format_standards     IETF / W3C / ECMA / ISO standard      json, xml, graphql,
                       names (public-domain facts, each      grpc, jwt, saml, iban
                       the proper name of a spec)            -> GLiNER ORG FPs

  user_agent_tokens    RFC 9110 §10.1.5 product tokens +     Mozilla, Gecko, Safari,
                       MDN User-Agent reference, plus         AppleWebKit, Windows NT,
                       ua-parser/uap-core family names        curl, Googlebot ...
                       (Apache-2.0)                           -> NER ORG/PERSON FPs on
                                                              every User-Agent header

  cloud_vendors        Major cloud providers + flagship      oracle, azure, aws,
                       services, per vendor product          amazon s3, gcp ...
                       catalogs (facts)                      -> GLiNER ORG FPs

  wappalyzer  (OPT-IN, OFF by default)  webappanalyzer       thousands of web
              --with-wappalyzer         (GPL-3.0!)           technologies
                       *** GPL-3.0 — do NOT vendor into a proprietary repo
                           unless you accept the copyleft implications. ***

SAFETY FILTERS (generic, applied to every fetched term)
-------------------------------------------------------
An allow-list is a suppression mechanism: a bad entry causes a LEAK by
un-masking something that should have been masked. So:
  * whole-token only — the runtime matcher compares the ENTIRE entity text to a
    term (case-insensitive). Allow-listing "oracle" can never un-mask
    "oracle-client-db-prod-4471" because that string != "oracle".
  * min length 3 — matches the detector stages' own floors.
  * drop English stopwords — a short function word ("now", "all", "for") is a
    dangerous allow-list entry and never a technology identifier.
  * ambiguous bare OAuth values ("password", "none", "basic", "code", "token")
    are NOT allow-listed: they can equal a real weak secret, and dropping them
    would cost recall — the one thing that must not happen.

USAGE
-----
    python3 tests/allowlist/build.py                 # fetch + write allowlist.yaml
    python3 tests/allowlist/build.py --cncf-file X   # use a local landscape.yml
    python3 tests/allowlist/build.py --offline       # skip all network, constants only
    python3 tests/allowlist/build.py --with-wappalyzer   # add GPL web-tech list

The committed allowlist.yaml is the offline, auditable artifact; this script is
its provenance. Nothing here is imported by the pipeline or the benchmark at
runtime — they read the YAML.
"""
import argparse
import datetime as _dt
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "allowlist.yaml")

CNCF_URL = "https://raw.githubusercontent.com/cncf/landscape/master/landscape.yml"
# ua-parser/uap-core — Apache-2.0. Browser and OS family names, the long tail
# behind the literal tokens in USER_AGENT_TOKENS below.
UAP_CORE_URL = "https://raw.githubusercontent.com/ua-parser/uap-core/master/regexes.yaml"
IANA_OAUTH_CSV = "https://www.iana.org/assignments/oauth-parameters/oauth-parameters-1.csv"
# webappanalyzer is GPL-3.0. Only fetched with --with-wappalyzer.
WAPP_BASE = "https://raw.githubusercontent.com/enthec/webappanalyzer/main/src/technologies/"


# ── generic safety filters ────────────────────────────────────────────────
MIN_LEN = 3

# A compact, standard English function-word / high-frequency stoplist. These are
# never technology identifiers and are the dangerous class for an allow-list
# (they could coincide with real data). Linguistic constant, not corpus-derived.
STOPWORDS = frozenset("""
a an the and or but if then else for while do done to of in on at by with from
into onto out up down over under again once here there all any both each few
more most other some such no nor not only own same so than too very can will
just should now new old get set put use used using make made new next last
this that these those it its is are was were be been being have has had he she
they we you your our their his her them us me my mine ok yes true false null none
one two three four five six seven eight nine ten first second third
""".split())

# Bare, ambiguous words that must NEVER be allow-listed as a value: each can be
# the LEFT side of a key/value pair (a field name, which is safe vocabulary) but
# as a bare VALUE could be a real weak secret or sensitive datum. Allow-listing
# them would cost recall — the one thing that must not happen. This is the union
# of the OAuth ambiguous values and the "do-NOT-value-allow-list" set documented
# across gitleaks / Datadog / Presidio. Compound forms (client_secret,
# public_key, secret_name) are NOT here — only the bare tokens.
DENY_VALUES = frozenset({
    "password", "passwd", "pwd", "passphrase", "passcode",
    "secret", "token", "key", "apikey", "credential", "credentials", "creds",
    "auth", "authorization", "authentication", "bearer", "basic",
    "session", "sessionid", "cookie", "code", "otp", "pin", "salt",
    "signature", "sign", "nonce", "seed", "mnemonic", "cipher", "hash",
    "none", "null", "nil", "client", "assertion", "implicit", "realm",
    "opaque", "private", "access",
})
# Back-compat alias — earlier call sites referenced this name.
OAUTH_VALUE_BLOCK = DENY_VALUES


# ── embedded, cited constants (public-domain facts) ───────────────────────
# OAuth 2.0 / OIDC / token-introspection / AS-metadata PARAMETER NAMES.
# Sources: RFC 6749, RFC 7591 (dynamic client registration), RFC 7662
# (introspection), RFC 8414 (AS metadata), OpenID Connect Core 1.0 / Discovery.
# These are field NAMES; as a *value* they are never a client secret, so
# suppressing an entity whose whole text is one of them is safe.
OAUTH_OIDC_PARAMS = [
    # core protocol params (RFC 6749)
    "client_id", "client_secret", "grant_type", "response_type", "redirect_uri",
    "scope", "state", "code", "access_token", "refresh_token", "token_type",
    "expires_in", "code_verifier", "code_challenge", "code_challenge_method",
    "nonce", "response_mode", "error", "error_description", "error_uri",
    # OIDC (Core / Discovery)
    "id_token", "id_token_hint", "userinfo", "claims", "acr_values", "prompt",
    "max_age", "login_hint", "display", "ui_locales", "claims_locales",
    "request_uri", "registration", "subject_type", "issuer", "jwks_uri",
    "authorization_endpoint", "token_endpoint", "userinfo_endpoint",
    "end_session_endpoint", "token_endpoint_auth_method",
    "token_endpoint_auth_signing_alg", "grant_types", "response_types",
    "application_type", "client_name", "client_uri", "logo_uri", "contacts",
    "tos_uri", "policy_uri", "sector_identifier_uri",
    # introspection / revocation / PKCE / device / PAR / DPoP / mTLS
    "active", "assertion", "client_assertion", "client_assertion_type",
    "device_code", "user_code", "verification_uri", "verification_uri_complete",
    "request", "cnf", "dpop", "dpop_jkt", "aud", "iss", "sub", "exp", "iat",
]

# Distinctive / compound OAuth enum VALUES (RFC 6749 §4, RFC 7591). Bare
# ambiguous words are excluded via OAUTH_VALUE_BLOCK above.
OAUTH_ENUM_VALUES = [
    "authorization_code", "client_credentials", "refresh_token",
    "urn:ietf:params:oauth:grant-type:jwt-bearer",
    "urn:ietf:params:oauth:grant-type:saml2-bearer",
    "urn:ietf:params:oauth:grant-type:device_code",
    "urn:ietf:params:oauth:grant-type:token-exchange",
    "client_secret_basic", "client_secret_post", "client_secret_jwt",
    "private_key_jwt", "tls_client_auth", "self_signed_tls_client_auth",
    "id_token token", "code id_token", "code token", "code id_token token",
]

# HTTP method names, common request/response header field names, and scheme
# words. Sources: IANA HTTP Method / Field Name registries, RFC 9110.
HTTP_VOCABULARY = [
    "http", "https",
    "get", "post", "put", "patch", "delete", "head", "options", "trace", "connect",
    "accept", "accept-encoding", "accept-language", "accept-ranges",
    "authorization", "cache-control", "content-type", "content-length",
    "content-encoding", "content-disposition", "content-language", "cookie",
    "set-cookie", "etag", "expires", "host", "location", "origin", "referer",
    "user-agent", "www-authenticate", "x-forwarded-for", "x-forwarded-proto",
    "x-frame-options", "x-content-type-options", "strict-transport-security",
    "transfer-encoding", "vary", "upgrade", "connection", "keep-alive",
    "if-match", "if-none-match", "if-modified-since", "last-modified",
    "access-control-allow-origin", "content-security-policy",
]

# Proper names of widely-used specs / formats / protocols. Each is the name of a
# standard published by IETF / W3C / ECMA / ISO / a foundation — public-domain
# facts, and exactly the tokens a zero-shot NER mislabels as ORGANIZATION.
FORMAT_STANDARDS = [
    # data / markup / query formats
    "json", "json5", "jsonl", "ndjson", "xml", "yaml", "toml", "csv", "tsv",
    "html", "xhtml", "css", "svg", "protobuf", "avro", "parquet", "cbor",
    "graphql", "sparql", "jsonpath", "xpath", "jmespath",
    # API / RPC / messaging styles & protocols
    "rest", "restful", "soap", "grpc", "rpc", "webhook", "websocket",
    "graphql", "openapi", "swagger", "asyncapi", "amqp", "mqtt", "stomp",
    "sse", "webrtc", "jsonrpc", "xmlrpc",
    # transport / network protocols
    "tcp", "udp", "ip", "icmp", "tls", "ssl", "mtls", "dns", "dhcp", "ntp",
    "ssh", "ftp", "sftp", "ftps", "smtp", "imap", "pop3", "ldap", "ldaps",
    "snmp", "kerberos", "radius", "saml", "scim", "spnego", "gssapi",
    # tokens / crypto / auth standards
    "oauth", "oidc", "openid", "jwt", "jws", "jwe", "jwk", "jwks", "jwa",
    "pkce", "hmac", "rsa", "ecdsa", "ed25519", "aes", "sha256", "sha512",
    "bcrypt", "scrypt", "argon2", "pbkdf2", "pgp", "gpg", "x509", "pkcs",
    # identifiers / financial / standard codes (ISO / ITU) — the WORD, not a value
    "iban", "bic", "swift", "isin", "lei", "uuid", "guid", "ulid", "urn",
    "uri", "url", "url", "iso8601", "rfc3339", "utc", "gtin", "ean", "upc",
]

# Major cloud providers and their flagship service/product names. Facts drawn
# from each vendor's public product catalog; these are the brand names a
# zero-shot NER most confidently mislabels as ORGANIZATION. Deliberately small
# and famous — extend per engagement rather than trying to be exhaustive.
CLOUD_VENDORS = [
    # AWS
    "aws", "amazon web services", "amazon s3", "s3", "ec2", "lambda",
    "cloudfront", "dynamodb", "rds", "route 53", "route53", "iam", "cognito",
    "eks", "ecs", "fargate", "cloudwatch", "sqs", "sns", "kms", "vpc",
    # Azure / Microsoft
    "azure", "microsoft azure", "azure ad", "entra", "entra id",
    "azure active directory", "blob storage", "cosmos db", "cosmosdb",
    # GCP / Google
    "gcp", "google cloud", "google cloud platform", "bigquery", "gke",
    "cloud run", "cloud storage", "firebase", "firestore", "pubsub",
    # other major vendors / brands commonly seen in traffic
    "oracle", "oracle cloud", "oci", "ibm", "ibm cloud", "sap", "salesforce",
    "vmware", "cloudflare", "akamai", "fastly", "digitalocean", "heroku",
    "netlify", "vercel", "snowflake", "databricks", "mongodb atlas",
    # ubiquitous web servers / runtimes / frameworks / databases not in CNCF
    "nginx", "apache", "apache http server", "httpd", "iis", "tomcat",
    "jetty", "gunicorn", "uvicorn", "php", "wordpress", "drupal", "joomla",
    "laravel", "symfony", "django", "flask", "fastapi", "rails",
    "spring", "spring boot", "express", "nestjs", "react", "angular", "vue",
    "svelte", "nextjs", "next.js", "nuxt", "dotnet", ".net", "asp.net",
    "node", "nodejs", "node.js", "deno", "bun",
    "mysql", "mariadb", "postgresql", "postgres", "oracle database", "mssql",
    "sql server", "sqlite", "mongodb", "cassandra", "elasticsearch",
    "opensearch", "rabbitmq", "kafka", "memcached",
]


# User-Agent product tokens, platform comments, HTTP client names and crawlers.
#
# WHY THIS GROUP EXISTS
# Every HTTP request carries a User-Agent, and RFC 9110 §10.1.5 defines its
# value as product tokens plus comments — i.e. software identifiers. A zero-shot
# NER has no way to know that, so it reads "Mozilla", "Safari" and "Gecko" as
# company names and masks them on literally every request in an engagement.
# Measured on one 36-app corpus run, "Mozilla" alone was the single most
# repeated over-mask at 62 hits. Nothing here is corpus-derived: the tokens are
# the ones the UA grammar and MDN's User-Agent reference document, and they are
# the same on any application's traffic.
#
# SAFETY: whole-token matching means allow-listing "chrome" cannot un-mask
# "chrome-admin-prod-4471". The residual risk is a client legitimately named
# after a browser or an OS — handle that with a custom rule or an `exclude:`
# entry in allowlist_overrides.yaml, exactly as for "Oracle" in cloud_vendors.
#
# Sources: RFC 9110 §10.1.5 (User-Agent grammar), MDN Web Docs "User-Agent"
# and "Firefox/Chrome user agent string reference" (CC-BY-SA facts — the token
# strings themselves are not copyrightable), vendor UA documentation, and the
# published UA strings of the major HTTP client libraries and crawlers.
USER_AGENT_TOKENS = [
    # ── the classic product tokens every browser still sends ──
    "mozilla", "gecko", "like gecko", "applewebkit", "webkit", "khtml",
    "blink", "trident", "presto", "edgehtml", "compatible", "version",
    # ── browser product names ──
    "chrome", "chromium", "safari", "mobile safari", "firefox", "edge", "edg",
    "edga", "edgios", "opera", "opr", "vivaldi", "brave", "seamonkey",
    "palemoon", "waterfox", "midori", "konqueror", "epiphany", "falkon",
    "samsungbrowser", "ucbrowser", "yabrowser", "qqbrowser", "maxthon",
    "crios", "fxios", "duckduckgo", "headlesschrome", "phantomjs",
    "electron", "qtwebengine", "wkwebview", "cfnetwork", "msie",
    # ── platform / architecture comment tokens ──
    "windows", "windows nt", "win64", "win32", "wow64", "x11", "macintosh",
    "mac os", "mac os x", "intel mac os x", "macos", "darwin", "linux",
    "ubuntu", "debian", "fedora", "gentoo", "freebsd", "openbsd", "solaris",
    "cros", "chrome os", "android", "iphone", "ipad", "ipod", "ios", "ipados",
    "watchos", "tvos", "webos", "kaios", "symbian", "blackberry",
    "windows phone", "windows mobile", "mobile", "tablet",
    "x86_64", "x64", "x32", "amd64", "arm64", "aarch64", "armv7l", "i686",
    # ── HTTP client libraries: the User-Agent of most API traffic ──
    "curl", "wget", "libwww-perl", "python-requests", "python-urllib",
    "aiohttp", "httpx", "go-http-client", "okhttp", "axios", "node-fetch",
    "undici", "apache-httpclient", "jakarta commons-httpclient", "guzzlehttp",
    "restsharp", "postmanruntime", "insomnia", "httpie", "powershell",
    "java", "dart", "reactornetty", "lwp-request", "symfony httpclient",
    # ── crawlers and link-preview fetchers seen in any public app's logs ──
    "googlebot", "bingbot", "applebot", "duckduckbot", "yandexbot",
    "baiduspider", "ahrefsbot", "semrushbot", "mj12bot", "dotbot",
    "facebookexternalhit", "twitterbot", "linkedinbot", "slackbot",
    "discordbot", "telegrambot", "whatsapp", "pingdom", "uptimerobot",
    "gtmetrix", "lighthouse", "chrome-lighthouse",
]


# Common API / JSON field-name conventions. Sources: JSON:API v1.1 (jsonapi.org),
# OpenAPI 3.x (spec.openapis.org), GraphQL introspection (graphql-spec §4),
# HAL/Hydra hypermedia, RFC 9457 Problem Details, and ubiquitous pagination /
# filtering / envelope conventions. These are structural member NAMES; as a bare
# value they are never client data, so suppressing an entity whose whole text is
# one of them is safe (a real value under `sort` is "createdAt", not "sort").
API_CONVENTIONS = [
    # JSON:API members + query params
    "data", "attributes", "relationships", "links", "meta", "included", "type",
    "lid", "self", "related", "first", "last", "prev", "next", "href", "rel",
    "describedby", "errors", "jsonapi", "include", "fields",
    # OpenAPI / Swagger structural keywords
    "openapi", "swagger", "info", "paths", "components", "servers", "schemas",
    "parameters", "requestbody", "responses", "security", "tags", "operationid",
    "summary", "description", "required", "properties", "schema", "content",
    "example", "examples", "securityschemes", "definitions", "produces", "consumes",
    # GraphQL introspection (reserved, never user data)
    "__typename", "__schema", "__type", "querytype", "mutationtype",
    "subscriptiontype", "directives", "isdeprecated", "deprecationreason",
    "oftype", "possibletypes", "enumvalues", "inputfields", "interfaces", "kind",
    # HAL / Hydra hypermedia
    "_links", "_embedded", "curies", "templated", "hydra",
    # pagination / filtering / sorting conventions
    "page", "pagesize", "page_size", "per_page", "limit", "offset", "cursor",
    "after", "before", "sort", "order", "order_by", "sort_by", "query", "search",
    "filter", "expand", "size", "count", "total", "total_count", "total_pages",
    "has_more", "next_cursor", "prev_cursor",
    # REST envelope / Problem Details (RFC 9457)
    "status", "error", "message", "detail", "title", "instance", "result",
    "results", "items", "payload", "success", "timestamp",
]

# Common enum / status / lifecycle VALUES. Universal structural constants that
# identify no client. Deliberately EXCLUDES two recall-unsafe classes:
#   * role words (admin/user/root/guest) — a pentester may want those visible;
#   * deployment-environment names (prod/dev/staging/test/...) — these are
#     embedded as COMPONENTS of sensitive resource identifiers (e.g. the host
#     "psql-payments-prod-rw...") exactly like cloud regions, so allow-listing
#     them risks suppressing a host-fragment detection and losing recall.
# Only pure status/lifecycle words and HTTP reason phrases remain — these do not
# appear as identifier components. Sources: HTTP status reason phrases (RFC 9110),
# common state-machine vocabulary.
ENUM_STATUS_VALUES = [
    "true", "false",
    "active", "inactive", "enabled", "disabled", "pending", "approved",
    "rejected", "success", "failure", "failed", "error", "warning", "info",
    "debug", "trace", "draft", "published", "archived", "deleted", "open",
    "closed", "read", "unread", "online", "offline", "available", "unavailable",
    "completed", "complete", "incomplete", "cancelled", "canceled", "expired",
    "valid", "invalid", "verified", "unverified", "processing", "queued",
    "running", "stopped", "started", "created", "updated", "high", "medium",
    "low", "critical", "asc", "desc", "ascending", "descending",
    # HTTP status reason phrases (RFC 9110)
    "accepted", "moved permanently", "found", "not modified",
    "bad request", "unauthorized", "forbidden", "not found", "conflict",
    "unprocessable entity", "too many requests", "internal server error",
    "bad gateway", "service unavailable", "gateway timeout",
]

# NOTE: a cloud-region group was intentionally REMOVED. Region identifiers
# (us-east-1, eu-central-1, ...) are embedded as components of sensitive cloud
# resource hostnames/ARNs (AWS RDS/ElastiCache/S3 endpoints, ARNs, and the
# Azure/GCP equivalents) and are tagged LOCATION by NER, so allow-listing them
# suppresses host-fragment detections and costs recall — measured on the
# benchmark (2 RDS endpoints lost). Regions as public infra are low-value to
# suppress standalone, so the class is excluded. A pentester who needs a bare
# region left unmasked can add it per-engagement via allowlist_overrides.yaml.

# JOSE / JWT structural header params, non-PII registered claims, and signing/
# encryption ALGORITHM names. Sources: RFC 7515/7517/7518/7519, IANA JWT
# registry. Excludes subject/audience/PII claims (sub, aud, email, name, ...).
JOSE_JWT = [
    # header params
    "alg", "typ", "cty", "kid", "jku", "jwk", "x5c", "x5t", "x5u", "crit",
    "enc", "zip", "b64",
    # non-PII registered claims
    "iss", "exp", "nbf", "iat", "jti", "azp", "amr", "acr", "sid", "scope",
    "at_hash", "c_hash", "nonce_supported",
    # signature / encryption algorithm names
    "rs256", "rs384", "rs512", "es256", "es384", "es512", "ps256", "ps384",
    "ps512", "hs256", "hs384", "hs512", "eddsa", "es256k",
    "rsa-oaep", "rsa-oaep-256", "a128kw", "a192kw", "a256kw", "dir",
    "ecdh-es", "a128gcm", "a192gcm", "a256gcm", "a128cbc-hs256", "a256cbc-hs512",
    # OIDC scopes (non-PII)
    "openid", "offline_access",
]

# Published false-positive vocabulary from secret scanners — words that CONTAIN
# key/api/token/secret but are never secrets. Source: gitleaks generic-api-key
# allow-list + DefaultStopWords (github.com/gitleaks/gitleaks, MIT). Compound
# forms are safe whole tokens; the bare ambiguous words are already in DENY.
DLP_FALSE_POSITIVES = [
    "public_key", "public_token", "publickeytoken", "api_version",
    "client_version", "map_key", "key_word", "secret_name", "client_name",
    "key_name", "keystone", "keyboard", "keying", "keychain", "monkey",
    "donkey", "rapid", "capital", "fake", "dummy", "example", "placeholder",
    "changeme", "test_key", "sample",
]

# Additional HTTP authentication scheme names (IANA HTTP Authentication Scheme
# Registry / RFC 7235). Basic/Bearer are intentionally omitted — they are in
# DENY_VALUES because the credential that follows the scheme word must be masked.
HTTP_AUTH_SCHEMES = [
    "digest", "negotiate", "ntlm", "dpop", "hoba", "mutual", "vapid",
    "concealed", "gnap", "scram-sha-1", "scram-sha-256", "aws4-hmac-sha256",
]


# ── fetch helpers ──────────────────────────────────────────────────────────
def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "kefir3000-allowlist-build"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def fetch_cncf(cncf_file=None, offline=False):
    """CNCF Landscape project names (Apache-2.0)."""
    try:
        import yaml
    except ImportError:
        print("  ! pyyaml not installed — skipping CNCF group", file=sys.stderr)
        return []
    text = None
    if cncf_file and os.path.exists(cncf_file):
        text = open(cncf_file, encoding="utf-8").read()
    elif not offline:
        try:
            text = _get(CNCF_URL)
        except Exception as e:
            print(f"  ! CNCF fetch failed ({e}) — skipping group", file=sys.stderr)
            return []
    if not text:
        return []
    data = yaml.safe_load(text)
    names = set()
    for cat in (data.get("landscape") or []):
        for sub in (cat.get("subcategories") or []):
            for item in (sub.get("items") or []):
                n = item.get("name")
                if n:
                    names.add(n.strip())
    return sorted(names)


def fetch_iana_oauth(offline=False):
    """Extend OAuth param names from the live IANA CSV when reachable."""
    if offline:
        return []
    try:
        text = _get(IANA_OAUTH_CSV)
    except Exception as e:
        print(f"  · IANA OAuth CSV not reachable ({e}) — using RFC constants only",
              file=sys.stderr)
        return []
    out = []
    for line in text.splitlines()[1:]:
        # CSV: Name,Parameter Usage Location,Change Controller,Reference
        cell = line.split(",", 1)[0].strip().strip('"')
        if cell and " " not in cell:
            out.append(cell)
    return out


def fetch_wappalyzer(offline=False):
    """OPT-IN. webappanalyzer technology names. GPL-3.0 — see module header."""
    if offline:
        return []
    print("  *** --with-wappalyzer: source is GPL-3.0. Do not vendor into a "
          "proprietary repo unless you accept copyleft. ***", file=sys.stderr)
    names = set()
    letters = list("abcdefghijklmnopqrstuvwxyz") + ["_"]
    for ch in letters:
        try:
            raw = _get(WAPP_BASE + f"{ch}.json", timeout=30)
            for k in json.loads(raw).keys():
                names.add(k.strip())
        except Exception as e:
            print(f"  ! wappalyzer {ch}.json failed: {e}", file=sys.stderr)
    return sorted(names)


# ── normalisation + safety filtering ───────────────────────────────────────
_SEP = re.compile(r"[ _\-.:]")


def fetch_uap_core(offline=False):
    """Browser and OS family names from ua-parser/uap-core (Apache-2.0).

    The long tail behind USER_AGENT_TOKENS: Amazon Silk, Basilisk, Aloha
    Browser, KaiOS and ~200 more that a curated constant will never keep up
    with. Only `family_replacement` (browsers) and `os_replacement` (operating
    systems) are read — device_parsers is deliberately skipped, because it is
    hundreds of handset brand/model strings whose collision surface with real
    data is far larger than their false-positive value.

    Replacement templates ("Apple $1 App", "Firefox ($1)") are dropped: they
    are regex substitution patterns, not names, and the literal "$1" would be
    a nonsense allow-list entry.

    Never fatal — an unreachable source leaves the curated constant in place.
    """
    if offline:
        return []
    try:
        import yaml
        raw = _get(UAP_CORE_URL)
        doc = yaml.safe_load(raw) or {}
    except Exception as e:
        print(f"  [uap-core] fetch failed ({e}) — using embedded tokens only")
        return []

    names = set()
    for parser in doc.get("user_agent_parsers", []):
        value = parser.get("family_replacement")
        if value:
            names.add(value)
    for parser in doc.get("os_parsers", []):
        value = parser.get("os_replacement")
        if value:
            names.add(value)

    out = [n for n in names if "$" not in n and _distinctive_family(n)]
    print(f"  [uap-core] {len(out)} browser/OS family names "
          f"({len(names) - len(out)} dropped as non-distinctive)")
    return out


def _distinctive_family(name):
    """Keep a FETCHED family name only if it cannot pass for an ordinary word.

    The curated USER_AGENT_TOKENS list carries every short bare token that
    actually matters (chrome, safari, curl, gecko, opr, edg). What the fetched
    long tail adds on top is niche browsers, and among them are short bare
    words that are also surnames or common nouns — Norton, Odin, Iron, Quark,
    Barca, Line, Spider. Allow-listing those buys almost no precision on real
    traffic and hands the suppression pass a way to un-mask a person's name.

    So a fetched term earns its place by being distinctive: it carries a
    separator or digit (Amazon Silk, 360 Secure Browser, Chrome Mobile WebView)
    or it is long enough that coincidence is implausible. Same reasoning as
    require_sep on the OAuth parameter group.
    """
    return len(name) >= 7 or bool(re.search(r"[\s._\-]|\d", name))


def _clean(terms, block=frozenset(), require_sep=False):
    """Lowercase, strip, dedupe, and drop unsafe entries. Returns sorted list.

    require_sep: keep only compound/identifier terms that contain a separator
    (_ - . : space). Used for the OAuth parameter group, where the real false
    positives are all compound (client_secret, refresh_token, id_token,
    grant_type) and the bare single words (code, state, scope, error, active,
    nonce, claims, aud ...) are common English words that would be a recall
    risk and were never the source of a false positive anyway.
    """
    seen = set()
    out = []
    for t in terms:
        if not isinstance(t, str):
            continue
        t = t.strip().lower()
        if len(t) < MIN_LEN:
            continue
        if t in STOPWORDS or t in DENY_VALUES or t in block:
            continue
        if require_sep and not _SEP.search(t):
            continue
        # a single alphabetic token that is a stopword-like function word is
        # already dropped; keep multi-word and punctuated tech names as-is.
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return sorted(out)


def build(args):
    groups = {}

    # protocol vocabulary (public-domain facts + optional live IANA extension)
    params = list(OAUTH_OIDC_PARAMS) + fetch_iana_oauth(args.offline)
    groups["oauth_oidc_params"] = {
        "source": "IANA OAuth Parameters registry; RFC 6749/7591/7662/8414; "
                  "OpenID Connect Core 1.0 / Discovery 1.0 (public domain)",
        "note": "compound protocol parameter NAMES (separator-containing); never "
                "a client secret as a value. Bare common words (code/state/scope) "
                "dropped as recall-risk and non-causes of FPs.",
        "terms": _clean(params, require_sep=True),
    }
    groups["oauth_enum_values"] = {
        "source": "RFC 6749 §4 grant/response types; RFC 7591 token-endpoint "
                  "auth methods (public domain)",
        "note": "distinctive/compound enum values only; bare ambiguous words "
                "(password/none/basic/code/token) excluded to protect recall",
        "terms": _clean(OAUTH_ENUM_VALUES, block=OAUTH_VALUE_BLOCK),
    }
    groups["http_vocabulary"] = {
        "source": "IANA HTTP Method/Field Name/Auth Scheme registries; RFC 9110/"
                  "7235 (public domain)",
        "terms": _clean(HTTP_VOCABULARY + HTTP_AUTH_SCHEMES),
    }
    groups["api_conventions"] = {
        "source": "JSON:API v1.1; OpenAPI 3.x; GraphQL introspection (graphql-spec "
                  "§4); HAL/Hydra; RFC 9457; common pagination/envelope conventions",
        "note": "structural member/field NAMES; never client data as a value",
        "terms": _clean(API_CONVENTIONS),
    }
    groups["enum_status_values"] = {
        "source": "HTTP status reason phrases (RFC 9110); common state-machine "
                  "vocabulary",
        "note": "excludes role words and deployment-environment names "
                "(identifier-component recall risk)",
        "terms": _clean(ENUM_STATUS_VALUES),
    }
    groups["jose_jwt"] = {
        "source": "RFC 7515/7517/7518/7519; IANA JWT registry — header params, "
                  "non-PII claims, and algorithm names (excludes sub/aud/PII claims)",
        "terms": _clean(JOSE_JWT),
    }
    groups["dlp_false_positives"] = {
        "source": "gitleaks generic-api-key allow-list + DefaultStopWords "
                  "(github.com/gitleaks/gitleaks, MIT) — words containing "
                  "key/api/token that are never secrets",
        "terms": _clean(DLP_FALSE_POSITIVES),
    }
    groups["format_standards"] = {
        "source": "Proper names of IETF/W3C/ECMA/ISO standards & widely-adopted "
                  "specs (public-domain facts)",
        "terms": _clean(FORMAT_STANDARDS, block=OAUTH_VALUE_BLOCK),
    }
    groups["user_agent_tokens"] = {
        "source": "RFC 9110 §10.1.5 User-Agent grammar; MDN User-Agent reference; "
                  "published HTTP client / crawler UA strings; "
                  "ua-parser/uap-core family names (Apache-2.0)",
        "note": "software identifiers carried by every HTTP request. A UA value "
                "names a program, never a person or a client organisation, so "
                "suppressing an entity whose WHOLE text is one of these is safe. "
                "A client named after a browser needs an exclude: entry.",
        "terms": _clean(USER_AGENT_TOKENS + fetch_uap_core(args.offline)),
    }
    groups["cloud_vendors"] = {
        "source": "Major cloud providers + flagship services and ubiquitous web "
                  "servers/frameworks/databases, per vendor product catalogs (facts)",
        "terms": _clean(CLOUD_VENDORS),
    }

    # cloud-native project names (Apache-2.0)
    cncf = fetch_cncf(args.cncf_file, args.offline)
    groups["cncf_projects"] = {
        "source": "CNCF Landscape, github.com/cncf/landscape/landscape.yml "
                  "(Apache-2.0)",
        "terms": _clean(cncf),
    }

    total = sum(len(g["terms"]) for g in groups.values())
    doc = {
        "_comment": "GENERATED by tests/allowlist/build.py — do not edit by hand. "
                    "Every term traces to a published industry vocabulary, never "
                    "to the test corpus. Regenerate to update.",
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "matching": "whole-token, case-insensitive (entity text == term)",
        "total_terms": total,
        "groups": groups,
    }
    return doc


def build_wappalyzer(args):
    """Separate GPL-3.0 technology-name list. Written to its own file so it is
    never silently merged into the permissive allowlist.yaml. The loader picks
    up every *.yaml in the allowlist dir, so dropping this file in enables it
    for local TESTING; delete it before shipping a proprietary build.

    Only bare technology NAMES are extracted (facts) — not the fingerprint
    regexes or descriptions, which are the creative, copyrightable content.
    """
    wapp = fetch_wappalyzer(args.offline)
    return {
        "_comment": "GENERATED (GPL-3.0 source). Technology NAMES only, extracted "
                    "from enthec/webappanalyzer for local testing/measurement. "
                    "Do NOT vendor into a proprietary repo — delete before shipping.",
        "license": "GPL-3.0 (source: github.com/enthec/webappanalyzer)",
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "matching": "whole-token, case-insensitive (entity text == term)",
        "total_terms": len(_clean(wapp)),
        "groups": {
            "wappalyzer_technologies": {
                "source": "enthec/webappanalyzer src/technologies (GPL-3.0)",
                "license_warning": "GPL-3.0 — testing only; not for a proprietary build.",
                "terms": _clean(wapp),
            }
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Build the technical-vocabulary allow-list.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output YAML path")
    ap.add_argument("--cncf-file", help="use a local landscape.yml instead of fetching")
    ap.add_argument("--offline", action="store_true", help="no network; embedded constants only")
    ap.add_argument("--with-wappalyzer", action="store_true",
                    help="also write allowlist_wappalyzer.yaml (GPL-3.0 tech names; "
                         "testing only — the loader auto-merges it)")
    args = ap.parse_args()

    import yaml
    doc = build(args)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, width=100)

    print(f"  wrote {args.out}")
    print(f"  {doc['total_terms']} terms across {len(doc['groups'])} groups:")
    for name, g in doc["groups"].items():
        print(f"    {name:24} {len(g['terms']):5}")

    if args.with_wappalyzer:
        wdoc = build_wappalyzer(args)
        wpath = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                             "allowlist_wappalyzer.yaml")
        with open(wpath, "w", encoding="utf-8") as f:
            yaml.safe_dump(wdoc, f, sort_keys=False, allow_unicode=True, width=100)
        print(f"  wrote {wpath}  ({wdoc['total_terms']} GPL-3.0 tech names — testing only)")


if __name__ == "__main__":
    main()
