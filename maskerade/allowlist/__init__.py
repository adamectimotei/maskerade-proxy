"""Technical-vocabulary allow-list — post-detection suppression.

Pipeline component. Pipeline.__init__ loads it ONCE (the agent loop calls
run() several times per tool call, so a per-run load would re-read 10k terms
each time) and run() applies it after every detection stage, before span
reconciliation and masking. Toggled by stages.allowlist; the run(whitelist=)
parameter still overrides it per call (used by measurement scripts).

Every *.yaml in this directory is merged at load time, so an optional,
separately-licensed file such as ``allowlist_wappalyzer.yaml`` is picked up
automatically when present and simply absent otherwise. A missing directory or
missing pyyaml yields an empty AllowList, which is falsy (``__len__``) and
therefore disables suppression on its own — it never raises.

What it does
------------
Drops detected entities whose whole text is universal technical vocabulary —
technology/product names, protocol parameter names, standard-format acronyms —
that identifies nobody. This is the industry-standard mechanism for the
"technical vocabulary flagged as sensitive" problem (Presidio allow_list, Google
DLP exclusion dictionaries, Datadog redaction-excluded-identifiers, gitleaks
allowlists): a sourced suppression pass applied after detection, not a change to
the detectors.

Matching contract (the safety property)
---------------------------------------
  * whole-token, case-insensitive: an entity is suppressed only when its ENTIRE
    text equals an allow-list term (after casefold + surrounding-quote/space
    trim). Substrings never match, so allow-listing "oracle" cannot un-mask
    "oracle-client-db-prod-4471" — that string is not equal to "oracle".

    SCOPE OF THAT SENTENCE, corrected 2026-10-14. It is true of THIS module
    and it is not true of the pipeline as a whole. `maskerade/stages/id.py`
    reads the same term list and tests a candidate's PREFIX against it, so
    `oracle-client-db-prod-4471` IS vetoed there — by the id stage, not by
    this filter. Measured; owner decision to keep, with the reasoning and
    the accepted cost recorded at the `_load_allowlist` comment in that file.
    Anyone relying on the whole-token property must read it as a property of
    this filter's own suppression pass, not as a guarantee that an
    allow-listed term can never cause a longer string to go unmasked.
  * custom-stage entities are never suppressed: explicit pentester intent wins,
    so a client literally named "Oracle" can be forced masked via a custom rule.
    (Only detector-slot entities reach this filter, so this holds by
    construction, but the guard is explicit for safety.)

Residual risk: a client legitimately named after a technology (company called
"Oracle") whose name is detected as a bare "Oracle" will not be masked by the
probabilistic detectors once suppressed here. Mitigation: a custom rule, or
listing the term under `exclude:` in the YAML. This is the accepted trade of any
allow-list and is why matching is whole-token and custom-exempt.
Beyond the vocabulary allow-list, this module also applies a few TYPE-AWARE
POLICY FILTERS (config-driven, in allowlist_overrides.yaml) that drop whole
classes of detection the pipeline should not mask:
  * email_requires_at — an EMAIL_ADDRESS detection with no '@' is never a real
    address (fixes a field/word merely containing "email" being masked);
  * drop_types — entity types out of scope for the engagement (e.g. FINANCIAL_DATA);
  * drop_single_token_types — one-word place names (bare city/region/timezone),
    which are public geography rather than client PII.
"""
import os
import re

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "allowlist.yaml")


class AllowList:
    """A loaded allow-list with a whole-token matcher + type-aware policy filters."""

    def __init__(self, terms_by_group=None, exclude=None, policy=None):
        # term (casefolded) -> group name, for per-source attribution
        self._term_group = {}
        # separator-collapsed term -> group, for space/case/separator-insensitive
        # matching of multi-part TECH names ("amazon s3" == "AmazonS3",
        # "content-type" == "ContentType"). Only built for terms that actually
        # contain a separator and whose collapsed form is >= 5 chars, so short
        # tokens can't collide; single tokens are already handled by exact match.
        self._collapsed = {}
        for group, terms in (terms_by_group or {}).items():
            for t in terms:
                key = self._norm(t)
                if key:
                    self._term_group.setdefault(key, group)
                    ck = self._collapse(key)
                    if ck and len(ck) >= 5 and ck != key:
                        self._collapsed.setdefault(ck, group)
        # `exclude` lets a user force-keep a term that would otherwise be
        # suppressed (e.g. their client really is called "Oracle").
        for t in (exclude or []):
            k = self._norm(t)
            self._term_group.pop(k, None)
            self._collapsed.pop(self._collapse(k), None)
        self.policy = policy or {}

    # ── normalisation: casefold + strip surrounding quotes/space ──
    @staticmethod
    def _norm(text):
        if not text:
            return ""
        s = text.strip()
        while len(s) >= 2 and s[0] in "\"'`" and s[0] == s[-1]:
            s = s[1:-1].strip()
        return s.casefold()

    # collapse spaces / underscores / hyphens / dots for separator-insensitive
    # matching of multi-part technology names.
    @staticmethod
    def _collapse(norm_text):
        return re.sub(r"[ _.\-]", "", norm_text)

    def __len__(self):
        return len(self._term_group)

    def group_of(self, entity_text):
        """Return the allow-list group that matches this text, or None."""
        n = self._norm(entity_text)
        g = self._term_group.get(n)
        if g is not None:
            return g
        # fall back to separator-collapsed match (multi-part tech names)
        ck = self._collapse(n)
        if len(ck) >= 5:
            return self._collapsed.get(ck)
        return None

    def _policy_reason(self, e):
        """Return a policy label if this entity should be dropped by a type-aware
        policy rule, else None. Applied only to non-custom, non-vocab entities.
        """
        p = self.policy
        if not p:
            return None
        etype = (e.get("entity_type") or "").upper()
        text = e.get("text", "") or ""

        # 1. A real email always contains '@'. An EMAIL_ADDRESS detection without
        #    one is a false positive (a field/word merely containing "email").
        #    The '@' may be URL-encoded: the email pattern in
        #    default_config.yaml deliberately matches %40 and %2540, because
        #    addresses cross the wire encoded. Until 2026-10-21 this check
        #    tested for a literal '@' only and silently dropped every such
        #    match, so `jan.novak%40acmebank.cz` was sent whole.
        if p.get("email_requires_at") and etype in ("EMAIL_ADDRESS", "EMAIL"):
            if not any(at in text for at in ("@", "%40", "%2540")):
                return "policy:email_no_at"

        # 2. Entity types dropped entirely (out of scope for the engagement).
        if etype in {t.upper() for t in (p.get("drop_types") or [])}:
            return f"policy:drop_type:{etype}"

        # 3. One-word place names are public geography, not client PII. A single
        #    token = no internal whitespace (multi-word street addresses stay).
        if etype in {t.upper() for t in (p.get("drop_single_token_types") or [])}:
            if text.strip() and not re.search(r"\s", text.strip()):
                return f"policy:single_token:{etype}"

        return None

    def filter(self, entities, text=None):
        """Split entities into (kept, suppressed).

        Suppressed items are annotated with `_allowlist_group` (a vocabulary
        group name, or a `policy:...` label). Custom-stage entities are always
        kept — explicit pentester intent overrides both vocab and policy.

        `text` (the source the entities were detected in) enables the span-aware
        guard: a vocabulary or single-word-location match is NOT suppressed when
        the entity is a fragment of a larger identifier in the text (e.g. the
        region in `...cluster-x.eu-central-1.rds.amazonaws.com`). The guard can
        only ever KEEP more, so it is strictly recall-safe.

        CALLER CONTRACT: `text` must be the exact string the entities' start/end
        offsets index into. Entities from a different coordinate frame (the
        regex stage detects against pre-regex-masking text) must be passed in a
        SEPARATE call with text=None — offsets that are merely in range but
        point at the wrong characters make the guard decide confidently and
        wrongly. See Pipeline.run(), which splits the list for this reason.
        """
        kept, suppressed = [], []
        for e in entities:
            if e.get("stage") == "custom":
                kept.append(e)
                continue
            group = self.group_of(e.get("text", ""))
            is_vocab = group is not None
            reason = group if is_vocab else self._policy_reason(e)
            if reason is not None:
                # The embedding guard applies to vocabulary matches and to the
                # single-word-location policy (tokens that could be a fragment of
                # a sensitive composite). It does NOT apply to email_no_at or
                # drop_type, which are type policies unrelated to embedding.
                guardable = is_vocab or reason.startswith("policy:single_token")
                if guardable and text is not None and self._is_embedded(e, text):
                    reason = None
            if reason is None:
                kept.append(e)
            else:
                item = dict(e)
                item["_allowlist_group"] = reason
                suppressed.append(item)
        return kept, suppressed

    # identifier continuation chars — a run of these around an entity means the
    # entity is a sub-token of a larger identifier (hostname, ARN, resource id).
    _ID_CHARS = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )

    def _is_embedded(self, e, text):
        """True if the entity is a fragment of a larger identifier in `text`.

        Expands outward from the entity span over identifier characters; if the
        surrounding maximal run (minus edge punctuation) is longer than the
        entity itself, the entity is embedded (e.g. a region inside a hostname).
        """
        try:
            s = int(e["start"]); en = int(e["end"])
        except (KeyError, TypeError, ValueError):
            return False
        if s < 0 or en > len(text) or s >= en:
            return False
        i = s
        while i > 0 and text[i - 1] in self._ID_CHARS:
            i -= 1
        j = en
        while j < len(text) and text[j] in self._ID_CHARS:
            j += 1
        run = text[i:j].strip("._-")
        return len(run) > (en - s)


def load_allowlist(path=None):
    """Load the allow-list. Returns an AllowList (empty if unavailable).

    If `path` is a file, load just that file. If `path` is a directory (or None,
    meaning this package's own directory), load and MERGE every ``*.yaml`` in it
    — so an optional, separately-licensed file such as ``allowlist_wappalyzer.yaml``
    is picked up automatically when present and simply absent otherwise.

    Never raises: a missing file/dir or missing pyyaml yields an empty (no-op)
    allow-list so the pipeline still runs.
    """
    try:
        import yaml
    except ImportError:
        print("  [ALLOWLIST] pyyaml not installed — suppression disabled")
        return AllowList()

    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))

    if os.path.isdir(path):
        import glob
        files = sorted(glob.glob(os.path.join(path, "*.yaml")))
    elif os.path.exists(path):
        files = [path]
    else:
        print(f"  [ALLOWLIST] {path} not found — suppression disabled "
              f"(run maskerade/allowlist/build.py from a source checkout)")
        return AllowList()

    terms_by_group = {}
    exclude = []
    policy = {}
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        for name, g in (doc.get("groups") or {}).items():
            terms_by_group.setdefault(name, [])
            terms_by_group[name].extend(g.get("terms") or [])
        exclude.extend(doc.get("exclude") or [])
        pol = doc.get("policy")
        if isinstance(pol, dict):
            policy.update(pol)
    return AllowList(terms_by_group, exclude=exclude, policy=policy)
