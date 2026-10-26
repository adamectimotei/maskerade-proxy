"""CPU orchestrator — the control plane for the standalone tool.

Derived from the original GPU implementation's orchestrator, with the `ner`
(Flair/Presidio) and `llm` (Ollama) branches deleted and the CPU detectors
promoted from a benchmark-harness patch into real, configurable stages.

Flow:
  1. Custom rules (user intent, highest priority)
  2. Regex
  3. Intermediate masking — regex/custom hits become placeholders
  4. Form (entropy / format standards) on the intermediate text
  5. The DETECTOR SLOT on the intermediate text (see _run_detectors)
  6. Allow-list suppression
  7. Span reconciliation
  8. Final masking against the ORIGINAL text
  9. Validation

Steps 4-5 read the SAME intermediate text. That shared coordinate frame is
what makes _reconcile_spans() able to compare offsets; masking between them
would invalidate it.

WHY THE DETECTOR SLOT IS ONE BLOCK
----------------------------------
In the original implementation every detector in this slot was composed into
a single `ner_detect` by the benchmark harness, and the orchestrator applied
placeholder-filtering and _remove_overlaps() to the COMBINED output, once.

_remove_overlaps() dedupes by entity TEXT. Splitting the slot into stages that
each extend all_entities and each call _remove_overlaps() would start deduping
keyval against gliner by text — which never happens today. Intra-slot dedup is
positional (`already=out`), and only checksum and id use it. So the slot is one
block here too: the toggles are per detector, the execution is not.

The order inside it is load-bearing. See _run_detectors().
"""
import importlib
import os
import re
import time

from .config import load_config
from .masker import apply_masks, unmask_response, is_low_risk_numeric_id
from .validator import validate
from .stages.custom import apply_custom_masks
from .stages.regex import regex_detect


# Stages whose entity offsets index the intermediate text. Everything else
# (custom, regex) was detected against an earlier coordinate frame and its
# offsets are NOT comparable to these. Both the allow-list's embedding guard
# and _reconcile_spans depend on this distinction, so it is defined once.
#
# "llm" is retained although this package has no LLM stage: it costs nothing,
# and removing a member of a frozenset the reconciler keys on is a behavioural
# edit this port is not allowed to make.
FRAME_STAGES = frozenset({"ner", "llm"})


# Catch-all entity types: the detector knows the value is sensitive but not
# what specific kind it is. keyval emits SENSITIVE_FIELD ("this key names a
# sensitive value"); the entropy net emits HIGH_ENTROPY_SECRET ("this is
# random"). Used by span reconciliation: when a generic type covers a span
# that a more authoritative detector typed specifically (IBAN, UUID, ...),
# the specific label is promoted onto the covering extent.
GENERIC_ENTITY_TYPES = frozenset({"SENSITIVE_FIELD", "HIGH_ENTROPY_SECRET"})


# ── Stage key set ────────────────────────────────────────────────────────
# CLOSED, deliberately. The original orchestrator did stages.get(name, True), so an
# override written before a stage existed silently ran it. With nine keys that
# is how a "pure-Python tier" run quietly includes gliner and every latency
# number becomes meaningless. Unknown keys raise; missing keys come from the
# named tier, never from a bare True.
STAGE_KEYS = (
    "custom", "regex", "form",
    "gliner", "keyval", "checksum", "id", "org_evidence",
    "allowlist", "validator",
)

# Detectors living in the one slot. Order here is documentation only; the
# executed order is fixed in _run_detectors().
SLOT_DETECTORS = ("gliner", "keyval", "checksum", "id")

TIER_FAST = {
    "custom": True, "regex": True, "form": True,
    "gliner": False, "keyval": True, "checksum": True, "id": True,
    # org_evidence is OFF here, and the reason is not the one that used to be
    # written on this line. It said "gliner-only filter; nothing to filter",
    # which is false: org_evidence governs ORG_LABELS — ORGANIZATION, ORG,
    # COMPANY_NAME, COMPANY — whatever produced them, and keyval emits
    # ORGANIZATION for companyName / vendorName / tenantName / merchantName
    # (stages/keyval.py, Rule 2b, NAME_ORG_CONTEXT). So with gliner off there
    # ARE governed spans in this tier; there is simply nothing wired to filter
    # them, because _run_detectors() gates the filter on `stages["gliner"] and
    # stages["org_evidence"]`.
    #
    # It stays off, and stays off for a measured reason rather than a
    # structural one: switching it on here would start dropping keyval's
    # ORGANIZATION spans, which is a DETECTION CHANGE in the under-masking
    # direction on a tier nobody has measured yet. That needs a prediction
    # and a measured run, not a packaging change. Leave it.
    "org_evidence": False,
    "allowlist": True, "validator": True,
}

# C2 — the shipped configuration: every CPU detector on.
TIER_FULL = {
    "custom": True, "regex": True, "form": True,
    "gliner": True, "keyval": True, "checksum": True, "id": True,
    "org_evidence": True,
    "allowlist": True, "validator": True,
}

TIERS = {"fast": TIER_FAST, "full": TIER_FULL}

# The tier a caller gets when it does not name one.
#
# "fast" since 2026-10-17, and this is the whole of the light-install change:
# `pip install maskerade` brings three pure-Python dependencies, and the tier
# it defaults to is the one those dependencies can actually run. The previous
# default was "full", which on a light install constructs fine, prints a mode
# string naming GLINER, and then raises DetectorError at warmup — correct
# fail-closed behaviour, and a terrible first run.
#
# This is a TIER default, not a config default. Writing `gliner: false` into
# default_config.yaml would have been the other way to do it and would have
# been wrong: _resolve_stages() merges tier < config < override, so config
# BEATS tier, and `--tier full` on a machine with [ner] installed would have
# silently kept gliner off while the operator believed they had asked for C2.
# The failure would have been invisible in every place anyone looks.
#
# The cost, stated: a [ner] user who upgrades and does not pass --tier full
# drops from C2 to the fast tier. That is under-masking, which is the
# direction this project refuses, so it is not left to be discovered — the
# proxy prints a line at startup when the extra is installed and the fast
# tier is what is running. It reports; it never switches on its own, because
# a tier that depends on what happens to be installed is a configuration that
# differs between two machines with nothing in the output to explain why.
DEFAULT_TIER = "fast"


# A string that is one placeholder and nothing else, e.g. [ADMIN_1].
_ONLY_PLACEHOLDER_RE = re.compile(r"\[[A-Z0-9_]+_\d+\]")


def _emails_cut_by_rules(text, custom_masks, config):
    """Email addresses in `text` that a custom rule would cut in half.

    Added 2026-10-21. The custom stage runs before regex and substitutes
    rule values wherever they occur, including inside an address:
    `jan.novak@acmebank.cz` became `jan.novak@[CLIENT_DOMAIN_1]`, which the
    email pattern no longer matches, so the local part reached the provider.

    Each address the configured email pattern finds is passed through the
    REAL custom matcher on its own. If a rule changes it without replacing
    it entirely, the rule would cut it, and it is returned so the caller can
    mask it whole first. If a rule replaces the whole address, the operator
    named it explicitly and the rule wins.
    """
    out = []
    for e in regex_detect(text, config):
        if e.get("entity_type") != "EMAIL_ADDRESS":
            continue
        value = e.get("text") or ""
        masked, _ents, _map = apply_custom_masks(value, custom_masks)
        if masked == value or _ONLY_PLACEHOLDER_RE.fullmatch(masked):
            continue
        out.append(dict(e, pre_custom=True))
    return out


def _warmup_remedies(names):
    """What to do about each failed stage, for the DetectorError message.

    Per stage, because the old single sentence gave every failure the same
    advice -- `warm --full` and a library-only `stages_override=` -- which
    was wrong for the commonest case: a fast-tier install that has not run
    `python -m maskerade.warm` yet, and needs neither the model nor Python.
    """
    out = []
    if "form" in names:
        out.append("  form:   fetch the BPE encoder once, with network: "
                   "python -m maskerade.warm")
    if "gliner" in names:
        out.append("  gliner: the full tier needs the 'ner' extra and its "
                   "model (pip install \"maskerade[ner]\", then "
                   "python -m maskerade.warm --full), or run the fast tier "
                   "(the default; drop --tier full)")
    other = [n for n in names if n not in ("form", "gliner")]
    if other:
        out.append("  %s: these ship with maskerade itself; reinstall it "
                   "(pip install --force-reinstall maskerade) and report the "
                   "error above if it persists" % ", ".join(other))
    out.append("To run without a stage on purpose, switch it off explicitly: "
               "a --config file with `stages: {<stage>: false}`, or "
               "stages_override={'<stage>': False} from Python.")
    return "\n".join(out)


class DetectorError(RuntimeError):
    """A stage was enabled but could not run.

    Raised rather than logged. An under-masked result that reports success is
    worse than no result, because the operator sends it anyway.
    """


class Pipeline:
    """CPU-only masking pipeline."""

    def __init__(self, config_path=None, tier=DEFAULT_TIER):
        self.config = load_config(config_path)
        self.tier = tier
        self.allowlist = self._load_allowlist()

    # ── construction helpers ─────────────────────────────────────────────

    def _load_allowlist(self):
        """Load the technical-vocabulary allow-list once, at construction.

        Returns an AllowList, or None when disabled/unavailable. Loading it
        here rather than per run() matters: the merged list is ~10k terms
        across every *.yaml in the package directory.

        Never fatal. A missing directory, missing pyyaml or a malformed YAML
        disables suppression and says so — masking more than necessary is the
        safe direction to fail.
        """
        allowlist_config = self.config.get("allowlist", {})
        if not allowlist_config.get("enabled", True):
            return None

        path = allowlist_config.get("path")
        if path and not os.path.isabs(path):
            # Relative to the PACKAGE directory. The original implementation
            # resolved it against the repo root; here the allowlist ships inside the
            # package, so a relative config path means "beside the code".
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

        try:
            from .allowlist import load_allowlist
            loaded = load_allowlist(path)
        except Exception as e:
            print(f"  [ALLOWLIST] Load failed ({e}) — suppression disabled")
            return None

        if not len(loaded):
            print("  [ALLOWLIST] Empty — suppression disabled")
            return None

        print(f"  [ALLOWLIST] {len(loaded)} terms loaded")
        return loaded

    def _resolve_stages(self, stages_override=None):
        """Merge tier default < config['stages'] < stages_override.

        Raises on unknown keys. A typo in a stage name is otherwise
        indistinguishable from leaving the stage at its default, which is
        exactly the failure the closed key set exists to prevent.
        """
        resolved = dict(TIERS[self.tier])
        for source in (self.config.get("stages") or {}, stages_override or {}):
            unknown = set(source) - set(STAGE_KEYS)
            if unknown:
                raise ValueError(
                    f"Unknown stage key(s): {sorted(unknown)}. "
                    f"Valid keys: {list(STAGE_KEYS)}"
                )
            resolved.update({k: bool(v) for k, v in source.items()})
        return resolved

    def mode_string(self, stages=None):
        """Human-readable record of which stages are actually live.

        The one cheap guard against measuring a configuration you did not mean
        to run. Print it; put it in result JSON; compare it between runs.
        """
        stages = stages or self._resolve_stages()
        parts = []
        if stages["custom"]:
            parts.append("CUSTOM")
        if stages["regex"]:
            parts.append("REGEX")
        if stages["form"]:
            entropy = (self.config.get("form") or {}).get("entropy_method", "bpe")
            parts.append(f"FORM({entropy})")
        for name in SLOT_DETECTORS:
            if stages[name]:
                label = name.upper()
                if name == "gliner":
                    # The gate shipped at 0 for a day because the mechanism was
                    # live and the policy was not, and nothing in the run
                    # header said so. Every run now states the policy it ran
                    # under, so a wrong value is visible in the first line of
                    # output instead of inferable from a missing skip.
                    limit = (self.config.get("gliner") or {}).get(
                        "max_input_chars", 0)
                    if limit:
                        label = "GLINER(<=%d)" % limit
                parts.append(label)
        if stages["gliner"] and stages["org_evidence"]:
            parts.append("ORG_EVIDENCE")
        return " + ".join(parts) if parts else "NONE"

    # ── the detector slot ────────────────────────────────────────────────

    def _run_detectors(self, text, stages, debug=False):
        """Run the CPU detectors over the intermediate text, in fixed order.

        ORDER IS LOAD-BEARING. Carried verbatim from the original
        implementation's combined detector:

          gliner    probabilistic semantic spans
          keyval    structural; claims freely, so its spans are in `already`
                    for the two detectors below
          checksum  FIRST of the proved tier: a value proved by published
                    arithmetic outranks every shape-based guess that follows
          id        after checksum: an ID is a specific shape, and whatever
                    claims a span first keeps it
          org_evidence  STRICTLY LAST. It corroborates ORGANIZATION spans
                    against the high-confidence detections in this same list,
                    so every other detector must have contributed first.

        Returns (entities, per_detector_seconds, notes). `notes` carries any
        "this stage declined the input" record — currently only gliner's size
        gate. It is NOT an error path and not an exception: the run is valid,
        but it masked with fewer layers than the mode string advertises, and
        the caller is about to send the result to a third party. Reporting it
        is the difference between a documented gate and a silent one.

        The timing dict is additive
        instrumentation — it is reported separately from `timing` so that
        stats.total_time keeps summing to the same number as before.
        """
        out = []
        secs = {}
        notes = []

        def _step(name, fn):
            """Run one detector. A requested detector that cannot run is an
            ERROR, never a silent skip.

            This used to be a single try/except around the whole slot, and it
            cost a day: gliner raised ModuleNotFoundError, the handler set the
            slot's output to [], and keyval + checksum + id — all installed,
            all working — contributed nothing. The pipeline then reported
            `validation passed: True` on a run that masked a fraction of what
            it should have.

            That is the worst failure mode this tool has, wearing a
            different hat: the tool quietly masks less than the operator
            believes, and nothing in the output says so. For a tool whose
            failure mode is sending client PII to a third party, "a stage you
            asked for is unavailable" must stop the run. Switch the stage off
            explicitly, or use tier="fast", if you mean to run without it.
            """
            t0 = time.time()
            try:
                result = fn()
            except Exception as e:
                raise DetectorError(
                    f"stage {name!r} was enabled but failed: "
                    f"{type(e).__name__}: {e}. Disable it explicitly "
                    f"(stages_override={{'{name}': False}}) or install what it "
                    f"needs — do not let it fail silently, the output would "
                    f"be under-masked."
                ) from e
            secs[name] = round(time.time() - t0, 4)
            return result

        if stages["gliner"]:
            from .stages.gliner import gliner_detect
            out.extend(_step("gliner", lambda: gliner_detect(
                text, self.config, debug=debug, notes=notes)))

        if stages["keyval"]:
            from .stages.keyval import keyval_detect
            loose = bool((self.config.get("keyval") or {}).get("loose", False))
            out.extend(_step("keyval", lambda: keyval_detect(
                text, self.config, loose=loose, debug=debug)))

        if stages["checksum"]:
            from .stages.checksum import checksum_detect
            out.extend(_step("checksum", lambda: checksum_detect(
                text, self.config, already=out, debug=debug)))

        if stages["id"]:
            from .stages.id import id_detect
            out.extend(_step("id", lambda: id_detect(
                text, self.config, already=out, debug=debug)))

        # Tied to gliner: it is the only detector in this slot that emits
        # ORGANIZATION, so with gliner off there is nothing for it to filter.
        if stages["gliner"] and stages["org_evidence"]:
            from .stages.org_evidence import org_evidence_filter
            out = _step("org_evidence", lambda: org_evidence_filter(
                text, self.config, out, debug=debug))

        return out, secs, notes

    # ── main entry point ─────────────────────────────────────────────────

    def run(self, raw_text, verbose=False, existing_mask_map=None,
            stages_override=None, custom_masks=None, whitelist=None,
            debug=False):
        """Run detection → mask → validate.

        Args:
            raw_text:          text to anonymise
            verbose:           print progress to stdout
            existing_mask_map: cumulative mask map from previous turns, so the
                               same entity always gets the same placeholder
            stages_override:   partial dict of STAGE_KEYS overriding the tier
            custom_masks:      per-chat user rule dicts (see stages/custom.py)
            whitelist:         values never to mask. A list/set of exact
                               strings, or an object with
                               .filter(entities, text=...) -> (kept, dropped).
                               When omitted, the allow-list loaded at
                               construction is used unless stages.allowlist
                               is False.
            debug:             pass through to each detector's own debug output

        Returns a dict with masked_text, mask_map, entities, validation,
        timing, mode, stats, stage_notes, masker_unmasked and
        masker_realigned.

        masker_unmasked is the fail-open set and is NOT covered by
        validation["passed"] — see the comment at the final apply_masks call.
        A caller deciding whether it is safe to send masked_text to a third
        party must check both.
        """
        stages = self._resolve_stages(stages_override)
        all_entities = []
        intermediate_text = None
        timing = {}
        detector_secs = {}
        stage_notes = []

        # Track custom-rule entries to merge into existing_mask_map BEFORE the
        # final apply_masks call. Without this merge the custom values would
        # still be in raw_text (we only modified intermediate copies) and
        # apply_masks would re-detect and re-number them.
        custom_mask_entries = {}

        # ── Email-first (only when custom rules are in force) ──
        # A custom rule for the client's domain or name cuts an address in
        # half before regex can see it whole, and the local part leaks. So
        # every address a rule WOULD cut is masked whole here, through the
        # standard masker and the conversation's existing map, and the rest
        # of the pipeline runs on that text. raw_text and existing_mask_map
        # are rebound on purpose: everything below, including the final
        # masking, must start from the text with these addresses already
        # substituted. With no rules this block does not run and nothing
        # below changes. See _emails_cut_by_rules().
        email_first_entities, email_first_unmasked = [], []
        if stages["custom"] and stages["regex"] and custom_masks:
            email_first_entities = _emails_cut_by_rules(
                raw_text, custom_masks, self.config)
            if email_first_entities:
                pre_report = {}
                raw_text, existing_mask_map = apply_masks(
                    raw_text, email_first_entities, self.config,
                    existing_mask_map, report=pre_report)
                email_first_unmasked = pre_report.get("unmaskable", [])
                if verbose:
                    print(f"  [EMAIL] {len(email_first_entities)} address(es) "
                          f"masked whole before the custom stage")

        # ── Custom stage (highest priority — runs before regex) ──
        text_for_detection = raw_text
        if stages["custom"] and custom_masks:
            t0 = time.time()
            try:
                text_for_detection, custom_entities, custom_mask_entries = \
                    apply_custom_masks(raw_text, custom_masks)
                all_entities.extend(custom_entities)
                timing["custom"] = round(time.time() - t0, 3)
                if verbose:
                    print(f"  [CUSTOM] {len(custom_entities)} rules fired "
                          f"({timing['custom']}s)")
            except Exception as e:
                timing["custom"] = round(time.time() - t0, 3)
                text_for_detection = raw_text
                custom_mask_entries = {}
                if verbose:
                    print(f"  [CUSTOM] Error: {e}")

        # ── Regex ──
        if stages["regex"]:
            t0 = time.time()
            regex_entities = regex_detect(text_for_detection, self.config)
            timing["regex"] = round(time.time() - t0, 3)
            all_entities.extend(regex_entities)
            if verbose:
                print(f"  [REGEX] {len(regex_entities)} entities "
                      f"({timing['regex']}s)")

        # ── Intermediate masking ──
        # Replace regex/custom hits with placeholders before the later stages
        # see the text, so no detector ever processes an 800-char JWT.
        # Custom entities are EXCLUDED: apply_custom_masks() already
        # substituted them in text_for_detection, so asking the masker
        # to find them produces a false 'could not be masked' warning
        # on every request that uses a rules file. The final
        # apply_masks() call below filters them for the same reason.
        intermediate_text, _ = apply_masks(
            text_for_detection,
            [e for e in all_entities if e.get("stage") != "custom"],
            self.config)

        # ── Form and the detector slot both run on intermediate_text ──
        # One shared coordinate frame is a hard requirement, not a convenience:
        # _reconcile_spans() compares entities by offset, and every stage that
        # tags stage="ner" (form does, deliberately, so the masker and
        # validator treat it like any other slot output) lands in that
        # comparison. Masking form's hits before the slot runs would shift
        # every later offset and silently make the reconciler compare spans
        # from two different texts. Form still runs FIRST so its values
        # suppress duplicate slot detections via _remove_overlaps.

        if stages["form"]:
            t0 = time.time()
            form_entities = []
            try:
                from .stages.form import form_detect
                form_entities = form_detect(intermediate_text, self.config,
                                            notes=stage_notes)
                form_entities = self._remove_overlaps(form_entities, all_entities)
                all_entities.extend(form_entities)
            except Exception as e:
                # Owner decision 2026-10-05: this tool does not fail open.
                # form was the one requested stage that swallowed its own
                # failure and returned a result reporting success, while the
                # detector slot's _step() raised. Same contract now.
                raise DetectorError(
                    "stage 'form' was enabled but failed: "
                    "%s: %s. Disable it explicitly "
                    "(stages_override={'form': False}) or install what it "
                    "needs — do not let it fail silently, the output would "
                    "be under-masked." % (type(e).__name__, e)) from e
            timing["form"] = round(time.time() - t0, 3)
            if verbose:
                print(f"  [FORM] {len(form_entities)} entities "
                      f"({timing['form']}s)")

        # ── Detector slot ──
        # ONE call, one placeholder filter, one _remove_overlaps — exactly as
        # the original orchestrator treated the composed ner_detect. See the module
        # docstring for why this must not be split per detector.
        if any(stages[name] for name in SLOT_DETECTORS):
            t0 = time.time()
            slot_entities = []
            # org_evidence corroborates against the ORIGINAL text, which it
            # reads from module state. The original implementation set this from
            # its benchmark harness around each run(); it is detection state, so
            # it lives here.
            org_ctx = stages["gliner"] and stages["org_evidence"]
            if org_ctx:
                try:
                    from .stages import org_evidence as _org
                    _org.set_corroboration_context(raw_text=raw_text)
                except Exception:
                    org_ctx = False
            # No try/except here on purpose. DetectorError propagates: a
            # requested detector that cannot run means the output would be
            # under-masked, and a caller about to send this text to a third
            # party must find that out by exception, not by reading counts.
            try:
                # EXTEND, never rebind. form appends its degradation note to
                # stage_notes before this point, and assigning the slot's own
                # list here discarded it — the note was built and thrown away,
                # which is the exact failure stage_notes exists to prevent.
                slot_entities, detector_secs, slot_notes = self._run_detectors(
                    intermediate_text, stages, debug=debug)
                stage_notes.extend(slot_notes)
                # Our own placeholders are not entities.
                slot_entities = [
                    e for e in slot_entities if not e["text"].startswith("[")]
                slot_entities = self._remove_overlaps(slot_entities, all_entities)
                all_entities.extend(slot_entities)
            finally:
                if org_ctx:
                    try:
                        _org.clear_corroboration_context()
                    except Exception:
                        pass
            timing["detect"] = round(time.time() - t0, 3)
            if verbose:
                detail = ", ".join(f"{k} {v}s" for k, v in detector_secs.items())
                print(f"  [DETECT] {len(slot_entities)} new entities "
                      f"({timing['detect']}s — {detail})")

        # ── Allow-list / whitelist filter ──
        # Applied after every detection stage so it overrides all of them: a
        # technical term like "Mozilla" is equally wrong coming from a regex or
        # from gliner, so suppression belongs here and not inside any stage.
        active_whitelist = whitelist
        if active_whitelist is None and stages["allowlist"]:
            active_whitelist = self.allowlist

        if active_whitelist:
            t0 = time.time()
            before = len(all_entities)
            if hasattr(active_whitelist, "filter"):
                # TWO calls, not one. The allow-list's embedding guard reads
                # entity start/end as offsets into the text it is handed. Only
                # slot/form entities index intermediate_text; regex entities
                # were detected against text_for_detection, before regex
                # masking shifted everything. Their offsets are usually still
                # in range, so a single call would not error — it would slice
                # the wrong characters and decide confidently and wrongly.
                # They get text=None instead: exact whole-token matching, guard
                # off. Custom entities are exempted inside filter() itself.
                frame_text = (intermediate_text if intermediate_text is not None
                              else text_for_detection)
                framed = [e for e in all_entities
                          if e.get("stage") in FRAME_STAGES]
                unframed = [e for e in all_entities
                            if e.get("stage") not in FRAME_STAGES]

                kept_framed, dropped_framed = active_whitelist.filter(
                    framed, text=frame_text)
                kept_unframed, dropped_unframed = active_whitelist.filter(
                    unframed, text=None)

                # Rebuild in the ORIGINAL order. filter() returns the very same
                # dict objects for kept entities, so identity is a safe key and
                # the downstream stage ordering stays exactly as it was.
                kept_ids = ({id(e) for e in kept_framed}
                            | {id(e) for e in kept_unframed})
                all_entities = [e for e in all_entities if id(e) in kept_ids]
                wl_dropped = dropped_framed + dropped_unframed

                if verbose and wl_dropped:
                    groups = {}
                    for d in wl_dropped:
                        g = d.get("_allowlist_group", "?")
                        groups[g] = groups.get(g, 0) + 1
                    detail = ", ".join(f"{g}: {n}"
                                       for g, n in sorted(groups.items()))
                    print(f"  [ALLOWLIST] Dropped {len(wl_dropped)} "
                          f"entities ({detail})")
            else:
                wl_set = set(active_whitelist)
                all_entities = [e for e in all_entities
                                if e["text"] not in wl_set]
                if verbose and len(all_entities) < before:
                    print(f"  [ALLOWLIST] Dropped {before - len(all_entities)} "
                          f"whitelisted entities")
            timing["allowlist"] = round(time.time() - t0, 3)

        # ── Reconcile overlapping spans across detection stages ──
        # Several detectors run over the SAME intermediate text and can mask
        # overlapping spans with different entity types. The masker is
        # text-keyed, so each redundant (value, type) pair becomes a separate
        # mask_map entry — a phantom placeholder that inflates the over-mask
        # count, and, for nested/staggered spans, a dangling value whose
        # characters a longer replacement already consumed. Collapse each
        # overlapping cluster to one entity that still covers the union of
        # sensitive characters, choosing the type by detector authority.
        all_entities = self._reconcile_spans(
            all_entities, intermediate_text, verbose=verbose)

        # ── Final masking on the ORIGINAL text ──
        t0 = time.time()
        if custom_mask_entries:
            # Re-apply custom rules to raw_text so the final masked_text has
            # them substituted with their flag-aware semantics.
            base_text, _, _ = apply_custom_masks(raw_text, custom_masks or [])
            merged_existing = dict(existing_mask_map or {})
            merged_existing.update(custom_mask_entries)
        else:
            base_text = raw_text
            merged_existing = existing_mask_map

        # Custom entities are already in merged_existing as completed
        # substitutions and already substituted in base_text; re-running them
        # through the standard masker would word-boundary-mask them again,
        # possibly missing user-intended substring or case-insensitive matches.
        # Low-risk purely-numeric IDs (user_id=7) are left unmasked on purpose
        # so the AI keeps path/parameter context.
        non_custom_entities = [
            e for e in all_entities
            if e.get("stage") != "custom"
            and not is_low_risk_numeric_id(e, self.config)
        ]

        # The masker's diagnostics are collected, not discarded. `unmaskable`
        # is the fail-open set: entities the masker was handed, recorded as
        # detected, and then left in the output because no substitutable span
        # could be found. Until now the only trace was a print() to stdout, so
        # the number was unmeasurable and landed in the benchmark's recall as
        # ordinary false negatives — indistinguishable from "never detected",
        # which needs the opposite fix.
        #
        # Note the validator CANNOT see this set: validate() tests for leaks
        # with the same word_boundary_pattern() the masker failed to match on,
        # so every one of these passes validation at 100% coverage. Reporting
        # it here is the only way it becomes visible.
        mask_report = {}
        masked_text, mask_map = apply_masks(
            base_text, non_custom_entities, self.config, merged_existing,
            report=mask_report)
        timing["masking"] = round(time.time() - t0, 3)
        # Anything the email-first pass could not substitute is a leak like
        # any other, and goes where the proxy refuses on it.
        masker_unmasked = (email_first_unmasked
                           + mask_report.get("unmaskable", []))
        # The email-first entities join only now. They were substituted before
        # the final masking ran, so handing them to apply_masks above would
        # look for text that is no longer there and report it as unmaskable.
        # From here they are validated and counted like every other entity.
        all_entities.extend(email_first_entities)
        masker_realigned = mask_report.get("realigned", [])
        masker_widened = mask_report.get("widened", [])
        if verbose:
            print(f"  [MASK] {len(mask_map)} placeholders applied "
                  f"({timing['masking']}s)")
            if masker_unmasked:
                print(f"  [MASK] {len(masker_unmasked)} detected entit"
                      f"{'y' if len(masker_unmasked) == 1 else 'ies'} NOT "
                      f"substituted — these remain in the text that would be "
                      f"sent")
            if masker_realigned:
                print(f"  [MASK] {len(masker_realigned)} span(s) trimmed to a "
                      f"token boundary — less was masked than was detected")
            if masker_widened:
                print(f"  [MASK] {len(masker_widened)} span(s) widened to the "
                      f"containing token — MORE was masked than was detected")

        # ── Validate ──
        validation = {"passed": True, "leaked": [], "warnings": [], "stats": {}}
        if stages["validator"]:
            t0 = time.time()
            # Custom entities are excluded: they were substituted with their
            # own flags, but the validator is always-word-boundary and
            # always-case-sensitive, so validating them would produce
            # false-positive leak warnings. Low-risk numeric IDs are excluded
            # for the same reason — they were never masked on purpose.
            entities_to_validate = [
                e for e in all_entities
                if e.get("stage") != "custom"
                and not is_low_risk_numeric_id(e, self.config)
            ]
            validation = validate(masked_text, entities_to_validate, mask_map)
            timing["validation"] = round(time.time() - t0, 3)
            if verbose:
                status = "PASSED" if validation["passed"] else "FAILED"
                pct = validation["stats"].get("coverage_pct", 0)
                print(f"  [VAL] {status} — {pct}% coverage")
                for leak in validation.get("leaked", []):
                    print(f"         LEAK: '{leak['text'][:40]}' "
                          f"({leak['entity_type']})")
                for w in validation.get("warnings", []):
                    print(f"         {w}")

        # ── Build result ──
        by_stage, by_type, by_detector = {}, {}, {}
        for e in all_entities:
            s = e.get("stage", "?")
            by_stage[s] = by_stage.get(s, 0) + 1
            t = e.get("entity_type", "?")
            by_type[t] = by_type.get(t, 0) + 1
            # form reports stage="ner" on purpose (the masker and validator
            # must treat it like slot output), so by_stage alone cannot tell
            # the detectors apart.
            d = e.get("detector")
            if d:
                by_detector[d] = by_detector.get(d, 0) + 1

        return {
            "masked_text": masked_text,
            "mask_map": mask_map,
            "entities": all_entities,
            "validation": validation,
            "timing": timing,
            # Stages that declined this input. Empty on a normal run. A caller
            # that reports "masked with mode X" should report this too.
            "stage_notes": stage_notes,
            # Entities the masker was handed and did NOT substitute. Each one
            # is a detection that is still present in masked_text, so a
            # non-empty list means this result leaks and the caller must treat
            # it as a leak — `validation` will NOT, it is blind to this set by
            # construction (same word-boundary predicate). Same contract as
            # stage_notes: a declined or failed operation is visible in the
            # output, never inferred from a missing number.
            "masker_unmasked": masker_unmasked,
            # Spans shrunk to a token boundary so they could be substituted.
            # Not a leak on its own, but the masked value is SHORTER than the
            # detected one, so the remainder of the detected span is still in
            # the output. Usually an over-greedy pattern running past its field.
            "masker_realigned": masker_realigned,
            # Spans grown to the containing token because they could not be
            # substituted as detected. Not a leak — the opposite, an
            # over-mask — but it is the masker doing something other than
            # what the detector asked for, so it is reported rather than
            # silently absorbed into the false-positive count.
            "masker_widened": masker_widened,
            "mode": self.mode_string(stages),
            "stats": {
                "total_entities": len(all_entities),
                "by_stage": by_stage,
                "by_detector": by_detector,
                "by_type": by_type,
                "detector_seconds": detector_secs,
                "total_time": round(sum(timing.values()), 1),
            },
        }

    def unmask(self, response_text, mask_map):
        """Restore original values in a reply. Thin passthrough to masker."""
        return unmask_response(response_text, mask_map)

    def warmup(self, stages_override=None):
        """Pre-load models and indexes before any timing starts.

        FAILS CLOSED. Owner decision 2026-10-05: every "stage unavailable"
        path in this tool refuses rather than degrades, because the
        operator's next action after a run is to send the text to a third
        party.

        This used to print each stage's warmup failure and return normally.
        The run then reported success, printed a mode string naming the
        stage, and raised DetectorError from _step() on the first message.
        Under C2 gliner is always enabled, so that was the shipped path: a
        broken model cache produced a clean-looking startup and a dead
        pipeline. Same contract as _step() now.

        Every enabled stage is attempted and ALL failures are collected
        before raising. A cold install fails both form (no tiktoken cache)
        and gliner (no HF cache); failing on the first would cost two round
        trips to discover two problems.

        A stage that is switched off is never warmed and can never raise,
        which is the documented way to run without it.
        """
        stages = self._resolve_stages(stages_override)
        failures = []

        def _warm(name, fn):
            try:
                fn()
            except Exception as e:
                failures.append((name, type(e).__name__, str(e)))

        def _warm_form():
            from .stages.form import warmup as form_warmup
            form_warmup(self.config)

        def _warm_gliner():
            from .stages.gliner import warmup as gliner_warmup
            gliner_warmup(self.config)

        def _warm_module(name):
            # Relative import, so the package can be renamed or vendored.
            # The old code hardcoded __import__("maskerade.stages." + name)
            # inside a bare except that only printed, so under any other
            # top-level name these three silently never warmed up and
            # nothing said so.
            mod = importlib.import_module(".stages." + name, __package__)
            fn = getattr(mod, "warmup", None)
            if fn is not None:
                fn(self.config)

        if stages["form"]:
            _warm("form", _warm_form)
        if stages["gliner"]:
            _warm("gliner", _warm_gliner)
        for name in ("keyval", "checksum", "id"):
            if stages[name]:
                _warm(name, lambda n=name: _warm_module(n))

        if failures:
            lines = "".join("\n    %s: %s: %s" % (n, t, m)
                            for n, t, m in failures)
            raise DetectorError(
                "warmup failed for %d enabled stage(s):%s\n"
                "Each of these stages is enabled, so each would fail again "
                "on the first message - with the output under-masked and the "
                "mode string still naming the stage.\n%s"
                % (len(failures), lines,
                   _warmup_remedies([n for n, _t, _m in failures])))
    # ── span reconciliation (copied verbatim — must not drift) ───────────

    @staticmethod
    def _detector_priority(entity):
        """Rank a detector by authority — lower number wins an overlap.

        The ordering is a generic statement about how each detector decides,
        not about any value or corpus:
          1 custom   explicit pentester intent
          2 regex    deterministic, structurally validated patterns
          3 form (checksum/format)  mod-97 / Luhn / RFC grammar — the type is
                     mathematically certain
          4 keyval / hostname  structural: a key name binds the value's
                     meaning; a Public Suffix List answer decides a name's
                     namespace. Deterministic given the string, but about
                     where the value sits rather than what it is.
          5 form (entropy)          knows it is a secret, not which kind
          6 gliner   probabilistic semantic guess
          7 llm      contextual guess
        form distinguishes tier 3 vs 5 by entity_type: the entropy net is the
        only path that emits HIGH_ENTROPY_SECRET.

        The "hostname" and "llm" arms are dead in this package. They are kept
        because deleting a branch of the key the reconciler sorts on is a
        behavioural edit, and this port must produce identical output.
        """
        stage = entity.get("stage")
        if stage == "custom":
            return 1
        if stage == "regex":
            return 2
        if stage == "ner":
            detector = entity.get("detector")
            if detector == "form":
                return 5 if entity.get("entity_type") == "HIGH_ENTROPY_SECRET" else 3
            if detector in ("keyval", "hostname"):
                return 4
            return 6
        if stage == "llm":
            return 7
        return 8

    @classmethod
    def _reconcile_key(cls, entity):
        """Total, order-independent sort key. First element wins the cluster."""
        return (
            cls._detector_priority(entity),
            -(entity["end"] - entity["start"]),   # longer span first
            -float(entity.get("score", 0.0)),     # higher confidence first
            entity.get("entity_type", ""),        # stable, value-independent
            entity["start"],
        )

    @classmethod
    def _resolve_cluster_type(cls, cluster, base_type):
        """Specific-over-generic promotion.

        If the covering entity's type is a catch-all (SENSITIVE_FIELD /
        HIGH_ENTROPY_SECRET) but a more authoritative detector typed the span
        specifically (IBAN, UUID, CREDIT_CARD, ...), adopt the specific label —
        it is more useful downstream and covers the same characters.
        """
        if base_type not in GENERIC_ENTITY_TYPES:
            return base_type
        specific = [e for e in cluster
                    if e.get("entity_type") not in GENERIC_ENTITY_TYPES]
        if not specific:
            return base_type
        return min(specific, key=cls._reconcile_key)["entity_type"]

    def _reconcile_spans(self, entities, frame_text, verbose=False):
        """Collapse overlapping detections to one entity per cluster.

        Guarantees (the acceptance gate):
          - Coverage never drops. Every cluster collapses to an entity whose
            text spans the union of the cluster, so every character any
            detector flagged is still masked. Redundant masks are removed,
            never coverage.
          - Deterministic. Clustering and winner selection depend only on the
            entity set, never on stage execution order.

        Only entities in a shared coordinate frame can be compared by offset.
        Slot and form entities are detected against `frame_text` (the
        intermediate, regex/custom-masked text), so their spans are mutually
        comparable. regex/custom entities live in a different frame, but they
        were already replaced by placeholders in `frame_text`, so a slot span
        can never positionally overlap them — they pass through untouched.
        """
        frame, passthrough = [], []
        for e in entities:
            if (e.get("stage") in FRAME_STAGES
                    and isinstance(e.get("start"), int)
                    and isinstance(e.get("end"), int)
                    and e["end"] > e["start"]):
                frame.append(e)
            else:
                passthrough.append(e)

        # Cluster by positional overlap (transitive). Sort by (start, end),
        # sweep: an entity joins the running cluster if it starts before the
        # cluster's running max end. Adjacent (start == end) does NOT merge,
        # matching the masker's treatment of distinct neighbouring values.
        frame.sort(key=lambda e: (e["start"], e["end"]))
        clusters, current, current_end = [], [], None
        for e in frame:
            if current and e["start"] < current_end:
                current.append(e)
                current_end = max(current_end, e["end"])
            else:
                if current:
                    clusters.append(current)
                current, current_end = [e], e["end"]
        if current:
            clusters.append(current)

        result = list(passthrough)
        collapsed = 0
        for cluster in clusters:
            if len(cluster) == 1:
                result.append(cluster[0])
                continue

            u_start = min(e["start"] for e in cluster)
            u_end = max(e["end"] for e in cluster)

            # Members that span the whole union: exact-span duplicates and the
            # containing span of a nesting. If present, one of them is the
            # cover.
            covers = [e for e in cluster
                      if e["start"] == u_start and e["end"] == u_end]

            if covers:
                base = min(covers, key=self._reconcile_key)
                winner = dict(base)
                winner["entity_type"] = self._resolve_cluster_type(
                    cluster, base["entity_type"])
            else:
                # Staggered: no single member covers the union. Merge to the
                # union slice so no sensitive character is left uncovered. The
                # union of a connected overlap chain is gap-free and (since the
                # detectors never emit placeholder spans) placeholder-free, so
                # the slice is a verbatim substring of the final base text and
                # the text-keyed masker will find it.
                if 0 <= u_start < u_end <= len(frame_text):
                    slice_text = frame_text[u_start:u_end]
                else:
                    slice_text = ""
                if not slice_text:
                    # Offsets inconsistent with frame_text — refuse to guess.
                    # Keep every member (reintroduces the redundancy for this
                    # one cluster, but never loses coverage).
                    result.extend(cluster)
                    continue
                lead = min(cluster, key=self._reconcile_key)
                winner = dict(lead)
                winner["entity_type"] = self._resolve_cluster_type(
                    cluster, lead["entity_type"])
                winner["start"], winner["end"], winner["text"] = \
                    u_start, u_end, slice_text

            result.append(winner)
            collapsed += len(cluster) - 1
            if verbose:
                kinds = ", ".join(
                    f"{e.get('entity_type')}({e.get('detector') or e.get('stage')})"
                    for e in cluster
                )
                print(f"  [RECONCILE] {len(cluster)} overlapping "
                      f"[{kinds}] -> {winner['entity_type']}")

        if verbose and collapsed:
            n = len([c for c in clusters if len(c) > 1])
            print(f"  [RECONCILE] removed {collapsed} redundant "
                  f"placeholder(s) across {n} cluster(s)")

        return result

    def _remove_overlaps(self, new_entities, existing_entities):
        """Drop entities from new_entities that overlap with existing ones.

        Overlap = same text appears in both lists (exact match). We keep the
        existing one because it came from a more deterministic stage.
        """
        existing_texts = {e["text"] for e in existing_entities}
        return [e for e in new_entities if e["text"] not in existing_texts]
