# -*- coding: utf-8 -*-
"""The `maskerade` command.

    maskerade init --engagement acme-q4
    maskerade sessions list
    maskerade sessions purge --older-than 30d
    maskerade proxy --engagement acme-q4 --port 8787

WHY THIS EXISTS
---------------
store.py has implemented list_sessions() and purge() since the proxy was
built, and nothing could reach them except Python. The threat model states a
30-day retention policy; a policy an operator cannot execute is a sentence,
not a control. This is the hand on that lever.

IT NEVER PRINTS A MASK MAP
--------------------------
`sessions list` reports id, label, last use and a COUNT. The values behind
those placeholders are the client data this whole tool exists to keep off
other people's machines, and a `--show-values` flag would be a one-keystroke
way to put them on a terminal, in a scrollback buffer, in a screen share.
There is deliberately no such flag. An operator who genuinely needs a map
can read the JSON file the listing names.

The UNSCANNED column added 2026-10-17 holds to the same line. It lists field
NAMES -- `system`, `tools`, `safeguards` -- which are protocol vocabulary,
not request content. The store never writes a value or a path there, so
there is nothing in it that a screen share could leak. It answers the
question a CISO asks after reading the threat model: what did this
engagement send without inspecting.

IT IMPORTS NOTHING HEAVY
------------------------
`sessions` touches store.py alone -- stdlib plus nothing. The Pipeline, and
with it torch/transformers/gliner2, is imported only inside the `proxy`
subcommand, lazily. That is not a micro-optimisation: a full-tier construct
is a measured 9.31 s cold start on the laptop (decision record S8), and
`maskerade sessions list` must not pay it to print four lines. If this file
ever grows a top-level `from maskerade import Pipeline`, that property is
gone and nothing will say so.

`proxy` FORWARDS, IT DOES NOT RE-DECLARE
----------------------------------------
Every argument after `maskerade proxy` is handed to
maskerade.proxy.server.main() verbatim. Re-declaring --port/--tier/--host
here would create a second copy of that interface which drifts from the
first -- and the flag most likely to drift is --allow-remote, where a
divergence means the bind guard does not fire.

The forwarding is done by slicing argv BEFORE argparse runs, not by an
argparse.REMAINDER subcommand. REMAINDER does not collect an argument that
LOOKS like an option before any positional has been seen, so
`maskerade proxy --port 8787` exits 2 with "unrecognized arguments: --port"
from the top-level parser, which never gets a chance to reach the
subparser. The slice has no such edge: everything after the word `proxy` is
the server's, including `--help`.
"""
import argparse
import os
import sys
import time

from .proxy.store import SessionStore, StoreError


def _fmt_time(ts):
    if not ts:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _fmt_age(ts):
    if not ts:
        return "-"
    days = (time.time() - ts) / 86400.0
    if days < 1.0:
        return "today"
    return "%dd ago" % int(days)


def _parse_days(text):
    """`30d`, `30`, `30 days` -> 30.0. Raises ValueError otherwise."""
    s = (text or "").strip().lower()
    for suffix in ("days", "day", "d"):
        if s.endswith(suffix):
            s = s[:-len(suffix)].strip()
            break
    value = float(s)
    if value < 0:
        raise ValueError("a negative age is not a window")
    return value


def _open_store(args):
    return SessionStore(root=args.state_dir,
                        retention_days=args.retention_days)


def _warn(store):
    for w in store.warnings:
        sys.stderr.write("  WARNING: %s\n" % w)


# ------------------------------------------------------------- version --

def cmd_version():
    """Print the version, the default tier, and which extras resolve.

    S3.6: a bug report has to say which install it came from. Two installs
    share this version string and run different detector sets, so the version
    alone identifies nothing.

    NOTHING HEAVY IS IMPORTED. maskerade._extras probes with
    importlib.util.find_spec, which locates a module without executing it, so
    this stays inside the property the module docstring above describes and
    bench/no_torch_check.py asserts.
    """
    from . import __version__
    from ._extras import ner_available, pin_mismatch, summary
    from .pipeline import DEFAULT_TIER, TIERS

    stages = sorted(k for k, v in TIERS[DEFAULT_TIER].items() if v)
    print("maskerade %s" % __version__)
    print("  default tier  %s  (%s)" % (DEFAULT_TIER, ", ".join(stages)))
    print("  extras        %s" % summary())

    # The pin is enforced when pip RESOLVES, and says nothing about the
    # environment a run happens in. An older venv or a manual `pip install
    # -U gliner2` leaves the frozen snapshots describing a library that is
    # no longer the one doing the detecting, and nothing else would mention
    # it.
    mismatch = pin_mismatch()
    if mismatch:
        req, got = mismatch
        print("  !! PIN        this package asks for %s, and %s is installed."
              % (req, got))
        print("                The frozen snapshots were recorded against "
              "the pinned version,")
        print("                so they do not describe this install. "
              "Reinstall the extra, or")
        print("                re-check both snapshots and record the result "
              "before trusting it.")

    if DEFAULT_TIER != "full" and ner_available():
        print("  note          the 'ner' extra is installed, so tier 'full' "
              "(C2) is available.")
        print("                It is NOT the default: pass --tier full to "
              "run it.")
    elif DEFAULT_TIER == "full" and not ner_available():
        # The inverse, and the worse one: the default names a tier this
        # install cannot run, so every run fails closed at warmup.
        print("  note          the default tier needs the 'ner' extra, which "
              "is not installed.")
    return 0


# ------------------------------------------------------------ sessions --

def cmd_list(args):
    store = _open_store(args)
    _warn(store)
    rows = store.list_sessions()
    if not rows:
        print("no sessions in %s" % store.sessions_dir)
        return 0

    rows.sort(key=lambda r: r.get("updated") or 0, reverse=True)
    print("%-34s %-16s %-17s %-10s %-13s %s"
          % ("SESSION", "LABEL", "LAST USED", "AGE", "PLACEHOLDERS",
             "UNSCANNED"))
    for r in rows:
        count = r.get("n_placeholders")
        if r.get("corrupt"):
            shown = "CORRUPT"
        else:
            shown = "-" if count is None else str(count)
        unscanned = r.get("unscanned") or []
        print("%-34s %-16s %-17s %-10s %-13s %s"
              % (r.get("id", "?"),
                 (r.get("label") or "-")[:16],
                 _fmt_time(r.get("updated")),
                 _fmt_age(r.get("updated")),
                 shown,
                 len(unscanned) if unscanned else "-"))

    print("\n%d session(s) in %s" % (len(rows), store.sessions_dir))
    print("retention %d days; `maskerade sessions purge --older-than %dd` "
          "applies it now." % (store.retention_days, store.retention_days))

    # The names, once, under the table rather than in it -- a column wide
    # enough for "context_management, messages(role=system), safeguards"
    # would be wider than a terminal, and these repeat heavily across the
    # sessions of one engagement.
    every = sorted({n for r in rows for n in (r.get("unscanned") or [])})
    if every:
        print("\nUNSCANNED means the field was forwarded to the provider "
              "WITHOUT being inspected.")
        print("Across these sessions: %s" % ", ".join(every))
        print("These are field NAMES, not values; no request content is "
              "stored here or printed.")
        print("Why each one is not scanned: maskerade/proxy/fields.py, "
              "TOP_UNSCANNED.")
    if any(r.get("corrupt") for r in rows):
        print("a CORRUPT session refuses every request that resolves to it. "
              "Remove it with: maskerade sessions purge --session <id>")
    return 0


def cmd_purge(args):
    # Exactly one selector, and never a default. A bare `purge` that fell
    # back to --all would delete every conversation's numbering on a typo,
    # and there is no undo: the map is the only copy of the association
    # between a placeholder and its value.
    chosen = [n for n, v in (("--all", args.all),
                             ("--older-than", args.older_than is not None),
                             ("--session", args.session is not None)) if v]
    if len(chosen) != 1:
        sys.stderr.write(
            "  give exactly one of --all, --older-than Nd, --session ID "
            "(got %s).\n" % (", ".join(chosen) if chosen else "none"))
        return 2

    days = None
    if args.older_than is not None:
        try:
            days = _parse_days(args.older_than)
        except ValueError as e:
            sys.stderr.write("  --older-than %r is not a number of days "
                             "(%s). Try 30d.\n" % (args.older_than, e))
            return 2

    store = _open_store(args)
    _warn(store)

    if args.all:
        what = "EVERY session"
    elif days is not None:
        what = "sessions last used more than %g day(s) ago" % days
    else:
        what = "session %s" % args.session

    if not args.yes:
        if not sys.stdin.isatty():
            sys.stderr.write(
                "  refusing to purge %s without confirmation. stdin is not a "
                "terminal, so pass --yes if you mean it.\n" % what)
            return 2
        sys.stderr.write(
            "  About to delete %s from %s.\n"
            "  This cannot be undone. A conversation whose map is deleted "
            "restarts its placeholder numbering on the next turn, so "
            "[USER_ID_1] in an earlier reply and in a later one stop meaning "
            "the same account.\n" % (what, store.sessions_dir))
        try:
            answer = input("  type 'yes' to continue: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer != "yes":
            print("  nothing deleted.")
            return 1

    try:
        n_s, n_i = store.purge(older_than_days=days,
                               session_id=args.session,
                               everything=bool(args.all))
    except StoreError as e:
        sys.stderr.write("  purge failed: %s\n" % e)
        return 1

    print("removed %d session(s) and %d index entr%s"
          % (n_s, n_i, "y" if n_i == 1 else "ies"))
    if args.session is not None and n_s == 0:
        print("no session named %r; `maskerade sessions list` shows what is "
              "there." % args.session)
    return 0


# --------------------------------------------------------------- rules --

def cmd_rules_check(args):
    """Parse a rules file and show what it would mask. No daemon.

    A rules file is the operator's statement of what must never reach
    the provider, so it needs a way to be checked that does not involve
    starting a proxy and sending a real request to find out.
    """
    from .proxy.rules import RulesError, load_rules
    try:
        rules = load_rules(args.file)
    except RulesError as e:
        sys.stderr.write("  %s\n" % e)
        return 1

    if not rules:
        print("%s parses, and contains no rules." % args.file)
        return 0

    # The TYPE, never a specific index. Indices are allocated per
    # DISTINCT MATCHED STRING at request time, longest rule first, so
    # any number printed here would be a guess that the run contradicts
    # as soon as two spellings of one value appear.
    print("%-44s %s" % ("VALUE", "PLACEHOLDER"))
    for r in rules:
        print("%-44s [%s_n]" % (r["value"][:44], r["placeholder_type"]))
    print("\n%d rule(s) in %s" % (len(rules), args.file))
    print("the index in [TYPE_n] is assigned per distinct spelling "
          "actually found in\nthe traffic, so acmebank.cz and "
          "ACMEBANK.CZ get different numbers and each\nunmasks to the "
          "bytes that were really there.")
    print("matching is case-insensitive and word-boundary anchored, so a "
          "rule for\nacme.com also covers www.acme.com and "
          "internal.acme.com, and leaves notacme.com alone.")

    if args.against:
        from .stages.custom import apply_custom_masks
        try:
            with open(args.against, "r", encoding="utf-8",
                      errors="replace") as f:
                sample = f.read()
        except OSError as e:
            sys.stderr.write("  cannot read %s: %s\n" % (args.against, e))
            return 1
        masked, ents, mask_map = apply_custom_masks(sample, rules)
        fired = {}
        for e in ents:
            fired[e["text"]] = fired.get(e["text"], 0) + 1
        print("\nagainst %s:" % args.against)
        if not fired:
            print("  no rule matched. Either the file does not contain "
                  "these values,\n  or they are spelled differently "
                  "there.")
        for value, n in sorted(fired.items(), key=lambda kv: -kv[1]):
            print("  %5dx  %s" % (n, value))
        unused = [r["value"] for r in rules
                  if not any(r["value"].casefold() in k.casefold()
                             or k.casefold() in r["value"].casefold()
                             for k in fired)]
        if unused:
            print("  rules that never fired: %s" % ", ".join(unused[:10]))
    return 0


# --------------------------------------------------------------- proxy --

def cmd_proxy(rest):
    # Imported HERE, not at module scope. See the module docstring.
    from .proxy.server import main as proxy_main
    return proxy_main(rest)


# ---------------------------------------------------------------- main --

def _add_store_flags(parser):
    parser.add_argument("--state-dir", default=None,
                        help="where the mask maps live. Defaults to the same "
                             "place the proxy uses (MASKERADE_STATE_DIR, or "
                             "the per-OS state directory).")
    parser.add_argument("--retention-days", type=int, default=30,
                        help="retention window reported by `list` (default 30)")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="maskerade",
        description="Local anonymisation of pentest prompts and HTTP traffic "
                    "before they reach a third-party AI.")
    # Not argparse's `action="version"`: that prints one string and exits
    # from inside parse_args(), and what is wanted here is three lines
    # assembled from the package, including a probe for the extras.
    ap.add_argument("--version", action="store_true",
                    help="print the version, the default tier and which "
                         "optional extras are installed")

    sub = ap.add_subparsers(dest="command")

    sessions = sub.add_parser(
        "sessions",
        help="inspect and delete the stored mask maps",
        description="The mask map is client data. These commands report what "
                    "is stored and delete it; they never print the values.")
    ssub = sessions.add_subparsers(dest="sessions_command")

    p_list = ssub.add_parser(
        "list", help="id, label, last use and placeholder COUNT",
        description="Never prints mask-map contents: those are client "
                    "values, and a terminal is not where they belong.")
    _add_store_flags(p_list)
    p_list.set_defaults(func=cmd_list)

    p_purge = ssub.add_parser(
        "purge", help="delete stored maps (this is the retention policy)",
        description="Deletes session files and the index entries pointing at "
                    "them. Irreversible. Running it against a live proxy is "
                    "safe but not free: a conversation whose map is deleted "
                    "mints a new session and restarts numbering on its next "
                    "turn.")
    p_purge.add_argument("--all", action="store_true",
                         help="every session")
    p_purge.add_argument("--older-than", metavar="Nd", default=None,
                         help="sessions not used for N days, e.g. 30d")
    p_purge.add_argument("--session", metavar="ID", default=None,
                         help="one session, by the id `list` shows")
    p_purge.add_argument("--yes", action="store_true",
                         help="skip the confirmation prompt")
    _add_store_flags(p_purge)
    p_purge.set_defaults(func=cmd_purge)

    rules = sub.add_parser(
        "rules",
        help="check an operator rules file before trusting it",
        description="A rules file lists literal values that must never "
                    "reach the provider. These commands read it; they "
                    "never send anything anywhere.")
    rsub = rules.add_subparsers(dest="rules_command")
    p_check = rsub.add_parser(
        "check", help="parse a rules file and show what it would mask")
    p_check.add_argument("file")
    p_check.add_argument("--against", metavar="SAMPLE", default=None,
                         help="a text file to dry-run the rules over, to "
                              "see which ones actually fire")
    p_check.set_defaults(func=cmd_rules_check)

    # Imported here rather than at module scope, for the reason in the
    # module docstring. wizard.py pulls in proxy/rules.py (stdlib only) and
    # imports stages/custom.py lazily, inside its own dry-run, so
    # registering it costs nothing a `sessions list` has to pay.
    from .wizard import add_init_parser
    add_init_parser(sub)

    # Registered for `maskerade --help` only. main() intercepts `proxy`
    # before parsing and this parser is never actually run -- see the
    # module docstring on why REMAINDER cannot do the job.
    sub.add_parser(
        "proxy",
        help="run the localhost masking proxy "
             "(`maskerade proxy --help` for its options)",
        add_help=False)

    return ap, sessions, rules


def main(argv=None):
    # The console is cp1252 on the owner's machine and a redirected child
    # inherits it, so one non-ASCII byte from a dependency kills the
    # process. server.py does the same thing for the same reason.
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    argv = list(sys.argv[1:] if argv is None else argv)

    # Before argparse, deliberately. Everything after `proxy` belongs to the
    # server and must reach it unexamined.
    if argv and argv[0] == "proxy":
        try:
            return cmd_proxy(argv[1:])
        except StoreError as e:
            sys.stderr.write("\n  %s\n" % e)
            return 1
        except KeyboardInterrupt:
            return 130

    ap, sessions, rules = build_parser()
    args = ap.parse_args(argv)

    # Before the subcommand dispatch, so `maskerade --version` works on its
    # own rather than falling through to the "no command" help text.
    if getattr(args, "version", False):
        return cmd_version()

    if getattr(args, "func", None) is None:
        if args.command == "sessions":
            sessions.print_help()
        elif args.command == "rules":
            rules.print_help()
        else:
            ap.print_help()
        return 2

    try:
        return args.func(args)
    except StoreError as e:
        # The store refuses rather than guessing -- an unset LOCALAPPDATA, an
        # unsafe id, an unreadable directory. Its message names the fix.
        sys.stderr.write("\n  %s\n" % e)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
