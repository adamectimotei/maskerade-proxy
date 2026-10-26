# -*- coding: utf-8 -*-
"""`maskerade init --engagement NAME` -- the per-engagement setup wizard.

    maskerade init --engagement acme-q4
    maskerade init --engagement acme-q4 --client "Acme Bank a.s." \
                   --domain acmebank.cz --non-interactive

WHY THIS EXISTS, AND WHY IT IS NOT COSMETIC
-------------------------------------------
Two things converge in this command.

**The custom-rules layer is the only layer that can carry the client's own
name.** No detector knows it in advance, and org_evidence's precision-first
ORGANIZATION design explicitly assumes this layer is populated. Before this
command that meant a README step, which is a step people skip. The rules
file is now the thing that happens first.

**It is the mitigation for the fields the proxy forwards unscanned.** An
agentic client's working directory, home directory, git remote and branch are
forwarded to the provider UNINSPECTED, so an engagement run in
`C:\\clients\\acmebank\\...` puts the client's name in the clear. The
pre-flight below is the discharge of that promise -- it is where the
operator finds out, before the first request, which of those surfaces
carries the name they just typed.

WHAT IT WRITES, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------
Writes, in the engagement directory:

    .maskerade/rules.txt            the custom-rules file (rules.py's format)
    .claude/settings.local.json     merged, never overwritten, .bak first
    .gitignore                      an entry for .maskerade/, if this is a repo

Does NOT write:

  - **Any credential.** Never an API key, never a token, in any file. The
    competitor this command is modelled on writes provider keys into
    settings.local.json; that is not copied. Credentials stay in the
    environment.
  - **A CLAUDE.md note.** maskerade/proxy/system_note.txt ALREADY tells the
    model that bracketed tokens stand for withheld values, that they are
    stable across the conversation, that [USER_ID_1] and [USER_ID_2] are
    different accounts, and that it must not ask for the real values. It is
    appended to every request and disclosed in X-Maskerade-System-Note. A
    CLAUDE.md would be a second copy of that contract with no mechanism
    keeping the two in step, and the copy in the repo is the one that wins
    silently when they drift. `init` prints one line naming the header
    instead.
  - **A marker file.** One more file carrying the engagement name, for a
    convenience the `--engagement` flag already provides explicitly.

IT WRITES NOTHING IT HAS NOT PARSED BACK
----------------------------------------
After writing the rules file it is loaded through `RulesFile` and the parsed
count and types are reported, and then dry-run through the real
`apply_custom_masks` over a sample synthesised from the operator's own
entries. The format has already produced one silent bug -- a UTF-8 BOM from
`Set-Content -Encoding utf8` meant the operator's first rule never fired --
and a wizard that writes a file it has not parsed is the same class of
mistake. Everything this module writes goes through
`_write_text()`, which writes UTF-8 with NO BOM.

THREE CONSTRAINTS THE PARSER PUTS ON THIS FILE
----------------------------------------------
`rules.py` RAISES rather than skipping a bad line, and the daemon's first
load is allowed to propagate -- so a rules file this wizard writes badly is
a proxy that refuses to start. Hence, before writing:

  1. **Case-insensitive de-duplication.** `acmebank.cz` and `ACMEBANK.CZ` in
     one file is a RulesError, not an ignored line.
  2. **Drop, with a warning, anything the parser would refuse**: a value
     under MIN_VALUE_LEN, or one containing a square bracket.
  3. **Drop a subdomain of a domain already given.** custom.py's documented
     behaviour: the word-boundary anchors only bite on word characters and
     `.` is not one, so one apex rule already covers every subdomain.
     `portal.acmebank.cz` alongside `acmebank.cz` is not an error -- it is
     two placeholders for one host, which spends the "same entity, same
     placeholder" property for nothing.
"""
import json
import os
import re
import shutil
import sys
import time

from .proxy.rules import MAX_RULES, MIN_VALUE_LEN, RulesError, load_rules

# Where `init` puts the rules file, and the conventional location the proxy
# looks in when no --custom-rules is given. Project-local and inside the
# engagement directory: it is the file an operator edits at 11pm when the
# engagement turns up a new hostname, so discoverability is the property
# that matters most. The proxy's resolution order is in server.py.
RULES_DIRNAME = ".maskerade"
RULES_BASENAME = "rules.txt"

CLAUDE_DIRNAME = ".claude"
SETTINGS_BASENAME = "settings.local.json"

DEFAULT_PORT = 8787

# Placeholder types. Both are plain [A-Z_]+ so rules.py normalises them
# unchanged, and the index is appended per distinct matched spelling by
# stages/custom.py at request time -- which is why neither carries a number.
TYPE_CLIENT = "CLIENT_NAME"
TYPE_DOMAIN = "CLIENT_DOMAIN"


# ───────────────────────────────────────────────────────── small helpers ──

def _write_text(path, text):
    """UTF-8, no BOM, LF-normalised, parent directories created.

    No BOM is not a style choice. PowerShell 5.1's `Set-Content -Encoding
    utf8` writes one, rules.py strips one defensively, and the bug it was
    written to defend against -- the operator's first rule silently never
    firing -- is exactly what this function exists to not reproduce.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None
    except UnicodeDecodeError:
        return None


def _backup(path):
    """Copy `path` to `path.bak` before it is modified. Returns the backup."""
    bak = path + ".bak"
    shutil.copy2(path, bak)
    return bak


def _say(msg=""):
    sys.stdout.write(msg + "\n")


def _norm_path(p):
    return os.path.abspath(p).replace("\\", "/").lower()


# ────────────────────────────────────────────────────── entry normalising ──

class Entry(object):
    """One accepted rules-file record: a literal value and its type."""

    __slots__ = ("value", "etype")

    def __init__(self, value, etype):
        self.value = value
        self.etype = etype


def normalise_entries(clients, domains):
    """Turn raw operator input into records `RulesFile` is guaranteed to take.

    Returns (entries, dropped), where `dropped` is a list of
    (value, reason) the caller must print. Nothing is dropped silently:
    a rule the operator believes is in force and is not is the one failure
    this whole file exists to prevent.
    """
    entries, dropped = [], []
    seen = {}                       # casefolded value -> the value kept

    def consider(raw, etype):
        value = (raw or "").strip()
        if not value:
            return
        if len(value) < MIN_VALUE_LEN:
            dropped.append((value, "shorter than %d characters; the masker's "
                                   "length floor would drop it, so the rule "
                                   "would never fire" % MIN_VALUE_LEN))
            return
        if "[" in value or "]" in value:
            dropped.append((value, "contains a square bracket; brackets "
                                   "delimit placeholders, so this value "
                                   "cannot be substituted safely"))
            return
        key = value.casefold()
        if key in seen:
            dropped.append((value, "already given as %r; two rules for one "
                                   "value would make the file unparseable "
                                   "and the proxy would refuse to start"
                            % seen[key]))
            return
        seen[key] = value
        entries.append(Entry(value, etype))

    # Domains first, so that the apex-subsumption check below has the apex
    # list complete before any subdomain is considered, whatever order the
    # operator passed the flags in.
    for d in domains or []:
        consider(d, TYPE_DOMAIN)
    for c in clients or []:
        consider(c, TYPE_CLIENT)

    entries, subsumed = _drop_subsumed_domains(entries)
    dropped.extend(subsumed)
    if len(entries) > MAX_RULES:
        dropped.extend((e.value, "over the %d-rule cap" % MAX_RULES)
                       for e in entries[MAX_RULES:])
        entries = entries[:MAX_RULES]
    return entries, dropped


def _drop_subsumed_domains(entries):
    """Remove a domain that an apex already in the list covers.

    custom.py: "An apex rule covers the whole estate ... Do not add
    per-subdomain rules; they are redundant and multiply placeholders for
    one host." This is that sentence, enforced, because the operator typing
    both is the expected case rather than the exceptional one.
    """
    apexes = [e.value.casefold().lstrip(".")
              for e in entries if e.etype == TYPE_DOMAIN]
    kept, dropped = [], []
    for e in entries:
        if e.etype != TYPE_DOMAIN:
            kept.append(e)
            continue
        low = e.value.casefold().lstrip(".")
        parent = next((a for a in apexes
                       if a != low and low.endswith("." + a)), None)
        if parent:
            dropped.append((e.value,
                            "already covered by %s -- the word-boundary "
                            "anchors do not bite on '.', so one apex rule "
                            "masks every subdomain. A second rule here "
                            "would be two placeholders for one host"
                            % parent))
            continue
        kept.append(e)
    return kept, dropped


def render_rules_text(entries, engagement):
    """The rules file, in rules.py's format. ASCII only, no BOM, LF.

    Deliberately written as `value = [TYPE]` rather than bare lines even for
    the default type: the file is edited by hand mid-engagement, and a
    reader who can see the type on every line can add one without first
    going to find out what the default is.
    """
    out = [
        "# maskerade custom rules -- engagement %s" % engagement,
        "# written by `maskerade init` on %s"
        % time.strftime("%Y-%m-%d %H:%M"),
        "#",
        "# One value per line. Blank lines and lines starting with # are",
        "# ignored. `value = [TYPE]` sets the placeholder type; the number",
        "# in [TYPE_n] is allocated per distinct spelling found in the",
        "# traffic, so it is not written here.",
        "#",
        "# Literal strings only -- no regexes, no wildcards, no CIDR. An",
        "# apex domain already covers every subdomain.",
        "#",
        "# Check it after editing:  maskerade rules check %s"
        % os.path.join(RULES_DIRNAME, RULES_BASENAME).replace("\\", "/"),
        "",
    ]
    for etype, heading in ((TYPE_CLIENT, "client names"),
                           (TYPE_DOMAIN, "client domains")):
        group = [e for e in entries if e.etype == etype]
        if not group:
            continue
        out.append("# %s" % heading)
        for e in group:
            out.append("%s = [%s]" % (e.value, etype))
        out.append("")
    return "\n".join(out)


# ──────────────────────────────────────────────────────── leak pre-flight ──
#
# The part no competitor has. Everything checked here is a surface the
# proxy forwards UNSCANNED, or one the tool never sees at all. The
# competitor this command is modelled on checks the folder name alone.
# Checking the rest is cheap and is directly motivated by the decision to
# forward those fields rather than refuse them.
#
# It WARNS. It never refuses. The operator may have good reasons -- a
# directory they cannot rename, a branch the client's own process named --
# and this tool does not override its user.

def _git_dir(start):
    """The .git directory for `start`, following a worktree's gitdir file.

    Read off disk rather than shelled out to `git`: the operator may not
    have git on PATH, and a subprocess that is not there is a crash where a
    missing file is an answer.
    """
    d = os.path.abspath(start)
    while True:
        candidate = os.path.join(d, ".git")
        if os.path.isdir(candidate):
            return candidate
        if os.path.isfile(candidate):
            text = _read_text(candidate) or ""
            m = re.match(r"\s*gitdir:\s*(.+)\s*$", text)
            if m:
                p = m.group(1).strip()
                if not os.path.isabs(p):
                    p = os.path.join(d, p)
                return os.path.abspath(p)
            return None
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _git_facts(cwd):
    """(remotes, branch). Either may be empty; this is best-effort."""
    gd = _git_dir(cwd)
    if not gd:
        return [], None
    remotes = []
    config = _read_text(os.path.join(gd, "config")) or ""
    for m in re.finditer(r"^\s*url\s*=\s*(\S+)\s*$", config, re.M):
        remotes.append(m.group(1))
    branch = None
    head = _read_text(os.path.join(gd, "HEAD")) or ""
    m = re.match(r"\s*ref:\s*refs/heads/(.+?)\s*$", head)
    if m:
        branch = m.group(1)
    return remotes, branch


def _needles(entries):
    """What to look for: each value, plus the non-trivial words inside it.

    `Acme Bank a.s.` will never appear in a directory name, but `acmebank`
    and `acme` will. Word parts under MIN_VALUE_LEN are dropped: `a` and
    `s` from a legal form would match everything.
    """
    out = {}
    for e in entries:
        for part in [e.value] + re.split(r"[^0-9A-Za-z]+", e.value):
            p = (part or "").strip().casefold()
            if len(p) >= MIN_VALUE_LEN:
                out.setdefault(p, e.value)
    # A domain's registrable label matters more than the whole domain:
    # a directory is named `acmebank`, not `acmebank.cz`.
    for e in entries:
        if e.etype == TYPE_DOMAIN:
            label = e.value.casefold().lstrip(".").split(".")[0]
            if len(label) >= MIN_VALUE_LEN:
                out.setdefault(label, e.value)
    return out


def _hits(haystack, needles):
    if not haystack:
        return []
    low = str(haystack).casefold()
    return sorted({src for n, src in needles.items() if n in low})


def preflight(entries, cwd, engagement, settings_path):
    """Return a list of (surface, detail, why, what_to_do).

    Each row names the surface, WHY it is visible, and what to do about it.
    "Why" is the load-bearing column: an operator who is told only that
    their folder name matches learns nothing they can act on.
    """
    needles = _needles(entries)
    rows = []

    def add(surface, detail, why, fix, found):
        if found:
            rows.append((surface, detail, why, fix,
                         ", ".join(found)))

    cwd = os.path.abspath(cwd)
    add("working directory", cwd,
        "an agentic client puts its working directory in the system prompt, "
        "in its own environment block and in `safeguards`. All three are "
        "forwarded to the provider UNINSPECTED.",
        "work from a neutral path, or rename the directory. "
        "`C:\\work\\eng-2026-014` leaks nothing.",
        _hits(cwd, needles))

    home = os.path.expanduser("~")
    add("home directory", home,
        "it travels with every absolute path in the same unscanned fields, "
        "and it carries the operator's account name as well.",
        "nothing to do about the username; it is named here so it is not a "
        "surprise in a capture. If the client's name is in it, that is "
        "worth moving.",
        _hits(home, needles))

    remotes, branch = _git_facts(cwd)
    for url in remotes:
        add("git remote", url,
            "the client's environment block carries git remote URLs, branch "
            "and working-tree status, forwarded uninspected.",
            "if the repository is the client's, run the engagement from a "
            "working copy with no remote, or from a directory that is not a "
            "repository at all.",
            _hits(url, needles))
    add("git branch", branch or "",
        "same field. Branch names carry ticket ids and client names.",
        "work on a neutrally-named branch for the duration.",
        _hits(branch, needles))

    add("engagement label", engagement,
        "the label is a FILE NAME in the state directory and is printed by "
        "`maskerade sessions list`. It never leaves the machine -- it is not "
        "sent to the provider -- but it is on disk and on any terminal where "
        "that command is run.",
        "use a neutral label: `eng-2026-014` rather than the client's name.",
        _hits(engagement, needles))

    for p in (settings_path,
              os.path.join(os.path.dirname(settings_path), "settings.json")):
        text = _read_text(p)
        if text is None:
            continue
        add("existing %s" % os.path.basename(p), p,
            "permission rules in a client's settings are exactly where a "
            "pentester writes in-scope hostnames and URLs, and that whole "
            "block is forwarded uninspected.",
            "read the file and move client identifiers out of it, or accept "
            "that they reach the provider.",
            _hits(text, needles))

    return rows


# ───────────────────────────────────────────── .claude/settings.local.json ──
#
# The riskiest thing this command does, so it does the least that works.
#
# VERIFIED against the Claude Code documentation on 2026-10-18:
#
#   `env`                                   documented top-level settings key,
#                                           a flat map of name -> string.
#   ANTHROPIC_BASE_URL                      documented. Also documented, and
#                                           printed by this command because an
#                                           operator should not discover it
#                                           later: pointing it at a
#                                           non-first-party host disables MCP
#                                           tool search by default and
#                                           disables Remote Control.
#   CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC  documented. Reads only WHETHER it
#                                           is set: `0` and `false` still
#                                           disable the traffic, so it is
#                                           written as "1" and removed rather
#                                           than zeroed to re-enable.
#
# NOT WRITTEN: a proposed third key, a "web-fetch pre-flight setting".
# There is no such documented key. The nearest real one is
# CLAUDE_CODE_DISABLE_WEB_FETCH, which turns the WebFetch TOOL off -- a
# change to what the agent can do, not a redirection of traffic through this
# proxy. Guessing at a schema is how a user's client config gets broken, so
# it is left out and the manual note below is printed instead.

SETTINGS_ENV_KEYS = ("ANTHROPIC_BASE_URL",
                     "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")


def desired_env(host, port):
    return {
        "ANTHROPIC_BASE_URL": "http://%s:%d" % (host, port),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


def plan_settings(path, host, port):
    """Decide what would change in settings.local.json. Writes nothing.

    Returns (new_text_or_None, changes, problem). `new_text_or_None` is
    None when the file is already correct. `problem` is set when the file
    exists and cannot be merged safely, in which case NOTHING is written
    and the caller prints the manual instruction: breaking an operator's
    client configuration is worse than not writing it.
    """
    want = desired_env(host, port)
    raw = _read_text(path)

    if raw is None and os.path.exists(path):
        return None, [], ("%s exists but is not readable UTF-8. Leaving it "
                          "alone." % path)

    if raw is None:
        doc = {"env": dict(want)}
        return (json.dumps(doc, indent=2, ensure_ascii=True) + "\n",
                [("create", k, None, v) for k, v in sorted(want.items())],
                None)

    try:
        doc = json.loads(raw)
    except ValueError as e:
        return None, [], ("%s is not valid JSON (%s). Refusing to touch it: "
                          "a merge would have to guess what the operator "
                          "meant." % (path, e))
    if not isinstance(doc, dict):
        return None, [], ("%s does not contain a JSON object. Leaving it "
                          "alone." % path)
    env = doc.get("env", {})
    if not isinstance(env, dict):
        return None, [], ("%s has an `env` that is not an object. Leaving it "
                          "alone." % path)

    changes = []
    new_env = dict(env)
    for k, v in sorted(want.items()):
        old = env.get(k)
        if old == v:
            continue
        new_env[k] = v
        changes.append(("set" if old is None else "replace", k, old, v))
    if not changes:
        return None, [], None

    # Merge, never overwrite: every other key in the document and every
    # other variable in `env` is carried through untouched.
    doc = dict(doc)
    doc["env"] = new_env
    return json.dumps(doc, indent=2, ensure_ascii=True) + "\n", changes, None


# ───────────────────────────────────────────────────────────── .gitignore ──

def plan_gitignore(cwd):
    """An entry for the rules directory, if this is a repository.

    The rules file holds the client's name, domains and tenant slugs. If the
    operator runs this inside any repository -- the client's own, most
    dangerously -- it must not end up in a commit.
    """
    if not _git_dir(cwd):
        return None, None
    path = os.path.join(cwd, ".gitignore")
    entry = "%s/" % RULES_DIRNAME
    text = _read_text(path)
    if text is None and os.path.exists(path):
        return None, "%s is not readable; add `%s` to it by hand." % (
            path, entry)
    existing = text or ""
    if entry in [ln.strip() for ln in existing.splitlines()]:
        return None, None
    block = ""
    if existing and not existing.endswith("\n"):
        block += "\n"
    block += ("\n# maskerade: the custom-rules file holds the client's own\n"
              "# names and domains. It must never be committed.\n%s\n"
              % entry)
    return existing + block, None


# ──────────────────────────────────────────────────────────── the command ──

def _prompt_list(what, example):
    _say("")
    _say("  %s, one per line. Empty line when done." % what)
    _say("  example: %s" % example)
    out = []
    while True:
        try:
            line = input("    > ").strip()
        except (EOFError, KeyboardInterrupt):
            _say("")
            break
        if not line:
            break
        out.append(line)
    return out


def cmd_init(args):
    cwd = os.path.abspath(args.directory or os.getcwd())
    engagement = args.engagement
    rules_path = os.path.join(cwd, RULES_DIRNAME, RULES_BASENAME)
    settings_path = os.path.join(cwd, CLAUDE_DIRNAME, SETTINGS_BASENAME)

    interactive = not args.non_interactive
    if interactive and not sys.stdin.isatty():
        # Same convention as `sessions purge`: a non-tty is not a terminal to
        # prompt at, and silently proceeding with no rules would produce an
        # engagement whose client name no layer carries.
        sys.stderr.write(
            "  stdin is not a terminal, so there is nothing to prompt. Pass "
            "--non-interactive with --client/--domain, or run this from a "
            "terminal.\n")
        return 2

    clients = list(args.client or [])
    domains = list(args.domain or [])

    _say("")
    _say("  maskerade init -- engagement %s" % engagement)
    _say("    directory   %s" % cwd)

    if interactive:
        _say("")
        _say("  The client's own name and domains are the ONE thing no "
             "detector knows in")
        _say("  advance. This file is the only layer that carries them.")
        clients += _prompt_list("Client names as they appear in traffic",
                                "Acme Bank a.s.")
        domains += _prompt_list("Client domains (apex only -- subdomains are "
                                "covered)", "acmebank.cz")

    entries, dropped = normalise_entries(clients, domains)

    if dropped:
        _say("")
        _say("  NOT written:")
        for value, why in dropped:
            _say("    %-28s %s" % (value[:28], why))

    if not entries:
        _say("")
        _say("  No usable values, so no rules file is written. Without one "
             "the client's own")
        _say("  name is left to the detectors, and org_evidence's "
             "ORGANIZATION arm is")
        _say("  precision-first precisely because it assumes this layer is "
             "populated.")
        _say("  Re-run with --client / --domain.")
        return 1

    # ── the pre-flight, BEFORE anything is written ──

    rows = preflight(entries, cwd, engagement, settings_path)
    _say("")
    if rows:
        _say("  LEAK PRE-FLIGHT -- %d surface(s) carry a value you just "
             "entered." % len(rows))
        _say("  These are forwarded to the provider WITHOUT being inspected. "
             "This is a")
        _say("  warning, not a refusal: nothing below stops the proxy "
             "running.")
        for surface, detail, why, fix, found in rows:
            _say("")
            _say("    [%s] %s" % (surface, detail))
            _say("      matches  %s" % found)
            _say("      why      %s" % why)
            _say("      do       %s" % fix)
    else:
        _say("  LEAK PRE-FLIGHT -- clean. No checked surface carries these "
             "values.")
        _say("  Checked: working directory, home directory, git remote, git "
             "branch,")
        _say("  engagement label, existing .claude settings.")

    # ── what would change ──

    rules_text = render_rules_text(entries, engagement)
    rules_old = _read_text(rules_path)
    rules_change = None
    if rules_old is None and os.path.exists(rules_path):
        _say("")
        _say("  %s is not readable UTF-8. Refusing to overwrite it."
             % rules_path)
        return 1
    if rules_old is None:
        rules_change = "create"
    elif _body(rules_old) != _body(rules_text):
        rules_change = "rewrite"

    settings_text, settings_changes, settings_problem = plan_settings(
        settings_path, args.host, args.port)
    gitignore_text, gitignore_problem = plan_gitignore(cwd)

    planned = []
    if rules_change:
        planned.append("%-8s %s  (%d rule(s))"
                       % (rules_change, rules_path, len(entries)))
    if settings_text:
        verb = "create" if not os.path.exists(settings_path) else "merge"
        planned.append("%-8s %s" % (verb, settings_path))
        for kind, key, old, new in settings_changes:
            if kind == "replace":
                planned.append("         %s: %r -> %r" % (key, old, new))
            else:
                planned.append("         %s = %r" % (key, new))
    if gitignore_text:
        planned.append("%-8s %s  (+ %s/)"
                       % ("append", os.path.join(cwd, ".gitignore"),
                          RULES_DIRNAME))

    _say("")
    if not planned:
        _say("  Nothing to change. Everything this command writes is already "
             "in place:")
        _say("    %s" % rules_path)
        if os.path.exists(settings_path):
            _say("    %s" % settings_path)
        _verify(rules_path, entries)
        _closing(engagement, rules_path, args)
        return 0

    _say("  About to change:")
    for line in planned:
        _say("    %s" % line)
    if rules_change == "rewrite":
        _say("")
        _say("    the existing rules file differs from what these values "
             "produce. It is")
        _say("    backed up to %s.bak first." % RULES_BASENAME)

    if settings_problem:
        _say("")
        _say("  NOT touching the client settings: %s" % settings_problem)
        _say("  Set these yourself instead:")
        for k, v in sorted(desired_env(args.host, args.port).items()):
            _say("    %s=%s" % (k, v))
    if gitignore_problem:
        _say("")
        _say("  %s" % gitignore_problem)

    if interactive and not args.yes:
        _say("")
        try:
            answer = input("  type 'yes' to write: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer != "yes":
            _say("  nothing written.")
            return 1

    # ── write ──

    _say("")
    if rules_change:
        if rules_old is not None:
            _say("  backed up  %s" % _backup(rules_path))
        _write_text(rules_path, rules_text)
        _say("  wrote      %s" % rules_path)
    if settings_text:
        if os.path.exists(settings_path):
            _say("  backed up  %s" % _backup(settings_path))
        _write_text(settings_path, settings_text)
        _say("  wrote      %s" % settings_path)
    if gitignore_text:
        gi = os.path.join(cwd, ".gitignore")
        if os.path.exists(gi):
            _say("  backed up  %s" % _backup(gi))
        _write_text(gi, gitignore_text)
        _say("  wrote      %s" % gi)

    if not _verify(rules_path, entries):
        return 1
    _closing(engagement, rules_path, args)
    return 0


def _body(text):
    """A rules file's meaning: its records, without comments or blanks.

    Idempotency has to be judged on this rather than on the bytes, because
    the header carries a timestamp -- comparing bytes would make every run
    a rewrite, which is exactly the "running it twice changes something"
    failure this command must not have.
    """
    return [ln.strip() for ln in (text or "").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def _verify(rules_path, entries):
    """Parse the file back, then dry-run it. Reports; returns ok.

    Two separate claims, and the second is the one that cannot be made by
    reading the parser:

      1. `RulesFile` accepts the file -- which is what the daemon does at
         boot, where a failure means it refuses to start.
      2. the rules actually FIRE, through the real `apply_custom_masks`,
         over a sample built from the operator's own entries.

    (2) catches the case the code alone does not settle: `_build_pattern`
    anchors only on an edge that is a word character, so a client name
    ending in a dotted legal form -- `Acme Bank a.s.` -- has no trailing
    anchor. It matches, and it also matches inside `Acme Bank a.s.r.o.`.
    That is over-masking, the accepted direction, but it is reported rather
    than discovered.
    """
    _say("")
    try:
        parsed = load_rules(rules_path)
    except RulesError as e:
        _say("  PARSE FAILED. The file this command just wrote is not one "
             "the proxy would")
        _say("  accept, and `maskerade proxy` would refuse to start:")
        _say("    %s" % e)
        return False

    types = {}
    for r in parsed:
        types[r["placeholder_type"]] = types.get(r["placeholder_type"], 0) + 1
    _say("  parsed back %d rule(s): %s"
         % (len(parsed),
            ", ".join("%s=%d" % kv for kv in sorted(types.items()))))
    if len(parsed) != len(entries):
        _say("  !! %d written, %d parsed. Do not trust this file; run "
             "`maskerade rules check`." % (len(entries), len(parsed)))
        return False

    try:
        from .stages.custom import apply_custom_masks
    except Exception as e:
        _say("  could not dry-run the rules (%s: %s). The file parses; "
             "check it with `maskerade rules check FILE --against SAMPLE`."
             % (type(e).__name__, e))
        return True

    # A sample built from the operator's own values, in the spellings that
    # actually turn up in traffic: as given, lowercased, and as a host.
    lines = []
    for e in entries:
        lines.append("value: %s" % e.value)
        lines.append("lower: %s" % e.value.lower())
        if e.etype == TYPE_DOMAIN:
            lines.append("Host: portal.%s" % e.value)
    sample = "\n".join(lines)
    masked, ents, mask_map = apply_custom_masks(sample, parsed)
    fired = sorted({en["text"] for en in ents})
    _say("  dry run     %d substitution(s), %d placeholder(s)"
         % (len(ents), len(mask_map)))
    # The one property of these rules that reading rules.py does not settle,
    # because it lives in stages/custom.py's _build_pattern: the
    # word-boundary anchors are applied only on an edge that IS a word
    # character. A value ending in a dotted legal form has no trailing
    # anchor. Verified 2026-10-18: a rule for `Acme Bank a.s.` masks
    # `Acme Bank a.s.` and also the first 14 characters of
    # `Acme Bank a.s.r.o.`, leaving `r.o.` behind. Over-masking is the
    # accepted direction of failure, and it is not a leak -- but an
    # operator who sees `[CLIENT_NAME_1]r.o.` in a transcript should
    # recognise it rather than file it as a bug.
    loose = sorted({e.value for e in entries
                    if e.value and not re.match(r"\w", e.value[-1])})
    if loose:
        _say("")
        _say("  Worth knowing rather than discovering: %s"
             % ", ".join(repr(v) for v in loose))
        _say("  do not end in a letter or digit, so there is no trailing word")
        _say("  boundary. Each fires as written AND at the start of a longer")
        _say("  string -- a rule for 'Acme Bank a.s.' also masks the first")
        _say("  part of 'Acme Bank a.s.r.o.', leaving 'r.o.' behind. That is")
        _say("  over-masking, the direction this tool fails in deliberately.")
        _say("  It is not a leak. Drop the trailing punctuation if you would")
        _say("  rather have the tighter rule.")

    never = [r["value"] for r in parsed
             if not any(r["value"].casefold() in f.casefold() for f in fired)]
    if never:
        _say("  !! these rules did not fire on a sample built from their own "
             "values: %s" % ", ".join(never))
        _say("     That should not happen. Report it rather than working "
             "around it.")
        return False
    return True


def _proxy_tier_flag():
    """` --tier full` when this install can run the full tier, else "".

    find_spec only (maskerade._extras), so asking costs a light install
    nothing. Any failure to answer means "", the default tier: a command
    that runs beats one that names a tier this install may not have.
    """
    try:
        from ._extras import ner_available
        return " --tier full" if ner_available() else ""
    except Exception:
        return ""


def _quote(path):
    """Quote a path for a command line the operator will copy."""
    return '"%s"' % path if any(c.isspace() for c in path) else path


def _closing(engagement, rules_path, args):
    # The ABSOLUTE path, and the `cd` with it. This printed a path relative
    # to the CURRENT directory until 2026-10-19, which was wrong twice over
    # when --directory pointed somewhere else: the path itself came out as
    # `../../<USER>~1/AppData/...`, and the sentence implied the file is
    # found wherever the proxy runs. It is not. server.py's
    # _resolve_rules_path looks for .maskerade/rules.txt under os.getcwd(),
    # so the proxy has to be STARTED FROM the engagement directory or told
    # where the file is. An operator who starts it elsewhere gets no custom
    # rules at all, and the only signal is one word on the banner.
    eng_dir = os.path.dirname(os.path.dirname(os.path.abspath(rules_path)))
    tier = _proxy_tier_flag()
    _say("")
    _say("  Start the proxy for this engagement, FROM this directory:")
    _say("    cd %s" % _quote(eng_dir))
    _say("    maskerade proxy --engagement %s%s" % (engagement, tier))
    if not tier:
        # Printed instead of `--tier full`, which used to be unconditional:
        # on a light install that command fails closed at warmup, so the
        # first thing a new operator copied out of this wizard did not run.
        _say("")
        _say("    That is the fast tier, the only one this install can run. "
             "The full tier")
        _say("    adds a local model for names and organisations; see "
             "\"Full tier\" in the")
        _say("    README, then add --tier full.")
    _say("")
    _say("    The proxy looks for %s in the directory it is started from, "
         "so" % os.path.join(RULES_DIRNAME, RULES_BASENAME))
    _say("    that `cd` is load-bearing. To run it from anywhere else, name "
         "the file:")
    _say("")
    _say("      maskerade proxy --engagement %s%s --custom-rules "
         "\"%s\"" % (engagement, tier, rules_path))
    _say("")
    _say("    Either way the banner's `custom rule` line says which file won. "
         "If it")
    _say("    says `none`, the client's own name is reaching the provider "
         "with only")
    _say("    the detectors between it and them.")
    _say("")
    _say("    Edit the rules file any time -- the proxy re-reads it without a "
         "restart,")
    _say("    and keeps the last good list if an edit does not parse.")
    _say("")
    _say("  Delete everything from this engagement, later:")
    _say("    maskerade sessions purge --session %s" % engagement)
    _say("")
    _say("  Two things this command did NOT do, deliberately:")
    _say("")
    _say("    No credential was written to any file. ANTHROPIC_API_KEY and "
         "any token")
    _say("    stay in your environment. Nothing this command writes needs "
         "one.")
    _say("")
    _say("    No CLAUDE.md was written. The proxy already appends the "
         "redaction")
    _say("    contract to every request -- what placeholders mean, that they "
         "are")
    _say("    stable, that [USER_ID_1] and [USER_ID_2] are different "
         "accounts -- and")
    _say("    says so in the X-Maskerade-System-Note response header. A "
         "second copy")
    _say("    in the repo would be one more thing to keep in step.")
    _say("")
    _say("  What pointing a client here does NOT cover:")
    _say("")
    _say("    A client makes its own requests outside this proxy. Version "
         "checks,")
    _say("    telemetry and release-note fetches do not go through "
         "ANTHROPIC_BASE_URL at")
    _say("    all; CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC turns most of "
         "them off, and")
    _say("    that variable reads only WHETHER it is set -- '0' still "
         "disables them.")
    _say("    Claude Code also disables MCP tool search and Remote Control "
         "while the")
    _say("    base URL is not api.anthropic.com, and may stop fetching the "
         "policy its")
    _say("    vendor would otherwise apply to it.")
    _say("    'Only masked text leaves the machine' is a claim about this "
         "proxy's")
    _say("    egress, not about the machine's.")
    _say("")


def add_init_parser(sub):
    """Register `init`. Called from cli.build_parser()."""
    p = sub.add_parser(
        "init",
        help="set up an engagement: custom rules, client config, leak "
             "pre-flight",
        description="Writes the custom-rules file that carries the client's "
                    "own name, points a client at the proxy, and warns "
                    "which surfaces already carry that name in the clear. "
                    "Idempotent: it says what it will change and running it "
                    "twice changes nothing.")
    p.add_argument("--engagement", required=True, metavar="NAME",
                   help="the engagement label. Also the session name, so "
                        "`sessions purge --session NAME` deletes exactly "
                        "this engagement. Use a neutral one.")
    p.add_argument("--client", action="append", metavar="NAME", default=None,
                   help="a client name as it appears in traffic. Repeatable.")
    p.add_argument("--domain", action="append", metavar="DOMAIN", default=None,
                   help="a client domain, apex only -- subdomains are "
                        "already covered. Repeatable.")
    p.add_argument("--non-interactive", action="store_true",
                   help="never prompt. Takes the values from --client / "
                        "--domain and writes without confirming.")
    p.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt")
    p.add_argument("--directory", default=None, metavar="DIR",
                   help="the engagement directory (default: the current one)")
    p.add_argument("--host", default="127.0.0.1",
                   help="the host to point the client at (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help="the port to point the client at (default %d)"
                        % DEFAULT_PORT)
    p.set_defaults(func=cmd_init)
    return p
