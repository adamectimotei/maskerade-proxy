# -*- coding: utf-8 -*-
"""Where the mask map lives, and how a request finds the one it belongs to.

THE PROBLEM
-----------
Placeholder numbering is cumulative: apply_masks() takes an existing_mask_map
so that the same value keeps the same placeholder across turns. That is not a
cosmetic property. When the model is reasoning about an IDOR between two
accounts it has to be able to tell [USER_ID_1] from [USER_ID_2] and to know
that the [USER_ID_1] in turn 6 is the same account as the one in turn 2.
Renumbering per request destroys exactly that.

So the map has to outlive the request, which means it needs a key. A stateless
HTTP proxy has no conversation identity: one operator on one machine still runs
several conversations at once in several terminals.

THE KEY IS DERIVED, NOT SUPPLIED
--------------------------------
The Messages API is stateless, so the client resends the whole conversation
every turn. Turn 6's request therefore CONTAINS turn 5's request as a prefix.
That is the identity, and it is already in the request:

  on each request, for k from len(messages)-1 down to 1, hash (system,
  messages[:k]) and look it up. The longest hit names the session. Then record
  (system, messages) under this session so the next turn hits it.

Longest-first matters: a client that retries or branches a conversation sends a
shorter prefix, and the most specific match is the right parent.

Hashing what the client SENT, rather than predicting what it will send next,
is deliberate. The obvious alternative -- after replying, hash this request
plus the assistant message the client is about to append -- requires the proxy
to reproduce the client's serialisation of that message byte for byte,
including tool_use blocks. It would work until a client formatted one field
differently, and then it would fail by silently minting a new session, which
looks like nothing at all.

WHAT BREAKS IT, VISIBLY
-----------------------
A client that compacts or summarises its history changes the prefix, so no
hash matches and a new session is minted. Numbering restarts from there. That
is not a leak and not a correctness problem inside any one request, but it IS
the kind of thing that otherwise looks like a mystery, so resolve() reports
whether the session was continued or created and the server puts it on the
response as X-Maskerade-Session.

CONCURRENCY
-----------
One lock per session, in-process. That is sufficient rather than lazy: the
proxy owns a TCP port, so a second instance on the same machine cannot start,
which makes the port the mutex. Writes are still atomic (temp file + rename)
because a crash mid-write is a different failure from a concurrent one, and a
half-written map means a reply that cannot be unmasked.

READ FAILURE REFUSES
--------------------
A map that cannot be read is not an empty map. Returning masked text as though
it were the answer would have the operator read "[PERSON_NAME_1]" as the
model's output. load() raises; the caller turns that into a refusal.
"""
import errno
import hashlib
import json
import os
import re
import sys
import threading
import time


RETENTION_DAYS_DEFAULT = 30

# Session ids are ours, but they end up as path components, so they are
# validated rather than trusted -- an operator-supplied --label reaches this.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# How far back to look for a parent prefix. Bounded so a 400-turn conversation
# does not cost 400 hashes per request; the parent is almost always at k=n-1
# and the scan exists for branching and retries.
MAX_PREFIX_LOOKBACK = 24


class StoreError(RuntimeError):
    """The map could not be read or written. The caller must refuse."""


# ─────────────────────────────────────────────────────────────── paths ──

def default_root():
    """Per-OS state directory. Never a directory that syncs to a cloud.

    On Windows this is %LOCALAPPDATA%, NOT ~/Documents, which is
    OneDrive-synced by default on a consumer install -- a mask map there is
    client PII in Microsoft's cloud, which inverts the entire product. If
    LOCALAPPDATA is somehow unset we raise rather than falling back to the
    home directory, because the fallback is the failure.
    """
    env = os.environ.get("MASKERADE_STATE_DIR")
    if env:
        return os.path.abspath(env)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            raise StoreError(
                "LOCALAPPDATA is not set, so there is no non-syncing place to "
                "put the mask map. Set MASKERADE_STATE_DIR to a local path. "
                "Refusing to fall back to the home directory: ~/Documents is "
                "OneDrive-synced by default and a mask map there is client "
                "data in a third party's cloud.")
        return os.path.join(base, "maskerade")
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library",
                            "Application Support", "maskerade")
    base = (os.environ.get("XDG_STATE_HOME")
            or os.path.join(os.path.expanduser("~"), ".local", "state"))
    return os.path.join(base, "maskerade")


_SYNC_MARKERS = ("onedrive", "dropbox", "google drive", "googledrive",
                 "icloud", "nextcloud", "pcloud", "box sync", "mega")


def sync_warning(path):
    """Return a warning string if `path` looks like it is inside a sync root.

    Heuristic and deliberately shallow: it reads the path, not the
    filesystem. A false positive costs a printed line; a false negative is
    the OneDrive case above, which the default path already avoids. This
    exists for the operator who sets MASKERADE_STATE_DIR by hand.
    """
    low = os.path.abspath(path).replace("\\", "/").lower()
    for marker in _SYNC_MARKERS:
        if marker in low:
            return ("state directory %s looks like it is inside a %s folder. "
                    "The mask map is client data and must not leave this "
                    "machine. Set MASKERADE_STATE_DIR to a local path."
                    % (path, marker))
    return None


def _restrict(path, is_dir):
    """0600 / 0700 on POSIX; owner-only ACL on Windows.

    On Windows os.chmod cannot express an ACL, so icacls is shelled out to,
    and a failure there is returned as a warning string for the caller to
    report: a mask map with default permissions on a multi-user box is the
    shared-host risk, and the operator should find out at startup.

    On POSIX a failure REFUSES (StoreError), changed 2026-10-21. Measured
    as a non-root user on a volume root it does not own: os.chmod raised a
    bare PermissionError out of SessionStore.__init__, so `maskerade proxy`
    said "cannot start: PermissionError" with no fix and `maskerade sessions
    list` printed a traceback. The mode is then READ BACK, because a chmod
    the filesystem accepts and ignores -- a Windows-drive bind mount under
    Docker Desktop -- would otherwise leave the map at default permissions
    and say nothing. Refuse, not warn: the state directory is chosen once,
    and a map that cannot be restricted is a map that should not be written.
    """
    if sys.platform != "win32":
        want = 0o700 if is_dir else 0o600
        try:
            os.chmod(path, want)
        except OSError as e:
            raise StoreError(
                "cannot restrict %s to mode %o (%s). This process (uid %d) "
                "does not own it. In a container: the image must create the "
                "state directory owned by the app user, and it must be "
                "mounted as a NAMED volume, not a bind mount. Otherwise: set "
                "MASKERADE_STATE_DIR to a directory this user owns."
                % (path, want, e.strerror or e, os.getuid()))
        got = os.stat(path).st_mode & 0o777
        if got != want:
            raise StoreError(
                "%s is mode %o after chmod %o: the filesystem ignored the "
                "change, so the mask map would be readable beyond this user. "
                "This is typical of a Windows-drive bind mount under Docker "
                "Desktop. Use a named volume, or a MASKERADE_STATE_DIR on a "
                "native Linux filesystem." % (path, got, want))
        return None
    import subprocess
    user = os.environ.get("USERNAME")
    if not user:
        return "USERNAME is unset; could not restrict the ACL on %s" % path
    try:
        subprocess.run(
            ["icacls", path, "/inheritance:r",
             "/grant:r", "%s:(OI)(CI)F" % user if is_dir else "%s:F" % user],
            check=True, capture_output=True)
    except Exception as e:
        return "could not restrict the ACL on %s (%s)" % (path, e)
    return None


# ────────────────────────────────────────────────────── prefix hashing ──

def canonical(obj):
    """Stable bytes for a JSON fragment. sort_keys so key order cannot

    split one conversation into two sessions just because a client
    serialised a dict differently between turns.

    ensure_ascii=True since 2026-10-29. With ensure_ascii=False the result
    was encoded to UTF-8, and a LONE SURROGATE -- which json.loads accepts
    from a "\\ud83d" escape, and which a client produces when it truncates a
    tool output in the middle of an emoji -- raised UnicodeEncodeError.
    Measured on 0.1.4: the request was masked and answered, then persist()
    raised, the turn was never remembered, and every later turn of that
    conversation died in resolve_prefix() with a traceback and a dropped
    connection. Escaping makes every string encodable, and the session
    document stays plain JSON that any version reads back.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":")).encode("ascii")


# What the prefix hash ignores. Changed 2026-10-29, measured on a real
# Claude Code session of 114 messages: no turn ever matched the previous one.
#
# The cause is two things a client legitimately changes in messages it has
# already sent:
#
#   cache_control  Claude Code puts its prompt-cache breakpoint on the
#                  newest message and removes it from the previous one. The
#                  previous request's last message is therefore never
#                  byte-identical to the same message one turn later.
#   one text block it then re-sends that message's content
#                  [{"type": "text", "text": X}] as the plain string X.
#
# Both are invisible to masking: `cache_control` is BLOCK_STRUCTURAL in
# fields.py and never read as text, and extract() sends a bare string and a
# lone text block through the same _leaf() -- same text, same
# system-reminder rule. So the normalised form is equal exactly when the
# TEXT extract() would scan is equal.
#
# What the match is used for changed on 2026-10-30. It was used to mask the
# matched messages from the stored map without detection, and that turned
# out not to reproduce what detection had produced (proxy/maskcache.py has
# the measurements). It now only decides which SESSION a request belongs
# to -- for an unlabelled conversation, the only thing that does.
#
# Deliberately NOT normalised: anything extract() reads. In particular the
# walk does not descend into tool_use `input`, which is arbitrary JSON and
# may legitimately contain a key named "cache_control".

def _norm_content(val):
    """Message content, tool_result content or `system`: a string or blocks."""
    if not isinstance(val, list):
        return val
    out = [_norm_block(b) for b in val]
    if len(out) == 1:
        only = out[0]
        if (isinstance(only, dict) and set(only) == {"type", "text"}
                and only["type"] == "text" and isinstance(only["text"], str)):
            return only["text"]
    return out


def _norm_block(blk):
    if not isinstance(blk, dict):
        return blk
    out = {k: v for k, v in blk.items() if k != "cache_control"}
    if out.get("type") == "tool_result" and "content" in out:
        out["content"] = _norm_content(out["content"])
    return out


def _norm_message(msg):
    if not isinstance(msg, dict) or "content" not in msg:
        return msg
    out = dict(msg)
    out["content"] = _norm_content(msg["content"])
    return out


def _all_prefix_hashes(system, messages):
    """[h0, h1, ..., hn] where hk = prefix_hash(system, messages[:k]).

    One serialisation of the conversation instead of one per prefix. The
    bytes hashed are exactly canonical({"messages": [...], "system": ...}):
    sort_keys puts "messages" first, so every prefix shares the leading
    bytes and differs only in where the array closes. sha256 objects are
    copied at each message boundary. The scan that used this per prefix
    re-serialised messages[:k] for every k -- quadratic in the conversation,
    and it runs twice per request.
    """
    msgs = list(messages or [])
    tail = b'],"system":' + canonical(_norm_content(system)) + b"}"
    h = hashlib.sha256(b'{"messages":[')
    out = []
    for i in range(len(msgs) + 1):
        if i:
            if i > 1:
                h.update(b",")
            h.update(canonical(_norm_message(msgs[i - 1])))
        c = h.copy()
        c.update(tail)
        out.append(c.hexdigest())
    return out


def prefix_hash(system, messages):
    """Identity of a conversation state. Computed over the RAW request.

    Raw, not masked: the client stores and resends real values, so the raw
    prefix is the stable thing. A masked prefix would depend on the map we
    are trying to look up, which is circular. Normalised as described
    above, which changes nothing extract() reads.
    """
    return _all_prefix_hashes(system, messages)[-1]


def prefix_hashes(system, messages, lookback=MAX_PREFIX_LOOKBACK):
    """Candidate parent hashes, longest prefix first."""
    hs = _all_prefix_hashes(system, messages)
    n = len(hs) - 1
    return [hs[k] for k in range(n - 1, 0, -1)][:lookback]


# ───────────────────────────────────────────────────────────── the store ──

class SessionStore:
    """Mask maps on disk, keyed by conversation prefix.

        <root>/sessions/<id>.json   the map itself
        <root>/index/<hash>         a pointer file holding a session id

    One small file per pointer rather than a central index: there is nothing
    to lock, a partial write affects one lookup instead of all of them, and
    purge is a directory sweep by mtime in both places.
    """

    def __init__(self, root=None, retention_days=RETENTION_DAYS_DEFAULT):
        self.root = os.path.abspath(root or default_root())
        self.retention_days = int(retention_days)
        self.sessions_dir = os.path.join(self.root, "sessions")
        self.index_dir = os.path.join(self.root, "index")
        self.warnings = []
        self._locks = {}
        self._locks_guard = threading.Lock()

        w = sync_warning(self.root)
        if w:
            self.warnings.append(w)
        for d in (self.root, self.sessions_dir, self.index_dir):
            try:
                os.makedirs(d, exist_ok=True)
            except OSError as e:
                raise StoreError("cannot create %s: %s" % (d, e))
            w = _restrict(d, is_dir=True)
            if w:
                self.warnings.append(w)

    # ── locking ──

    def _lock(self, session_id):
        with self._locks_guard:
            lk = self._locks.get(session_id)
            if lk is None:
                lk = self._locks[session_id] = threading.Lock()
            return lk

    # ── paths ──

    def _session_path(self, session_id):
        if not _SAFE_ID.match(session_id or ""):
            raise StoreError("refusing unsafe session id %r" % (session_id,))
        return os.path.join(self.sessions_dir, session_id + ".json")

    def _index_path(self, h):
        if not re.match(r"^[0-9a-f]{64}$", h or ""):
            raise StoreError("refusing unsafe index key %r" % (h,))
        return os.path.join(self.index_dir, h)

    # ── resolve ──

    def resolve(self, system, messages, label=None):
        """Find or mint the session this request belongs to.

        Returns (session_id, continued: bool). `label`, when given, overrides
        the derived key entirely -- it is the X-Maskerade-Session escape
        hatch, never required, and it is what makes "purge everything from
        engagement X" answerable.
        """
        if label:
            if not _SAFE_ID.match(label):
                raise StoreError(
                    "session label %r is not usable as a file name. Use "
                    "letters, digits, dot, dash and underscore." % (label,))
            return label, os.path.exists(self._session_path(label))

        for h in prefix_hashes(system, messages):
            sid = self._read_index(h)
            if sid and os.path.exists(self._session_path(sid)):
                return sid, True
        return self.new_id(), False

    def resolve_prefix(self, system, messages, label=None):
        """resolve(), plus HOW MANY messages the matched prefix covers.

        Since 2026-10-30 the count is informational: the server no longer
        masks a matched prefix from the stored map, because replaying the
        map is not replaying detection (see proxy/maskcache.py). What the
        prefix still decides is WHICH session a request belongs to, which
        is why the normalisation above matters: a moved cache marker must
        not split one conversation into two.

        A LABEL IS A NAMESPACE, NOT A BYPASS. Changed 2026-10-18.

        It used to return 0, on the reasoning that a label is an operator's
        choice of bucket and says nothing about what the request contains.
        The effect was that --engagement DISABLED THE PREFIX CACHE: with
        matched=0 the server's _cached_paths() is empty, _is_cached() is
        False for every path, and every message in the history is
        re-detected on every turn -- at the measured p50 of 314 ms per
        message, climbing with conversation length. So the one flag that
        makes "purge everything from engagement X" answerable also made a
        long agent session progressively slower, which is a cost the label
        does not need to pay for the property it is there for.

        The prefix scan therefore runs under a label too, and the matched
        count is honoured ONLY when the index entry points at THIS SAME
        label. A prefix that resolves to any other session -- an unlabelled
        run, or another engagement that happens to share a prefix -- is
        ignored and the turn is fully re-detected, because the map stored
        for this label says nothing about what that other session masked.
        That is what keeps the label a hard boundary between engagements
        while giving turn-to-turn continuation inside it.

        `purge --session <label>` is unaffected: the session file is still
        <label>.json.

        The scan below is unbounded, matching the unlabelled arm rather
        than resolve()'s MAX_PREFIX_LOOKBACK. That discrepancy predates
        this change and is left alone rather than fixed on one arm only,
        which would leave two paths with different bounds and nothing
        saying which.
        """
        if label:
            sid, continued = self.resolve(system, messages, label=label)
            if not continued:
                # Nothing stored under this label yet, so there is no map
                # for a prefix to be covered BY.
                return sid, continued, 0
            hs = _all_prefix_hashes(system, messages)
            for k in range(len(hs) - 2, 0, -1):
                if self._read_index(hs[k]) == sid:
                    return sid, True, k
            return sid, continued, 0
        hs = _all_prefix_hashes(system, messages)
        for k in range(len(hs) - 2, 0, -1):
            sid = self._read_index(hs[k])
            if sid and os.path.exists(self._session_path(sid)):
                return sid, True, k
        return self.new_id(), False, 0

    def new_id(self):
        return "s-%s-%s" % (time.strftime("%Y%m%d"),
                            hashlib.sha256(
                                os.urandom(16)).hexdigest()[:12])

    def _read_index(self, h):
        try:
            with open(self._index_path(h), "r", encoding="utf-8") as f:
                sid = f.read().strip()
        except OSError as e:
            if e.errno in (errno.ENOENT, errno.ENOTDIR):
                return None
            raise StoreError("cannot read index entry: %s" % e)
        return sid if _SAFE_ID.match(sid or "") else None

    def remember(self, session_id, system, messages):
        """Record this exact conversation state as belonging to `session_id`,
        so the next turn's prefix scan finds it."""
        h = prefix_hash(system, messages)
        self._atomic_write(self._index_path(h), session_id.encode("utf-8"))

    # ── map I/O ──

    def load(self, session_id):
        """The cumulative mask map, or {} for a session that does not exist.

        A session file that exists and cannot be parsed RAISES. An
        unreadable map is not an empty map: continuing with {} would
        renumber every placeholder mid-conversation and, worse, leave the
        reply unmaskable while reporting success.
        """
        path = self._session_path(session_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except OSError as e:
            if e.errno in (errno.ENOENT, errno.ENOTDIR):
                return {}
            raise StoreError("cannot read session %s: %s" % (session_id, e))
        except ValueError as e:
            raise StoreError(
                "session %s is corrupt (%s). Refusing to continue with an "
                "empty map: the reply would not be unmaskable and the "
                "operator would read placeholders as the model's output. "
                "Delete it with: maskerade sessions purge --session %s"
                % (session_id, e, session_id))
        mm = doc.get("mask_map")
        if not isinstance(mm, dict):
            raise StoreError(
                "session %s has no usable mask_map" % session_id)
        return mm

    def save(self, session_id, mask_map, label=None, unscanned=None):
        """Write the session document.

        `unscanned` is a list of FIELD NAMES forwarded without inspection on
        this request. Names only -- never a path, never a value. The session
        document is the one place a name could turn into a value by
        accident, and `sessions list` promises it never prints request
        content.

        The names are UNIONED with what is already stored, the same way
        `created` is preserved. Overwriting would make the record answer
        "what did the LAST request not inspect", and turn 9 of a
        conversation that sends no tools would erase the fact that turn 1
        forwarded `tools` and `safeguards`. The question worth answering is
        the engagement-level one.
        """
        path = self._session_path(session_id)
        now = time.time()
        created = now
        prior = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f) or {}
            created = doc.get("created", now)
            if isinstance(doc.get("unscanned"), list):
                prior = [n for n in doc["unscanned"] if isinstance(n, str)]
        except Exception:
            pass
        merged = sorted(set(prior) | {n for n in (unscanned or [])
                                      if isinstance(n, str)})
        doc = {
            "id": session_id,
            "label": label,
            "created": created,
            "updated": now,
            "n_placeholders": len(mask_map),
            # Added 2026-10-17. Backward compatible in both directions:
            # load() requires only that `mask_map` is a dict, and
            # list_sessions() defaults a document written before this to an
            # empty list.
            "unscanned": merged,
            "mask_map": mask_map,
        }
        self._atomic_write(path, canonical(doc))

    def _atomic_write(self, path, data):
        """Temp file in the same directory, fsync, rename.

        Same directory so the rename is atomic (a cross-filesystem rename is
        a copy). Permissions are set on the temp file BEFORE the rename, so
        the map is never briefly world-readable under its real name.
        """
        tmp = "%s.tmp.%d.%d" % (path, os.getpid(), threading.get_ident())
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)
            w = _restrict(tmp, is_dir=False)
            if w and w not in self.warnings:
                self.warnings.append(w)
            os.replace(tmp, path)
        except (OSError, StoreError) as e:
            # StoreError is _restrict refusing. The temp file already holds
            # the map, so it goes either way.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            if isinstance(e, StoreError):
                raise
            raise StoreError("cannot write %s: %s" % (path, e))

    # ── lifecycle ──

    def list_sessions(self):
        out = []
        for fn in sorted(os.listdir(self.sessions_dir)):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(self.sessions_dir, fn)
            try:
                st = os.stat(path)
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception:
                out.append({"id": fn[:-5], "label": None, "updated": None,
                            "n_placeholders": None, "unscanned": [],
                            "corrupt": True})
                continue
            unscanned = doc.get("unscanned")
            out.append({
                "id": doc.get("id", fn[:-5]),
                "label": doc.get("label"),
                "updated": st.st_mtime,
                "n_placeholders": doc.get("n_placeholders"),
                # Names only. A document written before 2026-10-17 has no
                # such key and reports an empty list, which is honest: that
                # session was served by a version that refused unknown
                # fields rather than forwarding them.
                "unscanned": unscanned if isinstance(unscanned, list) else [],
                "corrupt": False,
            })
        return out

    def purge(self, older_than_days=None, session_id=None, everything=False):
        """Delete maps and the index entries pointing at them.

        Returns (n_sessions, n_index_entries). Index entries are swept by
        their own mtime as well as by orphanhood, so a pointer to a session
        that is already gone cannot accumulate forever.
        """
        n_s = n_i = 0
        keep = set()

        for meta in self.list_sessions():
            sid = meta["id"]
            path = os.path.join(self.sessions_dir, sid + ".json")
            drop = everything or (session_id is not None and sid == session_id)
            if not drop and older_than_days is not None:
                try:
                    age = (time.time() - os.stat(path).st_mtime) / 86400.0
                except OSError:
                    age = 0.0
                drop = age > float(older_than_days)
            if drop:
                try:
                    os.unlink(path)
                    n_s += 1
                except OSError:
                    pass
            else:
                keep.add(sid)

        cutoff = (time.time() - float(older_than_days) * 86400.0
                  if older_than_days is not None else None)
        for fn in os.listdir(self.index_dir):
            path = os.path.join(self.index_dir, fn)
            try:
                sid = open(path, "r", encoding="utf-8").read().strip()
                mtime = os.stat(path).st_mtime
            except OSError:
                continue
            stale = everything or sid not in keep
            if not stale and cutoff is not None:
                stale = mtime < cutoff
            if stale:
                try:
                    os.unlink(path)
                    n_i += 1
                except OSError:
                    pass
        return n_s, n_i

    def purge_expired(self):
        """The retention policy, applied at startup. 30 days by default."""
        return self.purge(older_than_days=self.retention_days)


# ─────────────────────────────────────────────────────────── self-test ──

def _self_test():
    import shutil
    import tempfile

    failures = []

    def check(cond, label):
        print(("ok   " if cond else "FAIL ") + label)
        if not cond:
            failures.append(label)

    root = tempfile.mkdtemp(prefix="maskerade-store-test-")
    try:
        st = SessionStore(root=root, retention_days=30)

        sysp = "you are a pentest assistant"
        t1 = [{"role": "user", "content": "host is portal.acme.cz"}]
        sid1, cont1 = st.resolve(sysp, t1)
        check(not cont1, "first request mints a new session")
        st.save(sid1, {"[INTERNAL_HOSTNAME_1]": "portal.acme.cz"})
        st.remember(sid1, sysp, t1)

        # Turn 2: the client resends turn 1 plus the reply plus a new turn.
        t2 = t1 + [{"role": "assistant", "content": "try IDOR on it"},
                   {"role": "user", "content": "second user is bob@acme.cz"}]
        sid2, cont2 = st.resolve(sysp, t2)
        check(cont2 and sid2 == sid1,
              "turn 2 is matched to turn 1 by its prefix")
        mm = st.load(sid2)
        check(mm.get("[INTERNAL_HOSTNAME_1]") == "portal.acme.cz",
              "the cumulative map survives the round trip")

        # A different conversation that happens to share no prefix.
        other = [{"role": "user", "content": "unrelated engagement"}]
        sid3, cont3 = st.resolve(sysp, other)
        check(not cont3 and sid3 != sid1,
              "an unrelated conversation gets its own session")

        # A different system prompt is a different conversation even when the
        # messages match -- the system prompt is part of what the model sees.
        sid4, cont4 = st.resolve("different system", t1)
        check(not cont4, "system prompt is part of the identity")

        # Branching: resend turn 1 with a DIFFERENT turn 2. Same parent.
        branch = t1 + [{"role": "assistant", "content": "try IDOR on it"},
                       {"role": "user", "content": "actually, check auth"}]
        sid5, cont5 = st.resolve(sysp, branch)
        check(cont5 and sid5 == sid1,
              "a branched conversation keeps the parent's numbering")

        # ── Normalisation, 2026-10-29: what a client changes in history ──
        # Claude Code's pattern, measured: the cache breakpoint moves to the
        # newest message, and the previous one comes back as a plain string.
        cc = {"type": "ephemeral", "ttl": "1h"}
        sys_b = [{"type": "text", "text": "agent prompt", "cache_control": cc}]
        r1 = [{"role": "user", "content": [
            {"type": "text", "text": "scan crm.acme.cz", "cache_control": cc}]}]
        cid = st.new_id()
        st.save(cid, {"[INTERNAL_HOSTNAME_1]": "crm.acme.cz"})
        st.remember(cid, sys_b, r1)
        r2 = [{"role": "user", "content": "scan crm.acme.cz"},
              {"role": "assistant", "content": [
                  {"type": "tool_use", "id": "t1", "name": "Bash",
                   "input": {"command": "nmap crm.acme.cz"}}]},
              {"role": "user", "content": [
                  {"type": "tool_result", "tool_use_id": "t1",
                   "content": [{"type": "text", "text": "80/tcp open",
                                "cache_control": cc}]}]}]
        sys_b2 = [{"type": "text", "text": "agent prompt"}]
        got = st.resolve_prefix(sys_b2, r2)
        check(got == (cid, True, 1),
              "a moved cache_control and a collapsed text block still "
              "match the parent (got %r)" % (got,))
        r3 = r2 + [{"role": "assistant", "content": "done"},
                   {"role": "user", "content": "next"}]
        st.remember(cid, sys_b2, r2)
        check(st.resolve_prefix(sys_b, r3)[2] == 3,
              "the next turn matches the longest prefix again")

        # NOT normalised: anything extract() reads. A key named
        # cache_control inside tool input is data, and must split.
        a = [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "t", "name": "x",
             "input": {"cache_control": "secret-1"}}]}]
        b = [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "t", "name": "x",
             "input": {"cache_control": "secret-2"}}]}]
        check(prefix_hash("s", a) != prefix_hash("s", b),
              "cache_control inside tool input is hashed as data")
        cit = [{"role": "user", "content": [
            {"type": "text", "text": "x", "citations": []}]}]
        check(prefix_hash("s", cit) != prefix_hash(
            "s", [{"role": "user", "content": "x"}]),
            "a text block with another key is not collapsed")
        two = [{"role": "user", "content": [{"type": "text", "text": "x"},
                                            {"type": "text", "text": "y"}]}]
        check(prefix_hash("s", two) != prefix_hash(
            "s", [{"role": "user", "content": "xy"}]),
            "two text blocks are not merged into one string")
        check(prefix_hash("s", [{"role": "user", "content": "x"}])
              != prefix_hash("s", [{"role": "assistant", "content": "x"}]),
              "the role is part of the identity")

        # The incremental hash is byte-for-byte the documented one.
        msgs = r2 + [{"role": "user", "content": "\u0159\u00e1dek \U0001f600"}]
        hs = _all_prefix_hashes(sys_b, msgs)
        ok = len(hs) == len(msgs) + 1
        for k in range(len(msgs) + 1):
            ref = hashlib.sha256(canonical({
                "system": _norm_content(sys_b),
                "messages": [_norm_message(m) for m in msgs[:k]]})).hexdigest()
            ok = ok and hs[k] == ref
        check(ok, "incremental prefix hashes equal canonical() of each prefix")
        check(prefix_hashes(sys_b, msgs, lookback=2) == [hs[3], hs[2]],
              "prefix_hashes is longest-first and bounded")

        # The invariant the cache rests on: normalising changes nothing
        # extract() reads, in content or in order.
        from . import fields as _F
        for body in ({"messages": r2}, {"messages": r3}, {"messages": cit},
                     {"messages": two}, {"messages": msgs}):
            raw_t = [t for _, t in _F.extract(body)]
            norm_t = [t for _, t in _F.extract(
                {"messages": [_norm_message(m) for m in body["messages"]]})]
            check(raw_t == norm_t, "normalising preserves the scanned text "
                                   "(%d strings)" % len(raw_t))

        # A lone surrogate: json.loads accepts "\\ud83d", a client that cuts
        # a tool output inside an emoji sends one, and it used to kill the
        # conversation in persist() and every turn after.
        lone = json.loads('[{"role":"user","content":"cut \\ud83d here"}]')
        raised = None
        try:
            sid_l = st.new_id()
            st.save(sid_l, {"[X_1]": "cut \ud83d"})
            st.remember(sid_l, "s", lone)
            nxt = lone + [{"role": "assistant", "content": "ok"},
                          {"role": "user", "content": "more"}]
            back = st.resolve_prefix("s", nxt)
            mm_l = st.load(sid_l)
        except Exception as e:
            raised = e
        check(raised is None and back == (sid_l, True, 1)
              and mm_l == {"[X_1]": "cut \ud83d"},
              "a lone surrogate is stored, remembered and matched (%r)"
              % (raised,))

        # Corruption refuses rather than silently renumbering.
        with open(st._session_path(sid1), "w", encoding="utf-8") as f:
            f.write("{ this is not json")
        raised = False
        try:
            st.load(sid1)
        except StoreError:
            raised = True
        check(raised, "a corrupt map raises instead of returning {}")

        # The unscanned record: names only, and UNIONED across turns.
        u1 = st.new_id()
        st.save(u1, {"[A_1]": "b"}, unscanned=["system", "tools"])
        st.save(u1, {"[A_1]": "b"}, unscanned=["safeguards"])
        row = [r for r in st.list_sessions() if r["id"] == u1][0]
        check(row["unscanned"] == ["safeguards", "system", "tools"],
              "unscanned names are unioned across turns, not overwritten")
        with open(st._session_path(u1), "r", encoding="utf-8") as f:
            raw_doc = json.load(f)
        check(all(isinstance(n, str) for n in raw_doc["unscanned"]),
              "the stored unscanned list holds names, nothing else")
        check(st.load(u1) == {"[A_1]": "b"},
              "adding the key did not disturb the mask map")

        # A document written before the key existed still lists.
        old_doc = st.new_id()
        with open(st._session_path(old_doc), "w", encoding="utf-8") as f:
            json.dump({"id": old_doc, "label": None, "created": 1.0,
                       "updated": 1.0, "n_placeholders": 1,
                       "mask_map": {"[A_1]": "b"}}, f)
        row = [r for r in st.list_sessions() if r["id"] == old_doc][0]
        check(row["unscanned"] == [] and not row["corrupt"],
              "a pre-2026-10-17 session document still lists, with no names")

        # Labels override the derived key.
        st.save("acme-q4", {"[EMAIL_ADDRESS_1]": "bob@acme.cz"},
                label="acme-q4")
        sid6, cont6 = st.resolve(sysp, other, label="acme-q4")
        check(sid6 == "acme-q4" and cont6, "an explicit label wins")
        raised = False
        try:
            st.resolve(sysp, other, label="../../etc/passwd")
        except StoreError:
            raised = True
        check(raised, "a label that is a path is refused")

        # Permissions, POSIX only.
        if sys.platform != "win32":
            mode = os.stat(st._session_path("acme-q4")).st_mode & 0o777
            check(mode == 0o600, "session file is 0600 (got %o)" % mode)
            mode = os.stat(st.sessions_dir).st_mode & 0o777
            check(mode == 0o700, "sessions dir is 0700 (got %o)" % mode)

            # A chmod that raises, and a chmod the filesystem ignores. Both
            # must refuse; neither may leave a temp copy of the map behind.
            real_chmod = os.chmod
            for label, fake in (
                    ("raises", lambda p, m: (_ for _ in ()).throw(
                        PermissionError(1, "Operation not permitted", p))),
                    ("is ignored", lambda p, m: real_chmod(p, 0o644))):
                os.chmod = fake
                raised = False
                try:
                    st.save("perm-probe", {"[A_1]": "b"})
                except StoreError:
                    raised = True
                finally:
                    os.chmod = real_chmod
                check(raised, "a chmod that %s refuses the write" % label)
                left = [f for f in os.listdir(st.sessions_dir)
                        if f.startswith("perm-probe")]
                check(not left, "and leaves nothing behind (%s)" % left)

        # Purge by age: backdate one session past the retention window.
        old = st.new_id()
        st.save(old, {"[X_1]": "y"})
        st.remember(old, "sys", [{"role": "user", "content": "old"}])
        past = time.time() - 31 * 86400
        os.utime(st._session_path(old), (past, past))
        n_s, _ = st.purge(older_than_days=30)
        check(n_s >= 1, "purge removes a session past the retention window")
        check(not os.path.exists(st._session_path(old)),
              "the purged session file is gone")
        check(st._read_index(prefix_hash(
            "sys", [{"role": "user", "content": "old"}])) is None,
            "its index pointer is swept with it")

        # The sync-folder guard.
        check(sync_warning("C:/Users/x/OneDrive/maskerade") is not None,
              "a OneDrive path is flagged")
        check(sync_warning(root) is None, "a plain temp path is not flagged")

        n_s, n_i = st.purge(everything=True)
        check(st.list_sessions() == [], "purge --all empties the store")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\n%s" % ("PASSED" if not failures
                    else "%d FAILED: %s" % (len(failures), failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_self_test())
