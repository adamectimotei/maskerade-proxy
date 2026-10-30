# -*- coding: utf-8 -*-
"""What each piece of history was masked AS, so the next turn can reuse it.

THE PROBLEM
-----------
The client resends the whole conversation every turn. Detecting every
message every turn costs seconds per request and grows with the
conversation. The first answer, used until 2026-10-30, was: a message
inside the matched conversation prefix is masked from the stored mask map
alone, with no detection.

That is NOT the same result. Detection finds a value's span; re-applying
the map replaces every known value, longest first, and the two disagree:

  * a value recorded under two types, [CREDENTIAL_12] and
    [HIGH_ENTROPY_SECRET_201], comes out as whichever the map's iteration
    order reaches first -- and a map loaded from disk is ordered
    differently from the same map in memory, so it flipped between turns;
  * a longer value learned LATER can overlap an older one. Replaced first,
    it leaves the older value no longer matching whole:
    "user=jan.novak@acme.cz" with "login user=jan.novak" learned later
    becomes "[CREDENTIAL_7]@acme.cz", and the domain is back in clear.

A dogfood session showed both: 16 messages whose placeholders changed
between turns, and client names that had been masked going out in clear.

THE RULE
--------
Keep what a text was actually masked as, and only ever ADD to it.

  first sight   full detection, exactly as before. The masked result is
                remembered, keyed by a hash of the raw text, together with
                how much of the session's map it already accounts for.
  later turns   the remembered result, plus the map entries learned since,
                substituted OUTSIDE existing placeholders. Nothing already
                masked can change or come back; a newly learned value is
                still masked in old history, as before.

Keyed by the TEXT, not by its position: the same tool output re-read in a
later message is reused, and nothing depends on the conversation prefix.

WHAT IT DOES NOT TRUST
----------------------
  * Memory only. A restart costs one full re-detection, which is the old,
    correct, slow path -- never a guess.
  * Remembered only after the map that numbers it is persisted (commit()),
    so a request that failed upstream cannot leave placeholders behind that
    the stored map does not hold.
  * Checked against the stored map on every request (sync()). If any
    placeholder this cache relies on is missing or now means something
    else -- `sessions purge`, a hand-edited file -- the session's entries
    are dropped and everything is detected again.
  * A text whose detection was degraded (a stage failed) is not
    remembered, so the next turn detects it properly.
  * A changed rules file clears everything: the rules decide what the
    first detection found.
"""
import collections
import hashlib
import re
import threading

from maskerade.masker import word_boundary_pattern

# A complete placeholder. Mirrors masker._CONTAINS_PLACEHOLDER_RE.
_PLACEHOLDER_RE = re.compile(r"(\[[A-Z][A-Z0-9_]*_\d+\])")

# Characters of masked text kept across all sessions. A 114-message Claude
# Code session measured about 1 MB of scanned text, so this holds a working
# day of several engagements. Past it the least recently used entries go,
# and those texts are simply detected again.
MAX_CHARS_DEFAULT = 256 * 1024 * 1024

# Sessions remembered with no text in them. Without --engagement every new
# conversation -- and every side request Claude Code makes -- is a session
# of its own, so empty ones are pruned rather than kept for the life of
# the process.
MAX_EMPTY_SESSIONS = 256


def text_key(text):
    """Hash of a raw text. surrogatepass: a lone surrogate is valid input."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).digest()


def substitute_outside_placeholders(text, entries):
    """Replace each (placeholder, value) in `text`, never inside a placeholder.

    Same boundary rule as the masker (word_boundary_pattern), longest value
    first, ties in the order given. Placeholders already in the text are
    never matched into, and a value cannot match across one: it was masked
    in part, and the part that is left is not this value.
    """
    for ph, value in sorted(entries, key=lambda e: -len(e[1] or "")):
        if not value or value not in text:
            continue
        pattern = re.compile(word_boundary_pattern(value))
        parts = _PLACEHOLDER_RE.split(text)
        # split() with one group: even indices are literal text, odd ones
        # are the placeholders it split on.
        for i in range(0, len(parts), 2):
            if value in parts[i]:
                parts[i] = pattern.sub(lambda _m: ph, parts[i])
        text = "".join(parts)
    return text


class _Session(object):
    __slots__ = ("order", "values", "keys", "gen")

    def __init__(self, gen):
        self.order = []       # placeholders, in the order this cache met them
        self.values = {}      # placeholder -> value, as committed
        self.keys = set()     # text keys stored for this session
        self.gen = gen


class MaskCache(object):
    """Thread-safe. One instance per Masker, shared by every request."""

    def __init__(self, max_chars=MAX_CHARS_DEFAULT):
        self.max_chars = int(max_chars)
        self._lock = threading.Lock()
        self._sessions = {}
        # (sid, text key) -> [masked text, entries of session.order applied]
        self._texts = collections.OrderedDict()
        self._chars = 0
        self._gen = 0

    # ── lifecycle ──

    def clear(self):
        """Forget everything. Called when the rules file changes."""
        with self._lock:
            self._gen += 1
            self._sessions.clear()
            self._texts.clear()
            self._chars = 0

    def _drop_session(self, sid):
        st = self._sessions.pop(sid, None)
        if st is None:
            return
        for k in st.keys:
            ent = self._texts.pop((sid, k), None)
            if ent is not None:
                self._chars -= len(ent[0])

    def sync(self, sid, mask_map):
        """Reconcile with the map loaded for this request. Returns a token.

        Every placeholder the cache has committed must still be in the map
        with the same value, or the session's entries are dropped. Map
        entries the cache has not met are appended to its order, so every
        remembered text gets them applied on its next use.
        """
        with self._lock:
            st = self._sessions.get(sid)
            if st is not None:
                for ph, v in st.values.items():
                    if mask_map.get(ph) != v:
                        self._drop_session(sid)
                        st = None
                        break
            if st is None:
                if len(self._sessions) >= MAX_EMPTY_SESSIONS:
                    for osid in [k for k, v in self._sessions.items()
                                 if not v.keys]:
                        del self._sessions[osid]
                st = self._sessions[sid] = _Session(self._gen)
            for ph in sorted(mask_map):
                if ph not in st.values:
                    st.values[ph] = mask_map[ph]
                    st.order.append(ph)
            return (sid, st.gen, id(st))

    # ── per text ──

    def lookup(self, token, text, mask_map):
        """The remembered masking of `text`, brought up to date, or None.

        Up to date means: every entry of `mask_map` -- the request's map AS
        IT IS NOW, including values learned from texts earlier in this same
        request -- that the remembered result does not yet account for is
        substituted. The caller records the result with len(mask_map).
        """
        sid = token[0]
        with self._lock:
            st = self._live(token)
            if st is None:
                return None
            key = (sid, text_key(text))
            ent = self._texts.get(key)
            if ent is None:
                return None
            self._texts.move_to_end(key)
            masked, n = ent
            missing = [ph for ph in st.order[n:] if ph in mask_map]
            known = st.values
        # Entries of this request's map the session has not committed yet.
        missing += [ph for ph in mask_map if ph not in known]
        if missing:
            masked = substitute_outside_placeholders(
                masked, [(ph, mask_map[ph]) for ph in missing])
        return masked

    def commit(self, token, final_map, pending):
        """Remember this request's texts. Call only after the map is saved.

        `pending` is [(raw text, masked text, k)], where the masked text
        already accounts for the first k entries of the request's map in
        its insertion order -- which is final_map's order, because the map
        only ever grows by appending (Masker checks that).
        """
        sid = token[0]
        keys = list(final_map)
        prepared = []
        for raw, masked, k in pending:
            later = keys[k:]
            if later:
                masked = substitute_outside_placeholders(
                    masked, [(ph, final_map[ph]) for ph in later])
            prepared.append((text_key(raw), masked))
        with self._lock:
            st = self._live(token)
            if st is None:
                return
            # Two requests of one session that ran concurrently can each
            # have issued the same new placeholder for different values
            # (unlabelled sessions only; under --engagement they are
            # serialised). Whichever committed second would leave text
            # cached under a meaning the cache no longer records. Drop
            # the session instead: the next request detects again.
            if any(st.values.get(ph, final_map[ph]) != final_map[ph]
                   for ph in keys):
                self._drop_session(sid)
                return
            for ph in keys:
                if ph not in st.values:
                    st.values[ph] = final_map[ph]
                    st.order.append(ph)
            n = len(st.order)
            for key, masked in prepared:
                old = self._texts.pop((sid, key), None)
                if old is not None:
                    self._chars -= len(old[0])
                self._texts[(sid, key)] = [masked, n]
                st.keys.add(key)
                self._chars += len(masked)
            while self._chars > self.max_chars and self._texts:
                (osid, okey), ent = self._texts.popitem(last=False)
                self._chars -= len(ent[0])
                ost = self._sessions.get(osid)
                if ost is not None:
                    ost.keys.discard(okey)

    def _live(self, token):
        sid, gen, ident = token
        st = self._sessions.get(sid)
        if st is None or st.gen != gen or id(st) != ident:
            return None
        return st

    def stats(self):
        with self._lock:
            return {"sessions": len(self._sessions),
                    "texts": len(self._texts), "chars": self._chars}


# ─────────────────────────────────────────────────────────── self-test ──

def _self_test():
    failures = []

    def check(cond, label):
        print(("ok   " if cond else "FAIL ") + label)
        if not cond:
            failures.append(label)

    # Substitution never reaches into a placeholder, and keeps boundaries.
    t = "id [CLIENT_NAME_1] and leo, leopard, leo."
    out = substitute_outside_placeholders(t, [("[CLIENT_NAME_2]", "leo")])
    check(out == "id [CLIENT_NAME_1] and [CLIENT_NAME_2], leopard, "
                 "[CLIENT_NAME_2].", "word-bounded, outside placeholders")
    out = substitute_outside_placeholders(
        "[A_1] x", [("[B_1]", "A_1"), ("[C_1]", "1]")])
    check(out == "[A_1] x", "a value never matches inside a placeholder")
    out = substitute_outside_placeholders(
        "user=[EMAIL_ADDRESS_1] ok", [("[CREDENTIAL_7]", "user=jan.novak")])
    check(out == "user=[EMAIL_ADDRESS_1] ok",
          "a later, longer value cannot cut an older placeholder open")
    out = substitute_outside_placeholders(
        "a abc ab", [("[X_1]", "ab"), ("[Y_1]", "abc")])
    check(out == "a [Y_1] [X_1]", "longest value first")

    c = MaskCache()
    m0 = {"[EMAIL_ADDRESS_1]": "jan@acme.cz"}
    tok = c.sync("s", m0)
    check(c.lookup(tok, "mail jan@acme.cz", m0) is None, "a miss is None")
    c.commit(tok, m0, [("mail jan@acme.cz", "mail [EMAIL_ADDRESS_1]", 1)])
    m1 = dict(m0)
    m1["[PERSON_NAME_1]"] = "mail"            # learned later
    tok = c.sync("s", m1)
    got = c.lookup(tok, "mail jan@acme.cz", m1)
    check(got == "[PERSON_NAME_1] [EMAIL_ADDRESS_1]",
          "a value learned later is applied to remembered text")

    # A placeholder that changed meaning drops the session.
    tok = c.sync("s", {"[EMAIL_ADDRESS_1]": "someone@else.cz"})
    check(c.lookup(tok, "mail jan@acme.cz",
                   {"[EMAIL_ADDRESS_1]": "someone@else.cz"}) is None,
          "a re-assigned placeholder invalidates the session")
    check(c.stats()["texts"] == 0, "and its texts are gone")

    # commit() brings a text up to date with entries learned after it in
    # the same request.
    c = MaskCache()
    tok = c.sync("s", {})
    final = {"[A_1]": "alpha", "[B_1]": "beta"}
    c.commit(tok, final, [("alpha beta", "[A_1] beta", 1)])
    tok = c.sync("s", final)
    check(c.lookup(tok, "alpha beta", final) == "[A_1] [B_1]",
          "commit applies entries learned later in the same request")

    # A token from before clear() cannot commit.
    c = MaskCache()
    tok = c.sync("s", {})
    c.clear()
    c.commit(tok, {"[A_1]": "x"}, [("x", "[A_1]", 1)])
    check(c.stats()["texts"] == 0, "a stale token commits nothing")

    # Eviction keeps the bound.
    c = MaskCache(max_chars=10)
    tok = c.sync("s", {})
    c.commit(tok, {}, [("one", "aaaaaa", 0), ("two", "bbbbbb", 0)])
    st = c.stats()
    check(st["chars"] <= 10 and st["texts"] == 1, "eviction keeps the bound")
    tok = c.sync("s", {})
    check(c.lookup(tok, "two", {}) == "bbbbbb"
          and c.lookup(tok, "one", {}) is None, "least recent goes first")

    # Concurrent requests that numbered differently: nothing is kept.
    c = MaskCache()
    t1 = c.sync("s", {})
    t2 = c.sync("s", {})
    c.commit(t1, {"[E_1]": "a@x.cz"}, [("a@x.cz", "[E_1]", 1)])
    c.commit(t2, {"[E_1]": "b@x.cz"}, [("b@x.cz", "[E_1]", 1)])
    check(c.stats()["texts"] == 0,
          "a placeholder committed with two meanings drops the session")

    # Empty sessions do not accumulate.
    c = MaskCache()
    for i in range(MAX_EMPTY_SESSIONS * 3):
        c.sync("s-%d" % i, {})
    check(c.stats()["sessions"] <= MAX_EMPTY_SESSIONS + 1,
          "empty sessions are pruned (%d kept)" % c.stats()["sessions"])

    # A lone surrogate is a valid key.
    c = MaskCache()
    tok = c.sync("s", {})
    lone = "cut \ud83d"
    c.commit(tok, {}, [(lone, lone, 0)])
    check(c.lookup(tok, lone, {}) == lone, "a lone surrogate is a valid key")

    print("\n%s" % ("PASSED" if not failures
                    else "%d FAILED: %s" % (len(failures), failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
