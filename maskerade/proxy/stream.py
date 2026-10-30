# -*- coding: utf-8 -*-
"""Unmasking a streaming response, without destroying the reason to stream.

SSE responses arrive token by token, so a placeholder
splits across chunks: `[PERSON_NA` lands in one delta and `ME_1]` in the next.
Unmasking per chunk cannot work.

THE THREE FAILURES, AND WHICH ONE MATTERS
-----------------------------------------
Too lenient: emit `[PERSON_NA` and then `ME_1]`. Cosmetic -- the operator sees
a mangled placeholder.

Too conservative: hold the whole stream and release it at the end. Correct,
and it destroys the only reason anyone streams.

The one that is neither cosmetic nor obvious: a PARTIAL placeholder that is
also a COMPLETE shorter one. A map holding both `[USER_ID_1]` and
`[USER_ID_12]` can emit `[USER_ID_1` and, if that is flushed, it unmasks to
the wrong account -- silently, with no mangled text to notice. In a tool whose
job is telling one user from another in an IDOR, that is the failure that
matters. unmask_response() length-sorts for exactly this reason; the streaming
path has to preserve the property, and it does so by never flushing a tail
that could still grow.

THE RULE
--------
Hold back the longest suffix that could still become a placeholder; flush
everything in front of it; flush all of it at end of stream.

"Could still become a placeholder" is decided by shape, not by the map. A tail
is viable when it starts at a `[`, has not yet met its `]`, and contains only
characters a placeholder body may contain. The moment a character arrives that
cannot appear in a placeholder, the tail provably is not one and goes out. In
practice that means at most a few tens of bytes are ever held, and ordinary
prose -- which has no `[` -- is never held at all.

The shape is the one masker.py emits, from `placeholder_format`. It is matched
here rather than imported because the masker's regexes describe a whole token
and this needs to recognise a prefix of one, which is a different question.

TOOL USE
--------
`input_json_delta` carries fragments of the JSON a tool_use block is being
built from, and placeholders appear inside its string literals. Substituting a
raw value there can break the JSON: a password containing a quote or a
backslash turns a valid fragment into an invalid one. Values are therefore
JSON-escaped on that path and only on that path.

This assumes a placeholder in `partial_json` sits inside a string literal,
which is true because that is the only place the masker ever put one -- it
masks values, and a JSON value that was a string is still a string.
"""
import json
import re


# A lone UTF-16 surrogate. json.loads() turns a "\\ud83d" escape into one, and
# a client that cuts a tool output inside an emoji sends exactly that, so a
# masked value -- and therefore an unmasked reply -- can contain one.
_SURROGATE_RE = re.compile("[\ud800-\udfff]")

# A COMPLETE placeholder: [TYPE_1]. Mirrors masker._CONTAINS_PLACEHOLDER_RE.
PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*_\d+\]")

# A tail that is still an OPEN placeholder candidate: an unclosed `[`
# followed only by characters a placeholder body may contain. Deliberately
# permissive -- it decides what to HOLD, and holding something that turns out
# not to be a placeholder costs a few bytes of latency, while flushing
# something that turns out to be one costs correctness.
OPEN_TAIL_RE = re.compile(r"\[[A-Z0-9_]*$")

# Longest a real placeholder can be before it is certainly not one. Entity
# types are code constants and the index is a small counter; 96 is far above
# anything the masker emits. Without this, a `[` in ordinary output -- a
# markdown link, a log line, an array index -- would hold the stream open to
# the end.
MAX_HELD = 96


class HoldBackBuffer:
    """Feed text in, get text out that is safe to unmask and emit.

    Safe means: no suffix of what was emitted could have grown into a
    different placeholder than the one it was read as.
    """

    def __init__(self, max_held=MAX_HELD):
        self.max_held = max_held
        self._pending = ""

    def feed(self, text):
        """Append `text`; return the part that can be released now."""
        if not text:
            return ""
        self._pending += text
        cut = self._safe_cut(self._pending)
        out, self._pending = self._pending[:cut], self._pending[cut:]
        return out

    def flush(self):
        """End of stream: everything left is final, whatever shape it is."""
        out, self._pending = self._pending, ""
        return out

    @property
    def held(self):
        return self._pending

    def _safe_cut(self, s):
        """Index up to which `s` can be released."""
        start = s.rfind("[")
        if start < 0:
            return len(s)
        tail = s[start:]
        # A closed bracket means this candidate resolved one way or the
        # other; nothing after it is open, or rfind would have found it.
        if "]" in tail:
            return len(s)
        if not OPEN_TAIL_RE.match(tail):
            # Contains a character no placeholder body can hold, so it is
            # not a placeholder and never will be.
            return len(s)
        if len(tail) > self.max_held:
            return len(s)
        return start


def _json_escape_inner(value):
    """JSON-escape `value` for insertion inside an existing string literal."""
    return json.dumps(value)[1:-1]


def unmask_text(text, mask_map):
    """Restore originals in plain text. Longest placeholder first.

    Same contract as masker.unmask_response(); repeated here because the
    streaming path substitutes into fragments rather than whole documents and
    must not acquire a second, divergent implementation by accident.
    """
    out = text
    for ph, original in sorted(mask_map.items(), key=lambda kv: -len(kv[0])):
        if ph in out:
            out = out.replace(ph, original)
    return out


def unmask_json_fragment(fragment, mask_map):
    """Restore originals inside a partial JSON document, escaping as it goes."""
    out = fragment
    for ph, original in sorted(mask_map.items(), key=lambda kv: -len(kv[0])):
        if ph in out:
            out = out.replace(ph, _json_escape_inner(original))
    return out


def leftover_placeholders(text):
    """Placeholders still present after unmasking.

    A model can emit a placeholder that was never in the map -- it invents
    `[PERSON_NAME_9]` by pattern, or refers to one from a conversation this
    session does not hold. Unmasking leaves it literal, and the operator reads
    a placeholder as the model's output. The caller reports these on the
    response rather than letting them pass unremarked.
    """
    return sorted(set(PLACEHOLDER_RE.findall(text)))


# ───────────────────────────────────────────────────── the SSE pipeline ──

class SSEUnmasker:
    """Rewrite an Anthropic SSE stream, unmasking the text it carries.

    Events are parsed rather than pattern-matched over raw bytes: the text
    lives inside a JSON object, so a regex over the wire format would have to
    reimplement JSON string parsing to know where a value ends. Each event is
    re-serialised after rewriting, which is also what allows the unmasked text
    to be a different length from the masked text.

    Events that carry no text pass through with their bytes untouched.
    """

    TEXT_DELTAS = {"text_delta": "text", "input_json_delta": "partial_json"}

    def __init__(self, mask_map):
        self.mask_map = mask_map or {}
        self._buffers = {}          # content block index -> HoldBackBuffer
        self._kinds = {}            # content block index -> delta type
        self.leftovers = set()

    def feed(self, chunk_text):
        """Take a decoded chunk of the SSE body; yield rewritten event text."""
        out = []
        for raw_event in self._split_events(chunk_text):
            out.append(self._handle(raw_event))
        return "".join(out)

    def finish(self):
        """Flush every open block. Called when the upstream body ends.

        A stream that is cut off mid-placeholder still gets its held bytes,
        because the alternative is silently swallowing output. They are
        emitted as they stand, and any placeholder among them is reported.
        """
        tail = []
        for idx, buf in sorted(self._buffers.items()):
            rest = buf.flush()
            if not rest:
                continue
            kind = self._kinds.get(idx, "text_delta")
            text = self._unmask(rest, kind)
            self.leftovers.update(leftover_placeholders(text))
            tail.append(self._event("content_block_delta", {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": kind,
                          self.TEXT_DELTAS.get(kind, "text"): text},
            }))
        self._buffers.clear()
        return "".join(tail)

    # ── internals ──

    def __init_subclass__(cls, **kw):      # pragma: no cover
        super().__init_subclass__(**kw)

    def _split_events(self, text):
        """Yield complete SSE events; keep an incomplete trailer for later."""
        buf = getattr(self, "_wire", "") + text
        parts = buf.split("\n\n")
        self._wire = parts.pop() if parts else ""
        return [p for p in parts if p.strip()]

    def _handle(self, raw_event):
        data_line = None
        for line in raw_event.split("\n"):
            if line.startswith("data:"):
                data_line = line[5:].strip()
                break
        if data_line is None:
            return raw_event + "\n\n"
        try:
            payload = json.loads(data_line)
        except ValueError:
            # Not JSON we understand. Pass it through rather than dropping
            # it: an unparsed event is the provider's business, and swallowing
            # one would truncate the stream.
            return raw_event + "\n\n"

        if payload.get("type") != "content_block_delta":
            return raw_event + "\n\n"

        delta = payload.get("delta") or {}
        kind = delta.get("type")
        field = self.TEXT_DELTAS.get(kind)
        if field is None:
            return raw_event + "\n\n"

        idx = payload.get("index", 0)
        self._kinds[idx] = kind
        buf = self._buffers.get(idx)
        if buf is None:
            buf = self._buffers[idx] = HoldBackBuffer()

        releasable = buf.feed(delta.get(field) or "")
        if not releasable:
            # Everything is held. Emitting an empty delta would be noise, so
            # nothing goes out for this event; the bytes arrive with the next.
            return ""

        text = self._unmask(releasable, kind)
        self.leftovers.update(leftover_placeholders(text))
        payload["delta"] = dict(delta)
        payload["delta"][field] = text
        return self._event(payload.get("type"), payload)

    def _unmask(self, text, kind):
        if kind == "input_json_delta":
            return unmask_json_fragment(text, self.mask_map)
        return unmask_text(text, self.mask_map)

    @staticmethod
    def _event(event_type, payload):
        """One SSE event. ensure_ascii=False keeps Czech text as UTF-8.

        Except a lone surrogate, which json.dumps() then writes out raw and
        which cannot be encoded as UTF-8: measured 2026-10-29, a reply that
        restored a value containing one raised UnicodeEncodeError in the
        server's chunk writer and cut the stream. Inside a JSON string --
        the only place a surrogate can be -- its \\uXXXX escape is valid, and
        is exactly what the client sent us.
        """
        data = json.dumps(payload, ensure_ascii=False)
        if _SURROGATE_RE.search(data):
            data = _SURROGATE_RE.sub(lambda m: "\\u%04x" % ord(m.group()),
                                     data)
        return "event: %s\ndata: %s\n\n" % (event_type, data)


# ─────────────────────────────────────────────────────────── self-test ──

def _self_test():
    import sys

    failures = []

    def check(cond, label):
        print(("ok   " if cond else "FAIL ") + label)
        if not cond:
            failures.append(label)

    # ── the hold-back rule ──
    b = HoldBackBuffer()
    check(b.feed("the host is ") == "the host is ",
          "prose with no bracket is never held")
    check(b.feed("[PERSON_NA") == "", "an open candidate is held")
    check(b.feed("ME_1] and") == "[PERSON_NAME_1] and",
          "the candidate is released once it closes")
    check(b.held == "", "nothing is left held after it closes")

    b = HoldBackBuffer()
    check("[not-a-placeholder" in b.feed("see [not-a-placeholder"),
          "a bracket that cannot be a placeholder is released")
    check(b.held == "", "...and nothing stays held behind it")

    b = HoldBackBuffer()
    b.feed("x" * 10 + "[VERY" + "Y" * 200)
    check(b.held == "", "a bracket run past the cap is not held forever")

    # THE case that is not cosmetic: a complete short placeholder that is
    # also the prefix of a longer one.
    mm = {"[USER_ID_1]": "alice", "[USER_ID_12]": "bob"}
    b = HoldBackBuffer()
    out = b.feed("user [USER_ID_1")
    check(out == "user ", "an ambiguous prefix is NOT flushed")
    out += b.feed("2] owns it")
    check(unmask_text(out, mm) == "user bob owns it",
          "the longer placeholder resolves to the right value")

    b = HoldBackBuffer()
    out = b.feed("user [USER_ID_1]") + b.flush()
    check(unmask_text(out, mm) == "user alice",
          "the shorter placeholder still resolves correctly")

    # ── SSE rewriting ──
    def sse(idx, kind, field, value):
        return ("event: content_block_delta\ndata: "
                + json.dumps({"type": "content_block_delta", "index": idx,
                              "delta": {"type": kind, field: value}})
                + "\n\n")

    mm2 = {"[INTERNAL_HOSTNAME_1]": "portal.acmebank.cz",
           "[CREDENTIAL_1]": 'p"a\\ss'}

    u = SSEUnmasker(mm2)
    body = (sse(0, "text_delta", "text", "target [INTERNAL_")
            + sse(0, "text_delta", "text", "HOSTNAME_1] is up"))
    got = u.feed(body) + u.finish()
    texts = []
    for ev in got.split("\n\n"):
        if ev.strip().startswith("event:"):
            texts.append(json.loads(ev.split("data: ", 1)[1])
                         ["delta"]["text"])
    joined = "".join(texts)
    check(joined == "target portal.acmebank.cz is up",
          "a placeholder split across two SSE events is unmasked once")

    # Non-delta events survive untouched.
    u = SSEUnmasker(mm2)
    ping = "event: ping\ndata: {\"type\": \"ping\"}\n\n"
    check(u.feed(ping) == ping, "a non-text event passes through verbatim")

    u = SSEUnmasker(mm2)
    start = ('event: message_start\ndata: '
             '{"type": "message_start", "message": {"id": "msg_1"}}\n\n')
    check(u.feed(start) == start, "message_start passes through verbatim")

    # tool_use: the value contains a quote and a backslash, so a raw
    # substitution would produce invalid JSON.
    u = SSEUnmasker(mm2)
    frag = u.feed(sse(1, "input_json_delta", "partial_json",
                      '{"pw": "[CREDENTIAL_1]"}')) + u.finish()
    payload = json.loads(frag.split("data: ", 1)[1].strip())
    emitted = payload["delta"]["partial_json"]
    check(json.loads(emitted)["pw"] == 'p"a\\ss',
          "a tool_use fragment stays valid JSON after unmasking")

    # A placeholder the map does not hold is reported, not hidden.
    u = SSEUnmasker({"[EMAIL_ADDRESS_1]": "a@b.cz"})
    u.feed(sse(0, "text_delta", "text", "see [PERSON_NAME_9] there"))
    u.finish()
    check("[PERSON_NAME_9]" in u.leftovers,
          "an invented placeholder is reported as a leftover")

    # A stream cut off mid-placeholder still emits what it held.
    u = SSEUnmasker(mm2)
    out = u.feed(sse(0, "text_delta", "text", "cut here [INTERNAL_"))
    out += u.finish()
    check("[INTERNAL_" in out, "a truncated stream is not swallowed")

    # An event arriving split across two network chunks.
    u = SSEUnmasker(mm2)
    whole = sse(0, "text_delta", "text", "host [INTERNAL_HOSTNAME_1] ok")
    half = len(whole) // 2
    out = u.feed(whole[:half]) + u.feed(whole[half:]) + u.finish()
    check("portal.acmebank.cz" in out,
          "an event split across network chunks is reassembled")

    # A restored value holding a lone surrogate: the event must still be
    # encodable, and must still carry the value the client sent.
    lone = json.loads('"Ab\\ud83dcd"')
    u = SSEUnmasker({"[CREDENTIAL_1]": lone})
    out = u.feed(sse(0, "text_delta", "text", "pw [CREDENTIAL_1] ok")) \
        + u.finish()
    enc = None
    try:
        enc = out.encode("utf-8")
    except UnicodeEncodeError:
        pass
    got = "".join(json.loads(ev.split("data: ", 1)[1])["delta"]["text"]
                  for ev in out.split("\n\n") if "data: " in ev)
    check(enc is not None and got == "pw %s ok" % lone,
          "a lone surrogate in a restored value is escaped, not fatal")
    u = SSEUnmasker({})
    cz = u.feed(sse(0, "text_delta", "text", "\u0159\u00e1dek")) + u.finish()
    check("\u0159\u00e1dek" in cz, "other non-ASCII text stays as UTF-8")

    print("\n%s" % ("PASSED" if not failures
                    else "%d FAILED: %s" % (len(failures), failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
