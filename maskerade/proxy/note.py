# -*- coding: utf-8 -*-
"""The redaction contract appended to every request's system prompt.

WHY THE PROXY ADDS THIS AT ALL
------------------------------
Only the proxy knows masking happened. Without a word of explanation the
model reads `[INTERNAL_HOSTNAME_1]` as an unfilled template, says so, and asks
for the real value -- measured on the first two live requests this proxy ever
served. The operator cannot be expected to remember to paste a redaction
contract into every client's settings, because "a thing you remember" is
exactly what the proxy exists to replace.

WHAT IT DOES NOT DO
-------------------
It carries ONLY what masking makes necessary. It says nothing about tone,
format, or how a security finding should be written. Those belong to the
operator: Claude Code, a custom agent or any other client each
bring their own system prompt, and a proxy that quietly overrode them would
be a surprise in the path rather than a service in it.

HOW IT IS ADDED
---------------
As an ADDITIONAL system block at the END of the array, never by editing a
block the client wrote. Two reasons. Prompt caching: a `cache_control`
breakpoint the client set on an earlier block still matches, so appending
costs a cache miss on this block alone rather than on the whole prefix.
Attribution: an operator reading their own request back can see which text is
theirs and which is the proxy's.

It does not affect session identity. The conversation prefix is hashed from
the RAW request before any of this runs, so appending the note cannot split
one conversation into two sessions.

Disclosed, never silent: X-Maskerade-System-Note on the response, a line in
the startup banner, --no-system-note to turn it off, and --system-note FILE
to replace it with your own.
"""
import os


DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "system_note.txt")


class SystemNote:
    """Loaded once at startup; applied to every request."""

    def __init__(self, text, source):
        self.text = (text or "").strip()
        self.source = source

    @classmethod
    def load(cls, path=None, enabled=True):
        if not enabled:
            return cls("", "disabled")
        target = path or DEFAULT_PATH
        try:
            with open(target, "r", encoding="utf-8") as f:
                return cls(f.read(), target)
        except OSError as e:
            # A missing note is not a reason to refuse a request -- it is our
            # text, not the operator's data -- but it must not pass unnoticed,
            # because the symptom is the model arguing with its own input.
            raise RuntimeError(
                "cannot read the system note at %s (%s). Fix the path, or "
                "pass --no-system-note to run without it and accept that the "
                "model will read placeholders as unfilled templates."
                % (target, e))

    def __bool__(self):
        return bool(self.text)

    def apply(self, body):
        """Return `body` with the note appended to its system prompt.

        Mutates nothing the caller still holds: the system value is rebuilt
        rather than edited in place.
        """
        if not self.text:
            return body

        block = {"type": "text", "text": self.text}
        existing = body.get("system")

        if existing is None:
            body["system"] = [block]
        elif isinstance(existing, str):
            # Promoting a string to a block array is the only shape that can
            # carry two pieces of text, and the API accepts either form.
            body["system"] = [{"type": "text", "text": existing}, block]
        elif isinstance(existing, list):
            body["system"] = list(existing) + [block]
        else:
            # fields.extract() would already have refused anything else.
            return body
        return body
