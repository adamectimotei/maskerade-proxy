# -*- coding: utf-8 -*-
"""maskerade.proxy — the localhost masking proxy.

This is NOT a hosted service. It runs on
the operator's own machine, binds 127.0.0.1, and nothing reaches any host the
author operates. Its whole reason to exist is structural: with the library the
operator must remember to call Pipeline.run() on every field, and with a daemon
in the path their client CANNOT reach the provider except through the masker.

The modules:

    store    where the mask map lives on disk, and how a request is matched
             to the conversation it belongs to
    fields   which request fields get masked, and what happens
             to one nobody has seen before
    stream   the hold-back buffer that unmasks an SSE response
    server   the socket, the bind guard, the error semantics
    note     the redaction contract appended to the system prompt
    rules    the per-engagement custom-rules file

Everything here is stdlib. The base install stays at three declared
dependencies; see pyproject.toml's note on why that property is easy to lose.
"""

__all__ = ["store", "fields"]
