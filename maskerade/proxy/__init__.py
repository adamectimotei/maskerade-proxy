# -*- coding: utf-8 -*-
"""maskerade.proxy — the localhost masking proxy.

Decision record and Week 4 plan S1: this is NOT a hosted service. It runs on
the operator's own machine, binds 127.0.0.1, and nothing reaches any host the
author operates. Its whole reason to exist is structural: with the library the
operator must remember to call Pipeline.run() on every field, and with a daemon
in the path their client CANNOT reach the provider except through the masker.

Four modules, built in dependency order:

    store    where the mask map lives on disk, and how a request is matched
             to the conversation it belongs to                  (this increment)
    fields   which request fields get masked, and what happens
             to one nobody has seen before                      (this increment)
    stream   the hold-back buffer that unmasks an SSE response  (next)
    server   the socket, the bind guard, the error semantics     (next)

Everything here is stdlib. The base install stays at three declared
dependencies; see pyproject.toml's note on why that property is easy to lose.
"""

__all__ = ["store", "fields"]
