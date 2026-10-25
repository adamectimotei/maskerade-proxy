# -*- coding: utf-8 -*-
"""The localhost masking proxy.

    python -m maskerade.proxy.server
    python -m maskerade.proxy.server --port 8787 --tier full

Point any Anthropic client at it and nothing else changes:

    $env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8787"

WHAT IT DOES, IN ORDER
----------------------
 1. locate every string in the request that is the OPERATOR'S OWN CONTENT
    (fields.py); every other field is forwarded uninspected and NAMED on the
    response in X-Maskerade-Unscanned
 2. find the conversation this request continues (store.py), by hashing its
    own prefix -- no header, no client configuration
 3. detect and mask in the content that is NEW, and apply the conversation's
    existing map to the content that is not
 4. forward only the masked body to api.anthropic.com
 5. unmask the reply, streaming or not, and hand it back whole

FAIL-LOUD ON AN UNKNOWN FIELD, FAIL-CLOSED EVERYWHERE ELSE
-----------------------------------------------------------
Changed 2026-10-17. An unrecognised field used to be a refusal. It is now
forwarded and reported, because a refusal does not protect the client -- it
removes the tool from the path, and a pentester whose client returns 400 runs
that client without the proxy and leaks 100%. fields.py's header carries the
full argument and the residual this accepts.

Reporting it means three things, all of which happen here:

  - X-Maskerade-Unscanned on every response, naming the fields
  - one line on stderr per distinct field name, per process, not per request
  - the names recorded on the session, so `maskerade sessions list` can
    answer "what did this engagement not inspect"

Everything else still refuses. Every "cannot complete"
path refuses rather than degrades, because the operator's next action after a
run is to send the text to a third party and a quietly degraded result is
worse than an error. In this file that means:

  - warmup() is called at boot and allowed to propagate. A daemon that starts
    clean and then refuses every request is worse than one that refuses to
    start.
  - a detected entity the masker could not substitute (`masker_unmasked`)
    refuses the request. The validator CANNOT see that set -- it tests with
    the same word-boundary predicate the masker failed on, so every one of
    them passes at 100% coverage -- which is why it is checked separately.
  - a failed validation refuses.
  - a mask map that cannot be read refuses, rather than continuing with an
    empty one and returning placeholders as though they were the answer.
  - binding anything but loopback requires an explicit flag.
  - an image or document block, whose bytes cannot be masked, is refused.
    That is a POLICY refusal and is unaffected by the change above.

Refusals use 400 with a body that reads as a refusal. That is deliberate: a
client retries 429 and 5xx, and a masker that refuses must not look like a
provider that is flaky.

WHAT IS NEVER LOGGED
--------------------
The request body, the response body, the mask map, and the Authorization /
x-api-key headers. The access log carries method, path, status, duration,
session id and counts. On a refusal the offending field's PATH is logged, its
value is not.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import fields as F
from . import stream as S
from .note import SystemNote
from .rules import RulesError, RulesFile
from .store import SessionStore, StoreError, prefix_hash

UPSTREAM = "https://api.anthropic.com"

# The OAuth token endpoint. MEASURED to be on the Messages API host, so
# this proxy still talks to exactly one host.
#
# A client on subscription auth -- which is what most operators use -- sends
# its token refresh to whatever ANTHROPIC_BASE_URL names, so it arrives here.
# Before 2026-10-18 it got a 404 from the unknown-route arm and the client
# could not authenticate through this proxy at all.
#
# It is forwarded verbatim: no masking, no inspection, no logging. The body
# carries the operator's own refresh token, which is in the same class as
# `x-api-key` -- masking it would break auth, and
# writing it anywhere would be worse than anything this tool prevents.
#
# HOW THIS VALUE WAS ARRIVED AT, because the first answer was wrong
# -----------------------------------------------------------------
# It shipped on 2026-10-17 as `console.anthropic.com`, taken from published
# descriptions of the flow rather than from a measurement, and the client
# then failed to authenticate with a 403 Cloudflare interstitial. Measured
# on 2026-10-18, three hosts x four header sets,
# with a deliberately invalid token:
#
#   console.anthropic.com   403 at the WAF without a browser-like
#                           User-Agent; 404 with one. It does not serve
#                           this path at all.
#   platform.claude.com     403 at the WAF bare; 400 "Refresh token not
#                           found or invalid" with a User-Agent. Live, but
#                           it needs the client's own User-Agent forwarded.
#   api.anthropic.com       400 "Refresh token not found or invalid" with
#                           NO extra headers. Live, and reachable with the
#                           six-header allowlist this proxy already sends.
#
# So this is UPSTREAM, and that is the better answer for two reasons
# beyond its working: the proxy keeps talking to one host, which is a claim
# a client can check in ten seconds, and the header allowlist below
# stays an allowlist -- forwarding the client's User-Agent and SDK headers
# to reach `platform.claude.com` would have widened what this proxy
# discloses, for no gain.
#
# The residual: `api.anthropic.com` serving this path is a measurement, not
# a documented guarantee. If subscription auth stops working through the
# proxy, repeat that measurement -- an invalid refresh token POSTed to each
# candidate host, with and without a User-Agent -- and this is the one line
# to change. Pointing it at `platform.claude.com` also needs the User-Agent
# forwarded, which is why the header set matters as much as the host.
OAUTH_UPSTREAM = UPSTREAM

# Forwarded to the provider untouched. Everything else the client sends is
# dropped: hop-by-hop headers break when proxied, and a header this version
# does not know about is not forwarded for the same reason an unknown body
# field is not.
FORWARD_HEADERS = {
    "x-api-key", "authorization", "anthropic-version", "anthropic-beta",
    "content-type", "accept",
}

# Read-only endpoints with no generative content. They carry no client data
# in the body, so they are proxied verbatim.
PASSTHROUGH_GET = {"/v1/models"}

# A passthrough POST forwards FORWARD_HEADERS and nothing else, the same
# allowlist the Messages path uses. Recorded here because the alternative
# was tried and rejected on evidence rather than on taste.
#
# The 403 of 2026-10-18 looked like an argument for forwarding the client's
# headers as it composed them: `console.anthropic.com` refuses a request
# carrying `User-Agent: Python-urllib/3.12` at its WAF. That would have
# meant a denylist on this one route, and with it the client's User-Agent,
# version and SDK headers reaching the provider -- a widening of what this
# proxy discloses, on a path whose body is already the operator's
# credential.
#
# It is not needed. `api.anthropic.com` serves the endpoint with no extra
# headers at all (see OAUTH_UPSTREAM). So the narrower thing is also the
# working thing, and this file has one header policy rather than two.

# POSTs that carry no generative content and are forwarded verbatim. The
# value is the upstream host, because these do not all live on UPSTREAM.
# Their bodies are never parsed, never masked and never logged.
PASSTHROUGH_POST = {"/v1/oauth/token": OAUTH_UPSTREAM}

MAX_BODY_BYTES = 64 * 1024 * 1024

# An HTTP header cannot carry 350 paths, so X-Maskerade-Unscanned carries
# deduplicated short NAMES and says how many it left out. Full paths are
# deliberately not put in a header.
UNSCANNED_MAX_NAMES = 20
UNSCANNED_MAX_CHARS = 200


class Refusal(Exception):
    """Something the masker will not forward. Always the operator's to fix."""

    def __init__(self, message, detail=None, kind="maskerade_refused"):
        self.kind = kind
        self.detail = detail
        super().__init__(message)


class Masker:
    """Pipeline plus session store. One instance, shared by every request."""

    def __init__(self, pipeline, store, system_note=None, rules=None,
                 engagement=None):
        self.pipeline = pipeline
        self.store = store
        # The engagement label, from --engagement. When set it is the
        # session id for every request this daemon serves: one bucket per
        # engagement, so `sessions purge --session <label>` deletes exactly
        # that engagement and nothing else. Since
        # 2026-10-18 it is a NAMESPACE rather than a bypass -- store.py's
        # resolve_prefix still runs the prefix scan inside it, so the cache
        # is live from turn 2 instead of disabled.
        #
        # It is passed to BOTH resolve_prefix calls: the handler's, which
        # takes the lock, and mask_request's. Under a label the two now
        # agree, which NARROWS the known double-resolve bug without fixing
        # it -- for an unlabelled new session new_id() is still random, the
        # two calls still disagree, and the lock is still taken on an id
        # nothing else uses. Noted, not fixed.
        self.engagement = engagement
        self.system_note = system_note or SystemNote("", "disabled")
        # Operator-supplied literal rules. The ONLY layer that can carry
        # the client's own name, and unreachable through the proxy until
        # this existed -- org_evidence's precision-first ORGANIZATION
        # design assumes this stage is populated.
        self.rules = rules or RulesFile(None)
        self._session_locks = {}
        self._guard = threading.Lock()
        # Field names already warned about on stderr. Process-level, so the
        # operator is told once rather than on every turn of a conversation
        # that sends the same eight fields each time.
        self._warned_unscanned = set()

    def lock_for(self, session_id):
        with self._guard:
            lk = self._session_locks.get(session_id)
            if lk is None:
                lk = self._session_locks[session_id] = threading.Lock()
            return lk

    def warn_unscanned(self, unscanned):
        """One stderr line per distinct field name, per process.

        Guarded by the same lock `lock_for` uses: this is a
        ThreadingHTTPServer, two concurrent first requests would otherwise
        both see an empty set and both print. The reason text comes from
        fields.py so there is one copy of it.
        """
        fresh = []
        with self._guard:
            for path, name, reason in unscanned or []:
                if name in self._warned_unscanned:
                    continue
                self._warned_unscanned.add(name)
                fresh.append((path, name, reason))
        for path, name, reason in fresh:
            sys.stderr.write(
                "  [UNSCANNED] %s  forwarded without inspection (first seen "
                "at %s)\n              %s\n"
                % (name, path,
                   reason or "this version has never seen this field. It is "
                             "forwarded and named on every response; nothing "
                             "in it was masked."))
        if fresh:
            sys.stderr.flush()

    # ── the masking pass ──

    def mask_request(self, body):
        """Return (masked_body, mask_map, info). Raises Refusal.

        Fields in the matched conversation prefix are masked with the stored
        map ALONE -- no detection. Re-detecting the whole history every turn
        costs the measured p50 of 314 ms per message, climbing with
        conversation length, to recompute an answer already on disk.

        The prefix is safe to treat this way because the prefix HASH covers
        it: if any byte of `system` or of an earlier message had changed, the
        hash would not have matched and there would be no prefix.

        Since 2026-10-17 the only thing extracted is message content, and the
        prefix hash covers `(system, messages)` in full -- so there is no
        longer any extracted field OUTSIDE the hash's coverage. The sentence
        that used to stand here, "anything the hash does not cover -- tools,
        metadata, stop_sequences -- is detected on every turn", described a
        set that is now empty: those fields are not detected on any turn.
        """
        from maskerade.masker import apply_masks

        # Pick up an edited rules file without a restart. A reload that
        # FAILS keeps the previous rules and reports the reason: running
        # on the last known-good list is safer than dropping to none,
        # which is what letting the error propagate would do.
        changed, rules_error = self.rules.maybe_reload()
        if changed:
            sys.stderr.write("  [RULES] reloaded %s\n"
                             % self.rules.summary())
        if rules_error:
            sys.stderr.write("  [RULES] reload FAILED, keeping %d rule(s): "
                             "%s\n" % (len(self.rules), rules_error))

        # ONE walk. `unscanned` is an out-parameter rather than a second
        # return value so that inject()'s re-walk, which compares the
        # returned paths for count and order, is untouched.
        unscanned = []
        extracted = F.extract(body, unscanned)   # raises on a POLICY refusal
        self.warn_unscanned(unscanned)
        system = body.get("system")
        messages = body.get("messages") or []

        sid, continued, matched = self.store.resolve_prefix(
            system, messages, label=self.engagement)
        mask_map = self.store.load(sid) if continued else {}

        cached = _cached_paths(matched)
        out = []
        notes, widened, unmasked = [], 0, []

        for path, text in extracted:
            if not text:
                out.append((path, text))
                continue
            if _is_cached(path, cached):
                masked, mask_map = apply_masks(
                    text, [], self.pipeline.config, mask_map)
                out.append((path, masked))
                continue

            result = self.pipeline.run(
                text, existing_mask_map=mask_map,
                custom_masks=self.rules.rules)

            if result.get("masker_unmasked"):
                # A detected entity still present in the text that would be
                # sent. Validation does not cover this; see the module note.
                raise Refusal(
                    "the masker detected %d value(s) it could not substitute, "
                    "so the request was not sent."
                    % len(result["masker_unmasked"]),
                    detail={"field": path,
                            "entity_types": sorted(
                                {u.get("entity_type")
                                 for u in result["masker_unmasked"]})})
            validation = result.get("validation") or {}
            if not validation.get("passed", True):
                raise Refusal(
                    "post-mask validation failed on %d value(s), so the "
                    "request was not sent." % len(validation.get("leaked", [])),
                    detail={"field": path,
                            "entity_types": sorted(
                                {l.get("entity_type")
                                 for l in validation.get("leaked", [])})})

            notes.extend(result.get("stage_notes") or [])
            widened += len(result.get("masker_widened") or [])
            mask_map = result["mask_map"]
            out.append((path, result["masked_text"]))

        masked_body = F.inject(body, out)
        # AFTER masking, deliberately. The note is our own text and must not
        # be fed through the detector: it is full of example placeholders,
        # and masking those would nest a placeholder inside a placeholder.
        masked_body = self.system_note.apply(masked_body)
        info = {
            "session": sid,
            "continued": continued,
            "mode": self.pipeline.mode_string(),
            "notes": notes,
            "widened": widened,
            "placeholders": len(mask_map),
            "system_note": bool(self.system_note),
            "rules": len(self.rules),
            "rules_error": self.rules.error,
            # NAMES only, never a path and never a value. These are written
            # to the session document, and `sessions list` must keep its
            # promise that it never prints anything from a request.
            "unscanned": F.short_names(unscanned),
        }
        return masked_body, mask_map, info

    def persist(self, session_id, mask_map, system, messages, unscanned=None):
        # label= was never passed, so doc["label"] was always None and the
        # LABEL column of `maskerade sessions list` always printed `-`.
        # Fixed 2026-10-18: an engagement the operator named has to be
        # visible in the listing, or the label buys nothing they can see.
        self.store.save(session_id, mask_map, label=self.engagement,
                        unscanned=unscanned)
        self.store.remember(session_id, system, messages)


def _cached_paths(matched):
    """Message indices whose content the prefix hash already covers."""
    return set(range(matched))


def _is_cached(path, cached):
    if not cached:
        return False
    # DORMANT since 2026-10-17: extract() emits no `system` path at all, so
    # this branch is unreachable. Kept, like fields.py's schema walk, for
    # the day `--scan system` turns it back on -- deleting it would mean
    # re-deriving the rule that the prefix hash covers the system prompt.
    if path == "system" or path.startswith("system["):
        return True              # covered by the hash whenever matched > 0
    # DORMANT for the same reason: nothing outside `messages[` is extracted.
    if not path.startswith("messages["):
        return False
    try:
        idx = int(path[len("messages["):path.index("]")])
    except (ValueError, IndexError):
        return False
    return idx in cached


# ───────────────────────────────────────────────────────────── handler ──

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "maskerade"
    sys_version = ""

    masker = None          # set on the server instance at startup
    verbose = False

    # ── plumbing ──

    def log_message(self, fmt, *args):
        """Silenced. _access() is the log, and it never sees a body."""

    def _access(self, status, started, extra=""):
        sys.stderr.write("  %s %s -> %s  %.0fms%s\n" % (
            self.command, self.path.split("?")[0], status,
            (time.time() - started) * 1000.0, (" " + extra) if extra else ""))
        sys.stderr.flush()

    def _send_json(self, status, payload, headers=None):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _refuse(self, status, kind, message, detail=None, started=None):
        """A refusal the client must not retry and a human can read.

        `detail` carries field PATHS and entity TYPES -- never values. The
        refused body is not logged and not echoed.
        """
        body = {"type": "error",
                "error": {"type": kind, "message": message}}
        if detail:
            body["error"]["maskerade"] = detail
        self._send_json(status, body,
                        {"X-Maskerade-Refused": kind})
        if started is not None:
            self._access(status, started, kind)

    def _client_headers(self):
        out = {}
        for k, v in self.headers.items():
            if k.lower() in FORWARD_HEADERS:
                out[k] = v
        return out

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise Refusal("request body is larger than %d bytes"
                          % MAX_BODY_BYTES)
        return self.rfile.read(length) if length else b""

    # ── routes ──

    def do_GET(self):
        started = time.time()
        path = self.path.split("?")[0]
        if path == "/healthz":
            self._send_json(200, {"ok": True,
                                  "mode": self.masker.pipeline.mode_string()})
            return self._access(200, started)
        if path in PASSTHROUGH_GET:
            return self._passthrough(started)
        self._refuse(404, "maskerade_unknown_route",
                     "this proxy does not serve %s. Only the Messages API and "
                     "a small set of read-only endpoints are proxied; "
                     "anything else is refused rather than forwarded blind."
                     % path, started=started)

    def do_POST(self):
        started = time.time()
        path = self.path.split("?")[0]
        if path == "/v1/messages":
            return self._messages(started)
        if path == "/v1/messages/count_tokens":
            return self._count_tokens(started)
        if path in PASSTHROUGH_POST:
            return self._passthrough_post(started, PASSTHROUGH_POST[path])
        self._refuse(404, "maskerade_unknown_route",
                     "this proxy does not serve %s." % path, started=started)

    # ── the main path ──

    def _messages(self, started):
        try:
            raw = self._read_body()
        except Refusal as e:
            return self._refuse(413, e.kind, str(e), started=started)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self._refuse(400, "maskerade_bad_request",
                                "request body is not valid JSON.",
                                started=started)
        del raw

        m = self.masker
        try:
            system = body.get("system")
            messages = body.get("messages") or []
            sid, continued, matched = m.store.resolve_prefix(
                system, messages, label=m.engagement)
        except StoreError as e:
            return self._refuse(400, "maskerade_store_error", str(e),
                                started=started)

        with m.lock_for(sid):
            try:
                masked_body, mask_map, info = m.mask_request(body)
            except F.RefusedField as e:
                # Renamed from `maskerade_unknown_field` 2026-10-17: there
                # is no longer any such thing. What reaches here is a
                # policy refusal -- bytes that cannot be masked -- or a
                # malformed body.
                return self._refuse(400, "maskerade_policy_refusal", str(e),
                                    {"field": e.path}, started)
            except Refusal as e:
                return self._refuse(400, e.kind, str(e), e.detail, started)
            except StoreError as e:
                return self._refuse(400, "maskerade_store_error", str(e),
                                    started=started)
            except Exception as e:
                # A detector that could not run. Never a pass-through.
                return self._refuse(
                    400, "maskerade_pipeline_error",
                    "the masking pipeline could not complete (%s: %s), so the "
                    "request was not sent."
                    % (type(e).__name__, e), started=started)

            headers = _info_headers(info)
            try:
                if body.get("stream"):
                    ok = self._proxy_stream(masked_body, mask_map, headers,
                                            started, info)
                else:
                    ok = self._proxy_once(masked_body, mask_map, headers,
                                          started, info)
            except urllib.error.URLError as e:
                return self._refuse(502, "maskerade_upstream_unreachable",
                                    "could not reach the provider: %s" % e,
                                    started=started)
            if ok:
                try:
                    m.persist(info["session"], mask_map, system, messages,
                              unscanned=info.get("unscanned"))
                except StoreError as e:
                    sys.stderr.write("  [STORE] %s\n" % e)

    def _count_tokens(self, started):
        """Counts the MASKED text.

        Counting the raw text would send the raw text to the provider, which
        is the one thing this proxy exists to prevent. It is also the more
        useful number: the client is budgeting against what will actually be
        sent.
        """
        return self._messages(started)

    # ── upstream ──

    def _upstream_request(self, payload):
        req = urllib.request.Request(
            UPSTREAM + self.path,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._client_headers(), method="POST")
        return urllib.request.urlopen(req, timeout=600)

    def _proxy_once(self, masked_body, mask_map, headers, started, info):
        try:
            resp = self._upstream_request(masked_body)
            status, raw = resp.status, resp.read()
        except urllib.error.HTTPError as e:
            # The provider's own error. Forwarded as-is: it is theirs to
            # explain, and it carries only masked text.
            raw, status = e.read(), e.code
            self._send_json(status, _safe_json(raw), headers)
            self._access(status, started, "upstream-error")
            return False

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send_json(502, {"type": "error", "error": {
                "type": "maskerade_bad_upstream",
                "message": "the provider's response was not JSON."}})
            self._access(502, started)
            return False

        payload, leftovers = _unmask_payload(payload, mask_map)
        if leftovers:
            headers["X-Maskerade-Leftover"] = ",".join(leftovers)
        self._send_json(status, payload, headers)
        self._access(status, started,
                     "session=%s ph=%d" % (info["session"],
                                           info["placeholders"]))
        return True

    def _proxy_stream(self, masked_body, mask_map, headers, started, info):
        try:
            resp = self._upstream_request(masked_body)
        except urllib.error.HTTPError as e:
            raw, status = e.read(), e.code
            self._send_json(status, _safe_json(raw), headers)
            self._access(status, started, "upstream-error")
            return False

        self.send_response(resp.status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()

        un = S.SSEUnmasker(mask_map)
        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                text = un.feed(chunk.decode("utf-8", "replace"))
                if text:
                    self._write_chunk(text)
            tail = un.finish()
            if tail:
                self._write_chunk(tail)
            self._write_chunk("")        # terminating zero-length chunk
        except (BrokenPipeError, ConnectionResetError):
            self._access(499, started, "client-disconnected")
            return False

        self._access(resp.status, started,
                     "session=%s ph=%d stream%s"
                     % (info["session"], info["placeholders"],
                        " leftover=%d" % len(un.leftovers)
                        if un.leftovers else ""))
        return True

    def _write_chunk(self, text):
        raw = text.encode("utf-8")
        self.wfile.write(b"%x\r\n" % len(raw) + raw + b"\r\n")
        self.wfile.flush()

    def _passthrough(self, started):
        req = urllib.request.Request(UPSTREAM + self.path,
                                     headers=self._client_headers())
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            raw, status = resp.read(), resp.status
        except urllib.error.HTTPError as e:
            raw, status = e.read(), e.code
        except urllib.error.URLError as e:
            return self._refuse(502, "maskerade_upstream_unreachable",
                                "could not reach the provider: %s" % e,
                                started=started)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        self._access(status, started, "passthrough")

    def _passthrough_post(self, started, upstream):
        """Forward a non-generative POST verbatim, to a named host.

        Nothing here parses, masks or logs the body. The one that exists is
        the OAuth token refresh a client on subscription auth sends when
        ANTHROPIC_BASE_URL points at this proxy, and its body holds the
        operator's refresh token.

        Headers go through the same FORWARD_HEADERS allowlist as everything
        else; see the note above that set for why the alternative was tried
        and dropped.

        The access log records the path and the status, as for any other
        request. It does not and must not record anything from the body.
        """
        try:
            raw = self._read_body()
        except Refusal as e:
            return self._refuse(413, e.kind, str(e), started=started)
        req = urllib.request.Request(upstream + self.path, data=raw,
                                     headers=self._client_headers(),
                                     method="POST")
        ctype = "application/json"
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            out, status = resp.read(), resp.status
            ctype = resp.headers.get("content-type") or ctype
        except urllib.error.HTTPError as e:
            out, status = e.read(), e.code
            ctype = e.headers.get("content-type") or ctype
        except urllib.error.URLError as e:
            return self._refuse(502, "maskerade_upstream_unreachable",
                                "could not reach %s: %s" % (upstream, e),
                                started=started)
        finally:
            del raw
        # The upstream's own content-type, because a WAF answers an HTML
        # page and labelling that `application/json` makes the client
        # report a parse error instead of the refusal it was handed.
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(out)))
        self.send_header("X-Maskerade-Passthrough", upstream)
        self.end_headers()
        self.wfile.write(out)
        self._access(status, started, "passthrough->%s" % upstream)


def _safe_json(raw):
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {"type": "error", "error": {
            "type": "upstream_error",
            "message": "the provider returned a non-JSON error."}}


def _unmask_payload(payload, mask_map):
    """Restore originals in a non-streaming response."""
    leftovers = set()
    for block in payload.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            block["text"] = S.unmask_text(block["text"], mask_map)
            leftovers.update(S.leftover_placeholders(block["text"]))
        elif block.get("type") == "thinking" and isinstance(
                block.get("thinking"), str):
            block["thinking"] = S.unmask_text(block["thinking"], mask_map)
        elif block.get("type") == "tool_use":
            block["input"] = _unmask_tree(block.get("input"), mask_map,
                                          leftovers)
    return payload, sorted(leftovers)


def _unmask_tree(node, mask_map, leftovers):
    if isinstance(node, str):
        out = S.unmask_text(node, mask_map)
        leftovers.update(S.leftover_placeholders(out))
        return out
    if isinstance(node, dict):
        return {k: _unmask_tree(v, mask_map, leftovers)
                for k, v in node.items()}
    if isinstance(node, list):
        return [_unmask_tree(v, mask_map, leftovers) for v in node]
    return node


def _info_headers(info):
    """Degradation reaches the client, or it does not exist."""
    h = {
        "X-Maskerade-Mode": info["mode"],
        "X-Maskerade-Session": "%s; %s" % (
            info["session"], "continued" if info["continued"] else "new"),
    }
    if info.get("rules"):
        # The operator must be able to see that their rules file is in
        # force without reading the daemon's stderr.
        h["X-Maskerade-Rules"] = str(info["rules"])
    if info.get("rules_error"):
        h["X-Maskerade-Rules-Error"] = "reload-failed"
    if info.get("system_note"):
        # The proxy added text the operator did not write. Say so on every
        # response rather than leaving it to be discovered.
        h["X-Maskerade-System-Note"] = "appended"
    if info["widened"]:
        h["X-Maskerade-Widened"] = str(info["widened"])
    if info["notes"]:
        h["X-Maskerade-Degraded"] = ",".join(
            "%s/%s" % (n.get("stage"), n.get("reason"))
            for n in info["notes"])
    if info.get("unscanned"):
        # The header that carries the new guarantee. Present on every
        # response that forwarded a field without inspecting it, which for
        # an agentic client is every response.
        h["X-Maskerade-Unscanned"] = _unscanned_header(info["unscanned"])
    return h


def _unscanned_header(names):
    """Deduplicated names, capped, with `+N more` when it does not fit.

    Capped twice -- by count and by characters -- because one long name
    would otherwise blow the budget that twenty short ones fit inside, and
    a header a client rejects reports nothing at all.
    """
    shown, dropped = list(names[:UNSCANNED_MAX_NAMES]), len(names)
    while shown:
        value = ", ".join(shown)
        if len(names) > len(shown):
            value += ", +%d more" % (len(names) - len(shown))
        if len(value) <= UNSCANNED_MAX_CHARS:
            return value
        shown.pop()
    return "+%d more" % dropped


# ──────────────────────────────────────────────────────────────── main ──

# What `maskerade init` writes, and the conventional location build()
# looks in when no --custom-rules is given. Project-local and inside the
# engagement directory, because this is the file an operator edits at 11pm
# when the engagement turns up a new hostname: discoverable beats tidy.
PROJECT_RULES = os.path.join(".maskerade", "rules.txt")


def _resolve_rules_path(args, store):
    """Where the operator's rules come from. Returns (path_or_None, why).

    Until 2026-10-18 the only answer was --custom-rules, so the one layer
    that can carry the client's own name was off unless the operator
    remembered a flag -- and org_evidence's ORGANIZATION arm is
    precision-first precisely because it assumes that layer is populated.
    `maskerade init` now writes a conventional location and this resolves
    it. First present wins, and the banner says which won:

      1. --custom-rules FILE   explicit. A missing path still RAISES,
                               unchanged: an operator who named a file and
                               mistyped it must not get a daemon running
                               without it.
      2. ./.maskerade/rules.txt            what `init` writes
      3. <state-dir>/engagements/<label>.rules.txt   with --engagement, for
                               a proxy started from some other directory
      4. none

    2 and 3 are used only IF PRESENT, and that is why neither can be a
    plain default on the argparse option: RulesFile(path) raises on a
    missing path, so a default would turn "no rules file" into "the daemon
    refuses to start".

    The label has already been validated by build() before this runs, so
    joining it into a file name here is safe.
    """
    if args.custom_rules:
        return args.custom_rules, "--custom-rules"
    local = os.path.join(os.getcwd(), PROJECT_RULES)
    if os.path.exists(local):
        return local, "found in the working directory"
    if args.engagement:
        p = os.path.join(store.root, "engagements",
                         "%s.rules.txt" % args.engagement)
        if os.path.exists(p):
            return p, "found for engagement %s" % args.engagement
    return None, "none"


def _warn_no_rules(args, store):
    """--engagement was given and no rules file resolved. Say so loudly.

    Every other degraded condition in this daemon is loud: [TIER] when a
    better tier is installed and is not what is running, [RULES] when a
    reload fails, [UNSCANNED] once per field forwarded without inspection.
    This one was the quietest thing in the process -- the word `none` on one
    banner line -- and it is precisely the condition `maskerade init` exists
    to prevent: a named engagement running with ZERO operator rules, which
    means the client's own name is left to the detectors.

    It is easy to reach by accident because _resolve_rules_path reads
    os.getcwd(): `maskerade init` writes .maskerade/rules.txt into the
    engagement directory, and a proxy started from anywhere else does not
    see it.

    WARNING, NOT A REFUSAL, and that is deliberate. An operator may mean to
    run a labelled session with no rules -- a scratch engagement, a
    reproduction, a client whose name is genuinely not in the traffic. A
    refusal that removes the tool from the path protects nobody, which is
    the whole reason the unknown-field refusal was
    abandoned on 2026-10-17. Fail-closed stays for a stage that cannot RUN;
    an absent operator rules file is not that.
    """
    local = os.path.join(os.getcwd(), PROJECT_RULES)
    state = os.path.join(store.root, "engagements",
                         "%s.rules.txt" % args.engagement)
    sys.stderr.write(
        "\n"
        "  [RULES] NO rules file found, and --engagement %s was given.\n"
        "\n"
        "          The custom stage is the ONLY layer that can carry the\n"
        "          client's own name, domains and tenant slugs -- no detector\n"
        "          knows them in advance, and org_evidence's ORGANIZATION arm\n"
        "          is precision-first BECAUSE it assumes this layer is\n"
        "          populated. This engagement is about to run with the\n"
        "          client's identity left to the detectors alone.\n"
        "\n"
        "          Looked for, in this order:\n"
        "            %s\n"
        "            %s\n"
        "\n"
        "          The first is relative to the directory this daemon was\n"
        "          STARTED FROM, which is why `maskerade init` prints a `cd`.\n"
        "\n"
        "          Either restart from the engagement directory, or name the\n"
        "          file:\n"
        "            maskerade proxy --engagement %s --custom-rules <path>\n"
        "\n"
        "          No rules file yet:  maskerade init --engagement %s\n"
        "\n"
        "          This is a WARNING. The proxy is starting and will mask; it\n"
        "          is the operator-supplied layer that is absent, and the\n"
        "          banner below says `custom rule none` for the same reason.\n"
        "\n"
        % (args.engagement, local, state, args.engagement, args.engagement))
    sys.stderr.flush()


def build(args):
    from maskerade import Pipeline

    store = SessionStore(root=args.state_dir,
                         retention_days=args.retention_days)
    for w in store.warnings:
        sys.stderr.write("  [STORE] WARNING: %s\n" % w)

    # Validate the label ONCE, at startup, through the store's own validator
    # rather than a second copy of its regex here. An operator who typed a
    # path separator finds out now instead of on the first request, and the
    # `<label>.rules.txt` file name built in _resolve_rules_path is then
    # safe. StoreError propagates to main(), which prints "cannot start".
    if args.engagement:
        store.resolve(None, [], label=args.engagement)

    n_s, n_i = store.purge_expired()
    if n_s or n_i:
        sys.stderr.write("  [STORE] retention %dd: removed %d session(s), "
                         "%d index entr%s\n"
                         % (store.retention_days, n_s, n_i,
                            "y" if n_i == 1 else "ies"))

    note = SystemNote.load(args.system_note,
                           enabled=not args.no_system_note)

    rules_path, rules_why = _resolve_rules_path(args, store)

    # Allowed to propagate: an operator who passed --custom-rules and
    # whose file does not parse must not get a daemon that silently runs
    # without the one layer carrying their client's identity.
    rules = RulesFile(rules_path)
    if rules_path:
        sys.stderr.write("  [RULES] %s  (%s)\n" % (rules.summary(), rules_why))
    elif args.engagement:
        _warn_no_rules(args, store)

    pipeline = Pipeline(config_path=args.config, tier=args.tier)

    # The [ner] extra is installed and the daemon is about to run WITHOUT
    # gliner, because nobody passed --tier full. That is under-masking
    # relative to what this machine can do, and under-masking is the one
    # direction this tool does not fail in silently. It is reported, never
    # auto-corrected: a tier chosen by what happens to be installed would
    # make two machines with the same config mask differently, with nothing
    # in the output to say which was which.
    #
    # find_spec only, so a light install pays nothing to answer this and the
    # no-torch gate stays green.
    if args.tier != "full":
        from maskerade._extras import ner_available
        if ner_available():
            sys.stderr.write(
                "  [TIER] running '%s'. maskerade[ner] IS installed, so "
                "tier 'full' (C2) is\n"
                "         available on this machine and is NOT what is "
                "running. Pass --tier full\n"
                "         if you meant C2.\n" % args.tier)

    # Allowed to propagate. A daemon that starts clean and refuses every
    # request is worse than one that refuses to start.
    pipeline.warmup()
    return Masker(pipeline, store, system_note=note, rules=rules,
                  engagement=args.engagement)


def main(argv=None):
    # Imported HERE rather than at module scope. server.py is imported by
    # `maskerade proxy`, which is the one subcommand allowed to pay for the
    # Pipeline import, but keeping it inside main() means that importing this
    # module for any other reason still costs nothing.
    from maskerade.pipeline import DEFAULT_TIER

    ap = argparse.ArgumentParser(prog="python -m maskerade.proxy.server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--allow-remote", action="store_true",
                    help="permit a non-loopback bind (see the refusal text)")
    # Follows the PACKAGE default rather than restating it. A second copy of
    # this default is a second thing to keep in step, and the flag whose
    # divergence would be least visible is exactly this one: the daemon would
    # run a different detector set from the library in the same install, and
    # both would report themselves as correct.
    ap.add_argument("--tier", choices=["fast", "full"], default=DEFAULT_TIER,
                    help="detector set to run (default %s). 'full' is C2 and "
                         "needs maskerade[ner]." % DEFAULT_TIER)
    ap.add_argument("--config", default=None)
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--retention-days", type=int, default=30)
    ap.add_argument("--engagement", default=None, metavar="NAME",
                    help="put every request in one named session. This is "
                         "what makes `maskerade sessions purge --session "
                         "NAME` delete exactly this engagement and nothing "
                         "else. `maskerade init --engagement NAME` sets it "
                         "up and writes the rules file this picks up.")
    ap.add_argument("--custom-rules", default=None, metavar="FILE",
                    help="literal values to mask, one per line. This is "
                         "how the client's own name, domains and tenant "
                         "slugs reach the masker; no detector can know "
                         "them in advance. Re-read when the file "
                         "changes. `maskerade rules check FILE` first.")
    ap.add_argument("--system-note", default=None, metavar="FILE",
                    help="replace the appended redaction contract with your "
                         "own file")
    ap.add_argument("--no-system-note", action="store_true",
                    help="append nothing. The model will then read "
                         "placeholders as unfilled templates and ask for the "
                         "real values.")
    args = ap.parse_args(argv)

    # The console is cp1252 on the owner's machine and a redirect inherits
    # it, so a single non-ASCII byte from a dependency kills the process.
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if args.host not in ("127.0.0.1", "::1", "localhost") \
            and not args.allow_remote:
        sys.stderr.write(
            "\n  REFUSING to bind %s.\n\n"
            "  This proxy holds the mask map and sees every value before it is\n"
            "  masked. Binding it to a reachable address puts an unauthenticated\n"
            "  unmasking service on the network, and the single property a\n"
            "  client can verify in ten seconds -- that it listens only on\n"
            "  loopback -- stops being true.\n\n"
            "  If you mean it, pass --allow-remote, and say so in the threat\n"
            "  model: it is a different deployment with a different boundary.\n"
            % args.host)
        return 2

    try:
        masker = build(args)
    except Exception as e:
        sys.stderr.write("\n  cannot start: %s: %s\n" % (type(e).__name__, e))
        return 2

    Handler.masker = masker
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True

    # The bind line used to say "(loopback only)" unconditionally, which is
    # false under --allow-remote -- and the container image runs with
    # exactly that flag, binding 0.0.0.0 INSIDE the container. There the
    # loopback guarantee is enforced by the port publish
    # (-p 127.0.0.1:8787:8787), not by this bind, and the banner says so
    # rather than repeating a claim the process cannot make about itself.
    if args.host in ("127.0.0.1", "::1", "localhost"):
        bind_note = "(loopback only)"
        client_host = args.host
    else:
        bind_note = (
            "(ALL INTERFACES: --allow-remote)\n"
            "                Reachability is decided by what surrounds this "
            "process --\n"
            "                in a container, the -p 127.0.0.1:%d:%d publish "
            "-- NOT by\n"
            "                this bind. Publish without 127.0.0.1: and this "
            "is an\n"
            "                unauthenticated unmasking service on your "
            "network." % (args.port, args.port))
        # 0.0.0.0 is not an address a client can connect to. For a wildcard
        # bind the useful hint is the host side of a loopback publish.
        client_host = ("127.0.0.1" if args.host in ("0.0.0.0", "::", "")
                       else args.host)

    sys.stderr.write(
        "\n  maskerade proxy\n"
        "    listening   http://%s:%d   %s\n"
        "    mode        %s\n"
        "    upstream    %s\n"
        "    mask maps   %s   (retention %d days)\n"
        "    engagement  %s\n"
        "    custom rule %s\n"
        "    system note %s\n\n"
        "    point a client at it:\n"
        "      $env:ANTHROPIC_BASE_URL = \"http://%s:%d\"\n\n"
        % (args.host, args.port, bind_note, masker.pipeline.mode_string(),
           UPSTREAM,
           masker.store.root, masker.store.retention_days,
           args.engagement or "none (derived from the conversation prefix)",
           masker.rules.summary(),
           masker.system_note.source,
           client_host, args.port))
    sys.stderr.flush()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n  stopped.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
