# -*- coding: utf-8 -*-
"""Which request fields get masked, which are forwarded, and which are named.

This module locates the strings in an Anthropic Messages API
request body that are the OPERATOR'S OWN CONTENT, so the server can mask them.
It also accounts for every other field in the body, so the operator can be told
what was forwarded without being inspected.

WHAT CHANGED, AND WHY (2026-10-17)
----------------------------------
This file used to be a closed specification whose default on an unrecognised
field was `RefusedField`. The argument for that is still true and is worth
restating rather than deleting:

    the Messages API gains fields over time, and what this code does with a
    field nobody has seen decides whether a future API version leaks by
    itself. A proxy written as "mask these seven paths, forward the rest" is
    correct on the day it ships and leaks on the day the provider adds the
    eighth.

That rule worked. Pointed at a real first-party client it caught `safeguards`,
an undocumented structured dump of the operator's machine, which no part of the
design had anticipated.

It also broke every agent. And a refusal does not protect the client: a
pentester whose client returns 400 runs that client WITHOUT the proxy and leaks
100%. A review of the field (redactproxy, PrivAiTe, PasteGuard, Cover, Kiji)
found that no competitor refuses, and none of them scans these fields either --
PrivAiTe explicitly does not scan `system`, `instructions`, `tools`, `metadata`
or `messages[].name`.

So the resolution is REPORTING, not refusing. The guarantee changes from

    "we refuse what we don't understand"                     (old)
to
    "we account for every field in the request, and we tell
     you which ones we did not inspect"                      (new)

and the second is checkable in one request, at no cost in agent compatibility.
The leak the old rule prevented is not prevented any more; it is NAMED, on the
response, on stderr, and in the session record. That is a weaker guarantee
honestly stated, which is the trade this project takes over a stronger
guarantee nobody can run.

THE FOUR CLASSES
----------------
Every key reached by the walk below lands in exactly one of these. There is no
fall-through: a key that is not MASK and not STRUCTURAL is reported, always.
That absence of a default is what makes the new claim mechanical rather than
aspirational.

    MASK        the operator's own content. Goes through the pipeline.
    STRUCTURAL  protocol machinery -- an enum, a provider-issued id, a number,
                a boolean -- forwarded verbatim, each justified where listed.
    UNSCANNED   forwarded verbatim AND recorded in `unscanned_out`. A known
                member carries a written reason; an unknown one carries its
                path and nothing else.
    REFUSED     raises RefusedField. Only for a MALFORMED body -- one the
                provider would reject too. There is no longer any such thing
                as an unknown-field refusal, and since 2026-10-31 no content
                refusal either: images and documents are forwarded and
                named (see ATTACHMENTS below).

WHAT IS MASKED IS DECIDED BY AUTHORSHIP, NOT BY CONTAINER
---------------------------------------------------------
The obvious rule -- "mask everything inside `messages`" -- is wrong twice, and
both cases are in real captured traffic:

 1. A first-party client sends a message with `role: "system"` whose content is
    the Environment block: the working directory, the OS build, the git state,
    and the full roster of agent types and skill names. The agent DISPATCHES on
    those names and resolves relative paths against that directory. Masking it
    changes the agent's behaviour, which this tool must never do. So the scope
    is by ROLE: `user` and `assistant` are content; any other role is machine
    configuration that happens to be delivered in the messages array.

 2. The same client injects a complete `<system-reminder>` block as the first
    text block of a user message. It is harness-authored, and it contains an
    email address and a URL that the detector fires on. Masked, the agent
    writes `Co-Authored-By: ... [EMAIL_ADDRESS_1]` into a real git commit --
    the placeholder leaves by a path the response-side unmask never sees. So a
    text block that is ENTIRELY a system-reminder is reported, not masked.

Case 2 is a content sniff and is a different kind of rule from everything else
here. It is deliberate, and it fails in the safe direction: if the marker
changes, the block is masked as before. It is reported like any other unscanned
field, so it is visible rather than silent.

WHAT IS NOT MASKED, AND THE RESIDUAL THAT CREATES
--------------------------------------------------
`system`, `tools`, `metadata`, `stop_sequences`, `thinking`,
`context_management`, `output_config`, `tool_choice` and `safeguards` are
forwarded unscanned. Masking a schema `enum` means the model cannot emit a
conforming value; masking the working directory in an agent's system prompt
breaks its path reasoning; masking a tool `description` changes what the model
believes the tool does.

The residual, stated here rather than left to be found:
with `system` and `safeguards` unscanned, the agent's working directory reaches
the provider. On an engagement run in `C:\\clients\\acmebank\\...` that is the
client's name in cleartext. This is a deliberate trade; the mitigation is the
`maskerade init --engagement` wizard, not masking.

A second residual, less obvious: an unknown BLOCK TYPE inside a user message
forwards its text unscanned. `server_tool_use.input` is the concrete case.
Masking the contents of a block this version does not understand could corrupt
a provider-issued token, so it is reported instead.

ATTACHMENTS (changed 2026-10-31)
--------------------------------
Images and PDF documents used to be refused, on the reasoning that their
bytes cannot be masked. Dogfooding showed what that costs: Claude Code
resends the whole conversation every turn, so ONE pasted screenshot, or one
image file the agent reads, refused every later request of the
conversation. That is the same failure the 2026-10-17 change describes --
an operator who cannot work through the proxy works without it. So they are
now forwarded as they are and NAMED ("image", "document"), like every other
field this tool does not inspect. The owner's decision, with the residual
stated: a screenshot of a client system reaches the provider unmasked.

What CAN be masked in a document still is: a plain-text source (`data`
under `"type": "text"`), a content-blocks source, and the `title` and
`context` strings. A PDF, a URL source and a file reference are forwarded.

A later version adds an opt-in `--scan tools,system` flag. The classification
below is a data table for exactly that reason: adding it is a flag, not a
rewrite. SCAN_FLAG_CANDIDATES names the members it would cover.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not mask. It locates. `extract()` returns the strings and where they
came from; `inject()` puts replacements back. The masking policy lives in the
server, because it needs one property this module should not know about:

    the conversation prefix was already masked on an earlier turn, and its
    map is in the session store. Re-DETECTING it every turn would re-run the
    detector over the whole history -- at the measured p50 of 314 ms per
    message that is seconds per turn, climbing with conversation length, for
    an answer already on disk. The server detects only in content it has not
    seen, and applies the stored map to the rest.

A consequence worth stating: detection sees one message at
a time rather than the whole conversation, so a client name established in
turn 1 and mentioned bare in turn 9 is corroborated only if the stored map
already holds it. In practice the map is exactly that memory.
"""
import copy


class RefusedField(Exception):
    """A POLICY refusal: something this tool will not forward because it
    cannot mask it, or could not put a replacement back.

    Carries the JSON path so the refusal names the thing to look at. The
    server turns this into a 4xx that reads as "the masker refused", never
    as "the provider is flaky".

    An unrecognised field is NOT one of these, and since 2026-10-31 neither
    is an image or a document: both are forwarded and named on the response
    in X-Maskerade-Unscanned. If this exception fires, the request is
    malformed -- not a JSON object, a messages array that is not an array,
    a block with no type -- and the provider would reject it as well.
    """

    def __init__(self, path, detail=""):
        self.path = path
        super().__init__(
            "refused at %s%s. The request is malformed, so it was not "
            "sent. Unknown fields, images and documents are forwarded and "
            "named in the X-Maskerade-Unscanned response header; only a "
            "body this tool cannot walk is refused."
            % (path or "<root>", (": " + detail) if detail else ""))


def _p(path, key):
    return "%s.%s" % (path, key) if path else str(key)


# ── The unscanned record ─────────────────────────────────────────────────
#
# (path, name, reason). The PATH is exact and internal -- `messages[3].foo`
# -- because that is what an operator needs to go and look at. The NAME is
# the deduplicated short label that reaches the response header and the
# session record, because an HTTP header cannot carry 350 paths. The REASON
# is a written sentence for a field this version knows about, and None for
# one it has never seen.

def _note(un, path, name, reason=None):
    """Record a forwarded-but-uninspected field. `un` may be None."""
    if un is None:
        return
    un.append((path, name, reason))


def short_names(unscanned):
    """Deduplicated, sorted NAMES from an unscanned_out list.

    One implementation, used by the response header, the stderr warning and
    the session record alike. A second copy of this would drift, and the
    three would disagree about what was reported.
    """
    seen = set()
    for rec in unscanned or []:
        if rec and len(rec) > 1 and rec[1]:
            seen.add(rec[1])
    return sorted(seen)


# ── Top level ────────────────────────────────────────────────────────────
#
# STRUCTURAL. Numbers, booleans and closed enums. None of these can carry a
# value the operator typed, so none can carry client data, and none is worth
# reporting -- reporting them would bury the fields that matter.
TOP_STRUCTURAL = {
    "model",            # a provider model id
    "max_tokens", "temperature", "top_p", "top_k",
    "stream",           # bool
    "service_tier",     # enum
    "anthropic_version", "anthropic_beta",
}

# UNSCANNED, by name, with the reason reported alongside. Data-driven so the
# future `--scan` flag is a lookup rather than a rewrite.
TOP_UNSCANNED = {
    "system":
        "the client's own system prompt. It holds the agent's instructions "
        "and, for an agentic client, its working directory and environment. "
        "Masking it changes the agent's behaviour",
    "tools":
        "tool definitions: names, descriptions and JSON Schemas. The model "
        "dispatches on these and must emit values conforming to them, so a "
        "masked description or enum breaks the agent",
    "tool_choice":
        "which tool the model must call. A tool name, not operator content",
    "metadata":
        "client-supplied request metadata. For an agentic client this is its "
        "own device and session identity, not the operator's prose",
    "stop_sequences":
        "operator-chosen stop strings. Forwarded so the sequence the "
        "provider stops on is the one the client asked for",
    "thinking":
        "extended-thinking configuration. A discriminator and an integer "
        "budget",
    "context_management":
        "server-side history-editing configuration. Discriminators, integer "
        "thresholds and tool names",
    "output_config":
        "effort level and structured-output schema. The schema is the "
        "model's output contract, in the same class as `tools`",
    "safeguards":
        "the client's dangerous-tool-use classifier context: working and "
        "home directories, trusted directories, permission rules, git state "
        "and the operator's account name. Masking it would degrade a safety "
        "control that runs on the operator's own machine. THIS IS THE "
        "LARGEST SINGLE RESIDUAL -- see `maskerade init`",
}

# What an opt-in `--scan` flag would be able to turn on. Each has a walk in
# this file that still works; see "code that is dormant" below.
SCAN_FLAG_CANDIDATES = {"system", "tools", "metadata", "stop_sequences",
                        "output_config"}

# Reasons used in more than one place.
R_REMINDER = ("a harness-authored <system-reminder> block injected into the "
              "message. Masking it corrupts text the agent copies verbatim, "
              "such as a commit trailer")
R_ROLE = ("a message whose role is neither user nor assistant. For an "
          "agentic client this carries the machine environment the agent "
          "reasons about -- working directory, platform, agent and skill "
          "names it dispatches on")
R_BLOCK = "an unrecognised key on a content block"
R_BLOCKTYPE = ("a content block type this version does not recognise. Its "
               "contents may be provider-issued, so they are reported "
               "rather than masked")
R_MESSAGE = "an unrecognised key on a message"
R_IMAGE = ("an image. Its pixels cannot be masked, so it is forwarded as it "
           "is: a screenshot of a client system reaches the provider "
           "unmasked")
R_DOCUMENT = ("a document whose bytes cannot be masked -- a PDF, a URL or a "
              "file reference -- forwarded as it is. A plain-text document's "
              "text IS masked")
R_OPAQUE = "opaque `data` on a block that is not redacted_thinking"
R_UNKNOWN = None        # a top-level field this version has never seen


# ── Blocks ───────────────────────────────────────────────────────────────
BLOCK_STRUCTURAL = {
    "type",             # the discriminator itself
    "cache_control",    # {"type": "ephemeral"}
    "is_error",         # bool
    "id",               # tool_use id, provider- or client-generated handle
    "tool_use_id",      # the matching handle
    "index",
    # Provider-issued and verified on return. Masking breaks the request,
    # and neither carries operator input.
    "signature",
    "data",             # opaque on redacted_thinking; reported elsewhere
}

# The marker that identifies harness-authored text injected into a message.
# Matched on the WHOLE stripped block, never on a substring: an operator who
# quotes the phrase mid-prompt is still masked.
_REMINDER_OPEN = "<system-reminder>"
_REMINDER_CLOSE = "</system-reminder>"


def is_system_reminder(text):
    """True when `text` is entirely one harness-authored reminder block."""
    if not isinstance(text, str):
        return False
    s = text.strip()
    return (s.startswith(_REMINDER_OPEN) and s.endswith(_REMINDER_CLOSE)
            and len(s) > len(_REMINDER_OPEN) + len(_REMINDER_CLOSE))


# ── Inside a JSON Schema ─────────────────────────────────────────────────
#
# DORMANT since 2026-10-17. `tools[]` and `output_config.format.schema` are
# no longer scanned, so nothing in the shipping walk reaches `_schema` or
# the five sets below. They are kept, not deleted, for two reasons: the
# code is correct (it fixed a latent break where any schema using
# `$ref`/`$defs` refused), and it is exactly what the opt-in `--scan tools`
# option needs. The self-test exercises them directly for the same reason.
#
# The keywords are split by SHAPE rather than kept in one set, because
# "forward this value" and "forward this key and walk into its value" are
# different behaviours and conflating them is how `$defs` came to be listed
# and still refuse.

# Value is a subschema, or an array of subschemas. Recursed into.
SCHEMA_APPLICATORS = {
    "items", "prefixItems", "additionalItems", "contains",
    "additionalProperties", "unevaluatedProperties", "unevaluatedItems",
    "anyOf", "oneOf", "allOf", "not",
    "if", "then", "else",
    "propertyNames", "contentSchema",
}

# Value is a keyword, a number, a boolean, or an array of property names.
# None of these can carry a string the operator typed.
#
# `$ref`, `$id`, `$anchor` and `$dynamic*` are the deliberate exception:
# they are URI references, and `$ref` must stay byte-identical to the
# `$id`/`$anchor` it resolves against.
SCHEMA_ASSERTIONS = {
    "$schema", "$id", "$anchor", "$dynamicAnchor",
    "$ref", "$dynamicRef", "$vocabulary",
    "type", "required", "nullable", "format", "pattern",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf",
    "minLength", "maxLength",
    "minItems", "maxItems", "uniqueItems",
    "minContains", "maxContains",
    "minProperties", "maxProperties",
    "dependentRequired",
    "contentMediaType", "contentEncoding",
    "readOnly", "writeOnly", "deprecated",
}

# Value is a map from an author-chosen NAME to a subschema. The names are
# the tool's own vocabulary and are forwarded; only the values are walked.
SCHEMA_SUBSCHEMA_MAPS = {
    "properties", "patternProperties", "dependentSchemas",
    "$defs", "definitions",
}

# Author-written free text.
SCHEMA_TEXT = {"description", "title", "$comment"}

# Author-supplied VALUES.
SCHEMA_VALUES = {"enum", "const", "default", "examples"}

# Retained so an importer that referenced the old name still resolves. It
# is the union of the two forwarding sets and is not consulted by _schema.
SCHEMA_STRUCTURAL = SCHEMA_APPLICATORS | SCHEMA_ASSERTIONS


# ─────────────────────────────────────────────────────────────── extract ──

def extract(body, unscanned_out=None):
    """Return a list of (path, text) for every string that must be masked.

    `unscanned_out`, when given, is a list this walk APPENDS to: one
    (path, name, reason) record per field forwarded without being
    inspected. It is an out-parameter rather than a second return value so
    that `inject()`, which re-walks and compares the returned paths for
    count and order, is completely unaffected -- and so that there is one
    walk rather than two that drift.

    Raises RefusedField only on a policy refusal or a malformed body.
    """
    out = []
    if not isinstance(body, dict):
        raise RefusedField("", "request body is not a JSON object")

    for key, val in body.items():
        if key in TOP_STRUCTURAL:
            continue
        if key == "messages":
            _messages(val, "messages", out, unscanned_out)
        elif key in TOP_UNSCANNED:
            _note(unscanned_out, key, key, TOP_UNSCANNED[key])
        else:
            # A field this version has never seen. Forwarded, and named --
            # this is the path that `safeguards` took before it had an
            # entry above, and the path the next such field will take.
            _note(unscanned_out, key, key, R_UNKNOWN)
    return out


def _contains_string(v):
    """DORMANT. Reached only by `_schema`, which is itself dormant.

    It answered "would forwarding this value leak a string?", which was the
    question the refuse-by-default rule asked. The reporting rule does not
    ask it: a field is reported whether or not it happens to carry a string
    today, because a field that carries none now may carry one next week
    and the report must not quietly change.
    """
    if isinstance(v, str):
        return True
    if isinstance(v, dict):
        return any(_contains_string(x) for x in v.values()) or any(
            isinstance(k, str) and k for k in v.keys())
    if isinstance(v, list):
        return any(_contains_string(x) for x in v)
    return False


def _leaf(path, value, out, un):
    """One string that is either operator content, or harness text.

    The single place the system-reminder rule is applied, so the bare-string
    and block forms of message content cannot disagree about it.
    """
    if is_system_reminder(value):
        _note(un, path, "system-reminder", R_REMINDER)
        return
    out.append((path, value))


def _text_or_blocks(val, path, out, un):
    if val is None:
        return
    if isinstance(val, str):
        _leaf(path, val, out, un)
        return
    if isinstance(val, list):
        for i, blk in enumerate(val):
            _block(blk, "%s[%d]" % (path, i), out, un)
        return
    raise RefusedField(path, "expected a string or a block array")


# Roles whose content is the operator's own. Anything else in the messages
# array is machine configuration delivered in a content envelope -- see the
# module header.
CONTENT_ROLES = {"user", "assistant"}


def _messages(val, path, out, un):
    if not isinstance(val, list):
        raise RefusedField(path, "expected an array")
    for i, msg in enumerate(val):
        mp = "%s[%d]" % (path, i)
        if not isinstance(msg, dict):
            raise RefusedField(mp, "expected an object")

        role = msg.get("role")
        if role not in CONTENT_ROLES:
            # Reported by role so the name says WHICH role was skipped; a
            # bare "messages" would be indistinguishable from the array
            # itself having been skipped, which never happens.
            _note(un, mp, "messages(role=%s)" % (role,), R_ROLE)
            continue

        for key, v in msg.items():
            if key == "role":
                continue
            if key == "content":
                _text_or_blocks(v, _p(mp, "content"), out, un)
            else:
                # Includes `output_config`, which is per-message effort and
                # an output schema: configuration, in the same class as its
                # top-level twin, not content.
                _note(un, _p(mp, key), key,
                      TOP_UNSCANNED.get(key, R_MESSAGE))


def _block(blk, path, out, un):
    if isinstance(blk, str):
        _leaf(path, blk, out, un)
        return
    if not isinstance(blk, dict):
        raise RefusedField(path, "expected an object")

    btype = blk.get("type")
    if not isinstance(btype, str):
        raise RefusedField(_p(path, "type"), "block has no type discriminator")

    known = btype in ("text", "thinking", "redacted_thinking", "tool_use",
                      "tool_result", "image", "document")

    for key, v in blk.items():
        if key in BLOCK_STRUCTURAL:
            # `data` is opaque on redacted_thinking and expected there. On
            # any other block it is not something this walk understands:
            # forwarded and named, not refused (2026-10-31).
            if key == "data" and btype != "redacted_thinking":
                _note(un, _p(path, "data"), "data", R_OPAQUE)
            continue
        if btype == "text" and key == "text":
            if not isinstance(v, str):
                raise RefusedField(_p(path, "text"), "expected a string")
            _leaf(_p(path, "text"), v, out, un)
        elif btype == "thinking" and key == "thinking":
            if not isinstance(v, str):
                raise RefusedField(_p(path, "thinking"), "expected a string")
            _leaf(_p(path, "thinking"), v, out, un)
        elif btype == "tool_use" and key == "name":
            continue                 # the tool's own name, not client data
        elif btype == "tool_use" and key == "input":
            _free(v, _p(path, "input"), out, un)
        elif btype == "tool_result" and key == "content":
            # THE path that matters. In agentic use this carries the `Read`
            # of a config file, nmap output, a Burp export.
            _text_or_blocks(v, _p(path, "content"), out, un)
        elif btype == "image" and key == "source":
            # Forwarded and named since 2026-10-31; see ATTACHMENTS.
            _note(un, _p(path, "source"), "image", R_IMAGE)
        elif btype == "document" and key == "source":
            _document_source(v, _p(path, "source"), out, un)
        elif btype == "document" and key in ("title", "context") \
                and isinstance(v, str):
            _leaf(_p(path, key), v, out, un)
        else:
            _note(un, _p(path, key), key,
                  R_BLOCK if known else R_BLOCKTYPE)


def _document_source(src, path, out, un):
    """What of a document's source can be masked is masked; the rest is named.

    "text"     plain text in `data`: masked, like any message text.
    "content"  a list of content blocks (or a string): walked like a
               message's content, so its text is masked and any image in it
               is named.
    anything else -- "base64" (a PDF), "url", "file" -- is bytes or a
               reference this tool cannot mask: forwarded and named. A URL
               is not masked on purpose: the provider fetches it, and a
               masked URL fetches nothing.
    """
    stype = src.get("type") if isinstance(src, dict) else None
    if stype == "text" and isinstance(src.get("data"), str):
        _leaf(_p(path, "data"), src["data"], out, un)
    elif stype == "content" and "content" in src:
        _text_or_blocks(src["content"], _p(path, "content"), out, un)
    else:
        _note(un, path, "document", R_DOCUMENT)


# A tool-input key is author-chosen, and the path string that inject() uses
# to find it again is "."- and "[n]"-delimited. These characters are
# escaped in a key so that any key round-trips. Until 2026-10-31 a key
# containing one was REFUSED, which -- like every refusal in a resent
# history -- blocked the rest of the conversation.
_ESCAPES = (("\\", "\\\\"), (".", "\\."), ("[", "\\["), ("]", "\\]"))


def _esc(key):
    if key == "":
        return "\\0"              # an empty key, which _tokens() reads back
    for raw, escaped in _ESCAPES:
        key = key.replace(raw, escaped)
    return key


def _free_seg(path, key):
    """Guard an author-chosen path segment before it enters a path string.

    `inject` finds a field by re-parsing the path with `_set`, which splits
    on "." and reads "[...]" as an index. A tool-input key containing any of
    those characters makes `_set` raise ValueError on int(...) or assign
    into the wrong node: fail-closed turns into fail-crash, and the server
    reports it as a pipeline error that names nothing useful.

    Since 2026-10-31 the key is ESCAPED rather than refused (see _esc), so
    `{"a.b": ...}`, `{"x[0]": ...}` and `{"": ...}` are masked like any
    other value. A non-string key cannot come from JSON; it is stringified.
    """
    return _esc(key if isinstance(key, str) else str(key))


def _free(val, path, out, un):
    """Arbitrary JSON: every string is a value, so every string is masked.

    This is a tool's input, built by the model out of text it was shown. The
    keys are the tool's schema and stay; the values are whatever the model
    put there, which on this path is masked text being handed back. The
    response-side unmask restores the real value before the client executes
    the call.
    """
    if isinstance(val, str):
        _leaf(path, val, out, un)
    elif isinstance(val, dict):
        for k, v in val.items():
            _free(v, _p(path, _free_seg(path, k)), out, un)
    elif isinstance(val, list):
        for i, v in enumerate(val):
            _free(v, "%s[%d]" % (path, i), out, un)


# ────────────────────────────────────────── dormant: the schema walk ──
#
# Nothing in `extract()` reaches anything below this line. See the note on
# SCHEMA_APPLICATORS. Kept because it is correct, and because `--scan tools`
# is a flag that calls `_tools` rather than a rewrite that re-derives it.
# The self-test calls these directly so they cannot rot unnoticed.

def _tools(val, path, out, un=None):
    """DORMANT. The `--scan tools` walk."""
    if not isinstance(val, list):
        raise RefusedField(path, "expected an array")
    for i, tool in enumerate(val):
        tp = "%s[%d]" % (path, i)
        if not isinstance(tool, dict):
            raise RefusedField(tp, "expected an object")
        for key, v in tool.items():
            if key in ("name", "type", "cache_control"):
                continue
            if key == "description":
                if not isinstance(v, str):
                    raise RefusedField(_p(tp, "description"))
                out.append((_p(tp, "description"), v))
            elif key == "input_schema":
                _schema(v, _p(tp, "input_schema"), out)
            else:
                _note(un, _p(tp, key), key, "an unrecognised key on a tool")


def _seg(path, key, name):
    """DORMANT. The schema-side twin of `_free_seg`; see that docstring."""
    if not isinstance(name, str):
        raise RefusedField(_p(path, key), "schema map key is not a string")
    if not name:
        raise RefusedField(_p(path, key), "schema map key is empty")
    bad = [c for c in (".", "[", "]") if c in name]
    if bad:
        raise RefusedField(
            _p(path, key),
            "map key %r contains %s, which the path parser used by inject() "
            "would mis-read. Refused rather than injected into the wrong "
            "field" % (name, " and ".join(repr(c) for c in bad)))
    return name


def _schema(val, path, out, un=None):
    """DORMANT. Walks a JSON Schema, masking author-written text."""
    if isinstance(val, list):
        for i, v in enumerate(val):
            _schema(v, "%s[%d]" % (path, i), out, un)
        return
    if not isinstance(val, dict):
        return
    for key, v in val.items():
        if key in SCHEMA_SUBSCHEMA_MAPS:
            if isinstance(v, dict):
                for pk, pv in v.items():
                    _schema(pv, "%s.%s.%s" % (path, key, _seg(path, key, pk)),
                            out, un)
            continue
        if key in SCHEMA_TEXT:
            if isinstance(v, str):
                out.append((_p(path, key), v))
            continue
        if key in SCHEMA_VALUES:
            _free(v, _p(path, key), out, un)
            continue
        if key in SCHEMA_APPLICATORS:
            if isinstance(v, (dict, list)):
                _schema(v, _p(path, key), out, un)
            continue
        if key in SCHEMA_ASSERTIONS:
            # Forwarded, never recursed into: `dependentRequired` and
            # `required` hold property NAMES, and walking them as if they
            # were subschemas is what made `$defs` refuse.
            continue
        _note(un, _p(path, key), key, "an unrecognised JSON Schema keyword")


# `output_config`, top-level or on one message. DORMANT.
OUTPUT_CONFIG_KEYS = {
    "effort",   # "low" | "medium" | "high" | "xhigh" | "max"
    "format",   # {"type": "json_schema", "schema": <JSON Schema>}
}

OUTPUT_FORMAT_KEYS = {"type", "schema"}


def _output_config(val, path, out, un=None):
    """DORMANT. The `--scan output_config` walk."""
    if not isinstance(val, dict):
        raise RefusedField(path, "expected an object")
    for key, v in val.items():
        if key not in OUTPUT_CONFIG_KEYS:
            _note(un, _p(path, key), key, "an unrecognised output_config key")
            continue
        if key == "effort":
            if not isinstance(v, str):
                raise RefusedField(_p(path, "effort"), "expected a string")
            continue
        if not isinstance(v, dict):
            raise RefusedField(_p(path, "format"), "expected an object")
        for fkey, fv in v.items():
            if fkey not in OUTPUT_FORMAT_KEYS:
                _note(un, _p(_p(path, "format"), fkey), fkey,
                      "an unrecognised output_config.format key")
                continue
            if fkey == "type":
                if not isinstance(fv, str):
                    raise RefusedField(
                        _p(_p(path, "format"), "type"), "expected a string")
                continue
            _schema(fv, _p(_p(path, "format"), "schema"), out, un)


def _tool_choice(val, path, un=None):
    """DORMANT."""
    if not isinstance(val, dict):
        raise RefusedField(path, "expected an object")
    for key in val:
        if key not in ("type", "name", "disable_parallel_tool_use"):
            _note(un, _p(path, key), key, "an unrecognised tool_choice key")


def _thinking_config(val, path, un=None):
    """DORMANT."""
    if not isinstance(val, dict):
        raise RefusedField(path, "expected an object")
    for key in val:
        # The KEY is pinned and the VALUE is not enumerated: the published
        # docs list "summarized" | "omitted" for `display`, and the observed
        # wire value is "updates". An enumeration taken from the docs would
        # have refused a legal request.
        if key not in ("type", "budget_tokens", "display"):
            _note(un, _p(path, key), key, "an unrecognised thinking key")


# Inside one `edits[]` entry. DORMANT.
CONTEXT_EDIT_KEYS = {
    "type", "trigger", "keep", "clear_at_least",
    "exclude_tools", "clear_tool_inputs",
}

CONTEXT_THRESHOLD_KEYS = {"type", "value"}


def _context_management(val, path, un=None):
    """DORMANT. Extracts nothing: there is no operator content in it."""
    if not isinstance(val, dict):
        raise RefusedField(path, "expected an object")
    for key, v in val.items():
        if key != "edits":
            _note(un, _p(path, key), key,
                  "an unrecognised context_management key")
            continue
        if not isinstance(v, list):
            raise RefusedField(_p(path, "edits"), "expected an array")
        for i, edit in enumerate(v):
            ep = "%s.edits[%d]" % (path, i)
            if not isinstance(edit, dict):
                raise RefusedField(ep, "expected an object")
            if not isinstance(edit.get("type"), str):
                raise RefusedField(
                    _p(ep, "type"), "context edit has no type discriminator")
            for ekey, ev in edit.items():
                if ekey not in CONTEXT_EDIT_KEYS:
                    _note(un, _p(ep, ekey), ekey,
                          "an unrecognised context_management edit key")
                    continue
                if ekey in ("trigger", "keep", "clear_at_least"):
                    if isinstance(ev, str):
                        continue           # `keep: "all"` on clear_thinking
                    if not isinstance(ev, dict):
                        raise RefusedField(
                            _p(ep, ekey), "expected an object or a string")
                    for tkey in ev:
                        if tkey not in CONTEXT_THRESHOLD_KEYS:
                            _note(un, _p(_p(ep, ekey), tkey), tkey,
                                  "an unrecognised threshold key")
                elif ekey == "exclude_tools":
                    if not isinstance(ev, list):
                        raise RefusedField(
                            _p(ep, "exclude_tools"), "expected an array")
                elif ekey == "clear_tool_inputs":
                    if not isinstance(ev, bool):
                        raise RefusedField(
                            _p(ep, "clear_tool_inputs"), "expected a boolean")


def _metadata(val, path, out, un=None):
    """DORMANT. The `--scan metadata` walk.

    `user_id` reaches the provider verbatim. For an SDK caller who sets it
    by hand it can be an internal identifier and masking it is right; for an
    agentic client it is that client's own device and session identity and
    masking it does nothing useful. The field cannot tell them apart, so the
    shipping default is to forward and report, and an operator who needs the
    first case covers it with a rules-file entry.
    """
    if not isinstance(val, dict):
        raise RefusedField(path, "expected an object")
    for key, v in val.items():
        if key != "user_id":
            _note(un, _p(path, key), key, "an unrecognised metadata key")
            continue
        if v is None:
            continue
        if not isinstance(v, str):
            raise RefusedField(_p(path, "user_id"))
        out.append((_p(path, "user_id"), v))


# ───────────────────────────────────────────────────────────── inject ──

def inject(body, replacements):
    """Return a deep copy of `body` with each extracted path replaced.

    `replacements` is the list `extract()` returned, with the second element
    swapped for the masked text. The paths are re-walked rather than parsed:
    a path parser would be a second, divergent implementation of the walk
    above, and the two would drift.

    No `unscanned_out` is passed, deliberately. This call exists to
    reproduce the path list, and collecting the report twice would double
    every name in it.
    """
    out = copy.deepcopy(body)
    paths = [p for p, _ in extract(out)]
    if len(paths) != len(replacements):
        raise RefusedField(
            "", "field count changed between extract and inject "
                "(%d -> %d)" % (len(paths), len(replacements)))
    for (path, _old), (rpath, new) in zip(
            [(p, None) for p in paths], replacements):
        if path != rpath:
            raise RefusedField(
                "", "field order changed between extract and inject "
                    "(%s != %s)" % (path, rpath))
        _set(out, path, new)
    return out


def _tokens(path):
    """Parse a path from extract(): keys split on ".", indices in "[n]".

    A backslash escapes the next character, so a key may contain ".", "["
    or "]"; a whole segment "\\0" is the empty key. See _esc().
    """
    tokens, name, have = [], [], False
    i, n = 0, len(path)
    while i < n:
        c = path[i]
        if c == "\\" and i + 1 < n:
            d = path[i + 1]
            if (d == "0" and not have
                    and (i + 2 == n or path[i + 2] in ".[")):
                have = True             # the empty key
            else:
                name.append(d)
                have = True
            i += 2
            continue
        if c == "." or c == "[":
            if have:
                tokens.append(("k", "".join(name)))
            name, have = [], False
            if c == "[":
                j = path.index("]", i)
                tokens.append(("i", int(path[i + 1:j])))
                i = j + 1
            else:
                i += 1
            continue
        name.append(c)
        have = True
        i += 1
    if have:
        tokens.append(("k", "".join(name)))
    return tokens


def _set(root, path, value):
    """Walk a dotted/indexed path and assign. Paths come from extract()."""
    node = root
    tokens = _tokens(path)
    for kind, tok in tokens[:-1]:
        node = node[tok]
    kind, tok = tokens[-1]
    node[tok] = value


# ─────────────────────────────────────────────────────────── self-test ──

def _self_test():
    failures = []

    def check(cond, label):
        print(("ok   " if cond else "FAIL ") + label)
        if not cond:
            failures.append(label)

    def refuses(body, label):
        """A POLICY refusal. The only kind left."""
        try:
            extract(body)
        except RefusedField as e:
            print("ok   %s  (refused %s)" % (label, e.path or "<root>"))
            return
        print("FAIL " + label)
        failures.append(label)

    def reports(body, want_name, label, want_path=None):
        """Forwarded AND reported -- the behaviour that replaces a refusal.

        Asserts both halves. "did not refuse" alone would pass for a field
        that was silently dropped on the floor, which is the failure this
        whole step exists to prevent.
        """
        un = []
        try:
            extract(body, un)
        except RefusedField as e:
            print("FAIL %s  (refused %s)" % (label, e.path))
            failures.append(label)
            return
        except Exception as e:
            print("FAIL %s  (%s: %s)" % (label, type(e).__name__, e))
            failures.append(label)
            return
        names = short_names(un)
        if want_name not in names:
            print("FAIL %s  (reported %r, wanted %r)"
                  % (label, names, want_name))
            failures.append(label)
            return
        if want_path is not None and want_path not in [r[0] for r in un]:
            print("FAIL %s  (paths %r lack %r)"
                  % (label, [r[0] for r in un], want_path))
            failures.append(label)
            return
        print("ok   %s  (reported %s)" % (label, want_name))

    # ── 1. the masked surface ───────────────────────────────────────────
    print("-- what is masked --")

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "stream": True,
        "system": "you are a senior pentester in C:\\clients\\acmebank",
        "messages": [
            {"role": "user", "content": "Host: portal.acmebank.cz"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "the host is internal",
                 "signature": "AAAAsig=="},
                {"type": "text", "text": "try IDOR"},
                {"type": "tool_use", "id": "tu_1", "name": "send_request",
                 "input": {"host": "portal.acmebank.cz", "n": 3}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1",
                 "content": [{"type": "text",
                              "text": "jan.novak@acmebank.cz"}]},
            ]},
        ],
        "tools": [{
            "name": "send_request",
            "description": "send a request to portal.acmebank.cz",
            "input_schema": {
                "type": "object",
                "properties": {
                    "host": {"type": "string",
                             "description": "target host",
                             "enum": ["portal.acmebank.cz"]},
                },
                "required": ["host"],
            },
        }],
        "stop_sequences": ["END-ACMEBANK"],
        "metadata": {"user_id": "op-4471"},
    }

    un = []
    got = extract(body, un)
    texts = [t for _, t in got]

    check("Host: portal.acmebank.cz" in texts, "plain string content masked")
    check("the host is internal" in texts, "thinking text masked")
    check("try IDOR" in texts, "text block masked")
    check("portal.acmebank.cz" in texts, "tool_use input VALUE masked")
    check("jan.novak@acmebank.cz" in texts,
          "tool_result nested text masked (THE path that matters)")
    check("AAAAsig==" not in texts,
          "thinking.signature is NOT masked (provider-issued, verified)")
    check("send_request" not in texts, "tool name is not masked")
    check("tu_1" not in texts, "tool_use id is not masked")
    check(3 not in texts, "non-string tool input values are untouched")

    # The four that used to be masked and are now reported instead.
    names = short_names(un)
    check("you are a senior pentester in C:\\clients\\acmebank" not in texts,
          "system is NOT masked")
    check("send a request to portal.acmebank.cz" not in texts,
          "tool description is NOT masked")
    check("target host" not in texts, "schema description is NOT masked")
    check("END-ACMEBANK" not in texts, "stop sequence is NOT masked")
    check("op-4471" not in texts, "metadata.user_id is NOT masked")
    check(names == ["metadata", "stop_sequences", "system", "tools"],
          "all four are reported by name (got %r)" % (names,))
    check(all(r[2] for r in un),
          "every known unscanned field carries a written reason")
    check(len(got) == 5, "exactly the 5 message strings extracted (got %d)"
          % len(got))

    # ── 2. round trip ───────────────────────────────────────────────────
    masked = [(p, "[MASKED_%d]" % i) for i, (p, _) in enumerate(got)]
    rebuilt = inject(body, masked)
    again = [t for _, t in extract(rebuilt)]
    check(all(t.startswith("[MASKED_") for t in again),
          "inject replaced every extracted field")
    check(rebuilt["messages"][1]["content"][0]["signature"] == "AAAAsig==",
          "inject left structural fields alone")
    check(rebuilt["messages"][1]["content"][2]["input"]["n"] == 3,
          "inject left non-string values alone")
    check(rebuilt["system"] == body["system"],
          "inject left the unscanned system prompt byte-identical")
    check(body["messages"][0]["content"] == "Host: portal.acmebank.cz",
          "inject did not mutate the original body")

    # inject() must not collect: it re-walks, and a collecting re-walk would
    # double every name in the report.
    un2 = []
    extract(body, un2)
    check(len(un2) == len(un), "a second extract reports the same count")

    # ── 3. scope by ROLE, not by container ──────────────────────────────
    print("\n-- scope: role, not container --")

    env = ("# Environment\n - Primary working directory: "
           "C:\\Users\\op\\mask-test\n - Platform: win32\n"
           " - dataviz: use this skill for charts")
    rolebody = {"messages": [
        {"role": "user", "content": "hello"},
        {"role": "system", "content": [{"type": "text", "text": env}]},
    ]}
    un = []
    got = extract(rolebody, un)
    check([t for _, t in got] == ["hello"],
          "a role=system message's content is NOT masked")
    check("messages(role=system)" in short_names(un),
          "...and it is reported, naming the role")
    check(un[0][0] == "messages[1]", "its path names the message index")

    reports({"messages": [{"role": "developer", "content": "x"}]},
            "messages(role=developer)", "an unknown role is reported too")

    # ── 4. harness-authored text inside a user message ──────────────────
    print("\n-- the <system-reminder> rule --")

    rem = ("<system-reminder>\nEnd commits with:\n"
           "Co-Authored-By: Claude <noreply@anthropic.com>\n"
           "</system-reminder>\n")
    rbody = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": rem},
        {"type": "text", "text": "test"}]}]}
    un = []
    got = extract(rbody, un)
    check([t for _, t in got] == ["test"],
          "a whole-block system-reminder is NOT masked")
    check("system-reminder" in short_names(un), "...and it is reported")
    check(un[0][0] == "messages[0].content[0].text",
          "its path names the exact block")

    # The sniff is on the WHOLE block. Quoting the phrase mid-prompt is
    # still operator content and is still masked.
    quoted = {"messages": [{"role": "user", "content":
                            "what does <system-reminder> mean at acmebank?"}]}
    check([t for _, t in extract(quoted)]
          == ["what does <system-reminder> mean at acmebank?"],
          "a quoted marker mid-prompt is still masked")
    check(is_system_reminder("<system-reminder></system-reminder>") is False,
          "an empty reminder shell is not treated as one")

    # ── 5. refusals: malformed bodies only ──────────────────────────────
    print("\n-- refusals: malformed bodies only --")

    refuses({"messages": "not an array"}, "malformed messages array")
    refuses({"messages": [{"role": "user", "content": 7}]},
            "malformed message content")
    refuses({"messages": [{"role": "user", "content": [
        {"text": "no discriminator"}]}]}, "block with no type")
    refuses("not a dict", "body that is not a JSON object")

    # ── 5b. attachments: forwarded and named (2026-10-31) ───────────────
    print("\n-- attachments: forwarded and named, never refused --")

    reports({"messages": [{"role": "user", "content": [
        {"type": "image",
         "source": {"type": "base64", "media_type": "image/png",
                    "data": "iVBORw0K"}}]}]},
            "image", "an image is forwarded and named",
            "messages[0].content[0].source")
    reports({"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "data": "iVBORw0K"}}]}]}]},
            "image", "an image inside a tool result (an agent's Read)")
    reports({"messages": [{"role": "user", "content": [
        {"type": "document",
         "source": {"type": "base64", "media_type": "application/pdf",
                    "data": "JVBERi0="}}]}]},
            "document", "a PDF is forwarded and named")
    reports({"messages": [{"role": "user", "content": [
        {"type": "document",
         "source": {"type": "url", "url": "https://portal.acmebank.cz/a.pdf"}}
    ]}]}, "document", "a URL document is forwarded and named")
    reports({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "hi", "data": "AAAA"}]}]},
            "data", "opaque data on a block that is not redacted_thinking")

    # A document's TEXT is still masked.
    doc = {"messages": [{"role": "user", "content": [
        {"type": "document", "title": "acmebank report",
         "context": "from jan.novak@acmebank.cz",
         "source": {"type": "text", "media_type": "text/plain",
                    "data": "host portal.acmebank.cz"}},
        {"type": "document",
         "source": {"type": "content", "content": [
             {"type": "text", "text": "ip 10.0.0.5"},
             {"type": "image", "source": {"type": "base64", "data": "x"}}]}},
    ]}]}
    un = []
    got = extract(doc, un)
    texts = [t for _, t in got]
    check(set(texts) == {"acmebank report", "from jan.novak@acmebank.cz",
                         "host portal.acmebank.cz", "ip 10.0.0.5"},
          "a document's title, context, text source and content text are "
          "masked (got %r)" % texts)
    check(short_names(un) == ["image"],
          "...and the image inside its content is named (got %r)"
          % short_names(un))
    back = inject(doc, [(p_, "M") for p_, _ in got])
    d0, d1 = back["messages"][0]["content"]
    check(d0["title"] == d0["context"] == d0["source"]["data"] == "M"
          and d0["source"]["media_type"] == "text/plain"
          and d1["source"]["content"][0]["text"] == "M"
          and d1["source"]["content"][1]["source"]["data"] == "x",
          "inject puts each masked document string back in place")

    # Tool-input keys the path syntax uses are escaped, not refused.
    odd = {"messages": [{"role": "assistant", "content": [
        {"type": "tool_use", "id": "t", "name": "n", "input": {
            "a.b": "v1", "x[0]": "v2", "": "v3", "\\": "v4",
            "\\0": "v5", "0": "v6", "n": {"c.d": ["v7", {"]": "v8"}]}}}]}]}
    got = extract(odd)
    check(sorted(t for _, t in got) == ["v%d" % i for i in range(1, 9)],
          "every value under an awkward key is extracted")
    back = inject(odd, [(p_, t.upper()) for p_, t in got])
    inp = back["messages"][0]["content"][0]["input"]
    check(inp == {"a.b": "V1", "x[0]": "V2", "": "V3", "\\": "V4",
                  "\\0": "V5", "0": "V6",
                  "n": {"c.d": ["V7", {"]": "V8"}]}},
          "and each is put back under its own key (got %r)" % (inp,))

    # ── 6. what used to refuse and now reports ──────────────────────────
    print("\n-- converted: forwarded AND reported --")

    reports({"messages": [], "future_field": "client data here"},
            "future_field", "an invented top-level field", "future_field")
    reports({"messages": [{"role": "user", "content": "hi",
                           "attachments": ["secret.pdf"]}]},
            "attachments", "an unknown key on a message",
            "messages[0].attachments")
    reports({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "hi", "annotations": ["x"]}]}]},
            "annotations", "an unknown key on a known block")
    reports({"messages": [{"role": "user", "content": [
        {"type": "server_tool_use", "id": "x", "name": "n",
         "input": {"q": "acme"}, "server": "acme"}]}]},
            "server", "an unknown BLOCK TYPE (a named residual)")
    reports({"messages": [], "tools": [
        {"name": "t", "description": "d", "handler_url": "https://acme/run"}]},
            "tools", "a tool with an unknown key: the whole field reports")
    reports({"messages": [], "metadata": {"tenant": "acmebank"}},
            "metadata", "an unknown metadata key")
    reports({"messages": [], "thinking": {"type": "adaptive",
                                          "transcript": "client data"}},
            "thinking", "an unknown key on thinking")
    reports({"messages": [], "output_config": {"verbosity": "terse"}},
            "output_config", "an unknown key under output_config")
    reports({"messages": [], "output_format": {"type": "json_schema"}},
            "output_format", "the deprecated output_format")
    reports({"messages": [], "context_management": {"strategy": "x"}},
            "context_management", "an unknown key under context_management")
    reports({"messages": [{"role": "user", "content": "hi",
                           "output_config": {"effort": "low"}}]},
            "output_config", "per-message output_config is config, not text")

    # The headline. `safeguards` is forwarded now, and the reason it carries
    # is the one a reader needs.
    un = []
    sg = {"messages": [{"role": "user", "content": "hi"}],
          "safeguards": [{"type": "dangerous_tool_use",
                          "classifier_context": {
                              "live_cwd": "/c/clients/acmebank",
                              "user_identity": "op"}}]}
    got = extract(sg, un)
    check([t for _, t in got] == ["hi"], "safeguards does not block the walk")
    check("safeguards" in short_names(un), "safeguards is reported by name")
    reason = [r[2] for r in un if r[1] == "safeguards"][0]
    check("maskerade init" in reason and "LARGEST SINGLE RESIDUAL" in reason,
          "its reason names the residual and points at its mitigation")

    # A real Claude Code shape: every top-level field it sends at once.
    cc = {"model": "claude-opus-5-5", "max_tokens": 128000, "stream": True,
          "messages": [{"role": "user", "content": [
              {"type": "text", "text": "test"}]},
              {"role": "system", "content": [{"type": "text", "text": env}]}],
          "system": [{"type": "text", "text": "You are Claude Code"}],
          "tools": [{"name": "Bash", "description": "runs a command",
                     "input_schema": {"type": "object"}}],
          "metadata": {"user_id": "{\"device_id\":\"abc\"}"},
          "thinking": {"type": "adaptive", "display": "updates"},
          "context_management": {"edits": [
              {"type": "clear_thinking_20251015", "keep": "all"}]},
          "output_config": {"effort": "medium"},
          "safeguards": [{"type": "dangerous_tool_use"}]}
    un = []
    got = extract(cc, un)
    check([t for _, t in got] == ["test"],
          "a full Claude Code body masks only the user's own text")
    check(short_names(un) == [
        "context_management", "messages(role=system)", "metadata",
        "output_config", "safeguards", "system", "thinking", "tools"],
        "...and names all eight unscanned fields (got %r)"
        % (short_names(un),))

    # ── 7. the dormant walks, called directly ───────────────────────────
    # Nothing in extract() reaches these any more. They are kept for the
    # future `--scan` flag, so they are tested where they live
    # rather than left to rot behind a branch nobody takes.
    print("\n-- dormant code, exercised directly --")

    def schema_texts(sch):
        o, u = [], []
        _schema(sch, "input_schema", o, u)
        return [t for _, t in o], short_names(u)

    def schema_ok(sch, label, expect=None):
        try:
            t, _ = schema_texts(sch)
        except Exception as e:
            print("FAIL %s  (%s: %s)" % (label, type(e).__name__, e))
            failures.append(label)
            return
        if expect is not None and expect not in t:
            print("FAIL %s  (%r not extracted)" % (label, expect))
            failures.append(label)
            return
        print("ok   " + label)

    schema_ok({"type": "object", "properties": {"files": {"anyOf": [
        {"type": "string"},
        {"type": "object", "propertyNames": {"type": "string"}}]}}},
        "propertyNames inside anyOf (the field Claude Code sent)")
    schema_ok({"type": "object",
               "properties": {"x": {"$ref": "#/$defs/Thing"}},
               "$defs": {"Thing": {"type": "string",
                                   "description": "thing at acmebank"}}},
              "$defs subschema map walked, its description masked",
              "thing at acmebank")
    schema_ok({"type": "object",
               "patternProperties": {"^x_hdr": {"type": "string",
                                                "title": "acme header"}}},
              "patternProperties map walked", "acme header")
    schema_ok({"type": "array", "prefixItems": [{"type": "string"}],
               "contains": {"type": "string"}, "minContains": 1},
              "prefixItems / contains / minContains")
    schema_ok({"type": "object", "if": {"required": ["a"]},
               "then": {"required": ["b"]}, "else": {"required": ["c"]}},
              "if / then / else")
    schema_ok({"type": "object", "dependentRequired": {"a": ["b"]},
               "dependentSchemas": {"a": {"required": ["b"]}}},
              "dependentRequired forwarded, dependentSchemas walked")
    schema_ok({"type": "object", "$comment": "staging only, acmebank"},
              "$comment is masked, not forwarded", "staging only, acmebank")
    schema_ok({"type": "string", "readOnly": True, "multipleOf": 2,
               "contentEncoding": "base64", "unevaluatedProperties": False},
              "assorted assertions forwarded")

    t, _ = schema_texts({"$id": "https://acme.internal/s.json",
                         "type": "object",
                         "properties": {"x": {"$ref": "#/$defs/T"}},
                         "$defs": {"T": {"type": "string"}}})
    check("https://acme.internal/s.json" not in t,
          "$id is forwarded unmasked (bounded exception: a URI reference)")
    check("#/$defs/T" not in t, "$ref is forwarded unmasked")

    t, n = schema_texts({"type": "object", "x-acme-internal": "client data"})
    check(n == ["x-acme-internal"],
          "an unknown schema keyword is reported, no longer refused")

    # _seg still guards the schema path for the day --scan turns it on.
    for bad, label in (({"patternProperties": {"^x-[a-z]+$": {}}},
                        "regex map key still refuses rather than crashing"),
                       ({"properties": {"a.b": {"type": "string"}}},
                        "dotted property name still refuses")):
        try:
            schema_texts(bad)
            check(False, label)
        except RefusedField:
            check(True, label)

    # The tools walk and the schema round trip.
    o, u = [], []
    _tools([{"name": "t", "description": "d at acme",
             "input_schema": {"type": "object", "properties": {
                 "h": {"type": "string", "enum": ["portal.acmebank.cz"]}}},
             "handler_url": "x"}], "tools", o, u)
    check("d at acme" in [t for _, t in o], "_tools still masks a description")
    check("portal.acmebank.cz" in [t for _, t in o],
          "_tools still masks a schema enum")
    check(u and u[0][1] == "handler_url",
          "_tools reports an unknown tool key")

    o, u = [], []
    _context_management({"edits": [
        {"type": "clear_tool_uses_20250919",
         "trigger": {"type": "input_tokens", "value": 30000},
         "keep": {"type": "tool_uses", "value": 3},
         "exclude_tools": ["web_search", "Read"],
         "clear_tool_inputs": False}]}, "context_management", u)
    check(u == [], "_context_management still extracts and reports nothing")
    u = []
    _context_management({"edits": [{"type": "compact_20260112",
                                    "summary_prompt": "summarise acme"}]},
                        "context_management", u)
    check([r[1] for r in u] == ["summary_prompt"],
          "_context_management reports an undocumented strategy's own field")

    o, u = [], []
    _output_config({"effort": "high", "format": {
        "type": "json_schema", "schema": {"type": "object", "properties": {
            "host": {"type": "string",
                     "description": "the acmebank portal"}}}}},
        "output_config", o, u)
    check("the acmebank portal" in [t for _, t in o],
          "_output_config still masks a format.schema description")

    o, u = [], []
    _metadata({"user_id": "op-4471", "tenant": "acme"}, "metadata", o, u)
    check([t for _, t in o] == ["op-4471"],
          "_metadata still masks user_id when called")
    check([r[1] for r in u] == ["tenant"], "_metadata reports an unknown key")

    u = []
    _thinking_config({"type": "adaptive", "display": "updates"},
                     "thinking", u)
    check(u == [], 'thinking.display "updates" is still an accepted VALUE')
    _tool_choice({"type": "tool", "name": "x"}, "tool_choice", u)
    check(u == [], "_tool_choice still accepts its three keys")

    # ── 8. redacted_thinking ────────────────────────────────────────────
    print("\n-- unchanged --")
    ok_body = {"messages": [{"role": "assistant", "content": [
        {"type": "redacted_thinking", "data": "AAAAopaque"}]}]}
    un = []
    check([t for _, t in extract(ok_body, un)] == [] and un == [],
          "redacted_thinking.data is forwarded, neither masked nor reported")

    check(SCHEMA_STRUCTURAL == SCHEMA_APPLICATORS | SCHEMA_ASSERTIONS,
          "the retained SCHEMA_STRUCTURAL alias still resolves")
    check(SCAN_FLAG_CANDIDATES <= set(TOP_UNSCANNED),
          "every --scan candidate is a known unscanned field")

    print("\n%s" % ("PASSED" if not failures
                    else "%d FAILED: %s" % (len(failures), failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
