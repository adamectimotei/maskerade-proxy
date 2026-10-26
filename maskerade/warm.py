# -*- coding: utf-8 -*-
"""Fetch the model artifacts this tool needs, once, while there is network.

    python -m maskerade.warm                 # tier 1: the BPE encoder (~1.6 MB)
    python -m maskerade.warm --full          # + the gliner model (~1.2 GB)
    python -m maskerade.warm --to DIR        # write a portable cache for an
                                             #   air-gapped box
    python -m maskerade.warm --check         # report, download nothing

Why this exists
───────────────
Setup is the only step that is allowed to need the internet. Everything after
it runs offline — verified: tier 1 with the interface down, and the full tier
under HF_HUB_OFFLINE=1 with a warm cache.

Two artifacts are fetched at runtime by libraries we do not control, and both
used to fail in ways that were easy to miss:

  cl100k_base    tiktoken downloads it on first use. With no network, form
                 used to warn once and silently run Shannon instead —
                 measured at 9 of 81 form detections lost on a 40-message
                 sample, in the under-masking direction. It now refuses, and
                 names this command. 1.6 MB.

  gliner model   fails closed already, with a clear error. 1.2 GB, full tier
                 only, and not part of the tier-1 product.

The files are NOT vendored into the repository. cl100k_base.tiktoken is served
from an OpenAI blob and carries no redistribution licence of its own —
tiktoken's MIT licence covers the code that downloads it, not the data. Rather
than make a legal judgement in a repository headed for open source, this
fetches at setup, where the network is available anyway.

When a download fails
─────────────────────
The operator gets one short block naming the file, the host it comes from,
the underlying reason and what to do next -- not a Python traceback. The
usual cause is a corporate proxy or an offline machine, and a traceback
reads as a crash in this tool rather than as "this machine cannot reach that
host". --traceback restores the full error for a bug report.

The proxy variable's VALUE is never printed, only whether one is set: a
proxy URL can embed a username and password.

Air-gapped installs
───────────────────
    (online box)   python -m maskerade.warm --full --to ./maskerade-cache
    (copy the directory across by whatever means)
    (offline box)  set TIKTOKEN_CACHE_DIR and HF_HOME to point into it

--to prints the exact variables to set, with the paths filled in.
"""
import argparse
import hashlib
import os
import shutil
import sys
from urllib.parse import urlsplit


# The model the full tier loads by default (stages/gliner.py, `gliner.model`
# in default_config.yaml). Downloaded by name; its cache directory is derived
# from the name the way huggingface_hub derives it.
GLINER_MODEL = "fastino/gliner2-privacy-filter-PII-multi"
GLINER_HOST = "huggingface.co"

# huggingface_hub's own reading of a boolean environment variable
# (constants.ENV_VARS_TRUE_VALUES). Matched exactly so that HF_HUB_OFFLINE=0
# means what it means to the library that does the downloading: not offline.
_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}

_PROXY_VARS = ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")


class WarmError(RuntimeError):
    """A failure this module has already explained in full.

    main() prints its message as it stands. Anything else that escapes a
    download is summarised instead (see _explain).
    """


def _form():
    """Import the form stage.

    warm.py lives at maskerade/warm.py; form lives at maskerade/stages/form.py.
    The fallbacks cover running this file directly, and a flat layout.
    """
    try:
        from .stages import form as _f            # maskerade/stages/form.py
        return _f
    except ImportError:
        pass
    try:
        from maskerade.stages import form as _f   # not run as a module
        return _f
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "stages"), here):
        if os.path.exists(os.path.join(cand, "form.py")):
            sys.path.insert(0, cand)
            import form as _f
            return _f
    raise ImportError(
        "could not locate the form stage; expected maskerade/stages/form.py")


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0


def _q(path):
    """Quote a path for a command line the operator will copy."""
    return f'"{path}"' if any(c.isspace() for c in path) else path


def _env_true(name):
    return (os.environ.get(name) or "").strip().upper() in _TRUE_VALUES


def hf_offline_var():
    """The variable that puts huggingface_hub offline, or None.

    The library honours TRANSFORMERS_OFFLINE as well as HF_HUB_OFFLINE, so
    both are checked; the name is returned so the message can say which one
    to unset.
    """
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if _env_true(name):
            return name
    return None


def _offline_message(var):
    return (f"{var} is set, so nothing can be downloaded from {GLINER_HOST}. "
            f"Unset it for this command:\n"
            f"    Remove-Item Env:{var}      (PowerShell)\n"
            f"    unset {var}                (bash/zsh)")


def _ner_available():
    """True when the 'ner' extra is installed. find_spec only, no import."""
    try:
        from ._extras import ner_available
    except ImportError:
        try:
            from maskerade._extras import ner_available
        except ImportError:
            return True          # cannot tell; let the download itself decide
    try:
        return bool(ner_available())
    except Exception:
        return True


def warm_bpe(cache_dir, verify=True):
    """Populate `cache_dir` with cl100k_base. Returns the file path.

    tiktoken keys its cache by sha1 of the blob URL. That is undocumented
    rather than private, so the name is derived from the URL here instead of
    being hardcoded — if tiktoken changes the scheme, this breaks loudly on
    the next run rather than writing a file nothing reads.
    """
    form = _form()
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, form.bpe_cache_filename())

    prev = os.environ.get("TIKTOKEN_CACHE_DIR")
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
    try:
        import tiktoken
        tiktoken.get_encoding(form.BPE_ENCODING)
    finally:
        if prev is None:
            os.environ.pop("TIKTOKEN_CACHE_DIR", None)
        else:
            os.environ["TIKTOKEN_CACHE_DIR"] = prev

    if not os.path.exists(target):
        raise WarmError(
            f"tiktoken loaded {form.BPE_ENCODING} but wrote nothing to "
            f"{target}. Its cache naming has probably changed; "
            f"bpe_cache_filename() needs updating.")

    if verify:
        # A truncated or substituted encoder would change the entropy profile
        # of every message while reporting success — exactly the silent
        # degradation this whole change set exists to remove. Check it.
        h = hashlib.sha256()
        with open(target, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != form.BPE_BLOB_SHA256:
            os.remove(target)
            raise WarmError(
                f"{form.BPE_ENCODING} failed its checksum and was deleted.\n"
                f"  expected {form.BPE_BLOB_SHA256}\n"
                f"  got      {got}\n"
                f"Do not use this file: a different encoder silently changes "
                f"detection.")
    return target


def warm_gliner(config_path=None):
    """Download the gliner model into the active HF cache. Returns its name."""
    var = hf_offline_var()
    if var:
        raise WarmError(_offline_message(var))
    name = GLINER_MODEL
    try:
        from .stages import gliner as _g
        name = (_g.DEFAULT_MODEL if hasattr(_g, "DEFAULT_MODEL") else name)
    except Exception:
        pass
    from gliner2 import GLiNER2
    GLiNER2.from_pretrained(name)
    return name


def hf_cache_dir(to=None):
    """The hub cache huggingface_hub reads and writes.

    `to` is warm's --to directory: its layout is fixed (<to>/huggingface/hub).
    Otherwise the library's own answer when it is installed, and when it is
    not -- a light install, asked by `--check` -- the same environment
    variables in the same order the library reads them, so the path printed
    is the path the full tier would use.
    """
    if to:
        return os.path.join(os.path.abspath(to), "huggingface", "hub")
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return HF_HUB_CACHE
    except Exception:
        pass

    def _x(p):
        return os.path.expandvars(os.path.expanduser(p))

    if os.environ.get("HF_HUB_CACHE"):
        return _x(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return _x(os.environ["HUGGINGFACE_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return os.path.join(_x(os.environ["HF_HOME"]), "hub")
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join("~", ".cache")
    return os.path.join(_x(xdg), "huggingface", "hub")


def gliner_model_dir(hub):
    """Where huggingface_hub keeps GLINER_MODEL inside `hub`."""
    return os.path.join(hub, "models--" + GLINER_MODEL.replace("/", "--"))


def dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


# ─────────────────────────────────────────────────── failure reporting ──

def _root_cause(exc):
    """The innermost exception in the chain, which is the one that names the
    real reason ('Tunnel connection failed: 403', 'connection refused')."""
    seen = set()
    cur = exc
    while id(cur) not in seen:
        seen.add(id(cur))
        nxt = cur.__cause__ or cur.__context__
        if nxt is None:
            break
        cur = nxt
    return cur


def _one_line(exc, limit=200):
    text = " ".join(str(exc).split())
    if len(text) > limit:
        text = text[:limit - 3] + "..."
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _is_network(exc):
    """True when anything in the chain is an OSError.

    Every transport failure on these paths is one: socket and DNS errors,
    ssl.SSLError, requests' RequestException family and huggingface_hub's
    HTTP and cache-miss errors all derive from it.
    """
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, OSError):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _proxy_configured():
    return any(os.environ.get(v) for v in _PROXY_VARS)


def _explain(exc, what, host, full):
    """Print a short account of a failed download. Never prints a traceback,
    and never prints a proxy variable's value."""
    err = sys.stderr
    err.write(f"\n  FAILED  could not fetch the {what} from {host}.\n")
    err.write(f"          {_one_line(_root_cause(exc))}\n")
    if _is_network(exc):
        err.write(f"\n          This step needs HTTPS access to {host}.\n")
        if _proxy_configured():
            err.write("          A proxy variable (HTTPS_PROXY or similar) is "
                      "set and was used:\n"
                      f"          check that it is reachable and allows "
                      f"{host}.\n")
        else:
            err.write("          Behind a proxy? Set HTTPS_PROXY for this "
                      "command and run it again.\n")
        flag = " --full" if full else ""
        err.write("          No access from this machine at all? On one that "
                  "has it, run\n"
                  f"              python -m maskerade.warm{flag} --to DIR\n"
                  "          copy DIR here, and set the variables it "
                  "prints.\n")
    else:
        err.write("\n          This is not a network error. Re-run with "
                  "--traceback and include\n"
                  "          the output in a bug report.\n")
    err.write("          (--traceback shows the full Python error.)\n")


# ──────────────────────────────────────────────────────────────── main ──

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m maskerade.warm",
        description="Fetch model artifacts once, so every later run is offline.")
    ap.add_argument("--full", action="store_true",
                    help="also fetch the gliner model (~1.2 GB, full tier only; "
                         "needs the 'ner' extra)")
    ap.add_argument("--to", metavar="DIR",
                    help="write a portable cache here instead of the default "
                         "locations, for copying to an air-gapped box")
    ap.add_argument("--check", action="store_true",
                    help="report what is present; download nothing")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the encoder checksum (not recommended)")
    ap.add_argument("--traceback", action="store_true",
                    help="on failure, show the full Python error instead of "
                         "the short explanation")
    args = ap.parse_args(argv)

    form = _form()
    bpe_dir = (os.path.join(os.path.abspath(args.to), "tiktoken")
               if args.to else form.bpe_cache_dir())

    if args.check:
        path = os.path.join(bpe_dir, form.bpe_cache_filename())
        ok = os.path.exists(path)
        print(f"  BPE encoder   {'present' if ok else 'MISSING'}  {bpe_dir}")
        if ok:
            print(f"                {human(os.path.getsize(path))}")
        else:
            print("                run: python -m maskerade.warm"
                  + (f" --to {_q(args.to)}" if args.to else ""))
        hub = hf_cache_dir(args.to)
        mdir = gliner_model_dir(hub)
        msize = dir_size(mdir) if os.path.isdir(mdir) else 0
        print(f"  gliner model  {'present' if msize else 'MISSING'}  {hub}")
        if msize:
            print(f"                {human(msize)}  {GLINER_MODEL}")
        else:
            print("                full tier only; run: "
                  "python -m maskerade.warm --full"
                  + (f" --to {_q(args.to)}" if args.to else ""))
        return 0 if ok else 1

    # Everything that can be decided before a byte is downloaded is decided
    # here, so a run that cannot finish fails before it has half-finished.
    if args.full:
        if not _ner_available():
            sys.stderr.write(
                "\n  --full fetches the model for the full tier, which needs "
                "the 'ner' extra.\n"
                "  It is not installed. Install it first (CPU PyTorch, then "
                "the extra):\n"
                "      pip install torch --index-url "
                "https://download.pytorch.org/whl/cpu\n"
                "      pip install \"maskerade[ner]\"\n"
                "  Nothing was downloaded.\n")
            return 2
        var = hf_offline_var()
        if var:
            sys.stderr.write("\n  " + _offline_message(var).replace(
                "\n", "\n  ") + "\n  Nothing was downloaded.\n")
            return 2

    what, host = "BPE encoder", urlsplit(form.BPE_BLOB_URL).hostname
    try:
        print(f"  BPE encoder -> {bpe_dir}")
        path = warm_bpe(bpe_dir, verify=not args.no_verify)
        print(f"  ok  {human(os.path.getsize(path))}"
              f"{'' if args.no_verify else '  checksum verified'}")

        hf_dir = None
        if args.full:
            what, host = f"gliner model ({GLINER_MODEL})", GLINER_HOST
            if args.to:
                # HF_HOME must be set before huggingface_hub computes its
                # paths, which happens at import. Set it here, before
                # warm_gliner's import, or the download lands in the default
                # cache instead.
                hf_dir = os.path.join(os.path.abspath(args.to), "huggingface")
                os.environ["HF_HOME"] = hf_dir
            print("  gliner model -> "
                  f"{hf_dir or hf_cache_dir()}  (~1.2 GB, this takes a while)")
            name = warm_gliner()
            print(f"  ok  {name}")
    except KeyboardInterrupt:
        sys.stderr.write(
            "\n  interrupted. Run the same command again to finish; partial "
            "downloads are\n  resumed or replaced, never used.\n")
        return 130
    except WarmError as e:
        if args.traceback:
            raise
        sys.stderr.write("\n  FAILED  " + str(e).replace("\n", "\n  ") + "\n")
        return 1
    except Exception as e:
        if args.traceback:
            raise
        _explain(e, what, host, args.full)
        return 1

    if args.to:
        root = os.path.abspath(args.to)
        print()
        print("  Copy this directory to the offline box, then set:")
        print(f"    TIKTOKEN_CACHE_DIR = {os.path.join(root, 'tiktoken')}")
        if args.full:
            print(f"    HF_HOME            = "
                  f"{os.path.join(root, 'huggingface')}")
            print("    HF_HUB_OFFLINE     = 1")
        print(f"  total {human(dir_size(root))}")
    else:
        print()
        print("  Done. Nothing needs the network from here on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
