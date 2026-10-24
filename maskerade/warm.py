# -*- coding: utf-8 -*-
"""Fetch the model artifacts this tool needs, once, while there is network.

    python -m maskerade.warm                 # tier 1: the BPE encoder (~1.7 MB)
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
                 names this command. 1.7 MB.

  gliner model   fails closed already, with a clear error. 1.2 GB, full tier
                 only, and not part of the tier-1 product.

The files are NOT vendored into the repository. cl100k_base.tiktoken is served
from an OpenAI blob and carries no redistribution licence of its own —
tiktoken's MIT licence covers the code that downloads it, not the data. Rather
than make a legal judgement in a repository headed for open source, this
fetches at setup, where the network is available anyway.

Air-gapped installs
───────────────────
    (online box)   python -m maskerade.warm --full --to ./maskerade-cache
    (copy the directory across by whatever means)
    (offline box)  set TIKTOKEN_CACHE_DIR and HF_HOME to point into it

--to prints the exact two variables to set, with the paths filled in.
"""
import argparse
import hashlib
import os
import shutil
import sys


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
        raise RuntimeError(
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
            raise RuntimeError(
                f"{form.BPE_ENCODING} failed its checksum and was deleted.\n"
                f"  expected {form.BPE_BLOB_SHA256}\n"
                f"  got      {got}\n"
                f"Do not use this file: a different encoder silently changes "
                f"detection.")
    return target


def warm_gliner(config_path=None):
    """Download the gliner model into the active HF cache. Returns its name."""
    if os.environ.get("HF_HUB_OFFLINE"):
        raise RuntimeError(
            "HF_HUB_OFFLINE is set, so nothing can be downloaded. Unset it "
            "for this command:  Remove-Item Env:HF_HUB_OFFLINE   (PowerShell)"
            "  /  unset HF_HUB_OFFLINE   (sh)")
    name = "fastino/gliner2-privacy-filter-PII-multi"
    try:
        from .stages import gliner as _g
        name = (_g.DEFAULT_MODEL if hasattr(_g, "DEFAULT_MODEL") else name)
    except Exception:
        pass
    from gliner2 import GLiNER2
    GLiNER2.from_pretrained(name)
    return name


def hf_cache_dir():
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return HF_HUB_CACHE
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".cache",
                            "huggingface", "hub")


def dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m maskerade.warm",
        description="Fetch model artifacts once, so every later run is offline.")
    ap.add_argument("--full", action="store_true",
                    help="also fetch the gliner model (~1.2 GB, full tier only)")
    ap.add_argument("--to", metavar="DIR",
                    help="write a portable cache here instead of the default "
                         "locations, for copying to an air-gapped box")
    ap.add_argument("--check", action="store_true",
                    help="report what is present; download nothing")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the encoder checksum (not recommended)")
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
            print("                run: python -m maskerade.warm")
        hf = hf_cache_dir()
        hf_ok = os.path.isdir(hf) and dir_size(hf) > 0
        print(f"  gliner model  {'present' if hf_ok else 'MISSING'}  {hf}")
        if hf_ok:
            print(f"                {human(dir_size(hf))}")
        else:
            print("                full tier only; run: "
                  "python -m maskerade.warm --full")
        return 0 if ok else 1

    print(f"  BPE encoder -> {bpe_dir}")
    path = warm_bpe(bpe_dir, verify=not args.no_verify)
    print(f"  ok  {human(os.path.getsize(path))}"
          f"{'' if args.no_verify else '  checksum verified'}")

    hf_dir = None
    if args.full:
        if args.to:
            # HF_HOME must be set before huggingface_hub computes its paths,
            # which happens at import. Set it here, before warm_gliner's
            # import, or the download lands in the default cache instead.
            hf_dir = os.path.join(os.path.abspath(args.to), "huggingface")
            os.environ["HF_HOME"] = hf_dir
        print("  gliner model -> "
              f"{hf_dir or hf_cache_dir()}  (~1.2 GB, this takes a while)")
        name = warm_gliner()
        print(f"  ok  {name}")

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
