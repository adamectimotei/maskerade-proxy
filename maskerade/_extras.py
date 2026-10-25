# -*- coding: utf-8 -*-
"""Which optional extras are installed — answered WITHOUT importing them.

A bug report has to say which install it came from. `pip install maskerade`
and `pip install maskerade[ner]` are the same version string, the same
commands and two different detector sets, so a version string on its own does not
identify the thing that ran.

IT MUST NOT IMPORT WHAT IT REPORTS ON
-------------------------------------
importlib.util.find_spec() locates a module without executing it, so probing
for torch costs a path search rather than the measured ~88 s of instrumented
import time that `import gliner2` triggers once torch is present. That is not
an optimisation here, it is a correctness requirement: the light install
promises that nothing heavy appears in sys.modules under the default tier, and
this module is reachable from `maskerade --version` and from the proxy's
startup path. A plain `import torch` inside a try/except would make the gate
fail and would make the light install pay for a question about itself.

find_spec can do more than return None — it raises ModuleNotFoundError when a
PARENT package is missing, and ValueError on a half-initialised module — so
every call is wrapped. An extra reported as absent when it is present is a
cosmetic error in a version banner; an exception there would take out the
command.
"""
from importlib import util as _util


# extra name -> the modules that must all be importable for it to be usable.
#
# gliner2 alone is not enough and that is the whole reason this mapping is not
# just the extra's name: `pip install gliner2` without [local] installs a
# 356 kB wheel with no torch and no transformers, which resolves here, imports
# there, and fails at model load. The three names below are what
# gliner2[local] actually brings (read from its wheel metadata, 2.1.0).
EXTRA_MODULES = {
    "ner": ("gliner2", "torch", "transformers"),
}


def module_present(name):
    """True when `name` can be located on the path. Never imports it."""
    try:
        return _util.find_spec(name) is not None
    except (ImportError, ValueError):
        # ModuleNotFoundError on a missing parent package, ValueError on a
        # module that is in sys.modules with a None __spec__.
        return False


def module_version(name):
    """The installed version of a distribution, or None. Never imports it.

    importlib.metadata reads the .dist-info on disk, so asking for torch's
    version costs a directory scan rather than the seconds an import costs.

    This exists because "ner: installed" is not enough to reproduce a run.
    gliner2 is pinned, torch and transformers are not, and all three decide
    what the model produces — so the version banner names the trio and a bug
    report stops being a guess.
    """
    try:
        from importlib import metadata as _md
    except ImportError:                      # pragma: no cover - py<3.8
        return None
    try:
        return _md.version(name)
    except Exception:
        # PackageNotFoundError, and anything a malformed dist-info raises. A
        # version we cannot read is reported as unknown, never as absent:
        # the module may be perfectly importable from a source checkout.
        return None


def _requirements(dist="maskerade"):
    """(requirements, sources) for `dist`, preferring .dist-info.

    WHY THIS IS NOT JUST importlib.metadata.requires(). Running from the
    repository root — which is where shipcheck and `maskerade --version` are
    run — puts the CWD on sys.path, and a `maskerade.egg-info` left there by
    `pip install -e .` is then discovered as a distribution. It is a
    SEPARATE, OLDER copy of the metadata: measured 2026-10-17, an egg-info
    written before the pin landed reported `gliner2[local]` while the
    installed .dist-info beside the package correctly reported
    `gliner2[local]==2.0.0`.

    So the probe that exists to avoid a second, drifting copy of the pin was
    reading a second, drifting copy of the pin. .dist-info wins, and every
    source found is returned so a caller can say when they disagree.
    """
    try:
        from importlib import metadata as _md
    except ImportError:                      # pragma: no cover - py<3.8
        return [], []
    want = dist.lower().replace("_", "-")
    found = []
    try:
        for d in _md.distributions():
            try:
                name = (d.metadata["Name"] or "")
            except Exception:
                continue
            if name.lower().replace("_", "-") != want:
                continue
            path = str(getattr(d, "_path", "") or "")
            # 0 sorts first: a real install beats a build artifact.
            rank = 0 if path.endswith(".dist-info") else 1
            try:
                reqs = list(d.requires or [])
            except Exception:
                reqs = []
            found.append((rank, path, reqs))
    except Exception:
        found = []
    if not found:
        try:
            return list(_md.requires(dist) or []), []
        except Exception:
            return [], []
    found.sort(key=lambda t: t[0])
    return found[0][2], [(p, r) for _rank, p, r in found]


def metadata_conflict(module="gliner2", extra="ner", dist="maskerade"):
    """Distinct requirement strings for `module` across metadata copies.

    More than one entry means a stale egg-info (or a second install) is
    stating a different constraint from the one in force. Returns a list of
    (path, requirement) or [] when everything agrees.
    """
    _reqs, sources = _requirements(dist)
    if len(sources) < 2:
        return []
    seen = {}
    for path, reqs in sources:
        found = None
        for req in reqs:
            head = req.split(";")[0].strip()
            name = head.split("[")[0].split("=")[0].split("<")[0] \
                       .split(">")[0].split("!")[0].split("~")[0].strip()
            if name.lower().replace("_", "-") != module.lower().replace("_", "-"):
                continue
            if extra and ("extra" not in req or extra not in req):
                continue
            found = head
            break
        seen[path] = found
    if len(set(seen.values())) < 2:
        return []
    return sorted(seen.items())


def declared_requirement(module, extra=None, dist="maskerade"):
    """What THIS package asks for, read from its own installed metadata.

    The pin lives in pyproject.toml and nowhere else. Restating it as a
    constant here would create a second copy that drifts silently, and a
    drifting copy of a pin is worse than no copy, because it would report a
    constraint the installer never applied. Reading it back from the
    installed metadata means this can only ever report what is actually in
    force.

    Returns the requirement string (e.g. "gliner2[local]==2.0.0") or None.
    """
    reqs, _sources = _requirements(dist)
    for req in reqs:
        head = req.split(";")[0].strip()
        name = head.split("[")[0].split("=")[0].split("<")[0] \
                   .split(">")[0].split("!")[0].split("~")[0].strip()
        if name.lower().replace("_", "-") != module.lower().replace("_", "-"):
            continue
        if extra and ("extra" not in req or extra not in req):
            continue
        return head
    return None


def pin_mismatch():
    """(requirement, installed) when an EXACT pin is not what is installed.

    The case this catches: the pin is raised, or an older environment is
    reused, and the frozen snapshots no longer describe the library doing
    the detecting. Only `==` is checked — a range is a deliberate statement
    that several versions are acceptable, so a version inside it is not a
    mismatch to report.
    """
    req = declared_requirement("gliner2", extra="ner")
    if not req or "==" not in req:
        return None
    wanted = req.split("==", 1)[1].strip()
    got = module_version("gliner2")
    if got is None or got == wanted:
        return None
    return (req, got)


def extras_status():
    """{extra: {"installed": bool, "present": [...], "missing": [...]}}"""
    out = {}
    for extra, modules in EXTRA_MODULES.items():
        present = [m for m in modules if module_present(m)]
        missing = [m for m in modules if m not in present]
        out[extra] = {
            "installed": not missing,
            "present": present,
            "missing": missing,
        }
    return out


def summary():
    """One line naming the extras, for a version banner.

    A PARTIAL extra is called out rather than rounded down to "absent": the
    half-installed state is the one that produces a confusing bug report,
    because the tier a caller asks for decides whether it is fatal.
    """
    bits = []
    for extra, st in sorted(extras_status().items()):
        if st["installed"]:
            # Name the versions. All three decide what the model emits, and
            # only one of them is pinned.
            seen = []
            for m in EXTRA_MODULES[extra]:
                v = module_version(m)
                seen.append("%s %s" % (m, v if v else "?"))
            bits.append("%s: installed (%s)" % (extra, ", ".join(seen)))
        elif st["present"]:
            bits.append("%s: PARTIAL (missing %s)"
                        % (extra, ", ".join(st["missing"])))
        else:
            bits.append("%s: not installed" % extra)
    return "; ".join(bits) if bits else "no optional extras defined"


def ner_available():
    """True when the full tier could actually run on this install."""
    return extras_status()["ner"]["installed"]
