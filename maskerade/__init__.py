"""maskerade — local anonymisation of pentest prompts and HTTP traffic.

Masks sensitive values out of text before it is sent to a third-party AI, and
restores them in the reply. Everything runs locally; nothing is sent anywhere
by this package.

    from maskerade import Pipeline

    p = Pipeline()                   # the default tier: "fast"
    r = p.run(text)
    r["masked_text"], r["mask_map"], r["entities"], r["validation"]
    p.unmask(reply, r["mask_map"])

Two tiers:
  fast  custom + regex + form + keyval + checksum + id
        pure Python, three dependencies, tens of milliseconds.
        THE DEFAULT, and what `pip install maskerade` can run.
  full  the above plus gliner (+ org_evidence). Needs
        `pip install maskerade[ner]` — plus a one-off model download,
        `python -m maskerade.warm --full`.

Both tiers need the BPE encoder, because `form` is in both:

    python -m maskerade.warm         # ~1.6 MB, once, with network

That is not optional. form fails closed without it, and fail-closed means the
proxy does not start.

WHICH TIER AM I RUNNING?
------------------------
    maskerade --version              names the default tier and the extras
    p.mode_string()                  names the stages actually live

Measured recall and precision for BOTH tiers, with the date and the test sets
they come from, are in the README under "Measured detection". The two tiers
detect different things; a number for one says nothing about the other.
"""
from .pipeline import (
    Pipeline, DetectorError, TIER_FAST, TIER_FULL, TIERS, STAGE_KEYS,
    DEFAULT_TIER,
)

__all__ = [
    "Pipeline", "DetectorError",
    "TIER_FAST", "TIER_FULL", "TIERS", "STAGE_KEYS", "DEFAULT_TIER",
]
__version__ = "0.1.2"
