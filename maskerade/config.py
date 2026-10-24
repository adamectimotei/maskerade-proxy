"""Load pipeline configuration from YAML.

Changed from the GPU repo's pipeline/config.py in exactly one way: the default
is the config shipped INSIDE the package rather than a config.yaml sitting in a
repo root. A pip-installed package has no repo root.

Priority:
  1. explicit path argument          Pipeline(config_path="...")
  2. MASKERADE_CONFIG env var
  3. ANON_CONFIG env var             (compatibility with the GPU repo's tooling)
  4. default_config.yaml inside this package
"""
import os

import yaml


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "default_config.yaml")


def load_config(path=None):
    """Load configuration from a YAML file. Returns a dict."""
    config_path = (
        path
        or os.environ.get("MASKERADE_CONFIG")
        or os.environ.get("ANON_CONFIG")
        or DEFAULT_CONFIG_PATH
    )

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Pass config_path=, or set MASKERADE_CONFIG."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} did not parse as a dict")

    return config
