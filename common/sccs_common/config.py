"""Shared TOML configuration loading for the SCCS Drone project."""

import os
import logging

logger = logging.getLogger(__name__)


def load_config(path=None):

    path = path or os.environ.get("DRONE_CONFIG", "config.toml")
    if not os.path.exists(path):
        logger.warning("Config file not found at %s; using built-in defaults.", path)
        return {}
    try:
        import tomllib as _toml
        with open(path, "rb") as f:
            return _toml.load(f)
    except Exception as e:
        logger.error("Failed to parse config file %s: %s; using built-in defaults.", path, e)
        return {}


def get_setting(section, key, default):
    value = section.get(key)
    if value is None or value == "":
        return default
    return value
