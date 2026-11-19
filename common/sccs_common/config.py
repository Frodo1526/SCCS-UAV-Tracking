"""Shared TOML configuration loading for the SCCS Drone project."""
import os
import logging

logger = logging.getLogger(__name__)

def load_config(path=None):
    '''
    Loads a TOML configuration file from disk, falling back to an empty
    dict (which callers then treat as "use built-in defaults" via
    get_setting()) if the file is missing or fails to parse.

    Args:
        path (str, optional): Filesystem path to the TOML config file. If
            not provided, the DRONE_CONFIG environment variable is used;
            if that is also unset, defaults to "config.toml" in the
            current working directory.

    Returns:
        dict: The parsed TOML document's top-level table if the file
            exists and parses successfully.
        dict: An empty dict if the file does not exist at the resolved
            path, or if opening/parsing it raises any exception (the
            exception is caught, logged, and not re-raised).
    '''
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
    '''
    Looks up a key in a config section dict, treating both a missing key
    and an explicit empty-string value as "not configured" and falling
    back to the given default. This differs from a plain dict.get(key,
    default), which only falls back when the key is absent -- config.toml
    ships with some keys explicitly set to "" as placeholders, and a plain
    .get() would return that empty string instead of the intended default.

    Args:
        section (dict): A config section (e.g. the "network" table loaded
            by load_config()) to look up key in.
        key (str): The setting name to retrieve from section.
        default: The value to return if key is absent from section, or if
            its value is None or "".

    Returns:
        The value of section[key] if that key is present and its value is
            neither None nor the empty string.
        The given default if key is missing, or its value is None or "".
            Note a legitimately falsy value such as 0 or False is returned
            as-is, not treated as "empty".
    '''
    value = section.get(key)
    if value is None or value == "":
        return default
    return value
