"""Regression tests for sccs_common.config, specifically get_setting().

The motivating bug: config.toml ships with drone_system_address = "" as an
explicit placeholder. A plain dict.get(key, default) returns "" in that case
(the key IS present), not the default -- so the MAVSDK connect() call
silently received an empty string and failed with no clear diagnostic. This
was a High-severity finding in the original code review. get_setting() is
the fix; these tests pin down the exact behavior it must have.
"""

from sccs_common.config import get_setting, load_config


def test_get_setting_returns_value_when_present_and_non_empty():
    '''
    Verifies that get_setting() returns the section's actual value when
    the key is present and non-empty, rather than the fallback default.

    Returns:
        None: This is a pytest-style test function; it does not return a
            value, and signals failure via an assert statement.
    '''
    section = {"drone_system_address": "serial:///dev/ttyUSB0:57600"}
    assert get_setting(section, "drone_system_address", "default") == "serial:///dev/ttyUSB0:57600"

def test_get_setting_falls_back_when_key_missing():
    '''
    Verifies that get_setting() returns the provided default when the
    requested key is not present in the section dict at all.

    Returns:
        None: This is a pytest-style test function; it does not return a
            value, and signals failure via an assert statement.
    '''
    section = {}
    assert get_setting(section, "drone_system_address", "the-default") == "the-default"

def test_get_setting_falls_back_when_value_is_empty_string():
    """This is the specific case that was broken before: the key is present
    but blank (config.toml's shipped placeholder), and must still fall back
    to the default rather than returning "".
    """
    section = {"drone_system_address": ""}
    assert get_setting(section, "drone_system_address", "the-default") == "the-default"

def test_get_setting_does_not_treat_zero_or_false_as_empty():
    """Only None and "" should trigger the fallback -- a legitimately falsy
    but meaningful value like 0 or False must be returned as-is.
    """
    section = {"max_disappeared": 0}
    assert get_setting(section, "max_disappeared", 30) == 0

    section = {"use_multicast": False}
    assert get_setting(section, "use_multicast", True) is False

def test_load_config_returns_empty_dict_for_missing_file():
    '''
    Verifies that load_config() returns an empty dict, rather than raising,
    when pointed at a path that does not exist.

    Returns:
        None: This is a pytest-style test function; it does not return a
            value, and signals failure via an assert statement.
    '''
    result = load_config(path="/nonexistent/path/config.toml")
    assert result == {}
