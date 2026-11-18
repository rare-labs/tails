"""The version users import is the version the distribution declares."""

import re
from pathlib import Path

import tails

ROOT = Path(__file__).parents[1]


def test_version_matches_pyproject():
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'(?m)^version = "([^"]+)"', text)
    assert match is not None
    assert tails.__version__ == match.group(1)
