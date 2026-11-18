"""Check that the tag, package version, and built distributions agree.

Run it before ``python -m build`` to compare ``pyproject.toml`` with
``tails.__version__`` (and with the git tag, when ``GITHUB_REF`` is a tag).
Run it again after the build to require one sdist, one wheel, the bundled
ECCC files, and the license texts.
"""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]

WHEEL_PATHS = (
    "tails/datasets/data/eccc_idf_v3-40_QC.csv.gz",
    "tails/datasets/data/702S006.csv",
    "tails/datasets/data/6158731.csv",
    "tails/datasets/data/1108446.csv",
    "tails/py.typed",
)


def _pyproject_version() -> str:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return meta["project"]["version"]


def _init_version() -> str:
    text = (ROOT / "src" / "tails" / "__init__.py").read_text()
    match = re.search(r'^__version__ = ["\']([^"\']+)["\']', text, re.M)
    if match is None:
        raise SystemExit("src/tails/__init__.py has no __version__")
    return match.group(1)


def main() -> None:
    version = _pyproject_version()
    init = _init_version()
    if init != version:
        raise SystemExit(f"tails.__version__ is {init}, pyproject version is {version}")

    ref = os.environ.get("GITHUB_REF", "")
    if ref.startswith("refs/tags/"):
        tag = ref.removeprefix("refs/tags/")
        expected = f"v{version}"
        if tag != expected:
            raise SystemExit(f"tag {tag} does not match {expected}")

    dist = ROOT / "dist"
    wheels = sorted(dist.glob("*.whl")) if dist.is_dir() else []
    sdists = sorted(dist.glob("*.tar.gz")) if dist.is_dir() else []
    if not wheels and not sdists:
        print(f"version {version}")
        return
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(f"expected one wheel and one sdist, found {wheels} and {sdists}")
    if version not in sdists[0].name or version not in wheels[0].name:
        raise SystemExit(f"distribution names do not contain {version}: {sdists[0].name}, {wheels[0].name}")

    names = zipfile.ZipFile(wheels[0]).namelist()
    missing = [item for item in WHEEL_PATHS if item not in names]
    if not any(name.endswith("LICENSE") for name in names):
        missing.append("LICENSE")
    if not any(name.endswith("NOTICE") for name in names):
        missing.append("NOTICE")
    if missing:
        raise SystemExit("wheel is missing " + ", ".join(missing))
    print(f"version {version}; {wheels[0].name}; {sdists[0].name}")


if __name__ == "__main__":
    main()
