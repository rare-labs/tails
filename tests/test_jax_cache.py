import os
import subprocess
import sys

import pytest

CODE = "import tails._jax as t, jax; print(t.compilation_cache_dir()); print(jax.config.jax_compilation_cache_dir)"


def _run(env_extra, tmp_path):
    env = {
        k: v for k, v in os.environ.items() if k not in {"TAILS_CACHE", "TAILS_JAX_CACHE", "JAX_COMPILATION_CACHE_DIR"}
    }
    env.update(env_extra)
    out = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True, check=True)
    return out.stdout.split("\n")[:2]


@pytest.mark.parametrize("flag", ["0", "off"])
def test_cache_can_be_disabled(tmp_path, flag):
    ours, _ = _run({"TAILS_CACHE": str(tmp_path), "TAILS_JAX_CACHE": flag}, tmp_path)
    assert ours == "None"
    assert not (tmp_path / "jax").exists()


def test_cache_defaults_under_tails_cache(tmp_path):
    ours, jaxdir = _run({"TAILS_CACHE": str(tmp_path)}, tmp_path)
    assert ours == jaxdir == str(tmp_path / "jax")
    assert (tmp_path / "jax").is_dir()


def test_user_jax_cache_dir_is_respected(tmp_path):
    mine = tmp_path / "mine"
    ours, jaxdir = _run({"TAILS_CACHE": str(tmp_path / "t"), "JAX_COMPILATION_CACHE_DIR": str(mine)}, tmp_path)
    assert ours == jaxdir == str(mine)
    assert not (tmp_path / "t" / "jax").exists()
