"""The Québec IDF tutorial's Open in Colab badge opens a real notebook."""

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
NB_REL = "docs/notebooks/tutorial_quebec_idf.ipynb"
COLAB = f"https://colab.research.google.com/github/rare-labs/tails/blob/main/{NB_REL}"
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"


def _sources(nb: dict) -> str:
    chunks = []
    for cell in nb["cells"]:
        src = cell["source"]
        chunks.append("".join(src) if isinstance(src, list) else src)
    return "\n".join(chunks)


def test_quebec_tutorial_colab_badge_opens_notebook():
    page = (ROOT / "docs/tutorial_quebec_idf.md").read_text()
    script = (ROOT / "docs/scripts/tutorial_quebec_idf.py").read_text()
    nb = json.loads((ROOT / NB_REL).read_text())
    text = _sources(nb)

    assert f"[![Open In Colab]({BADGE})]({COLAB})" in page
    assert COLAB in script
    assert BADGE in script
    assert COLAB in text
    assert nb["nbformat"] == 4
    assert any(cell["cell_type"] == "code" for cell in nb["cells"])

    # The notebook is the tutorial, not an empty launcher.
    assert 'IDFData.from_eccc("701S001")' in text
    assert 'fit_idf(data, "general", "matern")' in text
    assert 'scaling_test(data, "simple")' in text
    assert "load_eccc_quebec" in text
    assert 'ci="profile"' in text
    assert 'method="bootstrap"' in text
    assert "rare-tails[plots] @ git+https://github.com/rare-labs/tails" in text

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        code = "".join(src) if isinstance(src, list) else src
        compile(code, NB_REL, "exec")
