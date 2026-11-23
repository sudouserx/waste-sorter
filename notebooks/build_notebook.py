"""Regenerate kaggle_run.ipynb from kaggle_run.py (cells separated by '# %%' markers).
python notebooks/build_notebook.py"""
import json
import re
from pathlib import Path

here = Path(__file__).resolve().parent
src = (here / "kaggle_run.py").read_text()
cells = []
for block in re.split(r"^# %%", src, flags=re.M)[1:]:
    header, _, body = block.partition("\n")
    if header.strip() == "[markdown]":
        lines = [re.sub(r"^# ?", "", ln) for ln in body.strip("\n").splitlines()]
        cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
    else:
        lines = body.strip("\n").splitlines()
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
for c in cells:
    c["source"] = [ln + "\n" for ln in c["source"][:-1]] + c["source"][-1:]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(here / "kaggle_run.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {here / 'kaggle_run.ipynb'} with {len(cells)} cells")
