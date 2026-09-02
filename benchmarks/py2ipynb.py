#!/usr/bin/env python3
"""Convert a jupytext-style `# %%` Python script to a Jupyter .ipynb.

Depends only on the Python stdlib so it runs anywhere (the user's machine or
Kaggle). Cell markers supported:

    # %%                 -> start a code cell
    # %% [markdown]      -> start a markdown cell

Lines before the first marker that are comments (or blank) are treated as a
leading markdown cell.
"""

import json
import re
import sys
from pathlib import Path

MARKER = re.compile(r"^#\s*%%\s*(?:\[(\w+)\])?\s*(.*)$")
MAGIC = re.compile(r"^#\s*!(.*)$")


def convert(path: Path) -> dict:
    lines = path.read_text().splitlines()
    raw_cells: list[tuple[str, str]] = []
    cell_type = "code"
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        if body:
            raw_cells.append((cell_type, "\n".join(body)))
        body = []

    for line in lines:
        m = MARKER.match(line)
        if m:
            flush()
            cell_type = m.group(1) or "code"
            continue
        body.append(line)

    flush()

    cells = []
    for ctype, text in raw_cells:
        if ctype == "code":
            # Unmask `# !pip ...` cell magics into notebook `!pip ...`.
            code = "\n".join(MAGIC.sub(r"!\1", l) for l in text.splitlines())
            cells.append(
                {"cell_type": "code", "execution_count": None, "metadata": {},
                 "outputs": [], "source": code.splitlines(keepends=True)}
            )
        else:
            cells.append(
                {"cell_type": "markdown", "metadata": {},
                 "source": text.splitlines(keepends=True)}
            )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: py2ipynb.py <input.py> <output.ipynb>", file=sys.stderr)
        sys.exit(1)
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    nb = convert(src)
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out}")
