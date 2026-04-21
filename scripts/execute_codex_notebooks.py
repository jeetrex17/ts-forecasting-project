from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute_notebook(path: Path, timeout: int) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
    client.execute(cwd=str(path.parent))
    nbformat.write(nb, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute codex notebooks in order.")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-notebook execution timeout in seconds.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    notebooks = [
        root / "codex_notebooks" / "00_project_map.ipynb",
        root / "codex_notebooks" / "01_eda.ipynb",
        root / "codex_notebooks" / "02_classical.ipynb",
        root / "codex_notebooks" / "03_deep.ipynb",
    ]

    for path in notebooks:
        print(f"Executing {path.name} ...")
        execute_notebook(path, timeout=args.timeout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

