from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build processed artifacts for codex notebooks.")
    parser.add_argument("--force", action="store_true", help="Rebuild artifacts even if they already exist.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from codex_notebooks.support import build_codex_artifacts

    build_codex_artifacts(force=args.force, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

