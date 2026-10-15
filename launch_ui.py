#!/usr/bin/env python3
"""Start the optional desktop model builder."""
from __future__ import annotations

import argparse
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description="KinkSol desktop model builder")
    parser.add_argument("model", nargs="?", help="Optional .json builder project or .py model to open")
    args = parser.parse_args()
    try:
        from builder.gui import main as launch
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Install the UI dependencies with:", file=sys.stderr)
        print("  python -m pip install -r requirements-ui.txt", file=sys.stderr)
        return 1
    return launch(args.model)


if __name__ == "__main__":
    sys.exit(main())
