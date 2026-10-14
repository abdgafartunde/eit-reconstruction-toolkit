"""
scripts/_cli.py
===============
Shared command-line handling for the scripts in this directory.

Importing this module puts the repository root on ``sys.path``, so the
scripts work from a source checkout whatever the current working
directory is.  :func:`parse_args` adds ``--outdir`` (where figures are
saved) and ``--show`` (also open the figure in a window).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(
    description: str, default_outdir: Path = SCRIPTS_DIR
) -> argparse.Namespace:
    """Parse ``--outdir`` and ``--show`` and create the output directory."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=default_outdir,
        help=f"directory for output figures (default: {default_outdir})",
    )
    parser.add_argument(
        "--show", action="store_true", help="also open the figure in a window"
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    return args
