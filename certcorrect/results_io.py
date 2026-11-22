"""Result files: one ``.npz`` per (experiment, problem, seed) with arrays plus a JSON metadata record.

Every file carries the run arguments and ``environment_info()`` so figures and tables can be traced to a run.
Figure and table scripts read these files and never rerun full-order solves.
"""

import json
import logging
from pathlib import Path

import numpy as np

from .environment import environment_info

LOG = logging.getLogger(__name__)
META_KEY = "_meta_json"


def save_result(path: Path, arrays: dict[str, np.ndarray], meta: dict) -> Path:
    """Write ``arrays`` and ``meta`` (plus the environment record) to ``path``; creates parent folders."""
    if META_KEY in arrays:
        raise ValueError(f"array name {META_KEY!r} is reserved")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"meta": meta, "environment": environment_info()}
    np.savez_compressed(path, **arrays, **{META_KEY: np.array(json.dumps(record, default=str))})
    LOG.info("saved %s (%d arrays)", path, len(arrays))
    return path


def load_result(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """``(arrays, record)`` from a file written by ``save_result``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"result file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != META_KEY}
        record = json.loads(str(data[META_KEY])) if META_KEY in data.files else {}
    return arrays, record


def write_tex_macros(path: Path, macros: dict[str, str]) -> Path:
    """Write ``\\newcommand{\\name}{value}`` lines (numbers cited in the paper are never typed by hand)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for name in macros:
        if not name.isalpha():
            raise ValueError(f"LaTeX macro names must be letters only, got {name!r}")
    lines = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in sorted(macros.items())]
    path.write_text("% generated; do not edit\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return path
