"""Shared figure style and the single save helper (vector PDF in ``paper/figures`` plus a 220-dpi PNG next to it).

Call ``use_style()`` in a figure script's ``main()``; ``pyplot`` is imported with the headless ``Agg`` backend.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from .config import FIGURES  # noqa: E402

PNG_DPI = 220
STYLE = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.figsize": (3.4, 2.6),       # one journal column
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,                 # embeddable TrueType fonts
    "ps.fonttype": 42,
}


def use_style() -> None:
    plt.rcParams.update(STYLE)


def save_figure(fig, name: str, folder: Path = FIGURES) -> Path:
    """Save ``fig`` as ``<folder>/<name>.pdf`` and ``.png``; returns the PDF path and closes the figure."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pdf = folder / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(folder / f"{name}.png", dpi=PNG_DPI)
    plt.close(fig)
    return pdf
