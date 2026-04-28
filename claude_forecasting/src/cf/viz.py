"""Plot defaults so every figure renders consistently in the report."""
import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_report_style():
    """Set matplotlib defaults that match a single-column LaTeX page (~6.5in wide)."""
    mpl.rcParams.update({
        "figure.figsize": (6.5, 4.0),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def save_figure(fig, name, figures_dir):
    """Save a figure as PNG and PDF for inclusion in the report."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png")
    fig.savefig(figures_dir / f"{name}.pdf")
    plt.close(fig)
