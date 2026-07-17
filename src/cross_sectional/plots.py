"""
Phase 2 Chart

Exactly one committed chart: mean monthly forward return by score decile for
B1 momentum, B3 combo, and XGBoost. Default matplotlib style, no custom
colors, saved with the Agg backend so it works without a display.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import; no display on the run machine

import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = Path("outputs") / "xsec_decile_returns.png"


def plot_decile_returns(decile_returns: dict[str, list[float]], out_path: Path = OUT_PATH) -> None:
    """
    Grouped bar chart of mean monthly forward return by decile.

    Args:
        decile_returns: {label: 10 mean monthly returns}, e.g. B1 momentum,
            B3 combo, XGBoost.
        out_path: PNG destination.
    """
    x = np.arange(10)
    n = len(decile_returns)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (label, vals) in enumerate(decile_returns.items()):
        ax.bar(x + (k - (n - 1) / 2) * width, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(10)])
    ax.set_xlabel("Decile (1 = lowest score)")
    ax.set_ylabel("Mean monthly forward return")
    ax.set_title("Mean monthly forward return by score decile")
    ax.axhline(0, linewidth=0.8, color="black")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
