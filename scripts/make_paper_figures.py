"""Figures for the ACM-format Project Update 1 report (reports/update1_acm/figures/).

Same data as scripts/make_report_figures.py, drawn for a two-column page: no titles inside
the image (the LaTeX caption carries them), small fonts, column width.

Run: python -m scripts.make_paper_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "update1_acm" / "figures"

# Reference palette (dataviz skill, light mode) and the blue ordinal ramp for ordered classes.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT1 = "#2a78d6"
ORDINAL = ["#86b6ef", "#2a78d6", "#104281"]

plt.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 7,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
})


def label(name: str) -> str:
    return name.replace("Philippines", "Filipino").replace("-English", "–En").replace("German-French", "German–French")


def style(ax) -> None:
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def fig_f1() -> None:
    per = pd.read_csv(ROOT / "results" / "runs" / "logreg_val_seed13" / "per_strata.csv", index_col=0)
    f1 = per.sw_f1.sort_values()
    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    ax.barh([label(s) for s in f1.index], f1, 0.72, color=SLOT1, edgecolor=SURFACE, linewidth=1)
    ax.axvline(f1.mean(), color=INK2, linewidth=0.8)
    ax.text(f1.mean() + 0.006, -0.95, f"macro {f1.mean():.3f}", color=INK2, fontsize=6.5, va="center")
    for i, (name, v) in enumerate(f1.items()):
        if i in (0, len(f1) - 1):
            ax.text(v + 0.008, i, f"{v:.3f}", va="center", color=INK, fontsize=6.5)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Switch F1 (validation)")
    style(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "baseline_f1.pdf")
    plt.close(fig)


def fig_duration() -> None:
    s = pd.read_csv(ROOT / "results" / "data_summary.csv").groupby("strata")[
        ["switches", "dur_1_2", "dur_3_6", "dur_7plus"]].sum()
    d = s[["dur_1_2", "dur_3_6", "dur_7plus"]].div(s.switches, axis=0).sort_values("dur_1_2")
    fig, ax = plt.subplots(figsize=(3.35, 3.1))
    left = np.zeros(len(d))
    names = ["1-2 tokens", "3-6 tokens", "7+ tokens"]
    for col, color, name in zip(d.columns, ORDINAL, names):
        ax.barh([label(x) for x in d.index], d[col] * 100, 0.72, left=left * 100, color=color, edgecolor=SURFACE,
                linewidth=1.2, label=name)
        left += d[col].to_numpy()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of switch points (%)")
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.13), ncol=3, frameon=False, labelcolor=INK2,
              handlelength=1.2, columnspacing=1.0, fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "duration_mix.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_f1()
    fig_duration()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
