"""Figures for the weeks 4-6 progress report (reports/weeks4-6/figures/).

Reads only saved outputs (results/data_summary.csv, results/runs/logreg_val_seed13), so it
runs in seconds and does not need the processed data.

Run: python -m scripts.make_report_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "weeks4-6" / "figures"

# Reference palette (dataviz skill, light mode): surface, inks, hairlines, categorical slots,
# and the blue ordinal ramp (steps 250 / 450 / 650) for the ordered duration classes.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT1, SLOT2 = "#2a78d6", "#eb6834"
ORDINAL = ["#86b6ef", "#2a78d6", "#104281"]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
})


def strata_totals() -> pd.DataFrame:
    s = pd.read_csv(ROOT / "results" / "data_summary.csv")
    t = s.groupby("strata")[["rows", "sentences", "positions", "switches", "dur_1_2", "dur_3_6", "dur_7plus"]].sum()
    t["switch_rate"] = t.switches / t.positions
    return t


def style_x_grid(ax) -> None:
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def title(ax, text: str, sub: str) -> None:
    ax.set_title(text, loc="left", color=INK, fontsize=12, fontweight="semibold", pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")


def fig_sampling(t: pd.DataFrame) -> None:
    """Share of training sentences per pair: natural vs temperature sampling (alpha = 0.3)."""
    sents = t.sentences.copy()
    pairs = pd.Series(sents.index.str.replace(r" \(.*\)", "", regex=True), index=sents.index)
    by_pair = sents.groupby(pairs).sum().sort_values()
    natural = by_pair / by_pair.sum()
    # Same formula as src.dataset.pair_sampling_probs (not imported: loading torch next to
    # matplotlib triggers a duplicate OpenMP runtime error on Windows).
    sampled = by_pair**0.3 / (by_pair**0.3).sum()

    fig, ax = plt.subplots(figsize=(8, 6.2))
    y = np.arange(len(by_pair))
    h = 0.38
    ax.barh(y + h / 2, natural * 100, h, color=SLOT1, edgecolor=SURFACE, linewidth=1.5, label="Natural share")
    ax.barh(y - h / 2, sampled * 100, h, color=SLOT2, edgecolor=SURFACE, linewidth=1.5,
            label="Training share with temperature sampling (alpha = 0.3)")
    ax.set_yticks(y, by_pair.index)
    ax.set_xlabel("Share of training sentences (%)")
    top = by_pair.index[-1]
    ax.text(natural[top] * 100 + 0.8, len(by_pair) - 1 + h / 2, f"{natural[top]:.0%}", va="center", color=INK, fontsize=9)
    ax.text(sampled[top] * 100 + 0.8, len(by_pair) - 1 - h / 2, f"{sampled[top]:.0%}", va="center", color=INK, fontsize=9)
    style_x_grid(ax)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK2)
    title(ax, "Rebalancing the training data", "Cantonese-English is 64% of sentences; sampling cuts it to 16%")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_sampling.png", dpi=200)
    plt.close(fig)


def fig_switch_rate(t: pd.DataFrame) -> None:
    r = t.switch_rate.sort_values()
    fig, ax = plt.subplots(figsize=(8, 6.2))
    ax.barh(r.index, r * 100, 0.7, color=SLOT1, edgecolor=SURFACE, linewidth=1.5)
    for name in [r.index[0], r.index[-1]]:
        ax.text(r[name] * 100 + 0.4, name, f"{r[name]:.1%}", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Switch points among all target positions (%)")
    style_x_grid(ax)
    title(ax, "How often the next token switches language", "y_sw = 1 rate per stratum, all splits")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_switch_rate.png", dpi=200)
    plt.close(fig)


def fig_duration_mix(t: pd.DataFrame) -> None:
    d = t[["dur_1_2", "dur_3_6", "dur_7plus"]].div(t.switches, axis=0)
    d = d.sort_values("dur_1_2")
    fig, ax = plt.subplots(figsize=(8, 6.2))
    left = np.zeros(len(d))
    names = ["1-2 tokens (class 0)", "3-6 tokens (class 1)", "7+ tokens (class 2)"]
    for col, color, name in zip(d.columns, ORDINAL, names):
        ax.barh(d.index, d[col] * 100, 0.7, left=left * 100, color=color, edgecolor=SURFACE, linewidth=2, label=name)
        left += d[col].to_numpy()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of switch points (%)")
    style_x_grid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3, frameon=False, labelcolor=INK2)
    title(ax, "Length of the segment after a switch", "y_dur class mix per stratum (only defined where y_sw = 1)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_duration_mix.png", dpi=200)
    plt.close(fig)


def fig_baseline_f1() -> None:
    per = pd.read_csv(ROOT / "results" / "runs" / "logreg_val_seed13" / "per_strata.csv", index_col=0)
    f1 = per.sw_f1.sort_values()
    macro = f1.mean()
    fig, ax = plt.subplots(figsize=(8, 6.2))
    ax.barh(f1.index, f1, 0.7, color=SLOT1, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(macro, color=INK2, linewidth=1)
    ax.text(macro + 0.005, len(f1) - 0.4, f"macro average {macro:.3f}", color=INK2, fontsize=9, va="bottom")
    for name in [f1.index[0], f1.index[-1]]:
        ax.text(f1[name] + 0.008, name, f"{f1[name]:.3f}", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Switch F1 on the validation split")
    style_x_grid(ax)
    title(ax, "Logistic regression baseline: switch F1 per stratum",
          "Prefix-only features; the majority baseline scores 0 everywhere")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_baseline_f1.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = strata_totals()
    fig_sampling(t)
    fig_switch_rate(t)
    fig_duration_mix(t)
    fig_baseline_f1()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
