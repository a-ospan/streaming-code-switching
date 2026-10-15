"""Evaluation shared by every model (baselines and transformers).

Inputs are flat arrays over target positions, each tagged with its stratum (language pair,
Hindi split by script). Reported per stratum and overall, where "overall" is the macro
average over strata so the large Cantonese pair cannot dominate.

y_sw:  precision, recall, F1 of the SWITCH class ("anticipatory F1"), PR-AUC, accuracy.
y_dur: accuracy and macro-F1 on positions where the TRUE y_sw = 1, plus a confusion matrix.
Universality: standard deviation of switch F1 across strata (lower = more universal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.labels import DURATION_NAMES, IGNORE


def best_threshold(y_true: np.ndarray, p_switch: np.ndarray) -> float:
    """Threshold maximizing switch F1. Choose it on training data, never on the evaluated split."""
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(y_true, p_switch)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def save_run(run_dir, config: dict, table: pd.DataFrame, overall: dict, cm: pd.DataFrame) -> None:
    """Write a run's config and metrics to results/runs/<run_name>/."""
    import json
    from pathlib import Path

    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; pick a new run name instead of overwriting results")
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (run_dir / "overall.json").write_text(json.dumps(overall, indent=2))
    table.round(4).to_csv(run_dir / "per_strata.csv")
    cm.to_csv(run_dir / "dur_confusion.csv")


def switch_metrics(y_true: np.ndarray, p_switch: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (p_switch >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average=None, zero_division=0)
    return {
        "sw_precision": float(prec[0]), "sw_recall": float(rec[0]), "sw_f1": float(f1[0]),
        "sw_pr_auc": float(average_precision_score(y_true, p_switch)) if y_true.any() else float("nan"),
        "sw_accuracy": float((y_pred == y_true).mean()),
        "sw_positive_rate": float(y_true.mean()),
        "n_positions": int(len(y_true)),
    }


def duration_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"dur_accuracy": float("nan"), "dur_macro_f1": float("nan"), "n_switches": 0}
    return {
        "dur_accuracy": float((y_true == y_pred).mean()),
        "dur_macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "n_switches": int(len(y_true)),
    }


def evaluate(
    strata: np.ndarray,
    y_sw: np.ndarray,
    p_switch: np.ndarray,
    y_dur: np.ndarray,
    dur_pred: np.ndarray,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Return (per-stratum table, overall summary, duration confusion matrix over all strata).

    All arrays are aligned over target positions (y_sw != IGNORE). dur_pred must be given at
    every position; it is only scored where y_sw == 1 and y_dur != IGNORE.
    """
    strata, y_sw, p_switch = np.asarray(strata), np.asarray(y_sw), np.asarray(p_switch)
    y_dur, dur_pred = np.asarray(y_dur), np.asarray(dur_pred)
    assert (y_sw != IGNORE).all(), "pass only target positions"

    rows = []
    for s in sorted(set(strata)):
        m = strata == s
        d = m & (y_sw == 1) & (y_dur != IGNORE)
        rows.append({"strata": s, **switch_metrics(y_sw[m], p_switch[m], threshold),
                     **duration_metrics(y_dur[d], dur_pred[d])})
    table = pd.DataFrame(rows).set_index("strata")

    metric_cols = ["sw_precision", "sw_recall", "sw_f1", "sw_pr_auc", "sw_accuracy", "dur_accuracy", "dur_macro_f1"]
    overall = {f"macro_{c}": float(table[c].mean()) for c in metric_cols}
    overall["sigma_universality_sw_f1"] = float(table.sw_f1.std(ddof=0))
    overall["sigma_universality_dur_macro_f1"] = float(table.dur_macro_f1.std(ddof=0))
    pooled = switch_metrics(y_sw, p_switch, threshold)
    overall.update({f"pooled_{k}": v for k, v in pooled.items()})

    d = (y_sw == 1) & (y_dur != IGNORE)
    cm = confusion_matrix(y_dur[d], dur_pred[d], labels=[0, 1, 2])
    cm = pd.DataFrame(cm, index=[f"true_{n}" for n in DURATION_NAMES], columns=[f"pred_{n}" for n in DURATION_NAMES])
    return table, overall, cm
