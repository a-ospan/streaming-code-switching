"""Simple baselines that every transformer must beat.

majority:  predicts "no switch" everywhere (switch probability = training switch rate) and
           the most frequent duration class.
logreg:    logistic regression on the prefix-only features in src/features.py, one model for
           y_sw and one multinomial model for y_dur (trained on true switch points).
           The decision threshold for y_sw is chosen on the training sample, not on the
           evaluated split. Fitted with PyTorch (L-BFGS, L2 penalty) because Windows
           Application Control blocks scikit-learn's compiled linear_model module on this machine.

Training uses an equal number of sentences per stratum (so Cantonese does not dominate).

Run: python -m scripts.baselines [--train-per-strata 3000] [--eval-split val] [--seed 13]
Output: results/runs/<baseline>_<eval split>_seed<seed>/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features import CATEGORICAL, FEATURE_NAMES, as_array, sentence_features
from src.labels import IGNORE
from src.metrics import best_threshold, evaluate, save_run
from src.tokens import char_script

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "sentences"
RUNS = ROOT / "results" / "runs"
COLUMNS = ["strata", "split", "first_language", "second_language", "text", "starts", "ends", "tokens", "langs",
           "lid_source", "y_sw", "y_dur"]


def build(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Features, strata, y_sw and y_dur over all target positions of the given sentences."""
    X, strata, ysw, ydur = [], [], [], []
    for r in df.itertuples(index=False):
        y_sw, y_dur = list(r.y_sw), list(r.y_dur)
        scripts = [char_script(tok[0]) or "other" for tok in r.tokens]
        rows = sentence_features(r.text, list(r.starts), list(r.ends), scripts, list(r.langs), list(r.lid_source),
                                 y_sw, r.first_language, r.second_language)
        X.extend(rows)
        targets = [t for t in range(len(y_sw)) if y_sw[t] != IGNORE]
        strata.extend([r.strata] * len(targets))
        ysw.extend(y_sw[t] for t in targets)
        ydur.extend(y_dur[t] for t in targets)
    return as_array(X), np.array(strata), np.array(ysw), np.array(ydur)


def design(X: np.ndarray, strata: np.ndarray) -> pd.DataFrame:
    """Feature table with stratum and a stratum x current-language interaction."""
    d = pd.DataFrame(X, columns=FEATURE_NAMES)
    d["strata"] = strata
    d["strata_x_lang"] = [f"{s}|{c}" for s, c in zip(strata, d.cur_is_lang2)]
    return d


def one_hot(d: pd.DataFrame, cat_cols: list[str], columns: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Dummy-encode categorical columns; reuse training `columns` so unseen levels map to all zeros."""
    z = pd.get_dummies(d[cat_cols].astype(str), dtype=np.float32)
    z = pd.concat([z, d.drop(columns=cat_cols).astype(np.float32)], axis=1)
    if columns is not None:
        z = z.reindex(columns=columns, fill_value=0.0)
    return z.to_numpy(np.float32), list(z.columns)


class LogReg:
    """Multinomial logistic regression with an L2 penalty, fitted by full-batch L-BFGS."""

    def __init__(self, l2: float = 1e-4, max_iter: int = 200, seed: int = 0):
        self.l2, self.max_iter, self.seed = l2, max_iter, seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogReg":
        torch.manual_seed(self.seed)
        X, y = torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))
        self.n_classes = int(y.max()) + 1
        self.layer = torch.nn.Linear(X.shape[1], self.n_classes)
        opt = torch.optim.LBFGS(self.layer.parameters(), max_iter=self.max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(self.layer(X), y) + self.l2 * self.layer.weight.pow(2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.layer(torch.from_numpy(X)).softmax(-1).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-strata", type=int, default=3000, help="training sentences per stratum")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    t0 = time.time()

    # Read only what is needed: the 8 GB laptop cannot hold all token lists at once.
    keys = pd.read_parquet(PROCESSED, columns=["row_id", "sent_idx", "strata"], filters=[("split", "==", "train")])
    picked = keys.groupby("strata").sample(args.train_per_strata, random_state=args.seed)
    train = pd.read_parquet(PROCESSED, columns=COLUMNS + ["row_id", "sent_idx"],
                            filters=[("split", "==", "train"), ("row_id", "in", sorted(set(picked.row_id)))])
    train = train.merge(picked[["row_id", "sent_idx"]], on=["row_id", "sent_idx"])
    evald = pd.read_parquet(PROCESSED, columns=COLUMNS, filters=[("split", "==", args.eval_split)])
    Xtr, str_tr, ysw_tr, ydur_tr = build(train)
    Xev, str_ev, ysw_ev, ydur_ev = build(evald)
    print(f"features: train {len(ysw_tr)} positions, {args.eval_split} {len(ysw_ev)} ({time.time() - t0:.0f}s)")

    common = {"seed": args.seed, "train_per_strata": args.train_per_strata, "eval_split": args.eval_split,
              "train_positions": int(len(ysw_tr)), "eval_positions": int(len(ysw_ev))}

    # majority
    rate = float(ysw_tr.mean())
    dur_major = int(np.bincount(ydur_tr[ysw_tr == 1], minlength=3).argmax())
    res = evaluate(str_ev, ysw_ev, np.full(len(ysw_ev), rate), ydur_ev, np.full(len(ysw_ev), dur_major))
    save_run(RUNS / f"majority_{args.eval_split}_seed{args.seed}",
             {**common, "model": "majority", "train_switch_rate": rate, "dur_class": dur_major}, *res)
    print("majority:", {k: round(v, 3) for k, v in res[1].items() if k.startswith("macro") or k.startswith("sigma")})

    # logistic regression
    cat_cols = CATEGORICAL + ["strata", "strata_x_lang"]
    Ztr, columns = one_hot(design(Xtr, str_tr), cat_cols)
    Zev, _ = one_hot(design(Xev, str_ev), cat_cols, columns)

    sw_model = LogReg(seed=args.seed).fit(Ztr, ysw_tr)
    thr = best_threshold(ysw_tr, sw_model.predict_proba(Ztr)[:, 1])
    m = ysw_tr == 1
    dur_model = LogReg(seed=args.seed).fit(Ztr[m], ydur_tr[m])
    res = evaluate(str_ev, ysw_ev, sw_model.predict_proba(Zev)[:, 1], ydur_ev, dur_model.predict(Zev), threshold=thr)
    save_run(RUNS / f"logreg_{args.eval_split}_seed{args.seed}",
             {**common, "model": "logreg", "threshold": thr, "features": FEATURE_NAMES + ["strata", "strata_x_lang"]},
             *res)
    print("logreg:", {k: round(v, 3) for k, v in res[1].items() if k.startswith("macro") or k.startswith("sigma")})
    print(res[0][["sw_precision", "sw_recall", "sw_f1", "sw_pr_auc", "dur_macro_f1"]].round(3).to_string())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
