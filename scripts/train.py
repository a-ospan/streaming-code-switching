"""Train the dual-head prefix-only predictor and evaluate it per language pair.

Data: train sentences drawn with temperature sampling over pairs (alpha, default 0.3).
Model: XLM-R with a causal mask, or XGLM (natively causal); see src/model.py.
Loss: lambda_sw * CE(switch) + lambda_dur * CE(duration, ignore_index=-1).
Threshold for "switch" is chosen on a sample of TRAIN sentences, never on the evaluated split.

Outputs:
  results/runs/<run_name>/   config, per-stratum metrics, overall metrics, duration confusion, loss log
  checkpoints/<run_name>/    model.pt (final weights) and last.pt (resume state), gitignored

Long runs (Colab): --save-every N writes checkpoints/<run_name>/last.pt every N steps. After a
disconnect, rerun the same command with --resume. The run continues from that step with the
same optimizer and schedule; the sampler restarts with a seed derived from the step, so the
batches differ from an uninterrupted run but follow the same sampling distribution.

Examples:
  # pipeline check on CPU with a tiny random backbone (minutes, numbers meaningless)
  python -m scripts.train --model FacebookAI/xlm-roberta-base --tiny --max-steps 30 --train-per-strata 200 --eval-per-strata 50
  # real run on a GPU
  python -m scripts.train --model FacebookAI/xlm-roberta-base --max-steps 20000 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from src.dataset import SwitchDataset, make_collate, pair_sampling_probs, temperature_sampler  # noqa: E402
from src.labels import IGNORE  # noqa: E402
from src.metrics import best_threshold, evaluate, save_run  # noqa: E402
from src.model import SwitchPredictor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "sentences"
RUNS = ROOT / "results" / "runs"
CHECKPOINTS = ROOT / "checkpoints"
COLUMNS = ["row_id", "sent_idx", "pair", "strata", "text", "starts", "ends", "y_sw", "y_dur"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(split: str, per_strata: int | None, seed: int) -> pd.DataFrame:
    """Sentences of one split with at least one target; optionally a fixed sample per stratum."""
    filters = [("split", "==", split), ("n_tokens", ">", 1)]
    if per_strata is None:
        return pd.read_parquet(PROCESSED, columns=COLUMNS, filters=filters)
    keys = pd.read_parquet(PROCESSED, columns=["row_id", "sent_idx", "strata"], filters=filters)
    picked = keys.groupby("strata").sample(per_strata, random_state=seed)[["row_id", "sent_idx"]]
    df = pd.read_parquet(PROCESSED, columns=COLUMNS, filters=filters + [("row_id", "in", sorted(set(picked.row_id)))])
    return df.merge(picked, on=["row_id", "sent_idx"]).reset_index(drop=True)


@torch.no_grad()
def predict(model, loader, device) -> dict[str, np.ndarray]:
    """Switch probability and duration prediction at every target position."""
    model.eval()
    out = {k: [] for k in ("strata", "y_sw", "p_switch", "y_dur", "dur_pred")}
    for batch in loader:
        res = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        p = res.sw_logits.softmax(-1)[..., 1].cpu()
        d = res.dur_logits.argmax(-1).cpu()
        m = batch["sw_labels"] != IGNORE
        rows = m.nonzero()[:, 0]
        out["strata"].extend(batch["strata"][i] for i in rows.tolist())
        out["y_sw"].append(batch["sw_labels"][m].numpy())
        out["p_switch"].append(p[m].numpy())
        out["y_dur"].append(batch["dur_labels"][m].numpy())
        out["dur_pred"].append(d[m].numpy())
    model.train()
    return {k: np.array(v) if k == "strata" else np.concatenate(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. FacebookAI/xlm-roberta-base or facebook/xglm-564M")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--tiny", action="store_true", help="tiny randomly initialized backbone (pipeline check)")
    ap.add_argument("--alpha", type=float, default=0.3, help="temperature sampling exponent over pairs")
    ap.add_argument("--lambda-sw", type=float, default=1.0)
    ap.add_argument("--lambda-dur", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--grad-accum", type=int, default=1, help="batches per optimizer step (effective batch = product)")
    ap.add_argument("--fp16", action="store_true", help="mixed precision on CUDA (faster, less memory)")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (2 on Colab, 0 on Windows)")
    ap.add_argument("--save-every", type=int, default=0, help="write a resumable checkpoint every N steps (0 = off)")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<run_name>/last.pt")
    ap.add_argument("--train-per-strata", type=int, default=None, help="limit training sentences per stratum")
    ap.add_argument("--eval-per-strata", type=int, default=2000, help="val sentences per stratum (0 = all)")
    ap.add_argument("--threshold-per-strata", type=int, default=500, help="train sentences per stratum for the threshold")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    short = args.model.split("/")[-1]
    run_name = args.run_name or f"{short}{'-tiny' if args.tiny else ''}_a{args.alpha}_s{args.max_steps}_seed{args.seed}"
    run_dir = RUNS / run_name
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; choose another --run-name instead of overwriting results")
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.tiny:
        cfg = AutoConfig.from_pretrained(args.model)
        small = dict(num_hidden_layers=2, hidden_size=64, num_attention_heads=2, intermediate_size=128) \
            if cfg.model_type != "xglm" else dict(num_layers=2, d_model=64, attention_heads=2, ffn_dim=128)
        for k, v in small.items():
            setattr(cfg, k, v)
        model = SwitchPredictor.from_config(cfg, lambda_sw=args.lambda_sw, lambda_dur=args.lambda_dur)
    else:
        model = SwitchPredictor.from_pretrained(args.model, lambda_sw=args.lambda_sw, lambda_dur=args.lambda_dur)
    model.to(device).train()

    train_df = load_split("train", args.train_per_strata, args.seed)
    thr_df = load_split("train", args.threshold_per_strata, args.seed + 1)
    eval_df = load_split(args.eval_split, args.eval_per_strata or None, args.seed)
    print(f"sentences: train {len(train_df)}, threshold {len(thr_df)}, {args.eval_split} {len(eval_df)} on {device}")
    probs = pair_sampling_probs(train_df.pair, args.alpha)
    print("pair sampling probabilities:", probs.round(3).to_dict())

    head_params = list(model.sw_head.parameters()) + list(model.dur_head.parameters())
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": args.lr},
                             {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * args.max_steps), args.max_steps)
    use_amp = args.fp16 and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = CHECKPOINTS / run_name
    last = ckpt_dir / "last.pt"
    start_step, log = 0, []
    if args.resume:
        if not last.exists():
            raise SystemExit(f"--resume given but {last} does not exist")
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start_step, log = state["step"], state["log"]
        print(f"resumed from {last} at step {start_step}")

    collate = make_collate(tok.pad_token_id)
    n_samples = (args.max_steps - start_step) * args.grad_accum * args.batch_size
    sampler = temperature_sampler(train_df.pair, args.alpha, n_samples, args.seed + start_step)
    loader_kw = dict(collate_fn=collate, num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(SwitchDataset(train_df, tok, args.max_length), batch_size=args.batch_size,
                              sampler=sampler, **loader_kw)
    eval_loader = DataLoader(SwitchDataset(eval_df, tok, args.max_length), batch_size=64, **loader_kw)
    thr_loader = DataLoader(SwitchDataset(thr_df, tok, args.max_length), batch_size=64, **loader_kw)

    acc = {"loss": 0.0, "loss_sw": 0.0, "loss_dur": 0.0, "n": 0}
    step = start_step
    for i, batch in enumerate(train_loader, start=1):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            res = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                        batch["sw_labels"].to(device), batch["dur_labels"].to(device))
        scaler.scale(res.loss / args.grad_accum).backward()
        for k in ("loss", "loss_sw", "loss_dur"):
            acc[k] += getattr(res, k).item()
        acc["n"] += 1
        if i % args.grad_accum:
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad()
        step += 1
        if step % args.log_every == 0 or step == args.max_steps:
            row = {"step": step, **{k: acc[k] / acc["n"] for k in ("loss", "loss_sw", "loss_dur")},
                   "elapsed_s": round(time.time() - t0)}
            log.append(row)
            print(json.dumps(row), flush=True)
            acc = {k: 0.0 for k in acc} | {"n": 0}
        if args.save_every and (step % args.save_every == 0 or step == args.max_steps):
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            tmp = ckpt_dir / "last.pt.tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "step": step, "log": log, "args": vars(args)}, tmp)
            tmp.replace(last)  # swap in one move so a disconnect mid-write cannot corrupt last.pt

    thr_pred = predict(model, thr_loader, device)
    threshold = best_threshold(thr_pred["y_sw"], thr_pred["p_switch"])
    ev = predict(model, eval_loader, device)
    table, overall, cm = evaluate(ev["strata"], ev["y_sw"], ev["p_switch"], ev["y_dur"], ev["dur_pred"], threshold)

    config = {**vars(args), "run_name": run_name, "device": device, "threshold": threshold,
              "train_sentences": len(train_df), "eval_sentences": len(eval_df),
              "pair_sampling_probs": probs.to_dict(), "elapsed_s": round(time.time() - t0)}
    save_run(run_dir, config, table, overall, cm)
    pd.DataFrame(log).to_csv(run_dir / "train_log.csv", index=False)
    if not args.tiny:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
    print({k: round(v, 3) for k, v in overall.items() if k.startswith("macro") or k.startswith("sigma")})
    print(table[["sw_precision", "sw_recall", "sw_f1", "sw_pr_auc", "dur_macro_f1"]].round(3).to_string())
    print(f"saved {run_dir} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
