"""train.py — smoke-run trainer for the ZDA variant grid (HANDOFF section 5).

Usage:  python train.py configs/v2.json

Per HANDOFF requirements:
  * AdamW (0.9, 0.95), wd 0.1 (>=2-D params only, so gate beta and LN/bias
    are NOT decayed — decaying beta would force the gate off by fiat),
    lr 3e-4 cosine + warmup, grad clip 1.0.
  * EVERYTHING seeded, including dataloader order. Batch sampling uses a
    dedicated generator seeded by (seed) only, so all variants at the same
    seed see identical data order regardless of model-side RNG consumption.
  * CSV train-loss log + JSONL eval log with per-layer/per-head beta values
    every eval step (gated variants).
  * Step-0 regression assert for V2: loss(V2) == loss(B0) to 1e-5 on an
    identical batch with identical init seed (the beta=0 invariant,
    end to end).
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import replace

import numpy as np
import torch

from data import CharDataset
from model import ZDAModel, ZDAConfig, load_config, flops_per_token


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_at(step: int, tr: dict) -> float:
    warmup, total = tr["warmup_steps"], tr["steps"]
    if step < warmup:
        return tr["lr"] * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return tr["lr_min"] + 0.5 * (tr["lr"] - tr["lr_min"]) * (1 + math.cos(math.pi * t))


def make_optimizer(model: ZDAModel, tr: dict) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": tr["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tr["lr"], betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model: ZDAModel, ds: CharDataset, tr: dict,
                  cfg: ZDAConfig, step: int) -> dict:
    """Eval batches are drawn from a generator seeded by (seed, step) only,
    so every variant is evaluated on the same batches at the same step."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        g = torch.Generator().manual_seed(
            cfg.seed * 100_000 + step * 10 + (0 if split == "train" else 1))
        losses = []
        for _ in range(tr["eval_iters"]):
            x, y = ds.get_batch(split, tr["batch_size"], cfg.ctx, g)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


def step0_regression_check(cfg: ZDAConfig, ds: CharDataset, tr: dict):
    """The beta=0 invariant, end to end: V2 and B0 built from the same seed
    must produce the same loss on the same batch."""
    g = torch.Generator().manual_seed(cfg.seed)
    x, y = ds.get_batch("train", tr["batch_size"], cfg.ctx, g)
    set_seed(cfg.seed)
    m_v2 = ZDAModel(cfg)
    set_seed(cfg.seed)
    m_b0 = ZDAModel(replace(cfg, variant="B0"))
    m_v2.eval(); m_b0.eval()
    with torch.no_grad():
        _, l_v2 = m_v2(x, y)
        _, l_b0 = m_b0(x, y)
    diff = abs(l_v2.item() - l_b0.item())
    assert diff < 1e-5, f"step-0 V2 != B0: diff={diff:.3e}"
    print(f"[regression] step-0 V2 loss == B0 loss "
          f"({l_v2.item():.6f} vs {l_b0.item():.6f}, diff {diff:.1e})  OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to configs/<variant>.json")
    ap.add_argument("--out_root", default="runs")
    args = ap.parse_args()

    cfg, tr = load_config(args.config)
    ds = CharDataset()
    cfg = replace(cfg, vocab_size=ds.vocab_size)

    run_dir = os.path.join(args.out_root, f"{cfg.variant}_seed{cfg.seed}")
    os.makedirs(run_dir, exist_ok=True)

    if cfg.variant == "V2":
        step0_regression_check(cfg, ds, tr)

    set_seed(cfg.seed)
    model = ZDAModel(cfg)
    opt = make_optimizer(model, tr)
    n_params = model.n_params()
    print(f"variant={cfg.variant} d_model={cfg.d_model} mlp={cfg.mlp} "
          f"params={n_params:,} flops/tok={flops_per_token(cfg):,}")

    # dataloader order: seeded independently of model RNG -> identical
    # across variants for the same seed
    data_gen = torch.Generator().manual_seed(cfg.seed + 777)

    train_csv = open(os.path.join(run_dir, "train_log.csv"), "w", newline="")
    csv_w = csv.writer(train_csv)
    csv_w.writerow(["step", "lr", "train_loss"])
    eval_jsonl = open(os.path.join(run_dir, "eval_log.jsonl"), "w")

    t0 = time.time()
    model.train()
    for step in range(tr["steps"] + 1):
        lr = lr_at(step, tr)
        for group in opt.param_groups:
            group["lr"] = lr

        if step % tr["eval_interval"] == 0 or step == tr["steps"]:
            ev = estimate_loss(model, ds, tr, cfg, step)
            rec = {"step": step, "lr": lr,
                   "train_loss_est": round(ev["train"], 6),
                   "val_loss": round(ev["val"], 6),
                   "betas": model.betas(),
                   "tokens": step * tr["batch_size"] * cfg.ctx,
                   "elapsed_s": round(time.time() - t0, 1)}
            eval_jsonl.write(json.dumps(rec) + "\n")
            eval_jsonl.flush()
            print(f"step {step:5d} | lr {lr:.2e} | train {ev['train']:.4f} "
                  f"| val {ev['val']:.4f} | {rec['elapsed_s']:.0f}s",
                  flush=True)

        if step == tr["steps"]:
            break

        x, y = ds.get_batch("train", tr["batch_size"], cfg.ctx, data_gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        opt.step()

        if step % tr["log_interval"] == 0:
            csv_w.writerow([step, f"{lr:.6e}", f"{loss.item():.6f}"])
            train_csv.flush()

    train_csv.close()
    eval_jsonl.close()

    final = {"variant": cfg.variant, "seed": cfg.seed, "n_params": n_params,
             "flops_per_token": flops_per_token(cfg),
             "final_val_loss": ev["val"], "final_train_loss": ev["train"],
             "betas": model.betas(),
             "wall_time_s": round(time.time() - t0, 1),
             "config": {**tr, "d_model": cfg.d_model, "mlp": cfg.mlp,
                        "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                        "ctx": cfg.ctx}}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2)
    torch.save({"model": model.state_dict(), "config": final["config"],
                "variant": cfg.variant},
               os.path.join(run_dir, "ckpt.pt"))
    print(f"done: {run_dir}  final val {ev['val']:.4f}  "
          f"({final['wall_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
