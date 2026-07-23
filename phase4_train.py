"""phase4_train.py — Phase 4 smoke/calibration trainer
(ZDA_phase4_spec.md v0.3, gate SS8.4).

Logs the SS6 mandatory diagnostics every eval (the beta-trace
successors), enforces the live go/no-go guards, and supports both the
Shakespeare trainability smoke and the dial task (answer-position masked
loss).

Diagnostics per eval, per layer (K3 variants): r2 min/p5/median, frac
r2 < 1e-2 / 1e-4 (meaningful because the score path is fp32, spec SS9.2),
attention entropy per head, gamma per head, K1-guard score-row spread.
Dense variants log entropy only.

--train_loss (dial task only): "masked" (default) trains on the single
answer-position token only, same as the graded metric. "full" trains on
every position's next-token loss instead (PHASE4_three_cells_handoff.md
Candidate A: masked-only training is ~2% signal density per 51-token
episode, a plausible starvation cause distinct from model capacity).
Grading is unaffected either way: eval val_loss/acc are always computed
at the answer position only.

Go/no-go (spec SS8.4), recorded in summary.json:
  diverged      — val loss non-finite, or > 1.2x its step-0 value at the
                  final eval
  k1_guard_min  — min over evals of the score row spread (must stay > 0:
                  query-dependence alive in-training)
  wall_s        — for the S-vs-D0p < 2x overhead comparison (post-hoc)

Optimizer rule carried from Phase 3: only >=2-D params are weight-
decayed; gamma (1-D) is never decayed, same as beta was.

Usage (smoke):  python phase4_train.py --variant S --task shakespeare
                --steps 600 --out_dir runs_p4_smoke
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from phase4_model import Phase4Model, VARIANTS
from phase4_layers import K3Attention


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--task", default="shakespeare",
                   choices=("shakespeare", "dial"))
    p.add_argument("--delta", type=int, default=0)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_iters", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--mlp", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--data_seed", type=int, default=999)
    p.add_argument("--out_dir", default="runs_p4_smoke")
    p.add_argument("--device", default="cpu")
    p.add_argument("--train_loss", default="masked",
                   choices=("masked", "full"))
    p.add_argument("--dial_gen", default="v2", choices=("v2", "v3"))
    return p.parse_args()


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = (step - a.warmup) / max(a.steps - a.warmup, 1)
    return 0.1 * a.lr + 0.45 * a.lr * (1 + math.cos(math.pi * t))


def make_optimizer(model, a):
    decay, nodecay = [], []
    for p_ in model.parameters():
        (decay if p_.dim() >= 2 else nodecay).append(p_)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, 0.95))


def diagnose(model, x):
    """SS6 diagnostics from one batch: walks blocks manually so each
    attention's scores() sees its true input."""
    out = []
    with torch.no_grad():
        h = model.emb(x)
        for blk in model.blocks:
            xin = blk.ln1(h)
            aux, s = blk.attn.scores(xin)
            att = torch.softmax(s, dim=-1)
            ent = -(att.clamp_min(1e-12).log() * att).sum(-1)  # (B,H,T)
            d = {"entropy": ent.mean(dim=(0, 2)).tolist()}
            # K1 guard: raw-score row spread over shared causal support
            T = s.shape[-1]
            half = T // 2
            spread = (s[..., half:, :half]
                      - s[..., half:half + 1, :half]).abs()
            d["k1_row_spread"] = float(spread.max())
            if aux is not None:                       # K3 variants
                r2 = aux.flatten().float()
                # sort-based nearest-rank quantiles — GPU hotfixes
                # 2026-07-22 (PHASE4_colab_launch_log SS3): the original
                # torch.quantile call built its q tensor on CPU against
                # CUDA input, and torch.quantile has a hard 2^24-element
                # cap < the pooled r2 tensor at grid dims (~25.2M).
                # Pooled fields are DESCRIPTIVE-only; vs torch.quantile's
                # linear interpolation, p5/med shift by <=1 order
                # statistic and r2_min is exact either way.
                n = r2.numel()
                r2s, _ = r2.sort()
                q = torch.stack([r2s[0],
                                 r2s[max(0, int(0.05 * (n - 1)))],
                                 r2s[int(0.5 * (n - 1))]])
                d.update(r2_min=float(q[0]), r2_p5=float(q[1]),
                         r2_med=float(q[2]),
                         frac_r2_lt_1e2=float((r2 < 1e-2).float().mean()),
                         frac_r2_lt_1e4=float((r2 < 1e-4).float().mean()),
                         gamma=blk.attn.gamma.tolist())
            out.append(d)
            h = blk(h)
    return out


def main():
    a = parse()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    if dev.type == "cuda":
        # spec SS9.2 hygiene rule: TF32 quantizes the score path's near-
        # manifold distance the same way fp16 storage does (10-bit
        # mantissa); PyTorch defaults this on for CUDA matmul, and
        # zda_grid.py (Phase 1-3) explicitly enables it for speed, which
        # would be wrong to inherit here. Disabled globally, not just for
        # K3's score path -- Phase 4 dims are small enough that the cost
        # is negligible against the S/D0p < 2x wall-clock bound.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if a.task == "shakespeare":
        from data import CharDataset
        ds = CharDataset()
        vocab = ds.vocab_size
        gtrain = torch.Generator().manual_seed(a.data_seed)

        def train_batch():
            x, y = ds.get_batch("train", a.batch_size, a.ctx, gtrain)
            return x.to(dev), y.to(dev), None

        def eval_batches(step):
            g = torch.Generator().manual_seed(a.data_seed * 100003 + step)
            for _ in range(a.eval_iters):
                x, y = ds.get_batch("val", a.batch_size, a.ctx, g)
                yield x.to(dev), y.to(dev), None
    else:
        import dial_data
        dial_mod = dial_data if a.dial_gen == "v2" else __import__("dial_data_v3")
        vocab = dial_mod.vocab_size()
        rtrain = np.random.default_rng(a.data_seed)

        def train_batch():
            return dial_mod.make_batch(a.delta, a.batch_size, rng=rtrain,
                                       device=dev)

        def eval_batches(step):
            g = np.random.default_rng((a.data_seed, step, a.delta))
            for _ in range(a.eval_iters):
                yield dial_mod.make_batch(a.delta, a.batch_size, rng=g,
                                          device=dev)

    model = Phase4Model(a.variant, vocab, a.d_model, a.n_heads,
                        a.n_layers, a.mlp, seed=a.seed).to(dev)
    opt = make_optimizer(model, a)
    run_dir = os.path.join(a.out_dir, f"{a.variant}_seed{a.seed}"
                           + (f"_d{a.delta}" if a.task == "dial" else "")
                           + (f"_trainfull" if a.train_loss == "full" else "")
                           + f"_lr{a.lr:g}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"variant={a.variant} task={a.task} params={model.n_params():,} "
          f"vocab={vocab} device={dev}")

    def eval_loss_fn(x, y, mask):
        logits = model(x)
        if mask is None:
            return F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        from dial_data import masked_loss
        return masked_loss(logits, y, mask)

    def train_loss_fn(x, y, mask):
        if mask is not None and a.train_loss == "full":
            logits = model(x)
            return F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        return eval_loss_fn(x, y, mask)

    t0, val0, k1_min = time.time(), None, float("inf")
    log = open(os.path.join(run_dir, "eval_log.jsonl"), "w")
    for step in range(a.steps + 1):
        if step % a.eval_interval == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                losses, accs = [], []
                for b in eval_batches(step):
                    losses.append(eval_loss_fn(*b).item())
                    if a.task == "dial":       # answer-token accuracy at
                        xb_, yb_, mask_ = b    # every masked position
                        logits = model(xb_)
                        sel = mask_.reshape(-1)
                        pred = logits.reshape(-1, vocab)[sel].argmax(-1)
                        tgt = yb_.reshape(-1)[sel]
                        accs.append((pred == tgt).float().mean().item())
                vl = float(np.mean(losses))
                acc = float(np.mean(accs)) if accs else None
            xb = next(iter(eval_batches(step)))[0]
            diag = diagnose(model, xb)
            k1 = min(d["k1_row_spread"] for d in diag)
            k1_min = min(k1_min, k1)
            if val0 is None:
                val0 = vl
            rec = {"step": step, "val_loss": vl, "acc": acc,
                   "k1_row_spread": k1,
                   "elapsed_s": round(time.time() - t0, 1), "diag": diag}
            log.write(json.dumps(rec) + "\n"); log.flush()
            acc_s = f" | acc {acc:.3f}" if acc is not None else ""
            print(f"step {step:5d} | val {vl:.4f}{acc_s} | "
                  f"k1 spread {k1:.2e} | {rec['elapsed_s']:.0f}s")
            model.train()
        if step == a.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        x, y, mask = train_batch()
        loss = train_loss_fn(x, y, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not torch.isfinite(loss):
            print("DIVERGED: non-finite train loss"); break
    log.close()

    diverged = (not np.isfinite(vl)) or vl > 1.2 * val0
    summary = {"variant": a.variant, "task": a.task, "delta": a.delta,
               "seed": a.seed, "n_params": model.n_params(),
               "final_val_loss": vl, "val0": val0,
               "final_acc": acc if a.task == "dial" else None,
               "diverged": bool(diverged), "k1_guard_min": k1_min,
               "wall_s": round(time.time() - t0, 1),
               "config": vars(a)}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"done: {run_dir}  final val {vl:.4f}  "
          f"diverged={diverged}  k1_min={k1_min:.2e}  "
          f"({summary['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
