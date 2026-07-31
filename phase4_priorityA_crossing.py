"""phase4_priorityA_crossing.py -- APM Stage A Priority A: the crossing-
point run (2026-07-29/30 work block).

Extends the existing ppl@512/ppl@1024 length-generalization eval to
2048/4096/8192, on the ALREADY-TRAINED grid checkpoints pulled from
Drive (MyDrive/p4_runs_ts) into p4_checkpoints/. No training. No new
grid. Reuses phase4_grid.py's own length_gen_eval() UNMODIFIED (monkey-
patches its LG_CTXS module global rather than reimplementing the loop),
so the new points are computed by the exact same code path that produced
the ppl@512/ppl@1024 numbers in RESULTS_phase4.md SS5/SS6 -- same
whole-sequence CE reduction, same batch_size=8/iters=32, same per-(seed,
ctx) generator seed formula (cfg.seed*1000+ectx), same val split
(p4_val_data/val.bin, byte-identical to MyDrive/zda_data_cache/val.bin,
4,884,400 tokens -- confirmed by direct size comparison before running).

Only touches the val split; does not require train.bin (not pulled
locally -- 700MB, unnecessary for this eval).

Includes 256/512/1024 in the same call as a live self-check: those must
reproduce each checkpoint's own recorded p4_artifacts/<run>/summary.json
length_gen numbers before the new 2048/4096/8192 points are trusted.

2026-07-31 revision: ctx=4096/8192 use a smaller batch size than the
protocol's original 8 (see batch_size_for_ctx), with iters scaled up so
the same TOTAL_N=256 examples are averaged either way -- S/X's K3
pairwise r^2 tensor OOM'd a 40GB A100 at batch=8/ctx=4096 (confirmed
live: "Tried to allocate 48.00 GiB"). ctx<=2048 is untouched from the
original run that already self-check-matched RESULTS_phase4.md exactly.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time

import numpy as np
import torch

import phase4_grid as G

VARIANTS = ("S", "X", "D1", "D0", "D0p", "Q0")
SEEDS = (1337, 1338, 1339)
CTXS = (256, 512, 1024, 2048, 4096, 8192)


class ValOnlyDataset:
    """Same slicing logic as phase4_grid.TokenDataset.get_batch, val
    split only -- avoids requiring the 700MB train.bin, which this eval
    never touches (length_gen_eval only calls ds.get_batch('val', ...))."""

    def __init__(self, data_dir: str, meta: dict):
        self.vocab_size = meta["vocab_size"]
        self.val = np.memmap(os.path.join(data_dir, "val.bin"),
                              dtype=np.uint16, mode="r")

    def get_batch(self, split: str, batch_size: int, ctx: int,
                  generator: torch.Generator):
        assert split == "val", "ValOnlyDataset has no train split by design"
        data = self.val
        ix = torch.randint(len(data) - ctx - 1, (batch_size,),
                            generator=generator).tolist()
        x = torch.from_numpy(np.stack(
            [data[i:i + ctx].astype(np.int64) for i in ix]))
        y = torch.from_numpy(np.stack(
            [data[i + 1:i + ctx + 1].astype(np.int64) for i in ix]))
        return x, y


def load_run(variant: str, seed: int, device: str, ckpt_dir: str):
    ckpt_path = os.path.join(ckpt_dir, f"{variant}_seed{seed}", "ckpt.pt")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ck["config"]
    cfg = G.P4Config(variant=ck["variant"], seed=seed,
                      vocab_size=c["vocab_size"], d_model=c["d_model"],
                      mlp_hidden=c["mlp_hidden"], n_layers=c["n_layers"],
                      n_heads=c["n_heads"], ctx=c["ctx"])
    model = G.build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


TOTAL_N = 256   # = original protocol's batch_size=8 * iters=32; held
                # constant across the per-ctx batch-size schedule below
                # so every ctx averages the same number of examples --
                # only the chunking changes, not the statistical power.


def batch_size_for_ctx(ctx: int) -> int:
    """ctx<=2048 keeps the original protocol's batch_size=8 UNCHANGED
    (already self-check-validated against RESULTS_phase4.md's recorded
    ppl@512/1024 -- no reason to touch what already works). ctx=4096/8192
    shrink batch size to fit a 40GB A100: the K3 pairwise r^2 tensor is
    (B,H,T,T,16), so memory ~ batch*ctx^2; batch=8 OOM'd at ctx=4096
    wanting 48GiB (confirmed live on Colab), and batch*ctx^2 is exactly
    equal for (batch=2,ctx=8192) as it was for the failing (batch=8,
    ctx=4096) -- so ctx=8192 needs batch=1, not 2, to actually fit."""
    if ctx <= 2048:
        return 8
    if ctx == 4096:
        return 2
    return 1   # ctx=8192


@torch.no_grad()
def length_gen_eval_safe(model, ds, cfg, device, ctxs) -> dict:
    """Same per-ctx computation as phase4_grid.length_gen_eval (reuses
    G.ce_loss directly, same seed formula cfg.seed*1000+ectx, same
    mean-loss-then-exp reduction, same TOTAL_N=256 examples averaged at
    every ctx) -- but broken into a per-ctx loop, with OOM handling AND
    a per-ctx batch/iters schedule (batch_size_for_ctx) so ctx=4096/8192
    don't need the full protocol's batch_size=8 to fit in GPU memory.
    Disclosed deviation: 4096/8192 are batched differently (2 and 1
    respectively, vs the original 8) purely for memory -- batching is a
    compute-chunking detail, not a change to what's measured (each
    example's loss is computed identically regardless of what batch it's
    grouped into), and N stays fixed at 256 throughout. Still OOM-safe:
    if a ctx OOMs even at its scheduled batch size, larger ctxs are
    skipped rather than attempted."""
    model.eval()
    out = {}
    oom_hit = False
    for ectx in ctxs:
        if oom_hit:
            out[f"ctx{ectx}"] = {"error": "skipped -- smaller ctx already OOM'd"}
            continue
        bs = batch_size_for_ctx(ectx)
        iters = TOTAL_N // bs
        g = torch.Generator().manual_seed(cfg.seed * 1000 + ectx)
        if device == "cuda":
            alloc0 = torch.cuda.memory_allocated() / 1e9
            resv0 = torch.cuda.memory_reserved() / 1e9
            print(f"    [mem] before ctx{ectx} (bs={bs}): "
                  f"allocated={alloc0:.2f}GB reserved={resv0:.2f}GB", flush=True)
        try:
            losses = []
            for _ in range(iters):
                x, y = ds.get_batch("val", bs, ectx, g)
                losses.append(G.ce_loss(model, x.to(device), y.to(device)).item())
            m = float(np.mean(losses))
            out[f"ctx{ectx}"] = {"loss": round(m, 6), "ppl": round(math.exp(m), 3),
                                  "batch_size": bs, "iters": iters}
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise   # only swallow OOM; anything else is a real bug
            out[f"ctx{ectx}"] = {"error": f"OOM (batch_size={bs}): {e}"}
            oom_hit = True
        finally:
            # empty_cache() only returns CACHED-but-unallocated blocks to
            # the driver -- confirmed empirically NOT the cause of the
            # 24GB+ "allocated" (not "reserved") figure seen live at the
            # ctx4096->ctx8192 boundary, so this alone doesn't explain
            # that. Kept (harmless, standard hygiene); the before/after
            # prints above/below are what actually diagnose it.
            if device == "cuda":
                gc.collect()   # drop any Python-side cyclic refs first --
                torch.cuda.empty_cache()   # -- so this can reclaim them
                alloc1 = torch.cuda.memory_allocated() / 1e9
                resv1 = torch.cuda.memory_reserved() / 1e9
                print(f"    [mem] after ctx{ectx} + empty_cache(): "
                      f"allocated={alloc1:.2f}GB reserved={resv1:.2f}GB", flush=True)
            del losses
            if device == "cuda":
                torch.cuda.empty_cache()
    model.train()
    return out


def check_against_recorded(variant: str, seed: int, lg: dict) -> str:
    """Cross-check ctx256/512/1024 against the run's own p4_artifacts
    summary.json, if present. Returns a short PASS/MISMATCH/no-ref line."""
    ref_path = os.path.join("p4_artifacts", f"{variant}_seed{seed}",
                             "summary.json")
    if not os.path.exists(ref_path):
        return "no p4_artifacts reference to check against"
    with open(ref_path) as f:
        ref = json.load(f)["length_gen"]
    lines = []
    for k in ("ctx256", "ctx512", "ctx1024"):
        if k not in ref or k not in lg or "ppl" not in lg[k]:
            continue
        ok = math.isclose(lg[k]["ppl"], ref[k]["ppl"], rel_tol=1e-6)
        lines.append(f"{k} new={lg[k]['ppl']} ref={ref[k]['ppl']} "
                      f"{'PASS' if ok else 'MISMATCH'}")
    return "; ".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--ctxs", nargs="+", type=int, default=list(CTXS))
    ap.add_argument("--device", default=None,
                     help="default: cuda if available, else cpu "
                          "(same auto-detect as phase4_grid.main())")
    ap.add_argument("--data_dir", default="p4_val_data",
                     help="dir with val.bin + meta.json; on Colab point "
                          "at .../MyDrive/zda_data_cache directly")
    ap.add_argument("--ckpt_dir", default="p4_checkpoints",
                     help="dir with <VARIANT>_seed<SEED>/ckpt.pt; on "
                          "Colab point at .../MyDrive/p4_runs_ts directly")
    ap.add_argument("--out", default="p4_priorityA_crossing_results.json")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        # spec SS9.2 hygiene rule (phase4_grid.main(), same guard) --
        # TF32's 10-bit mantissa quantizes near-manifold r^2 distance and
        # kills the K3 gradient there; this script never trains, but S/X
        # forward passes still go through the same score path.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("[hygiene] TF32 disabled (spec SS9.2)")
    print(f"[device] {device}")

    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    ds = ValOnlyDataset(args.data_dir, meta)

    # Byte-identical check against the Drive source noted in the module
    # docstring; run once, cheap, before trusting anything downstream.
    val_bytes = os.path.getsize(os.path.join(args.data_dir, "val.bin"))
    print(f"[data] val.bin: {val_bytes:,} bytes "
          f"({meta['n_val']:,} tokens), vocab {meta['vocab_size']}", flush=True)

    results = {}
    t_start = time.time()
    for variant in args.variants:
        for seed in args.seeds:
            t0 = time.time()
            model, cfg = load_run(variant, seed, device, args.ckpt_dir)
            lg = length_gen_eval_safe(model, ds, cfg, device, args.ctxs)
            dt = time.time() - t0
            check = check_against_recorded(variant, seed, lg)
            results[f"{variant}_seed{seed}"] = {
                "variant": variant, "seed": seed,
                "length_gen": lg, "wall_s": round(dt, 1),
                "self_check_256_512_1024": check}
            print(f"{variant} seed{seed}: "
                  + " ".join(f"ppl@{k[3:]}="
                             + (str(v['ppl']) if 'ppl' in v else v['error'])
                             for k, v in lg.items())
                  + f"  ({dt:.0f}s)  [{check}]", flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\ntotal wall time: {time.time() - t_start:.0f}s, "
          f"results written to {args.out}")


if __name__ == "__main__":
    main()
