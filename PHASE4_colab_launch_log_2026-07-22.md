# PHASE4_colab_launch_log — 2026-07-22 (grid launch session, chat + Colab)

**For:** Claude Code, next repo session
**From:** Claude (claude.ai chat), working live with the owner during the
GPU launch
**Status at time of writing:** grid IN PROGRESS — 4 of 18 runs complete
(D0p/D0/D1/S at seed 1337), X seed 1337 mid-run, Q0 and seeds 1338/1339
pending. `p4_runs_ts/` on the owner's Drive is the live source of truth;
this document is the narrative + action items, not the results record.
**Prime directive unchanged:** the completeness gate holds. No grading,
no partial-seed conclusions. The grand summary fires the frozen §5
readout automatically at 18/18. Everything numeric below is single-seed
and descriptive.

---

## 1. Environment and layout

- Colab Pro, **A100-SXM4-40GB** (confirmed via nvidia-smi both before
  launch and after a VM recycle; driver 580.82.07, CUDA 13.0). All
  completed runs are same-silicon — no cross-class asterisks.
- Drive layout (owner's MyDrive):
  - `p4_code/` — the four source files. **The `phase4_grid.py` here is
    now AHEAD of the repo** (two GPU hotfixes, §3). Canonical until
    synced back.
  - `zda_data_cache/` — TinyStories build artifacts (train.bin
    700,000,000 bytes = 350M uint16 tokens; val.bin 9,768,800 = 4,884,400
    tokens; meta.json; tok4096.json). One-time build done; every future
    session restores from here.
  - `p4_runs_ts/` — run outputs (out_root). Eval logs, checkpoints,
    summaries land here directly.
- Launch banner verified on GPU: match table byte-identical to the
  step-4 repo table with asserts live; preflight [P1]–[P5] all OK —
  first-ever preflight pass on CUDA, including [P4] ladder verification
  and [P5] T=1024 forward.
- Grid arithmetic: 18,311 steps/run × 16,384 tokens/step = 300.0M
  tokens; eval every 1,526 steps.

## 2. Incident log (chronological, all resolved)

1. **Missing module at first launch.** The three-file Colab upload set
   in HANDOFF §4 omitted `sedenion_kernel.py`, which
   `phase4_layers.py` imports at module level. The correct set is FOUR
   files: `sedenion_kernel.py`, `phase4_layers.py`, `phase4_model.py`,
   `phase4_grid.py`. → HANDOFF correction, action item A3.
2. **GPU bug #1 — device mismatch in `diagnose()`.** The pooled-stats
   line `torch.quantile(r2, torch.tensor([0.0, 0.05, 0.5]))` builds its
   q tensor on CPU against CUDA input. Crashed S at step 0, first
   attempt. Line was inherited VERBATIM from `phase4_train.py` for
   smoke-log continuity — the smoke only ever ran on CPU, so the bug was
   latent there and remains so (action item A2). Dense variants never
   hit it (aux is None).
3. **VM recycle + zombie Drive mount.** Runtime recycled after the
   crash; the new VM reported Drive "already mounted" while the FUSE
   connection was dead (ls saw nothing). `force_remount=True` fixed it.
   Lesson baked into the recovery stack (§4). Silver lining: the queued
   cache-copy cell had executed after the crash, so `zda_data_cache/`
   existed on Drive and the data rebuild was skipped.
4. **GPU bug #2 — `torch.quantile` size cap.** Second S attempt crashed
   at the same line: torch.quantile has a hard 2^24 = 16,777,216-element
   cap, and the pooled r² tensor at grid dims is 64×6×256×256 ≈ 25.2M.
   Debug dims (~33k) and CPU smoke never approached it. Replaced with
   sort-based nearest-rank quantiles (exact, uncapped). **The graded
   path was never affected by either bug**: the per-head causal-support
   tensor is ~12.6M elements, under the cap, and its q-values were
   already device-correct scalars.
5. **Drive eventual consistency (operational finding, not a bug).** A
   second reader VM saw `eval_log.jsonl` at 0 bytes with fresh mtime
   for many minutes while the writer VM was demonstrably flushing
   records. Implication: cross-VM mid-run log reading is unreliable
   until file close, and the "resume loses at most one eval interval"
   guarantee weakens to "at most one interval BEYOND whatever
   checkpoint bytes had fully propagated" in a worst-case VM death.
   Atomic writes still preclude corrupt-checkpoint resumes — staleness,
   never corruption. → HANDOFF operational note, action item A3.

## 3. The two hotfixes (Drive `p4_code/phase4_grid.py` is canonical)

Fix 1, line ~470: `torch.tensor([0.0, 0.05, 0.5], device=r2.device)`.
Fix 2, same block, replacing the pooled torch.quantile call entirely:

    n = r2.numel()
    r2s, _ = r2.sort()
    q = torch.stack([r2s[0],
                     r2s[max(0, int(0.05 * (n - 1)))],
                     r2s[int(0.5 * (n - 1))]])

Definitional footnote for the results doc: nearest-rank vs
torch.quantile's linear interpolation — a negligible shift in
DESCRIPTIVE-only pooled stats (r2_min is exact either way; p5/median
differ at most by one order statistic). The graded per-head p5 still
uses torch.quantile on per-head tensors and is untouched. Continuity
caveat: pooled p5/med in grid logs are nearest-rank; the gate-4 smoke
logs were interpolated.

## 4. The recovery stack (owner's notebook, now stable)

Six cells, top to bottom, is the entire disconnect drill; Run-all
self-heals into resuming the grid:
1. `drive.mount('/content/drive', force_remount=True)` — always force.
2. nvidia-smi + torch/TF32 fingerprint — confirm A100 before spending.
3. `pip -q install datasets tokenizers`.
4. Code restore from `p4_code/` + `ls` of all four files +
   `grep -c "device=r2.device" phase4_grid.py` (must print 1).
5. Data restore: `cp -n` from `zda_data_cache/` to `/content/zda_data`
   + listing check for train.bin.
6. Launch (identical every time):
   `python phase4_grid.py --out_root /content/drive/MyDrive/p4_runs_ts
   --data_dir /content/zda_data`
Optional between 5 and 6: `python -c "import phase4_grid; ..."` import
check — run after any edit or environment change.

## 5. Results so far — SINGLE SEED, DESCRIPTIVE, UNGRADED

Completed (summary.json on Drive):
- **D0p seed 1337:** final val 1.6644, len-gen ppl 512/1024 =
  12.553/35.055, diverged=False, k1_min=2.76, wall 2172s (~36 min).
  Textbook run; train/val gap ≤ ~0.02 throughout (300M unique tokens,
  no repeats).
- **D0 seed 1337 and D1 seed 1337:** completed on the recycled VM;
  console output was never seen but both `summary.json` files were
  pulled and read 2026-07-22 (owner downloaded from Drive, pasted to
  Claude Code — action item A4 closed for these two). Numbers, still
  descriptive/single-seed/ungraded:
  - **D0** (dense, standard multi-freq RoPE, reference only, mlp 1536,
    13,784,064 params): final val 1.6404 (val0 8.4302), diverged=False,
    k1_guard_min 2.583, len-gen ppl 512/1024 = 7.375/18.684, wall 2171s
    (~36 min).
  - **D1** (dense, positional-matched, FLOP-matched to S, mlp 1992,
    15,888,048 params): final val 1.6367 (val0 8.4608), diverged=False,
    k1_guard_min 2.687, len-gen ppl 512/1024 = 8.673/24.628, wall 2406s
    (~40 min).
  - **The report-only positional-restriction price, quantified for the
    first time (D0 vs D0p, same width):** ppl@512 7.375→12.553 (+70%),
    ppl@1024 18.684→35.055 (+88%) — restricting every head to a single
    frequency (the constraint K3's positional derivation requires) costs
    a plain dense model real length-gen quality on its own, growing with
    context, before K3 enters at all. This is the number the positional
    derivation implied but nothing before this grid had measured.
  - **A second, separable width effect (D0p vs D1, same positional
    scheme):** D1 is only wider (FLOP-matched to S) and that alone
    recovers much of what the restriction cost (ppl@1024 35.055→24.628),
    with no change to position at all — two distinct, additive-looking
    effects at this one seed, not one.
  - Full four-way table at seed 1337 (all read, S included from below):
    D0 1.6404/7.375/18.684 · D0p 1.6644/12.553/35.055 · D1
    1.6367/8.673/24.628 · S 1.7979/12.137/39.806 (val / ppl@512 /
    ppl@1024). Note this is NOT the H4a comparison (which pools all 3
    seeds and reads specific clauses) — it's a plain juxtaposition of
    what's on disk so far.
- **S seed 1337:** final val 1.7979 (deficit vs D0p: 0.1335), len-gen
  ppl 512/1024 = 12.137/39.806, diverged=False, k1_min=1.34, wall
  5197s (~87 min). Gap-to-D0p trajectory at matched steps: 0.44 →
  0.48 → 0.36 → 0.27 → 0.22 → 0.19 → 0.17 → … monotone closing with
  decaying increments (delayed-parallel shape; the decay arithmetic
  predicted a final deficit of 0.10–0.14 and landed 0.134). k1 grew
  1.34 → ~2.85 plateau — query-differentiation deepened all run, guard
  never threatened.
- **Notable single-seed observation — the 512 crossover:** S BEAT D0p
  at ppl@512 (12.137 vs 12.553) despite the in-distribution deficit,
  then lost at 1024 (39.8 vs 35.1). Identical frozen ladder in both by
  construction, so this is the SCORE FUNCTION extrapolating
  differently: S degraded ~2.0× over the first context doubling vs
  D0p's ~2.4×, then ~3.3× vs ~2.8× over the second, order flipping
  near the coarsest rung's 1024-token wavelength. Non-monotone,
  mechanism-shaped, single-seed. Top replication watch item for seeds
  1338/1339. Report-only under the frozen rules (H4a clause 2 reads
  ppl@1024 only, where S trails).

In progress:
- **X seed 1337** (as of writing, through step 4578): val at matched
  steps 3.280 / 2.586 / 2.257 vs S's 3.326 / 2.752 / 2.403 — the
  SHUFFLED tensor is ahead of the true one, lead 0.046 → 0.166 →
  0.146 (growth stopped at the last reading). k1 init 2.70 (vs S's
  1.34 — the tensors are behaviorally distinct from the first forward
  pass) climbing 3.15 → 4.55 → 4.81. If X finishes ahead, the
  interpretive frame for S shifts from "impressive despite exotic
  geometry" toward "the smooth G₂ manifold as active constraint —
  maximally-regular null structure as an unwanted regularizer, the
  scattered/singular null set as the friendlier loss landscape."
  Watch items: whether S-vs-X converges late (S's own late-game
  pattern), and above all whether X reproduces the 512 crossover —
  if it does, the crossover belongs to the kernel FAMILY, not the
  geometry; if it doesn't, the smooth manifold bought something real
  out-of-distribution. H4b′ is conditional on an H4a win and will
  likely never be graded; this comparison matters descriptively.

Wall-clock finding (recorded, real): S/X run at ~433 s/interval vs
dense ~181 s ⇒ **~2.4×, exceeding the post-hoc 2× S/D0p bound** the
grand summary will flag. Decomposition for the writeup: 1.157× is the
audited FLOP-invariant cost; the residual is THIS einsum implementation
on THIS hardware (many small 16-dim contractions, poor A100
utilization, TF32 off). Owner's framing, endorsed: a systems question,
not a property of the mathematics — a fused Triton kernel exploiting
the structure tensor's ±1 sparsity is the cheap experiment that bounds
implementation slack, and dedicated non-associative silicon is the
owner's longer arc (contingent, as ever, on the geometry earning it).
Not a grading input.

Budget actuals: dense-family ~36 min/run, S/X ~87 min/run. Remaining
from X's midpoint: X finish + Q0 (~1.4h) closes seed 1337's column;
seeds 1338/1339 ≈ 8 dense (~4.8h) + 4 S/X (~5.8h) ≈ 10.6h. Grid total
lands well inside the original 25–40h envelope despite the 2.4×.

## 6. Action items for the next repo session

A1. **Sync the grid file.** Pull Drive `p4_code/phase4_grid.py` into
    the repo as canonical; log both hotfixes as a dated note (project
    precedent: fixes logged, not silent). Run `--debug` locally as
    regression (CPU path unaffected by fix 1; fix 2 changes pooled
    p5/med by ≤1 order statistic — eyeball, don't fret).
A2. **Fix `phase4_train.py`.** It carries bug #1 verbatim (fires on ANY
    CUDA run) and would hit bug #2 at grid-scale dims. Apply both
    fixes; keep the two files' diagnose() blocks textually aligned.
A3. **HANDOFF updates:** (i) Colab upload set is FOUR files incl.
    `sedenion_kernel.py`; (ii) recovery stack (§4) as the documented
    disconnect drill; (iii) Drive eventual-consistency note and the
    weakened checkpoint-loss bound; (iv) wall-clock exceedance with the
    invariant-vs-implementation framing; (v) canonical-file note
    resolves once A1 lands.
A4. **Do not analyze early.** Completeness gate. Descriptive reading of
    finished summaries (incl. the unread D0/D1) is fine; no margins, no
    verdict language until 18/18 and the grand summary's own readout.
A5. **Replication watch list for the analysis session (pre-registered
    curiosity, not new hypotheses):** the 512 crossover; the X>S
    mid-run ordering and whether it survives to finals; the k1-at-init
    separation (S 1.34, X 2.70, dense ~2.6–2.8) as a cheap
    tensor-fingerprint diagnostic.
A6. **Memory + provenance.** Update the phase-4 memory entries with
    launch state. The owner saved a posterity screenshot of the first S
    evals — suggested name `PHASE4_first_light_S_seed1337_2026-07-22.png`;
    if it enters the repo, give it a provenance line.

## 7. One-line state for CLAUDE.md's status paragraph

Grid launched 2026-07-22 on Colab Pro A100; 4/18 runs complete at seed
1337 (two GPU hotfixes to phase4_grid.py applied live, Drive copy
canonical pending repo sync); S trains stably, final val 1.798 vs D0p
1.664 with a single-seed 512-ppl crossover flagged for replication;
X mid-run and ahead of S; wall-clock 2.4× exceeds the post-hoc 2×
bound (recorded); completeness gate holding — no grading until 18/18.
