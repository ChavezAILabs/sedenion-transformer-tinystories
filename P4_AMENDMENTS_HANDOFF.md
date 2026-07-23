# P4_AMENDMENTS_HANDOFF — 2026-07-23

**For:** Claude Code, next repo session
**From:** Claude (claude.ai chat)
**Mode switch:** prior art is parked. Phase 4 completion is the priority.

---

## 1. Situation

Grid is 12/18. Seed 1339 is waiting on Colab quota. Nothing is graded,
the §5 readout has not fired, and the completeness gate holds — none of
that changes.

Three pre-registration amendments have been drafted during the wait.
Their only real property is that they were written **blind to seed
1339**, and that property is destroyed the moment 1339 runs.

**Hard ordering constraint: A1, A2 and A3 must be verified and
committed BEFORE any seed-1339 run starts.** If that ordering can't be
met, mark all three as ordinary post-hoc analysis and stop presenting
them as blind. Half-credit is not available here.

## 2. The three files

| file | fixes | blocking issue |
|---|---|---|
| `P4_AMENDMENT_A1_magnitude_statistic.md` | the descriptive magnitude report (§4.2's non-discriminating H4c problem) | **has an unfilled slot** — A1.1 needs the canonical r² definition from `phase4_grid.py` §1 |
| `P4_AMENDMENT_A2_rung_posture.md` | rung reporting, comparator discipline | proposes a **correction to session-close §3(d)** — needs adjudication |
| `P4_AMENDMENT_A3_layer0_ablation.md` | ablation design, before it's run | rests on an **unverified assumption** about the γ gate |

None of them adds a hypothesis, changes a graded rule, or touches
H4a/H4b/H4c. All three explicitly re-state the standing orders.

## 3. Verify before committing — nothing here was checked against the repo

Every number in these drafts was recomputed from the prose of
`PHASE4_session_close_2026-07-22.md`. **None was read from
`summary.json` or `eval_log.jsonl`.** Treat all of it as unverified.

1. **A1.1 slot** — insert the repo's r² definition verbatim. The
   amendment cannot be frozen until this is filled.
2. **A1.4 bars** — four bars (1.5 / 0.5 / 1.0 decades, and F(X,L0,1e-2)=0)
   derived from §3(e)'s quoted minima. If those were rounded for the
   write-up, the decades move. Recompute from the logs.
3. **A2.3 — likely under-claim in the current record.** S appears to
   beat X at ppl@1024 as well as ppl@512, on both seeds, by larger
   absolute margins (12.41 and 17.22). If that holds, the S–X result is
   rung-independent, and §4.1's "the effect lives at a rung the rules
   don't grade" is too weak for the control comparison. `HANDOFF.md` §5
   and `CLAUDE.md`'s status line would both need updating.
4. **A2.4 — proposed correction to §3(d).** On relative (ratio) spread,
   the 512 rung looks *more* cross-seed variable than 1024 on the dense
   family — up to 82% vs up to 44%. This contradicts "treat 512 as
   evidential and 1024 as noisy." Possible I'm testing a different
   claim than the one intended (ordering stability vs value stability,
   or absolute vs relative). **Adjudicate explicitly; don't leave the
   claim standing unexamined.** Note the direction: removing an
   unsupported noise claim makes the H4a clause 2 negative *cleaner*.
5. **A3 γ assumption** — the design assumes γ is per-head and that
   γ = 0 fully nulls the gate, exactly as β = 0 does under T4. Confirm
   against `zda_layers.py`. If any additive term survives γ = 0, the
   intervention is wrong and A3.2 check 1 is what catches it.

## 4. Compute budget (verify rates in the Colab UI)

Pro is **100 units/month, replenishing at subscription renewal** — not
a daily free-tier reset. Exhausted balance reverts to free-tier limits.
At the community-reported A100 rate of ~13 units/hr:

- seed 1339 (6 runs, ~5.5 h) ≈ **72 units**
- clean D0p re-run (~0.6 h) ≈ **8 units** — recommended; D0p is graded
  and its resume provenance is weakened by the §6/A3 Drive
  eventual-consistency issue
- A3 ablation — **inference-only**, revised to ~5–15 units, and
  **plausibly free-tier viable**

If the ablation runs on free tier it doesn't compete with 1339 for the
monthly allocation at all. Worth testing with a single arm-I
reproduction on a T4 before assuming either way.

## 5. Suggested order

**Before seed 1339:**
1. Fill A1.1; verify A1.4 against logs.
2. Adjudicate A2.3 and A2.4 against `summary.json`.
3. Confirm the A3 γ assumption.
4. Commit all three, dated; record hashes in `HANDOFF.md`.

**Then, still during the wait:**
5. Build the A3 ablation harness; pass all four A3.2 checks; run arm I
   only (reproduce recorded grid values). Generate **no** ablation
   numbers beyond arm I.
6. Test free-tier viability for the ablation.
7. Draft `RESULTS_phase4.md` skeleton with empty readout slots.

**Also pending, lower priority:** `phase4_grid.py` must stay frozen
until 18/18 — A1's Drive→repo sync is record-keeping only, since seeds
1337/1338 ran on that exact file. A2 (the same two fixes into
`phase4_train.py`) is a different file and safe now.

## 6. Parked

- `PRIOR_ART_REVIEW_zda_section8-4_draft.md` — hypercomplex-ML branch
  and the Lie-Algebra Attention adjacency (arXiv:2606.20547). Not
  merged. Contains the Comminiello et al. Table 1 citation that
  underpins the §8.3 framing recommendation, so it shouldn't be
  dropped — but it isn't on the critical path.
- Shared-phase null check on `shuffled_structure_tensor(seed)` — still
  unassigned, still CPU-only, still gates what §3(e) may claim. Blocked
  on one question: is the shuffle a consistent permutation across all
  three index slots, or arbitrary?

## 7. Unchanged

No grading, no verdict language, no `RESULTS_phase4.md` conclusions
until 18/18 and the grand summary's own readout. No Phase 5 spec
drafting. All closed doors stay closed. Negative results get the same
care as positive ones.
