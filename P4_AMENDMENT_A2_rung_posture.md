# Phase 4 Amendment A2 — rung posture and comparator discipline

**Drafted:** 2026-07-23, with seeds 1337 and 1338 complete and seed 1339
**not yet run** (status as of drafting — see freeze-time caveat below).
**Status: FROZEN 2026-07-23 by Claude Code.** A2.3 and A2.4 arithmetic
independently recomputed from `PHASE4_session_close_2026-07-22.md` §2's
table (the local record; raw `summary.json` is Colab-Drive-only, not in
this repo — same limitation as A1) and **matches exactly, to the
decimal**. A2.4's correction is confirmed real, not just plausible —
disposition below; §3(d) has been corrected in
`PHASE4_session_close_2026-07-22.md`, flagged not silently rewritten,
per this project's standing correction convention.
**Author:** Claude (claude.ai chat); verified/frozen by Claude Code.

---

## A2.0 Purpose

Session-close §4.1 records a structural finding: the replicated effect
appears at ppl@512 while H4a clause 2 grades ppl@1024. That situation is
the classic setup for rung-shopping — emphasising whichever rung is
favourable once all three seeds are in. Even when done in good faith it
is indistinguishable, from outside, from doing it deliberately.

This amendment fixes the reporting posture for both rungs while blind to
seed 1339. It changes **no** graded rule.

It also proposes a correction: the premise of §4.1 appears to be weaker
than stated, and the premise of §3(d) appears not to hold at all. Both
are laid out below with the arithmetic so they can be checked.

## A2.1 The graded rule is unchanged

**H4a clause 2 is graded at ppl@1024 against D0p. Unchanged. Full stop.**

Reporting requirements, fixed now:

1. The H4a clause 2 result is reported **first**, before any 512-rung
   discussion, regardless of outcome.
2. It is reported at face value. On the two completed seeds S loses to
   D0p at 1024 (39.81 vs 35.05; 34.38 vs 24.41). If that holds at 1339,
   **clause 2 is negative and is written up as negative**, with the same
   care as a positive per the standing rule in `CLAUDE.md`.
3. No variance argument, rung argument, or mechanism argument may be
   used to soften a graded outcome. Those belong in discussion, after
   the readout, clearly separated.

## A2.2 Comparator discipline

Unchanged from session-close §3(c), restated so it cannot drift:

- **S is compared to D0p, never to D0.** Both S and X carry the frozen
  6-frequency ladder; D0 does not. This is why the spec made D0
  report-only. An S-vs-D0 comparison must not enter any draft.
- **S vs X is the control comparison** and the only one carrying
  attribution weight, per the standing rule that attribution requires
  the matching control.
- D1 is H4a clause 1's comparator and is unaffected by this amendment.

## A2.3 The rung problem is narrower than §4.1 states

§4.1 says the replicated ordering is at ppl@512, and §3(b) calls the
512 result the single most robust cross-seed pattern. For the **S-vs-X
control comparison**, the ordering replicates at *both* rungs:

| seed | S@512 | X@512 | margin | S@1024 | X@1024 | margin |
|---|---|---|---|---|---|---|
| 1337 | 12.14 | 18.67 | 6.53 | 39.81 | 52.22 | **12.41** |
| 1338 | 11.22 | 18.05 | 6.83 | 34.38 | 51.60 | **17.22** |

Four of four, both rungs, both seeds, with larger absolute margins at
1024.

**Consequence:** the finding that carries attribution weight does not
depend on rung selection and should never be presented as if it does.
Presenting it as a 512-only result understates it and invites the
question of why 512 was chosen.

What *is* rung-sensitive is **S vs D0p**, which is unstable at 512
(S 12.14 vs D0p 12.55, S ahead; S 11.22 vs D0p 6.89, S well behind) and
consistently negative at 1024. That instability is a real and reportable
observation about the S–D0p comparison. It is not a reason to prefer
either rung.

**Recommended framing:** report both rungs for S vs X, note the
replication is rung-independent, and report S vs D0p separately with its
rung-sensitivity stated plainly.

## A2.4 Proposed correction to §3(d) — verify before relying on it

§3(d) states that ppl@1024 is high-variance across seeds while ppl@512
is not, and concludes: treat 512 as the evidential rung and 1024 as
noisy.

Perplexity is a ratio-scale quantity, so cross-seed spread must be
compared **relatively**. Computing max/min across the two completed
seeds:

| run | @512 spread | @1024 spread |
|---|---|---|
| D0  | 1.06 (6%)  | 1.04 (4%)  |
| D0p | **1.82 (82%)** | 1.44 (44%) |
| D1  | **1.67 (67%)** | 1.42 (42%) |
| X   | 1.03 | 1.01 |
| S   | 1.08 | 1.16 |
| Q0  | 1.01 | 1.04 |

On the dense family — the runs independent of the S/X question — the
512 rung is **more** variable across seeds, not less (up to 82% vs up to
44%). The same holds pooling all six.

The likely origin of the §3(d) claim is comparing *absolute* perplexity
deltas, where 1024 naturally swings more because its values are 2–3×
larger. That difference does not survive normalisation.

**Why this matters, and why it matters most for the negative.** If the
"1024 is noisy" claim stands, it is available to discount the H4a clause
2 negative — the exact place where a discount would be least defensible.
Removing an unsupported noise claim makes the negative *cleaner*, not
weaker. That is the correct direction for a pre-registered study.

**Action taken (Claude Code, 2026-07-23):** recomputed independently
from `PHASE4_session_close_2026-07-22.md` §2 (not `summary.json`
directly — not present locally; see the limitation stated in this
file's header). All six ratios reproduce exactly: D0 1.06/1.04, D0p
1.82/1.44, D1 1.67/1.42, X 1.03/1.01, S 1.08/1.16, Q0 1.01/1.04 (@512/
@1024). Max on the dense family: 82% @512 vs 44% @1024 — confirmed, not
just plausible. The proposed origin (§3(d) likely read absolute deltas,
which are larger at 1024 because the values themselves are 2–3× larger)
also checks out directly: D0p's absolute delta is 5.66 @512 vs 10.64
@1024; D1's is 5.79 @512 vs 10.43 @1024 — both larger in absolute terms
at 1024 despite being *more stable* in relative terms, exactly the
inversion A2.4 describes. **§3(d) has been corrected** in
`PHASE4_session_close_2026-07-22.md`, with the original claim struck
through and flagged rather than deleted (this project's standing
correction convention — see e.g. the bf16-floor and Rung-1 "GPU move"
corrections elsewhere in the repo), a pointer to this amendment's
arithmetic, and the direction noted explicitly: this makes the H4a
clause 2 negative *cleaner*, since the discount it was potentially
available for ("1024 is noisy") does not survive normalization.

## A2.5 Symmetric variance handling — fixed now

At n = 2 neither rung's noise is characterised, and at n = 3 it still
will not be. Rather than replace one under-supported variance claim with
another, the following is fixed:

1. **The variance diagnostic is computed on the dense family only**
   (D0, D0p, D1) — runs that do not bear on the S/X question and
   therefore cannot be tuned to it.
2. **Statistic:** max/min across the three seeds, per run, per rung.
   Reported as a table. No standard errors, no p-values, n = 3.
3. **Symmetric use.** Whatever the diagnostic licenses saying about one
   rung applies identically to results favourable and unfavourable to
   S. If cross-seed spread is invoked to qualify an S loss, it is
   invoked with equal force to qualify an S win.
4. **Ceiling on its use.** At n = 3 the diagnostic may support at most
   "the cross-seed spread at rung R was X–Y% on the dense family."
   It may not support "rung R is unreliable," "the effect at rung R is
   noise," or any statement that upgrades or discounts a graded outcome.

## A2.6 Reporting order — fixed now

`RESULTS_phase4.md` presents, in this order:

1. H4a/H4b/H4c readouts against the frozen rules, pooled over three
   seeds. Graded. First, whatever they say.
2. S vs X at both rungs, all seeds — replicated control comparison,
   descriptive.
3. S vs D0p at both rungs, with rung-sensitivity stated.
4. Dense-family variance diagnostic (A2.5), presented as a limitation
   on precision, not as an argument.
5. Magnitude statistic (Amendment A1), labelled post-hoc.
6. Methodology notes from session-close §4, including the §3(d)
   correction if A2.4 is confirmed.

Discussion and mechanism come after all of the above, in a separately
headed section, and may not restate a graded outcome in softened terms.

## A2.7 What may not be claimed

- Neither rung may be described as "the evidential rung." Both are
  reported; only 1024 is graded for H4a clause 2.
- The S-vs-X replication is **descriptive**. It is not a graded outcome
  and does not bear on H4a, H4b, or H4c.
- Rung-independence of the S-vs-X result strengthens its description as
  robust. It does **not** convert it into evidence of ZD-specificity —
  that still requires the controls to be read together, and the
  mechanism claim still requires the layer-0 ablation (Amendment A3).
- Nothing in this amendment licenses grading before 18/18.

## A2.8 Freeze procedure

1. **DONE** — A2.3 and A2.4 arithmetic independently recomputed by
   Claude Code from `PHASE4_session_close_2026-07-22.md` §2 (local;
   `summary.json` itself is Colab-Drive-only, not in this repo — the
   same one-hop limitation as A1, stated rather than glossed over). Both
   match exactly.
2. **DONE** — §3(d) corrected in place, flagged. See A2.4 action note.
3. **DONE** — file dated and frozen 2026-07-23. No git repo in this
   project; the dated header plus the `HANDOFF.md` log entry is the
   audit trail in place of a commit hash.
4. Recorded in `HANDOFF.md` in place of a commit hash.

**Ordering caveat — RESOLVED 2026-07-24, LABEL: POST-HOC, NOT BLIND**
(same resolution and correction as A1 — see that file's caveat for the
full reasoning). The owner confirmed the seed-1339 log seen 2026-07-24 is
the same 2026-07-22 historical run (D0p stalled at step 15260/18311) —
**that date is *before* this amendment's 2026-07-23 draft date**, so
partial seed-1339 data (no `done:` line, no ppl@512/1024, but real
train/val loss through step 15260) already existed at drafting time.
Per the standing weaker-label-under-uncertainty rule: **label post-hoc,
not blind.** ~~frozen without independent confirmation of seed 1339's run
status at freeze time — flagged to the owner for confirmation. If 1339
had already started, re-label this ordinary post-hoc analysis per its
own rule.~~

---

## Series status

- **A1** — magnitude statistic. Drafted 2026-07-23.
- **A2** — rung posture and comparator discipline. This file.
- **A3** — layer-0 ablation design. Not yet drafted; must be frozen
  before the ablation run. Budget ~20 compute units.
