# Project Roadmap — RL-Based Hybrid Vehicle EMS

> **PHASE 4 (2026-08-26).** Root cause of the residual gap identified as an
> **exploration deadlock**: engine-OFF at 30-50 Nm sits 3.9-6.7 sigma from the
> actor mean under the linear action map. Fixed via `modeaware_gated`.
> **FTP75 is now AT the rule-based benchmark** (mean 3.2460 vs 3.2323, best
> seed 3.2088 = -0.7%, 3/3 charge-sustaining). **NEDC regressed** (SoC runaway
> on 2/3 seeds) because at gamma=0.20 the terminal SoC penalty is invisible and
> per-step k_fb alone cannot contain the extra OFF freedom.
> Next single experiment: raise `k_fb` on NEDC. See `PHASE4_FINAL_REPORT.md`.

> **PHASE 2 UPDATE (2026-08-26).** The Phase-1 P0 diagnosis (action geometry)
> was **REJECTED** by Q-landscape forensics and replaced by **P0-REVISED: a
> reward battery-price unit mismatch** (the reward prices battery energy
> 4.83x above ECMS's proven costate, so the reward-optimal action is engine-OFF
> in **0.0%** of states vs ECMS's 90.0%). Full evidence:
> **`RL_DIAGNOSTIC_REPORT.md`**. Per-experiment record: **`EXPERIMENT_LOG.md`**.
> Machine-readable configs: **`experiments/experiment_registry.yaml`**.
>
> **LOCKED — do not modify during RL work:** `src/env/powertrain.py`,
> `src/env/driving_cycle.py`, `src/baselines/rule_based.py`,
> `src/baselines/advanced_rule_based.py`, `src/baselines/ecms.py`, the
> validated env<->plant wiring, and the validated feasibility constraints.
> *No validated plant/environment physics may be modified during the RL
> optimization experiments unless a separate validation failure is
> demonstrated* — report a `VALIDATION CONFLICT` instead of changing them.

This file is the single working checklist for the project. Update it (not
just chat history) at the end of every session: which step you're on, what
changed, what the next decision gate is. Do not start a new experiment axis
until the current investigation below is resolved one way or the other.

**Before re-deriving any number, constant, or "does X work" check, look in
[`VERIFIED_FACTS.md`](VERIFIED_FACTS.md) first.** That file is the
separate, append-only ledger of everything already confirmed — physics
constants, locked benchmark numbers, pipeline behavior, and dated training
snapshots. This file (`ROADMAP.md`) is about *status and what to do next*;
`VERIFIED_FACTS.md` is about *what is already known and proven*. Keep them
separate — don't let status/narrative creep back into the facts ledger.

---

## 1. Objective (locked success criteria)

Train a SAC RL agent that controls the ICE/EM torque split in the parallel
hybrid EMS Gymnasium environment (`src/env/ems_env.py`) such that, on both
validated driving cycles:

1. **Beats the advanced rule-based baseline** on fuel consumption (L/100km).
2. **Is charge-sustaining**: final SoC within ~50% ± 2%.
3. **Generalizes**: a config frozen on one cycle should not collapse on the
   other (checked, not assumed).

Baselines to beat (locked — do not re-derive, see `VALIDATION.md`):

| Cycle | Rule-based baseline | ECMS stretch target |
|---|---|---|
| NEDC  | 3.506 L/100km | 3.1887 |
| FTP75 | 3.232 L/100km | 2.8097 |

Phase 4 is **done** only when a single frozen hyperparameter config beats
the rule-based baseline on both cycles simultaneously and is
charge-sustaining. Beating it on one cycle only is not done.

---

## 2. Phase status

(Supersedes `README.md`'s status checklist, which is stale — it still shows
Phases 2–4 unchecked despite `CHANGELOG.md` documenting them through v3.1.0.)

| Phase | Status | Evidence |
|---|---|---|
| 0 — Scoping, repo structure | Done | — |
| 1 — Pure-Python powertrain, MATLAB validation | Done, locked | `VALIDATION.md` |
| 2 — Gymnasium environment | Done | `CHANGELOG.md` [2.0.0] |
| 3 — SAC pipeline (PER, n-step, lookahead, checkpoint/logging, audit) | Done | `CHANGELOG.md` [3.0.0]/[3.1.0], 211/211 tests pass |
| **4 — Train agent to beat baseline** | **In progress — currently failing** | see §3 below |

---

## 3. Where we actually are (as of 2026-08-26)

Two long runs are in flight, each targeting 1,500,000 timesteps
(`--lookahead 5 --n-step 5`, `--prefill-mode none`, blank-slate):

- `models/NEDC` — 420,900 / 1,500,000 steps (~28%)
- `models/FTP75` — 422,100 / 1,500,000 steps (~28%)

Objective verdict from `python -m results.figures --run models/<cycle>`
(built into the repo for exactly this purpose — trust it over eyeballing
the CSV):

- **NEDC**: plateaued ~4.9 L/100km vs. 3.506 benchmark (+~40%). Not
  charge-sustaining. "ASSIST BLOB" pattern flagged: `OFF=10.5% ASSIST=26.6%
  LPS=46.0% ONLY=0.0% REGEN=17.0%` (ECMS: `off=53.1% assist=0.2%`).
- **FTP75**: plateaued ~4.5 L/100km vs. 3.232 benchmark. Quartile trend is
  **worsening** (Q1 4.468 → Q4 4.517), SoC drifting 47%→56% — mild
  instability, not just slow convergence. Same ASSIST BLOB pattern:
  `OFF=12.6% ASSIST=27.4%` (ECMS: `off=40.4% assist=6.0%`).

**Mode terminology (see `VERIFIED_FACTS.md` §G for the full table —
corrected here 2026-08-26 after an earlier mix-up):** the diagnostic fields
are `OFF` (engine off / pure electric) and `ASSIST`, NOT `ONLY` (which
means pure-engine-with-no-motor-use, and being near 0% is *correct*,
matching both benchmarks). In both cycles the agent's `OFF` time is far
below ECMS's and its `ASSIST` time is far above — it charges via LPS
almost as much as or more than ECMS, but then spends that energy in small
ASSIST increments instead of committing to sustained OFF/EV stretches.
This is the single blocking failure mode right now.

**Housekeeping: resolved.** The Phase 3 audit is committed (`e263ed9`),
and the `k_fb` reward-shaping addition is committed (`2a8cdbe`). Run
outputs (`models/`, `logs/`) are gitignored, not tracked.

---

## 4. Root-cause hypotheses for the ASSIST BLOB plateau

Two hypotheses, in the order to test them (cheapest / most concretely
evidenced first). Full reward-function audit: `VERIFIED_FACTS.md` §F.

**#1 — reward under-prices battery energy (evidenced, test first).**
`EMSEnv`'s battery-energy price (`eq_factor`, default 1.0) is flat and
SoC-independent inside a wide ±10% deadband. The project's own ECMS solver
(`src/baselines/ecms.py`, tested & proven) shows the price actually needed
for charge-sustaining behavior is `1.3125` (NEDC) / `2.4062` (FTP75), with
closed-loop SoC feedback (`k_fb=8.0`) — not a flat 1.0. This means the
agent gets almost no marginal signal on battery use across the 40-60% SoC
band where a charge-sustaining policy spends most of its time, which would
independently explain both the failure to commit (no gradient to commit
on) and the FTP75 SoC drift (battery is genuinely underpriced). **This is
directly testable with a single CLI flag change, no code edit** — see
step 1 below.

**#2 — SAC entropy structurally resists boundary actions (unconfirmed,
needs a TensorBoard trace).** Continuous `Box(-1,1)` action space
(`src/env/ems_env.py:263`) + `ent_coef="auto"` (`src/agents/train_sac.py:392`)
may bias the tanh-squashed Gaussian policy away from committing to the
extremes of the action range, where "pure EV"/"pure engine" live.

These are not mutually exclusive — reward-pricing could be the primary
driver with entropy as a secondary effect. Test #1 first because it's
free (existing flag) and grounded in numbers already proven in this repo,
not a new derivation.

---

## 5. Immediate next steps, in order

**Do not skip ahead. Do not start a new experiment axis (PER, multi-cycle
interleave, etc.) until this is resolved.**

1. ~~**Commit the Phase 3 audit**~~ — **DONE 2026-08-26**, commit `e263ed9`.
   `.gitignore` updated to exclude run outputs (`models/`, `logs/`); each
   run's own `run_config.json` remains the provenance record.
2. **Test hypothesis #1 (reward pricing).**
   - **Round 1 — static `eq_factor` (REFUTED, `VERIFIED_FACTS.md` §E
     2026-08-26):** flat `eq_factor=1.3125`/`2.4062`, no feedback, made
     `ASSIST%` and SoC drift WORSE on both cycles vs. the flat-1.0
     baseline. Root cause: `ecms.py` already proves a constant lambda
     can't charge-sustain this plant even for the optimal controller —
     this was never a fair test of the hypothesis.
   - **Round 2 — dynamic `k_fb` costate feedback (implemented, commit
     `2a8cdbe`; smoke test IN PROGRESS as of 2026-08-26):** added `k_fb`
     to `EMSEnv` (`eq_factor_eff = eq_factor + k_fb*(SOC_TARGET - soc)`,
     mirroring `ecms.py`'s proven `k_fb=8.0` exactly). `k_fb=0` verified
     as an exact no-op; 212/212 tests pass including an exact algebraic
     check of the reward shift. Running: `models_trial_kfb/{NEDC,FTP75}`,
     `--eq-factor 1.3125/2.4062 --k-fb 8.0`. **This is the correctly
     implemented test of hypothesis #1 — round 1 was not.** Check `OFF%`
     rising toward ECMS's and `ASSIST%` falling toward it (NOT `ONLY%` —
     see §3).
   - **Round 2 result (2026-08-26): gate FAILED on both cycles** —
     `OFF` still 28-45pp below ECMS on both, the core ASSIST-BLOB gap.
     Real partial effect, not neutral: `ASSIST%` recovered to baseline
     levels on both cycles (undid round 1's regression), and FTP75's SoC
     trend became non-monotonic/self-correcting instead of runaway. Not
     sufficient alone. **Keeping `k_fb=8.0` going forward** (evidenced
     improvement) while moving to hypothesis #2 — see `VERIFIED_FACTS.md`
     §E for full numbers.
3. **Hypothesis #2 tested via TensorBoard (2026-08-26) — original form NOT
   supported; refined to #3.** `ent_coef` DECREASED substantially on both
   cycles (NEDC 0.157→0.023, FTP75 0.160→0.099) — entropy pressure was
   shrinking, not stuck high, so the "entropy blocks commitment" story
   doesn't match the data. What the traces actually show: `critic_loss`
   instability — NEDC rises to a peak (12.6 @ 278k) then recovers to 3.2
   by 420k; **FTP75 climbs almost monotonically to 53.5 and does not
   recover** within available data, matching `figures.py`'s independent
   "WORSENING" flag for FTP75 specifically. Full numbers:
   `VERIFIED_FACTS.md` §E 2026-08-26.
4. **Hypothesis #3 CONFIRMED (2026-08-26) — `gradient_steps=16` fixes
   critic instability.** Direct TensorBoard comparison at matched step
   counts: FTP75 `critic_loss` at ~150k steps was 25.9 (still climbing) at
   the original `gradient_steps=64`, vs. **4.3, bounded 1.8-8.0** at
   `gradient_steps=16` — same pattern, milder, on NEDC. Not inferred, a
   direct quantitative confirmation. Downstream: best `OFF%` of all
   rounds on both cycles (NEDC 22.9%, FTP75 15.6%), NEDC's SoC trend now
   non-monotonic/bounded, FTP75's SoC hugs ~50% in the run's second half
   (50.7%→50.0%→49.3%, closest any run has gotten to ECMS's 50.13%
   target). **Gate still NOT READY** — `OFF`/`ASSIST` gap vs. ECMS hasn't
   closed, though it's the smallest gap seen yet. Full numbers:
   `VERIFIED_FACTS.md` §E 2026-08-26 (gradient_steps confirmation entry).
   **Current best config: `--eq-factor 1.3125/2.4062 --k-fb 8.0
   --gradient-steps 16`.**
5. **Extended to 500k steps via `--resume` (2026-08-26) — cycle-specific
   split result.** Gate still FAILs both, but NOT uniformly: **NEDC's best
   checkpoint is frozen at step 65,880** (430k further steps produced zero
   improvement); **FTP75 kept improving to step 296,408**, its best score
   of the whole investigation (4.207), SoC now hugging ~50-52% in the back
   half. TensorBoard confirms critic stayed bounded on both throughout (no
   re-divergence — the gradient_steps=16 fix holds over the longer
   horizon). New finding: NEDC's `ent_coef` collapsed low (~0.02-0.03) and
   froze by step 44k; FTP75's climbed back up (0.01→0.06) over training —
   plausibly why FTP75 kept improving and NEDC didn't. Full numbers:
   `VERIFIED_FACTS.md` §E 2026-08-26 (extended-run entry).
   **Next test (in progress): NEDC with a different seed**, same config,
   to check whether the step-65,880 freeze is systematic (an entropy/
   exploration issue worth fixing) or this seed's particular luck (in
   which case the multi-seed requirement in step 7 already covers it).
6. **Pre-flight gate before ANY full-length run — mandatory, not optional.**
   Run `python -m results.readiness_gate --run <smoke-test-dir>`
   (`results/readiness_gate.py`, added 2026-08-26). It checks, with actual
   numbers, not a gut call: unit tests pass, git tree is clean, the smoke
   run reached completion, the OFF/ASSIST gap vs. ECMS is within
   tolerance, and the SoC quartile trend isn't diverging. Only scale a
   config up to 1.5M steps if this gate PASSes. One variable changed at a
   time, always smoke-tested first — never skip straight to a full run on
   an untested change.
7. Once a config beats **both** baselines and is charge-sustaining on a
   single seed, **confirm on 2-3 total seeds before declaring Phase 4
   done** — everything run so far (baseline and all smoke tests) has been
   `seed=0` only, and RL training variance can produce a one-off good run
   that doesn't replicate. Then freeze the hyperparameters, write it up as
   a new "Phase 4" section in `VALIDATION.md` (same rigor as the existing
   entries — actual commands run, actual output inspected), and update
   `README.md`'s status checklist and this file's §2 table.
8. **Stretch / generalization check**: evaluate the frozen NEDC-trained
   checkpoint on FTP75 and vice versa, to catch cycle-overfitting before
   declaring Phase 4 done.

---

## 6. Session-end checklist

Before ending any session that touches training or the pipeline, update:

- [ ] §3 ("Where we actually are") with current step counts / verdicts
- [ ] §5 with which numbered step you're now on
- [ ] `CHANGELOG.md` / `VALIDATION.md` if anything got fixed or measured
- [ ] **`VERIFIED_FACTS.md` §E** with a new dated snapshot if training
      state changed — append, never overwrite the previous snapshot
- [ ] Commit working-tree changes (or note explicitly why not, e.g.
      mid-experiment)
