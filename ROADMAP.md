# Project Roadmap — RL-Based Hybrid Vehicle EMS

> **PHASE 9 (2026-08-28) - critic value-fidelity forensics + CQL.** The Phase-8
> "critic value-fidelity" framing is **refined**: on the in-distribution
> (ECMS-trajectory) states the critic is NOT grossly wrong - region-averaged
> min-Q ranks HIGH_EFF >= ECMS_NBHD >= LOW >= OFF in every band on both cycles,
> matching the reward and the SoC consequence. Neither pre-registered error
> type (Type-1 OFF-overvaluation, Type-2 high-load-undervaluation) is cleanly
> triggered. The real defect is a **mild systematic LOW-load bias in the
> per-state argmax** across 15-35 Nm (argmaxQ in {OFF,LOW,ECMS_NBHD} ~98%,
> HIGH_EFF ~2%) that **compounds** over a cycle into the Q-oracle SoC collapse -
> not one gross misvalued action, and not a far-OOD spike (the Q-oracle's own
> states are no more OOD than the ECMS states). **Physical SAC-ECMS
> decomposition (new, BSFC-grounded): operating-point inefficiency = NEDC +0.19
> (39%) / FTP75 +0.08 (18%); mode-selection & timing = NEDC +0.31 (61%) / FTP75
> +0.37 (81%); battery/SoC ~0.** ECMS keeps the engine in the low-BSFC
> high-load island (255 vs 290 g/kWh, eta 0.35 vs 0.32), running it less often
> but harder (NEDC 260 vs 376 engine-on steps, 79 vs 55 Nm). **Experiment A =
> CQL(H) conservative critic FAILED at every coefficient (alpha in {0.01, 0.05,
> 1.0}): trained policy runs SoC away to 78-86% (V_CE 4.7-5.6, 0/3 CS) or is CS
> only with 100+ violations; every CQL Q-oracle is non-CS (dSoC +34..+46pp);
> gap "closed" = -213%. CQL cut the OFF argmax (38%->2% at 15-30 Nm) but shifted
> mass to LOW not the efficient region. Critic-regularisation route rejected;
> best validated controller unchanged (Phase-8 CONTROL).** Next authorised:
> (B) targeted high-engine-load training coverage (no reward change), then
> (H) a part-load-penalty reward term. Algorithm swap stays gated.
> See `PHASE9_FINAL_REPORT.md`.

> **PHASE 8 (2026-08-28) - "actor is the bottleneck" DEMOTED by the Q-oracle
> test.** A Policy-B "SAC-Q oracle" (greedy arg-max of the trained twin-critic
> over a dense feasible grid, rolled through the real env) was built to measure
> the ceiling of *any* policy representation on the current critic. Result: it
> is **WORSE than the current actor and loses charge-sustaining** - NEDC
> 3.7666 (3/3 CS) -> **3.9404 (1/3 CS)**, FTP75 3.2889 (3/3 CS) -> **3.3545
> (0/3 CS)**; neither beats the rule-based benchmark. Engine-op counterfactual:
> the trained min-Q rates the actor's soft-engine operating point **above** the
> ECMS hard-engine point in every torque band on both cycles, even though the
> *immediate reward's* optimum is at a **higher** engine load than ECMS
> everywhere. **⇒ the binding constraint is CRITIC value-fidelity off the
> on-policy distribution, not the policy class and not the reward.** A
> 2-component mixture actor (8C, everything else frozen) is training as the
> mandated falsification (3 seeds x 2 cycles); it can only help if co-training
> also repairs the critic. Reward change (8H) and algorithm swap (8I) are gated
> OUT by the evidence. Next: a critic-side intervention (conservative/ensemble
> critic, or targeted coverage of the ECMS operating region). See
> `PHASE8_REPORT.md`.

> **PHASE 7 (2026-08-27) - economic/costate hypothesis REFUTED (no training).**
> Pure forensic calibration on the existing checkpoints. The residual gap is
> **NOT an economic (equivalent-factor / costate) valuation error**: the CONTROL
> policy's effective battery price (median **2.82 ECMS units** on NEDC) matches
> ECMS's *own* closed-loop effective price (median **2.78**) - both run below
> SoC target. `k_fb` is **not the lever**: actor P(OFF) is flat at 48% across
> `k_fb` {1.656, 2.5, 3.0} at the NEDC operating SoC, and the trained
> `k_fb` sweep on disk is a flat fuel plateau in [2.0, 3.0]. **Confirmed
> bottleneck: actor-side.** At NEDC 30-35 Nm the SAC critic's own arg-max wants
> engine-OFF **87%** of the time and the actor delivers **0%**; `Q(a)` is
> bimodal with the actor mean ~1.5 action-units away on the LPS lobe.
> `ERROR_reward >= 0` and `corr(ERROR_critic, eq-price) ~ 0` -> not economic,
> not temporal. **Classification: CASE D** (critic ~right, actor displaced) ->
> **CASE E** (unimodal policy class) if the next actor-side lever fails.
> Cross-cycle transfer fails charge-sustaining 0/3 both directions.
> **Next (one experiment): the pre-registered actor-side A/B - target-entropy /
> entropy-temperature, 3 seeds, everything else frozen; its failure authorizes a
> mixture / discrete-continuous policy head.** See `PHASE7_FINAL_REPORT.md`.

> **PHASE 6 (2026-08-27) - conditional-coverage hypothesis REFUTED.** A
> controlled A/B raised OFF coverage at 30-35 Nm / SoC 40-50 from **4.5% to
> 36.7%** (4.9x) and the critic did **not** respond (dQ(OFF-ASSIST)
> -0.0071 -> -0.0066). Fuel worsened on both cycles (NEDC 3.7666 -> 3.8178;
> FTP75 3.2889 -> 3.2984) and NEDC charge sustainability fell 3/3 -> 2/3.
> **Phase-5B diagnosis corrected**: at 30-35 Nm reward and critic AGREE on
> ASSIST (85-87% of states), so there was no conflict to fix.
> **New bottleneck: actor displacement at 15-30 Nm.**
> See `PHASE6_FINAL_REPORT.md`.

> **PHASE 5B (2026-08-27) - forensic closure, no training.** Direct replay
> measurement **refutes** the Phase-5 starvation inference (OFF coverage is
> 15.6% at 30-35 Nm), but reveals a **conditional** hole: only **4.5%** of
> transitions at the operating SoC (40-50%) contain OFF. The reward favours OFF
> (`dr=+0.0011`) while the critic does not (`dQ=-0.0062..-0.0222`).
> **Bottleneck = critic misestimation from conditional replay coverage**; the
> bimodal-Q story is demoted to a symptom. Next (unauthorised): a controlled
> OFF-coverage exploration schedule. See `PHASE5B_FORENSIC_CLOSURE.md`.

> **PHASE 5 (2026-08-26).** Costate gain `k_fb` identified. NEDC charge
> sustainability **solved** (1/3 -> 3/3 seeds at k_fb=2.5) with zero violations,
> but NEDC fuel is **statistically tied** with the linear reference
> (3.7666 +/- 0.0785 vs 3.7727 +/- 0.0281). Optimal k_fb is **cycle-dependent**
> (NEDC 2.5, FTP75 1.656). Root cause of the residual gap now measured: the
> critic landscape is **BIMODAL** (LPS peak / valley / OFF peak) while SAC's
> policy is unimodal, and actor sigma collapses to 0.194 leaving the OFF mode
> 4-5 sigma away. Next: raise target_entropy. See `PHASE5_FINAL_REPORT.md`.

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

---

## 7. Status update (2026-08-27) — supersedes §3/§4 narrative above

§3/§4 above describe the pre-Phase-4 "ASSIST BLOB" diagnosis and are kept for
history; they no longer reflect current status. Current state: Phases 4, 5,
5B, and 6 are complete (`PHASE5_FINAL_REPORT.md`, `PHASE5B_FORENSIC_CLOSURE.md`,
`PHASE6_FINAL_REPORT.md`; log entries E11-E14 in `EXPERIMENT_LOG.md`).
Phase 6's controlled A/B (conditional-exploration coverage fix) was REFUTED,
including a 2026-08-27 gap-closure addendum (`PHASE6_FINAL_REPORT.md` §10,
`EXPERIMENT_LOG.md` E14 addendum) that ran the missing FTP75 forensics and the
full actor A/B/C/D classification for both cycles — the outcome is unchanged
and reinforced (SAC's OFF-usage gap vs ECMS in the 30-75 Nm band is essentially
untouched by the intervention on both cycles). Current primary bottleneck:
actor-side displacement at 15-30 Nm, plus a critic value-ranking gap vs ECMS at
30-75 Nm. Next step (entropy temperature at 15-30 Nm) is proposed but NOT
started — awaiting explicit authorisation, per the standing rule against
auto-starting a second intervention.

### 7b. Phase 7 update (2026-08-27) — economic/costate forensic, NO training

Phase 7 (`PHASE7_FINAL_REPORT.md`, log entry E15) tested whether the residual
gap is an economic (equivalent-factor / costate) valuation error, *before*
authorising the actor-side experiment. **It is not.**

- Effective battery price (CONTROL rollout): median **2.82 ECMS units** (NEDC) /
  **2.72** (FTP75). Against static λ₀=1.3125 that is ~2.1×; **against ECMS's own
  closed-loop effective price (median 2.78 / 2.85) it is a match** — both
  controllers run below SoC target (median visited SoC 37.5% / 47.4%).
- `k_fb` is not the lever: at the NEDC operating SoC, actor P(OFF) is flat at
  48% across `k_fb` ∈ {1.656, 2.5, 3.0}; trained `k_fb ∈ [2.0,3.0]` is a flat
  fuel plateau (3.766→3.784, both 3/3 CS); `k_fb=1.656` (ECMS-slope-matched)
  loses charge-sustaining on NEDC (1/3). `k_fb` that would put the *median*
  price at λ₀ is ≤ 0 (refuted Phase 4/5).
- Matched states (ECMS trajectory): at NEDC 30-35 Nm the SAC critic's arg-max-Q
  wants engine-OFF **87%** of the time (≥ ECMS's 40%); the actor delivers **0%**.
  `Q(a)` is bimodal, actor mean ~1.5 action-units away on the LPS lobe.
  `ERROR_reward ≥ 0`, `corr(ERROR_critic, eq-price) ≈ 0` → not economic, not
  temporal (γ=0.20, n_step=1).
- SAC−ECMS gap ≈ 60-65% mode-selection (recoverable — critic already ranks OFF
  first), ≈ 10-25% engine operating-point (SAC runs the engine softer than
  ECMS in every band — possible action-head limitation), ≈ 1% battery-energy
  management (solved).
- Cross-cycle transfer: NEDC→FTP75 3.382 (0/3 CS), FTP75→NEDC 4.311 (0/3 CS) —
  the policy over-specialises to a cycle-specific SoC trajectory.

**Classification: CASE D → CASE E.** The one remaining actor-side lever
(target-entropy / entropy-temperature, 3 seeds, all else frozen at γ=0.20,
n_step=1, gated, `k_fb`=2.5) is the correct next experiment; **its failure
authorises a mixture / discrete-continuous policy head (CASE E)**. Do NOT
run a new costate sweep (§8 precondition not met) and do NOT switch algorithm
(CASE F not reached). Initial-SoC generalization needs `_Q_BT_IC` made an env
arg — a scoped RL-layer change, separate authorised task.
