# PHASE 12 — STAGE B: Targeted Informative-Coverage Experiment

**Experiment:** PHASE 12 · **Stage:** B · **Date/time:** 2026-08-30 ·
**Code revision:** git `90af969` + Stage-A `clip_eq_eff` flag
(`src/env/ems_env.py`, +28 lines, default-OFF; no other `src/` change) ·
**Training:** 3 seeds × 150 000 env steps (authorised 12B budget). Nothing
beyond that budget was run.

**Artifacts** (all under `results/phase12/`, nothing overwritten):
`stage_b/config_frozen.json`, `stage_b/dry_run_checks.json`,
`stage_b/train_summary.json`, `stage_b/diagnostics.json`,
`stage_b/seed{0,1,2}/{sac_ems_{50k,100k,150k,best}.zip, replay_buffer.pkl,
te_stats.json, coverage_evolution.json, eval_history.csv}`,
`te_highload.py`, `stage_b_train.py`, `stage_b_diagnostics.py`.

---

## 1. EXPERIMENT OBJECTIVE

Test: *the trained policy under-visits the high-load region because the existing
exploration does not provide sufficient informative replay coverage, preventing
the critic (and actor) from learning the high-load solution.* Distinguish
Case A (coverage supported) / B (coverage up, no vehicle benefit) / C (coverage
up, critic still doesn't learn) / D (coverage did not materially improve) /
E (Stage-A reward correction contaminated the experiment).

**Not** an ECMS-imitation experiment — no ECMS action, warm-start, expert
trajectory, or preferred torque is used anywhere.

---

## 2. FROZEN CONFIGURATION (`stage_b/config_frozen.json`)

| item | value | frozen? |
|---|---|---|
| cycle | NEDC | ✓ |
| γ | 0.20 | ✓ |
| network | MLP `[256, 256]`, twin-Q, tanh-squashed Gaussian actor | ✓ |
| `eq_factor` / `k_fb` | 0.2717 / 2.5 | ✓ |
| n_step | 1 | ✓ |
| `lambda_soc` / `soc_deadband` | 2.0 / 0.10 | ✓ |
| target_entropy / ent_coef | auto (−1.0) / auto | ✓ |
| lr / buffer / batch / τ / train_freq / gradient_steps | 3e-4 / 300k / 512 / 0.005 / 64 / 16 | ✓ |
| action bounds / feasibility logic / observation space | unchanged | ✓ |
| deterministic evaluation policy | `predict(deterministic=True)` — unchanged | ✓ |
| **reward** | **only** Stage-A `clip_eq_eff=True` (verified bitwise no-op on all CONTROL transitions; §B8 below) | 1 change |
| **training-time exploration** | high-engine-load informative-coverage injection (§3) | 1 change |
| training budget | 150 000 steps × seeds {0, 1, 2} | ✓ |

Exactly **two** changes vs the CONTROL: the Stage-A safety correction, and the
exploration intervention. No `eq_factor` / `k_fb` / γ / net / action-map / obs /
optimiser / n_step / architecture change.

---

## 3. EXACT EXPLORATION INTERVENTION (`results/phase12/te_highload.py`)

Training-time only. Overrides `_sample_action` (SB3 calls it **only** from
`collect_rollouts`; `predict()` never calls it — verified §4). `te_enabled=False`
restores stock SB3 byte-for-byte.

**Activation** (all must hold), evaluated from the decoded observation:
1. `15 ≤ T_MGB < 50 Nm` (Phase-11 low/mid-demand problem region);
2. `w_MGB > 0` (moving traction step);
3. `soc < 0.70` — **safety cap**: never inject charging actions into the
   `eq_eff < 0` zone (well above the ~37–45 % operating band, below the
   FTP75 69.9 % zero-crossing — Stage A);
4. a **non-degenerate feasible high-load interval** exists:
   `T_CE_hi_nom = 0.9 · T_CE_max_feasible(state)` (from the env's true
   `_action_to_torques` clamp), `T_CE_lo_nom = 1.3 · T_MGB`; require
   `T_CE_hi_nom − T_CE_lo_nom ≥ 5 Nm` after intersecting with `[-1, 1]`;
   otherwise **skip**.

**Injection:** with `p = 0.25`, replace the sampled action with
`a ~ Uniform(a_lo, a_hi)`, where `[a_lo, a_hi]` maps (via the monotone,
continuous `a → T_CE` relation, Phase-11 §11B) to
`T_CE ∈ [T_CE_lo_nom, T_CE_hi_nom]` intersected with `[-1, 1]`. The env's own
feasibility masks then run unchanged. The draw is **uniform over a
feasibility-defined interval** — it carries no information about a good
controller beyond "a harder engine load is reachable here".

**Realised injection (`stage_b/*/te_stats.json`, 3 seeds):**

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| env steps | 150 016 | 150 016 | 150 016 |
| steps in activation region (= feasible) | 47 232 (31.5 %) | 47 232 | 47 232 |
| **injected** | **11 598** (7.7 %) | 11 807 | 11 727 |
| injected executed `T_CE` [min, max] Nm | [21.2, 81.5] | [21.2, 81.4] | [21.3, 81.4] |
| **injected executed `T_CE` mean** | **≈ 38.0 Nm** | ≈ 37.9 | ≈ 37.8 |

The mean injected engine load is **≈ 38 Nm** — only ~3 Nm above where the actor
already operates (~35 Nm) and **~25 Nm below the reward's arg-max (~63 Nm)**.
This is a **design limitation** analysed in §12: `[1.3·demand, 0.9·T_CE_max]`
collapses to mid-load at low demand.

---

## 4. FEASIBILITY VERIFICATION (pre-training dry-run, `stage_b/dry_run_checks.json`)

| check | result |
|---|---|
| `predict(deterministic=True)` does not trigger the intervention (500 calls, `te_stats` unchanged) | **PASS** |
| `predict` is stock `SAC.predict` (not overridden) | **PASS** |
| intervention occurs only during training (rollout `_sample_action` path) | **PASS** |
| every injected action ∈ `[-1, 1]` (4000-step dry rollout, 305 injections) | **PASS** (0 OOB) |
| every injected action produces engine-**ON** (`T_CE > T_CUTOFF`) | **PASS** (0 infeasible) |
| reachable physical action set unchanged (`Box(-1, 1, (1,))`) | **PASS** |
| `te_enabled=False` never injects (1000 steps) | **PASS** |
| injected action range / executed `T_CE` range in the dry-run | `a ∈ [−0.847, −0.548]`, `T_CE ∈ [21.3, 75.8]`, mean 37.3 |

**All B3 checks PASS.** Training proceeded.

---

## 5. THREE-SEED RESULTS (`stage_b/train_summary.json`)

All three runs are 150 000 steps, identical config, reported individually. No
seed was extended, retried, or retrospectively selected.

| seed | best-checkpoint V_CE_equiv | best-ckpt SoC_final | ΔSoC (pp) | injected transitions |
|---|---|---|---|---|
| 0 | **3.8360** | 50.79 % | +0.79 | 11 598 |
| 1 | **3.7486** | 49.93 % | −0.07 | 11 807 |
| 2 | **3.8308** | 52.26 % | +2.26 | 11 727 |
| **mean ± SD** | **3.8051 ± 0.0400** | — | — | — |
| **CONTROL (Phase 7/8)** | **3.7666 ± 0.0785** | — | — | — |

**Vehicle fuel did not improve — it is +0.038 L/100km worse than CONTROL, and
charge-sustaining fell 3/3 → 2/3** (seed 2 overcharges by +2.26 pp, driven by
the LPS/charging injections raising SoC).

---

## 6. REPLAY COVERAGE ANALYSIS (`diagnostics.json → B5`)

Fraction of replay transitions whose **executed engine torque** falls in each
`T_CE` bucket, by demand band (3 CONTROL buffers vs 3 12B buffers):

### `frac(executed T_CE ≥ 50 Nm)` — the deep-LPS region the reward prefers

| demand band | CONTROL (3 seeds) | **12B (3 seeds)** | change |
|---|---|---|---|
| 15–25 Nm | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | — (58 Nm unreachable here) |
| **25–30 Nm** | 0.038 / 0.078 / 0.125 (μ≈0.080) | **0.032 / 0.020 / 0.086 (μ≈0.046)** | **↓ (lower!)** |
| **30–35 Nm** | 0.077 / 0.143 / 0.121 (μ≈0.114) | **0.128 / 0.121 / 0.098 (μ≈0.116)** | **≈ unchanged** |
| 35–50 Nm | 0.314 / 0.370 / 0.458 (μ≈0.380) | 0.455 / 0.449 / 0.509 (μ≈0.471) | ↑ (+0.09, already best-covered) |

### Bucket detail (mean over seeds)

| band | bucket | CONTROL | 12B | Δ |
|---|---|---|---|---|
| 25–30 | **40–50 Nm** | 0.29 | **0.35** | **+0.06** |
| 25–30 | 50–60 Nm | 0.08 | 0.05 | −0.03 |
| 25–30 | 60–75 Nm | **0.00** | **0.00** | 0 |
| 30–35 | **40–50 Nm** | 0.25 | **0.33** | **+0.08** |
| 30–35 | 50–60 Nm | 0.11 | 0.11 | 0 |
| 30–35 | 60–75 Nm | ≈0.005 | ≈0.0016 | ≈0 |
| 35–50 | 50–60 Nm | 0.16 | **0.27** | **+0.11** |
| 35–50 | 60–75 Nm | 0.19 | 0.19 | 0 |

### Coverage evolution (`coverage_evolution.json`, seed 0)

Injections accumulate linearly (≈2000 / 25k steps); mean injected `T_CE` is
**stable at ≈38 Nm** throughout (step 2.4k → 148.8k: 37.6 → 38.0 Nm).

**Finding:** the intervention **raised coverage of the *moderate* engine-load
region (40–50 Nm) in every band, and the 50–60 Nm region only in the
already-well-covered 35–50 Nm demand band.** It added **essentially zero**
coverage in the **60–75 Nm bucket** (where the reward's arg-max `T_CE ≈ 63 Nm`
lives) in any band — that bucket stays at ~0 % in the critical 25–35 Nm bands.
In the 25–30 Nm band the `T_CE ≥ 50 Nm` fraction actually **fell**. **The
informative coverage of the region the reward actually prefers did not
materially improve.**

---

## 7. Q-SURFACE / Q-ARGMAX ANALYSIS (`diagnostics.json → B6`)

Fixed matched-state set (20 CONTROL-rollout states, bands 15–25 / 25–30 /
30–35 / 35–50 Nm; identical to Phase-11 §11CDE). CORE = 25–50 Nm demand.
`tce_R` = reward arg-max `T_CE` (N-independent) = **63.22 Nm**.

| run / checkpoint | critic arg-max `T_CE` (`tce_Q`) | actor `T_CE` (`tce_pi`) | `dT(R−Q)` | resid @ `a_R*` |
|---|---|---|---|---|
| CONTROL s0 | 24.58 | 36.87 | 38.66 | −0.0179 |
| CONTROL s1 | 32.52 | 35.09 | 30.69 | −0.0040 |
| CONTROL s2 | 58.83 | 42.47 | 4.42 | −0.0133 |
| **12B s0** 50k → 100k → 150k → best | **43.7 → 49.9 → 37.5 → 22.4** | 25.0 → 34.7 → 37.5 → 23.8 | 40.8 (best) | −0.024 (best) |
| **12B s1** 50k → 100k → 150k → best | **6.4 → 50.8 → 34.1 → 44.8** | 22.4 → 34.0 → 44.6 → 48.0 | 18.5 (best) | −0.015 (best) |
| **12B s2** 50k → 100k → 150k → best | **40.0 → 20.3 → 35.1 → 46.9** | 26.7 → 36.1 → 27.0 → 34.3 | 16.3 (best) | −0.027 (best) |

**The critic did NOT learn the high-load preference:**
* `tce_Q` **oscillates 6 → 51 Nm across checkpoints on every seed** — the same
  bistable, non-converging behaviour Phase-11 EXP-P11-S1 found for offline
  refinement. It does **not** move toward the reward's 63 Nm.
* 12B best `dT(R−Q)` mean ≈ **25 Nm** — **unchanged** from CONTROL (≈24.6 Nm).
* actor `T_CE` ≈ 35 Nm — unchanged; still part-load.

---

## 8. BELLMAN RESIDUAL ANALYSIS (`diagnostics.json → B7`)

Predefined diagnostic threshold: `|resid @ a_R*| < 0.004` (**a target, not
proof of success**).

| | resid @ `a_R*` (CORE mean) | trajectory across 100k → 150k → best |
|---|---|---|
| CONTROL (3 seeds) | −0.018 / −0.004 / −0.013 (μ ≈ **−0.012**) | — |
| **12B best (3 seeds)** | −0.024 / −0.015 / −0.027 (μ ≈ **−0.022**) | s0: +0.002 → +0.002 → −0.024 ; s1: +0.006 → −0.027 → −0.015 ; s2: −0.042 → −0.020 → −0.027 |
| `tce_Q` std over last 3 checkpoints | — | seed 0 ≈ 11, seed 1 ≈ 8, seed 2 ≈ 10 Nm — **not stable** |

**The deep-LPS Bellman residual did not close** — 12B best is if anything
**more negative than CONTROL** (−0.022 vs −0.012), never reaches the 0.004
threshold on any seed, and oscillates checkpoint-to-checkpoint. Q-arg-max is
**not stable** over the final ~50–100k steps. Mean `|min-Q|` ≈ 0.16 (comparable
to CONTROL). This reproduces Phase-11 EXP-P11-S1's finding on new online data.

---

## 9. DUAL-REWARD AUDIT (`diagnostics.json → B8`)

Each 12B best checkpoint evaluated deterministically through **both**
`EMSEnv(clip_eq_eff=True)` (`R_corrected`) and `EMSEnv(clip_eq_eff=False)`
(`R_original`, reconstructed offline) on the identical trajectory.

| seed | SoC_max (eval) | % transitions above 60.868 % | cum `R_corrected` | cum `R_original` | **ΔR** | n affected |
|---|---|---|---|---|---|---|
| 0 | 50.79 % | 0.00 % | −75.304 | −75.304 | **0.000000** | **0** |
| 1 | 49.93 % | 0.00 % | −47.223 | −47.223 | **0.000000** | **0** |
| 2 | 52.26 % | 0.00 % | −53.937 | −53.937 | **0.000000** | **0** |

**The Stage-A correction had exactly zero effect on the evaluated
controllers** — SoC_max ≤ 52.3 %, no transition in the affected region,
`ΔR = 0.0` to the bit. **Case E (contamination) is ruled out.**

**During training**, however, the correction *was* exercised
(`stage_b/*/replay_buffer.pkl` scan): seed 0 / 1 / 2 had **759 / 1204 / 1901**
replay transitions with SoC > 60.868 % (0.51 % / 0.80 % / 1.27 %; max SoC
66.7 % / 72.6 % / 74.2 %). Without `clip_eq_eff=True` those transitions would
have fed the critic an inverted `eq_eff` down to ≈ −0.6, paying the agent to
discharge — so the Stage-A correction was **necessary and load-bearing during
training** (the LPS/charging injections push SoC up), and it neutralised that
defect cleanly. Its invisibility at evaluation is exactly the Stage-A
prediction for a well-behaved policy.

---

## 10. VEHICLE-LEVEL EVALUATION (`diagnostics.json → B9`; `results/evaluate_policy.py`, unchanged protocol)

| run | V_CE_equiv | ΔSoC pp | CS | violations | OFF % | ASSIST % | LPS % | engine-on s |
|---|---|---|---|---|---|---|---|---|
| CONTROL s0 | 3.6862 | +0.28 | ✓ | 0 | 39.8 | 15.3 | 27.9 | 376 |
| CONTROL s1 | 3.8431 | −0.72 | ✓ | 0 | 38.2 | 12.0 | 32.9 | 390 |
| CONTROL s2 | 3.7704 | +0.23 | ✓ | 0 | 37.8 | 16.6 | 28.6 | 393 |
| **12B seed 0** | **3.8360** | +0.79 | ✓ | 0 | 36.9 | 13.8 | 32.3 | 401 |
| **12B seed 1** | **3.7486** | −0.07 | ✓ | 0 | 38.2 | 13.9 | 30.9 | 390 |
| **12B seed 2** | **3.8308** | **+2.26** | **✗** | 0 | 38.2 | 12.4 | 32.4 | 390 |
| **12B mean ± SD** | **3.8051 ± 0.0400** | — | **2/3** | 0 | ≈37.8 | ≈13.4 | ≈31.9 | ≈394 |
| advanced rule-based | 3.5056 | — | — | — | 59.0 | 0.0 | 23.8 | 209 |
| ECMS | 3.1887 | — | — | — | 53.1 | 0.2 | 29.7 | — |

**No vehicle-level benefit.** 12B mean fuel is +0.038 L/100km **worse** than
CONTROL (within the CONTROL seed SD, so "no material change"), charge-sustaining
degrades 3/3 → 2/3, and the mode split / engine-on time are essentially
unchanged (LPS +4 pp, consistent with the injected LPS actions). Still far from
rule-based (+8.5 %) and ECMS (+19.3 %).

---

## 11. SEED-BY-SEED RESULTS

| metric | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| best V_CE_equiv | 3.8360 | 3.7486 | 3.8308 |
| ΔSoC / CS | +0.79 pp / ✓ | −0.07 pp / ✓ | +2.26 pp / **✗** |
| injected transitions | 11 598 | 11 807 | 11 727 |
| replay SoC > threshold (training) | 759 (0.51 %) | 1 204 (0.80 %) | 1 901 (1.27 %) |
| `frac(T_CE ≥ 50)` 25–30 / 30–35 Nm | 0.032 / 0.128 | 0.020 / 0.121 | 0.086 / 0.098 |
| critic arg-max `T_CE` (best) | 22.4 | 44.8 | 46.9 |
| resid @ `a_R*` (best) | −0.024 | −0.015 | −0.027 |
| eval ΔR (corrected − original) | 0.000000 | 0.000000 | 0.000000 |

Seed 2 is the only non-CS run and also the one whose training SoC ran highest
(74.2 %) — the deep-LPS injections raised SoC and, on that seed, broke charge
balance. No seed shows any critic-argmax or Bellman-residual improvement.

---

## 12. SCIENTIFIC CLASSIFICATION

**Primary: CASE D — coverage did not materially improve in the target region;
the intervention failed to test the intended hypothesis adequately.**

* `frac(T_CE ≥ 50 Nm)` in the critical 25–35 Nm demand bands did **not** rise
  (25–30: 0.080 → 0.046; 30–35: 0.114 → 0.116). The **60–75 Nm bucket**, which
  contains the reward's arg-max `T_CE ≈ 63 Nm`, stayed at ≈ 0 % coverage in
  those bands.
* **Why:** the injection interval `[1.3 · demand, 0.9 · T_CE_max_feasible]`
  collapses toward mid-load at low demand — at demand 25–35 Nm it spans roughly
  `[33–46, 43–53] Nm`, so uniform draws average ≈ 38 Nm (measured mean
  injected `T_CE` = 38.0 Nm, stable across training). The intervention injected
  **"slightly harder than the actor" (~38 Nm)**, not **"reward-optimal deep-LPS"
  (~55–65 Nm)**. It tested the wrong sub-region.

**Secondary: CASE C — where coverage *did* rise, the critic did not learn the
high-load preference.** Coverage of the 40–50 Nm bucket rose +6–8 pp in every
band and the 50–60 Nm bucket +11 pp in the 35–50 Nm band (~11.7k injected
mid/high-load transitions per seed). Despite this, the critic arg-max `T_CE`
stayed **bistable / non-converging** (6 → 51 Nm across checkpoints), `dT(R−Q)`
was unchanged (~25 Nm), and the deep-LPS Bellman residual stayed ≈ −0.02
(never < 0.004, oscillating). This **reproduces Phase-11 EXP-P11-S1** (offline
critic refinement) on fresh online data: the deep-LPS value is not identifiable
and the near-degenerate bimodal Q is not resolved by adding data at these load
levels. This points toward **critic function-approximation / policy
parameterisation**, not coverage, as the deeper limiter.

**CASE A — NOT supported.** No stabilisation of critic Q toward the high-load
region; no evidence critic learning was coverage-limited (adding data did not
help).

**CASE B — partial fit only for the vehicle question:** the intervention did
change replay coverage of the *mid*-load region and left vehicle fuel
statistically unchanged (slightly worse) — but because it did **not** cover the
reward-relevant deep-LPS region (Case D), this is not a clean "coverage worked
but wasn't the bottleneck" result.

**CASE E — RULED OUT.** Dual-reward audit: `ΔR = 0.0` exactly on all 3
evaluated controllers; 0 affected transitions. The Stage-A correction did not
contaminate the experiment (it was a training-time safety net that activated in
0.5–1.3 % of replay and neutralised the `eq_eff < 0` defect as designed).

---

## 13. LIMITATIONS

1. **The injection interval is mis-scaled at low demand** (Case D root cause):
   `[1.3·demand, 0.9·T_CE_max]` never reaches the reward's ~63 Nm arg-max in the
   25–35 Nm band. The experiment therefore does **not** falsify the coverage
   hypothesis for the *reward-relevant* deep-LPS region — only for the
   *mid-load* region.
2. **150k-step budget.** Consistent with every prior CONTROL comparison, but a
   longer run could change the critic-stability picture (Phase-11 flagged this
   as untested on the stable config).
3. **Single cycle (NEDC).** FTP75 not run.
4. **Seed 2 lost charge-sustaining** — the deep-LPS/charging injections raise
   SoC; the `soc < 0.70` activation cap was not tight enough to prevent an
   overcharge on one seed.
5. **The critic-diagnostic state set is 20 states** (5 per band), matched to
   Phase-11 for comparability; small-sample noise is possible in the per-band
   means, though the bistability signal is unambiguous across checkpoints.
6. Coverage is measured as *executed* `T_CE` bucket occupancy with a
   band-and-torque criterion; it is a proxy for "informative" coverage (whether
   the transitions actually constrain the deep-LPS Bellman target), which
   Phase-11 §11CDE argued is thinner than raw occupancy because the frozen /
   slowly-moving actor reverts to part-load at every next-state.

---

## 14. EXACT NEXT-STEP RECOMMENDATION

**Do not run another training experiment yet.** Per Case D, first fix the
intervention so it actually tests the hypothesis, as a **single-variable**
re-run:

**Proposed EXP-12B2 (single variable = the injected `T_CE` interval; NOT executed):**
* Redefine the injection interval directly in the **reward-relevant deep-LPS
  band**: `T_CE ∈ [T_CE_max_feasible(state) − 15 Nm, T_CE_max_feasible(state)]`
  — i.e. the deepest-15-Nm window the U_MIN clamp allows (at demand 30 Nm ≈
  [45, 60]; at demand 40 Nm ≈ [63, 78]) — which is where the reward's arg-max
  (~55–65 Nm) actually sits.
* Tighten the SoC activation cap to `soc < 0.55` (below the SoC-deadband edge)
  to prevent the seed-2 overcharge; keep `clip_eq_eff=True` as the safety net.
* Everything else **frozen** at the §2 config. `p = 0.25`. 3 seeds × 150k.
* **Primary metric:** `frac(executed T_CE ∈ [55, 75] Nm)` in the 25–35 Nm
  demand bands (target: 0.00 → ≥ 0.30) **and** critic arg-max `T_CE` +
  its stability over the last 50k steps **and** the deep-LPS Bellman residual.
* **Success (coverage hypothesis supported):** the 55–75 Nm bucket coverage
  reaches ≥ 0.30 in 25–35 Nm on ≥ 2/3 seeds **and** the critic arg-max `T_CE`
  rises ≥ +10 Nm toward the reward and stabilises (`tce_Q` std over last 50k
  < 5 Nm) **and** the deep-LPS residual falls below 0.004 — with vehicle CS
  held 3/3.
* **Failure (coverage hypothesis falsified → critic/policy-structure):** the
  55–75 Nm bucket reaches ≥ 0.30 but the critic arg-max stays bistable /
  < +3 Nm and the residual stays worse than −0.008 on ≥ 2/3 seeds — then
  **both** offline refinement (EXP-P11-S1) **and** online reward-relevant
  coverage have failed to move the critic ⇒ the near-degenerate bimodal Q is
  **intrinsic**, and the next lever is **policy parameterisation** (discrete
  engine-mode head + continuous within-mode) or a **reward-shape** change — not
  more coverage.

---

## CONCLUSION — PASS / FAIL / INCONCLUSIVE

**Stage B: INCONCLUSIVE for the coverage hypothesis (Case D), with a Case-C
signal.** The intervention was mechanically sound (11.7k feasible high-load
injections/seed, all dry-run checks pass, no contamination — Case E ruled out),
but its `[1.3·demand, 0.9·T_CE_max]` interval collapsed to **mid-load (~38 Nm)**
at low demand and **did not materially raise informative coverage of the
reward-relevant deep-LPS region (~55–65 Nm) in the critical 25–35 Nm demand
bands**. Where coverage *did* rise (mid-load, and 50–60 Nm in the 35–50 band),
the critic **did not** learn the high-load preference — Q-arg-max stayed
bistable and the deep-LPS Bellman residual stayed ≈ −0.02, reproducing
Phase-11 EXP-P11-S1. Vehicle fuel did not improve (3.805 ± 0.040 vs CONTROL
3.7666 ± 0.079; CS 2/3 vs 3/3). The Stage-A `clip_eq_eff` correction was
verified non-contaminating at evaluation (ΔR = 0.0, 3/3) and load-bearing
during training (0.5–1.3 % of replay).

---

## FINAL HARD STOP — SATISFIED

Stage B report and all machine-readable outputs/plots are under
`results/phase12/` (nothing overwritten). **No 500k/1M run, no γ change, no
action reparameterisation, no `eq_factor`/`k_fb` recalibration, no additional
experiment, no multi-hypothesis training run has been performed or will be.**
The single next experiment (EXP-12B2) is *recommended only*. Awaiting human
review and authorisation.
