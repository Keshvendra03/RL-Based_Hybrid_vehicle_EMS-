# RL Diagnostic Report — SAC-based HEV EMS

Forensic audit of the reinforcement-learning layer. Physics/plant/benchmarks
are validated and out of scope (see **Locked Components**).

- **Phase 1** (2026-08-26): forensic audit -> P0 = action geometry.
- **Phase 2** (2026-08-26): instrumentation + Q-landscape -> **P0 REJECTED,
  replaced by a reward unit-mismatch (P0-REVISED).**
- **Phase 4** (2026-08-26): exploration deadlock (OFF 3.9-6.7 sigma from the
  actor mean) -> gated action map. FTP75 reaches benchmark; NEDC SoC runaway.
- **Phase 5** (2026-08-26): costate gain k_fb identified. NEDC charge
  sustainability solved (1/3 -> 3/3); fuel tied. Proposed a bimodal-Q /
  unimodal-policy explanation.
- **Phase 5B** (2026-08-27): **forensic closure. The Phase-5 replay-starvation
  inference is REFUTED by direct measurement, and the bimodal-Q conclusion is
  demoted from primary cause to symptom.** See `PHASE5B_FORENSIC_CLOSURE.md`.

- **Phase 6** (2026-08-27): controlled A/B conditional-exploration experiment.
  **The Phase-5B conditional-coverage hypothesis is REFUTED.** Coverage at
  30-35 Nm / SoC 40-50 was raised 4.5% -> 36.7% (274 -> 1,333 transitions) and
  Q(OFF-ASSIST) did not move (-0.0071 -> -0.0066). Fuel worsened on both
  cycles. See `PHASE6_FINAL_REPORT.md`.

### CORRECTION TO THE PHASE-5B DIAGNOSIS (issued in Phase 6)

Phase 5B stated "the reward favours OFF and the critic disagrees" at 30-35 Nm.
**This was overstated.** Measured at 30-35 Nm: `dr(OFF-ASSIST) = +0.0000`,
positive in only **10%** of states, and **85-87% of states are
`[reward=ASSIST, Q=ASSIST]` -- reward and critic AGREE**. The Phase-5B claim
generalised the 15-30 Nm aggregate (+0.0011) into a region where it does not
hold. There was no conflict at 30-35 Nm to repair.

### CURRENT PRIMARY BOTTLENECK (Phase 6, measured)

**Actor-side displacement at 15-30 Nm.**

  * 15-30 Nm is the largest remaining error term (+0.3677 L/100km), unchanged
    by every intervention in Phases 4-6.
  * There the reward unambiguously favours OFF (`dr>0` in **100%** of states).
  * The critic is partially correctable there: raising coverage cut the
    `[r=OFF, Q=ASSIST]` conflict from **86% -> 66%**.
  * But the actor moved the WRONG way while the critic improved: P(OFF)
    **68.9% -> 50.7%**, actor-Q displacement **0.066 -> 0.270**.
  * Alignment predicts performance: FTP75 (the only benchmark-level result) is
    the most aligned config measured (75.8% aligned / 11.7% displaced).

### SUPERSEDED (kept for the record)

### CURRENT PRIMARY BOTTLENECK (Phase 5B, measured)

**Critic misestimation of Q(OFF) at the operating SoC, caused by a
CONDITIONAL replay-coverage hole.**

  * true reward favours OFF: `dr(OFF-ASSIST) = +0.0011`
  * critic disagrees:        `dQ(OFF-ASSIST) = -0.0062 .. -0.0222`
  * cause: at 30-35 Nm / SoC 40-50 (the operating point) only **4.5%
    (~276 of 6,132)** replay transitions contain OFF; the OFF data that exists
    sits at SoC<40, visited during the early runaway phase.
  * critic is NOT flat (`D_flat = 0.0%`) -> capacity is not the limit.
  * actor displaced from its own critic in **37.5%** of states; k_fb=2.5
    DOUBLED mean displacement (0.149 -> 0.295).
  * costate median = **2.79 ECMS units vs proven lambda_0 = 1.3125** ->
    battery systematically over-priced, independently depressing Q(OFF).

Companion files: `EXPERIMENT_LOG.md` (per-experiment record),
`experiments/experiment_registry.yaml` (machine-readable),
`ROADMAP.md` (plan), `VERIFIED_FACTS.md` (proven facts),
`VALIDATION.md` (plant validation).

---

## 0. LOCKED COMPONENTS — DO NOT MODIFY

> **No validated plant/environment physics may be modified during the RL
> optimization experiments unless a separate validation failure is
> demonstrated.** A suspected defect in a locked component must be reported
> as a `VALIDATION CONFLICT`, never silently changed.

| File / component | Why locked |
|---|---|
| `src/env/powertrain.py` | MATLAB/Simulink-validated, 9 checks (`VALIDATION.md`) |
| `src/env/driving_cycle.py` | Sample-count, `dv`, `x_tot` conventions verified vs MATLAB |
| `src/baselines/rule_based.py` | Validated baseline controller |
| `src/baselines/advanced_rule_based.py` | Benchmark, reproduces 3.5056 / 3.2323 |
| `src/baselines/ecms.py` | Benchmark, reproduces 3.1887 / 2.8097 charge-sustaining |
| Env↔plant wiring in `ems_env.py` | `test_ems_env.py` proves 1e-9 agreement with `evaluate_advanced.py` |
| Feasibility masks (motor/engine envelope, SoC limits) | Verified: 0 violations in RL rollouts |

**Modifiable (RL layer only):** reward shaping parameters, action→u
*coordinate* mapping, observation preprocessing, SAC hyperparameters,
training loop, evaluation/instrumentation.

---

## 1. Executive Diagnosis (Phase 2, current)

```
P0-REVISED  CRITICAL — REWARD-DESIGN / UNIT MISMATCH.
            The reward's implicit battery price is 4.83 fuel-J per battery-J
            at eq_factor=1.0, because `elec_liters` is ALREADY EFC-converted
            (it carries 1/(eta_BT*eta_EM*eta_CE)). ECMS's proven optimal
            costate is lambda=1.3125 (NEDC). The env therefore prices battery
            energy 3.68x too high at eq_factor=1.0, and 4.83x too high at the
            eq_factor=1.3125 used in every experiment this session.
            MEASURED CONSEQUENCE: over 160 states, the reward-optimal action
            agrees with the ECMS-optimal action in 7.5% of states and selects
            engine-OFF in 0.0% of states, versus ECMS selecting OFF in 90.0%.
            Unit-matched (eq_factor=0.2717): agreement 86.9%, OFF 78.1%.

P0-OLD      REJECTED — ACTION GEOMETRY was NOT the primary cause.
            (Retained below for the record; the geometry measurement is still
            correct, it is simply not what is limiting performance.)

P1          MAJOR — BENCHMARK-FAIRNESS. The advanced rule-based benchmark
            drives SoC to 0.61% for 38/1220 steps on NEDC. The RL agent is
            hard-masked at SOC_MIN=0.05 and can never access that range.
            Unequal control authority; part of the measured gap is structural.

P1          MAJOR — TRAINING-PROCEDURE. Best checkpoint always occurs in the
            first half of training, then degrades (NEDC seed0 frozen at
            65,880/497,760; FTP75 peaked 296,408 then collapsed to 4.946 /
            SoC 32.2% by ~1M steps). "Train longer" is refuted, 2 seeds.

P1          MAJOR — REWARD-DESIGN. SoC deadband penalty contributes EXACTLY
            0.0% (0/1220, 0/1876 steps active); terminal penalty 0.77%.
            Both mechanisms are effectively dead code.

P1          MAJOR — HYPERPARAMETER. gamma=0.9999 was justified in-code by the
            need to propagate the terminal SoC signal; that signal is 0.77% of
            the reward, so the justification is void.

P2          MODERATE — REWARD BUG. eq_eff = eq_factor + k_fb*(0.5-SoC) crosses
            zero and INVERTS at SoC 66.41% (NEDC) / 80.06% (FTP75), paying the
            agent to discharge. Not triggered at best checkpoints; NEDC ran at
            60-66% SoC during training.

P3          VERIFIED / NO PROBLEM — SAC implementation is mathematically and
            programmatically correct (line-by-line vs SB3 2.9.0).
            Custom NStepSAC is formula-identical to SB3 native.

P3          MODERATE — STATE-DESIGN. obs[7] (v_next) is byte-identical to
            obs[8] (fut_v1); obs[19] (gear_oh6) is always 0.0. 2 wasted inputs.

P3          GENERALIZATION NOT DEMONSTRATED. Cross-cycle evaluation never run.
```

---

## 2. P0-REVISED — the reward unit mismatch (full evidence)

### OBSERVATION
The trained agent uses engine-OFF on 29.4% of moving steps (NEDC) where both
validated benchmarks use 53.1-59.0%, parking instead in ASSIST (20.6% vs
0.0-0.2%). This "ASSIST blob" survived every hyperparameter intervention.

### EVIDENCE
**(a) Q-landscape sweep** (`results/q_landscape.py`, baseline checkpoint,
41-point action sweep at matched physical state via env deep-copy):

| probe | T_MGB | best-OFF − best-ASSIST, **true one-step reward** | same, **min-Q** | verdict |
|---|---|---|---|---|
| low_torque | 8.69 Nm | **+0.0101** | +0.135 | CRITIC AGREES |
| med_torque | 22.2 Nm | **−0.0255** | −0.273 | CRITIC AGREES |
| high_torque | 48.93 Nm | **−0.0825** | −0.533 | CRITIC AGREES |

The critic tracks the reward faithfully in every probe. **The critic is not
the problem — the reward itself ranks ASSIST/LPS above OFF at medium and high
torque.**

**(b) Unit analysis:**
```
1 J fuel    -> K_FUEL_L_PER_KG / H_u = 3.177172e-08 L
1 J battery -> K_ELEC_L_PER_J        = 1.534866e-07 L
implicit lambda at eq_factor=1.0     = 4.8309  fuel-J per battery-J
ECMS proven optimal lambda_0 (NEDC)  = 1.3125
ECMS proven optimal lambda_0 (FTP75) = 2.4062
```
`elec_liters` already contains `EFC_GAIN = 1/(eta_BT*eta_EM*eta_CE*(H_u/3.6e6)*rho_f)`.
Multiplying it again by an "ECMS lambda" double-counts the conversion.

**(c) Behavioural test** — 160 states, reward-optimal action (81-point sweep,
env deep-copy) vs ECMS Hamiltonian-optimal action at identical state/SoC:

| eq_factor | agrees with ECMS | reward picks OFF | ECMS picks OFF |
|---|---|---|---|
| **1.3125** (used in all session experiments) | **7.5%** | **0.0%** | 90.0% |
| **0.2717** (unit-matched) | **86.9%** | **78.1%** | 90.0% |

### ROOT CAUSE
`eq_factor` is applied to an already-EFC-converted quantity, so its numerical
scale is **not** the ECMS costate scale. Correct conversion:
```
eq_factor_env = lambda_ECMS / 4.8309      k_fb_env = k_fb_ECMS / 4.8309
NEDC : 1.3125 / 4.8309 = 0.2717           8.0 / 4.8309 = 1.656
FTP75: 2.4062 / 4.8309 = 0.4981
```

### IMPACT ON EMS
Battery discharge is priced ~5x above its true equivalent cost. Engine-OFF
*requires* discharge, so it is systematically rejected; LPS (charging) is
symmetrically over-rewarded, explaining LPS 33.0% vs the benchmarks' 23.8%.
The agent has been **correctly optimizing a misspecified objective**.

### WHY EARLIER RESULTS NOW MAKE SENSE
- Flat `eq_factor=1.0` baseline: already 3.68x overpriced → OFF 10.5%.
- Raising to 1.3125 "to match ECMS" made pricing **worse** (4.83x) — which is
  exactly why that intervention was measured as a regression.
- `k_fb=8.0` copied from ECMS is likewise 4.83x too strong.

### NOTE ON METRIC INTEGRITY
This changes only the **training signal**. `v_ce_equiv` is computed
independently by the validated EFC block and is untouched, so evaluation
remains honest. The `eq_factor=1.0` "reward telescopes to −v_ce_equiv"
property is an *accounting* identity; the *optimal costate* for the
charge-sustaining constrained problem is a different quantity (Pontryagin),
and ECMS has already proven what it is.

---

## 3. P0-OLD — action geometry (measurement retained, hypothesis rejected)

Engine-OFF requires `(1-u)*T_MGB <= T_CUTOFF`, i.e. `u >= 1 - T_CUTOFF/T_MGB`.
Under the linear map that region is narrow and state-dependent:

| | NEDC | FTP75 |
|---|---|---|
| OFF band, median % of action range | 12.19% | 9.29% |
| steps with OFF band < 10% | 47.8% | 53.8% |
| `a_off` spread (p25→p75) | +0.396 → +0.857 | +0.651 → +0.891 |

**Candidate mapping analysis** (`ACTION_MAPS` in `ems_env.py`):

| mapping | OFF med% | p25 | p75 | LPS med% | verdict |
|---|---|---|---|---|---|
| ORIGINAL linear (p=1) | 12.19 | 7.13 | 30.22 | 45.95 | baseline |
| power p=1.5 | 8.30 | 4.81 | 21.33 | 59.54 | **REJECTED — narrows OFF** |
| power p=2.0 | 6.29 | 3.63 | 16.46 | 67.78 | **REJECTED — narrows OFF** |
| power p=3.0 | 4.24 | 2.43 | 11.30 | 77.16 | **REJECTED — narrows OFF** |
| power p=0.5 | 22.89 | 13.75 | 51.31 | 21.11 | rejected — LPS shrinking |
| power p=0.25 | 40.55 | 25.60 | 76.29 | **4.46** | **REJECTED — LPS collapses** |
| piecewise fixed 35/25/40 | 40.00 | 23.39 | 50.77 | 35.00 | good, still state-varying |
| **analytic (mode-aware) 35/25/40** | **40.00** | **40.00** | **40.00** | **35.00** | **SELECTED** |

> The power values suggested a priori (p=1.5/2/3) move the band in the **wrong
> direction** — testing rather than assuming was necessary.

**Selected: mode-aware.** OFF is exactly 40% of the action range at *every*
timestep (zero state variance), converting the moving boundary into a constant
`a_off = +0.20`. LPS keeps 35% (benchmarks need 22-30%).

**Control equivalence proved** — `tests/test_action_mapping.py`, 53 tests:
exact endpoints `u(-1)=U_MIN`, `u(+1)=U_MAX` for every T; strict monotonicity
(bijection); identical reachable `u` set; regen/sub-cutoff falls back to linear.
Only the *density* of the action coordinate changes. Plant untouched.

**Status:** implemented behind `--action-map modeaware` (default `linear` is
bit-identical to the original). Experiment B tests it in isolation; given
P0-REVISED, a large gain is no longer expected — it is now a **falsification
control**.

---

## 4. Benchmark-fairness conflict (P1)

**VALIDATION CONFLICT — reported, not silently changed.**

Running the advanced rule-based controller through the validated plant
(no env masking, exactly as `evaluate_advanced.py` does):
```
rule-based SoC min = 0.6147%   steps below 5% SoC: 38 / 1220   final 52.47%
```
The RL agent is hard-masked at `SOC_MIN=0.05` and can never enter that region.
The benchmark therefore has **strictly greater control authority** — it can
transiently deep-discharge to sustain pure-EV operation, which is precisely
the mode where it beats the agent (OFF 59.0% vs 29.4%).

The benchmark is not cheating on the energy ledger (it recovers to 52.47%, and
`v_ce_equiv` charges for net battery use), but the comparison is not authority-
equal. **Recommended controlled test:** re-run the rule-based controller *with*
the env's SoC masks and re-measure its fuel. Do not change `SOC_MIN` for the
agent without that measurement.

---

## 5. Reward audit (measured contributions)

```
eq_eff(t) = eq_factor + k_fb*(0.5 - SoC_t)
r_t = -100*[ dm_fuel*(k_cs/rho_f) + eq_eff(t)*dE_batt*(EFC_GAIN/3.6e6) ]
      - 2.0*max(|SoC-0.5|-0.10, 0)^2 - 1.0*max(|SoC-0.5|-0.10, 0)
terminal: -100*max(0, E_init-E_final)*K_ELEC - 50*e - 800*e^2,  e=max(|SoC_f-0.5|-0.02,0)
```

| Component | NEDC | FTP75 |
|---|---|---|
| Fuel term | 65.9% | 44.0% |
| Battery term | 34.1% | 56.0% |
| SoC deadband penalty | **0.0%** (0/1220 active) | **0.0%** (0/1876 active) |
| Terminal penalty | −0.348 of −45.15 = **0.77%** | **0.00%** |

Also measured, contradicting the in-code n-step justification: the OFF↔ASSIST
per-step reward gap is **0.052** (OFF −0.0244, ASSIST −0.0766) ≈ **2/3 of the
reward's own magnitude** — *not* "a tiny per-step fuel amount". The signal was
always strong; it simply pointed the wrong way (§2).

---

## 6. SAC implementation audit — **VERIFIED CORRECT**

Audited against live SB3 2.9.0 source. Actor arch `20→256→256→(mu, log_std)`;
`SquashedDiagGaussianDistribution` (tanh + log-prob correction) unmodified;
action scaling happens **inside the env** after log-prob, so no inconsistency;
twin-Q with `min(Q1,Q2)`; correct target nets + Polyak `tau=0.005`; correct
entropy sign; correct terminal masking; correct detachment. Custom
`NStepSAC.train()` is formula-identical to SB3's (only difference — target-
update counter — is inert at `target_update_interval=1`).
**Classification: NO PROBLEM / VERIFIED.**

---

## 7. Benchmark comparison (identical conditions, one evaluator)

All rows produced by `results/evaluate_policy.py`.

### NEDC
| Metric | SAC (best, seed1) | ECMS | Adv. rule-based |
|---|---|---|---|
| **V_CE_equiv** | **4.1245** | **3.1887** | **3.5056** |
| SoC final / ΔSoC | 52.63% / +2.63pp | 50.36% / +0.36pp | 52.47% / +2.47pp |
| SoC min | 44.16% | — | **0.61%** |
| OFF / ASSIST / LPS / ONLY / REGEN | 29.4/20.6/33.0/0.0/17.0 | 53.1/0.2/29.7/0.0/17.0 | 59.0/0.0/23.8/0.2/17.0 |
| Constraint violations | 0 | 0 | 38 (below SOC_MIN) |
| Engine on-time | 466 s | — | 209 s |
| **SAC vs** | — | **+29.3% worse** | **+17.7% worse** |

### FTP75
| Metric | SAC (best) | ECMS | Adv. rule-based |
|---|---|---|---|
| **V_CE_equiv** | **4.2072** | **2.8097** | **3.2323** |
| SoC final / ΔSoC | 48.56% / −1.44pp | 50.13% / +0.13pp | 53.86% / +3.86pp |
| OFF / ASSIST / LPS / REGEN | 17.4/34.3/22.7/25.7 | 40.4/6.0/27.9/25.7 | 46.3/0.4/22.4/25.7 |
| **SAC vs** | — | **+49.7% worse** | **+30.2% worse** |

**SoC fairness: the comparison is VALID** on the energy ledger — SAC's ΔSoC
matches or is worse than the benchmarks'. **There is no SoC exploitation and no
hidden win.** (Authority asymmetry is a separate issue, §4.)

---

## 8. Root-cause conclusion

```
P0-OLD (action geometry)          : REJECTED as primary cause
P0-REVISED (reward unit mismatch) : STRONGLY SUPPORTED, pending Experiment B2
```

The measurement behind P0-OLD is correct and the mode-aware mapping is a
sound, control-equivalent improvement — but it cannot be the primary cause,
because at the current pricing the reward-optimal action is **never** OFF
(0/160 states) regardless of how much action-space resolution OFF is given.
Widening a door the reward tells the agent not to walk through cannot help.

---

## 9. Verdict on algorithm choice

**SAC: APPROPRIATE WITH MODIFICATIONS.** The implementation is correct, the
policy has not collapsed (action std 0.64, spans full range, 1.6% saturation),
constraints are respected, and the critic faithfully learns the reward it is
given. No evidence supports replacing SAC *before* the reward is corrected.
TD3 remains the designated secondary experiment, to be run only if SAC still
trails the rule-based benchmark after P0-REVISED is fixed and entropy/gamma
are tuned. **Do not jump to PPO.**
