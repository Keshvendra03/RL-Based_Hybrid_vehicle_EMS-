# PHASE 11 — STAGE 0 FINAL DECISION GATE

**V1 (costate verification) + V2 (ECMS sensitivity) complete. NO TRAINING. NO
RL CHANGE. NO CALIBRATION APPLIED. DP not written (none exists).**

Inputs: `results/phase11/STAGE0_AUDIT.md`, `results/phase11/V1/report.md`,
`results/phase11/V2/report.md`, `results/phase11/data/*.json`,
`results/phase11/V1_V2/manifest.json`.
Git HEAD `90af969` (one docs-only commit above the `f1f45c5` Stage 0 quoted; no
`src/` change). Env: SB3 2.8.0 / torch 2.12 CPU / numpy 2.4.6 / scipy 1.17.1 /
Python 3.13.2. ECMS headline reproduced **exactly** (3.1887 / 2.8097).

---

## A. WHAT IS MATHEMATICALLY PROVEN?

### PROVEN (from frozen source + exact reproduction)
1. **The SAC per-step reward is NOT ECMS's Hamiltonian.** Reduced to common
   variables it is `−[P_CE(u) + λ_SAC(SoC)·P_EM(u)]` with
   `λ_SAC(SoC) = eqf_eff(SoC)·C(SoC)·4.8309·2`, where
   `C(SoC) = dE_ledger/(P_EM·Δt) = (dE/dQ)/u_bt ≈ 0.55–0.61` (capacitor factor,
   from `E(Q)=½·U_oc(Q)·Q`) and the extra `×2` is the trapezoidal ½-marginal on
   the fuel term (`dm_fuel = 0.5·(P_CE+P_CE_prev)/H_u`; the other half lands in
   step *t+1* via `Tank.p_fuel_prev`, which is **not observable**).
2. **ECMS's λ₀ is a whole-cycle bisection calibration** (`tune_lambda` runs the
   full cycle repeatedly). It **is handed to SAC** as `eq_factor = λ₀/4.8309`.
3. **Constant-λ ECMS (k_fb = 0) cannot charge-sustain this plant** (V2-B:
   SoC → 22 % NEDC / 14 % FTP75). The SoC-feedback term is load-bearing.
4. **The ECMS headline is faithfully reproducible** (3.1887 / 2.8097, exact),
   physically fair (identical plant blocks, feasibility masks, evaluator).

### EMPIRICALLY SUPPORTED (this session's analysis-only measurements)
5. **On a common `P_EM` basis the SAC reward is a *stiffer-battery* Hamiltonian
   than ECMS.** `λ_SAC/λ_ECMS ≈ 1.17` at target SoC → **1.38–1.42** at
   operating SoC ~0.375 (NEDC); **1.17 → 1.30** (FTP75). *[analytical, V1-B]*
6. **The SAC reward's matched-state arg-min prefers more engine / less battery
   than ECMS's Hamiltonian** in **56–58 %** of probes overall and **66–90 %**
   in the 15–35 Nm bands; `SAC-prefers-more-battery = 0 %` in every band on both
   cycles; arg-min engine torque **+25 Nm (15–30) / +52 Nm (30–35)** above
   ECMS. *[decisive matched-state test, V1-D, 121 + 183 probes, 161-pt grid]*
7. **The trained actor runs the engine *softer* (~35 Nm at 30–35 Nm demand)
   than the SAC reward's own arg-min (~58 Nm)** — it undershoots its own
   objective by ~23 Nm. *[V1-D + Phase 7/9]*
8. **ECMS fuel is insensitive to λ₀** (spread 1.3 % NEDC / 1.8 % FTP75 over
   ±20 %); the CS constraint pins λ₀ to ±5–10 %; the whole-cycle bisection
   landed **0.004 L/100km above** the CS-feasible fuel minimum of its class.
   *[V2-A]*
9. **The 81-pt ECMS benchmark is mildly conservative:** 81 → 321 pts lowers
   ECMS fuel by **0.022 (NEDC, 0.7 %) / 0.011 (FTP75, 0.4 %)** — the true
   constant-λ optimum is ≈ **3.167 / 2.799**, so grid resolution *widens* the
   SAC target rather than shrinking it. *[V2-C]*
10. **CS tolerance (0.5 % vs 2 %) changes achievable ECMS fuel by < 0.001
    L/100km.** *[V2-D]*

### PLAUSIBLE (consistent with evidence, not isolated)
11. The reward's stiffer-than-ECMS instantaneous costate is a **secondary,
    same-direction contributor** to the CONTROL's excess engine-on time (376 vs
    260 steps NEDC) — direction matches, magnitude not isolated from the
    arg-min↔actor optimisation gap.
12. Grid-resolution effect suggests ~0.5–0.7 % of "the gap" is the analytic
    optimiser's pointwise precision that a smooth 150k-step approximator
    structurally cannot match (Phase 8 "E").

### UNKNOWN / REQUIRES A FUTURE TASK
13. **The true global optimum.** `DP_STATUS = DEFERRED_NO_EXISTING_VALIDATED_
    SOLVER`. 3.1887 / 2.8097 is **not** a proven physical lower bound.
14. The precise split of the +0.50 L/100km gap into
    "RL-addressable" vs "irreducible for any causal controller" — needs DP.
    V2 bounds the *benchmark-artefact* fraction to ≈ 1 % (λ₀ tuning) − 0.7 %
    (grid, negative) ≈ near-zero net; the rest is RL-side + structural.

---

## B. C1 / H3 VERDICT

> **FALSIFIED (as stated in Stage 0).**

Stage 0 claimed the SAC effective battery costate is *below* ECMS's on a common
`P_EM` basis, predicting SAC should use *more* battery. Both the analytical
correction (V1-B) and the decisive matched-state test (V1-D) show the
**opposite direction**.

**Quantified refined finding (retained):**
* Capacitor factor `C(SoC) ≈ 0.58` is **real and confirmed** — Stage 0 got that
  part right.
* Stage 0 **omitted** the trapezoidal ½-marginal on the fuel term, which
  weights battery 2× and reverses the sign.
* Net: `λ_SAC / λ_ECMS ≈ **1.17 (SoC 0.50) → 1.4 (SoC 0.30)**`.
* Matched-state arg-min: SAC prefers **+25 to +52 Nm more engine** than ECMS in
  66–90 % of critical-band states; **0 %** prefer more battery.
* Realised rollout exchange rate (diagnostic, confounded): CONTROL burns
  **2.4–4.9× more fuel per joule of battery discharged** than ECMS.

**Status of the mechanism:** the reward ≠ ECMS-Hamiltonian *mismatch is
CONFIRMED*; its *direction is stiffer-battery, not softer*; its *magnitude is a
secondary contributor* (~15–40 % over-pricing in the operating band), smaller
than the arg-min↔actor optimisation gap.

---

## C. BEHAVIOURAL RELEVANCE

**Does the mismatch actually explain the observed 15–35 Nm SAC failure?
Partially — as a same-direction contributor, NOT as the dominant cause. No
causal claim is made from correlation.**

* **Same direction:** the SAC reward's per-step arg-min is engine-heavier than
  ECMS (V1-D), and the CONTROL indeed runs the engine more (Phase 9). So the
  reward's instantaneous preference and the policy's behaviour agree — the
  reward mildly biases *toward* engine-ON, *against* the ECMS "OFF-or-hard"
  strategy.
* **Not the dominant cause, three reasons:**
  1. The reward's arg-min at 30–35 Nm is **~58 Nm** (engine ON, moderately
     loaded); the trained actor delivers **~35 Nm**. The **arg-min↔actor gap
     (~23 Nm) is larger** than the reward↔ECMS arg-min gap and is a
     bimodal-value / σ-collapse / thin-coverage optimisation failure
     (Phases 5–9), untouched by V1.
  2. The reward's arg-min is engine-ON-moderate where ECMS is **OFF**. A
     reward-optimal policy would run part-load, not reproduce ECMS.
  3. Phase 7 already measured trained-policy `P(OFF)` **flat** across
     `k_fb ∈ {1.656, 2.5, 3.0}` — the policy does not respond to per-step
     costate changes as the instantaneous math predicts, because it
     self-selects a low operating SoC where the total effective price
     re-converges toward ECMS's.

**Conclusion:** "C1 is mathematically nuanced and real" ✔ ; "C1 is the dominant
cause of the RL gap" ✘. The 15–35 Nm failure remains dominated by the
**exploration/optimisation topology** (actor cannot reach the hard-engine / OFF
operating points that its own critic's arg-max — and, for the hard-engine part,
its own *reward's* arg-min — prefer).

---

## D. ECMS BENCHMARK STATUS

| Property | Finding |
|---|---|
| **Current benchmark** | NEDC **3.1887**, FTP75 **2.8097** `v_ce_equiv` (λ₀ tuned, k_fb 8, grid 81) — reproduced exactly. |
| **Charge-sustaining** | Yes: ΔSoC +0.36 pp (NEDC) / +0.13 pp (FTP75). |
| **Sensitivity to λ₀** | Low. Fuel spread **1.3 % / 1.8 %** over λ₀ ± 20 %. CS window ≈ λ₀ ± 5 % (0.5 % tol) / ± 10 % (2 % tol). Bisection landed 0.004 above the class fuel-minimum. |
| **Effect of λ tuning (whole-cycle info)** | ≈ **1 % of the SAC gap (~0.005 L/100km)** — and **handed to SAC** as `eq_factor`. Not a material explanation for +0.50. |
| **Effect of grid resolution** | 81 → 321 pts: **−0.022 / −0.011 L/100km**. True constant-λ optimum ≈ **3.167 / 2.799**. Grid resolution *widens* the SAC target. |
| **CS constraint / tolerance** | k_fb feedback is *necessary* (k_fb 0 → SoC 22 %/14 %). CS tolerance 0.5 % vs 2 % changes ECMS fuel by < 0.001. |
| **Per-step preview** | ECMS has **none**. SAC has a 5-second causal speed preview in its observation. |
| **Structural (non-information) advantage** | Exact per-step Hamiltonian minimisation with a perfect plant model over an 81-pt grid — zero function-approximation error. Irreducible for a smooth learned controller; ~0.5–0.7 % scale per V2-C. |
| **Valid DP reference?** | **No.** `DP_STATUS = DEFERRED_NO_EXISTING_VALIDATED_SOLVER`. 3.1887 / 2.8097 is **not** a proven global optimum or physical lower bound. |
| **Overall** | **Fair, reproducible, robust, near-optimal for the constant-λ class, mildly conservative.** Essentially none of the +0.50 gap is a benchmark artefact. |

---

## E. UPDATED HYPOTHESIS RANKING (H1–H10)

Re-ranked on the V1/V2 evidence. **Not** carried over automatically.

| # | Hypothesis | Δ vs Stage 0 | Confidence | Rationale from V1/V2 |
|---|---|---|---|---|
| **H1** | **Exploration/discovery deadlock for efficient *high-engine-load* operating points** (distinct from the fixed OFF deadlock; ECMS_NBHD/HIGH_EFF replay support 8–27 % vs LOW 54 %). | **↑ reinforced** | **HIGH** | V1-D: the SAC reward's *own* arg-min at 30–35 Nm is a hard-engine point (~58 Nm) the actor never reaches (~35 Nm). The actor fails to track even its own objective's hard-engine preference → coverage/topology, not pricing. |
| **H2** | **Optimisation topology:** bimodal `r`/`Q`, σ-collapse (0.19); tanh-Gaussian local ascent cannot cross the valley to the hard-engine / OFF lobe. | **↑ co-primary with H1** | **HIGH** | V1-D quantifies the arg-min↔actor gap (~23 Nm at 30–35 Nm) and shows it is larger than the reward↔ECMS gap. This is now the single largest identified term. Intertwined with H1 (the deadlock *is* why the actor can't get there). |
| **H4** | **Short value horizon (γ = 0.20)** blocks anticipatory bank-then-spend; myopic critic cannot value "charge now, coast later". | **= unchanged** | **MED** | Unaffected by V1/V2. V1-D adds indirect support: the myopic reward's per-step arg-min is engine-heavy precisely because it cannot see the downstream fuel saved by banking; a longer horizon might let the critic value it. |
| **H5** | **State lacks timing features** (prev-action/mode, demand history, time/distance remaining); cross-cycle CS 0/3. | **= unchanged** | **MED** | Unaffected. Still the best explanation for cross-cycle CS failure. |
| **H3** | **Costate calibration** (reward ≠ ECMS Hamiltonian). | **↓ downgraded / partly resolved** | **LOW as a lever** | V1: mechanism CONFIRMED but direction is *stiffer-battery* (not softer), magnitude secondary, and Phase 7 showed trained-policy P(OFF) is flat vs k_fb. Stage 0's "raise eq_factor to 0.466" is the **wrong direction**. Not a next-experiment candidate. |
| **H6** | **Advantage/scale conditioning** (advantage 3–20 % of \|Q\|, RMSE ~24 %). | **= unchanged** | **LOW-MED** | Unaffected. |
| **H9** | **Critic value-fidelity off-distribution.** | **= (downstream of H1)** | **MED (as symptom)** | Unaffected; addressed inside H1's test. |
| **H8** | **Action-mapping compression** (gated map). | **= unchanged** | **LOW-MED** | V1-D: reward arg-min still reaches deep engine-side (da −1.2), so the map does not prevent a hard-engine arg-min from *existing*; whether the actor can occupy it is the H1/H2 question. |
| **H7** | **Training budget (150k).** | **= unchanged** | **LOW** | Unaffected. |
| **H10** | **ECMS structural / whole-cycle-tuning advantage explains part of the gap.** | **↓ bounded** | **LOW (as an explanation of the gap)** | V2: λ₀ tuning worth ~1 % and *given* to SAC; grid resolution makes the true target *lower* (gap wider); CS tolerance negligible. Benchmark is fair. A structural ~0.5–0.7 % pointwise-precision floor exists but is not RL-addressable. |
| — | Plant / powertrain / evaluator bug | ruled out | VERY LOW | V2 reproduction exact; Stage 0 audit clean. |

**Top of the ranking: H1 + H2 (intertwined, co-primary).** The actor cannot
occupy the hard-engine / sustained-OFF operating points that ECMS uses and that
its own critic (and, for the hard-engine part, its own reward's arg-min)
prefer, because the on-policy replay distribution structurally never covers that
region and the bimodal-value topology traps the Gaussian actor.

---

## F. NEXT EXPERIMENT RECOMMENDATION — EXACTLY ONE

### EXP-P11-1 — Pure-RL targeted exploration of the efficient high-engine-load region

**Hypothesis (tests H1 and, by its falsification branches, H2).** The critic's
low-load arg-max bias and the actor's soft-engine operating point at 15–35 Nm
are a self-reinforcing coverage deadlock in the efficient hard-engine region
(ECMS_NBHD/HIGH_EFF replay support 8–27 % vs 48–54 % for LOW). Guaranteeing
feasible coverage there re-orders the per-state arg-max-Q and lets the plain
actor track it — recovering the operating-point component (20–40 % of the gap)
and part of the mode-timing component.

**PRIMARY INDEPENDENT VARIABLE (one):** a training-time exploration schedule.
When the engine is commanded ON **and** `15 ≤ T_MGB < 50 Nm` **and** a higher
feasible engine load exists, with probability `p = 0.25` replace the sampled
action with a **uniform draw from the feasible high-engine-load `u`-interval**
(`u` such that `T_CE ∈ [1.3 · T_MGB, 0.9 · T_CE_max(w)]`, clipped to the
existing feasibility set). Implemented in `src/agents/targeted_exploration.py`
(the existing Phase-6 mechanism, extended) behind a new flag;
`predict(deterministic=True)` is **never** affected (evaluation-safe by
construction — same proof as Phase 6: `_sample_action` is called once in
`collect_rollouts`, SAC does not override it, no evaluation path calls it).

**PURITY (Section 1 compliance):** the injected action is a **uniform draw over
a feasibility-defined interval** — it encodes only *"a harder engine load is
reachable here"*, never what a good controller would choose. **No ECMS action,
trajectory, demonstration, imitation loss, warm-start, or benchmark-derived
reward term.** No offline data.

**FROZEN (everything else):** plant / powertrain / `ems_env` wiring / feasibility
logic / action bounds; ECMS; rule-based; `evaluate_policy.py`; SAC
implementation; `net_arch [256,256]`; optimiser (`lr 3e-4`); replay
(`buffer 300k`, `batch 512`, `tau 0.005`, `train_freq 64`,
`gradient_steps 16`); `γ = 0.20`; `n_step = 1`; `k_fb = 2.5`;
`eq_factor 0.2717/0.4981`; `target_entropy auto`; `action_map modeaware_gated`;
`lookahead 5`; `soc_deadband 0.10`; `lambda_soc 2.0`.

**CONTROL:** the existing 3-seed CONTROL checkpoints (`models_p5s0_k2.5/NEDC`,
`models_p5_k2.5/NEDC`, `models_p5_k2.5_s2/NEDC` + FTP75 equivalents), replay
buffers retained — provably untouched by the treatment code (Phase-6 design).

**SEEDS:** {0, 1, 2}. **BUDGET:** **150 000 steps** (matches every prior CONTROL
comparison; no scientific reason to deviate). Both cycles.

**RECORDED METRICS** → `results/phase11/EXP_P11_1/` (or `results/phase12/`):
replay support by engine-load region (OFF / LOW / ECMS_NBHD / HIGH_EFF / MAX)
in each 15–35 Nm band; twin-Q disagreement `|Q1−Q2|` there; per-state
arg-max-Q region distribution (matched ECMS-trajectory states, Phase-9 method);
`V_CE_equiv` mean ± SD + per-seed; charge-sustaining count; ΔSoC per seed;
constraint violations; per-band regional fuel (the Stage-0 §6.1 table); engine
`T_CE|on`, mean BSFC, engine-on steps; action distribution (mean/std/sat);
critic-loss / actor-loss / entropy-coefficient curves; training-stability
classification.

**PRE-DEFINED SUCCESS (all four required):**
1. **(coverage)** ECMS_NBHD/HIGH_EFF replay support at 15–35 Nm rises from
   8–27 % to **≥ 40 %** (manipulation check);
2. **(critic re-ordering)** per-state arg-max-Q share in {ECMS_NBHD, HIGH_EFF}
   at 15–35 Nm rises by **≥ +15 pp** vs CONTROL;
3. **(vehicle)** 3-seed mean NEDC `V_CE_equiv` improves vs CONTROL by
   **≥ 0.10 L/100km**, with **≥ 2/3 charge-sustaining** and **0 violations**;
4. **(mechanism)** engine `T_CE|on` rises toward ECMS by **≥ +10 Nm** and mean
   BSFC falls by **≥ 10 g/kWh**.

**PRE-DEFINED FALSIFICATION BRANCHES:**
* **(1) holds, (2) does NOT** (coverage up, arg-max unmoved — the Phase-6
  outcome, now in a *different* region): **FALSIFIES "coverage is sufficient"
  for the efficient region as well as the OFF region.** ⇒ the value-based
  route is exhausted; the advantage is below the critic's noise floor at this
  reward scale / 150k budget. Next branch is **not** another coverage or
  critic-regularisation tweak — it is a value-conditioning / horizon / policy-
  structure change.
* **(1) and (2) hold, (3) does NOT** (critic re-orders, fuel/CS worse):
  **isolates an actor-tracking failure ⇒ H2 is the binding constraint.** Next
  branch: policy parameterisation (discrete engine-mode head + continuous
  within-mode) or a structural change — **not** more coverage.
* **all four hold:** **H1 confirmed.** Proceed to a 500k confirmation + full
  Phase-9-style decomposition; then re-audit the residual gap.

**DESIGNATED FALLBACK (pre-registered, not this experiment):** if EXP-P11-1
falsifies on branch (1)-not-(2), the next experiment is the **clean γ ∈
{0.20, 0.50, 0.75, 0.90} sweep, 3 seeds, current stable config, `n_step 1`
throughout, all else frozen** (H4) — with success = ≥ 0.10 L/100km NEDC
improvement at ≥ 2/3 CS and bounded critic loss, falsification = monotone
degradation with horizon (the E10 result) reproduced on the stable config.

**Why EXP-P11-1 over the γ sweep as the single next experiment:** H1+H2 are
**HIGH** confidence and now co-primary (V1-D reinforced them by showing the
actor undershoots its own reward's hard-engine arg-min); H4 is **MED** and
partly confounded by the abandoned-baseline E10 result. EXP-P11-1 attacks the
largest cleanly-attributed gap component and its falsification branches are
mutually exclusive and individually decisive.

---

## G. STOP

**Stage 0 is complete.** C1/H3 falsified (refined finding retained); ECMS
benchmark validated as fair, robust, and near-optimal for its class; DP
deferred (no solver exists). Updated ranking puts **H1 + H2 (exploration /
optimisation topology for efficient high-engine-load operation)** as
co-primary. One next experiment is specified (EXP-P11-1) with pre-defined
success and falsification criteria, plus a pre-registered fallback (H4 γ sweep).

**No experiment executed. No RL controller modified. No training run. No
calibration applied. Awaiting human approval before proceeding.**
