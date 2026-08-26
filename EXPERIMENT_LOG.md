# Experiment Log

Chronological record of every RL experiment. Machine-readable configuration
lives in `experiments/experiment_registry.yaml`; the *reasoning* lives here.

**Rules**
- Every entry records: hypothesis → configuration → result → interpretation →
  decision → next action.
- Never delete an entry. A superseded conclusion gets a `SUPERSEDED BY` line.
- All metrics come from `python -m results.evaluate_policy` (the single
  authoritative evaluator). No controller-specific shortcuts.
- **Primary metric: `V_CE_equiv`** (L/100km) with charge-sustaining SoC and
  zero constraint violations. Never judge from training reward alone.

**Targets** — NEDC: rule-based **3.5056**, ECMS 3.1887 · FTP75: rule-based
**3.2323**, ECMS 2.8097. Rule-based is the fair near-term target; ECMS is a
stretch (its λ is bisection-tuned with whole-cycle information).

---

## E0 — Baselines (reference)

| id | cycle | seed | steps | V_CE_equiv | ΔSoC | OFF% | ASSIST% | LPS% |
|---|---|---|---|---|---|---|---|---|
| BASELINE-seed0 | NEDC | 0 | 497,760 | 4.3702 | +2.04pp | 22.9 | 32.8 | 26.6 |
| **BASELINE-seed1** | NEDC | 1 | 148,840 | **4.1245** | +2.63pp | 29.4 | 20.6 | 33.0 |
| BASELINE-FTP75 | FTP75 | 0 | 998,032 | 4.2072 | −1.44pp | 17.4 | 34.3 | 22.7 |

Config: `--action-map linear --eq-factor 1.3125/2.4062 --k-fb 8.0
--gradient-steps 16`, γ=0.9999, target_entropy=−1.0, buffer 300k, n_step 5,
lookahead 5.

**Reproduction check (Phase-2 §5): PASS.** `results/evaluate_policy.py`
reproduces BASELINE-seed1 exactly — 4.1245 / SoC 52.63% / OFF 29.4% /
ASSIST 20.6% / LPS 33.0% / REGEN 17.0%, and the rule-based benchmark at
3.5056 / OFF 59.0%. Cleared to run Experiment B.

**Interpretation.** Best checkpoint always lands in the first half of
training, then degrades: seed0 frozen at step 65,880 of 497,760 (430k further
steps, zero gain); FTP75 peaked at 296,408 then collapsed (quintile means
4.540→4.431→4.408→4.704→**4.946**, SoC 52.3%→**32.2%**). "Train longer" is
refuted on 2 seeds and 2 cycles.

---

## E1 — Q-landscape forensics (no training)

**Hypothesis.** The critic cannot resolve the OFF/ASSIST distinction because
the fuel-cutoff discontinuity gets smoothed.

**Method.** `results/q_landscape.py` — 41-point action sweep at three torque
bands; true one-step reward obtained by deep-copying the env at the probe
state so every action starts from an identical physical state.

**Result.**

| probe | T_MGB | best-OFF − best-ASSIST (true reward) | (min-Q) | verdict |
|---|---|---|---|---|
| low | 8.69 Nm | **+0.0101** | +0.135 | CRITIC AGREES |
| med | 22.2 Nm | **−0.0255** | −0.273 | CRITIC AGREES |
| high | 48.93 Nm | **−0.0825** | −0.533 | CRITIC AGREES |

**Interpretation.** Hypothesis **REJECTED**. The critic tracks the reward
faithfully at every probe. The *reward itself* ranks ASSIST/LPS above OFF at
medium and high torque. This redirected the investigation to the reward.

**Decision.** Audit the reward's battery pricing units. → E2

---

## E2 — Reward unit analysis (no training) ⇒ **P0-REVISED**

**Hypothesis.** `eq_factor` is not on the same scale as ECMS's costate λ.

**Result — unit analysis.**
```
1 J fuel    -> 3.177172e-08 L      1 J battery -> 1.534866e-07 L
implicit lambda at eq_factor=1.0 = 4.8309 fuel-J per battery-J
ECMS proven optimal lambda_0     = 1.3125 (NEDC) / 2.4062 (FTP75)
```
`elec_liters` already carries `EFC_GAIN = 1/(η_BT·η_EM·η_CE·(H_u/3.6e6)·ρ_f)`,
so multiplying it by an "ECMS λ" **double-counts the conversion**.
`eq_factor=1.0` is already **3.68×** ECMS's λ; `eq_factor=1.3125` (used in
every session experiment) is **4.83×** too expensive.

**Result — behavioural test** (160 states; reward-optimal action from an
81-point sweep vs the ECMS Hamiltonian-optimal action at identical state/SoC):

| eq_factor | agrees with ECMS | reward picks OFF | ECMS picks OFF |
|---|---|---|---|
| **1.3125** (as used) | **7.5%** | **0.0%** | 90.0% |
| **0.2717** (unit-matched) | **86.9%** | **78.1%** | 90.0% |

**Interpretation.** At the pricing used in every experiment, the reward-optimal
action is **never** engine-OFF. The agent was correctly optimizing a
misspecified objective. This supersedes the Phase-1 P0 (action geometry) as
the primary cause, and explains why raising `eq_factor` to 1.3125 "to match
ECMS" earlier in the session made the ASSIST blob *worse*.

**Correct conversion:** `eq_factor_env = λ_ECMS / 4.8309`,
`k_fb_env = k_fb_ECMS / 4.8309` → NEDC 0.2717 / FTP75 0.4981, k_fb 1.656.

**Metric integrity.** This changes only the *training signal*. `V_CE_equiv` is
computed independently by the validated EFC block and is untouched.

**Decision.** Run EXP-B2 isolating the pricing fix. Keep EXP-B running as a
falsification control for the action-geometry hypothesis.

---

## E3 — EXP-FAIR: benchmark control-authority audit (no training)

**Hypothesis.** The rule-based benchmark reaches SoC 0.61% on NEDC; the RL
agent is hard-masked at `SOC_MIN=0.05`. Part of the gap may be unequal
authority rather than RL skill.

**Method.** Run the benchmark twice — raw through the plant (how 3.5056 is
produced) and with its `u` routed through the env's normal masked action path.

| cycle | RAW | MASKED | authority penalty |
|---|---|---|---|
| NEDC | 3.5056 (SoC_min 0.61%, 38 steps <5%) | 3.5792 (SoC_min 4.64%) | **+0.0736 (+2.10%)** |
| FTP75 | 3.2323 (SoC_min 13.60%) | 3.2318 | −0.0005 (−0.01%) |

**Interpretation.** The asymmetry is **real but small**. It accounts for ~2.1
of the 17.7 percentage-point NEDC gap and ~0 of the FTP75 gap. It does **not**
excuse the result. Priority downgraded **P1 → P2**.

**Authority-equal target (NEDC): 3.5792.** SAC 4.1245 is +15.2% worse on that
basis (vs +17.7% against the published number).

**Decision.** Keep `SOC_MIN=0.05` for the agent; report the authority-equal
number alongside the published one. Do not change the locked benchmark.

---

## E4 — EXP-B: action-space reparameterization (running)

**Hypothesis (P0-OLD).** Engine-OFF occupies a narrow, state-dependent sliver
of the action range (median 12.19% NEDC / 9.29% FTP75; <10% wide on ~half of
traction steps; boundary `a_off` spanning +0.40…+0.89). A control-equivalent
reparameterization making the OFF band a state-invariant 40% will let SAC
commit to OFF.

**Candidate analysis** (structural metrics first, per §9 — not fuel):

| mapping | OFF med% | p25 | p75 | LPS med% | verdict |
|---|---|---|---|---|---|
| ORIGINAL linear | 12.19 | 7.13 | 30.22 | 45.95 | baseline |
| power p=1.5 / 2 / 3 | 8.30 / 6.29 / 4.24 | — | — | — | **REJECTED — these NARROW the band** |
| power p=0.5 | 22.89 | 13.75 | 51.31 | 21.11 | rejected — LPS shrinking |
| power p=0.25 | 40.55 | 25.60 | 76.29 | **4.46** | **REJECTED — LPS collapses** |
| piecewise fixed 35/25/40 | 40.00 | 23.39 | 50.77 | 35.00 | good, still state-varying |
| **analytic mode-aware 35/25/40** | **40.00** | **40.00** | **40.00** | **35.00** | **SELECTED** |

The a-priori suggested p=1.5/2/3 move the band the **wrong way** — verifying
rather than assuming was necessary. Mode-aware anchors breakpoints on the true
boundary `u_thr = 1 − T_CUTOFF/T_MGB`, giving OFF exactly 40% at *every*
timestep (zero state variance) and a constant `a_off = +0.20`.

**Control equivalence — PROVED**, `tests/test_action_mapping.py` (53 tests):
exact endpoints for every T, strict monotonicity (bijection onto
[U_MIN,U_MAX]), identical reachable `u` set, regen/sub-cutoff unchanged,
`linear` bit-identical to the original formula. 265/265 suite passes.

**Configuration.** Identical to BASELINE-seed1 except `--action-map modeaware`.
`git bd5fe41`, seed 1, 150k steps, NEDC + FTP75, `out=models_expB`.

**Status: RUNNING.** Demoted to a **falsification control** after E2 — at the
current pricing the reward-optimal action is never OFF, so widening the door
the reward forbids should not help much. A large gain would *contradict* E2.

---

## E5 — EXP-B2: reward unit fix (running)

**Hypothesis (P0-REVISED).** Unit-matching the battery price to ECMS's proven
costate will let SAC learn benchmark-like engine-OFF behaviour.

**Configuration.** Identical to BASELINE-seed1 except
`--eq-factor 0.2717 --k-fb 1.656` (one coupled unit fix). **`action-map`
deliberately left `linear`** so the pricing effect is isolated from E4.
`git bd5fe41`, seed 1, NEDC, 150k steps, `out=models_expB2`.

**Success criterion.** OFF ≥ 40% AND ASSIST ≤ 10% AND V_CE_equiv < 4.1245.

**Status: RUNNING.**

---

## Pending

| id | hypothesis | isolated variable |
|---|---|---|
| EXP-C | target_entropy −1.0 forces stochasticity opposing commitment | target_entropy ∈ {−1,−2,−3} |
| EXP-D | γ=0.9999 justification void (terminal term = 0.77%) | γ ∈ {0.99, 0.995, 0.999, 0.9999} |
| EXP-E | `eq_eff` sign inversion above SoC 66.4% is harmful | clip `eq_eff` to positive floor |
| EXP-F | buffer 300k (vs SB3 1M default) drives late-run collapse | buffer_size |
| EXP-G | duplicate/dead observations (v_next≡fut_v1, gear_oh6≡0) | 20-state vs 18-state |
| EXP-H | TD3's deterministic policy commits better than SAC's Gaussian | algorithm (only if SAC still trails) |
