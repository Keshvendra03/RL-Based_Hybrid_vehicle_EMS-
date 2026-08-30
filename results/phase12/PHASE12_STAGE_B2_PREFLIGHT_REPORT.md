# PHASE 12 — STAGE B2: PRE-FLIGHT REPORT

> **Deep-LPS coverage falsification — pre-training feasibility / targeting audit.
> NO TRAINING PERFORMED IN THIS STEP.**

**Date/time:** 2026-08-30 · **Code revision:** git `90af969` + Stage-A
`clip_eq_eff` flag (`src/env/ems_env.py`, +28 lines, default-OFF). No other
`src/` change. · **Script:** `results/phase12/stage_b2_preflight.py` · **Data:**
`results/phase12/data/stage_b2_preflight.json`.

---

## 1. THE ONLY EXPERIMENTAL VARIABLE

The B2 injection interval, defined **entirely from the environment's own
authoritative feasibility calculation** (`EMSEnv._action_to_torques` scanned
over `a ∈ [-1, 1]`, 161 pts):

```
TCE_max_feasible(s) = max_a  _action_to_torques(a).T_CE          (deepest LPS, a→U_MIN)
TCE_min_feasible(s) = min_a  _action_to_torques(a).T_CE          (engine OFF,  a→U_MAX)
TCE_deep_low(s)     = max( TCE_min_feasible(s), TCE_max_feasible(s) - 15.0 )
TCE_deep_high(s)    = TCE_max_feasible(s)
TCE_injected        ~ Uniform( TCE_deep_low(s), TCE_deep_high(s) )
a_injected          = action whose EXECUTED T_CE equals TCE_injected
                      (monotone-decreasing a→T_CE inverted by interpolation on the same scan)
```

**Not** `1.3·demand` as a lower bound. **Not** a fixed `[55, 75] Nm` rule. The
lower bound is `TCE_max_feasible − 15 Nm`, purely state-dependent — the top
15 Nm of the physically feasible engine-torque range at that state.

Everything else is frozen at the Phase-12B config (§0 rules 1–17): reward
formulation (only the approved Stage-A `clip_eq_eff=True` safety correction),
`eq_factor 0.2717`, `k_fb 2.5`, `γ 0.20`, net `[256,256]`, optimiser, entropy
auto, `n_step 1`, observation, physical constraints, reachable action set.
`predict(deterministic=True)` untouched.

---

## 2. ELIGIBILITY CONDITIONS (§5)

All must hold, evaluated from the decoded observation:

| # | condition | source |
|---|---|---|
| 1 | `15 ≤ T_MGB < 50 Nm` | decoded `obs[2]·150` |
| 2 | engine action available (`T_MGB > T_CUTOFF`, `w > 0`) | decoded obs |
| 3 | non-empty feasible interval, `TCE_max_feasible − TCE_min_feasible ≥ 5 Nm` | env `_action_to_torques` scan |
| 4 | `TCE_max_feasible > TCE_min_feasible` | as above |
| 5 | within all existing physical constraints (the env's own masks run unchanged after injection) | env |
| 6 | **`SoC < 0.55`** | decoded `(obs[4]+1)/2` |
| 7 | intervention probability `p = 0.25` | fixed |

**Why SoC < 0.55 (§6):** Phase-12B showed targeted high-load intervention can
push SoC toward the region where the *original* (pre-Stage-A) `eq_eff` defect
activated (60.9 %). The Stage-A correction remains enabled as a safety net, but
this experiment studies *critic coverage*, not aggressive charging, so
injection is gated below 0.55 (below the SoC-deadband edge). The underlying
vehicle / SoC dynamics are unchanged.

---

## 3. PRE-FLIGHT METHOD (§3, §8, §9)

Representative states = **every transition of the 3 CONTROL deterministic NEDC
trajectories** (3 × 1220 = **3660 transitions**). For each eligible transition:
compute `TCE_min_feasible`, `TCE_max_feasible`, the injection interval, draw
`TCE_injected`, invert to `a`, then **re-execute `a` through a fresh
`EMSEnv._action_to_torques`** and record `T_CE_executed`, whether any clamp
modified it, and `execution_fidelity = T_CE_executed / T_CE_requested`.

---

## 4. RESULTS

### 4.1 Eligibility & coverage projection (§9)

* total transitions in the CONTROL sample: **3660**
* eligible transitions (all 7 conditions): **1152 (31.5 %)**
* expected interventions over one 3660-transition eval trajectory: ≈ 288
* **expected interventions over a 150 000-step training run: ≈ 11 803**
  (scale = 150 000 / 3660 ≈ 41; consistent with Phase-12B's measured 11.6–11.8k)

| demand band | n transitions | n eligible | frac eligible (of sample) | **expected interventions / 150k run** | mean `TCE_max_feasible` | mean requested `T_CE` | mean **executed** `T_CE` | mean ρ_executed | frac ρ ≥ 0.75 | frac within 15 Nm of max | execution fidelity | n clamped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 15–25 Nm | 567 | 567 | 0.155 | **5 809** | 34.07 | 26.81 | **26.81** | 0.782 | 0.582 | 1.00 | **1.000** | 0 |
| **25–30 Nm** | 111 | 111 | 0.030 | **1 137** | 52.86 | 45.23 | **45.23** | **0.856** | **0.865** | 1.00 | **1.000** | 0 |
| **30–35 Nm** | 351 | 351 | 0.096 | **3 596** | 57.95 | 50.41 | **50.41** | **0.870** | **0.963** | 1.00 | **1.000** | 0 |
| 35–50 Nm | 123 | 123 | 0.034 | **1 260** | 73.14 | 66.45 | **66.45** | 0.909 | 1.000 | 1.00 | **1.000** | 0 |

`ρ_executed = (T_CE_executed − TCE_min_feasible) / (TCE_max_feasible − TCE_min_feasible)`.

### 4.2 Executed-action audit (§8) — sample (25–30 / 30–35 Nm demand)

| band | T_MGB | SoC | feasible `T_CE` | injection interval | requested | **executed** | ρ | feasible? | survives clamps? | executed = intended? | fidelity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30–35 | 34.95 | 50 % | [0.0, 64.66] | [49.66, 64.66] | 61.9 | **61.9** | 0.957 | ✓ | ✓ | ✓ | 1.00 |
| 30–35 | 34.95 | 41 % | [0.0, 64.66] | [49.66, 64.66] | 62.8 | **62.8** | 0.971 | ✓ | ✓ | ✓ | 1.00 |
| 30–35 | 30.05 | 50 % | [0.0, 55.59] | [40.59, 55.59] | 54.3 | **54.3** | 0.977 | ✓ | ✓ | ✓ | 1.00 |
| 25–30 | 29.09 | 50 % | [0.0, 53.81] | [38.81, 53.81] | 47.9 | **47.9** | 0.890 | ✓ | ✓ | ✓ | 1.00 |
| 25–30 | 28.44 | 41 % | [0.0, 52.61] | [37.61, 52.61] | 49.5 | **49.5** | 0.942 | ✓ | ✓ | ✓ | 1.00 |
| 25–30 | 28.53 | 48 % | [0.0, 52.78] | [37.78, 52.78] | 38.3 | **38.3** | 0.725 | ✓ | ✓ | ✓ | 1.00 |

**Across all 1152 eligible transitions and all 4 bands: `execution_fidelity =
1.000` and `n_clamped = 0`.** The monotone `a → T_CE` inversion is exact; the
env's feasibility masks never modify the injected action (by construction the
interval is inside the true feasible set). **The environment executes precisely
the intended deep-LPS torque.** The 30–35 Nm injections reach `T_CE` up to
**62.8 Nm — coincident with the reward's arg-max (~63 Nm)**, which Phase-12B
never covered.

---

## 5. ACCEPTANCE CRITERIA (§10)

| criterion | requirement | result | verdict |
|---|---|---|---|
| **A — Feasibility** | injected actions inside the true physical feasible interval | every one of 1152; 0 clamped; all `T_CE > T_CUTOFF`, all `a ∈ [-1, 1]` | **PASS** |
| **B — Reachability** | no new physical action set | action space `Box(-1.0, 1.0, (1,), float32)` unchanged | **PASS** |
| **C — Targeting** | majority of executed injections at ρ ≥ 0.75 (after all transforms) | 25–30 Nm: **86.5 %**; 30–35 Nm: **96.3 %**; 35–50 Nm: 100 % | **PASS** |
| **D — Low-demand relevance** | meaningful intervention frequency in 25–30 and 30–35 Nm | ≈ **1 137** (25–30) and ≈ **3 596** (30–35) interventions per 150k run | **PASS** |
| **E — No hidden remapping** | executed action not collapsed back to part-load | mean executed `T_CE` = mean requested (45.2 / 50.4 / 66.5), all within 15 Nm of `TCE_max_feasible` (52.9 / 58.0 / 73.1); fidelity 1.0 | **PASS** |
| **F — SoC safety** | `SoC < 0.55` enforced before intervention | eligibility test excludes `SoC ≥ 0.55`; 0 eligible transitions violate it | **PASS** |

### **ALL SIX CRITERIA PASS.**

Minor note (not a failure): in the **15–25 Nm** demand band ρ ≥ 0.75 holds for
only **58 %** of injections — at very low demand `TCE_max_feasible ≈ 30–38 Nm`,
so the top-15-Nm window is wide relative to the small feasible span and uniform
draws occasionally land at ρ 0.5–0.75. This band is **not** the critical region
(§13: the critical region is 25–35 Nm, where targeting is 86–96 %). It is
retained for eligibility so the 35–50 Nm comparison band is also populated.

---

## 6. IMPLEMENTATION AMBIGUITIES (§19)

**None found.** The env exposes the exact feasible `T_CE` range through
`_action_to_torques` (used verbatim, no simplified re-model); the `a → T_CE`
relation is monotone-decreasing and continuous (Phase-11 §11B), so the
inversion is well-defined; the executed action equals the intended action to
machine precision.

---

## 7. CONCLUSION — PROCEED

The pre-flight demonstrates: (A) feasibility, (B) reachability, (C) targeting
into the top-15-Nm feasible window with fidelity 1.0, (D) thousands of
interventions in the critical 25–35 Nm bands over a 150k run, (E) no hidden
remapping, (F) SoC < 0.55 safety gate. The intervention is **statistically
capable** of producing substantial deep-LPS replay coverage in the region the
reward prefers, which Phase-12B could not.

**PRE-FLIGHT: PASS. Cleared to run the B2 training experiment (3 seeds × 150 000
steps).**
