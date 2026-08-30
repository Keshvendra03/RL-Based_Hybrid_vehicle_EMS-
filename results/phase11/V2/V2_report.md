# PHASE 11 — V2: ECMS BENCHMARK SENSITIVITY

**NO TRAINING. NO RL CHANGE. NO EDIT TO `src/baselines/ecms.py`.**
Script: `results/phase11/v2_ecms_sensitivity.py` (re-implements the ECMS cycle
loop locally, mirroring `ecms.run_ecms_fixed_lambda` exactly — same plant
blocks, same `_hamiltonian_best_u` feasibility masks, same per-step wiring — to
add engine-operating diagnostics without touching the frozen source).
Data: `results/phase11/data/v2_ecms_sensitivity.json`.
`enable_fast_interpolation()` used at runtime only (proven exact < 1e-12).

---

## HEADLINE

**The reported ECMS number is faithfully reproduced, charge-sustaining, robust,
and mildly conservative. ECMS's whole-cycle information advantage is real but
small (≈1 % of the SAC→ECMS gap) and is largely neutralised because SAC is
handed λ₀ as its `eq_factor` anchor.**

| | NEDC | FTP75 |
|---|---|---|
| **Reproduced `v_ce_equiv`** (λ₀ tuned, k_fb 8, grid 81) | **3.1887** (reported 3.1887 ✓) | **2.8097** (reported 2.8097 ✓) |
| SoC_end / engine-on / T_CE\|on | 50.36 % / 29.9 % / 76.9 Nm | 50.13 % / 33.9 % / 64.8 Nm |
| Fuel spread over λ₀ ± 20 % | 3.185 → 3.227 (**±0.7 % / 1.3 % total**) | 2.809 → 2.859 (**±0.9 % / 1.8 % total**) |
| Lowest **CS** fuel in the λ₀ sweep | **3.1848** @ λ₀ × 0.95 (−0.004 vs tuned) | **2.8088** @ λ₀ × 0.95 (−0.001) |
| Fine-grid optimum (321 pts, k_fb 8) | **3.1669** (−0.022 / −0.7 %) | **2.7987** (−0.011 / −0.4 %) |
| Constant-λ (k_fb 0) charge-sustaining? | **NO** — SoC → 22.1 % | **NO** — SoC → 13.8 % |

---

## V2-A — λ₀ SENSITIVITY (k_fb = 8, grid = 81)

| mult | λ₀ | `v_ce_equiv` | `v_liter` | SoC_end % | ΔSoC pp | CS 0.5 % | CS 2 % | eng-on % | T_CE\|on |
|---|---|---|---|---|---|---|---|---|---|
| **NEDC** (tuned λ₀ = 1.3125) | | | | | | | | | |
| 0.80 | 1.0500 | 3.1902 | 3.1530 | 47.28 | −2.72 | ✗ | ✗ | 29.7 | 76.5 |
| 0.85 | 1.1156 | 3.1868 | 3.1592 | 47.99 | −2.00 | ✗ | ✗ | 29.7 | 76.7 |
| 0.90 | 1.1813 | 3.1855 | 3.1680 | 48.73 | −1.27 | ✗ | ✓ | 29.7 | 76.9 |
| 0.95 | 1.2469 | **3.1848** | 3.1790 | 49.58 | −0.42 | ✓ | ✓ | 29.8 | 76.9 |
| **1.00** | **1.3125** | **3.1887** | 3.1887 | 50.36 | +0.36 | ✓ | ✓ | 29.9 | 76.9 |
| 1.05 | 1.3781 | 3.1995 | 3.1995 | 51.22 | +1.22 | ✗ | ✓ | 30.0 | 76.8 |
| 1.10 | 1.4438 | 3.2081 | 3.2081 | 51.89 | +1.89 | ✗ | ✓ | 30.1 | 76.8 |
| 1.15 | 1.5094 | 3.2183 | 3.2183 | 52.66 | +2.66 | ✗ | ✗ | 30.1 | 77.1 |
| 1.20 | 1.5750 | 3.2265 | 3.2265 | 53.38 | +3.38 | ✗ | ✗ | 30.5 | 76.5 |
| **FTP75** (tuned λ₀ = 2.4062) | | | | | | | | | |
| 0.80 | 1.9250 | 2.8115 | 2.7663 | 44.59 | −5.41 | ✗ | ✗ | 33.9 | 63.9 |
| 0.90 | 2.1656 | 2.8089 | 2.7858 | 47.26 | −2.74 | ✗ | ✗ | 33.9 | 64.2 |
| 0.95 | 2.2859 | **2.8088** | 2.7981 | 48.73 | −1.27 | ✗ | ✓ | 34.0 | 64.3 |
| **1.00** | **2.4062** | **2.8097** | 2.8097 | 50.13 | +0.13 | ✓ | ✓ | 33.9 | 64.8 |
| 1.10 | 2.6468 | 2.8337 | 2.8337 | 53.09 | +3.09 | ✗ | ✗ | 34.1 | 64.9 |
| 1.20 | 2.8874 | 2.8592 | 2.8592 | 56.05 | +6.05 | ✗ | ✗ | 34.2 | 65.4 |

**Findings.**
* `v_ce_equiv` (which charges for net battery use) is **nearly flat**: total
  spread over λ₀ ± 20 % is **1.3 % (NEDC) / 1.8 % (FTP75)**. `v_liter` (raw
  engine fuel) falls monotonically as λ₀ drops — but that "gain" is un-repaid
  battery drain that `v_ce_equiv` prices back in.
* The **CS window is narrow**: only λ₀ ∈ [0.90, 1.10]× is CS at 2 % on NEDC;
  only [0.95, 1.00]× at 0.5 %. A causal controller that mis-estimated λ₀ by
  > 10 % would **fall out of charge-sustaining**, not merely lose a little fuel.
* The **whole-cycle bisection landed essentially on the CS-constrained fuel
  optimum for the constant-λ class**: the lowest CS fuel in the sweep is
  3.1848 (λ₀ × 0.95), only **0.004 L/100km (0.1 %) below** the tuned 3.1887.
* The **ECMS operating strategy is robust to λ₀**: engine-on fraction
  (29.7–30.5 %) and T_CE\|on (76.5–77.1 Nm) barely move across the whole sweep.
  λ₀ shifts the SoC endpoint, not the operating-point distribution.

---

## V2-B — k_fb SENSITIVITY (λ₀ tuned, grid = 81)

| k_fb | `v_ce_equiv` | `v_liter` | SoC_end % | ΔSoC pp | CS 0.5 % | CS 2 % |
|---|---|---|---|---|---|---|
| **NEDC** | | | | | | |
| 0 | 3.4982 | 3.1444 | **22.14** | **−27.86** | ✗ | ✗ |
| 4 | 3.2362 | 3.0324 | 34.55 | −15.45 | ✗ | ✗ |
| **8** | **3.1887** | 3.1887 | 50.36 | +0.36 | ✓ | ✓ |
| 16 | 3.3087 | 3.3087 | 58.04 | +8.04 | ✗ | ✗ |
| **FTP75** | | | | | | |
| 0 | 3.1094 | 2.8338 | **13.80** | **−36.20** | ✗ | ✗ |
| 4 | 2.7968 | 2.7549 | 44.99 | −5.01 | ✗ | ✗ |
| **8** | **2.8097** | 2.8097 | 50.13 | +0.13 | ✓ | ✓ |
| 16 | 2.8555 | 2.8555 | 50.91 | +0.91 | ✗ | ✓ |

**Findings.**
* **Constant-λ ECMS (k_fb = 0) CANNOT charge-sustain this plant** — SoC runs to
  22 % (NEDC) / 14 % (FTP75). The `ecms.py` docstring claim is **confirmed
  empirically**. The SoC-feedback term is load-bearing, not cosmetic.
* SoC_end is very sensitive to k_fb (NEDC: 0 → 22 %, 4 → 35 %, 8 → 50 %,
  16 → 58 %). **k_fb = 8 is the unique CS setting** among {0, 4, 8, 16} on
  NEDC (FTP75 tolerates 8 and 16).
* This is a **closed-loop necessity**, not an "unfair" advantage: any real-time
  ECMS-style controller (including SAC's `k_fb` reward term) needs an
  equivalent SoC-feedback mechanism.

---

## V2-C — ACTION-GRID RESOLUTION (λ₀ tuned, k_fb = 8)

| grid pts | `v_ce_equiv` | SoC_end % | ΔSoC pp | eng-on % | T_CE\|on |
|---|---|---|---|---|---|
| **NEDC** | | | | | |
| 41 | 3.2225 | 51.03 | +1.03 | 31.4 | 73.2 |
| 81 (reported) | **3.1887** | 50.36 | +0.36 | 29.9 | 76.9 |
| 161 | 3.1721 | 49.82 | −0.18 | 29.7 | 77.1 |
| 321 | **3.1669** | 49.85 | −0.15 | 29.5 | 77.3 |
| **FTP75** | | | | | |
| 41 | 2.8398 | 50.69 | +0.69 | 32.6 | 67.5 |
| 81 (reported) | **2.8097** | 50.13 | +0.13 | 33.9 | 64.8 |
| 161 | 2.8010 | 50.31 | +0.31 | 36.7 | 60.1 |
| 321 | **2.7987** | 50.47 | +0.47 | 38.2 | 57.9 |

**Findings.**
* **The 81-point benchmark is mildly grid-limited.** 81 → 321 lowers ECMS fuel
  by **0.022 L/100km (0.7 %) on NEDC** and **0.011 (0.4 %) on FTP75**, staying
  charge-sustaining.
* **This makes the "true" constant-λ ECMS target ≈ 3.167 / 2.799**, i.e. the
  reported number is *conservative* by ~0.7 % / 0.4 %. Grid resolution
  therefore **does NOT explain away any of the SAC gap** — it slightly *widens*
  the true target (NEDC SAC−ECMS_fine ≈ 3.767 − 3.167 = **+0.600**).
* On FTP75 the finer grid shifts ECMS toward *more* engine-on (32.6 → 38.2 %)
  at *lower* T_CE\|on (67.5 → 57.9) for lower total fuel — the coarse grid was
  over-committing to a few hard operating points.

---

## V2-D — CHARGE-SUSTAIN TOLERANCE (diagnostic; project criterion unchanged)

Applying the two tolerances as labels to the V2-A sweep:

| | NEDC | FTP75 |
|---|---|---|
| # of λ₀ points CS @ 0.5 % | 2 (× 0.95, 1.00) | 1 (× 1.00) |
| # of λ₀ points CS @ 2 % | 5 (× 0.90 … 1.10) | 3 (× 0.95 … 1.05) |
| lowest CS fuel @ 0.5 % | 3.1848 | 2.8088 |
| lowest CS fuel @ 2 % | 3.1848 | 2.8088 |

**Finding.** Relaxing the CS tolerance from 0.5 % to 2 % changes the
lowest achievable ECMS fuel by **≈ 0** (< 0.001 L/100km) — ECMS's
fuel-vs-SoC-endpoint curve is flat near the target. The CS tolerance is **not**
a source of ambiguity in the benchmark. (The project's official criterion
remains the 2 % check in `evaluate_policy.py`; SAC and ECMS are both judged by
it.)

---

## V2-E — WHOLE-CYCLE INFORMATION ADVANTAGE

1. **Does determining λ₀ require repeated full-cycle rollouts?** **Yes.**
   `ecms.tune_lambda` runs `run_ecms_fixed_lambda` over the **entire cycle**
   repeatedly, bisecting λ₀ until `|SoC_end − 0.5| ≤ 0.5 %`, keeping the
   closest-to-target run.
2. **Does it depend on knowledge of the complete future cycle?** **Yes** — the
   bisection target (`SoC_end`) is a whole-cycle outcome.
3. **Could a strictly causal online controller reproduce that procedure?**
   **No** — it cannot pre-simulate the cycle. It would have to *estimate* λ
   online (which the `k_fb` SoC-feedback term does — V2-B shows this is
   feasible and necessary).
4. **Is the tuned λ₀ supplied to SAC?** **Yes.** The CONTROL's `eq_factor` is
   set to `λ₀ / 4.8309` (0.2717 NEDC / 0.4981 FTP75) — SAC is *handed* ECMS's
   whole-cycle-tuned scalar as its per-step price anchor.
5. **What information advantage remains after λ₀ is supplied?**
   * The *global consistency* that a single whole-cycle-tuned scalar buys:
     every ECMS decision uses the same λ₀, guaranteed to charge-sustain the
     whole cycle. SAC's `k_fb` closed-loop term approximates this but with a
     different (steeper) slope and learning noise.
   * **Bounded magnitude:** V2-A shows that if a causal controller lands λ₀
     within ±5 % and stays CS, it loses only **~0.004–0.005 L/100km (≈1 % of
     the +0.50 SAC gap)** vs the whole-cycle-tuned value. Outside ±10 % it
     falls out of CS entirely.
   * ECMS also re-solves the exact instantaneous Hamiltonian over an 81-point
     grid with a perfect plant model every step (zero function-approximation
     error) — a *structural* precision advantage that is not "information" and
     not addressable by giving SAC more data (Phase 8 "E").

**Stated plainly (no exaggeration):**
* **ECMS has:** whole-cycle-tuned λ₀ (worth ~1 % of the gap, and *given* to
  SAC); exact per-step Hamiltonian minimisation with a perfect plant model
  (structural, not information).
* **SAC has:** the same λ₀ anchor; a 5-second causal speed preview in its
  observation (ECMS has **no** per-step preview); a learned SoC-feedback term.
* **Neither** has an unfair ledger/physics advantage (identical plant blocks,
  identical feasibility masks, identical evaluator — V2 reproduction exact).

---

## DP DECISION

`grep` over `src/` and `results/` for dynamic-programming / value-iteration /
Bellman-grid solvers returned **nothing**. No validated DP implementation
exists.

> **`DP_STATUS = DEFERRED_NO_EXISTING_VALIDATED_SOLVER`**

Per task rules, **no DP solver was written**. DP is a separate future benchmark
task. **This does NOT make ECMS the global optimum, and 3.1887 / 2.8097 are NOT
a proven physical lower bound.** Constant-λ ECMS is documented (`ecms.py`) to
sit "a hair above DP", but there is no DP number for this plant.

---

## V2 QUESTIONS — ANSWERS

**V2-Q1 — Sensitivity to λ₀?** **Low.** `v_ce_equiv` spread over λ₀ ± 20 % is
**1.3 % (NEDC) / 1.8 % (FTP75)**. The CS constraint pins λ₀ to ±5 % (0.5 % tol)
/ ±10 % (2 % tol). The whole-cycle bisection landed **0.004 L/100km above** the
CS-feasible fuel minimum of its class.

**V2-Q2 — Sensitivity to k_fb?** **High on SoC, and k_fb is *necessary*.**
k_fb = 0 (constant λ) cannot charge-sustain (SoC → 22 % NEDC / 14 % FTP75).
k_fb = 8 is the unique CS value among {0, 4, 8, 16} on NEDC. Fuel at the CS
setting is not sensitive; the mechanism is load-bearing.

**V2-Q3 — Action-grid resolution?** **Small and one-directional.** 81 → 321
points lowers ECMS fuel by **0.022 (NEDC, 0.7 %) / 0.011 (FTP75, 0.4 %)**,
CS-preserving. The reported 81-point number is *conservative*; the true
constant-λ optimum is ≈ **3.167 / 2.799**. This **widens**, not narrows, the
SAC target.

**V2-Q4 — CS tolerance?** **Negligible.** Lowest achievable ECMS fuel changes
by < 0.001 L/100km between 0.5 % and 2 % tolerance.

**V2-Q5 — How much of the nominal SAC–ECMS gap is plausibly ECMS's whole-cycle
tuning?** **≈ 1 % of the gap (~0.005 L/100km), and *given* to SAC.** A causal
controller within ±5 % of λ₀ and charge-sustaining loses only ~0.004–0.005
vs the tuned value; SAC already gets λ₀ as its `eq_factor` anchor. The
whole-cycle tuning is **not** a material explanation for the +0.50 L/100km gap.
Grid resolution accounts for **−0.7 % / −0.4 %** (it makes the true target
*lower*). Net: essentially all of the SAC→ECMS gap is **RL-side and/or a
structural function-approximation-vs-analytic-optimiser difference**, not a
benchmark artefact.

**V2-Q6 — Best defensible interpretation of the current ECMS number?**
> **3.1887 / 2.8097 is a faithfully reproduced, charge-sustaining (ΔSoC ≤ 0.4
> pp), robust (±10 % λ₀ → < 2 % fuel), constant-λ-plus-SoC-feedback ECMS
> optimum at 81-point action resolution.** It is a *firm, near-optimal
> reference for the constant-λ class* and an *upper bound* on that class's
> achievable fuel at 81 points (the fine-grid value is ≈ 3.167 / 2.799). It is
> a *lower-confidence proxy for the true global optimum* — no DP reference
> exists. It is **not** benchmark-inflated: the whole-cycle λ₀ tuning is worth
> ~1 % of the SAC gap and is handed to SAC anyway.

**V2-Q7 — Is DP currently available as a validated reference?** **No.**
`DP_STATUS = DEFERRED_NO_EXISTING_VALIDATED_SOLVER`. Recorded as a separate
future benchmark task.

---

## FILES

```
results/phase11/v2_ecms_sensitivity.py
results/phase11/data/v2_ecms_sensitivity.json
```
No `src/` file changed. No RL training. Reproduction of the headline ECMS
number is exact (3.1887 / 2.8097).
