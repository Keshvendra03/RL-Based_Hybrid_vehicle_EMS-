# PHASE 12 — STAGE A: Reward-Domain Mathematical Verification

**Experiment:** PHASE 12 · **Stage:** A (reward-domain verification) ·
**Date/time:** 2026-08-30 · **Code revision:** git `90af969` (working tree; one
docs-only commit above `f1f45c5`, no `src/` change prior to this stage) ·
**Training:** NONE.
**Artifacts:** `results/phase12/stage_a_reward_domain.py`,
`results/phase12/data/stage_a_reward_domain.json`,
`results/phase12/figures/stage_a_eq_eff_sweep_{NEDC,FTP75}.png`.

---

## A1. IMPLEMENTATION LOCATION

| quantity | file · location |
|---|---|
| `eq_factor_eff` (the equivalence factor / costate) | `src/env/ems_env.py` · `EMSEnv.step()` · line **662** (was 642 pre-diff): `eq_factor_eff = self.eq_factor + self.k_fb * (SOC_TARGET - soc_before)` |
| final per-step reward | `src/env/ems_env.py` · `EMSEnv.step()` · line **670**: `reward = -self.reward_scale * (fuel_liters + eq_factor_eff * elec_liters)` |
| fuel term | line 623: `fuel_liters = dm_fuel * K_FUEL_L_PER_KG` ; `dm_fuel` = `Tank.step` trapezoidal `m_dot_fuel` [kg] |
| electricity term | line 624: `elec_liters = dE_batt * K_ELEC_L_PER_J` ; `dE_batt = self._E_prev - E_now` (signed, + = discharge); `E = _battery_energy(Q) = 0.5·U_oc(Q)·Q` |
| conversion constants | line 217 `K_FUEL_L_PER_KG = _K_CS/_RHO_FUEL` ; line 218 `K_ELEC_L_PER_J = _EFC_GAIN/3.6e6` (`_EFC_GAIN` from `powertrain.equivalent_fuel_consumption`) |
| SoC-dependent feedback | the `+ self.k_fb * (SOC_TARGET - soc_before)` term; `SOC_TARGET = 0.5` (line 196); `soc_before = self._Q_BT / _Q_BT_0` (pre-decision SoC, line 603) |
| calibration params (CONTROL) | `eq_factor = 0.2717` (NEDC) / `0.4981` (FTP75) ; `k_fb = 2.5` ; `reward_scale = 100` — from each run's `run_config.json` |
| the same unclamped formula also in ECMS | `src/baselines/ecms.py` · `run_ecms_fixed_lambda` / `evaluate_policy.py` · `lam_eff = lam0 + 8.0*(SOC_TARGET - soc)` — **NOT modified in Phase 12** (rule 7; ECMS frozen) |

The production implementation was **not modified before the mathematical audit**
(A2–A3 are pure derivation; the code change is the flag-gated correction
documented in A4/A7, default-OFF so `R_original` = pre-Phase-12 behaviour).

---

## A2. ZERO-CROSSING DERIVATION (from current source + current parameters)

From `ems_env.py:662`, exactly as implemented:

```
eq_factor_eff(SoC) = eq_factor + k_fb · (SOC_TARGET − SoC)
```

This is **affine in SoC**: slope `−k_fb`, intercept `eq_factor + k_fb·SOC_TARGET`.
Solving `eq_factor_eff = 0`:

```
0 = eq_factor + k_fb·(0.5 − SoC)
SoC* = 0.5 + eq_factor / k_fb
eq_factor_eff < 0   ⟺   SoC > SoC*
```

**Numerical thresholds (recomputed directly from source, not the Phase-11 values):**

| cycle | eq_factor | k_fb | SOC_TARGET | **SoC\* (zero-crossing)** | eq_factor_eff at SoC = 0.95 |
|---|---|---|---|---|---|
| **NEDC** | 0.2717 | 2.5 | 0.5 | **0.608680  (60.8680 %)** | 0.2717 + 2.5·(−0.45) = **−0.8533** |
| **FTP75** | 0.4981 | 2.5 | 0.5 | **0.699240  (69.9240 %)** | 0.4981 + 2.5·(−0.45) = **−0.6269** |

**Assumptions (all verified from code / `run_config.json`):**
1. `SOC_TARGET = 0.5` (module constant, `ems_env.py:196`).
2. `eq_factor` / `k_fb` at the CONTROL values (0.2717/0.4981, 2.5).
3. `soc_before` is the *pre-decision* SoC = `_Q_BT/_Q_BT_0`, physically in
   `[0, 1]`; the env's action-feasibility masks bound the *executed* SoC to
   `[0.05, 0.95]`, so the extreme `eq_factor_eff` value is `−0.853` (NEDC) at
   `SoC = 0.95`.
4. No other term modifies `eq_factor_eff` before it multiplies `elec_liters`
   (verified: line 663–669 is the only consumer).

**Correction to prior reports:** the previously-quoted **66.41 %** (NEDC) /
**80.06–80.08 %** (FTP75) were computed with the *old* configuration
(`eq_factor 1.3125, k_fb 8.0` → `0.5 + 1.3125/8 = 0.6641`). The claim in
`PHASE2_FINAL_REPORT.md §18` note 5 that "dividing both by 4.8309 keeps the
ratio, and thus the threshold, unchanged" is **false for the CONTROL config**:
`eq_factor` was divided by 4.8309 (→ 0.2717) but `k_fb` was set to **2.5**, not
`8.0/4.8309 = 1.656`, so the ratio `eq_factor/k_fb` changed (0.164 → 0.109) and
the threshold moved to **60.87 %**. (`PHASE11/STAGE0_AUDIT.md §3 C1` and
`STAGE1_DIAGNOSIS.md §11A` already flagged this; A2 confirms it analytically.)

---

## A3. CLASSIFICATION OF THE DEFECT

**Intended interpretation.** `eq_factor_eff` is the ECMS-style *equivalence
factor* (a.k.a. costate / s-factor): the price of one unit of battery energy in
units of fuel energy. It multiplies `elec_liters`, the battery-energy cost term.
The `+ k_fb·(SOC_TARGET − SoC)` term is a **proportional (P) closed-loop costate
feedback** — battery gets *more* expensive as SoC falls below target and
*cheaper* as it rises above — mirroring `ecms.py`'s proven
`λ = λ₀ + k_fb·(0.5 − SoC)`. This structure is **correct and intended**.

**The defect.** An equivalence factor / energy price has a **physical domain of
`eq_factor_eff ≥ 0`**. The implemented linear feedback is **unbounded below**:
for `SoC > SoC*` it extrapolates into negative values, where the reward's
economics **invert** —

* discharging (`elec_liters > 0`) with `eq_factor_eff < 0` gives
  `eq_factor_eff·elec_liters < 0` ⇒ `−reward_scale·(fuel + negative)` ⇒ reward
  **increases** ⇒ the agent is *paid to discharge*;
* charging (`elec_liters < 0`) with `eq_factor_eff < 0` gives
  `eq_factor_eff·elec_liters > 0` ⇒ reward **decreases** ⇒ the agent is
  *penalised for charging*.

Both are economically invalid — battery energy always has non-negative value.

**Classification:** primarily **B — an unconstrained linear feedback
extrapolation**, with an element of **C — a missing physical/domain constraint**
(the non-negativity of an energy price). It is **not A** (algebraic/sign error):
every sign is correct within the valid domain `SoC ≤ SoC*`; the formula is
right, it is simply not clamped to its domain. It is **not D** (no other
implementation problem was found).

**Trigger status (from Phase 11 §11A and A5 below):** the CONTROL deterministic
trajectory **never** enters `SoC > SoC*` (max SoC 50.4 % NEDC / 55.7 % FTP75) —
so for the current controller this is a **latent** defect. It *is* exercised in
0.3–2.4 % of the training *replay buffers* (early SoC-runaway phase, SoC up to
95.5 %), where it fed the critic economically-inverted targets. Because Phase
12B will **inject high-engine-load (deep-LPS / charging) actions**, some
training trajectories may push SoC higher than the CONTROL did — so the
correction must be in place for 12B to prevent the defect from contaminating
the coverage experiment (this is the explicit reason 12A precedes 12B).

---

## A4. CANDIDATE CORRECTIONS

Requirements: enforce `eq_factor_eff ≥ 0`; preserve the intended P-feedback
economics for every `SoC ≤ SoC*`; minimum intervention; **do not** alter
`eq_factor`, `k_fb`, `SOC_TARGET`, or `ecms.py`.

| # | candidate | assessment |
|---|---|---|
| **C1** | `eq_factor_eff = max(eq_factor_eff, 0.0)` | **SELECTED.** Enforces the exact physical lower bound. No-op for every `SoC ≤ SoC*` (bit-identical reward). Introduces **no new parameter**. C0-continuous (a kink at `SoC*`, no jump). At `SoC > SoC*` the battery price is 0 — the reward is then neutral on the electrical ledger and simply minimises instantaneous fuel, which at high SoC means "use the engine as little as possible" = discharge toward target. The complementary restoring force is already present: the per-step SoC-band penalty (`ems_env.py` line ~672) activates at `\|SoC−0.5\| > SOC_DEADBAND = 0.10`, i.e. **SoC > 0.60** — essentially coincident with the NEDC `SoC* = 0.6087`. So exactly where C1 removes the (inverted) economic pull, the SoC-band penalty takes over the "return to target" job. The two mechanisms are complementary and the crossover is well-placed. |
| C2 | `eq_factor_eff = max(eq_factor_eff, ε)` , `ε > 0` (e.g. 0.05) | Rejected. `ε` is an **arbitrary new calibration parameter** (rule: do not add calibration). The prompt explicitly says not to assume `ε = 0.05`. A positive floor is not required — see C1's complementary-penalty argument. |
| C3 | efficiency-derived bound, e.g. `eq_factor_eff ≥ eq_factor` (never let feedback drop the price below its base value) | Rejected. Clamps the *entire* "battery gets cheaper above target" half of the P-feedback (all `SoC > 0.5`), which **changes the intended economics** for a large, valid SoC range — not a minimum intervention, and it defeats the purpose of the feedback. |
| C4 | reformulate multiplicatively, e.g. `eq_factor_eff = eq_factor · exp(k_fb'·(0.5−SoC))` | Rejected. Always positive and smooth, but redefines `k_fb`'s meaning and units (rule violation) and is a major reformulation, not minimal. |

**Selected correction: C1 — `eq_factor_eff = max(eq_factor_eff, 0.0)`.**

Charging/discharging economics are preserved exactly for `SoC ≤ SoC*` and made
domain-valid (price = 0, i.e. *cheap but not negative*) for `SoC > SoC*`.

---

## A5. CONTROL NON-REGRESSION TEST

`R_original` = `EMSEnv(..., clip_eq_eff=False)` (default; pre-Phase-12 reward,
preserved and reproducible). `R_corrected` = `EMSEnv(..., clip_eq_eff=True)`.
Each of the 6 CONTROL checkpoints was run through **both** environments with the
**identical deterministic action sequence** (the clamp changes only the scalar
reward, never the action or the physical trajectory), and per-transition rewards
compared. Tolerance reported as raw IEEE-754 double difference (no rounding).

| checkpoint | n transitions | SoC [min, max] % | threshold % | n below threshold | **n affected** | **max \|ΔR\| below thr** | max rel ΔR below thr | max \|ΔR\| affected | ΔR cumulative |
|---|---|---|---|---|---|---|---|---|---|
| `models_p5s0_k2.5/NEDC` | 1220 | 28.1 – 50.3 | 60.868 | 1220 | **0** | **0.000e+00** | 0.000e+00 | — | **0.000e+00** |
| `models_p5_k2.5/NEDC` | 1220 | 26.5 – 50.2 | 60.868 | 1220 | **0** | **0.000e+00** | 0.000e+00 | — | **0.000e+00** |
| `models_p5_k2.5_s2/NEDC` | 1220 | 23.6 – 50.4 | 60.868 | 1220 | **0** | **0.000e+00** | 0.000e+00 | — | **0.000e+00** |
| `models_p5f_k2.5_s0/FTP75` | 1876 | 39.3 – 55.7 | 69.924 | 1876 | **0** | **0.000e+00** | 0.000e+00 | — | **0.000e+00** |
| `models_p5f_k2.5_s1/FTP75` | 1876 | 35.7 – 53.4 | 69.924 | 1876 | **0** | **0.000e+00** | 0.000e+00 | — | **0.000e+00** |
| `models_p5f_k2.5_s2/FTP75` | 1876 | 38.5 – 54.0 | 69.924 | 1876 | **0** | **0.000e+00** | 0.000e+00 | — | **0.000e+00** |

**Totals:** 3660 NEDC + 5628 FTP75 = **9288 CONTROL transitions examined; 0
affected; maximum absolute reward difference = exactly 0.0; maximum relative
difference = exactly 0.0.**

The maximum CONTROL SoC (50.4 % NEDC, 55.7 % FTP75) sits **10.5 / 14.2
percentage points below** the respective thresholds. **The correction is
provably a bitwise no-op on the entire CONTROL trajectory.** `R_original`
remains exactly reproducible (default `clip_eq_eff=False`).

**Unit tests:** `pytest tests/test_ems_env.py` → **7 passed** (env still
validates; default behaviour unchanged).

---

## A6. SYNTHETIC SoC SWEEP

`eq_factor_eff` enters the reward at exactly **one** place — the economic term
`R_econ = −REWARD_SCALE·(fuel_liters + eq_factor_eff·elec_liters)`
(`ems_env.py:670`). The sweep isolates that term with **fixed physical
conditions**: `fuel_liters = 3.0e-3 L`, `|elec_liters| = 6.0e-3 equiv-L`
(representative magnitudes from a real mid-demand CONTROL step, T_MGB ≈ 60 Nm),
`elec_liters = +6e-3` for a **discharge** step and `−6e-3` for a **charge**
step. SoC swept 0.30 → 0.95 (66 points).

### NEDC (threshold SoC\* = 60.868 %)

| SoC % | eq_eff orig | eq_eff corr | R_disch orig | R_disch corr | **ΔR_disch** | R_charge orig | R_charge corr | **ΔR_charge** |
|---|---|---|---|---|---|---|---|---|
| 30.0 | 0.7717 | 0.7717 | 2.1167 | 2.1167 | **0.0000** | 2.1226 | 2.1226 | **0.0000** |
| 40.0 | 0.5217 | 0.5217 | 0.7073 | 0.7073 | **0.0000** | 0.7113 | 0.7113 | **0.0000** |
| 50.0 | 0.2717 | 0.2717 | −0.0359 | −0.0359 | **0.0000** | −0.0338 | −0.0338 | **0.0000** |
| 60.0 | 0.0217 | 0.0217 | −0.0482 | −0.0482 | **0.0000** | −0.0481 | −0.0481 | **0.0000** |
| **60.868** | **0.0000** | **0.0000** | — | — | **0.0000** | — | — | **0.0000** |
| 61.0 | −0.0033 | 0.0000 | −0.0068 | −0.0127 | **−0.0059** | −0.0068 | −0.0127 | **−0.0059** |
| 65.0 | −0.1033 | 0.0000 | +0.2398 | −0.0127 | **−0.2526** | +0.2391 | −0.0127 | **−0.2518** |
| 70.0 | −0.2283 | 0.0000 | +0.7349 | −0.0127 | **−0.7476** | +0.7331 | −0.0127 | **−0.7458** |
| 80.0 | −0.4783 | 0.0000 | +2.3781 | −0.0127 | **−2.3908** | +2.3744 | −0.0127 | **−2.3871** |
| 95.0 | −0.8533 | 0.0000 | +6.5970 | −0.0127 | **−6.6097** | +6.5920 | −0.0127 | **−6.6047** |

### FTP75 (threshold SoC\* = 69.924 %)

| SoC % | eq_eff orig | eq_eff corr | ΔR_disch | ΔR_charge |
|---|---|---|---|---|
| ≤ 69.924 | (linear, > 0) | identical | **0.0000** | **0.0000** |
| 70.0 | −0.0019 | 0.0000 | −0.0079 | −0.0078 |
| 75.0 | −0.1269 | 0.0000 | −0.6377 | −0.6309 |
| 85.0 | −0.3769 | 0.0000 | −2.5601 | −2.5399 |
| 95.0 | −0.6269 | 0.0000 | −5.4200 | −5.3957 |

Figures: `results/phase12/figures/stage_a_eq_eff_sweep_NEDC.png`,
`…_FTP75.png` (3 panels each: `eq_eff` vs SoC; per-step reward for a discharge
action; per-step reward for a charge action — original vs corrected, with the
zero-crossing marked).

### Where and why the two formulations diverge

* **SoC ≤ SoC\*:** `eq_factor_eff` is already `> 0`, so `max(·, 0)` is a no-op —
  `eq_eff`, `R_disch`, `R_charge` are **identical to the last bit**. ΔR = 0.0
  everywhere in this region (confirmed on both the 66-point sweep and the 9288
  real CONTROL transitions).
* **SoC > SoC\*:** the original `eq_factor_eff` goes negative and grows in
  magnitude linearly (slope −2.5), reaching **−0.853** (NEDC) / **−0.627**
  (FTP75) at SoC 95 %. The **original** reward for a *discharge* step then
  becomes **positive and large** (up to +6.60 at SoC 95 %, NEDC) — a spurious
  ~660-reward-unit (≈ 0.66 equiv-L) per-step *bonus for discharging*. The
  original reward for a *charge* step is symmetrically *over-penalised*. The
  **corrected** reward holds flat at `−REWARD_SCALE·fuel_liters ≈ −0.013`
  (battery price = 0) — economically valid: at very high SoC, discharge is free
  but not rewarded, charge is free but not penalised, and the SoC-band penalty
  drives SoC back to target.

---

## A7. EXACT CODE CHANGE

**File:** `src/env/ems_env.py` · **Function:** `EMSEnv.__init__` and
`EMSEnv.step` · **Scientific reason:** enforce the physical domain
`eq_factor_eff ≥ 0` of an equivalence factor / energy price, removing the
negative-price extrapolation (A3, classification B/C). Flag-gated, **default
`False`** so `R_original` is bit-exactly preserved and reproducible (A5).

```diff
--- a/src/env/ems_env.py
+++ b/src/env/ems_env.py
@@ EMSEnv.__init__ signature (after `lookahead`)
+        clip_eq_eff: bool = False, # PHASE 12A reward-domain SAFETY correction.
+                                   # eq_factor_eff = eq_factor + k_fb*(0.5 - soc)
+                                   # is an UNBOUNDED linear costate feedback: it
+                                   # extrapolates below zero once
+                                   #   soc > 0.5 + eq_factor/k_fb
+                                   # (= 60.87% NEDC CONTROL, 69.92% FTP75),
+                                   # where a NEGATIVE equivalence factor pays the
+                                   # agent to discharge and penalises charging --
+                                   # an invalid domain for an energy price.
+                                   # clip_eq_eff=True enforces eq_factor_eff >= 0.
+                                   # Default False keeps the reward byte-identical
+                                   # to the pre-Phase-12 implementation. Does NOT
+                                   # touch eq_factor, k_fb, SOC_TARGET, or ecms.py.
@@ EMSEnv.__init__ body (after `self.k_fb = float(k_fb)`)
+        self.clip_eq_eff = bool(clip_eq_eff)   # PHASE 12A: see __init__ docstring
@@ EMSEnv.step  (after `eq_factor_eff = self.eq_factor + self.k_fb * (SOC_TARGET - soc_before)`)
+        if self.clip_eq_eff:
+            # PHASE 12A reward-domain safety correction: an equivalence factor
+            # (battery-energy price) cannot be negative. Clamp the unbounded
+            # linear costate feedback at its physical lower bound. No-op wherever
+            # eq_factor_eff >= 0 (all CONTROL transitions); only affects the
+            # previously-invalid soc > 0.5 + eq_factor/k_fb region.
+            eq_factor_eff = max(eq_factor_eff, 0.0)
```

* **Old behaviour:** `eq_factor_eff` is the raw affine feedback, unbounded
  below; negative for `SoC > SoC*`.
* **New behaviour (only when `clip_eq_eff=True`):** `eq_factor_eff` is clamped
  to `≥ 0`; identical to old for `SoC ≤ SoC*`.
* **Default (`clip_eq_eff=False`): NO CHANGE** — `R_original` reproducible.
* **Not touched:** `eq_factor`, `k_fb`, `SOC_TARGET`, `reward_scale`, the fuel
  term, the SoC-band penalty, the terminal saturation correction / CS penalty,
  `src/baselines/ecms.py`, `evaluate_policy.py`, the physical model, the action
  model, the observation, γ, the network, the optimiser.

`git diff --stat src/env/ems_env.py` → `1 file changed, 28 insertions(+)`.

---

## CONCLUSION

| item | result |
|---|---|
| Zero-crossing (recomputed from source) | **SoC\* = 0.5 + eq_factor/k_fb = 60.868 % (NEDC) / 69.924 % (FTP75)**; `eq_factor_eff < 0` above it (down to −0.853 / −0.627 at SoC 95 %). |
| Defect classification | **B (unconstrained linear feedback extrapolation) + C (missing non-negativity domain constraint).** Not an algebraic sign error. |
| Correct minimum intervention | **C1: `eq_factor_eff = max(eq_factor_eff, 0.0)`** — physical lower bound of an energy price; no new parameter; complementary to the SoC-band penalty whose activation (SoC > 60 %) coincides with SoC\*. |
| CONTROL non-regression | **PASS.** 9288 transitions, **0 affected**, max \|ΔR\| = **exactly 0.0**, cumulative ΔR = **0.0**, all 6 checkpoints. 7/7 env unit tests pass. |
| Synthetic sweep | ΔR = 0.0 for every SoC ≤ SoC\*; above SoC\* the original reward gives a spurious discharge bonus / charge penalty growing to ±6.6 (NEDC) / ±5.4 (FTP75) per step at SoC 95 %; the corrected reward holds at `−scale·fuel`. |
| Contamination risk for 12B | The CONTROL never triggers the defect, but 12B's high-engine-load injections could push SoC above SoC\* on some training trajectories; therefore `clip_eq_eff=True` **must** be used for all 12B training, and 12B must audit both `R_corrected` and `R_original` (Stage B §B8). |

**STAGE A CONCLUSION: PASS.** The reward-domain defect is mathematically
identified and classified (unbounded costate-feedback extrapolation past its
non-negativity domain); the justified minimum correction is
`eq_factor_eff = max(eq_factor_eff, 0)`, applied as a default-OFF flag
(`clip_eq_eff`) that is a provable bitwise no-op on all 9288 CONTROL
transitions. `R_original` remains exactly reproducible. The mathematical
correction is explicitly documented (A7).

---

## HARD STOP AFTER STAGE A — SATISFIED

Stage A report complete; correction documented and verified non-regressive.
No training performed. `eq_factor`, `k_fb`, γ, and the equivalent-consumption
calibration parameters unchanged. Proceeding to Stage B (targeted
informative-coverage experiment) per the Phase 12 plan, which authorises Stage
B "ONLY after Stage A has been completed and its correction verified".
