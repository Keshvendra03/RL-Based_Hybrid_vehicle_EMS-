# Phase 5B — Forensic Closure Audit

**No training performed.** No hyperparameter, reward, environment, benchmark
or ECMS change. Analysis only, closing the evidence gaps left by Phase 5.

Raw output: `results/phase5b/phase5b_forensics.txt`
Figures: `results/phase5b/q_landscape_NEDC.png`, `q_landscape_FTP75.png`

---

## 1. Objective

Determine conclusively whether the residual SAC-vs-benchmark gap is caused by
(A) replay/action coverage, (B) actor optimisation, (C) critic misestimation,
(D) SoC/equivalent-factor valuation, (E) action representation, or
(F) engine operating-point optimisation.

---

## 2/3. Replay-buffer forensics — **measured, not inferred**

NEDC candidate (`gated, k_fb=2.5`), 150,016 stored transitions:

| band | count | % of buffer | OFF | ASSIST | LPS | OFF feasible |
|---|---|---|---|---|---|---|
| 0-15 | 66,153 | 44.1% | 25.4% | 3.6% | 5.9% | 100% |
| **15-30** | 27,798 | 18.5% | **44.9%** | 19.0% | 36.1% | 100% |
| **30-35** | 14,391 | 9.6% | **15.6%** | 20.0% | 64.4% | 100% |
| 35-50 | 5,043 | 3.4% | 33.3% | 22.9% | 43.8% | 100% |
| 50-75 | 9,840 | 6.6% | 5.5% | 23.0% | 71.4% | 100% |
| >75 | 8,610 | 5.7% | 0.0% | 16.0% | 84.0% | 0% |

### **This REFUTES the Phase-5 §O claim.**

Phase 5 asserted that σ-collapse "means the buffer receives almost no OFF
transitions there." **That was an inference and it is wrong.** The buffer
holds ~2,245 OFF transitions in 30-35 Nm and ~12,480 in 15-30 Nm. Coverage is
substantial.

### But conditioning on SoC reveals the real gap

| band | SoC band | count | OFF% | ASSIST% | LPS% |
|---|---|---|---|---|---|
| 15-30 | SoC<40 | 14,321 | 45.7% | 19.3% | 35.0% |
| 15-30 | 40-50 | 13,091 | 44.1% | 18.3% | 37.6% |
| 15-30 | ≥50 | 386 | 47.2% | 30.8% | 22.0% |
| **30-35** | **SoC<40** | 8,091 | **24.4%** | 24.1% | 51.6% |
| **30-35** | **40-50** | 6,132 | **4.5%** | 15.0% | 80.6% |
| **30-35** | **≥50** | 168 | **3.0%** | 6.0% | 91.1% |
| 35-50 | SoC<40 | 1,963 | 33.2% | 21.3% | 45.4% |
| 35-50 | 40-50 | 2,843 | 34.2% | 23.7% | 42.1% |

The deterministic policy operates at **SoC 40-50%**. In exactly that region,
30-35 Nm OFF coverage is **4.5% (~276 transitions of 6,132)**. The OFF data
that exists is concentrated at **SoC<40**, i.e. states visited during the early
SoC-runaway phase, not at the operating point.

### Diagnosis → **CASE R3**

> *replay contains OFF data but mostly at incorrect SOC states.*

Not R2 (coverage is not absent), not R1 in the naive sense. `Q(OFF | 30-35 Nm,
SoC 40-50)` is estimated from ~276 samples — a **conditional** coverage hole.

---

## 4. Matched before/after `k_fb` (1.656 → 2.5), identical states

| band | actor mean | actor σ | P(OFF) |
|---|---|---|---|
| | k=1.656 → k=2.5 | k=1.656 → k=2.5 | k=1.656 → k=2.5 |
| 15-30 | −0.247 → **−0.398** | 0.385 → 0.295 | 31.1% → **22.9%** |
| 30-35 | −0.549 → **−0.802** | 0.308 → 0.419 | 0.2% → 0.4% |
| 35-50 | −0.054 → **−0.424** | 0.407 → 0.411 | 36.2% → **20.3%** |

| band | ΔQ(OFF−ASSIST) | ΔQ(OFF−LPS) | Δr(OFF−ASSIST) |
|---|---|---|---|
| 15-30 | +0.0003 → **−0.0062** | −0.0105 → **+0.0010** | +0.0008 → +0.0011 |
| 30-35 | −0.0016 → **−0.0222** | −0.0217 → **+0.0054** | +0.0001 → +0.0001 |
| 35-50 | +0.0036 → +0.0023 | −0.0138 → −0.0135 | +0.0005 → +0.0007 |

### What `k_fb=2.5` actually changed

1. It **re-ranked OFF above LPS in the critic** (ΔQ(OFF−LPS) flips negative →
   positive in 15-30 and 30-35). This is the intended economic correction and
   it is exactly why SoC stopped running away — LPS over-charging is no longer
   the critic's preferred escape.
2. But it **pushed the actor mean further toward LPS in every band**
   (−0.247→−0.398, −0.549→−0.802, −0.054→−0.424) and **reduced P(OFF)**.
3. And it **worsened ΔQ(OFF−ASSIST)** (−0.0062, −0.0222).
4. Meanwhile **Δr(OFF−ASSIST) stayed positive and essentially unchanged**
   (+0.0008 → +0.0011). **The true reward still prefers OFF; the critic does
   not.**

**So `k_fb` fixed the SoC mechanism (OFF vs LPS) while degrading the OFF-vs-
ASSIST valuation and displacing the actor.** That is precisely why SoC became
3/3 charge-sustaining while fuel stayed tied.

---

## 5. Q(a) + actor-density figures

Produced for 15-30 / 30-35 / 35-50 Nm on both cycles, overlaying actor mean,
±1σ, ±2σ, actor density, LPS/ASSIST/OFF regions and argmax Q:
`results/phase5b/q_landscape_NEDC.png`, `q_landscape_FTP75.png`.

---

## 6. Actor-vs-critic classification (distribution, n=120 states each)

`action_distance = |actor_mean − argmax_a Q(a)| / action_range`

| config | mean | median | p90 | A aligned | B displaced | C Q prefers non-OFF | D flat |
|---|---|---|---|---|---|---|---|
| NEDC k=1.656 | 0.149 | 0.079 | 0.420 | 60.8% | 21.7% | 17.5% | **0.0%** |
| **NEDC k=2.5 (candidate)** | **0.295** | 0.093 | 0.868 | 51.7% | **37.5%** | 10.8% | **0.0%** |
| **FTP75 k=1.656 (best result)** | 0.154 | 0.042 | 0.657 | **75.8%** | **11.7%** | 12.5% | **0.0%** |

Two decisive observations:

- **`D_flat = 0.0%` everywhere.** The critic is *not* uninformative. It has
  clear action preferences. "Insufficient critic resolution" is eliminated.
- **Actor-critic alignment tracks performance.** The best configuration
  (FTP75, at the benchmark) is the most aligned (75.8% A, 11.7% B). The NEDC
  candidate is the *least* aligned — `k_fb=2.5` **doubled** the mean
  displacement (0.149 → 0.295) and grew CASE B from 21.7% to **37.5%**.

---

## 7. Costate percentile forensics (NEDC)

| k_fb | eq min / p5 / median / mean / p95 / max | ECMS units (×4.8309) | time priced above ECMS λ₀ |
|---|---|---|---|
| 1.656 | 0.083 / 0.242 / 0.526 / 0.499 / 0.756 / 0.887 | 0.40 / 1.17 / 2.54 / 2.41 / 3.65 / 4.29 | **91.1%** |
| 2.5 | 0.265 / 0.272 / 0.578 / 0.543 / 0.670 / 0.820 | 1.28 / 1.31 / 2.79 / 2.62 / 3.24 / 3.96 | **95.3%** |

**Answer to the §7 question — `k_fb=2.5` does BOTH, but predominantly
suppresses.** It genuinely improves the *floor* (p5 rises 1.17 → 1.31 ECMS
units, so the battery is never absurdly cheap, removing the over-discharge
escape). But the median price sits at **2.79 ECMS units — 2.1× ECMS's proven
λ₀=1.3125 — and 95.3% of the episode is priced above λ₀.** Battery energy is
systematically over-priced relative to the proven optimum, which discourages
engine-OFF (OFF *requires* discharge). This is the economic reason ΔQ(OFF−ASSIST)
worsened.

---

## 8. Error budget, all three configurations (NEDC, ΔFuel vs rule-based)

| config | brake | 0-15 | 15-30 | 30-50 | 50-75 | >75 | TOTAL |
|---|---|---|---|---|---|---|---|
| linear ref | +0.0002 | −0.0052 | +0.3502 | **+0.5434** | −0.2937 | −0.3351 | +0.2598 |
| gated k=1.656 | +0.0030 | −0.0123 | +0.3431 | **+0.1318** | −0.1756 | **+0.1222** | +0.4121 |
| **gated k=2.5** | +0.0006 | −0.0122 | +0.3677 | +0.3705 | −0.2317 | **−0.3142** | **+0.1807** |

**Regional source of the k_fb gain, explicitly:** k=2.5 did **not** improve
30-50 Nm relative to k=1.656 (it got *worse*, +0.1318 → +0.3705). Its entire
net gain comes from **restoring the >75 Nm advantage** (+0.1222 → −0.3142, a
swing of 0.436) that the ungated/under-damped configuration had destroyed by
running LPS 100% of the time there. Total improves to the best value seen
(+0.1807), but the low-torque deficit is **untouched**: 15-30 Nm remains
+0.3677 in every configuration.

---

## 9. Combined Pareto table

| Configuration | k_fb | NEDC mean (n=3) | FTP75 mean (n=3) | NEDC CS | FTP75 CS | OFF% | ASSIST% | LPS% | ΔSoC NEDC | ΔSoC FTP75 |
|---|---|---|---|---|---|---|---|---|---|---|
| linear reference | 1.656 | **3.7727 ± 0.0281** | 3.3821 ± 0.0846 | 3/3 | 3/3 | 35.3 | 24.8 | 22.9 | +0.15 | — |
| gated baseline | 1.656 | 3.8824 ± 0.1371 | **3.2460 ± 0.0434** | **1/3** | 3/3 | 41.6 | 8.9 | ~31 | +11.4/+8.4/−1.8 | +0.62/−2.00/−0.64 |
| **gated candidate** | **2.5** | 3.7666 ± 0.0785 | 3.2889 ± 0.0174 | **3/3** | 3/3 | 38.6 | 14.6 | ~29 | +0.28/−0.72/+0.23 | −0.53/−0.51/−0.07 |
| gated | 3.0 | 3.7840 ± 0.0483 | — | 3/3 | — | 39.2 | 10.3 | ~30 | +0.98/+1.21/+0.73 | — |
| advanced rule-based | — | 3.5056 | 3.2323 | — | — | 59.0 | 0.0 | 23.8 | +2.47 | +3.86 |
| ECMS | — | 3.1887 | 2.8097 | — | — | 53.1 | 0.2 | 29.7 | +0.36 | +0.13 |

No seed cherry-picked; all three seeds shown for ΔSoC.

---

## 10. FTP75 matched-state: SAC vs rule-based vs ECMS

| controller | V_CE_equiv | SoC final | ΔSoC | OFF | ASSIST | LPS | REGEN | engine-ON | mean\|T_CE\| | mean\|T_EM\| |
|---|---|---|---|---|---|---|---|---|---|---|
| **SAC (gated k=1.656)** | **3.2356** | 50.62% | **+0.62pp** | 44.3 | 3.9 | 26.1 | 25.7 | 444 s | 25.3 | 21.3 |
| advanced rule-based | 3.2323 | 53.86% | +3.86pp | 46.3 | 0.4 | 22.4 | 25.7 | 414 s | 22.9 | 20.2 |
| ECMS | **2.8097** | 50.13% | +0.13pp | 40.4 | 6.0 | 27.9 | 25.7 | 501 s | 24.1 | 21.9 |

SAC is **+0.10% from the rule-based benchmark** on this seed and is *better
charge-balanced* than it (+0.62 vs +3.86pp).

### SAC − ECMS gap decomposition (FTP75)

| region | ΔFuel | ΔElec | SAC OFF% | ECMS OFF% |
|---|---|---|---|---|
| braking | +0.0005 | +0.1022 | — | — |
| 0-15 | +0.0202 | +0.0783 | 99.6% | 54.6% |
| 15-30 | **+0.0950** | −0.0009 | 84.1% | 89.4% |
| **30-50** | **+0.1429** | −0.0125 | 51.2% | 59.5% |
| **50-75** | **+0.1151** | −0.1181 | 2.7% | **12.4%** |
| >75 | +0.0521 | −0.0532 | 0.0% | 0.0% |

**The SAC−ECMS gap is broadly distributed, not concentrated**: 30-50 Nm
(+0.143), 50-75 Nm (+0.115) and 15-30 Nm (+0.095) each contribute comparably.
Notably ECMS uses **more** engine-ON time (501 s vs 444 s) yet burns less fuel
— so its advantage is **engine operating-point quality**, not mode selection.
Two concrete signatures: ECMS runs OFF only 54.6% at 0-15 Nm where SAC runs
99.6% (SAC over-uses EV at trivial loads, then must recharge), and ECMS keeps
12.4% OFF at 50-75 Nm where SAC has 2.7%.

**Not forced equivalence (§11):** SAC's deviation at 0-15 Nm is *harmful*
(+0.0202 fuel and +0.0783 elec — it pays twice). Its deviation at braking is
neutral. So the differences are diagnostic, not stylistic.

---

## 13. Generalization inventory (inspection only, no training)

Legitimately variable **today**, without touching validated physics:

| lever | status | scientific value |
|---|---|---|
| `cycle_name` (NEDC ↔ FTP75) | **available** — both CSVs present | already used; the cross-cycle test |
| `DrivingCycle(dt=…)` | available (constructor arg) | **low** — changes the plant's time base; would invalidate the 1 s-step validation |
| `lookahead` | available | preview-horizon sensitivity, not a physical shift |
| `eq_factor`, `k_fb`, `lambda_soc`, `soc_deadband` | available | reward-side, not a condition shift |
| **initial SoC** | **NOT a parameter** — `_Q_BT_IC` is hard-coded at 50% in `powertrain.py`, and `Battery.reset()`/`EMSEnv.reset()` both use it | **highest value**, but requires a code change |
| speed scaling / traffic variation / accessory load / temperature | **NOT SUPPORTED** — no such parameters exist in `params.json` or the env | would require inventing physics — **prohibited** |

**Recommendation:** the only *physically meaningful* shifted-condition test the
project can support without inventing physics is **varied initial SoC**
(e.g. 40% / 60%), which needs a small, clearly-scoped change to make
`_Q_BT_IC` an env parameter. Cross-cycle evaluation (train NEDC → test FTP75
and vice versa) is available **now** at zero cost and should be done first.
Do not attempt speed/temperature/accessory shifts — they are not modelled.

---

## 14. Bottleneck classification

Applying the §14 decision tree to the measurements:

- Replay OFF coverage **adequate in aggregate** (15.6% at 30-35 Nm) but
  **conditionally deficient at the operating SoC** (4.5% at SoC 40-50) → R3.
- Critic is **not flat** (D_flat = 0.0%).
- Critic **does not favour OFF over ASSIST** (ΔQ = −0.0062 / −0.0222) while the
  **true reward does** (Δr = +0.0011). → critic misestimation.
- Actor is **displaced from its own critic in 37.5% of states**, and `k_fb=2.5`
  doubled that displacement.

### **PRIMARY BOTTLENECK: critic misestimation of `Q(OFF)` at the operating SoC, caused by a conditional replay-coverage hole — with actor displacement as a secondary, `k_fb`-induced effect.**

This **supersedes the Phase-5 "bimodal-Q / unimodal-policy" conclusion as the
primary explanation.** The bimodality observed in Phase 5 is real but is a
*symptom*: the OFF lobe of Q is fitted from ~276 relevant samples, so its
height is unreliable. It is not evidence that a unimodal policy class is
fundamentally inadequate.

Also note: `eq` median sits at **2.79 ECMS units vs λ₀=1.3125**, i.e. battery
is systematically over-priced — an independent contributor pushing Q(OFF) down.

---

## 15. Unresolved questions

1. Is the 15-30 Nm deficit (+0.3677, unchanged across every configuration)
   caused by the same conditional-coverage mechanism, or by engine
   operating-point quality (as the FTP75-vs-ECMS decomposition suggests)?
2. Would `eq_factor`/`k_fb` re-scaled so the *median* price lands near ECMS λ₀
   (rather than 2.1× it) restore ΔQ(OFF−ASSIST) without reopening SoC drift?
3. Does the NEDC/FTP75 divergence reduce to regen availability (17.0% vs
   25.7%) driving different SoC occupancy, hence different coverage holes?

---

## 16. Recommendation — ONE next intervention

### **Restore OFF-region critic coverage at the operating SoC, via a targeted exploration schedule — NOT a reward, gamma, entropy or architecture change.**

**Evidence that this is the correct change:**

1. **The reward is already right**: Δr(OFF−ASSIST) = **+0.0011 > 0** at the
   matched states — OFF is genuinely economically better there.
2. **The critic disagrees with the reward**: ΔQ(OFF−ASSIST) = **−0.0062 to
   −0.0222** — a value-estimation error, not an economics error.
3. **The cause of that error is measured**: only **4.5% (~276) of the 6,132**
   replay transitions at 30-35 Nm / SoC 40-50 contain OFF. The critic cannot
   fit what it has not seen at the states that matter.
4. **The critic is not flat** (D_flat = 0.0%), so it *can* represent the
   distinction once given data — this is not a capacity limit.
5. **Alignment predicts performance**: FTP75, the only benchmark-level result,
   is also the most actor-critic-aligned (75.8% A / 11.7% B). NEDC's candidate
   is the least (51.7% / 37.5%).
6. **Entropy is the wrong lever here**: `k_fb=2.5` already *raised* σ at
   30-35 Nm (0.308 → 0.419) and P(OFF) still stayed at 0.4%, because the actor
   *mean* moved away faster (−0.549 → −0.802). Widening a distribution whose
   mean is being pushed away does not fix coverage.

**Proposed experiment (to be authorised separately):** a controlled,
documented exploration schedule that guarantees feasible-OFF action coverage
in 15-35 Nm **at SoC 40-55%** during early training — compared head-to-head
against normal SAC with all other variables identical, per §21/§22. Measure:
replay OFF-share at the operating SoC, ΔQ(OFF−ASSIST), actor P(OFF), and
finally fuel. This is a *coverage* intervention, not imitation: no benchmark
or ECMS action is used as a label (§23 respected).

**Do not start it.** Authorisation required.
