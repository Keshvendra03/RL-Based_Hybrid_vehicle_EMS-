# Phase 4 — Final Report

Root-cause analysis of the residual SAC-vs-benchmark gap and the single
targeted intervention. Does **not** supersede Phase-1/2/3 conclusions.
Raw forensics output: `results/phase4/forensics_NEDC.txt`.
Plots: `results/phase4/reward_counterfactual_NEDC.png`,
`results/analysis/{q_landscape,policy_law}_NEDC.png`.

---

## A. Gamma decision (FROZEN)

Two reference configurations are retained and **never mixed**:

| ref | gamma | horizon | role |
|---|---|---|---|
| **REF-SHORT** | 0.20 | 1.25 s | current performance reference |
| **REF-INTERMEDIATE** | 0.90 | 10 s | physically-meaningful intermediate horizon |

`gamma = 0.90` is retained deliberately (not because it is optimal — it is
not, 3.8795 vs 3.7775) so that a moderate temporal horizon can be re-tested
once the policy problem is fixed. **The gamma investigation is closed.**

## B. Current best SAC (pre-intervention)

`gamma 0.20, n_step 1, linear map, eq_factor 0.2717/0.4981, k_fb 1.656,
gradient_steps 16, lookahead 5, buffer 300k, batch 512, lr 3e-4, [256,256]`

| cycle | mean (n=3) | std | rule-based | gap |
|---|---|---|---|---|
| NEDC | 3.7727 | 0.0281 | 3.5056 | +7.6% |
| FTP75 | 3.3821 | 0.0846 | 3.2323 | +4.6% |

## C. Policy-boundary diagnosis (30-50 Nm)

**Exploration deadlock.** The probability mass the actor's own Gaussian places
in the engine-OFF region predicts its OFF usage almost exactly:

| band | P(OFF) under policy | actual SAC OFF% |
|---|---|---|
| 15-30 Nm | 54.4% | **53.1%** |
| 30-50 Nm | **3.6%** | **0%** |

Under the linear map, OFF at 30-50 Nm sits **+3.87σ (γ=0.20) / +6.71σ
(γ=0.90)** from the actor mean. The chain: never proposed → no OFF data in the
replay buffer at those torques → critic cannot learn Q(OFF) → no gradient
pulls the actor there → self-reinforcing.

**Sub-band split (section 7)** — the divergence is really **15-35 Nm**, not the
whole 30-50 band. At 30-35 Nm the benchmark uses OFF 84.6%, SAC 0%
(feasibility 100%). Above 35 Nm the benchmark *also* abandons OFF.

## D. Q-value diagnosis (section 8)

| | REF-SHORT (γ=0.20) | REF-INTERMEDIATE (γ=0.90) |
|---|---|---|
| ΔQ(OFF−ASSIST) | −0.0071 (>0 in 30%) | **+0.0020 (>0 in 60%)** |
| ΔQ(OFF−LPS) | −0.0158 (>0 in 15%) | +0.0013 (>0 in 65%) |
| sign(ΔQ)==sign(Δr) | **20%** | **85%** |
| **CASE (section 17)** | **C** — critic bias | **A** — actor fails to select |

At γ=0.90 the reward says OFF, the critic **agrees**, and the actor still
chooses ASSIST 62% / LPS 35%. **Both** references refuse OFF → per section 19,
**gamma is not the root cause.**

## E. Reward diagnosis (section 9)

Immediate reward **favours OFF** in the target band:
Δr(OFF−ASSIST) = **+0.0047** (γ=0.20) / **+0.0084** (γ=0.90), positive in 55%
of states; heatmap mean over 30-50 Nm = **+0.00414**. The reward is correct.

## F. Costate diagnosis (section 11)

`eq_factor(SoC) = 0.2717 + 1.656·(0.5 − SoC)`, `d(eq)/dSoC = −1.656` (constant).
Over the visited SoC (28.6-50.2%) it spans **0.2685-0.6255**, i.e. **1.297-3.022
in ECMS units** (ECMS λ₀ = 1.3125) — an **80%** price variation.

**This explains why γ=0.90 does not beat γ=0.20**: the only inter-temporal
coupling (SoC) is already supplied explicitly per-step, so a long-horizon value
function adds variance without adding information.

## G. Error budget (section 13, timestep-aligned; demand verified controller-independent, max|ΔT| = 0.0)

| region | time% | ΔFuel (SAC−RB) | ΔElec | ΔTotal |
|---|---|---|---|---|
| braking/regen | 12.1 | +0.0002 | +0.0862 | +0.0865 |
| standstill | 28.7 | +0.0000 | 0.0000 | +0.0000 |
| 0-15 Nm | 15.4 | −0.0052 | −0.0452 | −0.0504 |
| **15-30 Nm** | 18.5 | **+0.3502** | −0.3197 | +0.0305 |
| **30-50 Nm** | 13.0 | **+0.5434** | −0.2873 | **+0.2561** |
| 50-75 Nm | 6.6 | **−0.2937** | +0.2922 | −0.0015 |
| >75 Nm | 5.7 | **−0.3351** | +0.3203 | −0.0148 |
| **TOTAL** | 100 | **+0.2598** | +0.0465 | +0.3063 |

Actual gap +0.2720. **The gap is not uniform:** SAC loses **+0.894 L/100km**
across 15-50 Nm but *wins* **−0.629** above 50 Nm. The net gap is ~3.4x smaller
than the low-torque loss, which had been masking it.

## H. Highest-impact remaining limitation

The **exploration deadlock in 15-35 Nm** (section C), worth ~+0.89 L/100km of
excess fuel before high-torque offsets.

## I/J. Intervention selected, and its measured effect

**One variable changed: action representation.** Everything else frozen.

**Attempt 1 — ungated `modeaware`.** Mechanism **confirmed** in the target band
(30-35 Nm: OFF 0→12.8%, fuel **−0.1129**) but **regressed overall** (3.8775):
the fixed 40% OFF allocation is spent on an infeasible mode above ~50 Nm,
compressing ASSIST and driving **LPS to 100% at >75 Nm (+0.1464)**.

**Attempt 2 — `modeaware_gated`** (apply the reparameterization only where the
motor envelope makes OFF reachable; linear elsewhere). Endpoints, strict
monotonicity and reachable-`u` set preserved; 267/267 tests pass.

## K/L. Multi-seed results (section 21), n=3 per cycle

| cycle | gated mean | std | min | max | 95% CI | linear mean | Δ | Cohen d | CS | viol |
|---|---|---|---|---|---|---|---|---|---|---|
| **NEDC** | 3.8824 | 0.1371 | 3.7311 | 3.9985 | [3.727, 4.038] | 3.7727 | **+0.1097 worse** | **−1.11** | **1/3** | 0 |
| **FTP75** | **3.2460** | 0.0434 | **3.2088** | 3.2937 | [3.197, 3.295] | 3.3821 | **−0.1361 better** | **+2.02** | **3/3** | 0 |

Per-seed:

| | s0 | s1 | s2 |
|---|---|---|---|
| NEDC | 3.9177 (**+11.41pp, CS NO**) | 3.7311 (−1.80pp, CS yes) | 3.9985 (**+8.37pp, CS NO**) |
| FTP75 | 3.2356 (+0.62pp) | 3.2937 (−2.00pp) | 3.2088 (−0.64pp) |

**Mode split:** NEDC OFF 41.6% / ASSIST 8.9%; FTP75 OFF 43.2% / ASSIST 4.6%
(benchmark ASSIST 0.4%). The deadlock **is** broken — OFF rose from 35→42%
and ASSIST fell from 24.8→~9% — but on NEDC that did not convert to fuel.

## M. Has the authority-equal benchmark been beaten?

- **FTP75: essentially YES on the best seed.** 3.2088 vs rule-based 3.2323 =
  **−0.7%**, charge-sustaining, zero violations. Mean 3.2460 = **+0.4%**, i.e.
  statistically *at* the benchmark (CI [3.197, 3.295] straddles 3.2323).
- **NEDC: NO.** Mean 3.8824 = +10.7% vs 3.5056, +8.5% vs authority-equal
  3.5792 — a **regression** versus the linear reference.

**No overall success is claimed.**

## Why the intervention splits by cycle — root cause of the NEDC failure

On NEDC seeds 0 and 2 SoC ran away to **+11.41pp / +8.37pp**. The terminal
charge-sustaining penalty cannot prevent this at γ=0.20:

```
gamma=0.20 discount, k steps before episode end:
   k=1  -> 2.0e-01     k=3  -> 8.0e-03     k=10 -> 1.0e-07
SoC drift +11.41pp -> terminal penalty 11.79 reward units,
   but seen 10 s earlier as 1.2e-06  -> INVISIBLE
```

So at γ=0.20 the **only** effective SoC control is the per-step `k_fb` costate.
The gated map gives the agent far more engine-OFF freedom; on NEDC the weak
per-step SoC control could not contain it, and the policy over-charged via LPS
to compensate. FTP75 is immune because its denser braking (REGEN 25.7% vs
17.0%) supplies regen energy without needing LPS charging.

**This is a γ×action-map interaction, not a flaw in either alone.**

## N. Next single experiment (one variable)

**Option B — `k_fb`.** Raise the per-step costate feedback gain (currently
1.656 liter-units = 8.0 ECMS units) for the **gated map at γ=0.20 on NEDC**.

- *Hypothesis:* SoC runaway on 2/3 NEDC seeds is caused by per-step SoC control
  being too weak once the action map grants more OFF freedom, with the terminal
  penalty structurally invisible at γ=0.20.
- *Expected effect:* NEDC charge-sustaining returns to 3/3; NEDC mean falls
  from 3.8824 toward the 3.73 already achieved on the compliant seed.
- *Success criterion:* 3/3 charge-sustaining **and** mean < 3.7727 (the linear
  reference) on NEDC, without degrading the FTP75 result.
- *Do not* also change gamma, reward scale, entropy or architecture.

**Fallback if that fails:** make the gate SoC-aware as well as
feasibility-aware, or revert NEDC to the linear map and keep the gated map for
FTP75 only (cycle-specific configuration, which would itself be a reportable
finding about cycle-dependent action geometry).

---

## Status summary

| | NEDC | FTP75 |
|---|---|---|
| best validated config | **linear**, 3.7727 ± 0.0281 | **gated**, 3.2460 ± 0.0434 |
| rule-based | 3.5056 | 3.2323 |
| gap | **+7.6%** | **+0.4%** (best seed −0.7%) |

FTP75 is now at the benchmark. NEDC remains the open problem, with a specific,
measured cause and a specific next experiment.
