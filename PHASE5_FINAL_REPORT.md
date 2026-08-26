# Phase 5 — Final Report: Costate Gain Identification & SoC Stabilisation

Does not supersede Phase-1/2/3/4 conclusions. Raw output:
`results/phase5/phase5_NEDC.txt`, `results/phase5/phase5_NEDC.png`.

---

## A. Objective

Find the costate feedback gain `k_fb` that lets the gated action
representation exploit the engine-OFF opportunity **while keeping SoC
charge-sustaining**, then recover NEDC fuel economy without sacrificing FTP75.

## B. Frozen configuration (only `k_fb` varied)

```
gamma 0.20 · n_step 1 · modeaware_gated · eq_factor 0.2717 (NEDC) / 0.4981 (FTP75)
gradient_steps 16 · lookahead 5 · buffer 300k · batch 512 · lr 3e-4 · [256,256]
150k steps · checkpoint rule = zero-violations → charge-sustaining → min V_CE_equiv
```

## C. k_fb sweep

### C.1 Screening on seed 1 — and a methodological error

| k_fb | 1.656 | 2.0 | 2.5 | 3.0 | 4.0 | 5.0 |
|---|---|---|---|---|---|---|
| V_CE_equiv | **3.7311** | 3.7961 | 3.8431 | 3.8340 | 3.7911 | 3.9573 |
| CS | Y | Y | Y | Y | **N** | **N** |
| ΔSoC pp | −1.80 | +0.28 | −0.72 | +1.21 | +3.84 | +3.87 |

This suggested `k_fb=1.656` was already optimal and higher values only hurt.
**That conclusion was wrong, because the screen used the wrong seed.** The SoC
runaway being investigated occurred on **seeds 0 and 2**; seed 1 was already
SoC-stable, so raising `k_fb` there could only over-penalise. Recorded as an
error rather than hidden.

### C.2 Correct test — k_fb on the FAILING seeds

| run | k_fb | V_CE_equiv | ΔSoC pp | CS |
|---|---|---|---|---|
| seed 0 (Phase 4) | 1.656 | 3.9177 | **+11.41** | **N** |
| **seed 0** | **2.5** | **3.6862** | **+0.28** | **Y** |
| seed 0 | 3.0 | 3.7376 | +0.98 | Y |
| seed 0 | 4.0 | 3.7724 | +2.64 | N |
| seed 2 (Phase 4) | 1.656 | 3.9985 | **+8.37** | **N** |
| **seed 2** | **3.0** | **3.7803** | +0.73 | **Y** |

**Both diverging seeds are rescued.** `k_fb` does supply the per-step SoC
regulation the terminal penalty cannot provide at γ=0.20.

**Both readings are real and describe different regimes:** on an already-stable
seed, extra `k_fb` over-penalises (CASE C); on a diverging seed it is
corrective. Screening on a healthy seed hides the effect entirely.

### C.3 Section 9 — shape

Non-monotonic / U-shaped, optimum at **k_fb ≈ 2.5** (V=3.6862 on seed 0);
degrades again by k=4.0 where CS fails in the *over-charging* direction.

## D. SoC trajectory analysis (section 5)

Trajectory stability improves monotonically with `k_fb` (seed-1 sweep):

| k_fb | mean \|SoC−50%\| | drift pp/1000s | SoC min | SoC max |
|---|---|---|---|---|
| 1.656 | 14.25 | −14.46 | 14.1 | 50.0 |
| 2.5 | 11.47 | −8.31 | 26.5 | 50.2 |
| 3.0 | 9.98 | −5.99 | 30.7 | 51.2 |
| 5.0 | **4.34** | **−3.28** | 38.2 | 54.2 |

The mechanism works as designed. At k≥4 the trajectory is tight but shifted
**upward**, so terminal CS fails in the opposite direction.

## E. OFF/ASSIST/LPS policy analysis (section 6)

OFF% by torque band (seed-1 sweep):

| k_fb | 0-15 | 15-30 | 30-35 | 35-50 | 50-75 | >75 |
|---|---|---|---|---|---|---|
| linear ref | 99.5 | 53.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1.656 gated | 100.0 | 52.2 | 26.5 | 61.0 | 3.8 | 0.0 |
| 3.0 | 100.0 | 57.5 | 9.4 | 14.6 | 0.0 | 0.0 |
| 5.0 | 100.0 | 37.6 | 5.1 | **0.0** | 0.0 | 0.0 |
| RULE-BASED | 100.0 | **98.2** | **84.6** | 9.8 | 0.0 | 0.0 |
| ECMS | 100.0 | 76.5 | 48.7 | 63.4 | 22.5 | 0.0 |

**CASE C is visible**: raising `k_fb` progressively suppresses OFF (35-50 Nm
goes 61.0 → 0.0). SoC stability is bought with the very behaviour Phase 4
unlocked.

## F. Costate analysis (section 8)

All candidates keep `eq_factor > 0` everywhere (no sign inversion). In ECMS
units (×4.8309), over the actually-visited SoC:

| k_fb | ECMS λ min | max | mean | SoC flip point |
|---|---|---|---|---|
| 1.656 | 1.312 | 4.183 | 2.452 | 66.4% |
| 2.5 | 1.289 | 4.150 | 2.697 | 60.9% |
| 5.0 | 0.303 | 4.158 | 2.308 | 55.4% |

Higher `k_fb` widens the price range and makes the battery *nearly free* above
50% SoC — which is precisely what drives the over-charging at k≥4.

## G. NEDC results (3 seeds)

| config | mean | std | min | max | 95% CI | CS | viol |
|---|---|---|---|---|---|---|---|
| linear reference | 3.7727 | 0.0281 | — | — | — | 3/3 | 0 |
| gated k=1.656 (Phase 4) | 3.8824 | 0.1371 | 3.7311 | 3.9985 | — | **1/3** | 0 |
| **gated k=2.5** | **3.7666** | 0.0785 | **3.6862** | 3.8431 | [3.678, 3.855] | **3/3** | 0 |
| gated k=3.0 | 3.7840 | 0.0483 | 3.7376 | 3.8340 | [3.729, 3.839] | **3/3** | 0 |

**k=2.5 vs linear: −0.0061 L/100km — a statistical TIE** (0.08σ). Charge
sustainability is fully restored (1/3 → 3/3), but **fuel is unchanged**.

## H. FTP75 results (3 seeds) — section 11 Pareto

| config | mean | std | min | CS | vs rule-based |
|---|---|---|---|---|---|
| linear reference | 3.3821 | 0.0846 | — | 3/3 | +4.6% |
| **gated k=1.656** | **3.2460** | 0.0434 | **3.2088** | 3/3 | **+0.4%** |
| gated k=2.5 | 3.2889 | 0.0174 | 3.2699 | 3/3 | +1.8% |

**k=2.5 materially regresses FTP75** (+0.0429 vs k=1.656; Cohen d ≈ 1.30).
Per section 11 this is not an acceptable trade.

**Pareto conclusion: the optimal costate gain is CYCLE-DEPENDENT.**
NEDC needs k≈2.5; FTP75 is best at k=1.656. Physically consistent: FTP75 has
far more braking (REGEN 25.7% vs 17.0%), so regen supplies battery energy and
less costate pressure is required to stay charge-sustaining.

## I. Error-budget comparison (sections 24/25)

ΔFuel vs rule-based by region (negative = SAC better), NEDC seed 0:

| config | brake | 0-15 | 15-30 | 30-50 | 50-75 | >75 | TOTAL |
|---|---|---|---|---|---|---|---|
| linear ref | +0.0002 | −0.0052 | +0.3502 | **+0.5434** | −0.2937 | −0.3351 | +0.2598 |
| gated k=1.656 | +0.0030 | −0.0123 | +0.3431 | **+0.1318** | −0.1756 | **+0.1222** | +0.4121 |
| **gated k=2.5** | +0.0006 | −0.0122 | +0.3677 | +0.3705 | −0.2317 | **−0.3142** | **+0.1807** |
| gated k=3.0 | +0.0008 | −0.0251 | +0.5006 | +0.2713 | −0.2063 | −0.3092 | +0.2320 |

Two results:
1. **k=2.5 gives the best total budget (+0.1807 vs linear +0.2598)** — a real
   reduction, not pure redistribution.
2. **Section 25 satisfied**: k=2.5 *preserves* the high-torque advantage
   (>75 Nm −0.3142 ≈ linear's −0.3351), whereas k=1.656 destroyed it (+0.1222,
   LPS 24%→100%). This was an explicit risk and it is now controlled.

The dominant remaining deficit is **15-30 Nm (+0.3677)**, where the benchmark
uses OFF 98.2% and SAC 58.4%.

## J. Multi-seed validation

Done, 3 seeds per cycle, per section 21. See G and H. No cherry-picking:
means, std, min, max and 95% CI reported.

## K. Root-cause conclusion — **section 13 CASE A, then a deeper structure**

SoC stabilised, but NEDC fuel did not improve → *action access is no longer
the limiting factor*. Probing further:

**Actor reachability at k=2.5 (seed 0)** — P(OFF) under the policy still
predicts actual OFF usage almost exactly:

| band | mean z | P(OFF) | σ | actual OFF% | benchmark |
|---|---|---|---|---|---|
| 15-30 | −1.01 | 59.0% | 0.278 | 58.4% | **98.2%** |
| **30-35** | **+2.70** | **3.3%** | **0.265** | **4.3%** | **84.6%** |
| 35-50 | +0.85 | 44.1% | 0.630 | 51.2% | 9.8% |

**30-35 Nm is still deadlocked even though the gate made OFF reachable there**
(a_off = +0.20).

**Section 18 — the critic landscape is BIMODAL.** At T=30 Nm:

```
a=-0.50  Q=-0.0410   <- LPS local max   (actor mean a=-0.62, sigma=0.194)
a=+0.10  Q=-0.0594   <- VALLEY, barrier 0.0184 deep
a=+0.30  Q=-0.0461   <- OFF local max
```

Over 10 sampled 30-35 Nm states the critic's *global* argmax lies in the OFF
region in 6/10, yet `dQ(OFF_boundary − actor) = −0.0107` (<0 in 10/10) and
`dr = −0.0043` (<0 in 10/10).

**Mechanism:** the gated map moved the OFF boundary from a=+0.82 to a=+0.20,
but the actor then **converged onto the LPS mode and shrank σ from ~0.55 to
0.194**, so the OFF mode is ~4-5σ away *again*, separated by a genuine Q
valley. A unimodal tanh-Gaussian performing local gradient ascent cannot
cross it.

**This is the first direct evidence that the optimal control law here is
BIMODAL (commit to LPS *or* to OFF, not to anything between), while SAC's
policy class is unimodal.** It vindicates — with measurement rather than
speculation — the bang-bang hypothesis raised and left unconfirmed in Phase 1.

## L. Should the gated representation be retained?

**Yes for FTP75** (3.2460, +0.4% from benchmark, 3/3 CS).
**Yes for NEDC at k=2.5** — it is CS-valid, ties the linear reference on fuel,
and gives the best error budget while preserving the high-torque advantage.

## M. Should the linear representation be retained?

**Yes, as the hard reference** (section 10). It remains statistically tied with
the best gated configuration on NEDC, so it cannot be discarded.

## N. Actor/critic diagnosis

Not a critic-accuracy failure: the critic's global optimum is in OFF for the
majority of 30-35 Nm states. Not a pure actor failure either: at the actor's
own location the *local* Q gradient correctly points away from the marginal
OFF boundary. **It is a policy-class / optimisation-topology failure** —
bimodal Q, unimodal policy, collapsed σ.

## O. Replay/exploration diagnosis

σ collapse (0.55 → 0.194 in the affected band) means the buffer receives almost
no OFF transitions there, so the OFF branch of Q stays poorly estimated —
the Phase-4 deadlock re-forming one mode inward. Quantifying replay OFF-share
in 15-35 Nm is the confirmatory measurement (section 20/22) if needed.

## P. ECMS comparison

Deferred by design (section 26) — benchmark-level performance is reached on
FTP75 (+0.4%) but not NEDC (+7.4%).

## Q. Next single highest-value intervention

**Raise entropy in the affected region — section 19 priority 1: target
entropy.** Evidence: σ collapsed to 0.194 while the competing mode sits ~4-5σ
away behind a 0.0184-deep Q valley. Higher entropy keeps probability mass on
both modes long enough for the critic to estimate the OFF branch.

- **Test:** `target_entropy ∈ {−0.5, 0.0}` (higher entropy than the current
  −1.0), gated map, **k_fb = 2.5**, γ=0.20, NEDC. One variable.
- **Success:** 15-30 Nm OFF% rises above 58.4% toward the benchmark's 98.2%
  **and** NEDC mean < 3.7727 with 3/3 CS.
- **If it fails**, the conclusion is that the unimodal Gaussian policy class is
  itself the limitation, which would justify evaluating a mixture/discrete-
  continuous policy — a larger change to be proposed, not taken unilaterally.

---

## Success hierarchy status (section 29)

| level | status |
|---|---|
| 1 — zero constraint violations | **PASS** (0 across all 12 runs) |
| 2 — charge sustainability | **PASS** — NEDC 1/3 → **3/3**; FTP75 3/3 |
| 3 — beat linear SAC reference | **NOT MET** — 3.7666 vs 3.7727 is a tie |
| 4 — beat authority-equal benchmark | **NOT MET** (NEDC +5.2% best) · FTP75 **+0.4%, best seed −0.7%** |
| 5 — approach ECMS | not attempted |
| 6 — beat ECMS | not attempted |

**Phase 5 delivered LEVEL 2, not LEVEL 3.** SoC stability is solved; NEDC fuel
is unchanged. Claiming otherwise would be unsupported.
