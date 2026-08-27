# Phase 6 — Controlled Conditional-Exploration Experiment

**Verdict: REFUTED** (primary region). The conditional-coverage hypothesis
from Phase 5B is rejected as the cause of the residual gap.

Raw: `results/phase6/phase6_forensics_NEDC.txt`, `results/phase6/phase6_NEDC.json`
Figures: `results/phase6/figures/q_landscape_ab_NEDC.png`

---

## 1. Design

**CONTROL** = the Phase-5 validated candidate, already trained (commit
`9a125ad`), 3 seeds, replay buffers retained — therefore provably untouched by
any Phase-6 code.

**TREATMENT** = identical in every respect except one flag.

```
gamma 0.20 · n_step 1 · modeaware_gated · k_fb 2.5
eq_factor 0.2717 (NEDC) / 0.4981 (FTP75) · gradient_steps 16 · buffer 300k
batch 512 · lr 3e-4 · lookahead 5 · target_entropy auto(-1.0) · 150k steps
seeds {0,1,2} (same as all prior validated multi-seed work)
```

**Intervention** (`src/agents/targeted_exploration.py`, commit `6e2d475`):
override `_sample_action`; when `15 <= T_MGB < 35 Nm`, `0.40 <= SoC <= 0.55`
**and** engine-OFF is physically reachable, replace the action with a
**uniform draw from the feasible OFF interval** with probability `p = 0.30`.

Evaluation safety proved by inspection, not assumed: `_sample_action` is
called exactly once (`collect_rollouts` L538), SAC does not override it,
`predict()` never calls it, and every evaluation path uses
`predict(deterministic=True)`. Contamination is **structurally impossible**.

Not imitation (§C/§23): the injected action is a *uniform* draw over a
feasibility-defined interval. It encodes only "engine-off is possible here",
never what a good controller would choose. No benchmark/ECMS label is used.

Measured injection: 100% feasible; 4,364 / 3,780 / 4,846 injections per seed
(~2.9% of all steps).

---

## 2. §G — Conditional replay coverage: **the intervention worked**

| cell | CONTROL n / OFF | CONTROL OFF% | TREATMENT n / OFF | TREATMENT OFF% | Δ |
|---|---|---|---|---|---|
| **30-35 Nm / SoC 40-50** | 6,132 / **274** | **4.5%** | 3,632 / **1,333** | **36.7%** | **+32.2pp** |
| 15-30 Nm / SoC 40-50 | 13,091 / 5,769 | 44.1% | 10,923 / 5,995 | **54.9%** | +10.8pp |
| 35-50 Nm / SoC 40-50 | 2,843 / 973 | 34.2% | 2,830 / 463 | 16.4% | −17.8pp |

OFF transitions in the primary target cell rose **4.9×**. The manipulation
check passes unambiguously.

---

## 3. §H/§I/§J — the causal chain breaks at the first link

### 30-35 Nm (primary diagnostic region), n=117 matched states

| metric | CONTROL | TREATMENT |
|---|---|---|
| coverage | 4.5% | **36.7%** |
| ΔQ(OFF−ASSIST) mean / median | −0.0071 / −0.0051 | **−0.0066 / −0.0067** |
| ΔQ(OFF−ASSIST) > 0 | 7% | **6%** |
| ΔQ(OFF−LPS) mean | −0.0079 | −0.0043 |
| **Δr(OFF−ASSIST) mean / >0** | **+0.0000 / 10%** | **+0.0000 / 10%** |
| P(OFF) | 0.1% | 1.5% |
| actor–Q displacement | 0.944 | 0.467 |
| `[r=OFF, Q=ASSIST]` conflict | 8% | 7% |
| `[r=ASSIST, Q=ASSIST]` agree | **85%** | **87%** |

**An 8× coverage increase moved ΔQ(OFF−ASSIST) by +0.0005 — i.e. nothing.**

### CORRECTION TO THE PHASE-5B DIAGNOSIS

Phase 5B stated *"the reward favours OFF and the critic disagrees"* at
30-35 Nm. **That was overstated and is now corrected.** At 30-35 Nm
Δr(OFF−ASSIST) = **+0.0000, positive in only 10% of states**, and **85-87% of
states are `[reward=ASSIST, Q=ASSIST]` — reward and critic AGREE.**
The Phase-5B claim generalised the 15-30 Nm aggregate (+0.0011) into a region
where it does not hold. **There was no reward-vs-critic conflict at 30-35 Nm
to repair**, so filling the coverage hole there could not have helped. The
experiment was still worth running: it is what exposed the error.

### 15-30 Nm (n=29) — where the reward *does* prefer OFF (100% of states)

| metric | CONTROL | TREATMENT |
|---|---|---|
| ΔQ(OFF−ASSIST) mean | −0.0022 | −0.0023 |
| ΔQ(OFF−ASSIST) > 0 | 14% | **34%** |
| `[r=OFF, Q=ASSIST]` conflict | **86%** | **66%** |
| P(OFF) | **68.9%** | **50.7%** |
| actor–Q displacement | **0.066** | **0.270** |

Coverage **partially corrected the critic** here (conflict 86% → 66%,
ΔQ>0 14% → 34%) — a genuine, if partial, confirmation of the coverage→Q link
in the region where a conflict actually existed. **But the actor moved the
wrong way**: P(OFF) fell 18pp and displacement quadrupled.

### 35-50 Nm (n=41)

ΔQ(OFF−ASSIST) −0.0001 → **+0.0048** (critic now favours OFF); displacement
0.268 → 0.183; P(OFF) 14.9% → 19.3%. The clearest coverage→Q→actor response —
but this region is only ~3% of the cycle and the benchmark itself uses OFF
just 9.8% here, so it carries little fuel weight.

---

## 4. §L — Vehicle-level results (3 seeds, no cherry-picking)

### NEDC

| arm | seed | V_CE_equiv | ΔSoC | CS | OFF% | ASSIST% | LPS% | viol |
|---|---|---|---|---|---|---|---|---|
| CONTROL | 0 | **3.6862** | +0.28 | Y | 39.8 | 15.3 | 27.9 | 0 |
| CONTROL | 1 | 3.8431 | −0.72 | Y | 38.2 | 12.0 | 32.9 | 0 |
| CONTROL | 2 | 3.7704 | +0.23 | Y | 37.8 | 16.6 | 28.6 | 0 |
| TREATMENT | 0 | 3.8285 | +1.42 | Y | 38.2 | 14.3 | 30.6 | 0 |
| TREATMENT | 1 | 3.7488 | **+2.89** | **N** | 40.9 | 9.2 | 32.9 | 0 |
| TREATMENT | 2 | 3.8762 | +1.47 | Y | 35.3 | 18.3 | 29.4 | 0 |

| arm | mean | std | min | max | 95% CI | CS | viol |
|---|---|---|---|---|---|---|---|
| CONTROL | **3.7666** | 0.0785 | 3.6862 | 3.8431 | [3.678, 3.855] | **3/3** | 0 |
| TREATMENT | 3.8178 | 0.0644 | 3.7488 | 3.8762 | [3.745, 3.891] | **2/3** | 0 |

**Δ = +0.0513 L/100km WORSE, Cohen d = −0.71.**

### FTP75

| arm | mean | std | min | max | 95% CI | CS | viol |
|---|---|---|---|---|---|---|---|
| CONTROL | **3.2889** | 0.0174 | 3.2699 | 3.3041 | [3.269, 3.309] | 3/3 | 0 |
| TREATMENT | 3.2984 | 0.0184 | 3.2831 | 3.3188 | [3.278, 3.319] | 3/3 | 0 |

**Δ = +0.0095 WORSE, d = −0.53.** Neither arm beats the best-known FTP75
configuration (gated `k_fb=1.656`, **3.2460**).

---

## 5. §M — Error budget (NEDC, ΔFuel vs rule-based)

| config | brake | 0-15 | 15-30 | 30-50 | 50-75 | >75 | TOTAL |
|---|---|---|---|---|---|---|---|
| linear ref | +0.0002 | −0.0052 | +0.3502 | +0.5434 | −0.2937 | −0.3351 | +0.2598 |
| gated k=1.656 | +0.0030 | −0.0123 | +0.3431 | +0.1318 | −0.1756 | +0.1222 | +0.4121 |
| **CONTROL k=2.5** | +0.0006 | −0.0122 | +0.3677 | +0.3705 | −0.2317 | −0.3142 | **+0.1807** |
| **TREATMENT** | +0.0006 | −0.0231 | **+0.4636** | +0.3345 | −0.3091 | **−0.1436** | **+0.3229** |

30-50 Nm improved marginally (+0.3705 → +0.3345) but **15-30 Nm worsened
(+0.3677 → +0.4636)** and **~55% of the >75 Nm advantage was lost**
(−0.3142 → −0.1436). Net: the error was **redistributed and increased**.

---

## 6. §O — Causal classification

> ### **CASE 3** — coverage ↑, Q(OFF) did not improve.
> Replay coverage alone is **not** sufficient in the primary region.

Secondary observations: **CASE 2** partially at 35-50 Nm (Q improved, actor
responded weakly); **CASE 4** partially at 15-30 Nm (critic conflict fell
86%→66%, yet fuel worsened).

---

## 7. §X — Explicit answers

| # | question | answer |
|---|---|---|
| 1 | Conditional OFF coverage ↑ at SoC 40-50? | **YES** |
| 2 | By how much? | 30-35 Nm: **4.5% → 36.7%** (274 → 1,333 transitions, 4.9×) |
| 3 | ΔQ(OFF−ASSIST) improved? | **NO** at 30-35 (−0.0071 → −0.0066). Partially at 15-30 (>0: 14%→34%) and 35-50 (−0.0001→+0.0048) |
| 4 | ΔQ(OFF−LPS) improved? | Marginally: −0.0079 → −0.0043 (still negative) |
| 5 | Actor P(OFF) ↑? | **NO** — 30-35: 0.1%→1.5% (negligible); 15-30: **68.9%→50.7% (fell)** |
| 6 | Actor–critic alignment improved? | **MIXED** — 30-35: 0.944→0.467 (better); 15-30: 0.066→0.270 (**much worse**) |
| 7 | SoC charge-sustaining? | **NEDC 3/3 → 2/3 (WORSE)**; FTP75 3/3 → 3/3 |
| 8 | NEDC fuel improved? | **NO** — 3.7666 → 3.8178 (+0.0513, d=−0.71) |
| 9 | FTP75 improved or regressed? | **REGRESSED** — 3.2889 → 3.2984 (+0.0095, d=−0.53) |
| 10 | 15-30 Nm error decreased? | **NO** — +0.3677 → +0.4636 |
| 11 | 30-50 Nm error decreased? | Marginally — +0.3705 → +0.3345 |
| 12 | >75 Nm advantage retained? | **NO** — −0.3142 → −0.1436 (~55% lost) |
| 13 | SAC−ECMS gap decreased? | **NO** (no arm improved on either cycle) |
| 14 | % of original gap remaining? | NEDC: best config 3.7666 vs rule-based 3.5056 → **+7.4%**; of the session-start +17.7%, **~42% of the original gap remains**. FTP75 best 3.2460 vs 3.2323 → **+0.4%** |
| 15 | Highest-impact bottleneck now? | **15-30 Nm**, where the reward genuinely favours OFF (100% of states) yet the critic conflicts in 66-86% and the actor is displaced |
| 16 | Single next experiment? | See §8 |

### **CLASSIFICATION: REFUTED**

The conditional-coverage hypothesis is **rejected** as the cause of the
residual gap in the primary region. Coverage rose 8× and produced **no** Q
correction there, **no** actor response, and a **worse** vehicle outcome on
both cycles.

**Rejected causal links (explicit, per §R):**
- coverage → Q(OFF) at 30-35 Nm: **REJECTED**
- Q → actor at 15-30 Nm: **REJECTED** (Q conflict fell, actor still retreated)
- intervention → fuel: **REJECTED** on both cycles

**Retained:** coverage → Q is *partially* real where a genuine reward-vs-critic
conflict exists (15-30 Nm, 35-50 Nm) — but it is not the dominant mechanism.

---

## 8. §Q/§16 — Single next experiment

**Do NOT pursue exploration further** (§R: do not escalate strength to force a
result). **Do not stack interventions** (§P).

**Recommended next: actor-side alignment at 15-30 Nm — specifically the
entropy temperature.**

Evidence:
1. 15-30 Nm is now the largest single error term (**+0.3677**, unchanged by
   every intervention across Phases 4-6).
2. There, the reward **unambiguously** favours OFF (Δr>0 in **100%** of states)
   — unlike 30-35 Nm, where reward and critic agree on ASSIST.
3. The critic is **partially correctable** there (conflict 86%→66% under
   coverage alone).
4. The failure is now demonstrably **actor-side**: displacement rose 0.066 →
   0.270 and P(OFF) fell 68.9% → 50.7% *while the critic was improving*.
5. Phase-5B showed alignment predicts performance (FTP75, the only
   benchmark-level result, is the most aligned at 75.8%).

**Caveat, stated up front:** Phase 5 already showed `target_entropy` −1/−2/−3
is statistically indistinguishable at n=3 on the *linear* configuration. This
experiment is only justified because the configuration is now different
(gated + k_fb=2.5) and the diagnosis is now actor-displacement-specific. If it
also fails, the honest conclusion is that the **unimodal Gaussian policy class**
is the limitation, which would justify a mixture/discrete-continuous policy —
a scope increase to be proposed, not taken unilaterally.

**NOT STARTED. Awaiting authorisation.**

---

## 9. Unresolved

1. Why does 15-30 Nm resist every intervention while the reward clearly
   favours OFF there?
2. Is the >75 Nm advantage inherently in tension with low-torque OFF usage?
   Every intervention that improved one degraded the other.
3. FTP75 sits at the benchmark with `k_fb=1.656` but `k_fb=2.5` costs it
   +0.043 — is the cycle-specific costate a real requirement or an artifact of
   the 150k budget?
