# PHASE 11 — STAGE 1: NO-TRAINING FORENSIC DIAGNOSIS

**No training. No code modification. No calibration. No DP solver.** All results
are deterministic offline diagnostics on the existing CONTROL checkpoints
(`models_p5{s0,,_s2}_k2.5/NEDC`, `models_p5f_k2.5_s{0,1,2}/FTP75`) and their
replay buffers, git HEAD `90af969` (docs-only above `f1f45c5`; `src/` clean),
SB3 2.8.0 / torch 2.12 CPU / numpy 2.4.6 / Python 3.13.2.

Scripts / data (analysis-only, under `results/phase11/`):
`s1_11A_reward_safety.py` → `data/s1_11A.json`;
`s1_11B_action_transform.py` → `data/s1_11B_{NEDC,FTP75}.json`;
`s1_11CDE_matched_forensic.py` → `data/s1_11CDE_{NEDC,FTP75}.json`.

Tags: **[F]** measured fact · **[M]** mathematical implication · **[H]** hypothesis.

---

## 11A — REWARD / SAFETY FORENSIC (eq_eff sign inversion)

**Source expressions (verified in `ems_env.py::EMSEnv.step`):**
```
eq_eff  = eq_factor + k_fb*(SOC_TARGET - soc_before)        soc_before = pre-decision SoC
reward  = -reward_scale * (fuel_liters + eq_eff * elec_liters)  - SoC-band penalties
eq_eff < 0   <=>   soc_before > SOC_TARGET + eq_factor/k_fb
```

**Threshold — prior reports are STALE. [F][M]**

| | prior report | **actual CONTROL** (eq_factor/k_fb) |
|---|---|---|
| NEDC | 66.41 % | **60.87 %**  (0.5 + 0.2717/2.5) |
| FTP75 | 80.06–80.08 % | **69.92 %**  (0.5 + 0.4981/2.5) |

The 66.41 % / 80.08 % figures were computed with the **old** config
(`eq_factor 1.3125, k_fb 8.0`). Phase 2 §18 note 5 claimed "dividing both by
4.8309 keeps the ratio" — **false**, because CONTROL `k_fb` is 2.5, not
`8.0/4.8309 = 1.656`, so `eq_factor/k_fb` changed and the threshold moved down
~5.5 pp.

**Trigger scan:**

| source | NEDC | FTP75 |
|---|---|---|
| **CONTROL deterministic rollouts** (3 seeds each, 1220 / 1876 steps) | **0 steps** SoC > threshold, **0** eq_eff < 0, min eq_eff +0.263 | **0 steps**, min eq_eff +0.356 |
| **CONTROL replay buffers** (150 016 transitions each) | s0: **3 534 (2.36 %)** ; s1: **1 370 (0.91 %)** ; s2: **1 086 (0.72 %)** — SoC up to 95.5 %, min eq_eff **−0.866** | s0: 0 ; **s1: 424 (0.28 %)**, min eq_eff −0.399 ; s2: 0 |

**Verdict [F].** The sign inversion is **NEVER triggered by any existing
CONTROL trajectory** (the converged deterministic policy). It **is** present in
the replay buffers — 0.3–2.4 % of NEDC transitions, all from the **early-training
SoC-runaway phase** (SoC 61–95 %), where discharging at high SoC received a
**positive** reward contribution up to `|eq_eff|_max·|elec term|` (bounded by
`|eq_eff| ≤ 0.866`, i.e. at most 0.87× the normal battery-cost magnitude). Those
~700–3 500 transitions gave the critic economically-inverted training signal in
a SoC region (> 61 %) the converged policy does not use.

**Classification: LATENT IMPLEMENTATION DEFECT** for the current controller
(0 current-trajectory triggers), with **minor historical critic-training
contamination** confined to an abandoned high-SoC region.
**Per the rules: reward NOT modified. Flagged as a required objective/safety
decision — not acted on in this phase.**

---

## 11B — ACTION-TRANSFORMATION FORENSIC (is ~58 Nm hard to reach/represent?)

Production path traced exactly through `EMSEnv._action_to_torques`
(`modeaware_gated` map → motor-envelope clamp → SoC masks → engine over-torque
guard). 4001-point action grid, real CONTROL-rollout states.

**Findings [F]:**

1. **The map is clean.** Monotone-decreasing `T_CE` in `a` in **every** state
   tested; **no clipping flats** (`longest_flat(a) = 0.0`); max single-grid-step
   jump ≈ 0.02–0.03 Nm (Δa = 5e-4) — fully continuous. `dT_CE/da ≈ −27 to
   −42 Nm per unit action` in the operating region (a mild −56 to −60 kink at
   one `modeaware_gated` segment breakpoint, not a discontinuity).

2. **~58 Nm engine torque is UNREACHABLE across most of the 15–35 Nm band**,
   because the LPS depth is capped at `U_MIN = −0.85`:

   | demand band | max feasible `T_CE` | `a` for `T_CE = 58` |
   |---|---|---|
   | 15–25 Nm (T ≈ 22–24) | **40–45 Nm** | NONE (even 50 Nm NONE) |
   | 25–30 Nm (T ≈ 28–29) | **52–54 Nm** | NONE (55 Nm NONE; 50 Nm needs a ≈ −0.9) |
   | 30–35 Nm, low end (T ≈ 31) | **57.5 Nm** | NONE |
   | 30–35 Nm, high end (T ≈ 34–35) | 63–65 Nm | a ≈ **−0.84 to −0.86** (= the `U_MIN` clamp) |

3. **Policy-space width** for `T_CE = 58 ± 1 Nm`: **0.0** (unreachable) or
   ≈ **0.05** where reachable (30–35 Nm high end); for `58 ± 5 Nm`: 0.0–0.24;
   for `T_CE 50–60`: 0.0–0.25.

**Answer to 11B's explicit question [F][M]:**
> **The actor is fully capable of representing any T_CE that is physically
> feasible. ~58 Nm is NOT in a numerically/pathologically difficult part of the
> transformation.** It is simply **at or beyond the U_MIN = −0.85 LPS clamp**
> for most 15–35 Nm demand — a *reachability* limit that ECMS's feasible grid
> shares (`_feasible_u_grid` uses the same `U_MIN`). The "~58 Nm reward
> preference" reported in V1-D / Phase 8 is the **boundary value** of the
> deepest-LPS action, not a free interior optimum. The actor's ~35 Nm sits well
> in the interior (`a ≈ −0.3 to −0.4`), ~0.5 action-units from the feasible-LPS
> boundary — a large **interior displacement**, not a boundary-resolution issue.

**⇒ "action-parameterization difficulty" is REJECTED as the mechanism.**

---

## 11C / 11D / 11E — STATE-MATCHED REWARD / CRITIC / ACTOR / REPLAY / BELLMAN

20 real CONTROL-rollout states (NEDC), bands 15–25 / 25–30 / 30–35 / 35–50 Nm,
121-point action grid; `r(s,a)` = actual implemented reward via env deep-copy;
`min-Q` averaged over the 3 CONTROL critics; Bellman
`Q_target(a) = r(a) + γ·(1−done)·V(s'(a))`,
`V(s') = min_i Q(s', π_det(s')) − ent_coef·log π` with **ent_coef = 0.0017**
(entropy collapsed → term ≈ 0.002, negligible). γ = 0.20.

### 11C-A — immediate reward argmax [F]
* In **~17 / 20** states (all with SoC < ~45 %) `a_R* = −1.0` exactly →
  `T_CE(a_R*)` = **the maximum feasible `T_CE`** (45 / 54 / 64 / 78 Nm by band).
  The immediate reward's argmax over the feasible grid is **always the U_MIN /
  deepest-LPS clamp**. (The few SoC ≈ 50 % states pick OFF: at target SoC
  `eq_eff` is small and the reward is battery-neutral → prefers no engine.)
* **The 58-vs-35 preference is WEAK.** `r(T_CE=58) − r(T_CE=35)` (where both are
  reachable, 30–50 Nm bands) = **+0.0006 … +0.0098** reward units — a shallow,
  monotone up-slope, never reversing, so the argmax lands at the boundary but
  the gradient is small.

### 11C-B — learned critic argmax [F]
* `min-Q` as a function of `T_CE` **peaks near T_CE ≈ 40–47 Nm and then
  DECLINES toward deep LPS.** So `a_Q*` gives `T_CE ≈ 31–47` (median ~44) —
  **well below the reward's boundary argmax (55–78)**.
* Yet `min-Q(T_CE=58) − min-Q(T_CE=35)` is still **positive** (+0.002 … +0.015):
  both 58 and 35 are on the critic's up-slope; the critic's *decline* is beyond
  ~50 Nm.
* Twin-Q disagreement at the deep-LPS end: **0.004–0.013** — an order of
  magnitude *below* Phase 9's 0.056–0.066 for genuinely infeasible/unsupported
  OFF actions. The deep-LPS region is **moderately covered**, not far-OOD.

### 11C-C — actor [F]
* `T_CE(a_pi)` is the **lowest of the three** in every band: typically **30–40 Nm**;
  in 2 / 20 low-SoC states it collapses to **OFF** (T_CE 2.6 and 17.0 Nm) with a
  **policy-variance spike** (`log_std ≈ 0` → σ ≈ 1, vs σ ≈ 0.13–0.39 elsewhere).
* So the operating cascade is **reward (~55–78) > critic (~40–47) > actor
  (~30–40)** — each stage drops the engine load by ~10–20 Nm.

### 11C-D — replay `T_CE` occupancy (NEDC, 3-seed aggregate) [F]

| demand band | `T_CE` 50–60+ Nm coverage (band-wide / state-matched) | note |
|---|---|---|
| 15–25 Nm | 0 % / 0 % | 58 Nm **unreachable** (max 40–45) — not a hole |
| 25–30 Nm | **8 % / 7–9 %** | thin |
| 30–35 Nm | **11 % / 7–23 %** | thin–moderate |
| 35–50 Nm | **38 % / 28–54 %** | **substantial** |

### 11D — BELLMAN DECOMPOSITION — the decisive result [F][M]

| quantity | value |
|---|---|
| `argmax_a r(a)` == `argmax_a [r(a) + γ·V(s'(a))]` | **19 / 20 states** |
| `γ·V(s')` spread across the action grid (fixed state) | **< 0.001** (near-constant offset ≈ −0.007 … −0.03) |
| mean Bellman residual `Q̂ − Q_target` at `a_R*` (deep-LPS) | **−0.0118** |
| … at `a_Q*` (critic's own argmax) | **−0.0051** |
| … at `a_pi` (actor's action) | **−0.0064** |
| `|resid @ a_R*|` > `|resid @ a_Q*|` | **19 / 20 states** |
| corr(SoC, resid @ a_R*) | +0.30 (weak: lower SoC → slightly worse) |
| residual at `a_R*` in the 35–50 Nm band (**38–54 % replay coverage**) | still **−0.009 … −0.015** |

**Mechanism assignment (11D menu):**
* **Mechanism 2 (future-value contribution flips the preference): FALSIFIED.**
  `γ·V(s')` is essentially constant across the action grid (one action moves
  SoC by ~0.1–0.3 %, so `V(s')` barely depends on `a`). Adding it does **not**
  change the argmax (19 / 20). The immediate reward and the 1-step Bellman
  target have the **same** argmax — deep LPS.
* **Mechanism 1 (critic's immediate component ≠ measured reward): not
  separately testable** from the fitted `Q̂` alone; but the residual analysis
  below supersedes it.
* **Mechanism 3 (critic approximation / Bellman inconsistency): CONFIRMED as
  dominant.** The fitted `Q̂` is systematically **below its own 1-step Bellman
  target at the deep-LPS end** (mean −0.0118, 19 / 20 states), ~2.3× the
  residual at the critic's own argmax. Its `Q(T_CE)` curve invents a downslope
  beyond ~50 Nm that neither the immediate reward nor the Bellman target has.
* **Mechanism 4 (coverage / extrapolation): CONTRIBUTING, not dominant.** The
  deep-LPS under-fit **persists in the 35–50 Nm band where replay coverage is
  38–54 %** (residual −0.009 … −0.015), and twin-Q disagreement there is low
  (0.004–0.013). Coverage amplifies the error at low SoC / low demand
  (corr 0.30, 8 % coverage at 25–30 Nm) but is not the whole story.

### 11E — ACTOR-TRACKING TEST [F]
* Mean `Q_loss = min-Q(a_Q*) − min-Q(a_pi)` = **0.0044** (~2 % of |Q| ≈ 0.2).
  Only **2 / 20** states exceed 0.015 (the low-SoC OFF-flip states, 0.017 &
  0.027).
* `policy-space distance(a_pi, a_Q*)` is often large (0.03–1.45) but the Q
  surface between them is **flat**, so little value is lost.
* `dQ/da` at `a_pi` ≈ ±0.02 (near-zero) in most states → the actor sits in a
  locally near-optimal spot **of its own (mis-fit) critic**.

**⇒ 11E: the actor is NOT grossly failing to track its critic.** It undershoots
the critic's argmax by ~10 Nm of engine load but loses only ~0.004 Q (flat
surface); it diverges materially (to OFF) only in ~10 % of low-SoC states.
**"Actor optimisation failure" (Case A) is WEAKLY supported — a secondary
term.**

---

## 11F — CASE CLASSIFICATION

| Case | fit | why |
|---|---|---|
| A — strong actor-tracking discrepancy | **WEAK / secondary** | actor undershoots critic by ~10 Nm but `Q_loss` mean 0.0044; gross divergence only 2/20 states |
| B — critic disagrees with immediate reward | **YES** — and 11D shows it is **not** the future term |
| C — poor coverage + critic disagreement (→ exploration bottleneck) | **PARTIAL** | coverage thin (8 %) at low demand, but **substantial (38–54 %) at 35–50 Nm where the critic under-fit persists** ⇒ not primarily coverage |
| **D — substantial coverage + critic inconsistent with reward/Bellman (→ critic function-approximation / critic-training failure)** | **BEST FIT** | `Q̂` below its own Bellman target at deep-LPS in 19/20 states, including in 38–54 %-covered bands; γ does not re-rank; actor follows the mis-fit critic |
| E — actor ≈ critic, both disagree with reward, explained by future value | **FALSIFIED** | future-value term is action-flat (19/20 same argmax) |

---

## 11G — GAMMA: IS THERE A MECHANISTIC REASON TO TEST IT?

**No, not from this evidence. [F][M]**

* At γ = 0.20 the critic target is `r_t + 0.2·V(s_{t+1})`; `V` recursively
  carries geometrically down-weighted future rewards (0.2, 0.04, 0.008, …).
  Future effects are **present**, just attenuated.
* **Measured:** `V(s')` varies by **< 0.001 across the action grid** at a given
  state (one action changes SoC by ~0.1–0.3 %, and `V` is smooth in SoC), so
  the discounted future term is a **near-constant offset**, not an
  action-ranking signal. `argmax_a r(a) == argmax_a [r(a) + γV(s'(a))]` in
  **19 / 20** states.
* The immediate reward **already** prefers deep LPS; the failure is the fitted
  critic not reproducing that. Raising γ would add more (still roughly
  action-flat) future value — it would **not** address the critic's systematic
  under-fit at the deep-LPS end.
* **A gamma experiment is not justified.** It stays **below** the
  critic-fidelity hypothesis in priority. (A gamma change could in principle
  make the future term more action-dependent by compounding SoC divergence over
  more steps — but that is a speculative second-order effect, not something
  this evidence supports.)

---

# FINAL DECISION REPORT

## 1. VERIFIED FACTS

1. **eq_eff sign-inversion threshold is 60.87 % SoC (NEDC) / 69.92 % (FTP75)**
   for the actual CONTROL config — not the 66.41 % / 80.08 % in prior reports
   (stale, old config). [F]
2. **The inversion never triggers in any CONTROL deterministic trajectory**
   (0 / 1220 steps, 0 / 1876, all seeds). It occurs in 0.3–2.4 % of replay
   transitions, all in the early-training SoC-runaway phase (SoC > 61 %). [F]
3. **The action→T_CE map is continuous, monotone, and well-conditioned**
   (`dT_CE/da ≈ −35 Nm/unit`, no clipping flats, max step-jump ~0.03 Nm). [F]
4. **~58 Nm engine torque is unreachable across most of the 15–35 Nm band**
   (LPS capped at `U_MIN = −0.85`; max feasible `T_CE` ≈ 1.85 × demand ≈ 40–54
   Nm below ~33 Nm demand). Where reachable (T ≥ ~34 Nm) it sits at `a ≈ −0.85`,
   i.e. at the LPS clamp. ECMS's feasible grid uses the same `U_MIN`. [F]
5. **The immediate reward's argmax over the feasible grid is the deepest-LPS
   (U_MIN) action** whenever SoC < ~45 %. The 58-vs-35 preference is a shallow
   monotone up-slope (Δr ≈ +0.001 … +0.010). [F]
6. **The learned critic's `Q(T_CE)` curve peaks near 40–47 Nm and declines
   toward deep LPS** — its argmax `T_CE ≈ 40–47` is ~15–30 Nm below the
   reward's. [F]
7. **The fitted `Q̂` is systematically below its own 1-step Bellman target at
   the deep-LPS end** — mean residual **−0.0118** at `a_R*` vs **−0.0051** at
   `a_Q*`; `|resid@a_R*| > |resid@a_Q*|` in 19 / 20 states. [F]
8. **The discounted future term `γ·V(s')` is action-flat** (spread < 0.001);
   `argmax r == argmax (r + γV)` in 19 / 20 states → γ = 0.20 provides **no
   action-ranking signal** in this region. [F][M]
9. **The deep-LPS critic under-fit persists in the 35–50 Nm band with 38–54 %
   replay coverage** (residual −0.009 … −0.015); twin-Q disagreement there is
   0.004–0.013 (not far-OOD). [F]
10. **The actor mostly tracks its (mis-fit) critic** — mean `Q_loss` 0.0044
    (~2 % of |Q|); `dQ/da @ a_pi ≈ 0`. It undershoots the critic's argmax by
    ~10 Nm; it flips to OFF with a policy-variance spike (σ ≈ 1) in ~10 % of
    low-SoC mid-torque states. [F]
11. **Entropy has collapsed:** `ent_coef = 0.0017`; actor `log_std ≈ −1` to
    −2 (σ ≈ 0.14–0.39) in most states, spiking to σ ≈ 1 in the OFF-flip states. [F]
12. **ECMS at 30–35 Nm demand runs the engine OFF** (V1-D), while the reward's
    deep-LPS argmax runs it ~55–65 Nm — the reward is NOT pointing at the
    ECMS-optimal action here (V1: reward is a stiffer-battery Hamiltonian than
    ECMS). [F, from V1]

## 2. REJECTED HYPOTHESES

| hypothesis | why rejected |
|---|---|
| **Action-parameterization / representation difficulty** | 11B: map is smooth, monotone, well-conditioned; no clipping/saturation/dead-zone/discontinuity near the region; ~58 Nm is a *reachability* boundary (shared with ECMS), not a numerically pathological zone. Actor's ~35 Nm is a clean interior point. |
| **Temporal-credit / γ failure (future value flips the preference)** | 11D: `γ·V(s')` is action-flat (< 0.001 spread); `argmax r == argmax(r+γV)` in 19/20 states. γ = 0.20 gives no action-ranking signal here. Case E falsified. |
| **`eq_factor` mis-scaling drives the 15–35 Nm behaviour** | 11A + V1: the sign inversion never triggers in the CONTROL trajectory; V1's decisive matched-state test already showed the reward is *stiffer*-battery than ECMS, not softer. Not the 15–35 Nm mechanism. |
| **Actor cannot represent / reach the reward-preferred point** | 11B: fully representable where feasible; where "58 Nm" is infeasible it is a demand-dependent `U_MIN` limit, not an actor limitation. |
| **Pure exploration/coverage bottleneck (Case C) as the sole cause** | 11D/11C-D: the critic under-fit persists at 38–54 % replay coverage (35–50 Nm band); twin-Q disagreement is low. Coverage contributes at low SoC/demand but is not the dominant mechanism. |

## 3. SURVIVING HYPOTHESES (ranked by evidence)

### H-CRITIC — Critic function-approximation / training inconsistency at high engine load (Case D) — **PRIMARY**
* **Evidence for:** `Q̂` below its own 1-step Bellman target at the deep-LPS
  end in 19/20 states (mean −0.0118 vs −0.0051 at the critic argmax); the
  critic invents a `Q(T_CE)` downslope beyond ~50 Nm absent from both the
  immediate reward and the Bellman target; the effect persists with 38–54 %
  replay coverage; twin-Q disagreement is low (not far-OOD).
* **Evidence against:** the residual is modest (~2–12 % of |Q|); a weak SoC
  correlation (0.30) shows coverage still matters; the reward-optimal target it
  fails to reach (deep LPS) is itself *not* the ECMS-optimal action (V1), so
  "fix the critic" ≠ "close the ECMS gap" automatically.
* **Confidence: strongly supported (primary).**
* **Falsifier:** refine the critic (more fitting) on the **frozen** replay
  buffer with the actor/reward/γ/network frozen. If the deep-LPS Bellman
  residual shrinks toward 0 and the critic argmax moves toward the reward's
  argmax → confirmed (data was sufficient, fitting was not). If the residual
  persists → the data is insufficient → H-COVERAGE, not H-CRITIC.

### H-COVERAGE — Thin replay coverage of the high-engine-load region at low demand/SoC (Case C) — **CONTRIBUTING**
* **Evidence for:** `T_CE` 50–60 Nm coverage is 0–8 % in the 15–30 Nm bands;
  Bellman residual worsens at low SoC (corr 0.30); Phase 4 precedent (an
  analogous deadlock for OFF was coverage-fixable).
* **Evidence against:** the under-fit persists at 38–54 % coverage (35–50 Nm);
  twin-Q disagreement is low; the residual is not larger where coverage is
  thinner (35–50 band vs 25–30 band).
* **Confidence: plausible, contributing, confounded with H-CRITIC.**
* **Falsifier:** the H-CRITIC falsifier above separates them. If offline
  critic-refinement on frozen data does *not* fix the residual, targeted
  high-engine-load exploration becomes justified.

### H-ACTOR — Actor mildly undershoots its own critic; bimodal/unstable in low-SoC mid-torque states (Case A) — **SECONDARY**
* **Evidence for:** actor `T_CE` is ~10 Nm below the critic's argmax
  everywhere; in ~10 % of low-SoC states it flips to OFF with σ ≈ 1
  (`ent_coef 0.0017` → near-deterministic elsewhere).
* **Evidence against:** mean `Q_loss` only 0.0044; `dQ/da @ a_pi ≈ 0`; the
  actor is near-optimal *w.r.t. its critic* in 18/20 states.
* **Confidence: weakly supported, secondary.**
* **Falsifier:** if H-CRITIC is fixed and the actor still lags the corrected
  critic by > ~15 Nm / > 0.01 Q, H-ACTOR is promoted.

### H-REWARD-TARGET — The reward's deep-LPS preference is itself not the right target — **STRUCTURAL, from V1**
* **Evidence for:** V1-D — the reward's per-step argmin prefers more engine /
  less battery than ECMS; ECMS at 30–35 Nm runs OFF, the reward's argmax runs
  ~58 Nm LPS. Both differ from the actor's part-load ~35 Nm.
* **Confidence: established (V1), orthogonal to H-CRITIC/H-COVERAGE.** Means:
  even a perfectly-fit critic + perfectly-tracking actor would converge to the
  reward's deep-LPS point, not ECMS's OFF — so this diagnostic explains the
  *internal* inconsistency but not the full ECMS gap.

## 4. ROOT-CAUSE ASSESSMENT

**Single most likely current mechanism (for the observed ~58 Nm reward vs
~35 Nm actor discrepancy in 15–35 Nm):**

> **The trained critic systematically under-estimates the value of
> high-engine-load / deep-LPS actions relative to both the immediate reward and
> its own one-step Bellman target; the near-deterministic actor then faithfully
> tracks that mis-fit critic, operating the engine ~10 Nm softer than the
> critic's argmax and ~20–40 Nm softer than the reward.**

Distinguishing certainty:
* **Proven [F]:** (a) `γ·V(s')` is action-flat and does not re-rank actions
  (19/20) — **not a temporal/γ problem**; (b) the action map is clean and
  ~58 Nm is a demand-dependent `U_MIN` reachability boundary — **not an
  action-representation problem**; (c) the eq_eff inversion never fires in the
  CONTROL trajectory — **not that reward defect**; (d) the fitted `Q̂` is below
  its own Bellman target at the deep-LPS end in 19/20 states.
* **Strongly supported [H, high confidence]:** the dominant mechanism is
  **critic function-approximation / training inconsistency (Case D)**, because
  the under-fit persists at 38–54 % replay coverage and with low twin-Q
  disagreement.
* **Plausible [H]:** thin coverage at low demand/SoC (Case C) amplifies the
  critic error there; the actor's mild undershoot and low-SoC OFF-flips
  (Case A) are secondary contributors.
* **Unknown:** whether the critic under-fit is curable by more
  fitting/capacity/target-construction on the **existing** replay data, or
  needs **new** data — H-CRITIC and H-COVERAGE are confounded and only an
  intervention separates them. Also unknown (from V1, not this phase): whether
  the reward's deep-LPS target is even desirable — it is not the ECMS-optimal
  action for 15–35 Nm demand.

## 5. NEXT EXPERIMENT (ONE — proposed, NOT executed)

### EXP-P11-S1 — Offline critic refinement on the frozen replay buffer

**Objective.** Separate H-CRITIC (Case D) from H-COVERAGE (Case C): does more
critic fitting on the **existing** data remove the systematic deep-LPS Bellman
residual and move the critic's `T_CE` argmax toward the reward's preference?

**Single independent variable.** Number of additional **critic-only** gradient
steps on the frozen CONTROL replay buffer: `N ∈ {0 (= CONTROL), 50k, 150k, 400k}`
(one sweep axis).

**Everything frozen.** Actor weights (frozen — **not updated**); reward;
`eq_factor`; `k_fb`; `γ = 0.20`; `n_step = 1`; entropy coefficient (frozen at
the checkpoint value 0.0017); critic network architecture `[256,256]`, twin-Q;
`τ = 0.005`; optimiser (`lr 3e-4`, batch 512); **replay-buffer contents
(no new rollouts, no targeted exploration, no environment interaction at
all)**; target-construction formula. Only the critic parameters change, via the
existing SAC critic loss on minibatches sampled from the frozen buffer.

**Seeds.** 3 (the 3 CONTROL checkpoints + their own replay buffers).

**Training budget.** ≤ 400k critic gradient steps per seed (no env steps).
Pure offline; ~minutes–low-tens-of-minutes on CPU.

**Primary metric.** Mean Bellman residual `Q̂ − Q_target` at the deep-LPS
action (`a_R*`) across the 15–35 Nm matched-state set (the 11D metric), and the
critic's per-state `argmax_a min-Q` engine torque in those bands.

**Secondary metrics.** Residual at `a_Q*` and `a_pi`; twin-Q disagreement at
`a_R*`; critic-loss curve; `min-Q(a_R*) − min-Q(a_pi)`; residual vs replay
coverage per band; residual vs SoC.

**Success criterion (H-CRITIC confirmed).** Mean deep-LPS Bellman residual
`|Q̂ − Q_target|` at `a_R*` falls below **0.004** (from −0.0118) **and** the
critic's argmax `T_CE` in the 30–50 Nm bands rises by **≥ +10 Nm** toward the
reward's argmax, on ≥ 2 / 3 seeds — with the improvement **largest in the
best-covered band (35–50 Nm)**.

**Failure criterion (H-COVERAGE promoted).** Residual at `a_R*` stays **worse
than −0.008** and/or the critic argmax moves **< +3 Nm**, on ≥ 2 / 3 seeds,
after 400k critic steps — the fitted critic cannot reproduce the Bellman
target on this data ⇒ the data is insufficient ⇒ targeted high-engine-load
exploration becomes the justified next experiment.

**Rollback condition.** This experiment produces a **diagnostic critic only**;
it is never deployed as a controller (the actor is frozen and was trained
against the old critic). No rollback of the CONTROL is possible or needed —
nothing in `src/`, no checkpoint, and no CONTROL artefact is modified; all
outputs go to `results/phase11/EXP_P11_S1/`.

## 6. EXPERIMENTS NOT JUSTIFIED YET

| candidate | justified now? | reason |
|---|---|---|
| **`eq_factor` recalibration** | **NO** | V1 falsified the under-pricing direction; 11A shows the sign inversion never triggers in the CONTROL trajectory. |
| **`k_fb` recalibration** | **NO** | Phase 7: trained-policy P(OFF) flat across `k_fb ∈ {1.656, 2.5, 3.0}`; 11D shows the failure is critic-fit, not costate. |
| **γ sweep** | **NO** | 11D/11G: `γ·V(s')` is action-flat; `argmax r == argmax(r+γV)` (19/20). No mechanistic reason. |
| **Action reparameterisation / widening `U_MIN`** | **NO** | 11B: the map is clean; ~58 Nm is a demand-dependent `U_MIN` reachability boundary shared with ECMS. Widening `U_MIN` would push toward *more* LPS — the opposite of the ECMS strategy for this band (V1: ECMS runs OFF here). |
| **Targeted exploration (high-engine-load coverage)** | **NOT YET** | Confounded with the critic-fit hypothesis. Becomes justified **only if** EXP-P11-S1 hits its *failure* criterion (offline critic refinement on frozen data does not fix the residual). |
| **DP reference** | **DEFERRED** | `DP_STATUS = DEFERRED_NO_EXISTING_VALIDATED_SOLVER` (V2). Separate future task; not blocking. |
| **Actor-side change (policy head / entropy)** | **NOT YET** | 11E: the actor mostly tracks its critic (`Q_loss` 0.0044). Re-evaluate only after the critic is corrected. |

## 7. STOP

Diagnosis complete. Primary mechanism: **critic function-approximation /
training inconsistency at high engine load (Case D)**, with thin low-demand
coverage (Case C) contributing and a mild actor undershoot (Case A) secondary;
**temporal-credit/γ and action-parameterisation are falsified as mechanisms**;
the eq_eff inversion is a **latent defect** (never triggered by the CONTROL
trajectory). One next experiment is specified (EXP-P11-S1, offline critic
refinement on the frozen buffer) with pre-defined success/failure criteria that
separate H-CRITIC from H-COVERAGE.

**No code modified. No training run. No calibration applied. No fix
implemented. Awaiting human review before any intervention.**
