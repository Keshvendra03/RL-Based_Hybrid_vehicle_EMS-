# PHASE 7 — ECONOMIC-VALUE / COSTATE FORENSIC EXPERIMENT

**Status: FORENSIC CALIBRATION COMPLETE. No training run. No physics / plant /
benchmark / evaluator / SAC-algorithm / exploration change.**

**Verdict — the residual SAC-vs-benchmark gap is NOT primarily an economic
(equivalent-factor / costate) valuation error.** The CONTROL policy's effective
battery price already matches the ECMS closed-loop reference; the `k_fb` axis is
an already-measured flat plateau; and where the SAC critic *does* mis-rank the
ECMS action, the error is uncorrelated with the battery price. The dominant
remaining mechanism is **actor-side: the squashed-Gaussian policy will not place
its mass on the engine-OFF region of the action range at 15–35 Nm even though
its own critic's arg-max — and ECMS — put it there.**

→ **Decision-tree classification: CASE D** (critic ranking ≈ correct, actor
systematically displaced), shading into **CASE E** (unimodal policy class
inadequate for the bimodal Q) if the one remaining actor-side lever fails.

Raw output: `results/phase7/raw/phase7_forensics_{NEDC,FTP75}.txt`
Machine-readable: `results/phase7/data/*.json`, `results/phase7/data/matched_states_{NEDC,FTP75}.csv`
Figures: `results/phase7/figures/phase7_summary_{NEDC,FTP75}.png`,
`counterfactual_{NEDC,FTP75}.png`

> **How to read this report.** Every section quotes the corresponding
> instruction from the Phase 7 brief (in a block-quote) and then answers it
> with the measured evidence, so the question each number answers is explicit.

---

## §1 — LOCK THE BASELINE

> *"Use the currently validated best configuration as CONTROL. Do not silently
> change anything. Record the exact: git commit, model checkpoint, seed, gamma,
> n_step, action representation, eq_factor, k_fb, target_entropy, learning rate,
> batch size, replay size, gradient steps, lookahead, network architecture,
> training steps. Also retain the previously validated linear k_fb=1.656,
> gated k_fb=1.656, gated k_fb=2.5, ECMS, advanced rule-based."*

Full machine-readable lock: `results/phase7/data/00_baseline_lock.json`.

| field | CONTROL value |
|---|---|
| git commit (before Phase 7) | `f1f45c559e126e67a7fc01634895a22b6e08e8de` |
| CONTROL identity | gated `k_fb=2.5` — the Phase-5 validated candidate |
| checkpoints NEDC | `models_p5s0_k2.5/NEDC` (seed0, V_CE 3.6862), `models_p5_k2.5/NEDC` (seed1, 3.8431), `models_p5_k2.5_s2/NEDC` (seed2, 3.7704) |
| checkpoints FTP75 | `models_p5f_k2.5_s0/1/2` (3.2699 / 3.3041 / 3.2926) |
| training git commit | `9a125adc7577ec5a1d66962ef32ebb91ce5d5497` |
| seeds | {0, 1, 2} |
| gamma | **0.20** |
| n_step | **1** |
| action representation | **`modeaware_gated`** |
| eq_factor | **0.2717 (NEDC) / 0.4981 (FTP75)**  = λ₀ / 4.8309 |
| k_fb | **2.5** |
| target_entropy | **auto** (= −1.0, action_dim = 1) |
| learning rate | 3e-4 |
| batch size | 512 |
| replay size | 300 000 |
| gradient steps | 16 (per `train_freq`=64 env steps) |
| lookahead | 5 |
| network architecture | MLP [256, 256], twin-Q, tanh-squashed diag-Gaussian actor |
| training steps | 150 000 |
| lambda_soc / soc_deadband / tau | 2.0 / 0.10 / 0.005 |
| CONTROL V_CE (3-seed mean) | **NEDC 3.7666**, FTP75 3.2889 |

**Comparison references (NOT training targets), retained:**

| ref | V_CE NEDC | V_CE FTP75 | notes |
|---|---|---|---|
| linear `k_fb=1.656` | 3.7727 (3/3 CS) | — | `models_seed_NEDC_s0/s2`, `models_final_NEDC_s2` |
| gated `k_fb=1.656` | 3.8824 (**1/3 CS**) | **3.2460** (best FTP75) | `models_p4_gated_g20`,`models_p4g_N0/N2` / `models_p4g_F0/1/2` |
| gated `k_fb=2.5` | = CONTROL | = CONTROL | |
| gated `k_fb=3.0` | 3.7840 (3/3 CS) | — | `models_p5s0_k3.0`,`models_p5_k3.0`,`models_p5s2_k3.0` |
| ECMS (λ₀, k_fb=8) | **3.1887** (λ₀ 1.3125) | **2.8097** (λ₀ 2.4062) | `src/baselines/ecms.py` |
| advanced rule-based | **3.5056** | **3.2323** | `src/baselines/advanced_rule_based.py` |

**Validated conversion (Phases 2 & 5, unchanged):**
`lambda_ECMS = eq_factor_eff × 4.8309`, where
`eq_factor_eff = eq_factor + k_fb·(0.5 − soc_before)`.
ECMS's own costate law is `λ₀ + 8.0·(0.5 − soc)`, i.e. **`k_fb = 8.0/4.8309 =
1.656` in env units.** The CONTROL uses `k_fb = 2.5` (feedback slope 1.51× ECMS).

---

## §2 — PURE FORENSIC CALIBRATION: EFFECTIVE BATTERY PRICE

> *"Reconstruct the exact effective equivalent factor / costate seen by the
> reward at every timestep. Report min / p5 / p25 / median / p75 / p95 / max /
> mean / std. Convert into ECMS-equivalent units. Compare directly against
> lambda_ECMS = 1.3125. Do this separately for SoC < 40 / 40–45 / 45–50 /
> 50–55 / > 55 and for torque bands 0–15 / 15–30 / 30–35 / 35–50 / 50–75 / > 75.
> Do NOT merely report the global mean."*

Reconstructed over the **CONTROL deterministic evaluation rollout** (seed 0),
moving-traction steps only. Data: `results/phase7/data/effective_price_{C}.json`.

### Global percentiles — effective price in ECMS units (× 4.8309)

| cycle | min | p5 | p25 | median | p75 | p95 | max | mean | std |
|---|---|---|---|---|---|---|---|---|---|
| **NEDC** | 1.301 | 1.870 | 2.690 | **2.817** | 3.018 | 3.341 | 3.964 | 2.802 | 0.420 |
| **FTP75** | 1.721 | 2.083 | 2.523 | **2.721** | 3.052 | 3.400 | 3.699 | 2.764 | 0.390 |

(Same, in raw reward/liter units: NEDC median 0.583, FTP75 median 0.563.)

* **Fraction of the moving episode priced above λ_ref = 1.3125: NEDC 98.8%,
  FTP75 100%.** In isolation this looks like gross over-pricing (median ≈ 2.1–2.2×
  λ_ref).
* **BUT the correct reference is not the static λ₀.** ECMS is a *closed-loop*
  controller (`k_fb = 8.0`). Its **own effective λ over its own charge-sustaining
  rollout** has median **2.776 (NEDC) / 2.846 (FTP75)** — essentially identical
  to the SAC CONTROL's 2.82 / 2.72. Both controllers run a deep-SoC-excursion
  strategy (median visited SoC **NEDC 37.5 %**, FTP75 47.4 %), so both experience
  a feedback-inflated price ≈ 2.8 ECMS units. **The SAC CONTROL is not
  over-pricing battery energy relative to the proven-optimal reference.**

### By SoC band — effective price (ECMS units), NEDC / FTP75

| SoC band | NEDC median (×λ_ref) | FTP75 median (×λ_ref) |
|---|---|---|
| < 40 % | 2.87 (2.18×) | — |
| 40–45 % | 2.42 (1.84×) | 2.90 (2.21×) |
| 45–50 % | 1.83 (1.39×) | 2.79 (2.13×) |
| 50–55 % | 1.30 (0.99×) | 2.34 (1.78×) |
| > 55 % | — | 1.9 (1.5×) |

The SAC price **collapses to ≈ λ_ref exactly at SoC 50–55 %** on NEDC (by
construction — the base `eq_factor` = λ_ref/4.8309 and the feedback term
vanishes at target). The high multipliers are entirely a consequence of the
policy operating *below* target, which the ECMS reference also does.

### By torque band — price experienced while deciding in that band (NEDC)

| T band | n | price median (ECMS u) | SoC median | ×λ_ref |
|---|---|---|---|---|
| 0–15 | 156 | 2.798 | 0.377 | 2.13× |
| 15–30 | 226 | 2.793 | 0.377 | 2.13× |
| 30–35 | 117 | 2.738 | 0.382 | 2.09× |
| 35–50 | 41 | 2.727 | 0.383 | 2.08× |
| 50–75 | 80 | 3.095 | 0.352 | 2.36× |
| > 75 | 70 | 3.022 | 0.358 | 2.30× |

The price is nearly flat across torque bands (it is a function of SoC only); the
15–35 Nm decisions — the largest error region — are made at a price of
≈ 2.75 ECMS units, the *same* as everywhere else, i.e. there is no
torque-band-specific pricing pathology.

**§2 conclusion:** measured against the static λ₀ the price is 2.1× high; measured
against ECMS's actual closed-loop pricing it is a match. **No state-conditioned
over-pricing of battery discharge relative to the reference controller.**

---

## §3 — TEST THE CENTRAL HYPOTHESIS (matched states)

> *"For matched states, calculate Δr(OFF−ASSIST), ΔQ(OFF−ASSIST), ΔQ(OFF−LPS),
> and compare the SAC Q-ranking with the ECMS-selected action. … Focus
> particularly on 15–30, 30–35, 35–50, 50–75 Nm."*

Matched states = a fresh, unpatched `EMSEnv` deep-copied at **every traction
step along the charge-sustaining ECMS SoC trajectory** (realistic operating
SoC), then probed with the CONTROL SAC critic/actor and with candidate OFF /
ASSIST / LPS actions. Per-state records: `results/phase7/data/matched_states_{C}.csv`.
Summary: `..._summary.json`.

### NEDC

| region | n | Δr(OFF−ASSIST) median (>0%) | ΔQ(OFF−ASSIST) median (>0%) | OFF% SAC / argmaxQ / ECMS |
|---|---|---|---|---|
| **15–30** | 90 | −0.0034 (42 %) | **+0.0064 (62 %)** | **47 / 72 / 71** |
| **30–35** | 90 | −0.0021 (14 %) | **+0.0234 (91 %)** | **0 / 87 / 40** |
| 35–50 | 41 | −0.0045 (15 %) | +0.0008 (51 %) | 15 / 17 / 29 |
| 50–75 | 80 | ≈0 (median) † | ≈0 (median) † | 0 / 5 / 20 |

† at 50–75 Nm engine-OFF is mostly motor-infeasible; the OFF-probe is clamped
and the mean is a −1.0 artefact — the median (≈0) is the meaningful figure.

### FTP75

| region | n | Δr(OFF−ASSIST) median (>0%) | ΔQ(OFF−ASSIST) median (>0%) | OFF% SAC / argmaxQ / ECMS |
|---|---|---|---|---|
| **15–30** | 90 | **+0.0077 (82 %)** | **+0.0120 (89 %)** | **33 / 80 / 89** |
| **30–35** | 80 | +0.0020 (62 %) | **+0.0080 (78 %)** | **5 / 31 / 48** |
| 35–50 | 90 | −0.0006 (48 %) | +0.0050 (64 %) | 6 / 33 / 38 |
| 50–75 | 90 | −0.0065 (32 %) | −0.0178 (24 %) | 0 / 4 / 11 |

**Central finding.** In the two regions that carry the NEDC gap (15–35 Nm):

* **ΔQ(OFF−ASSIST) is positive** — the SAC critic *does* prefer engine-OFF
  (NEDC 30–35: +0.0234, positive in **91 %** of states).
* **The SAC critic's own arg-max-Q OFF share (72 % / 87 % NEDC, 80 % / 31 %
  FTP75) matches or exceeds ECMS's** (71 % / 40 %, 89 % / 48 %).
* **The SAC deterministic actor delivers 47 % / 0 % OFF (NEDC), 33 % / 5 %
  (FTP75).**

The disagreement is **not reward-vs-critic** and **not critic-vs-ECMS** — it is
**actor-vs-its-own-critic.** At NEDC 30–35 Nm the critic wants OFF 87 % of the
time and the actor never selects it.

This *sharpens* the Phase 6 correction: Phase 5B claimed "reward favours OFF,
critic disagrees"; Phase 6 corrected that to "reward and critic agree on
ASSIST at 30–35". Phase 7, using the realistic ECMS-trajectory matched states,
shows the **critic actually favours OFF at 30–35** and it is the **actor** that
sits in LPS.

---

## §4 — CRITICAL COUNTERFACTUAL ANALYSIS (dense action grid)

> *"For matched states, evaluate the SAME state under a dense action grid
> a ∈ [−1, 1]. For every action calculate immediate reward, SAC critic Q(a),
> torques, battery power, SoC transition, feasibility, mode. Identify the action
> maximizing immediate reward, maximizing SAC Q, ECMS, rule-based, actor.
> Produce Q(a) and reward(a) plots. This is the central diagnostic."*

121-point grid, env-deepcopy reward at every grid point, min-Q from the twin
critic. Arrays + representative-state plots:
`results/phase7/figures/counterfactual_{NEDC,FTP75}.png`,
`results/phase7/data/counterfactual_{C}.json`.

**NEDC 30–35 Nm representative state (SoC 25 %):**

* `r(a)` decreases almost monotonically toward `a = −1` (deep LPS/charge) — the
  **immediate reward gradient points at charging**, because at low SoC the
  feedback term makes stored energy valuable to bank.
* `min-Q(a)` is **bimodal**: a shoulder on the LPS side and a **strong global
  maximum at `a ≈ +0.75`, deep inside the engine-OFF band.**
* Markers: `argmax-Q ≈ +0.75` (OFF), `ECMS ≈ −1.0`†, **actor mean `≈ −0.8`
  (LPS)**. †the ECMS marker is the *inverse-mapped* Hamiltonian `u`; at this
  particular low-SoC snapshot ECMS also banks, but across the 90-state
  aggregate ECMS OFF-share is 40 %.
* **The deterministic actor sits ~1.5 action-units away from its own critic's
  arg-max, on the opposite side of the range.**

**NEDC 15–30 Nm:** `Q(a)` peaks near the LPS/OFF boundary and stays high through
OFF; actor at `a ≈ −0.85`, argmax-Q at `a ≈ −0.55` … `+0.3`.

**35–50 / 50–75 Nm:** `Q(a)` monotone or noisy-multimodal; actor ≈ argmax-Q
(aligned); a hard feasibility cliff at `a ≈ 0.4` where the motor envelope can no
longer carry the demand.

**§4 conclusion:** the central diagnostic confirms a **bimodal Q with the
unimodal actor parked on the wrong (LPS) lobe** in exactly the 15–35 Nm band,
while the immediate-reward surface — being myopic — genuinely does slope toward
charging at low SoC. The critic has correctly learned the long-run value of OFF;
the actor has not moved there.

---

## §5 — IS THE CRITIC ERROR ECONOMIC OR TEMPORAL?

> *"For each matched state calculate ERROR_critic = Q_SAC(a_ECMS) −
> Q_SAC(a_SAC) and ERROR_reward = r(a_ECMS) − r(a_SAC), and compare with
> Δcostate, ΔSoC, battery discharge, future SoC consequence. Determine whether
> SAC is rejecting ECMS primarily because: A battery discharge over-priced /
> B battery charging under-priced / C SoC feedback too strong / D SoC feedback
> too weak / E terminal reward improperly propagated / F critic bootstrapping
> bias / G immediate reward and long-term Q economically inconsistent /
> H continuous action representation creates a systematic operating-point
> problem. Do NOT choose the answer in advance. Measure it."*

`a_SAC` = deterministic actor action; `a_ECMS` = grid action whose mapped `u`
matches the Hamiltonian `u` at that state.

| | NEDC 15–30 | NEDC 30–35 | NEDC 35–50 | FTP75 15–30 | FTP75 30–35 | FTP75 35–50 |
|---|---|---|---|---|---|---|
| median **ERROR_reward** | +0.0020 | +0.0007 | +0.0008 | +0.0021 | +0.0013 | +0.0010 |
| median **ERROR_critic** | −0.0075 | −0.0022 | −0.0091 | −0.0058 | −0.0028 | −0.0019 |
| critic rejects ECMS *while reward accepts* | 63 % | 50 % | 51 % | 42 % | 30 % | 32 % |
| corr(ERROR_critic, eq-price) | **−0.00** | +0.11 | +0.18 | +0.13 | −0.04 | +0.15 |
| corr(ERROR_critic, ERROR_reward) | −0.12 | +0.61 | +0.64 | +0.43 | +0.71 | +0.61 |
| corr(ERROR_critic, ΔSoC) | −0.13 | +0.58 | +0.72 | +0.21 | +0.60 | +0.55 |

Reading against the A–H menu:

| option | verdict | evidence |
|---|---|---|
| **A** battery discharge over-priced | **REJECTED** | §2/§8: SAC effective price ≈ ECMS effective price; `corr(ERROR_critic, eq-price) ≈ 0` on both cycles |
| **B** battery charging under-priced | **REJECTED** | LPS is *over*-selected by the actor, but that is a low-SoC reward-gradient effect, correct in direction; no pricing asymmetry (`eq_factor` is symmetric) |
| **C** SoC feedback too strong | **PARTIAL, not actionable** | NEDC operates at SoC 37 %, so the feedback adds ≈ +1.5 ECMS units; but lowering `k_fb` 2.5→1.656 is already measured (§6) — it breaks charge-sustaining on NEDC and does **not** improve fuel, and barely moves actor P(OFF) at the NEDC operating SoC |
| **D** SoC feedback too weak | **REJECTED** | opposite direction |
| **E** terminal reward mis-propagated | **REJECTED** | γ = 0.20 → terminal signal has negligible reach; previously measured at 0.77 % of the reward |
| **F** critic bootstrapping bias | **MINOR** | `ERROR_critic < 0` at the exact ECMS point ⇒ a small negative bias on high-engine-load / OFF actions, but the critic's *arg-max mode* still matches ECMS (§3); n_step = 1, γ = 0.20 bound the bootstrap depth |
| **G** immediate reward and long-term Q economically inconsistent | **PRESENT, but in the RIGHT direction** | `r(a)` slopes to LPS at low SoC while `Q(a)` peaks at OFF — the critic is *more* correct than the myopic reward, not defective |
| **H** continuous action representation → operating-point problem | **STRONGEST MATCH** | bimodal `Q(a)`; actor mean ~1.5 action-units from arg-max-Q at 30–35 Nm; the gated map hands OFF a contiguous 40 % block at `a ∈ [+0.2, +1.0]` and the squashed-Gaussian actor will not place its mean there |

**§5 verdict: the critic error is NEITHER primarily economic (A/B/C/D) NOR
temporal (E/F).** `ERROR_reward ≥ 0` everywhere (the one-step reward is fine with
the ECMS action) and `ERROR_critic` is small and uncorrelated with the battery
price. The binding problem is **G + H**: a myopic reward surface that locally
favours charging, a critic that has *correctly* learned OFF's long-run value,
and a **unimodal continuous policy that cannot sit on the OFF lobe** the critic
prefers.

---

## §6 — k_fb = 1.656 vs 2.5 (vs 3.0)

> *"Determine whether k_fb=2.5 is economically over-penalizing battery
> discharge. Create a matched-state table: region / SoC band / k_fb /
> eq-price median / Δr OFF-ASSIST / ΔQ OFF-ASSIST / P(OFF) / fuel contribution.
> Use the same states wherever possible. Do NOT infer causality from aggregate
> fuel alone."*

Same matched states (ECMS trajectory), three trained checkpoints
(`ref_k1.656_gated`, `control_k2.5_gated`, `ref_k3.0_gated`). Full table:
`results/phase7/data/kfb_compare_{C}.json`.

### NEDC (excerpt — the operating SoC band 32–42 %)

| variant | k_fb | region | eq med (ECMS u) | Δr(OFF−ASST) | ΔQ(OFF−ASST) | P(OFF) actor |
|---|---|---|---|---|---|---|
| ref  | 1.656 | 15–30 | 2.63 | −0.0005 | +0.0042 | **48.0 %** |
| CTRL | 2.500 | 15–30 | 3.31 | −0.0005 | +0.0128 | **48.0 %** |
| ref  | 3.000 | 15–30 | 3.70 | −0.0005 | −0.0009 | **47.7 %** |
| ref  | 1.656 | 35–50 | 2.63 | −0.0028 | +0.0035 | 38.9 % |
| CTRL | 2.500 | 35–50 | 3.31 | −0.0028 | +0.0011 | 37.5 % |
| ref  | 3.000 | 35–50 | 3.71 | −0.0028 | −0.0079 | 38.2 % |

**At the NEDC operating SoC the eq-price moves 2.63 → 3.31 → 3.70 ECMS units as
`k_fb` goes 1.656 → 2.5 → 3.0, and the actor's P(OFF) does not respond
(48 % → 48 % → 48 %).** `k_fb` is not the lever on OFF usage there.

### FTP75 (SoC band 40–48 %)

| variant | k_fb | region | P(OFF) actor |
|---|---|---|---|
| ref 1.656 | 15–30 | **82.4 %** |
| CTRL 2.5  | 15–30 | 58.0 % |
| ref 1.656 | 30–35 | **53.2 %** |
| CTRL 2.5  | 30–35 | 11.0 % |
| ref 1.656 | 35–50 (46–54 %) | 47.8 % |
| CTRL 2.5  | 35–50 (46–54 %) | 28.4 % |

On FTP75 (operating SoC ≈ 47 %, near target) the actor **is** responsive to
`k_fb`, and gated `k_fb = 1.656` is the best FTP75 config (3.2460 vs 2.5's
3.2889). But FTP75 is already at the rule-based benchmark, so the prize is small.

**Is k_fb = 2.5 economically over-penalizing discharge?** *Relative to the static
λ₀* — numerically yes (2.1×). *Relative to ECMS's actual closed-loop pricing* —
no (§2). And the **behavioural** answer from the trained multi-seed sweep already
on disk is decisive:

| gated k_fb | NEDC 3-seed mean | NEDC CS | note |
|---|---|---|---|
| **1.656** | 3.8824 | **1/3** | "ECMS-matched" slope — loses charge-sustaining, worse fuel |
| 2.0 | (1 seed) | — | |
| **2.5** | **3.7666** | 3/3 | CONTROL |
| **3.0** | 3.7840 | 3/3 | statistically tied with 2.5 |
| 4.0 / 5.0 | (seeded) | — | |

`k_fb ∈ [2.0, 3.0]` is a **flat plateau** (3.766 → 3.784, both 3/3 CS). Moving
toward the ECMS-slope-matched 1.656 trades charge-sustaining for nothing.
**`k_fb` is not the correct lever for the residual gap.**

---

## §7 — ECMS IS A REFERENCE, NOT A LABEL

> *"Do NOT train SAC to imitate ECMS. Do NOT add ECMS actions to the replay
> buffer. Do NOT add an imitation loss. ECMS is being used only as an
> independent optimal-control reference for forensic comparison."*

Respected. ECMS appears in Phase 7 only as (a) a reference costate law for the
price comparison, (b) a reference SoC trajectory for matched-state selection,
and (c) a reference action in `ERROR_reward`/`ERROR_critic`. **No ECMS action
enters any buffer, target, or loss. No training was run at all.**

---

## §8 — COSTATE SWEEP: JUSTIFIED?

> *"If — and ONLY if — the forensic evidence demonstrates systematic
> over-pricing of battery discharge, perform a one-variable sweep of the
> economic feedback parameter. … First calculate what k_fb would be required
> for the median effective price to approach the validated ECMS λ₀ = 1.3125.
> Then construct a small scientifically justified sweep around that value.
> Do not assume that the correct answer is k_fb=1.656 or 2.5."*

Derivation from the **measured** visited-SoC distribution
(`results/phase7/data/required_kfb_{C}.json`):

* NEDC median visited SoC (pre-decision, moving) = **37.5 %** (p25 35.9 %, p75 38.6 %).
* FTP75 median visited SoC = **47.4 %**.
* `eq_factor_eff(median_SoC)` in ECMS units `= 1.3125 + k_fb·4.8309·(0.5 − SoC_med)`.
* **To bring the *median* effective price to λ₀ = 1.3125** requires
  `k_fb·(0.5 − SoC_med) = 0`. Because the base `eq_factor` already equals
  `λ₀ / 4.8309`, and the policy operates *below* 0.5, this needs **`k_fb ≤ 0`**
  — which is `k_fb = 0` flat pricing, already refuted in Phases 4–5 (SoC
  runaway, `1/3` CS).
* **To reproduce ECMS's own costate-law slope**: `k_fb = 8.0 / 4.8309 = 1.656`.
* **To match ECMS's *median effective λ*** over its own rollout:
  **NEDC `k_fb = 2.43`** (≈ the current 2.5!), FTP75 `k_fb = 3.50`.

So the CONTROL's `k_fb = 2.5` already puts the NEDC median effective price
(2.82) essentially on top of ECMS's own median effective λ (2.78). **There is no
positive `k_fb` that both (a) brings the median price to the static λ₀ and
(b) keeps the policy charge-sustaining.** The only defensible calibration
targets are `k_fb = 1.656` (slope match — *already trained, 1/3 CS, worse fuel*)
and `k_fb ≈ 2.43` (median match — *already the CONTROL*).

Nominal "scientifically-justified minimal sweep" = **{1.656, 2.08, 2.5}**. Every
point in it is already covered by trained checkpoints (`k_fb ∈ {1.656, 2.0, 2.5,
3.0, 4.0, 5.0}` all exist), and the 2.0–3.0 sub-range is a measured flat
plateau.

> **DECISION: the §8 precondition ("forensic evidence demonstrates systematic
> over-pricing of battery discharge") is NOT met.** Against the correct
> closed-loop reference there is no over-pricing; against the static λ₀ the only
> corrective `k_fb` is ≤ 0, which is refuted; and the corrective sweep has, in
> effect, already been run. **No new costate sweep is authorized by this
> diagnosis.** (If the user wants a formal 3-seed confirmation, the single
> missing grid cell is `k_fb = 2.0` at seeds 0 & 2 — low expected value,
> ~35 min compute — see §9.)

---

## §9 — TRAINING DESIGN (for the record; not executed)

> *"If a sweep is justified: one parameter only, gamma frozen at 0.20, n_step 1,
> action representation frozen, target entropy frozen, architecture frozen, lr /
> batch / replay / lookahead frozen, reward frozen except the single
> investigated economic parameter, same seeds, same training budget. Use at
> least 3 seeds for any candidate that survives the smoke test."*

Not executed — §8 did not justify a sweep. Timing was measured for feasibility:
a 150 k-step NEDC seed ≈ **12–15 min** on this machine (CPU torch 2.13, sb3
2.9.0); a full 3-seed × 3-candidate NEDC sweep ≈ 2–3 h; FTP75 ≈ 1.4× that. The
frozen-variable protocol above is recorded so any future sweep is
one-variable-clean.

---

## §10 — REQUIRED PERFORMANCE METRICS (CONTROL, 3 seeds)

> *"For every candidate report vehicle-level (V_CE_equiv, fuel, elec, final SoC,
> ΔSoC, SoC min/max, violations), mode-level (OFF/ASSIST/LPS/REGEN%, engine-on
> time), operating-point level (mean |T_CE|, |T_EM|, torque distributions), and
> a regional error budget. Do NOT report only the total V_CE."*

No new candidate exists (no training). The CONTROL field set is reproduced from
the authoritative evaluator and the Phase-5B/6 records:

**NEDC CONTROL (gated k_fb=2.5), 3 seeds:** V_CE 3.6862 / 3.8431 / 3.7704 →
**mean 3.7666 ± 0.0785**, 95 % CI [3.678, 3.855]; ΔSoC +0.28 / −0.72 / +0.23 pp
→ **3/3 charge-sustaining**; 0 constraint violations; OFF 37.8–39.8 %,
ASSIST 12.0–16.6 %, LPS 27.9–32.9 %, REGEN ≈ 17 %; engine-on ≈ 430–470 s.

**FTP75 CONTROL, 3 seeds:** V_CE 3.2699 / 3.3041 / 3.2926 → **mean 3.2889 ±
0.0174**; ΔSoC −0.53 / −0.51 / −0.07 pp → 3/3 CS; 0 violations.

**Operating-point signature (matched-demand, §11):** in **every** torque band
from 15 Nm up, the SAC CONTROL runs the engine at **lower torque when it is on**
than ECMS does — NEDC engine |T_CE| when on: 32 vs 40 (15–30), **35 vs 58**
(30–35), 50 vs 68 (35–50), 70 vs 95 (50–75), 94 vs 107 (>75). SAC runs the
engine **softer and more often**; ECMS runs it **harder and less often**.

**Regional ΔFuel vs advanced rule-based** (NEDC, from Phase-5B/6, unchanged —
Phase 7 ran no training): brake +0.001, 0–15 −0.012, **15–30 +0.368**,
30–50 +0.371, 50–75 −0.232, >75 −0.314, TOTAL +0.181. The **15–30 Nm term
(+0.368) is unchanged by every intervention in Phases 4–7.**

---

## §11 — SAC vs ECMS GAP DECOMPOSITION

> *"For the best surviving candidate, perform matched-state SAC-vs-ECMS
> decomposition. Calculate ΔFuel, ΔElec, ΔTotal for every torque region. Then
> separate the gap into 1 mode-selection, 2 battery-energy-management,
> 3 engine operating-point, 4 SoC-equivalence. We need to know what fraction of
> the remaining ECMS gap is actually theoretically addressable by this SAC
> formulation."*

CONTROL seed 0 vs ECMS, **demand-aligned exactly** (`max|T_SAC − T_ECMS| =
0.00e+00`). Data: `results/phase7/data/ecms_gap_{C}.json`.

### NEDC — total gap +0.4975 L/100km (SAC 3.686 vs ECMS 3.189)

| region | ΔFuel | ΔElec | ΔTotal | SAC OFF% / ECMS OFF% | SAC eng \|T_CE\| / ECMS |
|---|---|---|---|---|---|
| brake | −0.001 | +0.012 | +0.010 | 0 / 0 | — |
| 0–15 | +0.018 | +0.123 | **+0.141** | 35 / 35 | — |
| **15–30** | **+0.315** | −0.120 | **+0.196** | 58 / 77 | 32 / 40 |
| **30–35** | **+0.240** | −0.118 | **+0.122** | **4 / 49** | 35 / 58 |
| 35–50 | +0.050 | −0.024 | +0.026 | 51 / 63 | 50 / 68 |
| 50–75 | +0.002 | +0.001 | +0.003 | 0 / 22 | 70 / 95 |
| > 75 | −0.127 | +0.127 | +0.000 | 0 / 0 | 94 / 107 |

### FTP75 — total gap +0.4601 L/100km (SAC 3.270 vs ECMS 2.810)

| region | ΔFuel | ΔElec | ΔTotal | SAC OFF% / ECMS OFF% | SAC eng \|T_CE\| / ECMS |
|---|---|---|---|---|---|
| brake | +0.001 | +0.048 | +0.049 | 0 / 0 | — |
| 0–15 | +0.018 | +0.101 | +0.119 | 38 / 21 | 14 / 6 |
| **15–30** | **+0.149** | −0.013 | **+0.136** | 81 / 89 | 31 / 32 |
| 30–35 | +0.080 | −0.043 | +0.037 | 42 / 64 | 43 / 53 |
| **35–50** | **+0.200** | −0.119 | **+0.081** | 30 / 58 | 54 / 71 |
| 50–75 | +0.040 | −0.028 | +0.011 | 1 / 12 | 76 / 86 |
| > 75 | −0.032 | +0.060 | +0.028 | 0 / 0 | 105 / 113 |

### Four-way split

| component | NEDC | FTP75 | interpretation |
|---|---|---|---|
| **1 — mode selection** (SAC uses OFF less than ECMS where it should) | **≈ +0.32** (15–35 Nm) | **≈ +0.22** (30–50 Nm) | SAC OFF 4 % vs ECMS 49 % at NEDC 30–35; the single largest lever |
| **2 — battery-energy management** (net ΔElec) | ≈ +0.001 | ≈ +0.006 | negligible — SAC's SoC ledger ≈ ECMS's; **not** the problem |
| **3 — engine operating-point** (fuel/on-second at matched OFF%) | ≈ +0.03–0.13 | ≈ +0.06–0.11 | SAC runs the engine 10–40 % softer than ECMS in every band; real, secondary |
| **4 — SoC-equivalence residual** | ≈ −0.14 | ≈ −0.08 | SAC banks slightly more regen than ECMS at high torque (partial credit) |
| **0–15 Nm over-EV penalty** (separate) | +0.141 | +0.119 | SAC runs OFF at trivial loads then pays to recharge — Phase-5B signature, persists |

**What fraction is theoretically addressable by this SAC formulation?**

* **Mode-selection (≈ 60–65 % of the gap): addressable in principle** — the
  critic *already* ranks OFF above ASSIST at 15–35 Nm; only the actor fails to
  act on it. A policy that tracked its own arg-max-Q would recover most of this.
* **Engine operating-point (≈ 10–25 %): partially addressable** — needs the
  continuous action to select a *higher engine load when the engine is on*; the
  gated map compresses the ASSIST/engine-load sub-range, so this may be
  representation-limited.
* **0–15 Nm over-EV (≈ 25 %): addressable** — a mild penalty / preview signal
  discouraging OFF at trivial load; but note this partly *funds* the SoC that
  makes later OFF possible.
* **Battery-energy management (≈ 1 %): already solved.**

Net: **≈ 70–80 % of the SAC−ECMS gap is theoretically reachable within this SAC
formulation, gated mostly by the actor, not the reward.** The remainder
(engine-load fine-positioning) may need a richer action head.

---

## §12 — DECISION TREE CLASSIFICATION

> *"CASE A: costate calibration fixes Q ranking and fuel improves → continue
> with tuned economic factor. CASE B: fixes Q ranking but fuel does not improve
> → dominant problem is engine operating-point / action representation.
> CASE C: does not fix Q ranking even though price matches ECMS → investigate
> critic bias / bootstrapping / target construction. CASE D: critic ranking
> becomes correct but actor remains systematically displaced → investigate SAC
> actor optimization and policy parameterization. CASE E: neither critic nor
> actor can reproduce the desired continuous action structure despite correct
> reward valuation → propose a different policy parameterization. CASE F: SAC
> remains inferior after economic calibration and critic/actor diagnostics →
> consider an alternative RL algorithm."*

| case | applies? | why |
|---|---|---|
| A | **NO** | calibration was not needed and would not help — the Q ranking already prefers OFF at 15–35 Nm; fuel does not track `k_fb` in [2.0, 3.0] |
| B | **PARTIALLY** | engine operating-point *is* a real secondary component (§10/§11: SAC runs the engine softer than ECMS everywhere); the gated map may compress the engine-load sub-range |
| C | **NO (mostly)** | the critic's *arg-max mode* is right (matches ECMS); only a small negative `ERROR_critic` on the exact ECMS point, uncorrelated with price — not a gross bias |
| **D** | **YES — primary** | **"critic ranking ≈ correct, actor systematically displaced."** NEDC 30–35 Nm: argmax-Q OFF 87 %, actor OFF 0 %. FTP75 15–30: argmax-Q 80 %, actor 33 %. §4: actor mean ~1.5 action-units from arg-max-Q, on the wrong lobe of a bimodal Q |
| **E** | **LIKELY, pending one test** | the actor failure has the shape of a unimodal-policy-vs-bimodal-Q limit (Phase 5/6 flagged this); becomes the classification if the §13/§15 actor-side lever fails |
| F | **NOT YET** | do not switch algorithms — the brief and the evidence both say the diagnosis is not "SAC is wrong", it is "the Gaussian actor / gated action head is wrong" |

**CLASSIFICATION: CASE D**, with **CASE E** as the pre-registered fallback if the
single remaining actor-side lever (entropy temperature / policy parameterization)
does not close the 15–35 Nm mode-selection gap. **CASE F (algorithm swap) is not
reached and not authorized.**

---

## §13 — GENERALIZATION

> *"1 train NEDC → evaluate FTP75; 2 train FTP75 → evaluate NEDC; 3 (if the env
> is safely parameterized) varied initial SoC 40/50/60 %. Do NOT invent
> temperature / traffic / accessory load / road grade / new physics."*

Cross-cycle, CONTROL (gated k_fb=2.5) checkpoints, authoritative evaluator,
deterministic. Data: `results/phase7/data/generalization_crosscycle.json`.

| train → eval | V_CE (3-seed) | vs rule-based | charge-sustaining |
|---|---|---|---|
| NEDC → NEDC | 3.7666 ± 0.079 | +7.4 % | **3/3** |
| FTP75 → FTP75 | 3.2889 ± 0.017 | +1.8 % | **3/3** |
| **NEDC → FTP75** | **3.3818 ± 0.062** | +4.6 % | **0/3** |
| **FTP75 → NEDC** | **4.3107 ± 0.144** | +23.0 % | **0/3** |

**Cross-cycle transfer is poor, and specifically fails on charge balance
(0/3 CS both directions).** FTP75→NEDC is a hard failure (+23 %). This confirms
Phase 5's "the optimal costate is cycle-dependent" finding: the policy has fitted
cycle-specific SoC management (a fixed `k_fb` + a fixed SoC trajectory), not a
transferable costate law. This is itself weak corroboration that the residual
problem is not a mis-set global price but a policy that over-specializes.

**Initial-SoC sweep (item 3): NOT run.** `_Q_BT_IC` is hard-coded at 50 % in
`powertrain.py`, which is a **LOCKED** component. Per the brief, this test is
only permitted "if the environment has been safely parameterized without
modifying validated physics" — it is not. Making `_Q_BT_IC` an `EMSEnv`
constructor argument (threaded to `Battery.reset()`) is a small, clearly-scoped
RL-layer change that does **not** alter any plant equation; it is recommended as
a separate authorized task, not taken here.

---

## §14 — DOCUMENTATION

> *"Update/create results/phase7/, raw forensic output, figures, machine-readable
> summaries, EXPERIMENT_LOG.md, RL_DIAGNOSTIC_REPORT.md, ROADMAP.md,
> VALIDATION.md, experiments/experiment_registry.yaml. Record hypothesis,
> control, treatment, exact changed/unchanged variables, git commit before/after,
> seeds, training budget, numerical results, conclusion, rejected hypotheses,
> next decision. Do not overwrite previous phase results."*

Produced:

```
results/phase7/
  raw/phase7_forensics_NEDC.txt          raw/phase7_forensics_FTP75.txt
  data/00_baseline_lock.json             data/effective_price_{NEDC,FTP75}.json
  data/matched_states_{NEDC,FTP75}.csv   data/matched_states_{NEDC,FTP75}_summary.json
  data/counterfactual_{NEDC,FTP75}.json  data/kfb_compare_{NEDC,FTP75}.json
  data/required_kfb_{NEDC,FTP75}.json    data/ecms_gap_{NEDC,FTP75}.json
  data/generalization_crosscycle.json
  figures/phase7_summary_{NEDC,FTP75}.png  figures/counterfactual_{NEDC,FTP75}.png
results/phase7_forensics.py   results/phase7_figures.py   (analysis scripts, reusable)
PHASE7_FINAL_REPORT.md   (this file)
```

Appended (not overwritten): `EXPERIMENT_LOG.md` (E15), `RL_DIAGNOSTIC_REPORT.md`
(Phase 7 section), `ROADMAP.md`, `VALIDATION.md`,
`experiments/experiment_registry.yaml` (`phase7:` block).

* **Hypothesis:** the residual gap is primarily an economic (equivalent-factor /
  costate) valuation error.
* **Control:** gated `k_fb=2.5` Phase-5 candidate (unchanged; §1).
* **Treatment:** none — pure forensic analysis on existing checkpoints.
* **Changed variable:** none. **Unchanged:** everything (plant, benchmarks,
  evaluator, SAC, exploration, all hyperparameters).
* **git before:** `f1f45c5`. **git after:** *(this commit)*.
* **Seeds:** existing {0,1,2}. **Training budget:** 0 (no training).
* **Conclusion:** hypothesis **REJECTED**; bottleneck is actor-side (CASE D).
* **Rejected:** A (discharge over-priced), B (charging under-priced),
  D (feedback too weak), E (terminal propagation), and "`k_fb` is the lever".
* **Next decision:** §15 Q11 — one actor-side A/B, then CASE E if it fails.

---

## §15 — PHASE 7 FINAL FORENSIC REPORT (the twelve questions)

> *"It must answer, with measured evidence:"*

**1. Why does SAC still lose to the advanced rule-based benchmark on NEDC?**
The entire net deficit is the **15–30 Nm band (+0.368 L/100km vs rule-based,
unchanged across Phases 4–7)** plus a smaller 30–50 Nm term. In that band the
SAC critic prefers engine-OFF (ΔQ(OFF−ASSIST) = +0.006…+0.023, positive in
62–91 % of matched states) but the **deterministic Gaussian actor selects OFF in
0–47 % of states vs its own arg-max-Q's 72–87 %**. SAC parks in LPS/ASSIST where
the benchmark (and ECMS) use pure-electric. Secondary: SAC runs the engine
~10–40 % softer than optimal when it is on.

**2. Why does SAC remain above ECMS on both cycles?**
Total gap NEDC +0.498, FTP75 +0.460. Decomposition (§11): **≈ 60–65 %
mode-selection** (SAC OFF 4 % vs ECMS 49 % at NEDC 30–35 Nm), **≈ 10–25 % engine
operating-point** (SAC runs the engine softer/more often; ECMS harder/less
often), **≈ 25 % (NEDC 0–15 Nm) over-EV at trivial load** then paying to
recharge, **≈ 1 % battery-energy management** (already solved).

**3. Is battery energy being economically over-priced?**
**Against the static λ₀ = 1.3125: numerically yes** (median effective price
2.82 ECMS units NEDC / 2.72 FTP75, ≈ 2.1×, 98.8–100 % of the episode above λ₀).
**Against the correct reference (ECMS's own closed-loop pricing): NO** — ECMS's
own effective λ over its own charge-sustaining rollout has median **2.78 / 2.85**,
essentially identical. Both controllers price battery ≈ 2.8 ECMS units *because
both operate below SoC target* (median visited SoC 37.5 % / 47.4 %). The CONTROL
`k_fb=2.5` already matches ECMS's *median effective* λ (derived `k_fb` = 2.43 on
NEDC). **No over-pricing relative to the proven optimum.**

**4. Is k_fb the correct lever?**
**No.** (a) At the NEDC operating SoC (37 %) the actor's P(OFF) is flat at 48 %
across `k_fb` ∈ {1.656, 2.5, 3.0} while the eq-price moves 2.63 → 3.31 → 3.70
ECMS units. (b) The trained multi-seed sweep already on disk shows `k_fb ∈
[2.0, 3.0]` is a flat fuel plateau (3.766 → 3.784, both 3/3 CS), and `k_fb =
1.656` (ECMS-slope-matched) loses charge-sustaining on NEDC (1/3) for worse
fuel. (c) The only positive `k_fb` that would bring the *median* price to λ₀ is
`k_fb ≤ 0`, refuted in Phase 4–5. On FTP75 lower `k_fb` helps marginally, but
FTP75 is already at the benchmark.

**5. What is the dominant remaining source of the 15–30 Nm deficit?**
**Actor-side displacement.** ΔQ(OFF−ASSIST) is positive (critic prefers OFF),
arg-max-Q OFF share (72 % NEDC / 80 % FTP75) matches ECMS (71 % / 89 %), and the
actor delivers 47 % / 33 %. The immediate reward at low SoC slopes toward
charging (myopically correct), and the unimodal squashed-Gaussian policy sits on
that LPS lobe instead of the OFF lobe its critic prefers. Neither coverage
(Phase 6, refuted) nor economics (Phase 7, refuted) explains it.

**6. What is the dominant remaining source of the 30–75 Nm ECMS gap?**
Two parts: (i) **mode-selection at 30–50 Nm** — SAC OFF 4–51 % vs ECMS 49–64 %
(same actor-displacement mechanism as Q4/Q5: at NEDC 30–35 Nm argmax-Q OFF =
87 %, actor = 0 %); (ii) **engine operating-point at 35–75 Nm** — when the engine
*is* on, SAC runs it at |T_CE| ≈ 50–70 Nm vs ECMS's 68–95 Nm, i.e. lower-load,
worse-BSFC operation. Part (ii) is plausibly limited by the gated action map
compressing the engine-load sub-range (CASE B component).

**7. Is the critic wrong?**
**Mostly no.** Its arg-max mode matches ECMS at 15–35 Nm; `D_flat = 0 %`
(informative). There is a small negative `ERROR_critic` (−0.002…−0.009) on the
*exact* ECMS operating point, uncorrelated with the battery price
(corr ≈ −0.00…+0.18) — a minor high-engine-load undervaluation, not a gross
economic or bootstrapping bias (γ = 0.20, n_step = 1 bound the temporal reach).

**8. Is the actor wrong?**
**Yes — this is the primary finding.** The deterministic policy is displaced
~1.5 action-units from its own arg-max-Q at 30–35 Nm, on the opposite (LPS) lobe
of a bimodal Q. Across 15–35 Nm the actor under-delivers OFF by 25–87
percentage points relative to its own critic.

**9. Is the action representation wrong?**
**Contributing (CASE B/H).** The gated `modeaware` map is control-equivalent and
was a real improvement, but (a) it presents OFF as a contiguous `a ∈ [+0.2,+1.0]`
block that the Gaussian actor — pushed toward `a ≈ 0` by entropy and toward LPS
by the low-SoC reward gradient — will not occupy, and (b) it compresses the
ASSIST/engine-load sub-range, which correlates with SAC's soft-engine operating
point. It is not the *sole* cause (the actor fails to reach even the OFF share
its own critic wants), but a mode-aware **policy head** (not just a coordinate
remap) is the natural fix.

**10. Is the SAC algorithm itself wrong?**
**No — not reached.** The implementation is verified correct (Phase 2). The
failure is specific to the **unimodal continuous policy class** interacting with
a **bimodal Q** and a **myopic-at-low-SoC reward surface**. That is CASE D/E, not
CASE F. Do not switch to TD3/PPO/DDPG before the actor-side lever is tried.

**11. What single intervention has the highest expected probability of closing
the remaining gap?**
A **mode-aware policy parameterization** — a 2-component mixture policy, or a
discrete engine-OFF/ON head with a continuous within-mode action — so the policy
can place mass on the OFF lobe of Q at 15–35 Nm. This directly targets the
measured failure (unimodal actor on a bimodal Q). The disciplined path that
respects the brief's decision gate: **first run the pre-registered one-variable
actor-side A/B — target-entropy / entropy-temperature (and optionally an actor
LR bump), gamma 0.20 and everything else frozen, 3 seeds** — as the final CASE-D
check; **if it fails to move actor P(OFF) at 15–35 Nm, that result authorizes the
mixture/discrete-continuous policy (CASE E).**

**12. What evidence justifies that intervention?**
(a) §3 — at NEDC 30–35 Nm the critic's arg-max wants OFF 87 % of the time, the
actor delivers 0 %; the gap is actor-vs-own-critic, not reward or coverage.
(b) §4 — `Q(a)` is explicitly bimodal with the actor mean on the wrong lobe,
~1.5 action-units from `argmax-Q`.
(c) §5 — `ERROR_reward ≥ 0` and `corr(ERROR_critic, eq-price) ≈ 0`: the problem
is not economic; §8 confirms `k_fb` is not the lever.
(d) Phase 6 already refuted coverage; Phase 5 showed `k_fb` fixes SoC but
displaces the actor; Phase 5B/6 pre-registered "if the actor-side lever fails,
the unimodal Gaussian class is the limitation."
(e) §11 — ≈ 70–80 % of the ECMS gap is mode-selection that the critic already
ranks correctly, i.e. recoverable by a policy that tracks its own Q.

---

## Objective scorecard

> **PRIMARY:** beat the advanced rule-based benchmark on NEDC and FTP75 while
> remaining charge-sustaining and physically valid.
> **STRETCH:** move materially toward ECMS.

| | current best SAC | rule-based | ECMS | status |
|---|---|---|---|---|
| NEDC | 3.7666 (3/3 CS) | 3.5056 | 3.1887 | **+7.4 % — primary NOT met** |
| FTP75 | 3.2460 (gated k1.656) / 3.2889 (CONTROL) | 3.2323 | 2.8097 | **+0.4 % / +1.8 % — primary essentially met on FTP75** |

Phase 7 did **not** change these numbers (no training) and does **not** claim
progress on fuel. Its deliverable is the **diagnosis** that unblocks the next
authorized intervention: the residual NEDC gap is an **actor / policy-class**
problem at 15–35 Nm, **not** an economic-pricing problem, and **`k_fb` is not the
lever**.

---

## Rejected hypotheses (Phase 7)

| # | hypothesis | verdict | key evidence |
|---|---|---|---|
| P7-A | battery discharge is economically over-priced vs the reference | **REJECTED** | SAC effective price 2.82 ≈ ECMS effective price 2.78 (§2, §8) |
| P7-B | battery charging is under-priced | **REJECTED** | symmetric `eq_factor`; LPS over-use is a low-SoC reward-gradient effect, correct in direction |
| P7-C | `k_fb` is the corrective lever | **REJECTED** | actor P(OFF) flat vs `k_fb` at NEDC operating SoC; trained `k_fb ∈ [2.0,3.0]` a flat plateau; 1.656 breaks CS (§6, §8) |
| P7-D | the critic has a gross economic / bootstrapping bias | **REJECTED** | arg-max mode matches ECMS; small `ERROR_critic` uncorrelated with price; γ=0.2, n_step=1 (§5) |
| P7-E | terminal reward mis-propagation | **REJECTED** | γ = 0.20; terminal term ≈ 0.77 % of reward (prior measurement) |

## Retained / newly supported

* **Actor-side displacement at 15–35 Nm** (from Phase 6) — **confirmed and
  sharpened**: the critic *does* prefer OFF there; the actor does not follow it.
* **Engine operating-point quality** (from Phase 5B FTP75) — **confirmed on both
  cycles**: SAC runs the engine softer than ECMS in every torque band.
* **Cycle-dependent costate** (from Phase 5) — **confirmed**: cross-cycle
  transfer fails charge-sustaining 0/3 both directions.
* **Unimodal-policy-vs-bimodal-Q** (from Phase 5/6) — **now the leading
  candidate** for the terminal classification (CASE E), pending one actor-side
  A/B.
