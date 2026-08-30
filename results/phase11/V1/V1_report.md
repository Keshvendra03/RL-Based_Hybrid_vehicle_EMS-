# PHASE 11 — V1: COSTATE-CURRENCY VERIFICATION

**NO TRAINING. NO CONTROLLER CHANGE. NO CALIBRATION IMPLEMENTED.**
Scripts (analysis-only, under `results/phase11/`):
`v1_analytical.py`, `v1_action_ranking.py`, `v1_rollout_R.py`.
Data: `results/phase11/data/v1_analytical.json`,
`v1_action_ranking_{NEDC,FTP75}.json`, `v1_rollout_R.json`.
Env: SB3 2.8.0, torch 2.12 CPU, numpy 2.4.6, scipy 1.17.1, Python 3.13.2.
`ems_env.enable_fast_interpolation()` used at runtime only (provided API, proven
exact < 1e-12; powertrain clamps all lookups). No `src/` file modified.

---

## HEADLINE

**C1 / H3 is FALSIFIED in the direction Stage 0 claimed.**

Stage 0 asserted: *"after conversion to a common `P_EM` basis, the SAC effective
battery costate is **substantially below** ECMS's ⇒ battery usage is more
attractive to SAC."* The decisive matched-state test (V1-D) shows the
**opposite**: at matched states the SAC per-step reward's arg-min prefers
**more engine / less battery** than ECMS's Hamiltonian in **56–58 %** of probes
overall and **66–90 %** in the critical 15–35 Nm bands, with **`SAC-picks-more-
battery` = 0 % in every band on both cycles**.

**What Stage 0 got right:** the battery-energy ledger `E(Q) = ½·U_oc(Q)·Q`
does make `dE_ledger` ≈ `0.583·P_EM·dt` near operating SoC — the capacitor
factor is real. **What Stage 0 missed:** the reward's fuel term is trapezoidal,
`dm_fuel = 0.5·(P_CE + P_CE_prev)/H_u`, so the **marginal** fuel cost of the
current action is only `0.5·P_CE(u)/H_u` (the other half lands in step *t+1*
via `Tank.p_fuel_prev`, which is **not in the observation**). That factor of ½
on the fuel marginal weights battery **2× heavier** relative to fuel, which
**more than cancels** the 0.583 capacitor factor. Net, on a common `P_EM`
basis:

| | Stage-0 (incomplete) basis | trapezoidal marginal basis | decisive empirical test |
|---|---|---|---|
| `λ_SAC / λ_ECMS` at operating SoC (NEDC ~0.375) | **0.69** (SAC below) | **1.38** (SAC above) | **SAC arg-min prefers more engine 66–90 %** of critical-band states |

---

## V1-A/B — analytical (from frozen source)

`v1_analytical.py`, data `v1_analytical.json`.

### Battery model (`powertrain.py`)
```
U_oc(Q)      = (15.6/36000)·Q + 39                       [V]
E(Q)         = 0.5·U_oc(Q)·Q                             [J, "capacitor-analogy"]
dE/dQ        = (15.6/36000)·Q + 19.5  =  U_oc(Q) − 19.5
Battery step : Q_t = Q_{t−1} − I·Δt ,  I = P_EM/u_bt ,  Δt = 1 s
⇒ dE_ledger  = E(Q_{t−1}) − E(Q_t) ≈ (dE/dQ)·(P_EM/u_bt)·Δt   (2nd-order term ~1e-3 J, negligible)
⇒ C(SoC)     = dE_ledger / (P_EM·Δt) = (dE/dQ) / u_bt
```

| SoC | U_oc (V) | dE/dQ | **C = dE_ledger / (P_EM·Δt)** (at ~3 kW discharge) |
|---|---|---|---|
| 0.30 | 43.68 | 24.60 | **0.555** |
| 0.375 | 44.85 | 25.35 | **0.566** |
| 0.40 | 45.24 | 25.79 | **0.570** |
| 0.50 | 46.80 | 27.30 | **0.584** |
| 0.60 | 48.36 | 28.80 | **0.598** |

C is ≈ **0.55–0.61** across the operating range — the ledger charges **~58 %**
of the electrical work delivered at the terminals. **This part of Stage 0 is
CONFIRMED.**

### Reward-to-Hamiltonian reduction (common `P_EM` basis)

Reward per step, u-dependent part:
```
r(u) ∝ − [ 0.5·P_CE(u)/H_u · K_FUEL_L_PER_KG   +   eqf_eff · dE_ledger(u) · K_ELEC_L_PER_J ]
        with dE_ledger(u) = C(SoC)·P_EM(u)·Δt
     ∝ − [ P_CE(u)  +  eqf_eff · C(SoC) · (K_ELEC_L_PER_J / K_FUEL_L_PER_KG · H_u) / 0.5 · P_EM(u) ]
     = − [ P_CE(u)  +  λ_SAC(SoC) · P_EM(u) ]
     λ_SAC(SoC) = eqf_eff(SoC) · C(SoC) · 4.8309 · 2        (the ×2 = trapezoidal ½-marginal on fuel)
     eqf_eff(SoC) = eq_factor + k_fb·(0.5 − SoC)             (CONTROL: eq_factor 0.2717/0.4981, k_fb 2.5)
λ_ECMS(SoC)  = λ₀ + 8.0·(0.5 − SoC)                          (λ₀ 1.3125 NEDC / 2.4062 FTP75)
```

| NEDC SoC | eqf_eff | **λ_ECMS** | **λ_SAC (marginal)** | λ_SAC/λ_ECMS | λ_SAC (Stage-0 basis, no ×2) | s0/ECMS |
|---|---|---|---|---|---|---|
| 0.30 | 0.772 | 2.913 | **4.136** | **1.42** | 2.068 | 0.71 |
| 0.375 (operating) | 0.584 | 2.313 | **3.197** | **1.38** | 1.598 | 0.69 |
| 0.40 | 0.522 | 2.113 | **2.873** | **1.36** | 1.437 | 0.68 |
| 0.50 (target) | 0.272 | 1.313 | **1.534** | **1.17** | 0.767 | 0.58 |

FTP75 same pattern: λ_SAC/λ_ECMS = **1.30 (SoC 0.30) → 1.17 (SoC 0.50)**,
operating-SoC (0.474) ratio **1.21**.

**The analytical result is basis-dependent** (this is why V1-C is only a
diagnostic and V1-D is decisive):
* **marginal / per-step-decision basis** (½ on fuel, C on battery): `λ_SAC ≈ 1.2–1.4 × λ_ECMS` — SAC **over-prices** battery.
* **Stage-0 basis** (raw P_CE, C on battery): `λ_SAC ≈ 0.58–0.71 × λ_ECMS` — SAC under-prices (Stage 0's claim).
* **discounted-return basis** (effective fuel weight `0.5·(1+γ) = 0.60` at γ=0.20): between the two, roughly `λ_SAC ≈ 0.95–1.1 × λ_ECMS`.

---

## V1-D — DECISIVE matched-state action-ranking test

`v1_action_ranking.py`. States on the ECMS charge-sustaining SoC trajectory
(pass 1 = ECMS rollout records SoC per step; pass 2 = fresh env, SoC overwritten
to the ECMS value, 161-point action sweep, **actual implemented reward**
`r(a)` from `env.step`, ECMS `H(a) = p_ce(a) + λ_ECMS(SoC)·p_em(a)` on the
**same executed operating points**). `da = a_sac − a_ecms`; more `a` ⇒ more
`u` ⇒ **less** engine. NEDC 121 probes, FTP75 183 probes.

### NEDC

| band | n | `da` median | ΔT_CE median (SAC−ECMS) | SAC picks more engine | SAC picks more battery | same arg-min | SAC OFF% | ECMS OFF% | mean T_CE\|SAC / ECMS |
|---|---|---|---|---|---|---|---|---|---|
| 0–15 | 32 | 0.000 | 0.0 | 0 % | 0 % | 100 % | 100 | 100 | 5.0 / 5.0 |
| **15–30** | 38 | **−1.200** | **+25.2 Nm** | **66 %** | **0 %** | 34 % | 18 | 74 | 30.7 / 13.6 |
| **30–35** | 20 | **−1.200** | **+52.5 Nm** | **90 %** | **0 %** | 10 % | 0 | 90 | 58.5 / 10.9 |
| 35–50 | 6 | −0.788 | +33.6 | 50 % | 0 % | 50 % | 17 | 67 | 59.3 / 25.4 |
| 50–75 | 13 | −0.225 | +16.4 | 85 % | 0 % | 15 % | 0 | 31 | 106.1 / 69.2 |
| >75 | 12 | −0.750 | +39.0 | 92 % | 0 % | 8 % | 0 | 0 | 145.6 / 108.7 |
| **overall** | 121 | **−0.225** | — | **56 %** | **0 %** | **44 %** | — | — | — |

### FTP75

| band | n | `da` median | ΔT_CE median | SAC more engine | SAC more battery | same | SAC OFF% | ECMS OFF% | T_CE SAC/ECMS |
|---|---|---|---|---|---|---|---|---|---|
| 0–15 | 39 | 0.000 | 0.0 | 0 % | 0 % | 100 % | 59 | 59 | 4.9 / 4.9 |
| **15–30** | 57 | **−1.225** | **+29.9** | **65 %** | **0 %** | 35 % | 23 | 88 | 32.4 / 8.0 |
| **30–35** | 11 | **−1.200** | **+52.3** | **55 %** | **0 %** | 45 % | 18 | 73 | 50.2 / 19.6 |
| **35–50** | 24 | **−1.256** | **+63.4** | **83 %** | **0 %** | 17 % | 0 | 62 | 78.3 / 28.9 |
| 50–75 | 37 | −0.375 | +22.1 | 76 % | 0 % | 24 % | 0 | 11 | 107.8 / 78.5 |
| >75 | 15 | −0.663 | +38.9 | 100 % | 0 % | 0 % | 0 | 0 | 144.3 / 104.7 |
| **overall** | 183 | **−0.375** | — | **58 %** | **0 %** | **42 %** | — | — | — |

**Reading.**
* In the two bands that carry the NEDC gap (15–35 Nm) the SAC reward's arg-min
  is **~25–52 Nm harder on the engine** than ECMS's, and it is **engine-ON in
  90–100 %** of 30–35 Nm states where **ECMS's arg-min is OFF in 90 %**.
* `SAC-picks-more-battery` is **0 % in every band on both cycles** — the SAC
  per-step reward **never** prefers a more battery-heavy action than ECMS at
  matched states.
* The ~44 % "same arg-min" probes are the 0–15 Nm (both pick min-engine) and
  the forced high-demand splits.

**⇒ On a common physical (`P_EM`) basis the SAC per-step reward is a
*stiffer-battery* Hamiltonian than ECMS. C1's stated direction is FALSIFIED.**

---

## V1-C — rollout-level realised exchange rate (DIAGNOSTIC ONLY)

`v1_rollout_R.py`. `R = Σ d_fuel_L / Σ (P_EM·Δt)`, CONTROL SAC (3 seeds, mean)
vs ECMS, real env rollout. **Task §5 caveat honoured:** this is confounded by
*which states each controller visits* (SAC discharges during ASSIST with the
engine also on; ECMS discharges during pure-EV with the engine off), so it is
**not** a clean costate measurement — reported as corroboration of direction
only.

| bucket | metric | ECMS | SAC (3-seed) | SAC / ECMS |
|---|---|---|---|---|
| **discharge side** | `Σ d_fuel_L / Σ |P_EM·Δt|` NEDC | 2.65e-8 | **1.28e-7** | **4.85×** |
| | same, FTP75 | 2.13e-8 | **5.14e-8** | **2.42×** |
| engine-on | `R/|batt|` NEDC | 3.53e-7 | 6.35e-7 | 1.80× |
| charge side | `R/|batt|` NEDC | 1.85e-7 | 2.21e-7 | 1.19× |
| regen | `R/|batt|` NEDC | 3.32e-9 | 3.69e-9 | 1.11× |

The CONTROL realises **2.4–4.9× more fuel per joule of battery discharged** than
ECMS. Consistent with V1-D: the SAC controller behaves as if battery discharge
is much more "expensive" (in realised fuel terms) than ECMS treats it — the
opposite of C1. (Part of the 4.85× is behavioural — SAC's discharge co-occurs
with engine-on ASSIST — not pure pricing; V1-D, which holds the state fixed, is
the clean signal.)

---

## V1-E — prediction vs observed behaviour

Stage 0's prediction: *"a lower SAC battery price should make battery usage
more attractive relative to ECMS."*

* **The measured direction is the reverse** (V1-D, V1-C): the SAC reward's
  per-step arg-min uses **less** battery / **more** engine than ECMS.
* **This direction IS consistent with the observed CONTROL behaviour**
  (Phase 9: 376 vs 260 engine-on steps on NEDC; T_CE\|on 55 vs 79 Nm;
  part-load, worse BSFC). So the reward's instantaneous preference and the
  trained policy's behaviour **agree in direction** — SAC runs the engine more
  partly because its per-step reward, on a common physical basis, tells it to.
* **BUT this does NOT make the reward "the cause of the gap":**
  1. The reward's arg-min engine torque at 30–35 Nm is **~58 Nm** (engine
     ON, moderately loaded); the *trained actor* delivers **~35 Nm** (Phase
     7/9) — the actor **undershoots its own reward's arg-min by ~23 Nm**. That
     residual is the optimisation/exploration-topology failure documented in
     Phases 5–9 (bimodal `r`/`Q`, σ-collapse, thin coverage of hard-engine
     operating points), unchanged by V1.
  2. The reward's arg-min is **engine-ON-moderate** where ECMS is **OFF**.
     Following the reward perfectly would **not** reproduce the ECMS OFF
     strategy — it would run the engine at part load. So the reward, on a
     common basis, is a *stiffer-battery* guide than ECMS and mildly biases the
     agent **against** the ECMS "OFF or hard" strategy.

**"C1 is mathematically nuanced" ≠ "C1 is the dominant cause."** The dominant
cause remains the arg-min↔actor gap (optimisation/exploration), with the
reward's stiffer-than-ECMS instantaneous costate a **secondary, same-direction
contributor** to the excess engine-on time.

---

## V1 QUESTIONS — ANSWERS

**V1-Q1 — Is the Stage-0 mathematical derivation correct?**
**Partially.** The capacitor factor `C(SoC) = dE_ledger/(P_EM·Δt) ≈ 0.58` is
correct and confirmed. The derivation **omitted** the trapezoidal ½-marginal
weighting of the fuel term (`dm_fuel = 0.5·(P_CE + P_CE_prev)/H_u`), which
weights battery 2× relative to marginal fuel and **reverses the sign** of the
conclusion.

**V1-Q2 — Is a costate mismatch present on a common `P_EM` basis?**
**Yes, but with the opposite sign to Stage 0.** On the marginal (per-step-
decision) basis `λ_SAC ≈ 1.2–1.4 × λ_ECMS` in the operating SoC band; the
decisive matched-state test confirms the SAC reward's arg-min systematically
prefers less battery / more engine.

**V1-Q3 — How large is the mismatch?**
Analytical: `λ_SAC/λ_ECMS` = **1.17 at SoC 0.50 → 1.38–1.42 at SoC 0.30**
(NEDC); **1.17 → 1.30** (FTP75). Behavioural: SAC arg-min engine torque is
**+25 Nm (15–30 Nm band) / +52 Nm (30–35 Nm band)** above ECMS's; SAC arg-min
is engine-ON in 90–100 % of 30–35 Nm states vs ECMS 10 %.

**V1-Q4 — Intercept error, slope error, or both?**
**Predominantly intercept (base costate), amplified at low SoC.** After the
capacitor + trapezoidal corrections, the *slope* term `eqf_eff`'s `k_fb`
contribution scales the same way for both; the ratio `λ_SAC/λ_ECMS` is largest
(1.4) at low SoC and smallest (1.17) at target — i.e. the divergence grows as
`(0.5 − SoC)` grows, which is a **combined intercept + slope** effect but the
intercept ratio (1.17 at target) is the irreducible part.

**V1-Q5 — Does it produce different instantaneous action preferences?**
**Yes, systematically.** V1-D: 56–58 % of matched states overall (66–90 % in
15–35 Nm) have the SAC-reward arg-min at a **harder engine / less battery**
point than the ECMS arg-min; **0 %** anywhere have SAC preferring more battery.

**V1-Q6 — Does it explain the observed SAC behaviour in the 15–35 Nm region?**
**Partially, as a same-direction contributor — not as the primary cause.** The
reward's stiffer instantaneous costate biases toward engine-ON, matching the
CONTROL's excess engine-on time. But (a) the trained actor runs the engine
*softer* (~35 Nm) than the reward's own arg-min (~58 Nm) — an optimisation gap
V1 does not touch; and (b) the reward's arg-min is still engine-ON-moderate,
not the ECMS OFF, so a reward-optimal policy would not reproduce ECMS. The
15–35 Nm gap remains dominated by the arg-min↔actor optimisation/exploration
failure (Phases 5–9).

**V1-Q7 — C1 / H3 verdict:**
> **FALSIFIED (as stated).** Refined finding retained: the SAC reward, on a
> common `P_EM` basis, is a *stiffer-battery* Hamiltonian than ECMS
> (`λ_SAC/λ_ECMS ≈ 1.17–1.4` in the operating band; matched-state arg-min
> prefers more engine in 56–90 % of states). This is a **secondary,
> same-direction contributor** to the CONTROL's excess engine-on time, not the
> dominant cause of the ECMS gap.

**V1-Q8 — What can legitimately be concluded about changing `eq_factor` or
`k_fb`?** *(No calibration change is implemented or recommended for execution.)*
1. The analytical basis is genuinely **ambiguous** (marginal → SAC over-prices;
   Stage-0 basis → under-prices; return basis → near-parity). Only the
   empirical matched-state test is decisive, and it says **SAC over-prices
   battery per-step relative to ECMS**.
2. Therefore, **Stage 0's specific suggestion — raise `eq_factor` to 0.466 — is
   the wrong direction.** If a costate-calibration experiment is ever approved,
   it should test **lowering** the effective per-step costate (lower
   `eq_factor` and/or lower `k_fb`) to bring the reward's per-step arg-min
   toward ECMS's — while checking SoC stability (the reason `k_fb` was raised
   to 2.5 in Phases 4–5).
3. **However, three facts cap the expected value of any such experiment:**
   (a) Phase 7 measured trained-policy `P(OFF)` **flat** across `k_fb ∈
   {1.656, 2.5, 3.0}` — the trained policy does not respond to the costate the
   way the per-step math predicts, because it self-selects a low operating SoC
   (~37.5 %) where the total effective price re-converges toward ECMS's;
   (b) the arg-min↔actor gap (~23 Nm) is larger than the reward↔ECMS arg-min
   gap and is not a calibration issue;
   (c) lowering the costate risks re-opening the SoC-runaway that `k_fb=2.5`
   fixed.
4. **Net:** costate recalibration is a **low-priority, secondary** lever. It
   should not be the next experiment. It could be folded, as a *documented
   sub-variable with its own justification*, into a later reward-shaping study —
   but only after the primary optimisation/exploration lever is tested.

---

## FILES

```
results/phase11/v1_analytical.py          results/phase11/data/v1_analytical.json
results/phase11/v1_action_ranking.py      results/phase11/data/v1_action_ranking_{NEDC,FTP75}.json
results/phase11/v1_rollout_R.py           results/phase11/data/v1_rollout_R.json
```
No `src/` file changed. No RL training. No calibration applied.
