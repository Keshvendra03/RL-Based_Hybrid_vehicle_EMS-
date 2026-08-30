# PHASE 11 — STAGE 0 REPORT

**Forensic + Bellman + ECMS-fairness + physical-energy-balance audit.
NO TRAINING. NO CODE CHANGES.**

Git commit at audit: `f1f45c5` (working tree has uncommitted Phase 7–10 docs).
Scope: `src/env/ems_env.py`, `src/env/powertrain.py`, `src/env/driving_cycle.py`,
`src/baselines/ecms.py`, `src/baselines/advanced_rule_based.py`,
`results/evaluate_policy.py`, `src/agents/train_sac.py`, `data/params.json`,
`data/maps/engine_maps_data.py`, plus `results/phase{7,8,9}/data/*.json` and
Phases 2–10 reports. Every numeric claim is tagged **[F]** measured fact from
code/data, **[D]** derived analytically here, **[I]** interpretation,
**[H]** hypothesis.

---

## 1. VERIFIED FACTS

### 1.1 CONTROL configuration (what "current SAC" actually is)

The validated CONTROL is **not** reproduced by `train_sac.py` defaults — it is
a specific CLI flag set. **[F]**

| Item | CONTROL value | `train_sac.py` default | Source |
|---|---|---|---|
| γ | **0.20** | 0.9999 | registry `PHASE5-kfb`, Phase 7 §1 |
| n_step | **1** | 5 | registry |
| k_fb | **2.5** (env liter-units) | 0.0 | registry |
| eq_factor | **0.2717** (NEDC) / **0.4981** (FTP75) | 1.0 | registry; = λ_ECMS / 4.8309 |
| action_map | **`modeaware_gated`** | `linear` | registry |
| target_entropy | **auto** (= −dim A = −1.0) | auto | `train_sac.py:316` |
| lr | 3e-4 | 3e-4 | `train_sac.py:473` |
| batch | 512 | 512 | `:478` |
| buffer | 300 000 | 300 000 | `:474` (SB3 default is 1e6) |
| τ (Polyak) | 0.005 | 0.005 | `:477` |
| train_freq | 64 | 64 | `:479` |
| gradient_steps | **16** | 64 | registry; `:480` |
| net_arch | [256, 256] | [256, 256] | `:485` |
| lookahead | 5 causal speeds | 5 | `:362` |
| training budget | **150 000 steps** (≈123 NEDC / ≈100 FTP75 episodes) | 1.5e6 | registry |
| SoC hard mask | [0.05, 0.95] | — | `ems_env.py:194` |
| CS evaluation | \|SoC_final − 0.5\| ≤ 0.02 | — | `evaluate_policy.py:68` |
| reset | fixed SoC 0.5 (`_Q_BT_IC` hard-coded) | — | `powertrain.py:955`, LOCKED |
| action space | `Box(-1, 1, (1,))` | — | `ems_env.py:389` |

**Config inconsistency [F]:** the registry contains **two** "validated"
configs — `FINAL-VALIDATED` (Phase 2: **linear** map, k_fb **1.656**,
target_entropy **−2**, NEDC 3.7727 ± 0.028) and the Phase-5+ **CONTROL**
(gated, k_fb 2.5, entropy auto, NEDC 3.7666 ± 0.079). Phases 7–9 use the
gated k_fb=2.5 as CONTROL. The NEDC difference is a statistical tie
(−0.0061, 0.08 σ). Phase 11 must pick **one** and freeze it. Recommend the
Phase-7+ CONTROL (gated, k_fb 2.5) for continuity with the forensic chain.

### 1.2 Reward — exact implementation (`ems_env.py:598-711`) **[F]**

Per non-terminal step:
```
r_t = -100 · ( fuel_L,t  +  eqf_eff,t · elec_L,t )
      -  2.0 · excess_t²                              (LAMBDA_SOC)
      -  1.0 · excess_t                               (LAMBDA_SOC_LIN)
      -  1.0 · [safety flags: uv / oc / motor OL / engine OL]   (masking ⇒ ~never)

fuel_L,t   = dm_fuel,t · K_FUEL_L_PER_KG ,   K_FUEL_L_PER_KG = _K_CS/_RHO_FUEL = 1.15/0.843 = 1.36418
dm_fuel,t  = (P_CE,t + P_CE,t-1) · 0.5 · Δt / H_u        (trapezoidal, kg)
elec_L,t   = ΔE_batt,t · K_ELEC_L_PER_J ,    K_ELEC_L_PER_J  = _EFC_GAIN/3.6e6 = 1.534866e-7
ΔE_batt,t  = E(Q_{t-1}) − E(Q_t)                        (J; + = discharge)
E(Q)       = 0.5 · U_oc(Q) · Q ,  U_oc(Q) = (15.6/36000)·Q + 39      ("capacitor-analogy" energy)
eqf_eff,t  = eq_factor + k_fb · (0.5 − SoC_t^{pre-decision})
excess_t   = max( |SoC_t − 0.5| − 0.10 , 0 )            (SOC_DEADBAND = 0.10)
```
Terminal step additionally:
```
E_net = E(Q_IC) − E(Q_T)
if E_net < 0:  r_T -= 100 · (−E_net) · K_ELEC_L_PER_J          (saturation correction)
t_excess = max( |SoC_T − 0.5| − 0.02 , 0 )                     (TERM_TOL = 0.02)
r_T -= 50 · t_excess + 800 · t_excess²                         (TERM_W_LIN, TERM_W_QUAD)
```

Measured component weights **[F]** (Phase 2 §5 / RL_DIAGNOSTIC §5, CONTROL
rollout): fuel term 66 % (NEDC) / 44 % (FTP75); battery term 34 % / 56 %;
**SoC deadband penalty 0.0 %** (0/1220, 0/1876 steps active — SoC stays inside
±10 %); **terminal penalty 0.77 %** (NEDC) / **0.00 %** (FTP75) of episode
reward.

### 1.3 Constants (`data/params.json`, `engine_maps_data.py`) **[F]**

* Battery: 10 Ah, 48 V nom, `Q_BT_0 = 36000 As`, `Q_BT_IC = 18000 As` (50 %),
  0.48 kWh; `I_BT_MAX = 300 A`; U_oc linear 39 → ~53.8 V over 5–95 % SoC.
* Engine map: η peaks at **≈ 0.343** ⇒ BSFC_min ≈ **245 g/kWh**. Map speed grid
  **125.7–439.8 rad/s (1200–4200 rpm)**. **Idle = 8000 W flat** whenever
  `0 < w_gear ≤ 105 rad/s` (≈ v < 10 km/h in gear 1), torque-independent.
  Fuel cutoff at `T_CE ≤ 5 Nm`. **No start cost, no minimum-on time, no
  warm-up.** `_K_CS` cold-start = **1.15** applied to all fuel.
* Motor: `T_max` 60 Nm (≤ ~200 rad/s), falling to ~7.7 Nm at 800 rad/s;
  12 kW. Signed-torque efficiency map (η for regen, 1/η for motoring).
* **Engine speed is not a control DOF:** `w_CE = w_MGB = w_wheel · i_gt`,
  fixed by prescribed speed × gear. The only instantaneous freedom is the
  split factor `u` (`T_EM = u·T_MGB`, `T_CE = (1−u)·T_MGB`),
  `u ∈ [U_MIN = −0.85, U_MAX = 1.0]`.

### 1.4 SAC implementation **[F]**
Verified line-by-line vs SB3 2.9.0 in Phase 3; `NStepSAC` formula-identical.
Not re-audited here; taken as correct.

---

## 2. MATHEMATICAL EQUATIONS

### 2.1 Implemented SAC objective

SB3 SAC (n_step = 1, auto-α to hit target entropy `H̄ = −1`):
```
J(π) = E_π [ Σ_{t≥0} γ^t ( r_t + α · H(π(·|s_t)) ) ] ,      γ = 0.20
critic target:  y_t = r_t + γ (1 − d_t) ( min_{i} Q̄_i(s_{t+1}, a') − α log π(a'|s_{t+1}) ) ,  a' ~ π(·|s_{t+1})
actor:          max_π E_{a~π} [ min_i Q_i(s,a) − α log π(a|s) ]
```
Effective value horizon **[D]**: `1/(1−γ) = 1.25` steps. Discount weights:
γ¹ = 0.20, γ² = 0.04, γ³ = 8e-3, γ⁵ = 3.2e-4, γ¹⁰ = 1.0e-7. **Only the current
step and ~1–2 following steps carry any weight.** The bootstrapped term is 20 %
of the target.

### 2.2 ECMS objective (`ecms.py`, `evaluate_policy.py:170-195`)

Per step, grid-search `u` over 81 feasible points minimising the Hamiltonian:
```
H(u) = P_fuel(u)  +  λ_eff · P_batt(u) ,     λ_eff = λ₀ + k_fb^{ECMS} · (0.5 − SoC)
P_fuel(u) = combustion_engine( (1−u)·T ).p_ce        [W]
P_batt(u) = electric_motor( u·T ).p_em               [W, + = discharge]
k_fb^{ECMS} = 8.0 ;   λ₀ = 1.3125 (NEDC) / 2.4062 (FTP75)   [bisection-tuned, §5]
Braking (T < 0):  u is NOT optimised — forced to max feasible regen.
```

### 2.3 Costate-currency derivation — the SAC reward is NOT ECMS's Hamiltonian **[D]**

Reduce the SAC per-step reward (drop the inactive deadband penalty and the
γ-dead terminal term — §1.2) to a Hamiltonian in the **same variables ECMS
uses** (`P_fuel`, `P_batt`):
```
r_t ≈ -100 · [ P_fuel · Δt · (K_FUEL_L_PER_KG / H_u)  +  eqf_eff · ΔE_batt · K_ELEC_L_PER_J ]
    = -100 · Δt · (K_FUEL_L_PER_KG / H_u) · [ P_fuel  +  eqf_eff · (ΔE_batt/Δt) · 4.8309 ]
        with  4.8309 = K_ELEC_L_PER_J / (K_FUEL_L_PER_KG / H_u)          [F, RL_DIAGNOSTIC §2]
```
The bracket is a Hamiltonian **only if `ΔE_batt/Δt` is proportional to
`P_batt`**. It is not. From the battery integrator `Q_t = Q_{t-1} − I·Δt`,
`I = P_batt/U_bt`:
```
ΔE_batt = E(Q_{t-1}) − E(Q_t) ≈ (dE/dQ) · (P_batt/U_bt) · Δt
dE/dQ   = (15.6/36000)·Q + 19.5  =  U_oc(Q) − 19.5
(dE/dQ)/U_oc  =  1 − 19.5/U_oc(Q)   ≈  0.583  at SoC 0.50 (Q = 18000)
                                   ≈  0.565  at SoC 0.375 (operating median)
                                   ∈ [0.51, 0.61] over the full 5–95 % SoC range
```
So `ΔE_batt/Δt ≈ 0.583 · P_batt` (near operating SoC), and the SAC reward's
**effective per-step costate on battery electrical power** is
```
λ_SAC(SoC) ≈ eqf_eff · 4.8309 · 0.583
          = [ eq_factor + k_fb·(0.5 − SoC) ] · 2.816
          = 0.765  +  7.04 · (0.5 − SoC)                    (CONTROL: eq_factor 0.2717, k_fb 2.5)

λ_ECMS(SoC) = 1.3125  +  8.00 · (0.5 − SoC)
```

**Result [D]:** `λ_ECMS(SoC) − λ_SAC(SoC) = 0.5475 + 0.96·(0.5 − SoC) > 0` for
every `SoC ∈ [0, 1]`. **The SAC reward prices battery discharge below ECMS at
every SoC** — by ~42 % at target SoC (0.77 vs 1.31), by ~29 % at the operating
median (1.65 vs 2.31). The **slope** is nearly matched (7.04 vs 8.0) once the
capacitor factor is applied; the **base costate** is not.

Equivalently: the `eq_factor = λ_ECMS/4.8309` calibration (Phase 2) equates the
reward's price with ECMS's costate **per joule of `E(Q) = ½U_oc Q`**, but
ECMS's Hamiltonian charges **per watt of electrical power `P_EM`**. The two
currencies differ by `dE/dQ ÷ U_oc ≈ 0.583`. To match ECMS's costate on a
common `P_EM` basis the CONTROL would need `eq_factor ≈ 0.2717/0.583 ≈ 0.466`.

**Status:** this is a **[D] derivation from the source**, consistent with —
but not previously identified by — any phase. It **requires one numerical
cross-check** before it is load-bearing (§8, task V1): compute the realised
`Σ Δfuel_L / Σ (P_EM·Δt)` for the CONTROL and for ECMS over their rollouts, and
the per-step arg-min of each controller's own instantaneous objective on a
common `P_EM` basis.

---

## 3. CONTRADICTIONS FOUND

### C1 — Reward costate calibration is on the wrong currency (see §2.3) **[D, needs V1]**

Canonical project claim (RL_DIAGNOSTIC §2, registry `reward_unit_conversion`,
Phases 2/5/7): *"implicit λ at eq_factor = 1.0 is 4.8309 fuel-J per battery-J;
`eq_factor = λ_ECMS/4.8309` makes the reward's costate equal ECMS's λ₀."*

The equality holds for battery **`E(Q)`-joules**; ECMS's Hamiltonian uses
battery **electrical power `P_EM`**. Because `E(Q) = ½U_oc(Q)Q`,
`dE/dQ ≈ 0.583·U_oc`, so on a common `P_EM` basis the CONTROL's effective
costate is **base 0.77 vs ECMS 1.31 (−42 %)**, slope **7.04 vs 8.0 (≈ match)**.

**Implications if V1 confirms:**
* Phase 7's headline *"SAC effective price (median 2.82 ECMS units) ≈ ECMS's
  own effective price (2.78) ⇒ no over-pricing"* is likely a **nominal-vs-true
  comparison**: SAC's 2.82 is `eqf_eff·4.8309` (the `E(Q)` basis); ECMS's 2.78
  is its true `P_EM` costate. On a common basis SAC's is **lower**, not equal.
  The Phase-7 conclusion "the gap is not an economic/costate error" would need
  re-examination.
* Phases 4–5 needed a **steeper-than-ECMS `k_fb`** (2.5 → 12.1 nominal units)
  to keep NEDC charge-sustaining. Under C1 this is explained: a base price
  ~40 % low makes battery too cheap → SoC drifts down → the strong `k_fb`
  proportional term is a *crutch* compensating a mis-set intercept, and the
  system self-selects a low operating SoC (median 37.5 %) where the total
  effective price rises back toward ECMS's.
* The Phase-2 "reward unit mismatch" fix (eq_factor 1.0 → 0.2717) may have
  **over-corrected**: on a common `P_EM` basis, `eq_factor = 1.0` is ~2.15×
  ECMS's λ₀ (not the quoted 3.68–4.83×), and the matched value is ~0.47, not
  0.27.

### C2 — `ems_env.py` docstring vs code: braking regen **[F]**

Module docstring (lines ~44–49): *"When T_MGB < 0 (braking) the action is
overridden by maximum feasible recuperation — identical policy to BOTH
rule-based benchmarks."* The actual `_action_to_torques` (lines ~536–572, with
its own comment) does the **opposite**: *"no forced regen — agent owns u on
every step"*; the forced-regen override was **removed** as a "0.31 L/100km
interface artefact." Meanwhile `ecms.py._hamiltonian_best_u` **still forces**
`_max_regen_u` on `T < 0`, and `evaluate_policy.py` routes ECMS through that.

**Net:** on braking steps the **RL policy chooses `u`** (regen amount, bounded
only by the motor envelope + SoC), while **ECMS is hard-forced to max regen**.
`VERIFIED_FACTS §G` ("env hard-codes max regen for every controller equally")
is **stale** for the training env. Empirically small — Phase 7 brake-band
ΔTotal = +0.010, REGEN % matches (~17 % NEDC) — but it is a real asymmetry and
a misleading docstring. **Recommend:** fix the docstring; keep the code
(agent-owns-`u` is the fairer design); note the asymmetry in any ECMS
comparison.

### C3 — "Don't train longer" conclusion is stale **[F]**

Phases 1–2 established "best checkpoint always in the first half; train longer
degrades" — measured at **γ = 0.9999, gradient_steps = 64**, i.e. the
**diverging-critic** baseline (critic_loss climbed to 12–53). That config was
abandoned; `gradient_steps = 16` fixed the divergence (VERIFIED_FACTS §E), and
Phase 8C explicitly reports the **current** config "trains stable and
progressive, no early-peak-collapse." **The anti-long-training result does not
transfer, and long training has never been run on the current stable config.**

### C4 — Gap-attribution methods disagree 6× **[F]**

`phase7/data/ecms_gap_NEDC.json` (`gap_split`): mode-selection **+0.608**,
engine-operating-point **+0.032**, battery +0.001, soc-equiv-residual −0.143.
`phase9/data/engine_physics_NEDC.json` (BSFC-grounded): operating-point
**+0.192 (39 %)**, mode/timing **+0.306 (61 %)**, battery ≈0. Both sum to
≈ +0.50 but the operating-point *share* differs 6× (6 % vs 39 %). The report
has been quoting whichever is convenient. Honest bracket: **operating-point
20–40 %, mode/timing 55–75 %, battery ≈ 0** on NEDC.

### C5 — Phase-9 FTP75 instrumentation artefact **[F]**

`phase9/data/engine_physics_FTP75.json`: `mean_bsfc_ecms = 457`,
`mean_eff_ecms = 2.4e11` — divide-by-near-zero when ECMS engine torque is
tiny. **The cycle-mean ECMS BSFC/efficiency figures for FTP75 in the Phase-9
report are artefacts.** Per-band numbers are sane.

### C6 — "Reward is sufficient, do not modify" rests on an instantaneous argmax **[F/I]**

Phase 8 §16 concluded the reward is fine because `argmax_a r(a)` engine torque
> ECMS torque in every band. That tests only the **instantaneous** optimum. It
does **not** establish that `argmax_π E[Σγ^t r_t]` (the actual training
objective, with the γ-dead terminal and the C1 costate) has the ECMS
trajectory as a fixed point — §2.3 shows it does **not** (λ_SAC < λ_ECMS
everywhere).

---

## 4. CURRENT SAC OBJECTIVE — ANSWERS

**A. What physical behaviour is the implemented SAC objective actually
rewarding?** **[D]**
Per-step minimisation of an equivalent-fuel Hamiltonian
`P_fuel(u) + λ_SAC(SoC)·P_batt(u)` with `λ_SAC(SoC) ≈ 0.765 + 7.04·(0.5−SoC)`
(on a `P_EM` basis), plus a weak SoC-band restoring force that is **inactive**
inside ±10 % of 50 % SoC (i.e. always, in practice), plus a terminal
charge-sustaining penalty **discounted to ≈ 0** by `γ^{steps-to-go}` for all
but the last ~3 steps. In words: **"at each step, split torque to minimise
instant fuel + a (somewhat under-priced) instant battery cost; do not worry
about the end-of-cycle SoC."** It is an ECMS-style myopic Hamiltonian
controller with a proportional costate feedback whose *intercept is ~40 % below
ECMS* and whose *slope ≈ ECMS*.

**B. Effective temporal horizon at γ = 0.20?** **[D]**
`1/(1−γ) = 1.25` steps ≈ 1.25 s. Steps ≥ 3 in the future contribute < 1 % of
the discounted return. Anticipatory "bank charge now to run electric over the
upcoming low-load stretch in 10–20 s" is **not representable** in the value
function — though the agent *does* receive 5 s of causal speed preview in the
observation, so it can *perceive* an upcoming event even if it cannot *value*
acting on it.

**C. Is the objective time-consistent with charge-sustaining operation?** **[D]**
**No, not as an internal constraint.** Charge-sustaining is enforced only by
(i) the per-step `k_fb` proportional term and (ii) a terminal penalty that is
γ-discounted to ≈ 0. The deadband quadratic/linear penalty never activates
(§1.2). So the SAC-optimal trajectory is whatever **fixed-point SoC** the
`k_fb` regulator + the (under-priced) battery cost settle at — empirically
SoC ≈ 37.5 % median on NEDC — **not** the SoC ≈ 50 % charge-sustaining optimum
ECMS solves for explicitly via λ₀ bisection. The 2 %-tolerance CS check at
evaluation is satisfied only because the terminal penalty (visible for the last
~3 steps) plus `k_fb` pull the *endpoint* back, not because the trajectory
optimises a CS-constrained objective.

**D. Can the ECMS trajectory theoretically be an optimum of the SAC
objective?** **[D]**
**No.** `λ_SAC(SoC) < λ_ECMS(SoC)` for all `SoC ∈ [0,1]` (§2.3). A per-step
minimiser of the SAC reward would use **more** electric assist / **less**
engine than ECMS at every SoC, and would charge-sustain at a **lower** SoC.
The ECMS `(u_t)` sequence is not a stationary point of the SAC per-step
objective.

**E. If not, identify the mathematical mismatch.** **[D]**
Three, in decreasing certainty:
1. **Base costate (intercept):** reward's effective `λ` on `P_EM` ≈ 0.77 at
   target SoC vs ECMS 1.31 (C1: `E(Q) = ½U_oc Q` currency vs `P_EM` currency;
   factor ≈ 0.583). *Needs V1 to quantify precisely.*
2. **Charge-sustaining constraint absent from the objective** at γ = 0.20
   (terminal term ≈ 0.77 % of return; deadband penalty 0 %). ECMS solves the
   CS-constrained problem directly.
3. **`k_fb` slope:** nominal 12.1 vs 8.0 ECMS units; ≈ 7.0 vs 8.0 after the
   C1 correction — a minor residual mismatch, not the dominant term.

**F. If the ECMS trajectory could (approximately) be recovered, why does trained
SAC fail to discover it?** **[F/I]**
Because the mismatch in E points the **wrong way** for the observed failure:
a cheaper-battery objective predicts SAC uses **more** electric than ECMS, but
SAC uses **less** in the decisive 15–35 Nm band (engine-on 376 vs 260 steps,
T_CE|on 55 vs 79 Nm). So SAC is **not** at its own objective's optimum there.
The measured obstacles (all pre-existing **[F]**):
* `r(a)` and `Q(a)` are **bimodal** at 15–35 Nm (LPS/charge lobe — valley —
  engine-OFF lobe), Phase 5/7; actor σ collapsed to **0.194**; actor mean
  sits ~1.5 action-units from its own critic's arg-max, on the LPS lobe.
  A tanh-Gaussian doing local gradient ascent cannot cross the valley.
* The **efficient high-engine-load region** (hard-engine LPS at low demand)
  has **8–27 % replay support** vs **48–54 %** for part-load, and twin-Q
  disagreement 10× higher there (Phase 9) — the critic cannot rank operating
  points it has never seen.
* Greedy exploitation of the trained critic (**Q-oracle**) is **worse** than
  the actor and loses CS (Phase 8) — the critic is not exploitable either.
⇒ **proximate cause = optimisation / exploration topology**, with the C1
costate calibration as a secondary contributor (SoC drift, 0–15 Nm over-EV).

---

## 5. ECMS OBJECTIVE + FAIRNESS VERDICT

### 5.1 What ECMS optimises **[F]**
Instantaneous Hamiltonian `H(u) = P_fuel(u) + λ_eff·P_batt(u)`, 81-point
feasible `u`-grid (`Δu ≈ 0.023` ≈ 0.7 Nm engine torque at 30 Nm demand),
`λ_eff = λ₀ + 8·(0.5−SoC)`. Braking: forced max regen (no optimisation).
**No finite-horizon term; no per-step preview.** It calls the **same** plant
blocks and the **same** feasibility masks the env uses (`_feasible_u_grid`), and
`evaluate_policy.py` routes it through the **same** `env.step` (reward,
powertrain, EFC). Physics and ledger are **fair**.

### 5.2 Where does 3.1887 / 2.8097 come from? **[F]**
`tune_lambda` runs the **entire cycle repeatedly**, bisecting `λ₀` until
`|SoC_end − 0.5| ≤ 0.5 %`, keeping the closest-to-target run. **`λ₀` is an
offline, whole-cycle-outcome calibration.** A causal online controller cannot
perform this; it must estimate `λ` online.

### 5.3 Fairness verdict **[F/I]**

| Question | Finding |
|---|---|
| Correct physics / ledger? | **Yes** — same blocks, same masks, same evaluator, CS to 0.36 pp. |
| Future / preview info per step? | **No** — ECMS has none; **SAC has 5 s speed preview**, so SAC has *more* per-step information. |
| Information advantage? | **Yes, one:** whole-cycle `λ₀` bisection. **Mitigation:** SAC is *handed* `λ₀` (as `eq_factor = λ₀/4.8309`), so the residual advantage is the global *consistency* the tuned scalar buys, not the scalar itself. Magnitude **unquantified** — needs a sensitivity sweep (task V2). |
| Idle-8 kW quantum? | Symmetric — both `H(u)` and the reward see it; ECMS will pick OFF over an idling engine, and so should a reward-optimal SAC. |
| Engine start/stop penalty? | **None in the plant** for either controller. ECMS can toggle every step for free; so can SAC. Not an asymmetry (but not physical either). |
| Braking regen | **Asymmetric** (C2): ECMS forced to max; SAC chooses. Empirically ≈ 0 fuel effect. |
| Grid-resolution limited? | Unknown — 81 points may or may not be tight. Needs V2. |
| Is 3.1887 near-DP? | **Unverified** — no DP solver exists. Constant-λ ECMS is documented as "a hair above DP", but that is an assertion, not a number for this plant. |

**Verdict [I]:** ECMS is a **correctly implemented, physically fair,
cycle-specifically-tuned local optimiser** with **one bounded information
advantage** (offline `λ₀`) that is **largely handed to SAC**. It is a
legitimate *reference*; whether beating it outright is *achievable* for a
causal controller is **currently unknown** and requires V2 (ECMS sensitivity)
+ a DP solver. Treat **3.1887 / 2.8097** as a firm *upper-bound target for the
constant-λ+feedback class at this grid*, not a proven physical floor.

---

## 6. PHYSICAL GAP DECOMPOSITION

Best available, from `phase7/data/ecms_gap_NEDC.json` and
`phase9/data/engine_physics_{NEDC,FTP75}.json` (both seed 0, demand-aligned
exact, `max|ΔT_demand| = 0`). Not recomputed here (Stage 0 = no execution); a
clean independent recompute on a common basis is task V1/V3.

### 6.1 NEDC — total SAC−ECMS gap **+0.498** L/100km (seed 0: 3.686 vs 3.189) **[F]**

Per engine-demand band, `ΔTotal` (`ecms_gap_NEDC.json`):

| band | ΔFuel | ΔElec | **ΔTotal** | SAC / ECMS OFF % | SAC / ECMS T_CE\|on (Nm) |
|---|---|---|---|---|---|
| brake | −0.001 | +0.012 | +0.010 | 0 / 0 | — |
| **0–15** | +0.018 | **+0.123** | **+0.141** | 34.9 / 34.9 | — |
| **15–30** | **+0.315** | −0.120 | **+0.196** | 58.4 / 76.5 | 31.6 / 39.9 |
| **30–35** | **+0.240** | −0.118 | **+0.122** | **4.3 / 48.7** | **35.2 / 57.8** |
| 35–50 | +0.050 | −0.024 | +0.026 | 51.2 / 63.4 | 50.1 / 67.5 |
| 50–75 | +0.002 | +0.001 | +0.003 | 0 / 22.5 | 70.3 / 95.3 |
| >75 | −0.127 | +0.127 | +0.000 | 0 / 0 | 93.9 / 106.9 |

* **≈ 92 % of the gap is generated at engine demand < 35 Nm** (0–15: +0.141;
  15–30: +0.196; 30–35: +0.122). **[D]** from the table.
* **0–15 Nm** is *not* a mode-selection difference (OFF % identical) — it is
  **+0.123 ΔElec**: SAC goes pure-electric at trivial load and pays fuel later
  to recharge. Classic myopic bank/spend-timing failure. **[F/I]**
* **15–35 Nm:** SAC keeps the engine **ON far more** (30–35 Nm: 96 % vs 51 %)
  **and runs it 20–25 Nm softer** (35 vs 58 Nm) → part-load, worse BSFC.
* **≥ 50 Nm:** SAC ≈ ECMS or better (>75 Nm: −0.127 ΔFuel, offset by regen
  banking). The high-demand split is near-forced and SAC handles it.

### 6.2 BSFC-grounded engine decomposition (`engine_physics`) **[F]**

| | NEDC SAC / ECMS | FTP75 SAC / ECMS |
|---|---|---|
| engine-on steps | **376 / 260** | 504 / 501 |
| mean T_CE when on | **55 / 79 Nm** | 70 / 68 Nm |
| mean BSFC when on | **290 / 255 g/kWh** | 253 / (artefact — C5) |
| mean engine η | **0.324 / 0.352** | 0.392 / (artefact) |
| A — operating-point BSFC (both on) | **+0.192 (39 %)** | **+0.082 (18 %)** |
| B + D — mode-selection & timing (net) | **+0.306 (61 %)** | **+0.373 (81 %)** |
| C — battery / SoC-equivalence | **≈ 0** | ≈ 0.005 |

### 6.3 Physical partition (mandate §6 categories) **[D/I]**

| Category | NEDC | FTP75 | Evidence |
|---|---|---|---|
| 1. Engine operating-point inefficiency (part-load BSFC) | **20–40 %** (C4 bracket; BSFC 290 vs 255) | ~18 % | 6.2 |
| 2. Engine ON/OFF decision (SAC runs it when ECMS coasts on banked charge) | **largest single, ~45–60 %** | ~55–65 % | 6.1 (30–35 Nm OFF 4 % vs 49 %), 6.2 (376 vs 260 on-steps) |
| 3. Battery charge/discharge strategy | ≈ 0 (ΔSoC matches; C ≈ 0) | ≈ 0 | 6.2 |
| 4. Regenerative braking | ≈ 0 (brake ΔTotal +0.010; REGEN % matches) | ≈ 0 | 6.1 |
| 5. SoC drift | ≈ 0 (both CS; but SAC operates at ~37.5 % vs ECMS ~50 % — a *state* difference, not a *fuel* difference) | ≈ 0 | Phase 7 §2 |
| 6. Transient / start-stop | **0 by construction** — no start cost in the plant (§1.3) | 0 | code |
| 7. Action discretisation / mapping | **minor** — gated map compresses the engine-load sub-range (Phase 7 §9); reachable-`u` set proven identical | minor | Phase 4/7 |
| 8. Other: over-EV-then-recharge at 0–15 Nm | **~10–15 %** (+0.123 ΔElec, timing) | ~10 % | 6.1 |
| 9. **ECMS whole-cycle `λ₀` tuning advantage** | **unquantified** (needs V2) | unquantified | §5.2 |

**One-sentence physical diagnosis [I]:** ECMS's strategy is *"engine OFF, or
engine HARD, rarely in between — bank the surplus"*; SAC's learned strategy is
*"engine ON, near demand, most of the time"* — and the fuel is lost in the
**part-load middle at < 35 Nm demand** that ECMS structurally avoids.

---

## 7. RANKED HYPOTHESIS LEDGER

Ranked by **current evidence**, not by Phase-10 ranking.

| # | Hypothesis | Evidence FOR | Evidence AGAINST | Confidence | Cheapest decisive experiment |
|---|---|---|---|---|---|
| **H1** | **Exploration/discovery failure for efficient *high-engine-load* (hard-engine LPS) operation** — a second deadlock, distinct from the already-fixed OFF deadlock. | Phase 9: HIGH_EFF/ECMS_NBHD replay support 8–27 % vs LOW 54 %; twin-Q disagreement 10×. §6: 20–40 % of gap is part-load BSFC; SAC T_CE\|on 55 vs 79. Phase 4 precedent (OFF deadlock was real and fixable by a coordinate remap). Bimodal Q, actor on the soft lobe. | Phase 6 coverage intervention failed — **but that targeted the OFF region, a different hole**. | **High** | Pure-RL targeted exploration schedule: with prob p, replace the sampled action by a **uniform draw from the feasible high-engine-load `u`-interval** (feasibility bracket, **no ECMS action**); 150k / 3 seeds; measure replay support in the efficient region, critic per-state arg-max, regional fuel. |
| **H2** | **Optimisation-topology failure:** the objective is ~an ECMS Hamiltonian but the tanh-Gaussian actor (σ collapsed to 0.19) cannot cross the bimodal-`r`/`Q` valley to the OFF / hard-engine lobes. | Phase 7: actor P(OFF) **flat** across k_fb∈{1.656,2.5,3.0} — policy not tracking its costate. Phase 5: σ 0.55→0.19; actor 4–5 σ from the OFF lobe. Phase 8: mixture actor collapsed. Q-oracle worse than actor. | Q-oracle also fails ⇒ not *purely* the actor; the critic is imperfect off-distribution too (⇒ H1). | **Med-High** | 3-seed A/B on `target_entropy ∈ {−1, −2, −3}` + an explicit σ-floor / entropy-anneal; measure P(OFF) & P(high-load) at 15–35 Nm vs the critic's arg-max. **(EXP-C, authorised since Phase 6, never run.)** |
| **H3** | **Base-costate calibration error (C1):** reward under-prices battery on a `P_EM` basis by ~40 % at target SoC ⇒ SoC drift, 0–15 Nm over-EV, need for a `k_fb` crutch. | §2.3 derivation. §6.1: 0–15 Nm is +0.123 ΔElec (cheap-battery over-EV). Phase 5 §F: needed steeper-than-ECMS k_fb. Median operating SoC 37.5 %. | Predicts *more* electric than ECMS overall; SAC uses *less* in 15–35 Nm ⇒ not the dominant term for the main gap. | **Med** (real in the objective; secondary to H1/H2 behaviourally) | **V1** (no training): realised `Σ Δfuel_L / Σ P_EM·Δt` and per-step arg-min on a common basis, CONTROL vs ECMS. Then, if confirmed: retrain with `eq_factor ≈ 0.466` (C1-corrected) **and** `k_fb ≈ 1.656`, 3 seeds — one coupled calibration change. |
| **H4** | **Short horizon (γ=0.20) blocks anticipatory bank/spend** ⇒ the 0–15 Nm ΔElec waste + part of mode-timing. | 1.25 s value horizon (§2.1). §6.1: entire gap is at low demand where timing dominates. Cross-cycle CS 0/3 (Phase 7) — memorised trajectory. | ECMS is *also* per-step myopic and does fine ⇒ myopia per se isn't fatal *if the costate is right*. γ sweeps were single-seed / confounded (C3, E10). | **Med** | Clean γ ∈ {0.20, 0.50, 0.75, 0.90} sweep, **3 seeds**, current stable config, n_step 1 throughout; watch critic-loss / Q-growth / CS; measure the 0–15 Nm ΔElec term specifically. **(Stage 1 as prompted.)** |
| **H5** | **State insufficiency for timing:** no prev-action/mode, no demand history, no time/distance-remaining ⇒ a myopic critic cannot infer *when* to bank. | Cross-cycle CS 0/3. γ=0.20 ⇒ critic can't use downstream reward, so state must carry timing. 0–15 Nm over-EV. | lookahead=5 already gives speed preview; Phase 8/9 asserted (weakly) sufficiency. | **Med** | Add one channel at a time (prev-`u`; then 10-step demand history); 150k / 3 seeds; measure fuel, CS, **and cross-cycle CS**. |
| **H6** | **Advantage/scale conditioning:** REWARD_SCALE=100 ⇒ OFF-vs-ASSIST advantage is 3–20 % of \|Q\|, critic RMSE ~24 % of \|Q\| ⇒ SNR < 1. | E7 (SNR<1). Phase 9: CQL blew up because its term (≈ ln 30) dwarfed the TD loss (~5e-3) by ~700× — a scale symptom. | E7: lowering γ shrank signal and noise together, SNR unchanged; critic *does* rank regions right on-distribution. | **Low-Med** | Return-normalisation A/B (running-stats on returns / reward scale), semantics unchanged; 150k / 3 seeds. |
| **H7** | **Training budget (150k) too small on the *stable* config.** | C3: the anti-long-training result is from the diverging-critic config; current config trains stably (Phase 8C). | γ=0.20 ⇒ little temporal information to gain from more data; 8C curves plateau. | **Low** | Extend CONTROL to 500k, 3 seeds; inspect learning-curve slope + Q growth. |
| **H8** | **Action-mapping compression (gated map).** | Phase 7 §9: gated map compresses ASSIST/engine-load sub-range; correlates with soft-engine operating point. | Reachable-`u` set proven identical (53 tests); actor fails to reach even the OFF share its critic wants. | **Low-Med** | Dense-action audit (no training): `du/dT_CE` resolution across the gated map at representative 15–50 Nm states; add a `linear`-map arm only if compression is shown. |
| **H9** | **Critic value-fidelity off-distribution** (Phase 8's stated cause). | Q-oracle fails; Q@actor-load highest in every band. | Phase 9: critic **not** grossly wrong on-distribution; CQL made it worse. **Downstream of H1** (coverage), not independently actionable. | **Med** (as a *symptom* of H1) | Covered by H1's decisive experiment (does closing the coverage hole re-order the arg-max?). |
| **H10** | **ECMS structural / tuning advantage ⇒ part of the nominal gap is irreducible for a causal controller.** | λ₀ whole-cycle bisection (§5.2). No DP reference exists. | SAC is handed λ₀; physics identical; SAC has *more* per-step preview. | **Med** (that *some* is irreducible; magnitude unknown) | **V2** (no training): ECMS sensitivity sweep (λ₀ ± 20 %, grid 41/81/161/321, k_fb ∈ {0,4,8,16}, CS-tol 0.5 %/2 %) + a 1-D-SoC DP solver (100-bin SoC × 1221 steps × 81 u — seconds). Bounds addressable vs irreducible gap. |
| H11 | Reward semantics wrong (fuel/battery term structure). | — | Telescopes exactly to −v_ce_equiv at eq_factor 1, k_fb 0 (verified). The metric is the metric. | **Very Low** | none — do not modify reward semantics. |
| H12 | Plant / powertrain bug. | — | 9 MATLAB checks; env↔plant 1e-9; C1/C2 are RL-layer / definitional, not plant bugs. | **Very Low** | none. |
| H13 | Evaluator mismatch. | — | Single code path, all controllers through the same `env.step` (§`evaluate_policy.py`). | **Very Low** | none. |

---

## 8. THE SINGLE MOST INFORMATIVE NEXT EXPERIMENT

**Two no-training verifications must complete Stage 0 first** (hours, not days;
they change what "the target" and "the objective" even are):

* **V1 — costate-currency cross-check (C1/H3).** Over the CONTROL (3 seeds) and
  ECMS rollouts, compute the realised `Σ Δfuel_L ÷ Σ (P_EM·Δt)` and each
  controller's per-step arg-min of its **own** instantaneous objective on a
  common `P_EM` basis. Deliverable: a definitive statement of whether the SAC
  reward's effective costate is ~40 % below ECMS's, and by how much, as a
  function of SoC. *If confirmed, the Phase-2/5/7 costate narrative is revised
  and H3 becomes a coupled `eq_factor`+`k_fb` recalibration.*
* **V2 — ECMS sensitivity + DP reference (H10).** ECMS sweep {λ₀ ± 20 %, grid
  41/81/161/321, k_fb ∈ {0,4,8,16}, CS-tol} + a 1-D-SoC DP solver. Deliverable:
  the achievable floor, and the split of the nominal +0.50 gap into
  *addressable* vs *ECMS-tuning-advantage* vs *grid-resolution*.

**Then — the single most informative *training* experiment: pure-RL targeted
exploration of the efficient high-engine-load region (H1).**

* **Hypothesis:** the critic's low-load arg-max bias at 15–35 Nm is a
  self-reinforcing coverage deadlock in the efficient hard-engine region
  (support 8–27 % vs 54 % for part-load). Guaranteeing feasible coverage there
  re-orders the per-state arg-max and lets the plain actor track it — closing
  the 20–40 % operating-point component and part of the mode-timing component.
* **Change (one variable):** training-time exploration only. When
  `15 ≤ T_MGB < 50 Nm`, engine commanded ON, and a higher feasible engine load
  exists, with prob `p ≈ 0.25` replace the sampled action with a **uniform
  draw from the feasible high-engine-load `u`-interval** (`T_CE ∈ [1.3·demand,
  0.9·T_CE_max]` — a *feasibility* bracket). `predict()` untouched
  (evaluation-safe by construction, identical safeguard to Phase 6).
* **Purity:** the injected action is a *uniform draw over a feasibility-defined
  interval* — it encodes "a harder engine load is reachable here", never what a
  good controller would pick. **No ECMS/benchmark action, trajectory,
  demonstration, imitation loss, or warm-start.** Compliant with §1.
* **Frozen:** plant, env, ECMS, rule-based, evaluator, SAC impl, net_arch,
  optimiser, replay config, action bounds, feasibility logic, reward
  semantics, γ 0.20, n_step 1, k_fb 2.5, eq_factor, lookahead 5, seeds {0,1,2}.
* **Budget / protocol:** 150k × 3 seeds → `readiness_gate` → 500k for a
  survivor.
* **Measurements (records `results/phase11/`):** replay support in
  HIGH_EFF/ECMS_NBHD (target: 8–27 % → ≥ 40 %); twin-Q disagreement there;
  per-state arg-max-Q region distribution; V_CE ± SD + per-seed; CS count;
  ΔSoC; regional fuel (the §6.1 table); engine T_CE\|on, BSFC, on-steps;
  action distribution; critic/actor loss + entropy curves.
* **Falsification design (mandate §17):** if replay support rises but the
  arg-max and fuel **do not move** (the Phase-6 outcome, now in a *different*
  region), that falsifies "coverage is sufficient" for **both** OFF *and*
  efficient-load regions — strong evidence the value-based route is exhausted
  and the advantage genuinely sits below the critic's noise floor (→ H6/H2 or a
  structural change). If support rises **and** the arg-max re-orders **but
  fuel/CS worsen**, that isolates an actor-tracking failure (→ H2 becomes
  primary). If support rises, arg-max re-orders, **and** fuel improves with CS
  held, H1 is confirmed and the operating-point component is recovered.
* **Why this over the γ sweep (Stage 1 as literally prompted):** the γ sweep
  (H4) is worth running, but H4 is **Med** confidence and partly confounded
  (C3), whereas H1 is **High** confidence, attacks the largest cleanly-
  attributed component, is decisive either way, and is cheaper to interpret.
  Recommend running H1 as Stage 1 and folding the γ sweep in as Stage 1b.

---

## STAGE 0 DELIVERABLE CHECKLIST (mandate §19)

1. **Verified facts** — §1 (config table, exact reward, constants).
2. **Mathematical equations** — §2 (SAC objective, ECMS Hamiltonian,
   costate-currency derivation).
3. **Contradictions found** — §3 (C1 costate currency; C2 stale regen
   docstring; C3 stale "don't train longer"; C4 6× attribution disagreement;
   C5 Phase-9 FTP75 artefact; C6 "reward sufficient" rests on an instantaneous
   argmax).
4. **Current SAC objective** — §4 (A–F): a myopic equivalent-fuel Hamiltonian,
   1.25 s horizon, CS not an internal constraint, ECMS trajectory is **not** an
   optimum of it (`λ_SAC < λ_ECMS` ∀ SoC), and the trained policy additionally
   fails to reach its *own* optimum in 15–35 Nm.
5. **ECMS objective** — §5: instantaneous Hamiltonian, 81-pt grid, whole-cycle
   `λ₀` bisection; **fair on physics/ledger**, one bounded (largely-handed)
   information advantage; 3.1887/2.8097 = firm target for the class, not a
   proven physical floor (no DP exists).
6. **Physical gap decomposition** — §6: **≈ 92 % of the NEDC gap is below
   35 Nm demand**; ~45–60 % engine ON/OFF mode-selection, 20–40 % part-load
   BSFC, ~10–15 % 0–15 Nm over-EV-then-recharge, ≈ 0 battery/regen/transient.
7. **Ranked hypotheses** — §7 (H1 high-load exploration deadlock = highest
   confidence; H2 optimisation topology; H3 costate calibration; H4 horizon;
   H5 state; then conditioning, budget, mapping; plant/evaluator ruled out).
8. **Single most informative next experiment** — §8: V1 + V2 (no training) to
   close the audit, then the **pure-RL targeted high-engine-load exploration
   schedule (H1)**, 150k × 3 seeds, one variable, with an explicit
   falsification design.

---

**STAGE 0 COMPLETE. No code changed. No training run. Awaiting approval before
Stage 1.**
