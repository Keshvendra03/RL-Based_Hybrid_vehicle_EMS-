# PHASE 10 — INDEPENDENT GAP INVESTIGATION

**Full project re-audit + independent ECMS-gap diagnosis.**
No new training run. Analysis only, over the committed code
(`f1f45c5`), the Phase 1–9 reports, `results/phase{4..9}/data/*.json`, and the
external HEV/RL literature.

> **Standing instruction honoured:** this deliverable is the *diagnosis and
> ranked experimental plan*. It does **not** implement a Phase-10 training
> experiment. It is explicitly authorised (mandate §30) to contradict earlier
> phases, and it does so in several places — most importantly the "the optimum
> is myopic / γ is closed" conclusion and the "reward/state are fine"
> conclusions, which were reached under a 150k-step budget on a
> since-abandoned diverging-critic baseline and are over-claimed.

---

## 1. EXECUTIVE VERDICT

**Why is SAC still worse than ECMS after Phases 1–9?**

Because Phases 1–5 reformulated the problem — for sound tractability reasons —
into one whose optimum ECMS computes *analytically* and SAC must *learn by
trial and error from ~150k steps of its own data*, and that data never covers
the region where the two controllers differ. Concretely, the project drove
γ→0.20 (a ≈1.2 s value horizon), added a per‑step costate feedback term
(`k_fb`, slope ≈1.5× ECMS's), and reparameterised the action coordinate
(`modeaware_gated`). Under that reformulation the SAC agent is a *myopic
one‑step Hamiltonian minimiser with a learned, cycle‑overfit costate* — i.e. a
noisy, function‑approximated adaptive‑ECMS. The residual +0.58 L/100km NEDC /
+0.48 FTP75 gap to true ECMS decomposes (Phase 9 `engine_physics`, BSFC‑grounded)
into:

* **Engine ON/OFF selection and timing — NEDC ≈61 % / FTP75 ≈81 % of the gap.**
  ECMS runs the engine **fewer steps** (NEDC 260 vs SAC 376) because its single
  scalar λ₀ is bisection‑tuned over the *whole cycle*, making every
  "instantaneous" decision globally consistent. SAC's learned per‑step costate
  is a local, noisier surrogate, and — decisively — its stochastic Gaussian
  policy sits on the **wrong (LPS / part‑load) lobe of a bimodal action‑value
  surface** at 15–35 Nm demand, because the on‑policy replay distribution
  **never visits** the sustained‑engine‑OFF / high‑engine‑load region and the
  critic therefore has no data to value it (an exploration deadlock, confirmed
  in Phase 4 for OFF and re‑confirmed in Phase 9 for the efficient engine‑load
  region: HIGH_EFF/ECMS_NBHD replay support 8–27 % vs 48–54 % for LOW).

* **Engine operating‑point / BSFC quality — NEDC ≈39 % / FTP75 ≈18 %.**
  When SAC *does* run the engine at low demand it runs it **near demand**
  (part load, ~290 g/kWh, η 0.32) instead of loading it into the efficient
  island (~255 g/kWh, η 0.35) and banking the surplus as charge. The
  *instantaneous reward's* argmax actually wants the harder point
  (`reward_counterfactual`: argmax‑r T_CE > ECMS T_CE in every band), but the
  myopic critic — fit only on the narrow part‑load distribution its own actor
  generates — does not propagate that preference.

* **Battery / SoC energy ledger — ≈0 %.** SAC's ΔSoC ≈ ECMS's; the
  equivalent‑energy term is not the problem and has not been for four phases.

**What is *not* the answer** (all tested, all refuted or inconclusive):
economic over‑pricing of battery energy (Phase 7), `k_fb` value (Phase 7,
flat plateau), replay OFF‑coverage at the operating SoC (Phase 6), multimodal
policy class (Phase 8C), conservative/pessimistic critic (Phase 9 CQL, failed
at every α), and "the reward lacks engine‑efficiency information" (Phase 8 §16).

**What has NOT been cleanly tested and is where the remaining leverage lives:**

1. **Removing the requirement to *learn the pointwise optimiser at all*** — an
   architecture where RL outputs a state/history‑dependent **equivalent factor
   / costate** (or a correction Δλ around a nominal) and a **deterministic
   Hamiltonian minimiser** produces `u`. This attacks *both* dominant gap
   components simultaneously and reduces the RL problem to smooth 1‑D scalar
   regression. It is directly supported by the entire Phase 5–9 forensic chain
   and by a substantial recent literature (DRL‑ECMS, A‑ECMS‑by‑RL).
2. **Targeted exploration / offline warm‑start covering the efficient
   high‑engine‑load region** (Phase 9 "Experiment B", designed but never run;
   plus an offline non‑expert feasible dataset — never tried).
3. **State features and value horizon adequate for anticipatory
   load‑point‑shifting.** The "myopic optimum" conclusion was validated only at
   a 150k‑step budget on a *diverging‑critic* baseline (γ=0.9999,
   gradient_steps=64) and is not safe as a general claim (§6, §8, §21 below).

**If the answer is not yet fully identifiable:** the split between
"irreducible causal‑controller gap below ECMS" and "addressable RL
approximation error" is **unidentified**, because the project has **no dynamic
programming reference**. ECMS itself is a cycle‑tuned local optimiser (§5), so
part of "the gap" may not be recoverable by *any* causal controller and part
of ECMS's lead may be an unfair information advantage. Building a DP reference
is a prerequisite for closing this out honestly.

---

## 2. RECONSTRUCTED PROJECT HISTORY (Phases 1–9)

Convention: **objective → hypothesis → intervention → result → conclusion →
does the conclusion still hold?**

### Phase 1 — Powertrain port + MATLAB validation
* **Obj:** pure‑Python powertrain equivalent to the Simulink model.
* **Result:** 9 validation checks pass; NEDC baseline fuel 4.535 vs MATLAB
  4.513 (~0.5 %); advanced rule‑based 3.506 vs MATLAB 3.348 (systematic offset,
  relative comparisons valid).
* **Conclusion:** plant is locked, not to be modified during RL work.
* **Still valid?** **Yes**, with two caveats surfaced here (§4): (a) the FTP75
  benchmark 3.2323 was never MATLAB‑cross‑checked, only pipeline‑consistency
  confirmed; (b) `Q_BT` final differs from MATLAB by ~6 % — systematic, but it
  means the battery‑energy accounting carries a few‑percent absolute bias that
  is invisible in strategy‑vs‑strategy comparison and irrelevant to the gap.

### Phase 2 — Gym env + reward unit fix + myopic pivot
* **Obj / hyp:** diagnose why the trained agent parks in "ASSIST blob"
  (OFF 10–29 % vs benchmark 53–59 %). P0 = action geometry.
* **Interventions:** (a) reward battery‑price unit fix (`elec_liters` is
  *already* EFC‑converted, so `eq_factor` must be `λ_ECMS/4.8309` ≈ 0.2717 NEDC,
  not 1.0–1.3125); (b) γ sweep 0.9999→0.00.
* **Result:** P0 (geometry) partially confirmed, not primary. Reward fix alone
  at γ=0.9999 = no improvement (4.178). γ↓ + reward fix *together* → NEDC 3.777,
  FTP75 3.418, both charge‑sustaining for the first time. "Two coupled root
  causes: reward unit mismatch + horizon mismatch."
* **Conclusion:** "the optimum here is myopic (Pontryagin: the only
  inter‑temporal coupling is SoC, carried by the costate feedback), γ=0.20."
* **Still valid?** **Partially — and over‑claimed.** The reward *unit fix* is
  correct and necessary. The "myopic" framing is **not proven**: (i) single
  seed for most γ arms; (ii) high‑γ arms were run on the *diverging‑critic*
  baseline (gradient_steps=64) — the degradation was critic instability, later
  fixed independently (E10 / VERIFIED_FACTS §E), never re‑tested at high γ with
  the stable config; (iii) a truly myopic policy provably **cannot** perform
  anticipatory load‑point shifting, which is exactly the behaviour ECMS gets
  "for free" from its whole‑cycle‑tuned λ₀. See §6, §8, §21.

### Phase 3 — SAC pipeline audit
* **Result:** SAC implementation verified line‑by‑line vs SB3 2.9.0;
  custom `NStepSAC` formula‑identical; 211/211 tests pass.
* **Still valid?** **Yes.** The RL *implementation* is not in question.

### Phase 4 — Exploration‑deadlock diagnosis + gated action map
* **Hyp:** engine‑OFF at 30–50 Nm sits 3.9–6.7 σ from the actor mean under the
  linear map → never proposed → no buffer data → critic can't learn Q(OFF) →
  no gradient. Self‑reinforcing.
* **Intervention:** `modeaware_gated` action‑coordinate remap (OFF gets a
  fixed 40 % of the action range where physically reachable; linear elsewhere).
  Control‑equivalence proved (53 tests).
* **Result:** FTP75 → benchmark (3.246, best seed 3.209 = −0.7 %). NEDC
  **regressed** (SoC runaway on 2/3 seeds) — at γ=0.20 the terminal CS penalty
  is invisible, so only per‑step `k_fb` controls SoC and it was too weak for
  the extra OFF freedom.
* **Conclusion:** γ×action‑map interaction; raise `k_fb` on NEDC.
* **Still valid?** **Yes for the mechanism** (deadlock is real and recurs). The
  gated map is a genuine but *partial* fix — it made OFF reachable; it did
  **not** make the efficient high‑engine‑load / LPS‑banking region reachable
  (Phase 9), and Phase 7 §9 notes it *compresses* the engine‑load sub‑range.

### Phase 5 / 5B — costate gain `k_fb`, SoC stabilised, bimodal‑Q found
* **Intervention:** `k_fb` sweep (gated, γ=0.20).
* **Result:** `k_fb=2.5` → NEDC 3/3 CS, fuel **statistically tied** with the
  linear reference (3.7666 vs 3.7727). Optimal `k_fb` is **cycle‑dependent**
  (NEDC 2.5, FTP75 1.656). 5B: replay OFF‑coverage is *adequate in aggregate*
  (15.6 % at 30–35 Nm) but *conditionally deficient at the operating SoC*
  (4.5 %, ~276 samples). Critic landscape is **bimodal** (LPS lobe / valley /
  OFF lobe); actor σ collapsed to 0.194, OFF lobe 4–5 σ away.
* **Conclusion:** bottleneck = critic misestimation of Q(OFF) from a
  conditional coverage hole; bimodal‑Q is a *symptom*.
* **Still valid?** **Mechanism yes, primary‑cause label superseded** by
  Phase 6. `k_fb=2.5` = **12.1 ECMS units of feedback slope** vs ECMS's own
  8.0 — the *training* costate is 1.5× steeper than the proven‑optimal one and
  (Phase 5 §F) makes battery "nearly free" above 50 % SoC. This mismatch was
  patched, not resolved (§6).

### Phase 6 — controlled conditional‑exploration A/B → REFUTED
* **Intervention:** inject feasible‑OFF actions at 15–35 Nm / SoC 40–55 %,
  p=0.30. Coverage at 30–35 Nm / SoC 40–50: 4.5 %→36.7 % (4.9×).
* **Result:** ΔQ(OFF−ASSIST) moved −0.0071→−0.0066 (**nothing**). Fuel worse on
  both cycles; NEDC CS 3/3→2/3. **Correction to 5B:** at 30–35 Nm reward and
  critic *agree* on ASSIST in 85–87 % of states — there was no conflict to
  repair; the 5B claim generalised the 15–30 Nm aggregate.
* **Conclusion:** conditional‑coverage hypothesis **rejected**. New bottleneck
  label: actor displacement at 15–30 Nm.
* **Still valid?** **Yes** — but note the coverage that was added was OFF
  coverage where OFF was *already* the reward/critic preference. Coverage of
  the *efficient high‑engine‑load* region (a different lever) was never added.

### Phase 7 — economic/costate forensic (no training) → REFUTED
* **Hyp:** the gap is an economic (equivalent‑factor) valuation error.
* **Result:** SAC's effective battery price (median 2.82 ECMS units NEDC)
  *matches ECMS's own closed‑loop effective price* (2.78) — both run below SoC
  target (median visited SoC 37.5 %). `k_fb` P(OFF) response is **flat** across
  {1.656, 2.5, 3.0}. `ERROR_reward ≥ 0`, `corr(ERROR_critic, eq‑price) ≈ 0`.
  Cross‑cycle transfer fails CS 0/3 both directions.
* **Conclusion:** **CASE D** — critic ranking ≈ right, actor displaced onto the
  LPS lobe of a bimodal Q; → **CASE E** if the actor‑side lever fails.
* **Still valid?** **Yes.** Economic hypothesis is genuinely dead. The
  cross‑cycle CS failure is important and under‑exploited evidence (§7): the
  policy has fitted a *cycle‑specific SoC trajectory*, not a transferable
  control law — a hallmark of over‑fitting to a 150k‑step single‑cycle budget.

### Phase 8 — Q‑oracle ceiling + mixture actor → policy class is NOT the cause
* **Q‑oracle** (greedy argmax of trained min‑Q, real‑env rollout): NEDC
  3.7666→**3.9404 (1/3 CS)**, FTP75 3.2889→**3.3545 (0/3 CS)** — *worse* than
  the actor, loses CS.
* **Engine‑op counterfactual:** `Q@actor‑load` is the **highest** value in
  every band on both cycles; `argmax‑r T_CE > ECMS T_CE` everywhere.
* **8C mixture actor** (2‑component, all else frozen): NEDC 3.873 (1/3 CS,
  *worse*), components collapsed (separation 0.03–0.16); FTP75 3.246 (3/3 CS,
  joint‑best, still +0.4 % over RB).
* **Conclusion:** binding constraint = **critic value‑fidelity off the
  on‑policy distribution**; policy class is not it; reward is "sufficient, do
  not modify"; ECMS has an irreducible pointwise‑precision advantage.
* **Still valid?** **Q‑oracle result: yes and important** (an ideal actor on
  this critic does worse → the critic, not the policy head, is the ceiling).
  "**Reward is sufficient, do not modify**" is **over‑claimed** — it rests on
  an *instantaneous* argmax; the mandate's own chain
  (instantaneous ≠ finite‑horizon ≠ charge‑sustaining ≠ ECMS optimum) is
  precisely what an instantaneous argmax cannot test (§6).

### Phase 9 — critic forensics + CQL → critic regularisation FAILS
* **Critic error map:** on ECMS‑trajectory states the critic is **not grossly
  wrong** — region‑mean min‑Q ranks HIGH_EFF ≳ ECMS_NBHD ≳ LOW ≳ OFF in every
  band, matching reward and next‑SoC. Neither pre‑registered error type
  cleanly triggered. Real defect: a **mild systematic low‑load bias in the
  per‑state argmax** across 15–35 Nm (argmax ∈ {OFF,LOW,ECMS_NBHD} ≈98 %,
  HIGH_EFF ≈2 %), in a region with **8–27 % replay support** vs 48–54 % for LOW,
  and twin‑Q disagreement 0.056–0.066 (vs 0.004–0.01 elsewhere).
* **BSFC decomposition:** NEDC operating‑point +0.192 (39 %) / mode‑timing
  +0.306 (61 %) / battery ≈0; FTP75 +0.082 (18 %) / +0.373 (81 %) / ≈0. ECMS:
  260 vs 376 engine‑on steps, 79 vs 55 Nm when on, 255 vs 290 g/kWh, η 0.35 vs
  0.32.
* **Experiment A (CQL(H) conservative critic):** FAILS at α ∈ {0.01, 0.05, 1.0}
  — SoC runaway to 78–86 % (V_CE 4.7–5.6, 0/3 CS) or CS only with 100+
  violations; gap "closed" = −213 %. CQL cut the OFF argmax but shifted mass to
  **LOW**, not the efficient region.
* **Conclusion:** neither actor‑capacity nor critic‑regularisation closes the
  gap. Next: (B) targeted efficient‑region coverage, then (H) a part‑load
  reward term. Algorithm swap gated.
* **Still valid?** **Yes.** Phase 9 is the strongest single phase: it
  correctly relocates the defect from "gross critic error" to "a compounding
  low‑load argmax bias in a data‑starved region." That framing is the launch
  point for Phase 10.

### Attribution inconsistency to flag
Phase 7 `gap_split` (analysis‑based) puts NEDC at **mode‑selection +0.608 /
operating‑point +0.032**; Phase 9 `engine_physics` (BSFC‑grounded) puts it at
**mode/timing +0.306 / operating‑point +0.192**. Both sum to ≈+0.50, but the
*share* assigned to operating‑point differs 6× (6 % vs 39 %). The two methods
draw the "mode vs operating‑point" boundary differently (Phase 7 counts a
switched‑off engine step as pure mode‑selection; Phase 9 folds the electrical
credit ECMS earns from its extra OFF into residual D = −0.297). **Neither is
wrong, but the project has been quoting whichever is convenient.** For Phase 10
planning, treat the honest bracket as: **operating‑point 20–40 %, mode/timing
55–75 %, battery ≈0** on NEDC.

---

## 3. COMPLETE GAP DECOMPOSITION (NEDC + FTP75)

### 3.1 The true gap, all controllers

Latest validated numbers (`results/evaluate_policy.py`, `PHASE8_REPORT.md` §1,
`PHASE9_FINAL_REPORT.md` §1, `experiment_registry.yaml`). CONTROL = gated
`k_fb=2.5`, γ=0.20, n_step 1, eq_factor 0.2717/0.4981, 3 seeds, 150k steps.

| # | Controller | NEDC V_CE (L/100km) | NEDC ΔSoC / CS | FTP75 V_CE | FTP75 ΔSoC / CS | NEDC vs ECMS |
|---|---|---|---|---|---|---|
| 1 | **SAC CONTROL** (3‑seed mean) | **3.7666 ± 0.079** | +0.28/−0.72/+0.23 pp · **3/3** | **3.2889 ± 0.017** | −0.53/−0.51/−0.07 pp · **3/3** | **+0.578 / +18.1 %** |
| 1b | SAC CONTROL best seed (s0) | 3.6862 | +0.28 pp · CS | 3.2699 | CS | +0.498 / +15.6 % |
| 2 | Best SAC found so far | NEDC = CONTROL (3.7666) | — | **3.2460** (gated k_fb 1.656 / 8C mixture) | 3/3 | — |
| 3 | Advanced rule‑based | **3.5056** | +2.47 pp; SoC_min **0.61 %** (38 steps < 5 %) | **3.2323** | +3.86 pp | — |
| 3b | Rule‑based, authority‑equal (through env SoC masks) | 3.5792 | SoC_min 4.64 % | 3.2318 | — | — |
| 4 | **ECMS** (λ₀ bisection‑tuned, k_fb 8.0, 81‑pt grid) | **3.1887** | +0.36 pp; λ₀ 1.3125 | **2.8097** | +0.13 pp; λ₀ 2.4062 | — |
| 5 | DP / global optimum | **NOT AVAILABLE** — never built | — | **NOT AVAILABLE** | — | — |
| 6 | Phase‑8 Q‑oracle (greedy min‑Q, real env) | 3.9404 ± 0.47 | −2.48 pp · **1/3** | 3.3545 ± 0.03 | −4.05 pp · **0/3** | +0.752 / +23.6 % |
| 7 | Phase‑8 mixture actor (2‑component) | 3.8730 ± 0.12 | +2.4/+1.9/+3.4 pp · **1/3** | 3.2462 ± 0.003 | 3/3 | +0.684 |
| 8 | Phase‑9 CQL(H) critic (α=1.0) | 4.9996 ± 0.23 | −3.1/−2.4/+10.3 pp · **0/3** | not run | — | +1.811 / +56.8 % |

**Reading:** every intervention since the Phase‑2 pivot has moved NEDC V_CE
within ±0.13 of 3.77 or made it worse. FTP75 is essentially *at* the rule‑based
benchmark (+0.4–1.8 %) and the open problem is **NEDC vs both benchmarks** and
**both cycles vs ECMS**.

### 3.2 Per‑band gap, NEDC (Phase 7 `ecms_gap_NEDC.json`, seed 0, demand‑aligned exact)

| T_MGB band | ΔFuel | ΔElec | **ΔTotal** | SAC OFF% / ECMS OFF% | SAC T_CE|on / ECMS |
|---|---|---|---|---|---|
| brake | −0.001 | +0.012 | +0.010 | 0 / 0 | — |
| **0–15** | +0.018 | **+0.123** | **+0.141** | 34.9 / 34.9 | — |
| **15–30** | **+0.315** | −0.120 | **+0.196** | 58.4 / 76.5 | 31.6 / 39.9 |
| **30–35** | **+0.240** | −0.118 | **+0.122** | **4.3 / 48.7** | **35.2 / 57.8** |
| 35–50 | +0.050 | −0.024 | +0.026 | 51.2 / 63.4 | 50.1 / 67.5 |
| 50–75 | +0.002 | +0.001 | +0.003 | 0 / 22.5 | 70.3 / 95.3 |
| >75 | −0.127 | +0.127 | +0.000 | 0 / 0 | 93.9 / 106.9 |
| **TOTAL** | | | **+0.498** | | |

**92 % of the NEDC gap is generated below 35 Nm demand** (0–15: +0.141;
15–30: +0.196; 30–35: +0.122). The physical story per band:

* **0–15 Nm (+0.141, almost all ΔElec):** SAC and ECMS run OFF equally often
  (35 %), but SAC pays **+0.123 L/100km‑equiv of electrical cost** here — it
  goes pure‑electric at trivial load and then must burn fuel later to put that
  charge back. ECMS is more willing to let the engine idle/carry trivial load.
  This is a *timing* problem: cheap EV now, expensive recharge later.
* **15–35 Nm (+0.318):** SAC keeps the engine ON far more than ECMS
  (30–35 Nm: SAC 96 % on vs ECMS 51 %) **and runs it 20–25 Nm softer**
  (35 vs 58 Nm). ECMS either shuts it off and coasts on banked charge, or
  loads it hard into the efficient island. SAC does the in‑between everywhere.
* **50 Nm+ : SAC ≈ ECMS or better** (>75 Nm: SAC −0.127 ΔFuel). At high demand
  the split is nearly forced and SAC does fine; ECMS keeps a little OFF at
  50–75 Nm (22.5 %) that SAC never uses, but the fuel effect is ~0.

### 3.3 Per‑band, FTP75 (Phase 9 `engine_physics_FTP75.json`, Phase 5B §10)

FTP75's gap is **more broadly distributed** (15–30, 35–50, 50–75 each
contribute comparably) and **81 % mode/timing, 18 % operating‑point**. FTP75
engine‑on *counts* match ECMS (504 vs 501) but 249 steps are "exactly one of
the two engines on" — the **split is wrong at the right times, or right at the
wrong times**. Denser braking (REGEN 25.7 % vs NEDC 17 %) supplies charge
without LPS, which is why FTP75 is easier and why its optimal `k_fb` is lower.

> **⚠ Forensic‑instrumentation bug found (§4):** `engine_physics_FTP75.json`
> reports `mean_bsfc_ecms = 457` and `mean_eff_ecms = 2.4e11` — divide‑by‑near‑
> zero when ECMS torque is tiny. The per‑band FTP75 numbers are sane; the
> cycle‑mean ECMS BSFC/efficiency figures for FTP75 in the Phase‑9 report are
> artefacts and should not be quoted.

### 3.4 Operating‑point / engine map (from `data/maps/engine_maps_data.py`)

* Engine efficiency map peaks at **η ≈ 0.343** ⇒ BSFC_min ≈ 245 g/kWh.
* Map speed grid: **125.7–439.8 rad/s (1200–4200 rpm)**. Below 105 rad/s the
  engine is *idle‑clamped at 8 kW fuel power regardless of torque*.
* Efficient island: mid‑high torque (≈60–140 Nm) at 1800–3500 rpm, η 0.33–0.34.
* **Engine speed is NOT a control variable** — `w_CE = w_MGB = w_wheel · i_gt`,
  fixed by vehicle speed × prescribed gear. The only instantaneous DOF is the
  torque split `u`. So "operating‑point optimisation" is 1‑D (choose engine
  torque along a vertical line on the BSFC map). ECMS's grid search does this
  exactly; SAC's policy does it approximately.
* **No engine start cost, no minimum‑on time, no warm‑up model** in the plant
  (only fuel‑cutoff at T_CE ≤ 5 Nm). Therefore **switching frequency is not
  penalised by physics** and excessive switching is *not* a hidden cost source
  (§14). The 8 kW idle quantum at very low speed IS a real discrete cost.

---

## 4. IMPLEMENTATION AUDIT

Line‑by‑line review of `src/env/ems_env.py`, `src/env/powertrain.py`,
`src/env/driving_cycle.py`, `src/baselines/ecms.py`,
`src/baselines/advanced_rule_based.py`, `src/agents/train_sac.py`,
`data/params.json`, `data/maps/engine_maps_data.py`.

Classification: **CC** confirmed correct · **PC** probably correct ·
**U** uncertain · **LP** likely problematic · **BUG?** critical‑bug candidate.

### 4.1 Environment / plant

| # | Item | Finding | Class |
|---|---|---|---|
| E1 | Vehicle longitudinal dynamics (`VehicleDynamics.step`) | `v_a = 0.5(v+v_prev)`, F_roll switched at `v_a>0`, F_aero on `v_a²`, F_iner on raw `dv`, `w_wheel` from `v_a` not `v`. Matches MATLAB checks. | **CC** |
| E2 | Gearbox wheel→flywheel (`gearbox`) | Asymmetric efficiency (friction added before ÷η driving; ×η after for braking), neutral‑protection zeroes outputs. | **CC** |
| E3 | Engine inertia + idle clamp + fuel cutoff (`combustion_engine`) | `w_CE=max(w_gear,105)`; `T_CE=max(T_total,5)`; 2‑D map on **(speed, torque)** directly (no p_me conversion); idle → 8 kW whenever `0<w_gear≤105`; cutoff → 0 W when `T_CE≤5` and not idle. | **CC** |
| E4 | Idle 8 kW quantum | At `w_MGB≤105` (≈ v<10 km/h in gear 1) an "on" engine burns 8 kW *flat*, torque‑independent. Real discrete non‑convexity in the reward surface; both controllers see it. | **CC** (but see R‑audit §6) |
| E5 | Motor efficiency map (`electric_motor`) | Signed‑torque map: η for T<0 (regen), 1/η for T>0 (motoring); single formula `P_EM = T·w·η_factor`. Clamped to grid; sentinel 10.0 guarded. | **CC** |
| E6 | Battery (`Battery.step`) | Charge integrator `Q[k]=Q[k-1]−h·I[k-1]`; mode‑switched U_BT (charge/discharge/idle Fcn blocks); `_battery_energy(Q)=0.5·U_oc(Q)·Q` capacitor‑analogy; V_BT = (E_init−E_now)/x_tot/36. | **PC** — see E7 |
| E7 | "−C−" constant = `U_oc(Q_BT_IC)` = 46.8 V | Derived, not extracted from Simulink. The docstring flags it: "If your f(u) block returns something other than `U_oc(Q)·Q`, update `_battery_energy`." Battery ledger term is ≈0 % of the gap regardless, but the 6 % `Q_BT` mismatch vs MATLAB (Phase 1) traces here. | **U** (low impact) |
| E8 | EFC block (`equivalent_fuel_consumption`) | `gain = 1/(η_BT·η_EM·η_CE·(H_u/3.6e6)·ρ_f)` with η_BT/EM/CE = 0.9/0.8/0.25 (fixed, from `params.json`); saturation lower=0 (no credit for net charge), upper=∞. | **CC** |
| E9 | **η_CE = 0.25 in the EFC gain vs peak map η ≈ 0.34** | The *evaluation metric* prices battery energy using a fixed engine efficiency of **0.25**, but the engine's actual efficiency where it matters is 0.32–0.34. This makes the headline `v_ce_equiv` value 1 L of battery‑sourced energy as if it had to be regenerated by a **0.25‑efficient** engine. It is applied identically to SAC and ECMS, so it does not bias the *comparison* — but it does mean the whole project is optimising against a metric whose electric‑to‑fuel exchange rate is ~35 % more punitive than the real marginal engine. Worth stating explicitly. | **U** (metric definition, not a bug; identical for all controllers) |
| E10 | `_K_CS` cold‑start = **1.15** | All fuel (and thus the fuel term of the reward, via `K_FUEL_L_PER_KG = _K_CS/_RHO_FUEL`) is inflated 15 %. `VERIFIED_FACTS` §A lists it as "confirmed 1.15" but the `ems_env.py` docstring and `README` historically said 1.0. Present in `powertrain.py:744`. Applied to all controllers. | **CC** (consistent) — verify once more against Simulink |
| E11 | Action → `u` → torque map (`_action_to_torques`) | Linear split `t_em = u·T`, then motor‑envelope clamp, then SoC hard masks, then engine‑over‑torque shift‑to‑motor. **No "electric snap", no forced‑regen override** (removed in a prior fix — the docstring documents a 0.31 L/100km interface artefact that used to handicap the agent). Braking: agent owns `u`, only the envelope clamps. | **CC** |
| E12 | `modeaware_gated` map | Monotone bijection [−1,1]→[U_MIN,U_MAX]; reachable `u` set identical to linear (53 tests). OFF gets fixed 40 % of the range where `_off_reachable`, linear elsewhere. **Compresses the ASSIST / engine‑load sub‑range** (Phase 7 §9) — a genuine minor contributor to the soft‑engine operating point. | **LP** (works as designed but has a documented side effect on engine‑load resolution) |
| E13 | Feasibility masks | Motor envelope `T_EM_max(w) − |θ_EM·dw| − ε`; SoC hard limits 0.05/0.95; engine envelope shift. 0 violations in all validated RL rollouts. **Identical masks used in `ecms.py._feasible_u_grid`** ⇒ benchmark‑fair. | **CC** |
| E14 | Timestep / integration | dt = 1 s throughout; trapezoidal fuel integration; backward‑difference `dv`; trapezoidal `x_tot` from `v_a`. Matches `evaluate_advanced.py` to 1e‑9 (`test_ems_env.py`). | **CC** |
| E15 | `enable_fast_interpolation()` monkey‑patch | Replaces scipy `RegularGridInterpolator` with bilinear; verified equal to <1e‑12 in‑bounds; powertrain clamps all lookups so no extrapolation. **Global module mutation** (`pt._interp2d_linear`) — harmless here, but any process that imports `ecms.py` after `ems_env.enable_fast_interpolation()` gets the patched interpolator too. Both use the same clamped path, so numerically identical. | **PC** |
| E16 | Standstill handling | `T==0 or w≤0` → `(0,0,0,"stop")`; engine contributes 0 (true standstill, `w_gear==0`, not idle). | **CC** |
| E17 | Driving cycle trailing row | NEDC 1221 samples (1220 real + 1 trailing `v=0,gear=0`), required to match MATLAB. `done` fires at `t ≥ length−1`. | **CC** |
| E18 | `future_speeds` lookahead | Causal: next `horizon` prescribed speeds, clamped to final speed at cycle end. Replaces the absolute `progress` scalar when `lookahead>0`. | **CC** |

### 4.2 ECMS implementation

| # | Item | Finding | Class |
|---|---|---|---|
| M1 | Hamiltonian `H(u) = P_fuel(u) + λ·P_batt(u)` | Computed with the project's own `combustion_engine` / `electric_motor` blocks — same plant, same wiring as `evaluate_advanced.py`. Apples‑to‑apples. | **CC** |
| M2 | `u`‑grid | `np.linspace(U_MIN=−0.85, U_MAX=1.0, 81)`, filtered by the **same feasibility masks the env uses**. Resolution = 0.023 in `u` ≈ 2–4 Nm of engine torque at typical demand. | **CC** (see sensitivity §5) |
| M3 | Braking | `_max_regen_u` — deterministic maximum feasible regen, identical to the env's forced policy for *all* controllers. No decision, no advantage. | **CC** |
| M4 | Closed‑loop costate | `λ_eff = λ₀ + k_fb·(0.5 − SoC)`, `k_fb = 8.0`. | **CC** (this is what the env's `k_fb` mirrors) |
| M5 | **λ₀ selection = whole‑cycle bisection** (`tune_lambda`) | `λ₀` is chosen by *repeatedly simulating the entire cycle* and bisecting until `SoC_end ≈ 50 %`. **This uses whole‑cycle outcome information.** A causal online controller cannot do this; it would have to adapt λ online. This is the core benchmark‑fairness caveat (§5, §26). | **LP** (not a bug — it is what ECMS is — but it is an information advantage over SAC that must be quantified, not ignored) |
| M6 | Tie‑breaking | First `u` on the ascending grid that achieves `min H` wins (`if H < best_H`). Slight bias toward lower `u` (more charging) on ties. Minor. | **PC** |
| M7 | Engine ON/OFF in ECMS | Emerges from the grid: `u` s.t. `(1−u)·T ≤ 5 Nm`. Same mechanism as the env. Idle 8 kW quantum: ECMS *will* pick OFF over an idling engine because `H(OFF)` excludes the 8 kW. | **CC** |
| M8 | Terminal SoC | ECMS run reports `v_ce_equiv` which *does* charge for residual net battery use (EFC saturation). Reported runs land `SoC_end` within 0.36 pp of 50 %. Fair. | **CC** |
| M9 | `ECMS_TARGET` constants (3.1887 / 2.8097) | Hard‑coded in `ecms.py` and `train_sac.py`; docstring says "reproduce with `python -m src.baselines.ecms --cycle X`". Not re‑verified in this audit (no execution), but the script is deterministic and the code path is clean. | **PC** — **recommend one reproduction run** |

### 4.3 Reward

| # | Item | Finding | Class |
|---|---|---|---|
| R1 | `r_t = −100·(fuel_liters + eq_factor_eff·elec_liters) − λ_soc·excess² − λ_soc_lin·excess` | `excess = max(|SoC−0.5| − 0.10, 0)`. At k_fb=0 the economic part telescopes *exactly* to −v_ce_equiv (verified `test_ems_env.py`). | **CC** (as an objective) |
| R2 | `eq_factor_eff = eq_factor + k_fb·(0.5 − soc_before)` | k_fb=2.5 (env liter units) = **12.08 ECMS λ‑units of slope** vs ECMS's own 8.0. The *training* costate feedback is **1.5× steeper** than the proven‑optimal one. Phase 5 §F: this makes battery "nearly free" above 50 % SoC and progressively suppresses OFF as SoC rises. | **LP** — patched (γ×k_fb interaction) but not resolved; the slope mismatch is a live reward‑shaping error |
| R3 | SoC deadband penalty | `SOC_DEADBAND = 0.10`. Measured contribution to the reward: **0.0 %** (0/1220 steps active on NEDC, 0/1876 on FTP75). Effectively dead code at the current operating trajectory. | **U** — inert; not harmful, not helping |
| R4 | Terminal penalty (`TERM_W_LIN=50`, `TERM_W_QUAD=800`) | At γ=0.20 the discount 10 steps out is 1e‑7. Measured reach: **0.77 % of episode reward**. The terminal charge‑sustaining signal is **structurally invisible** to the critic — the *entire* SoC regulation burden falls on the per‑step `k_fb` term. This is a direct consequence of the γ=0.20 pivot and is why Phase 4/5 needed `k_fb` at all. | **LP** — the terminal reward is real code doing nothing at γ=0.20 |
| R5 | `REWARD_SCALE = 100` | 1 equiv‑liter → 100 reward units. Per‑step reward ≈ −0.2. TD errors ≈ 0.1. Twin‑Q ≈ −0.3. The *advantage* between adjacent actions (OFF vs ASSIST) is ≈ 0.005–0.05 — **3–20 % of |Q|**, and the critic RMS fit error is ≈ 24 % of |Q| (E7). The learning problem is being asked to resolve an advantage well below the critic's own noise floor. This scale analysis is *why CQL blew up* (Phase 9: the CQL(H) term ≈ ln 30 ≈ 3.4 dwarfed the TD loss ≈ 5e‑3 by ~700×). | **LP** — reward/advantage conditioning is poor; not a bug, but it makes value‑based RL structurally weak here |
| R6 | `eq_factor_eff` sign inversion | `eq_factor_eff` crosses 0 and goes negative (pays to discharge) at SoC 66.4 % (NEDC) / 80.1 % (FTP75). Measured exploited: **0 steps** (max SoC ≈ 50–56 %). Latent, never triggered. | **U** — latent defect, recommend a positive‑floor clip as hygiene (EXP‑E in the registry) |
| R7 | Fuel term carries `_K_CS = 1.15` | The reward's fuel term is 15 % heavier than raw tank‑to‑wheel. The `4.8309` unit conversion already folds this in, so `eq_factor` is internally consistent — but the fuel:battery exchange rate the agent optimises is a cold‑start‑inflated one. | **PC** (consistent, worth a sentence in any writeup) |

### 4.4 SAC / training config

| # | Item | Finding | Class |
|---|---|---|---|
| T1 | Implementation | Verified vs SB3 2.9.0 (Phase 2/3). `NStepSAC` formula‑identical. | **CC** |
| T2 | **Training budget = 150k steps** | ≈ 123 NEDC / 100 FTP75 episodes. **Every "closed" conclusion (γ, k_fb, actor class, critic) was measured at this budget.** Literature routinely uses 1–5M steps or 2000+ episodes. This is the single biggest methodological limitation in the project. | **LP** |
| T3 | γ = 0.20 | Chosen from a mostly‑single‑seed sweep; high‑γ arms confounded by the diverging‑critic baseline. 1.2 s value horizon. See §6/§8/§21. | **LP** |
| T4 | `gradient_steps = 16`, `train_freq = 64`, `buffer 300k` (SB3 default 1M) | The gradient‑steps fix (64→16) genuinely stabilised the critic (VERIFIED_FACTS §E). Buffer 300k vs 1M is untested (EXP‑F). | **PC / U** |
| T5 | `target_entropy = auto = −1.0` | Phase 2 §16: −2 marginally better on ASSIST but indistinguishable at n=3. Phase 5/6 flagged entropy as "the next lever" repeatedly; it was **never actually run as a 3‑seed A/B** on the current config (EXP‑C, still "AUTHORISED‑AS‑NEXT, NOT STARTED"). | **U** — a genuine untested lever, though low expected value given Phase 8C |
| T6 | `n_step = 1` | At γ=0.20 n‑step is meaningless; correct. | **CC** |
| T7 | Observation (15 + lookahead dims) | Contains SoC, SoC‑error, w/dw/T_MGB, dT_MGB, v, v_next, 5‑step speed preview, gear one‑hot. **Does NOT contain:** previous action, previous mode, engine ON/OFF history, time/distance remaining (the preview *replaced* the progress scalar), battery‑power history, recent‑regen indicator, a SoC *target schedule*. Phase 2 §21 found the two dead channels (`v_next≡fut_v1`, `gear_oh6≡0`) but removing them hurt (input‑scaling perturbation). **A state‑ablation adding demand history / longer preview / SoC‑schedule has never been run.** | **U** — see §7 |
| T8 | Checkpoint rule | zero‑violations → charge‑sustaining → min V_CE. Training reward never used. Sound. | **CC** |
| T9 | Best checkpoint always in first half of training (old config) | "Train longer degrades" was measured at γ=0.9999 / gradient_steps=64 (diverging critic). Phase 8C explicitly notes the *current* config trains "stable and progressive, no early‑peak‑collapse." So the anti‑long‑training result **does not transfer** to the current config and long training has never been run on it. | **LP** (a stale conclusion still being cited) |

### 4.5 Summary of the audit

* **No critical bug found** in the plant, the env↔plant wiring, the ECMS
  Hamiltonian, or the SAC implementation. The physics is sound and
  benchmark‑fair on the *masks and wiring*.
* **The metric** (`v_ce_equiv` via a fixed‑efficiency EFC block with η_CE=0.25)
  is a defensible‑but‑punitive electric‑to‑fuel exchange rate, applied
  identically to all controllers — not a bug, but it is what "the gap" is
  measured against and it inflates the apparent cost of any electric‑heavy
  strategy.
* **Likely‑problematic, all in the RL layer:** the 150k budget (T2), the
  over‑steep training costate slope (R2), the invisible terminal reward at
  γ=0.20 (R4), the poor advantage/noise conditioning (R5), the
  engine‑load‑range compression of the gated map (E12), the stale "don't train
  longer" conclusion (T9), and an untested state representation (T7).
* **One instrumentation artefact:** Phase‑9 FTP75 cycle‑mean ECMS BSFC/η are
  divide‑by‑near‑zero garbage (§3.3). Per‑band numbers are fine.

---

## 5. ECMS AUDIT — IS THE BENCHMARK FAIR AND CORRECTLY IMPLEMENTED?

### 5.1 Correctness

`src/baselines/ecms.py` re‑uses the validated plant blocks through the same
per‑step wiring as `evaluate_advanced.py`; the only difference from the
rule‑based evaluator is the controller (grid‑search Hamiltonian min instead of
if/else `u`). The feasible `u`‑grid uses the **same masks** the env applies.
The reported `v_ce_equiv` charges for residual battery use. **The
implementation is sound and the numbers are directly comparable to SAC's.**
(One recommendation: run `python -m src.baselines.ecms --cycle {NEDC,FTP75}`
once to re‑confirm 3.1887 / 2.8097 against the hard‑coded constants — this
audit did not execute code.)

### 5.2 Is ECMS a *true* benchmark, a *tuned* controller, a *local optimum*, or
does it exploit unavailable information?

**It is a cycle‑specifically tuned local optimiser with a whole‑cycle
information advantage over SAC.** Evidence:

1. **λ₀ is bisection‑tuned on the whole cycle** (`tune_lambda` simulates the
   entire NEDC/FTP75 repeatedly, bisecting λ₀ until `SoC_end ≈ 50 %`). This is
   an offline, non‑causal calibration. A real online controller must *estimate*
   λ; that is what A‑ECMS and the `k_fb` feedback do, imperfectly.
2. **ECMS re‑solves the exact instantaneous Hamiltonian with a perfect plant
   model at every step** over an 81‑point grid. Zero function‑approximation
   error, zero exploration cost, zero credit‑assignment problem. A 150k‑step
   SAC critic is a smooth approximator that cannot match pointwise BSFC
   precision (Phase 8 "E", the irreducible floor).
3. **Constant‑λ ECMS sits a hair *above* the DP optimum** (the docstring says
   so). So 3.1887 / 2.8097 are a *tight, achievable reference*, not a lower
   bound — a preview‑equipped causal policy could in principle edge past them.
4. **Partial mitigation:** the SAC agent's *base* `eq_factor` **is** set to
   `λ₀/4.8309` — i.e. SAC is *handed* ECMS's cycle‑tuned λ₀ as its price
   anchor. So the "SAC doesn't know λ₀" gap is only the `k_fb` *slope*
   difference (2.5 vs 1.656 env units) plus the online‑adaptation noise, not
   the whole scalar.

### 5.3 ECMS sensitivity analysis — **NOT YET RUN; SPECIFIED HERE**

The mandate asks for ECMS performance vs {λ₀, SoC correction, SoC target, SoC
tolerance, `u`‑grid resolution, engine‑torque resolution, ON/OFF penalty,
battery EF formulation}. This is cheap (each point = one full‑cycle
deterministic run) and should be **the first concrete task of Phase 10**
because it bounds how much of "the gap" is real:

| Sweep | Range | What it tells us |
|---|---|---|
| λ₀ around the tuned value | ±20 % | Sensitivity of ECMS fuel to the whole‑cycle calibration. If ECMS fuel jumps 3–5 % for a 10 % λ₀ error, then a causal controller that can't tune λ₀ perfectly *cannot* reach 3.1887 and the "true" SAC target is higher. |
| `u`‑grid resolution | 41 / 81 / 161 / 321 | Is 3.1887 grid‑limited? If 321 pts gives 3.15, the reported ECMS is itself ~1 % suboptimal and the real DP gap is larger. |
| `k_fb` (SoC‑feedback slope) | 0 / 4 / 8 / 16 | 0 cannot charge‑sustain (documented). Confirms the feedback is load‑bearing and shows how sharp the SoC‑vs‑λ cliff is. |
| SoC tolerance for "charge‑sustaining" | 0.5 % / 2 % | Does relaxing CS to ±2 % (the number SAC is judged on) let ECMS report a lower fuel? If so the comparison is not tolerance‑matched. |
| Idle‑8 kW ON/OFF treatment | with / without the quantum in `H` | Quantifies how much of ECMS's extra OFF is "avoid the idle quantum". |

**Deliverable:** a table `ECMS(λ₀ ± ε, grid, k_fb, tol)` so the report can say
"ECMS is a *local* optimum with sensitivity S; a causal controller's best
achievable is ECMS + S; DP would be ECMS − G". Until this exists, **the split
between irreducible and addressable gap is UNIDENTIFIED**.

### 5.4 Verdict

ECMS is **correctly implemented and benchmark‑fair on the physics**, but it is
a **cycle‑tuned, whole‑cycle‑informed local optimiser**, not an oracle. Beating
it outright is not a fair primary target for a *general* causal controller;
matching it on the *same cycle it is tuned on* (which is also the cycle SAC
trains on) is fair. The rule‑based benchmark has a small, measured
authority advantage on NEDC (+0.074 L/100km / +2.1 %, it deep‑discharges to
SoC 0.61 %). Use **3.5792 (authority‑equal RB)** and **3.1887 (ECMS)** as the
NEDC reference pair; **3.2318 / 2.8097** for FTP75.

---

## 6. PHYSICAL ENERGY‑FLOW DIAGNOSIS — WHERE IS THE FUEL ACTUALLY LOST?

Combining Phase 7 `ecms_gap` and Phase 9 `engine_physics` (both NEDC seed 0,
demand‑aligned exact), against the mandate's A–I menu:

**Mechanical demand is identical** (demand alignment `max|ΔT| = 0`). So the gap
is entirely in how the identical wheel work is *sourced*.

### Energy balance, NEDC, SAC vs ECMS (per Phase 9 + §3)

| Quantity | SAC | ECMS | Δ |
|---|---|---|---|
| Engine‑on steps | 376 | 260 | **+116 (SAC runs the engine 45 % more often)** |
| Mean engine T_CE when on | 55 Nm | 79 Nm | **−24 Nm (SAC runs it softer)** |
| Mean BSFC when on | 290 g/kWh | 255 g/kWh | **+35 (SAC ~14 % worse specific consumption)** |
| Mean engine efficiency | 0.324 | 0.352 | −0.028 |
| ΔSoC | +0.28 pp | +0.36 pp | ≈0 (both charge‑sustaining) |
| V_CE_equiv | 3.686 | 3.189 | **+0.498** |

### Attribution against the menu

* **B — SAC uses the engine in a less efficient region: YES, ≈20–40 % of the
  gap** (part‑load, worse BSFC). Confirmed by the BSFC map figure and the
  T_CE|on numbers. This is the "operating‑point" component.
* **A — SAC burns more fuel for the same engine work: partly, folds into B**
  (same total engine mech energy, worse BSFC ⇒ more fuel).
* **F — SAC starts/stops the engine inefficiently: partly.** SAC keeps the
  engine on 45 % more steps at part load. But **there is no start penalty in
  the plant** (§4 E4/§14), so this is not a *switching‑cost* effect — it is a
  *mode‑selection* effect (SAC should have shut the engine off and coasted on
  banked charge in ≈116 of those steps, as ECMS does). ≈55–75 % of the gap.
* **C — SAC uses battery energy at the wrong times: mildly, at 0–15 Nm.** The
  +0.141 L/100km in the 0–15 Nm band is +0.123 *electrical* — SAC goes EV at
  trivial load then pays to recharge. This is *timing*, and it is the one place
  the "myopic policy can't plan the bank/spend cycle" hypothesis bites
  measurably.
* **D — SAC fails to recover braking energy: NO.** REGEN is env‑forced maximum
  for all controllers; brake‑band ΔTotal ≈ +0.01.
* **E — SAC maintains the wrong SoC trajectory: NO.** ΔSoC matches ECMS;
  battery ledger ≈0.
* **G — SAC's reward valuation is wrong: NOT at the margin.** Effective battery
  price matches ECMS's (Phase 7). The `k_fb` slope is 1.5× steep (R2) but
  Phase 7 showed P(OFF) is flat vs `k_fb` — the *policy* doesn't respond to it.
* **H — ECMS exploits future‑cycle information: YES, but bounded.** λ₀ is
  whole‑cycle bisection‑tuned (§5). Magnitude unquantified until the §5.3
  sensitivity sweep runs. Likely a small fraction (SAC is handed λ₀ as its
  anchor).
* **I — combination: YES.** Best statement:
  **≈55–75 % engine ON/OFF mode‑selection + timing (SAC keeps the engine on at
  part load where ECMS coasts on banked charge or loads it hard); ≈20–40 %
  engine operating‑point / BSFC (when on, SAC runs it too soft); ≈5–10 %
  over‑EV at trivial load then recharge (0–15 Nm); ≈0 % battery ledger;
  unquantified single‑digit % ECMS whole‑cycle‑tuning advantage.**

**The physical root:** ECMS's strategy is "engine OFF or engine HARD, rarely
in between; bank the surplus." SAC's learned strategy is "engine ON, near
demand, most of the time." SAC's is a *locally* reasonable myopic response;
ECMS's is what a globally‑consistent costate produces. The fuel is lost in the
**part‑load middle** that ECMS avoids and SAC lives in.

---

## 7. RL DIAGNOSIS

Separating the RL failure modes the mandate lists.

### 7.1 Representation (policy class) — **NOT the binding cause** (Phase 8, solid)
The Q‑oracle (ideal actor on the trained critic) is *worse* than the real
actor and loses CS. A 2‑component mixture actor collapsed to unimodal and did
not improve fuel/CS. The unimodal‑Gaussian‑vs‑bimodal‑Q story is *real*
(actor mean ~1.5 action‑units from argmax‑Q at NEDC 30–35 Nm) but it is
downstream of the critic and downstream of coverage.

### 7.2 Value learning (critic) — **the proximate binding constraint, but it is
a *coverage* problem, not a *capacity* or *pessimism* problem**
* On‑distribution the critic is **not grossly wrong** (Phase 9: min‑Q ranks
  HIGH_EFF ≳ ECMS_NBHD ≳ LOW ≳ OFF, matching reward and next‑SoC).
* Off the on‑policy distribution it has a **mild low‑load argmax bias** that
  *compounds* over a cycle into SoC collapse when exploited greedily.
* Cause: **8–27 % replay support** for HIGH_EFF/ECMS_NBHD at 15–35 Nm vs
  48–54 % for LOW; twin‑Q disagreement 10× higher there. The critic can't rank
  what it hasn't seen.
* CQL (Phase 9) *cannot fix this* — pessimism penalises the policy's own OFF
  actions, the actor charges to escape, CS collapses. Confirmed at every α.
* Advantage/noise conditioning is poor (R5): the OFF‑vs‑ASSIST advantage is
  3–20 % of |Q|, critic fit error ~24 % of |Q|.

### 7.3 Temporal credit assignment — **prematurely closed; see §8**
γ=0.20 (1.2 s horizon). The terminal CS reward is 0.77 % of episode reward,
invisible. The project *asserts* the optimum is myopic; it *demonstrated only*
that at 150k steps on a diverging‑critic baseline, high γ was worse. A myopic
policy structurally cannot do anticipatory bank/spend, which is exactly the
0–15 Nm over‑EV failure (§6 C) and part of the mode‑timing failure. **Not
re‑tested under the stable config; not tested with a longer budget.**

### 7.4 Exploration / data distribution — **the deepest recurring cause**
The same exploration deadlock has now been found **twice**: Phase 4 for
engine‑OFF at 30–50 Nm (fixed by the gated map), Phase 9 for the efficient
high‑engine‑load region at 15–35 Nm (**not fixed** — never targeted). The
on‑policy distribution is self‑reinforcing: actor doesn't go there → no data →
critic can't value it → no gradient → actor doesn't go there. Phase 6 injected
OFF coverage where the reward/critic *already* preferred OFF (wrong region) and
it failed. **Targeted coverage of the efficient engine‑load region, and/or
offline data covering it, has never been run.**

### 7.5 State information — **UNTESTED and now in scope**
Phase 8/9 dismissed state sufficiency with "the reward already carries the
fuel consequence." That is a non‑sequitur under γ=0.20: a myopic critic cannot
use a downstream reward consequence; it needs the *state* to distinguish "good
time to bank" from "good time to spend." The state lacks: previous
action/mode, engine ON/OFF history, time/distance remaining, demand history,
recent‑regen indicator, a SoC target *schedule*. The 5‑step speed preview is
the only forward signal and was deliberately kept short for generalisation.
Cross‑cycle transfer failing CS 0/3 both ways (Phase 7) is consistent with a
policy that memorised a cycle‑specific SoC trajectory *because the state
doesn't carry enough to compute a transferable one*. **A state ablation
(base / +prev‑action / +demand‑history / +longer‑preview / +SoC‑schedule) has
never been run on the current config** — Phase 8 §21 lists it as "in scope
only after 8C and the critic route fail." **Both have now failed.**

### 7.6 Action parameterisation — **contributing, minor**
The gated map compresses the engine‑load sub‑range (Phase 7 §9), correlating
with the soft‑engine operating point. But the actor fails to reach even the
OFF share its own critic wants, so the map is not the primary limiter.

### 7.7 Planning / horizon — **never tried.** No MPC, no receding‑horizon
action search, no "critic as terminal value + short rollout." The greedy
1‑step Q‑oracle collapsed; a short physical rollout with a critic terminal
would be more robust and has diagnostic value (§11).

### 7.8 Is SAC itself the constraint? — **Not demonstrated, but the *value‑based
end‑to‑end framing* is a poor fit.**
The implementation is correct. But the problem, as reformulated, is "learn
ECMS's analytic pointwise optimiser by bootstrapped value iteration from
150k steps of self‑generated data that never covers the decisive region."
An analytic minimiser wrapped by a *small* learned quantity (the costate)
sidesteps every one of 7.1–7.7 at once. That is not "swap SAC for TD3" — it is
"change what the RL has to represent."

---

## 8. LITERATURE INVESTIGATION

Ten+ technically relevant works, with the *mechanism* responsible for
their result (not just the headline). Figures are as reported by the sources
retrieved; verify against the papers before quoting numbers.

| # | Work / thrust | Architecture | Mechanism responsible for the gain | Relevance to this project |
|---|---|---|---|---|
| L1 | **DRL‑ECMS: adaptive hierarchical ECMS via DRL** (IEEE, 2022) | Upper: DRL agent outputs the **equivalent factor** from (SoC, demand, speed stats); Lower: 1‑D search picks engine power given EF. | RL never touches the low‑level continuous optimisation — it learns only the *economic scalar*. The pointwise BSFC‑optimal split is guaranteed by the deterministic minimiser. | **This is Architecture C.** Directly addresses this project's two dominant gap components at once. |
| L2 | **Adaptive hierarchical EMS combining heuristic domain knowledge + DRL** (IEEE T‑something, 2021) | Two‑layer: RL plans a reference (speed / power) using traffic + powertrain state; A‑ECMS allocates energy at the low level. | Domain‑knowledge structure (A‑ECMS) removes the hardest part of the search from the RL; RL supplies the adaptivity the fixed A‑ECMS lacks. | Confirms the pattern: RL for adaptation, analytic layer for the instantaneous optimum. |
| L3 | **Comparative study of 13 DRL EMS methods for an HEV** (Energy, 2022) | Unified framework, SAC/PPO/TD3/DDPG/DQN‑family + reward variants. | Every DRL EMS stayed **within 7.6 % of DP**; A‑ECMS was 8.6 % above DP; power‑following 10.3 %. The *reward design* (multi‑objective with a **dynamic SoC penalty derived from DP insight**) mattered more than the algorithm. | Says a well‑designed DRL EMS *can* beat A‑ECMS and approach DP — but via reward shaping informed by DP, and with far more training than 150k steps. |
| L4 | **Comparative study, SOC‑adaptive reward** (Applied Energy / eTransportation, 2023–24) | SAC + a multi‑objective reward with a **dynamic SoC penalty** that forces SoC_end → target regardless of initial SoC or trip length. | The dynamic SoC term is what makes the policy *charge‑sustaining and cycle‑transferable* — it removes the need for the value function to carry the long‑horizon SoC coupling. | This project's `k_fb` term is a cruder version of the same idea; the slope is mis‑set (R2) and the terminal term is dead (R4). A DP‑informed dynamic SoC reward is a cleaner replacement. |
| L5 | **SAC beats DP by 6.8 % on the training cycle; within 7.6 % on tests; beats A‑ECMS by 15.5 %** (range‑extender EV benchmark, 2025) | SAC, SOC‑adaptive multi‑objective reward, large training budget. | Careful reward + enough training + SOC‑adaptivity. Note "beats DP on the *training* cycle" = the same
cycle‑specific‑tuning caveat this project's ECMS has (§5). | Sets a realistic ceiling: a *well‑trained* SAC can match/exceed A‑ECMS. This project is not there because of budget + coverage + reward‑shape, not because SAC can't. |
| L6 | **PPO/SAC best, TD3/DDPG "smooth deterministic, moderate gains"** (comparative, 2023) | Same env, 4 algorithms. | Stochastic on‑policy / max‑entropy handle the exploration of the discrete‑ish ON/OFF structure better than deterministic off‑policy. | Argues *against* a naive TD3 swap for this project (matches Phase 2 §26: TD3 committed harder but lost CS). |
| L7 | **RL + experience augmentation for HEV EMS** (Applied Thermal Eng, 2025) | Off‑policy RL with an **augmented replay buffer** (synthetic / re‑labelled transitions covering under‑visited regions). | ~40–45 % faster convergence, ~18.9 % fuel reduction "without increasing battery power" — from *better data coverage*, not a better network. | Direct support for a **targeted‑coverage / experience‑augmentation** experiment (Phase 9 "Experiment B" generalised). |
| L8 | **Warm‑start / offline pre‑training for RL supervisory control** (arXiv 2010.14575) | Initialise the critic/policy from an offline dataset (rule‑based or ECMS rollouts) before online RL. | Cuts learning time drastically and lifts the floor — the online phase starts from a distribution that already covers the good operating region. | This project explicitly forbade benchmark‑seeded buffers ("guided"). A **non‑expert feasible** offline dataset (random/curriculum, not ECMS) is a legitimate middle path (§11). |
| L9 | **Imitation‑learning‑embedded DRL for FCHEV** (J. Cleaner Prod, 2024) | BC / imitation loss as an auxiliary term alongside the RL objective, warmed from a heuristic controller. | The imitation term keeps the policy near a known‑good manifold while RL refines; avoids the exploration collapse. | Even if pure imitation is off‑limits, an *auxiliary* term anchored to a *feasible‑diverse* (not optimal) controller is worth a diagnostic run. |
| L10 | **DRL + driving‑condition recognition (DCR)** (Energy, 2024; Applied Energy, 2024) | GRU speed predictor (~97 % DCR accuracy) feeding a condition embedding / predicted power to the RL state. | Adaptivity to cycle phase and near‑future demand — the policy conditions its strategy on *what's coming*, which is exactly what anticipatory bank/spend needs. | Direct support for the **state‑representation** lever (§7.5): add demand history + a short predicted‑power / cycle‑phase feature. |
| L11 | **A‑ECMS costate‑correction by RL** (from the DRL‑ECMS family) | RL outputs **Δλ around a nominal costate** given (SoC, current λ, demand, predicted power, predicted mean speed/accel). | Learning a *bounded correction* is a far easier and more stable regression than learning the whole control; the nominal analytic layer guarantees feasibility and pointwise optimality. | **Architecture G** — the lowest‑risk variant of the hybrid approach. |
| L12 | **Real‑time HEV EMS from naturalistic data + DRL, high generalisation** (Applied Energy, 2024) | DRL trained across many real trips; emphasis on transfer. | Broad training distribution ⇒ transferable policy. This project trains on one cycle and evaluates on it, then fails cross‑cycle 0/3 (Phase 7). | Confirms the cross‑cycle CS failure is a *distribution* problem, not an algorithm problem. |
| L13 | **MPC + RL hybrids / RL policy as MPC terminal value** (multiple, 2023–24) | Short‑horizon MPC around a learned value function or policy. | The rollout corrects the greedy‑1‑step error that made this project's Q‑oracle collapse; the learned terminal value supplies the long horizon cheaply. | Support for the **horizon** lever (§7.7, §11). |

**Synthesis of the literature vs this project:**

1. The strongest results that *match or beat A‑ECMS/DP* either **(a) learn the
   equivalent factor / costate with an analytic lower layer** (L1, L2, L11), or
   **(b) use a DP‑informed dynamic‑SoC reward + a large training budget +
   broad data** (L3, L4, L5), or **(c) fix the data distribution directly**
   (L7, L8, L12). None of them succeed by scaling actor capacity or adding
   critic pessimism on a small budget — exactly the two routes this project
   has exhausted (Phases 8–9).
2. **The mechanism this project is missing is not a better network — it is
   either removing the low‑level optimisation from the RL's job (a, most
   direct), or giving the RL the data and reward shape to learn it (b, c).**

Sources:
[DRL/ECMS comparative (13 methods)](https://www.sciencedirect.com/science/article/abs/pii/S0360544222033837) ·
[DRL EMS comparative (Energy Conv. Mgmt.)](https://www.sciencedirect.com/science/article/abs/pii/S0196890423007884) ·
[SOC‑adaptive DRL benchmark, REEV](https://www.sciencedirect.com/science/article/abs/pii/S2352152X25032657) ·
[DRL‑ECMS adaptive hierarchical](https://ieeexplore.ieee.org/document/9827234/) ·
[Adaptive hierarchical EMS, heuristic + DRL](https://ieeexplore.ieee.org/document/9635809/) ·
[Warm‑start methods for RL supervisory control](https://arxiv.org/pdf/2010.14575) ·
[RL with experience augmentation](https://www.sciencedirect.com/science/article/abs/pii/S1359431125011536) ·
[Imitation‑learning‑embedded DRL (FCHEV)](https://www.sciencedirect.com/science/article/abs/pii/S0959652624008072) ·
[Adaptive DRL EMS with driving‑condition recognition](https://www.sciencedirect.com/science/article/abs/pii/S0360544224038647) ·
[Real‑time HEV EMS, naturalistic data, high generalisation](https://www.sciencedirect.com/science/article/abs/pii/S0306261924017331) ·
[Progress & summary of RL for MPS‑EV EMS (survey)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10755288/) ·
[DDQN adaptive PHEV EMS](https://www.sciencedirect.com/science/article/abs/pii/S0360544224021765)

---

## 9. GLOBAL LEVER MATRIX

Status codes: **TR** tested & refuted · **TI** tested but inconclusive ·
**NT** never actually tested · **IX** tested under an invalid / confounded
experimental design.

| Lever | Status | Evidence | Remaining uncertainty | Expected impact | Cost | Risk | Priority |
|---|---|---|---|---|---|---|---|
| Actor capacity / multimodality | **TR** | Phase 8C mixture collapsed, fuel + CS worse; Q‑oracle (ideal actor) worse than actor | none — the actor is not the ceiling | ~0 | — | — | drop |
| Critic regularisation (CQL / pessimism) | **TR** | Phase 9 A: fails at α∈{0.01,0.05,1.0}, SoC runaway / 100+ viol, −213 % | none — pessimism destabilises the CS balance | negative | — | high | drop |
| Replay coverage of engine‑**OFF** at operating SoC | **TR** | Phase 6: 4.5 %→36.7 %, ΔQ moved 0.0005; fuel worse | none — reward/critic already agree on ASSIST there | ~0 | — | med | drop |
| Replay coverage of the efficient **high‑engine‑load** region | **NT** | Phase 9 designed "Experiment B", never ran it; HIGH_EFF support 8–27 % vs LOW 54 % | does closing *this* hole re‑order the argmax? | **med–high** | med | med | **HIGH** |
| Reward — equivalent‑factor *scale* | **TR** | Phase 2 unit fix (necessary); Phase 7 effective price matches ECMS | none | ~0 | — | — | done |
| Reward — `k_fb` *slope* (2.5 vs ECMS 1.656 env units) | **IX** | Phase 7 "flat plateau [2.0,3.0]" — but measured only at γ=0.20 where the terminal term is dead and only vs fuel, not vs the ECMS *strategy* | does ECMS‑matched slope + visible terminal term change the operating strategy? | low–med | low | low | med |
| Reward — DP‑informed dynamic‑SoC term (replace dead deadband + dead terminal) | **NT** | R3/R4: both current SoC mechanisms are inert at the operating trajectory; L4 shows this is the standard fix | could improve CS robustness + cross‑cycle transfer | med | low | low | med |
| Reward — explicit part‑load / low‑η penalty | **NT** | Phase 9 §16 "(H)"; 39 % of NEDC gap is part‑load BSFC | does a shaping term make the efficient region the gradient attractor? | med | med (metric distortion) | med | med |
| State — demand history / longer preview / SoC‑schedule / prev‑action | **NT** | Phase 8 §21 gated it behind "8C + critic fail"; both have now failed; cross‑cycle CS 0/3 | is the myopic critic starved of the *state* to time bank/spend? | **med–high** | low | low | **HIGH** |
| Temporal history in the critic (n‑step at higher γ) | **IX** | γ sweeps single‑seed, high‑γ arms on a diverging‑critic baseline | is "myopic optimal" real, or a 150k‑step / unstable‑critic artefact? | med | med | med | med |
| Value horizon γ (re‑sweep on stable config, 3 seeds, longer budget) | **IX** | E10 removed the n‑step confound but kept 150k budget + single seed on new arms | same as above | med | med | low | med |
| Preview / predicted power in the state | **NT** | lookahead=5 speeds only, deliberately short; L10 uses DCR + predicted power | does near‑future demand context enable anticipatory OFF? | med | low (still causal) | low | med |
| Equivalent‑factor learning (RL → λ, analytic lower layer) — **Architecture C** | **NT** | entire Phase 5–9 chain points here; L1/L2/L11 literature | how close to ECMS does "RL learns only λ(s)" get? | **HIGH** | med (new code, ~2 days) | low–med | **TOP** |
| RL → Δλ correction around a nominal costate — **Architecture G** | **NT** | L11; lowest‑variance version of the above | same | **HIGH** | med | **low** | **TOP (safest)** |
| RL → engine operating‑region + deterministic optimiser — **Architecture F** | **NT** | attacks operating‑point directly | how much of the 20–40 % op‑point share is recoverable | med–high | med | med | med |
| RL → battery/engine power split (vs torque split) | **NT** | reparam only; Phase 4 showed coordinate reparam ≠ fix | probably low given Phase 4 | low | low | low | low |
| Horizon / short‑MPC around the critic (critic as terminal value) | **NT** | Q‑oracle collapsed *because* greedy‑1‑step; a rollout would be more robust; L13 | does a 3–10 step rollout beat greedy‑Q and the actor? | med | med | med | med |
| Offline warm‑start from a **non‑expert feasible** dataset | **NT** | L8; project only forbade *benchmark‑seeded* buffers | does starting from a coverage‑complete distribution lift the floor? | med–high | med | low–med | **HIGH** |
| Offline RL (CQL/IQL) on a feasible‑diverse dataset + online fine‑tune | **NT** | distinct from Phase 9's *online* CQL that failed | can offline value learning on good coverage beat on‑policy? | med | med–high | med | med |
| Longer training on the current *stable* config | **IX** | "don't train longer" measured on the diverging‑critic baseline only (T9) | does 500k–2M steps at γ=0.20 stable config still plateau? | low–med | low (compute) | low | med (cheap sanity) |
| Exploration schedule (OU / param‑noise / ensemble‑disagreement) | **NT** | never varied from SAC's default entropy exploration | does uncertainty‑driven exploration find the efficient region? | med | low | low | med |
| Curriculum / mixed‑cycle training | **NT** | single‑cycle training; cross‑cycle CS 0/3 | does mixed‑cycle training produce a transferable costate law? | med | low | low | med |
| Target entropy A/B (EXP‑C) | **NT** | "AUTHORISED‑AS‑NEXT" since Phase 6, never run; Phase 8C makes it low‑value | probably little given the mixture result | low | low | low | low |
| Buffer size 300k → 1M (EXP‑F) | **NT** | SB3 default is 1M | late‑run stability only | low | low | low | low |
| Normalisation / numerical conditioning | **TI** | Phase 2 §21: removing dead channels *hurt* (input‑scaling sensitivity) — a red flag that the net is scale‑fragile | is obs/reward normalisation (VecNormalize / running stats) worth a run? | low–med | low | low | med |
| Algorithm swap (TD3/DDPG/MPO/distributional/ensemble) | **IX** | only TD3, one seed, Phase 2 §26 — committed harder but lost CS | would a distributional / ensemble critic help the coverage‑uncertainty problem? | low–med | med | med | low (gated) |
| ECMS implementation correctness | **PC** | code audit §5.1 clean; not re‑executed | reproduce 3.1887/2.8097 once | — | trivial | — | do it |
| ECMS sensitivity (λ₀, grid, k_fb, tol) — bounds "the gap" | **NT** | §5.3 | how much of the gap is irreducible vs addressable | **diagnostic** | trivial (no training) | none | **DO FIRST** |
| DP reference | **NT** | never built | the actual optimum | **diagnostic** | med (1D‑state DP is tractable here) | none | **HIGH** |
| Engine map / battery model correctness | **CC/PC** | §4; no bug | E7 "−C−" derivation; E10 K_CS re‑check | ~0 | trivial | none | low |

---

## 10. TOP 10 REMAINING HYPOTHESES (ranked)

Ranked by *evidence strength × expected impact × (1/experimental cost)*, with
confidence.

| Rank | Hypothesis | Evidence | Expected impact on the gap | Confidence | Cost |
|---|---|---|---|---|---|
| **H1** | **The RL is being asked to learn a pointwise optimisation ECMS solves analytically. Moving to RL→λ(s) (or Δλ) + a deterministic Hamiltonian minimiser removes the actor‑displacement AND the operating‑point error at once.** | Phase 7 CASE D (actor on the wrong lobe); Phase 8 Q‑oracle ceiling (critic can't be exploited); Phase 9 (defect is coverage of a region a pointwise minimiser reaches by construction); L1/L2/L11 literature | **50–90 % of the gap** (both dominant components) | **High** | Med (~2 days: minimiser wrapper + λ‑headed SAC) |
| **H2** | **The efficient high‑engine‑load region is a data‑starved hole (a second exploration deadlock). Targeted exploration + experience augmentation there re‑orders the critic argmax.** | Phase 9: HIGH_EFF/ECMS_NBHD replay support 8–27 % vs LOW 54 %, twin‑Q disagreement 10×; Phase 4 (same deadlock structure for OFF, fixable); L7 | **20–40 %** (the operating‑point component, possibly some mode‑timing) | Med‑High | Low‑Med (exploration schedule flag, 3 seeds) |
| **H3** | **The state lacks the features to time bank/spend under a myopic critic (prev‑action/mode, demand history, SoC‑schedule, longer preview). Cross‑cycle CS 0/3 is the tell.** | Phase 7 cross‑cycle failure; γ=0.20 makes downstream reward unusable by the critic; L10/L12; never tested | **10–30 %** (0–15 Nm over‑EV + some timing) + big transfer gain | Med | Low (obs ablation, 3 seeds) |
| **H4** | **"The optimum is myopic" is a 150k‑step / diverging‑critic artefact. A stable‑config γ re‑sweep + longer budget shifts the optimum upward and enables anticipation.** | γ arms single‑seed; high‑γ on gradient_steps=64; Phase 8C says current config trains stably (contradicts "don't train longer") | **10–25 %** | Low‑Med | Med (γ×budget grid, 3 seeds) |
| **H5** | **Offline warm‑start from a non‑expert *feasible‑diverse* dataset lifts the floor by starting online RL from coverage‑complete data.** | L8; Phase 9 CQL failed *online* but offline value learning on good coverage is different; project only banned *benchmark*‑seeded buffers | **20–40 %** | Med | Med (dataset gen + pretrain + finetune) |
| **H6** | **The `k_fb` slope (1.5× ECMS) + dead terminal reward together mis‑shape the operating strategy; ECMS‑matched slope + a live dynamic‑SoC term (DP‑informed) + a modestly higher γ fixes it.** | R2/R4; Phase 5 §F ("battery nearly free above 50 % SoC"); L3/L4 | **10–20 %** + CS robustness + transfer | Low‑Med | Med (coupled reward change, 3 seeds) |
| **H7** | **An explicit part‑load / low‑η penalty makes the efficient region the reward‑gradient attractor (Phase 9 "H").** | Phase 9 §10 (39 % NEDC part‑load BSFC); Phase 8 §16 (reward's own argmax already wants harder load but the policy never gets there) | **15–30 %** (operating‑point) | Med | Med (reward term, metric‑distortion risk) |
| **H8** | **Short receding‑horizon action search over the trained critic (critic as terminal value, 3–10 step physical rollout) beats both the greedy Q‑oracle and the actor.** | Q‑oracle collapsed *because* greedy‑1‑step on an imperfect critic; L13 | **10–30 %** (and a strong diagnostic on "is it the critic value or the 1‑step exploitation?") | Med | Med (rollout controller, no training) |
| **H9** | **ECMS's whole‑cycle λ₀ tuning is a non‑trivial information advantage; the causal‑controller‑achievable target is ECMS + S (S from the §5.3 sensitivity sweep), and part of "the gap" is not addressable.** | §5; `tune_lambda` is whole‑cycle bisection; constant‑λ ECMS is "a hair above DP" | reframes the target (could be 20–40 % of the *nominal* gap being irreducible) | Med | Trivial (no training) |
| **H10** | **The observation net is scale‑fragile (removing dead channels *hurt*). Proper obs/reward normalisation (running‑stats) improves critic conditioning and the argmax bias.** | Phase 2 §21; R5 (advantage below noise floor) | **5–15 %** | Low‑Med | Low (VecNormalize‑style wrapper, 3 seeds) |

---

## 11. TOP 3 EXPERIMENTS (full specification)

All obey the mandate §28 rules: one major hypothesis at a time, ≥3 seeds,
frozen evaluation (`results/evaluate_policy.py`), identical cycle/evaluator, no
benchmark action leakage (except where the experiment explicitly studies
offline learning), no unrelated hyperparameter changes, report mean ± SD +
per‑seed + CS + ΔSoC + regional fuel + engine operating point + action
distribution + training stability, and — for any failure — *why* it failed.

**Also run first (no training, hours not days):** the §5.3 ECMS sensitivity
sweep + one ECMS reproduction run + a 1‑D‑state DP solver (SoC is the only
continuous state; a 100‑bin SoC grid × 1221 steps × 81 `u` values is a few
seconds). These bound the addressable gap and are prerequisites for
interpreting everything below.

---

### EXPERIMENT 1 (THE SINGLE BEST — see §12) — Architecture G: RL learns Δλ around a nominal costate; a deterministic Hamiltonian minimiser produces `u`

**Hypothesis (H1, safest variant).** The residual gap is dominated by (a) the
actor sitting on the wrong lobe of a bimodal action‑value surface and (b)
soft‑engine operating points — both of which vanish if the instantaneous `u`
is chosen by a deterministic minimiser and the RL only supplies a *bounded
economic correction* to the costate.

**Exact code change.**
1. New module `src/agents/hamiltonian_layer.py`:
   `best_u(w, dw, T, soc, lam) -> u` = the *exact* body of
   `ecms._hamiltonian_best_u` (reuse it; it already calls the validated plant
   blocks and the env's feasibility masks). Grid 81 points (match ECMS).
2. New env wrapper `EMSEnvLambda(EMSEnv)` (RL layer only, no plant change):
   * `action_space = Box(-1, 1, (1,))` interpreted as
     `dlam = a * DLAM_MAX`, `DLAM_MAX = 1.0` (env liter‑units; ≈ 4.8 ECMS
     units — generous).
   * per step: `lam_eff = LAM0 + K_FB*(0.5 - soc) + dlam`, with
     `LAM0 = eq_factor` (0.2717 NEDC / 0.4981 FTP75) and `K_FB` fixed at the
     **ECMS‑matched 1.656** (not 2.5 — the minimiser + correction removes the
     need for the over‑steep patch).
   * `u = hamiltonian_layer.best_u(...)`; then the *existing*
     `_action_to_torques` feasibility path (so masks/clamps are byte‑identical).
   * reward, obs, `k_fb` telescoping, evaluator — **unchanged**.
3. `train_sac.py --arch lambda-correction` flag selecting the wrapper.

**Variables changed:** the action *semantics* (torque‑split → Δcostate) and
the addition of the deterministic minimiser. `K_FB` 2.5 → 1.656 (justified:
the minimiser makes the steep patch unnecessary; document as part of the
architecture, not a free hyperparameter).

**Variables frozen:** plant, powertrain blocks, feasibility masks, reward
equation, `eq_factor`/`LAM0`, obs (15 + lookahead 5), γ 0.20, n_step 1,
lr 3e‑4, batch 512, buffer 300k, tau 0.005, gradient_steps 16, net [256,256],
target_entropy auto, evaluator, ECMS, rule‑based, 3 seeds {0,1,2}.

**Training budget:** 150k steps first (smoke, `results/readiness_gate.py`),
then 500k for the survivor.

**No leakage:** the minimiser uses the *plant model*, not any ECMS/benchmark
*action* or *trajectory*. `LAM0` is the same cycle‑anchor SAC already gets. No
imitation loss, no seeded buffer.

**Expected outcome (positive):** NEDC V_CE **3.30–3.45** (closes 50–75 % of the
+0.578 gap), 3/3 CS, engine T_CE|on rises toward 70–80 Nm, engine‑on steps
fall toward ~280, BSFC toward ~260 g/kWh, OFF% at 30–35 Nm rises from 4 % toward
40 %+. The Δλ head should converge to a small‑magnitude, smooth function of SoC
and demand. FTP75 V_CE **2.85–2.95**.

**Interpretation of a positive result:** confirms H1 — the end‑to‑end
value‑based framing was the limiter; the RL only ever needed to learn the
1‑D economic scalar. This becomes the project's controller; remaining gap to
ECMS is the whole‑cycle λ₀‑tuning advantage (§5/H9) + minimiser grid
resolution, both quantifiable.

**Interpretation of a negative result** (Δλ head learns ≈0 and fuel ≈ pure
ECMS‑with‑k_fb‑1.656, i.e. ~3.35 but not better; OR it destabilises CS):
* If it lands at pure‑ECMS‑1.656 fuel with the head at ≈0 → the RL adds
  nothing but the analytic layer *itself* closes most of the gap ⇒ the gap was
  the pointwise optimisation, not adaptivity; ship the analytic layer, drop
  the RL. Still a win.
* If CS destabilises → the Δλ head is over‑powered; shrink `DLAM_MAX` to 0.3
  and re‑run. If it still destabilises, the reward's SoC handling (R4) is the
  problem and Experiment 3 becomes primary.
* If fuel is *worse* than the CONTROL → the minimiser grid (81 pts) or the
  idle‑8 kW handling interacts badly with the reward's `k_fb` telescoping;
  inspect `lam_eff` distribution vs ECMS's.

---

### EXPERIMENT 2 — Targeted efficient‑region coverage + experience augmentation (H2)

**Hypothesis.** The critic's low‑load argmax bias at 15–35 Nm is a coverage
hole in the efficient engine‑load region (HIGH_EFF/ECMS_NBHD support 8–27 %).
Guaranteeing feasible coverage there re‑orders the per‑state argmax and lets a
plain actor track it.

**Exact code change.** Extend `src/agents/targeted_exploration.py`:
when `15 ≤ T_MGB < 50 Nm` **and** the engine is commanded ON **and** a higher
feasible engine load exists, with probability `p = 0.25` replace the sampled
action with a **uniform draw from the feasible high‑engine‑load interval**
(`u` such that `T_CE ∈ [0.9·T_CE,ECMS_proxy, 0.9·T_CE_max]`, where the proxy is
`1.3× demand` — a *feasibility* bracket, NOT an ECMS action). Training‑time
only; `predict()` untouched (evaluation‑safe by construction, as in Phase 6).

**Variables changed:** the exploration distribution (one flag). **Frozen:**
everything else at CONTROL values.

**Budget:** 150k × 3 seeds, then 500k for the survivor.
**Leakage:** none — the injected action is a *uniform draw over a
feasibility‑defined interval*; it encodes "a harder engine load is possible
here", never what a good controller would pick (identical safeguard to
Phase 6).

**Expected (positive):** HIGH_EFF/ECMS_NBHD replay support 8–27 % → 40 %+;
twin‑Q disagreement there 0.06 → <0.02; per‑state argmax‑Q shifts from
{OFF,LOW} toward ECMS_NBHD; NEDC V_CE **3.55–3.70** (closes the ~39 %
operating‑point component + some timing), 3/3 CS.

**Interpretation positive:** the deadlock was the mechanism all along and the
gated‑map fix in Phase 4 only solved half of it; ship the exploration schedule
(and consider annealing it out). Combine with Experiment 1 if both land.

**Interpretation negative** (support rises, argmax/fuel don't move — the
Phase‑6 outcome again): the critic *cannot* re‑order even with data ⇒ the
advantage genuinely is below the critic's noise floor (R5) at this reward
scale/budget ⇒ escalate to Experiment 1 (remove the need to learn it) and/or a
2–5× longer budget + reward‑scale increase. Document that two coverage
interventions in different regions both failed to move the argmax ⇒ strong
evidence the value‑based route is exhausted.

---

### EXPERIMENT 3 — State sufficiency + reward SoC‑handling, cleanly separated (H3 + H6)

Run as **two** single‑variable arms (3 seeds each), not stacked.

**3a — state ablation.** Add to the observation, one variant at a time:
(i) previous executed `u` and previous mode one‑hot; (ii) a 10‑step demand
(T_MGB) history summary (mean, slope, min); (iii) lookahead 5 → 20 speeds;
(iv) `(time_remaining, distance_remaining)` normalised (restored alongside the
preview, not replacing it); (v) a SoC *reference schedule* value
`soc_ref(t) = 0.5` (trivial here, but the *deviation‑from‑schedule* channel is
the hook a transferable costate needs).
**Frozen:** everything else at CONTROL. **Measure:** NEDC/FTP75 V_CE + CS +
**cross‑cycle transfer** (train NEDC → eval FTP75 and vice versa — the Phase 7
0/3 CS result is the baseline to beat).

**Expected (positive):** variant (ii) or (iii) drops the 0–15 Nm over‑EV term
(+0.14) and improves cross‑cycle CS from 0/3 toward 2–3/3. NEDC V_CE
**3.60–3.72**.

**Interpretation positive:** the myopic critic was state‑starved for
timing; a modest feature addition is a cheap partial fix and a transferability
fix.
**Interpretation negative:** state is genuinely sufficient for the myopic
formulation ⇒ the limiter is the formulation itself ⇒ Experiment 1.

**3b — reward SoC handling.** Single coupled change: `k_fb` 2.5 → **1.656**
(ECMS‑matched slope) **and** replace the inert deadband+terminal (R3/R4) with a
**live per‑step dynamic‑SoC term** `-w_soc · (soc - 0.5)² · ramp(t)` where
`ramp` grows over the episode (DP‑informed shape, L4), tuned so its
cycle‑integral ≈ the current terminal penalty's *intended* magnitude.
**Frozen:** everything else. **Measure:** CS robustness across seeds, ΔSoC
distribution, whether the operating strategy (OFF% by band, T_CE|on) shifts
toward ECMS, cross‑cycle CS.

**Expected (positive):** 3/3 CS at the ECMS‑matched slope (which Phase 4/5
couldn't achieve *because* the terminal term was invisible — this fixes that),
NEDC V_CE roughly unchanged or −0.05, cross‑cycle CS improved.
**Interpretation:** if CS holds at slope 1.656 with the live term, then the
Phase‑7 "k_fb is a flat plateau" conclusion was an artefact of testing with a
dead terminal reward, and the ECMS‑matched slope should be the new default
(it also makes Experiment 1's `K_FB=1.656` choice clean).

---

## 12. THE SINGLE BEST NEXT EXPERIMENT

**Experiment 1 — Architecture G: RL learns a bounded costate correction Δλ; a
deterministic Hamiltonian minimiser produces `u`.**

**Why it has the highest expected information gain:**

* It is the *only* candidate that attacks **both** measured gap components
  (mode/timing 55–75 % **and** operating‑point 20–40 %) with a single change.
  Every other lever attacks one.
* Every Phase 5–9 forensic result *converges* on it:
  – Phase 7: the actor is displaced from its own critic's argmax (CASE D). A
    minimiser has no "actor" to displace.
  – Phase 8: the Q‑oracle proves the critic *can't be exploited* by any policy;
    the minimiser doesn't rely on the learned critic for `u` at all.
  – Phase 9: the defect is thin coverage of the efficient region. A pointwise
    minimiser *reaches that region by construction* on every step — no coverage
    needed.
  – The reward's own instantaneous argmax already wants the harder engine load
    (Phase 8 §16); the minimiser follows the reward's Hamiltonian exactly.
* It is strongly supported by the literature that actually matches/beats
  A‑ECMS (L1, L2, L11) and *not* supported‑against by anything the project has
  found.
* It is **low‑risk**: bounded correction around a nominal that is already
  known to be near‑optimal (the k_fb=1.656 ECMS law); if the head learns ≈0
  the result is still "analytic layer closes the gap", which is a shippable
  win and a clean diagnosis.
* It is **cheap**: ~2 engineering days (the minimiser already exists in
  `ecms.py`; the wrapper is ~80 lines), then the standard 3‑seed / 150k smoke +
  500k confirm.
* It produces a **decisive** outcome either way (see Experiment 1's
  positive/negative interpretations) — unlike another coverage or entropy
  tweak, which can come back "inconclusive."

**Highest probability of materially reducing the gap:** the two things ECMS
does that SAC cannot — pointwise BSFC‑optimal `u` and a globally consistent
costate — are *exactly* what this architecture grants SAC (the first exactly,
the second up to a learned bounded correction). The expected landing (NEDC
3.30–3.45, closing 50–75 % of the gap) would be the largest single move since
the Phase‑2 pivot.

**Pre‑req (run this week, no training):** the §5.3 ECMS sensitivity sweep + a
DP solver, so that when Experiment 1 lands at (say) 3.35 we can say whether the
remaining 0.16 is DP‑addressable or ECMS‑tuning‑advantage.

---

## 13. PATH TO ECMS — STAGED ROADMAP

Targets (mandate §27): **A** beat the (authority‑equal) rule‑based benchmark;
**B** within 5 % of ECMS; **C** within 1–2 % of ECMS.

| Cycle | RB (auth‑equal) | ECMS | Target A | Target B (≤5 %) | Target C (≤2 %) |
|---|---|---|---|---|---|
| NEDC | 3.579 | 3.189 | < 3.579 | < 3.348 | < 3.253 |
| FTP75 | 3.232 | 2.810 | < 3.232 | < 2.950 | < 2.866 |

Current: NEDC 3.767 (misses A by +5.2 %), FTP75 3.289 (misses A by +1.8 %).

### Stage 1 — Beat the rule‑based benchmark on both cycles (Target A)
**Requirements:** close ~0.19 L/100km on NEDC (FTP75 is essentially there).
**Route:** Experiment 1 (Δλ + minimiser) alone is expected to clear this by a
wide margin. If Experiment 1 is delayed, Experiment 2 (efficient‑region
coverage) is the fallback and should reach ~3.55–3.70.
**Also:** run the ECMS sensitivity sweep + DP solver; reproduce ECMS numbers;
add the positive‑floor `eq_factor_eff` clip (hygiene, EXP‑E).
**Exit criterion:** NEDC < 3.579 and FTP75 < 3.232, 3/3 CS, 0 violations,
3 seeds.

### Stage 2 — Close 50 % of the SAC→ECMS gap (NEDC ≤ 3.478, FTP75 ≤ 3.050)
**Requirements:** the operating‑point component (20–40 %) *and* about half the
mode/timing component must go.
**Route:** Experiment 1 as the controller + Experiment 3b (ECMS‑matched `k_fb`
slope 1.656 with a live dynamic‑SoC term) so the costate anchor is
proven‑optimal, not the 1.5× patch. Add Experiment 2's exploration schedule if
Experiment 1 alone under‑performs on the operating point.
**Exit criterion:** NEDC ≤ 3.48, FTP75 ≤ 3.05, 3/3 CS, and **cross‑cycle CS
≥ 2/3** (transfer must not be worse than the specialised policy — the Δλ layer
should make the costate law transferable).

### Stage 3 — Reach ≤ 5 % from ECMS (NEDC ≤ 3.348, FTP75 ≤ 2.950)
**Requirements:** the remaining gap is now (a) ECMS's whole‑cycle λ₀ tuning
advantage and (b) minimiser grid resolution and (c) online costate‑adaptation
noise.
**Route:**
* Increase the minimiser grid to 161–321 points (cheap; matches a finer ECMS).
* Add near‑future demand context to the Δλ head's input (short predicted‑power
  / cycle‑phase feature, L10) so the online costate correction anticipates the
  next low/high‑load stretch — this is the causal analogue of ECMS's
  whole‑cycle tuning.
* Offline warm‑start the Δλ head from a *feasible‑diverse* dataset (H5/L8) so
  it doesn't have to discover the small‑correction regime from scratch.
**Exit criterion:** NEDC ≤ 3.35, FTP75 ≤ 2.95, 3/3 CS, cross‑cycle CS 3/3.

### Stage 4 — Attempt ≤ 2 % (NEDC ≤ 3.253, FTP75 ≤ 2.866)
**Requirements:** at this point the DP solver's number is essential. If DP is,
say, 3.10 on NEDC, then ≤3.253 means "within 5 % of DP" and is plausible for a
preview‑equipped causal controller. If DP ≈ ECMS (3.19), then ≤2 % of ECMS is
≤ the ECMS‑tuning advantage itself and may be **structurally unreachable**
without whole‑cycle preview — in which case the honest deliverable is "matched
DP to within X %, cannot match cycle‑tuned ECMS causally, and here is why."
**Route:** receding‑horizon Δλ (short MPC over the Δλ head with the minimiser
inner loop, critic as terminal value — Experiment 8/H8) + full‑cycle
preview features + a DP‑distilled reward shaping term.
**Exit criterion:** NEDC ≤ 3.25 **or** a documented proof that the residual is
the ECMS cycle‑tuning advantage.

**Realistic assessment:** Stages 1–2 are high‑confidence with Experiment 1.
Stage 3 is plausible. Stage 4 depends on the DP number and on whether
whole‑cycle preview is deemed in‑scope (the project has kept preview short for
generalisation; Stage 4 likely requires relaxing that or accepting the
cycle‑tuning gap as irreducible).

---

## 14. FINAL RESEARCH VERDICT

### What is the most likely fundamental reason this project has failed to close the ECMS gap despite Phases 1–9?

**The problem was reformulated — correctly, for tractability — into one whose
optimum is a per‑step Hamiltonian minimisation with a costate, and then
end‑to‑end value‑based RL was used to *learn that minimisation by trial and
error* from a training budget (150k steps) and an on‑policy data distribution
that structurally never covers the region where SAC and ECMS diverge.** The
residual gap is the approximation error of a small‑budget bootstrapped
function approximator against a closed‑form optimiser, concentrated in a
data‑starved slice of the state–action space (part‑load / efficient‑engine‑load
at 15–35 Nm demand) that the actor never enters because the critic never learns
to value it because the actor never enters it. Phases 8 and 9 correctly proved
this is *not* fixable by more actor capacity or by critic pessimism, and Phase
7 proved it is *not* an economic‑pricing error. Every remaining phase kept
tuning the *learned* controller instead of asking whether the learned
controller should be doing this optimisation at all. Secondary contributors,
all real but smaller: an over‑steep training costate slope with a dead terminal
reward (the γ=0.20 pivot left SoC regulation entirely to a mis‑calibrated
per‑step term), a state representation with no features for anticipatory
bank/spend, and a 150k‑step budget on which "the optimum is myopic" and "don't
train longer" were concluded but not safely established.

### What is the most promising remaining technical lever?

**Hybridise: let a deterministic Hamiltonian minimiser (the project already has
one, in `ecms.py`) choose the instantaneous torque split, and restrict the RL
to learning a bounded, state‑and‑history‑dependent correction to the
equivalent factor / costate (Architecture G, escalating to a full learned
state‑dependent λ if the correction saturates).** This eliminates the actor
displacement (no actor to displace), eliminates the operating‑point error (the
minimiser is pointwise BSFC‑optimal by construction), eliminates the coverage
deadlock (the minimiser visits the efficient region every step), reduces the
RL problem to smooth 1‑D scalar regression, and is exactly the design that the
recent literature which *matches or beats A‑ECMS* converges on. It is
low‑risk (a bounded correction around a proven‑near‑optimal nominal), cheap
(~2 days), and produces a decisive result either way. The runner‑up levers —
targeted coverage of the efficient engine‑load region, a state ablation for
timing features, and an offline non‑expert warm‑start — are worth running in
parallel as they are cheap and independently informative, but Architecture G
is the shortest technically defensible path to ECMS.

---

## APPENDIX A — Files reviewed

`src/env/ems_env.py`, `src/env/powertrain.py`, `src/env/driving_cycle.py`,
`src/baselines/ecms.py`, `src/baselines/advanced_rule_based.py`,
`src/agents/train_sac.py`, `data/params.json`,
`data/maps/engine_maps_data.py`, `README.md`, `ROADMAP.md`,
`EXPERIMENT_LOG.md`, `VERIFIED_FACTS.md`, `RL_DIAGNOSTIC_REPORT.md`,
`experiments/experiment_registry.yaml`, `PHASE2_FINAL_REPORT.md`,
`PHASE4_FINAL_REPORT.md`, `PHASE5_FINAL_REPORT.md`,
`PHASE5B_FORENSIC_CLOSURE.md`, `PHASE6_FINAL_REPORT.md`,
`PHASE7_FINAL_REPORT.md`, `PHASE8_REPORT.md`, `PHASE9_FINAL_REPORT.md`,
`results/phase7/data/*.json`, `results/phase8/data/*.json`,
`results/phase9/data/*.json`.

## APPENDIX B — Prior conclusions this report challenges

| Prior conclusion | Phase | This report's position |
|---|---|---|
| "The optimum here is myopic; γ is closed at 0.20." | 2, 4, E10 | **Over‑claimed.** Established only at 150k steps; high‑γ arms confounded by a diverging critic; a myopic policy provably cannot do anticipatory load‑point shifting, which is part of the measured gap. Re‑test on the stable config, 3 seeds, longer budget. |
| "The reward is sufficient; do not modify it." | 8 §16 | **Over‑claimed.** Rests on an *instantaneous* argmax; the `k_fb` slope is 1.5× ECMS's and the terminal SoC term is inert (0.77 % of reward). A DP‑informed dynamic‑SoC term + ECMS‑matched slope is warranted. |
| "State is sufficient; the reward carries the fuel consequence." | 8, 9 | **Non‑sequitur under γ=0.20.** A myopic critic cannot use a downstream reward. State lacks timing features; cross‑cycle CS 0/3 is the tell. Ablation never run — now in scope. |
| "`k_fb` is a flat plateau [2.0, 3.0]; not the lever." | 7 | **Confounded.** Measured with a dead terminal reward, so lowering `k_fb` broke CS for a reason unrelated to the slope's correctness. Re‑test with a live dynamic‑SoC term. |
| "Conditional replay coverage is refuted." | 6 | **Correct for the OFF region tested.** But the *efficient high‑engine‑load* region (a different hole, Phase 9) was never targeted. |
| Gap attribution "≈60 % mode / ≈25 % operating‑point / ≈15 % other." | 7, 8 | Superseded by Phase 9's BSFC‑grounded 39/61 (NEDC); the honest bracket is operating‑point 20–40 %, mode/timing 55–75 %. Report whichever with its method. |
| ECMS 3.1887 / 2.8097 treated as a near‑oracle target. | throughout | ECMS's λ₀ is **whole‑cycle bisection‑tuned** — a real information advantage. Without a DP reference the addressable vs irreducible split is **unidentified**. Build DP; run the ECMS sensitivity sweep. |
