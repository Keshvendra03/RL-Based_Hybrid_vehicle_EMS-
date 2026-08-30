# PHASE 8 — ACTOR-SIDE BREAKTHROUGH + ECMS GAP CLOSURE

## Forensic Decision Report

**Status:** 8A / 8B / 8G / reward-sufficiency / **8C (both cycles)** complete.

**8C outcome (3 seeds each, everything except the actor policy class frozen):**

| | NEDC | FTP75 |
|---|---|---|
| CONTROL (unimodal) | 3.7666 (3/3 CS) | 3.2889 (3/3 CS) |
| **8C mixture actor** | **3.8730 ± 0.122 (1/3 CS)** — WORSE | **3.2462 ± 0.003 (3/3 CS)** — best FTP75 SAC yet, ties rule-based (3.2323) but still +0.4 % above it |

On **NEDC** (the real gap) the mixture is worse and loses charge-sustaining. On
**FTP75** it is a mild win (joint-best SAC config, 3/3 CS) — but FTP75 was
already at the benchmark. Both cycles: actor↔critic alignment **improved**
markedly (NEDC 30–35 Nm arg-max distance 0.718 → 0.226; FTP75 15–30 Nm 0.360 →
0.177) and the critic's own arg-max-Q-OFF share **fell toward the actor's**
(NEDC 30–35: 87 % → 23 %; FTP75 15–30: 80 % → 31 %). The mixture components
**partly collapsed** (separation 0.03–0.19). **Neither cycle beats rule-based.**
⇒ **the policy class is not the bottleneck for the NEDC gap** — consistent with
the 8B Q-oracle ceiling.

> Every section quotes the corresponding Phase-8 brief instruction (block-quote)
> then answers it with measured evidence, so the question each number answers is
> explicit — same convention as `PHASE7_FINAL_REPORT.md`.

**Headline (from the forensics that gate everything else):**

> The Phase-7 conclusion *"the critic already knows what to do; only the actor
> fails to realize it"* is **DEMOTED by the Q-oracle test.** When the policy is
> replaced by an ideal actor that follows the trained critic exactly
> (arg-max min-Q over a dense feasible grid, every step), it does **NOT** beat
> the rule-based benchmark and **loses charge-sustaining**:
>
> | | NEDC | FTP75 |
> |---|---|---|
> | A — current SAC actor (3 seeds) | **3.7666** (3/3 CS) | **3.2889** (3/3 CS) |
> | **B — SAC-Q oracle (3 seeds)** | **3.9404**, σ 0.47, **1/3 CS**, ΔSoC −2.5pp | **3.3545**, σ 0.03, **0/3 CS**, ΔSoC −4.1pp |
> | advanced rule-based | 3.5056 | 3.2323 |
> | ECMS | 3.1887 | 2.8097 |
>
> The Q-oracle is a **strict upper bound** on what *any* policy representation
> (unimodal, mixture, hierarchical) can extract from **this frozen critic**. That
> ceiling is *worse than the current actor* and *not charge-sustaining*.
> **⇒ the binding constraint is the CRITIC's value estimate, not the policy class.**
> The actor's conservatism (staying near its training distribution) is partly
> *protective*: naive exploitation of the critic's engine-OFF lobe over-discharges.

---

## §1 — FREEZE THE CURRENT KNOWLEDGE

> *"Before changing anything, create a formal Phase-8 baseline record from the
> latest validated configuration. Treat these as frozen reference facts …"*

Frozen (carried from Phases 2–7, machine-readable in
`results/phase8/baseline/data/00_baseline_lock.json`):

| item | value / status |
|---|---|
| git commit (Phase-8 start) | `f1f45c559e126e67a7fc01634895a22b6e08e8de` |
| SAC implementation | functionally validated (Phase 2) — **frozen** |
| NEDC / FTP75 env, ECMS, advanced rule-based | validated — **frozen** |
| gamma | investigation **CLOSED**; 0.20 primary, 0.90 secondary ref |
| n_step | **1** (frozen, primary) |
| k_fb | 2.5 (NEDC charge-sustaining candidate); **NOT the actor-selection lever** |
| reward / equivalent-factor redesign | **not the first intervention** |
| replay-buffer size | **not the first intervention** |
| TD3 / PPO / DDPG | **not authorized** |
| CONTROL | gated `k_fb=2.5`, γ 0.20, n_step 1, eq_factor 0.2717/0.4981, target_entropy auto, lr 3e-4, batch 512, buffer 300k, grad_steps 16, lookahead 5, net [256,256], 150k steps, 3 seeds |

**CONTROL 3-seed scorecard (reproduced, §3):**

| cycle | V_CE mean ± σ (95% CI) | ΔSoC per seed (pp) | SoC[min,max] | OFF% | ASSIST% | LPS% | engine-on | mean \|T_CE\| when ON | viol | CS |
|---|---|---|---|---|---|---|---|---|---|---|
| **NEDC** | **3.7666 ± 0.0785** ([3.678, 3.855]) | +0.28 / −0.72 / +0.23 | [23.6, 50.4] | 38.6 | 14.6 | 29.8 | 376–393 s | **53.0–53.5 Nm** | 0 | **3/3** |
| **FTP75** | **3.2889 ± 0.0174** ([3.269, 3.309]) | −0.53 / −0.51 / −0.07 | [35.7, 55.7] | 41.1 | 6.4 | 26.8 | 471–504 s | **66.6–72.5 Nm** | 0 | **3/3** |

Relative gaps: NEDC (SAC−RB)/RB **+7.4 %**, (SAC−ECMS)/ECMS **+18.1 %**;
FTP75 (SAC−RB)/RB **+1.8 %**, (SAC−ECMS)/ECMS **+17.1 %**.

---

## §2 — CURRENT ROOT CAUSE (do not regress) + §3 — REPRODUCE THE FORENSIC BASELINE

> *"The latest matched-state and dense-action evidence indicates … CRITIC → largely
> knows what to do; ACTOR → fails to realize what the critic wants. Therefore the
> next intervention must primarily attack the actor/policy representation. Do NOT
> revert to the old explanation that the replay buffer simply contains no OFF
> transitions."*
>
> *"Re-run or reproduce … NEDC/FTP75 matched-state Q landscape, actor mean, actor
> sigma, P(OFF), argmax-Q mode, ECMS mode, ΔQ(OFF−ASSIST), ΔQ(OFF−LPS),
> Δr(OFF−ASSIST), actor-to-argmax-Q distance, engine torque when ON, regional fuel,
> final SoC, constraint violations. Use exactly the Phase-7 state-generation
> methodology. Archive as results/phase8/baseline/."*

Reproduced with the **identical Phase-7 methodology** (matched states = a fresh
clean `EMSEnv` deep-copied at every traction step along the charge-sustaining
**ECMS SoC trajectory**). Archived: `results/phase8/baseline/` (raw txt, `data/*.json`,
`matched_states_{NEDC,FTP75}.csv`). Actor-to-arg-max-Q distance and P(mode):
`results/phase8/data/actor_alignment_{NEDC,FTP75}.json`.

### Matched-state summary (CONTROL, ECMS-trajectory states)

**NEDC**

| region | n | Δr(OFF−ASSIST) med (>0%) | ΔQ(OFF−ASSIST) med (>0%) | \|a_sac − argmaxQ\|/2 | OFF% actor / argmaxQ / ECMS |
|---|---|---|---|---|---|
| 15–30 | 90 | −0.0034 (42 %) | **+0.0064 (62 %)** | **0.322** | 47 / 72 / 71 |
| **30–35** | 90 | −0.0021 (14 %) | **+0.0234 (91 %)** | **0.718** | **0 / 87 / 40** |
| 35–50 | 41 | −0.0045 (15 %) | +0.0008 (51 %) | 0.103 | 15 / 17 / 29 |
| 50–75 | 80 | ≈0 (median) | ≈0 (median) | 0.054 | 0 / 5 / 20 |

**FTP75**

| region | n | Δr(OFF−ASSIST) med (>0%) | ΔQ(OFF−ASSIST) med (>0%) | \|a_sac − argmaxQ\|/2 | OFF% actor / argmaxQ / ECMS |
|---|---|---|---|---|---|
| 15–30 | 90 | +0.0077 (82 %) | **+0.0120 (89 %)** | 0.360 | 33 / 80 / 89 |
| 30–35 | 80 | +0.0020 (62 %) | +0.0080 (78 %) | 0.219 | 5 / 31 / 48 |
| 35–50 | 90 | −0.0006 (48 %) | +0.0050 (64 %) | 0.262 | 6 / 33 / 38 |
| 50–75 | 90 | −0.0065 (32 %) | −0.0178 (24 %) | 0.146 | 0 / 4 / 11 |

The Phase-7 forensic picture reproduces exactly: the critic's arg-max-Q prefers
engine-OFF at 15–35 Nm (≥ ECMS's OFF share); the deterministic actor does not,
with the misalignment maximal at NEDC 30–35 Nm (distance 0.718 = 72 % of the
half-range). **§2's instruction not to regress to "buffer has no OFF" is
respected** — the diagnosis stays conditional/actor-and-critic-representation,
and §8B below tests it properly.

---

## §17 / §4-preview — THEORETICAL CEILING: Policy A vs Q-oracle vs ECMS  (PHASE 8B)

> *"Construct three hypothetical policies … A = current SAC actor; B = SAC-Q
> oracle (at every matched state choose the feasible action maximizing the trained
> SAC critic); C = ECMS. If Q-oracle approaches ECMS while the actor does not →
> actor is the dominant bottleneck. If Q-oracle remains far behind ECMS → the
> critic/reward/state representation itself is limiting performance. This
> experiment must determine whether the current SAC formulation has enough
> information to approach ECMS before considering an algorithm switch."*

`results/phase8_qoracle.py` — Policy B = arg-max over a 121-point feasible action
grid of the trained twin-critic `min(Q1,Q2)`, rolled through the **real validated
env** (no deepcopy shortcut; the trajectory is allowed to diverge). 3 critics
(the 3 CONTROL seeds). Data: `results/phase8/data/qoracle_ceiling_{NEDC,FTP75}.json`.
Figure: `results/phase8/figures/ceiling_bars.png`, `qoracle_ceiling_{NEDC,FTP75}.png`.

| metric | NEDC A (actor) | NEDC B (Q-oracle) | FTP75 A | FTP75 B |
|---|---|---|---|---|
| V_CE mean (3 seeds) | **3.7666** | **3.9404** | **3.2889** | **3.3545** |
| V_CE σ | 0.079 | **0.470** | 0.017 | 0.026 |
| per-seed V_CE | 3.686 / 3.843 / 3.770 | **3.634 / 3.705 / 4.482** | 3.270 / 3.304 / 3.293 | 3.342 / 3.384 / 3.337 |
| ΔSoC mean (pp) | −0.07 | **−2.48** | −0.37 | **−4.05** |
| per-seed ΔSoC (pp) | +0.28 / −0.72 / +0.23 | **−2.55 / −5.95 / +1.07** | −0.53 / −0.51 / −0.07 | **−5.77 / −2.13 / −4.27** |
| charge-sustaining | **3/3** | **1/3** | **3/3** | **0/3** |
| OFF% | 38.6 | 40.7 | 41.1 | 43.3 |
| mean engine \|T_CE\| when ON | 53.3 | 61.6 | 69.2 | 74.0 |
| constraint violations | 0 | 0 | 0 | 0 |
| beats advanced rule-based? | no (+7.4 %) | **no (+12.4 %)** | no (+1.8 %) | **no (+3.8 %)** |
| vs ECMS | +18.1 % | **+23.6 %** | +17.1 % | **+19.4 %** |

**The Q-oracle closes −30 % (NEDC) / −14 % (FTP75) of the A→ECMS gap — i.e. it
makes things worse — and destroys charge-sustaining.**

Mechanism (from the per-seed traces): the greedy-Q policy follows the critic's
engine-OFF preference, SoC bleeds down, and the critic — trained only on the
narrow on-policy distribution the actor visits — keeps rating OFF/discharge
highly off-distribution, so SoC runs away. The one seed that stays CS (NEDC
seed2) does so by luck of trajectory and pays 4.48 L/100km. High σ (0.47) is the
signature of an unstable greedy policy on an inconsistent value function.

> **§17 verdict: Q-oracle remains far behind ECMS ⇒ the critic / reward / state
> representation is the limiting factor, NOT the policy class.** Per §18 this does
> **not** authorize an algorithm switch (other gate conditions unmet), and it
> reframes §4/§23-8C: a mode-aware actor built on this critic is *upper-bounded by
> this failing ceiling* unless co-training also repairs the critic.

---

## §14 / §15 — ENGINE OPERATING-POINT DIAGNOSIS  (PHASE 8G)

> *"SAC operates the engine at substantially lower torque than ECMS when
> engine-on. Determine whether this is: A actor cannot reach the high-efficiency
> region / B critic undervalues high-engine-load actions / C reward does not
> represent engine efficiency / D action mapping compresses the useful region /
> E ECMS exploits instantaneous efficiency the SAC reward does not / F state lacks
> BSFC information / G SAC produces excessive intermediate engine torque / H
> combination. DO NOT choose the answer before measuring it."*
>
> *"For matched ECMS states, dense action sweep … If Q(high-load) > Q(low-load)
> but actor selects low-load → actor problem. If Q(low-load) > Q(high-load) while
> ECMS prefers high-load → critic/reward/state problem."*

`results/phase8_qoracle.py::engine_op_counterfactual` +
`results/phase8_reward_state.py::reward_counterfactual`. 60 matched ECMS states
per torque band, 161-point action sweep. Data:
`results/phase8/data/engine_op_counterfactual_{C}.json`,
`reward_counterfactual_{C}.json`. Figure: `results/phase8/figures/engine_operating_point.png`.

### (a) What does the *immediate reward* want? (§16)

Reward-optimal engine torque `argmax_a r(a)` vs ECMS vs actor, per band:

| band | NEDC argmax-r / ECMS / actor T_CE | FTP75 argmax-r / ECMS / actor T_CE |
|---|---|---|
| 0–15 | 5.0 / 4.9 / 1.1 | 6.5 / 5.3 / 3.9 |
| 15–30 | **33.8** / 15.9 / 21.8 | **25.6** / 7.7 / 19.3 |
| 30–35 | **58.3** / 31.7 / 50.5 | **59.5** / 37.0 / 50.0 |
| 35–50 | **61.0** / 51.6 / 55.4 | **75.9** / 56.2 / 65.8 |
| 50–75 | **106.7** / 84.2 / 96.0 | **107.6** / 90.9 / 97.2 |
| > 75 | **150.4** / 111.3 / 133.0 | **153.6** / 127.6 / 131.1 |

**The reward-optimal engine load is HIGHER than ECMS's in every band on both
cycles.** The current reward, followed greedily, would run the engine *harder*
than ECMS, not softer. ⇒ **hypothesis C is REJECTED** — the reward does carry an
engine-efficiency preference (implicitly, via the fuel term + the battery price),
and it points the *right* way.

### (b) What does the *critic* value? (§15)

Q evaluated at the actor's operating point vs the ECMS point vs the max-feasible-
load point, matched ECMS states:

| band | NEDC Q@actor / Q@ECMS / Q@maxload | FTP75 Q@actor / Q@ECMS / Q@maxload |
|---|---|---|
| 0–15 | **−0.144** / −0.158 / −0.178 | **−0.055** / −0.066 / −0.071 |
| 15–30 | **−0.221** / −0.229 / −0.239 | **−0.078** / −0.085 / −0.081 |
| 30–35 | **−0.271** / −0.275 / −0.275 | **−0.101** / −0.104 / −0.101 |
| 35–50 | **−0.226** / −0.236 / −0.235 | **−0.119** / −0.123 / −0.121 |
| 50–75 | **−0.339** / −0.429 / −0.342 | **−0.141** / −0.145 / −0.143 |
| > 75 | **−0.374** / −0.376 / −0.389 | **−0.216** / −0.217 / −0.227 |

**`Q@actor-load` is the highest (least-negative) value in EVERY band on BOTH
cycles.** The critic rates the current policy's (softer, more-OFF) operating point
*above* the ECMS operating point and above the max-load point. It does **not**
know the ECMS engine loading is better.

### (c) Diagnosis against the A–H menu

| option | verdict | evidence |
|---|---|---|
| A — actor cannot reach the high-efficiency region | **REJECTED** | argmax-r reaches 150 Nm; the gated map exposes full authority; the Q-oracle *does* run harder (61.6 vs 53.3) yet still loses |
| **B — critic undervalues high-engine-load actions** | **CONFIRMED (primary)** | Q@actor ≥ Q@ECMS ≥ Q@maxload in every band, both cycles, while the reward prefers the harder point |
| C — reward lacks engine-efficiency info | **REJECTED** | argmax-r T_CE > ECMS T_CE everywhere (§16 satisfied) |
| D — action map compresses the engine-load region | **MINOR** | possible contributor to precision but not the cause — authority and reward gradient both reach the ECMS point |
| E — ECMS has a structurally different objective | **PARTLY TRUE** | ECMS re-solves the exact instantaneous Hamiltonian with perfect plant knowledge each step; a smooth critic+policy at 150k steps cannot match that pointwise BSFC precision — a genuine irreducible component |
| F — state lacks BSFC info | **UNLIKELY** | the reward already encodes the fuel consequence; the critic could learn it from data it does not have (coverage), not from a missing feature |
| G — SAC produces excessive intermediate engine torque | **SYMPTOM** | true of the actor, but it is following a critic that endorses it |
| H — combination | **YES** = **B (dominant) + E (irreducible) + D (minor)** | |

> **Engine-operating-point diagnosis: the critic undervalues hard engine load
> (B).** The reward is adequate (C rejected, §16 satisfied). The actor is a
> symptom (G). A structural ECMS advantage (E) sets a floor below which no
> function-approximation controller reaches.

---

## §21 — LIVING HYPOTHESIS TABLE (updated after 8A/8B/8G)

| Hypothesis | Evidence | Status | Next test |
|---|---|---|---|
| γ too low/high | Phase 5–7; γ closed | **CLOSED** | none |
| k_fb is primary bottleneck | Phase 7 §6/§8; flat plateau [2.0,3.0] | **REJECTED** | none |
| replay buffer globally lacks OFF | Phase 5B: 15.6 % OFF at 30–35 | **REJECTED** | none |
| conditional OFF coverage matters | Phase 6 refuted the coverage→Q link; Phase 8B: greedy-Q OFF → SoC collapse | **CONTRIBUTING (off-distribution)** | critic-coverage / conservative-Q |
| critic is grossly wrong | Phase 7: arg-max mode ≈ ECMS; **Phase 8B: greedy-Q underperforms actor & loses CS**; Phase 8G: Q@actor ≥ Q@ECMS | **CONFIRMED as the binding limit** (value-fidelity off-distribution, not gross mis-ranking) | conservative/ensemble critic; targeted coverage of the ECMS operating region |
| actor cannot represent bimodal Q | Phase 7 §3/§4; distance 0.718 at NEDC 30–35 | **REJECTED as the binding cause** — Q-oracle (ideal actor) does worse; **8C mixture actor: 3.873 (1/3 CS), WORSE than CONTROL**, components collapsed, alignment improved but fuel/CS did not | none — critic side |
| entropy prevents boundary commitment | not yet isolated | **TESTABLE** | 8E entropy A/B (only if 8C directional) |
| engine operating point inefficient | Phase 7 §11; Phase 8G | **CONFIRMED** | cause = critic undervaluation (B), see §14 |
| reward lacks engine-efficiency info | Phase 8G §16: argmax-r T_CE > ECMS T_CE everywhere | **REJECTED** | none |
| state lacks required information | reward already encodes fuel consequence | **UNLIKELY** | feature-sufficiency test if 8C+critic fail |
| SAC algorithm itself inadequate | §18 gate: 4/6 conditions unmet | **NOT PROVEN** | only after 8C + critic work |

---

## 8C RESULTS — MODE-AWARE (2-COMPONENT MIXTURE) ACTOR

> *"Test the hypothesis: the unimodal squashed-Gaussian SAC actor is structurally
> inadequate for a bimodal Q landscape. Implement an actor-side policy that can
> explicitly represent the two modes … The critic should remain as unchanged as
> possible … The ONLY intended scientific change is UNIMODAL GAUSSIAN ACTOR →
> MODE-AWARE / MULTIMODAL ACTOR. … First run with the current frozen
> hyperparameters. If the result improves P(OFF), actor/Q alignment, 15–35 Nm
> fuel, NEDC V_CE, then the actor representation hypothesis is supported."*

**Implementation** (`results/phase8_mixture_policy.py`, `phase8_train_mixture.py`):
a `MixtureSACPolicy` whose actor emits a **2-component tanh-squashed diagonal-
Gaussian mixture** (component means, log-stds, mixing logits). The SAC objective
is unchanged — the mixture log-prob is a correct single-sample estimate, so the
entropy-temperature auto-tuning, twin-Q target, `tau`, `gamma`, replay, batch,
lr, `net_arch`, observation, reward, env, feasibility masks, `k_fb`, `lookahead`,
`n_step`, training budget are **all identical to the CONTROL**. No benchmark /
ECMS action is used anywhere (§5). Action *coordinate* representation is the
existing `modeaware_gated` map, unchanged — only the *policy decision*
representation is multimodal (§6).

Reachable-action audit: the mixture actor's action still passes through the same
`map_action_to_u` + feasibility clamps as the CONTROL, so per-state feasible OFF
region / ON region / engine & motor torque ranges / battery power / action bounds
are byte-identical to the CONTROL env (verified — the env is untouched).

### 8D — 3-seed results

**FTP75 8C mixture (3 seeds):** V_CE **3.2462 ± 0.0028** (95 % CI [3.243, 3.249]),
**3/3 charge-sustaining**, 0 violations; per-seed 3.2481 / 3.2475 / 3.2430;
ΔSoC +0.01 / +0.35 / +0.56 pp; mean engine \|T_CE\| when ON 68.8–72.5 Nm; OFF ≈
42 %. This is the **joint-best FTP75 SAC configuration** (matches gated
`k_fb=1.656`'s 3.2460) and marginally beats CONTROL 3.2889 — but is still
**+0.4 % above rule-based 3.2323**, so the benchmark is not beaten. FTP75
alignment: |a_sac − argmaxQ|/2 at 15–30 / 30–35 Nm **0.360 / 0.219 → 0.177 /
0.111**; argmax-Q-OFF share 80 / 31 % → 31 / 12 % (converged to the actor).
FTP75 gap decomposition (best seed vs ECMS): total ≈ +0.433; 15–30 +0.121,
35–50 +0.171 dominate; engine still soft (57 vs 71 at 35–50).

### 8D — 3-seed results (NEDC)

`results/phase8c_forensics.py`, deterministic eval on the `sac_ems_best`
checkpoints. Data: `results/phase8/data/phase8c_forensics_NEDC.json`.
Training curves: `results/phase8/figures/phase8c_training_curves.png`.

| metric | CONTROL (unimodal) | 8C mixture (NEDC, 3 seeds) | Δ |
|---|---|---|---|
| V_CE mean ± σ (95 % CI) | **3.7666 ± 0.079** | **3.8730 ± 0.122** ([3.735, 4.011]) | **+0.106 WORSE** |
| per-seed V_CE | 3.686 / 3.843 / 3.770 | 3.948 / **3.733** / 3.939 | — |
| charge-sustaining | **3/3** | **1/3** (seed1 only; ΔSoC +2.4/+1.9/+3.4 pp) | **worse** |
| constraint violations | 0 | 0 | — |
| OFF% | 38.6 | 38.6 | ≈ |
| mean engine \|T_CE\| when ON | 53.3 | 54.8 | ≈ (still far below ECMS ~95) |
| actor↔argmaxQ dist, 15–30 / 30–35 Nm | 0.322 / **0.718** | **0.190 / 0.226** | **improved** |
| actor OFF%, 15–30 / 30–35 (matched ECMS states) | 47 / 0 | 33 / 1 | ~unchanged / worse |
| argmaxQ OFF%, 15–30 / 30–35 | 72 / 87 | **50 / 23** | critic's own OFF preference *fell* |
| mixture component separation (mean) | n/a | **0.03 – 0.16** (components collapsed) | — |

**Reading:** co-training with the more-capable actor *did* pull the actor closer
to its critic's arg-max (30–35 Nm distance 0.718 → 0.226) **and** moved the
critic's own arg-max away from extreme OFF (30–35 argmaxQ-OFF 87 % → 23 %) — the
latter is direct evidence that the Phase-7 "critic prefers OFF at 30–35" signal
was **overestimation that co-training partially corrected**. But the net vehicle
result is **worse fuel and 3/3 → 1/3 charge-sustaining**, and the mixture
**collapsed to ~unimodal** (component separation 0.03–0.16) — SAC's single-sample
entropy proxy does not hold two components apart. Training is **stable and
progressive** (no early-peak-collapse; the 8C curves overlay the CONTROL curves
almost exactly — figure `phase8c_training_curves.png`).

**Training-stability forensics (§10):** learns progressively → **stabilizes**
(category 5), at parity-minus with CONTROL. No early peak, no collapse, no
oscillation. So "train longer" is not the missing ingredient.

---

## §12 — CAN THE ACTOR FIX ALONE CLOSE THE NEDC GAP?

> *"Compare CURRENT SAC vs MODE-AWARE SAC at matched states. Mode-selection
> recovery: how much of the ≈ +0.32 L/100km NEDC 15–35 Nm penalty disappears?
> Total NEDC improvement? SoC impact? Operating point — does the new policy still
> operate the engine too softly? … We do NOT want to incorrectly conclude that
> 'the actor solved the problem' if it merely changes OFF selection while SAC
> still operates the engine inefficiently."*

**Answer: NO — the actor fix alone does not close the gap.**

| question | measured |
|---|---|
| Mode-selection recovery (15–35 Nm) | **≈ 0.** Mixture OFF% at 30–35 Nm = 12 % (vs CONTROL ~4 %, ECMS 49 %); the +0.34 L/100km 30–35 penalty is essentially unchanged (§8F) |
| Total NEDC ΔV_CE | **−0.106 (worse):** 3.7666 → 3.8730 |
| SoC impact | **destabilized:** 3/3 → **1/3 charge-sustaining**, ΔSoC drift +1.9…+3.4 pp |
| Operating point | **still soft:** mean engine \|T_CE\| when ON = 54.8 Nm (CONTROL 53.3, ECMS ~95). The mixture did **not** move the engine operating point |

Per the brief's explicit warning: even where alignment improved, **SAC still
operates the engine inefficiently** — so this is *not* "the actor solved the
problem". It confirms the operating-point gap is critic-side (§14).

---

## §13 — UPDATED SAC-vs-ECMS DECOMPOSITION (best 8C mixture seed, NEDC)

`results/phase8/data/phase8c_forensics_NEDC.json`. Best mixture seed (s1,
V_CE 3.7325, the only CS one) vs ECMS, matched demand:

| band | ΔFuel (mix − ECMS) | OFF% mix / ECMS | engine \|T_CE\| mix / ECMS |
|---|---|---|---|
| 0–15 | +0.025 | 35 / 35 | — |
| **15–30** | **+0.199** | 65 / 77 | 28 / 40 |
| **30–35** | **+0.142** | 12 / 49 | 34 / 58 |
| **35–50** | **+0.186** | 17 / 63 | 63 / 68 |
| 50–75 | +0.043 | 0 / 22 | 73 / 95 |
| > 75 | −0.050 | 0 / 0 | 103 / 107 |
| **total** | **≈ +0.544** (V_CE 3.7325 − 3.1887 = +0.5438) | | |

4-way split — **unchanged from Phase 7 / CONTROL:** ≈ 60 % mode-selection
(15–35 Nm, critic still under-uses OFF), ≈ 25 % engine operating-point (engine
still soft), ≈ 15 % other; battery-energy management ≈ 0. **The mixture actor
moved none of these components.**

---

## §25 — THE THREE SCIENTIFIC QUESTIONS

**Q1 — "If we give SAC a policy representation capable of expressing the
multimodal decision structure its own critic has already learned, how much of the
NEDC and FTP75 performance gap disappears?"**

Measured two ways: (a) the **Q-oracle** — the ideal such policy (unbounded
capacity, follows the critic exactly): **NEDC 3.767 → 3.940, 3/3 → 1/3 CS;
FTP75 3.289 → 3.354, 3/3 → 0/3 CS.** (b) a **trained 2-component mixture actor**,
everything else frozen, co-trained so the critic *could* move: **NEDC 3.7666 →
3.8730, 3/3 → 1/3 CS** — alignment improved (30–35 Nm arg-max distance 0.718 →
0.226) but the components collapsed and fuel/CS got worse.

**Answer: essentially none of the gap disappears; charge-sustaining is lost by
both the ideal and the trained multimodal policy.** ⇒ **the policy class is not
the dominant bottleneck.** The one genuinely new nuance: co-training the mixture
*moved the critic's own arg-max away from extreme engine-OFF* (30–35 Nm
argmax-Q-OFF 87 % → 23 %), confirming the Phase-7 "critic prefers OFF at 30–35"
was partly overestimation.

**Q2 — "After the actor is fixed, why does SAC still differ from ECMS, especially
in engine operating point and fuel consumption?"**

Because the *critic* differs from ECMS, not (only) the actor. The trained
`min(Q1,Q2)` rates the current soft-engine / moderate-OFF operating point **above**
the ECMS hard-engine point in every torque band on both cycles (§14b), even
though the *immediate reward's* optimum is at a **higher** engine load than ECMS
everywhere (§14a). The critic has fit an accurate value function only on the
narrow on-policy distribution and does not extrapolate the reward's own
operating-point preference into the ECMS region it never sees.

**Q3 — "Is the remaining gap fundamentally caused by the RL algorithm, the policy
class, the reward, the state representation, or the fact that ECMS has a
structurally different optimization objective?"**

Quantified attribution of the ≈ +0.58 L/100km (NEDC) / +0.48 (FTP75) SAC→ECMS gap:

| cause | share | evidence |
|---|---|---|
| **Critic value-fidelity off-distribution** (overvalues sustained OFF → SoC collapse when exploited; undervalues hard engine load) | **dominant** | §8B Q-oracle underperforms the actor & loses CS; §8G Q@actor ≥ Q@ECMS everywhere |
| Policy class (unimodal vs bimodal Q) | **small / none on its own** | §8B: the ideal multimodal policy (Q-oracle) does worse |
| Reward | **≈ none** | §8G/§16: argmax-r engine load > ECMS everywhere; reward telescopes to −V_CE at k_fb=0 |
| State representation | **unlikely** | reward already carries the fuel consequence; the missing ingredient is coverage, not a feature |
| **ECMS structurally different objective** (exact instantaneous Hamiltonian, perfect plant knowledge, pointwise BSFC-optimal) | **irreducible floor, ~non-zero** | ECMS uses *more* engine-on time yet less fuel (Phase 5B) — pure operating-point precision a smooth approximator cannot match at 150k steps |
| RL algorithm (SAC vs TD3/…) | **NOT the cause** (per §18, not demonstrated) | implementation verified; the failure is value-fidelity, which an algorithm swap alone does not fix |

---

## §26 — DELIVERABLE CHECKLIST

1. **What changed:** nothing in the plant / env / benchmarks / reward / SAC core.
   New: `results/phase8/` forensics; a `MixtureSACPolicy` actor (8C, isolated).
2. **What did not change:** γ 0.20, n_step 1, k_fb 2.5, eq_factor, target_entropy
   auto, lr, batch, buffer, grad_steps, lookahead, net_arch, observation, reward,
   env, feasibility masks, ECMS, rule-based, evaluator, 150k budget, seeds {0,1,2}.
3. **Baseline reproduction:** §1/§3 — CONTROL 3.7666 / 3.2889, 3/3 CS both, 0 viol
   (bit-identical to Phase 7).
4. **Mode-aware actor results:** §8D — NEDC mixture **3.8730 ± 0.122, 1/3 CS**
   (WORSE than CONTROL 3.7666, 3/3 CS); FTP75 mixture **3.2462 ± 0.003, 3/3 CS**
   (joint-best SAC, still +0.4 % above rule-based). Components partly collapsed
   (sep 0.03–0.19); alignment improved on both; training stable. Neither beats RB.
5. **3-seed statistics:** §8D — NEDC mixture per-seed 3.948 / 3.733 / 3.939,
   ΔSoC +2.4 / +1.9 / +3.4 pp, 0 violations, 1/3 CS.
6. **Actor/Q alignment before vs after:** §8D — |a_sac − argmaxQ|/2 at NEDC
   15–30 / 30–35 Nm: **0.322 / 0.718 (CONTROL) → 0.190 / 0.226 (mixture)** —
   improved; but argmaxQ-OFF% *fell* (72/87 → 50/23) and V_CE/CS worsened.
7. **NEDC regional fuel decomposition:** §14 + Phase-7 §11 (15–30 +0.196,
   30–35 +0.122 dominate).
8. **FTP75 regional fuel decomposition:** §14 + Phase-7 §11.
9. **Q-oracle ceiling:** §8B — **NEDC 3.9404 (1/3 CS), FTP75 3.3545 (0/3 CS);
   worse than the actor.**
10. **SAC-vs-ECMS decomposition:** §25-Q3 table.
11. **Engine operating-point diagnosis:** §14 — **cause = critic undervalues hard
    engine load (B)**; reward is fine (C rejected).
12. **Reward sufficiency diagnosis:** §16 — reward's optimum is at *higher* engine
    load than ECMS in every band ⇒ **reward is sufficient; do not modify it.**
13. **State sufficiency diagnosis:** unlikely to be the limit — the reward
    already encodes the fuel consequence; the gap is coverage/critic-fidelity.
14. **Remaining ECMS gap:** NEDC ≈ +0.58 L/100km (+18.1 %), FTP75 ≈ +0.48 (+17.1 %).
15. **% attribution:** critic value-fidelity **dominant**; policy class **~0
    alone**; reward **~0**; state **~0**; ECMS structural floor **non-zero,
    irreducible**; algorithm **not the cause**.
16. **Rejected hypotheses:** k_fb is the lever (Phase 7); buffer globally lacks
    OFF; reward lacks engine-efficiency info; **"actor alone can close the gap"
    (Phase 8B Q-oracle + Phase 8C mixture, both worse than CONTROL)**;
    **"a multimodal policy class is the missing ingredient" (Phase 8C: mixture
    trained, components collapsed, fuel + CS worse)**.
17. **Confirmed hypotheses:** engine operating point inefficient *because the
    critic undervalues hard load*; critic value-fidelity off-distribution is the
    binding limit; ECMS has a structural precision advantage; **the Phase-7
    "critic prefers OFF at 30–35" was partly overestimation** (8C co-training
    moved argmax-Q-OFF 87 % → 23 % there).
18. **Best validated configuration:** NEDC — **still the CONTROL** (gated
    `k_fb=2.5`, 3.7666, 3/3 CS); 8C did not beat it. FTP75 — the **8C mixture
    actor** is now joint-best (3.2462, 3/3 CS) alongside gated `k_fb=1.656`
    (3.2460); still +0.4 % above rule-based.
19. **Beaten the advanced rule-based benchmark?** **NO** — NEDC +7.4 %, FTP75
    +1.8 %. The Q-oracle ceiling also does not beat it.
20. **Distance to ECMS:** NEDC +18.1 %, FTP75 +17.1 %.
21. **Retain or replace SAC?** **RETAIN.** §18 gate fails: the Q-oracle shows the
    *critic*, not the algorithm, is the limit; an algorithm swap does not fix
    value-fidelity on a narrow on-policy distribution.
22. **Exact next experiment:** see §"NEXT". Critic-side, one variable, 3 seeds,
    everything else frozen: **Option A — a conservative / pessimistic critic**
    (CQL-style OOD penalty or min-over-ensemble target) to remove the
    sustained-engine-OFF overestimation that makes greedy-Q diverge; **Option B —
    targeted on-policy coverage of the ECMS-style hard-engine operating region.**
    Run A first. **Not** a reward change (§16), **not** an algorithm swap (§18),
    **not** a k_fb sweep, **not** more actor capacity (8C refuted).

---

## NEXT — the exact next experiment

8C **failed** exactly as 8B predicted (NEDC mixture 3.873, 1/3 CS, components
collapsed, alignment up but fuel/CS down). 8E (entropy A/B) is **not triggered**
(§8 requires 8C to be directionally successful — it is not). 8H (reward) is
gated out (§16: the reward's optimum is at *higher* engine load than ECMS). 8I
(algorithm swap) is gated out (§18: the Q-oracle shows the critic, not the
algorithm, is the limit).

**The single next experiment is critic-side, one variable, 3 seeds, everything
else frozen (γ 0.20, n_step 1, gated, k_fb 2.5, unimodal actor, arch/lr/batch/
buffer/lookahead):**

> **Option A (preferred): a conservative / pessimistic critic** — add a CQL-style
> penalty (or a min-over-ensemble target) that lowers `Q` for
> out-of-distribution actions. Target the measured failure directly: the critic
> overvalues sustained engine-OFF off the on-policy distribution (8B: greedy-Q
> → SoC collapse), so pessimism there should make the value function safe to
> exploit and let a plain actor track it. Measure: does the Q-oracle on the new
> critic become charge-sustaining and beat RB?
>
> **Option B: targeted on-policy coverage of the ECMS hard-engine operating
> region** — a training-time exploration schedule that, when the engine is ON,
> occasionally commands a *higher* engine load (uniform over the feasible
> high-load interval), so the critic gets data where the reward's own preference
> (argmax-r T_CE > ECMS T_CE, §16) can be learned. This is a coverage
> intervention, not imitation — no ECMS/benchmark action is used.

Run Option A first (it attacks the SoC-collapse mechanism that is the larger
effect). If the conservative-critic Q-oracle reaches charge-sustaining and
approaches RB, train a plain SAC actor on that critic and run the full 3-seed +
decomposition. Only if **both** A and B fail does the state-sufficiency test
(§13) or, last, an algorithm change (§18 gate) come into scope.
