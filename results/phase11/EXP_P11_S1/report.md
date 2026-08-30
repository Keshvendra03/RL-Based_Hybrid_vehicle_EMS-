# EXP-P11-S1 — OFFLINE CRITIC FALSIFICATION (H-CRITIC vs H-COVERAGE)

Strictly offline critic-only refinement on the frozen CONTROL replay buffers.
Script: `results/phase11/exp_p11_s1_offline_critic.py` ·
Data: `report_data.json`, `freeze_verification.json`, `run.log`.
Env: SB3 2.8.0 / torch 2.12 CPU / numpy 2.4.6 / Python 3.13.2. `src/` unchanged.

---

## 1. FREEZE VERIFICATION (exact)

Programmatically verified before and after each run (`freeze_verification.json`):

| seed (checkpoint) | actor unchanged | replay unchanged | critic changed | config frozen |
|---|---|---|---|---|
| `models_p5s0_k2.5/NEDC` | **✓** (SHA `c0e49444…`) | **✓** (`7e792b73…`) | ✓ | γ 0.2, τ 0.005, batch 512, critic_lr 3e-4, ent_coef 0.0017051, tgt_upd_int 1, net [256,256] |
| `models_p5_k2.5/NEDC` | **✓** (`3c20c352…`) | **✓** (`417a38b9…`) | ✓ | … ent_coef 0.0013218 … |
| `models_p5_k2.5_s2/NEDC` | **✓** (`ac32cac5…`) | **✓** (`61f887f6…`) | ✓ | … ent_coef 0.0017761 … |

* **No environment interaction** for training — every gradient step drew a
  512-sample minibatch from `replay_buffer.sample()` only. The env was
  instantiated **solely** to compute the diagnostic `r(s,a)` landscape and the
  next-state transitions for the Bellman residual (no such transition ever
  entered the buffer or the critic loss).
* **Actor weights byte-identical** before/after 400k critic steps (SHA-256 of
  the full actor `state_dict`), all 3 seeds. Actor params additionally set
  `requires_grad_(False)` during the loop.
* **Replay buffer byte-identical** before/after (SHA-256 of
  obs+actions+rewards+dones), all 3 seeds. No prioritisation, no re-weighting.
* Reward / `eq_factor` / `k_fb` / γ / n_step / entropy coefficient / network
  architecture / optimiser / `τ` / target-update interval / target-construction
  formula: **unchanged**. Only critic (and, via Polyak `τ`, critic-target)
  parameters moved.
* The critic update is the **exact SB3 SAC critic loss**
  (`0.5·Σ MSE(Qᵢ(s,a), r + γ(1−d)(min_j Q̄ⱼ(s',a') − α·log π(a'|s'))))`,
  `a' ~ frozen actor`), followed by `polyak_update(τ=0.005)` every step.

## 2. CHECKPOINTS & REPLAY BUFFERS USED

The 3 NEDC CONTROL seeds exactly as they exist:
`models_p5s0_k2.5/NEDC/{sac_ems_best.zip, replay_buffer.pkl}`,
`models_p5_k2.5/NEDC/…`, `models_p5_k2.5_s2/NEDC/…`.
Each replay buffer: **150 016 transitions** (`pos`, not full). No new data.

## 3. NUMBER OF CRITIC UPDATES

`N ∈ {0, 50 000, 150 000, 400 000}` additional critic-only gradient steps per
seed (snapshots taken at each N; N=0 = original CONTROL critic). Wall time
≈ 39 min/seed on CPU (~2 335 s for 400k steps).

## 4. BELLMAN RESIDUAL TABLE  `resid = Q̂(s,a) − [r + γ(1−d)·V(s')]`

`V(s') = min_i Q(s', π_det(s')) − α·log π` (frozen actor). Deep-LPS action
`a_R*` = `argmax_a r(s,a)`. **CORE = demand bands 25–30 / 30–35 / 35–50 Nm**
(the 15–25 Nm band is excluded from deep-LPS metrics — `T_CE ≈ 58` is
physically unreachable there, 11B). Mean over CORE states, per seed:

| seed | N | **resid @ a_R\*** | resid @ a_Q\* | disagree @ a_R\* |
|---|---|---|---|---|
| s0 | 0 | **−0.01787** | +0.00199 | 0.00625 |
| s0 | 50k | −0.00861 | −0.00214 | 0.00732 |
| s0 | 150k | −0.01193 | +0.00034 | 0.00749 |
| s0 | **400k** | **−0.00661** | +0.00184 | 0.00816 |
| s1 | 0 | **−0.00403** | +0.00906 | 0.00771 |
| s1 | 50k | −0.00018 | +0.01021 | 0.00779 |
| s1 | 150k | −0.01031 | +0.00287 | 0.00677 |
| s1 | **400k** | **−0.01110** | +0.00456 | 0.01367 |
| s2 | 0 | **−0.01326** | −0.00876 | 0.01089 |
| s2 | 50k | −0.01185 | +0.00658 | 0.00595 |
| s2 | 150k | −0.00273 | +0.00683 | 0.00621 |
| s2 | **400k** | **−0.00609** | +0.00600 | 0.01313 |

Full per-state × per-N detail: `report_data.json → per_seed`. Distribution
stats (mean/median/std/min/max/n) per seed×N: `report_data.json → tables`.

**Reading:** `|resid @ a_R*|` at N=400k = **0.0066 / 0.0111 / 0.0061** — none
below the pre-registered 0.004 threshold; **s1 got WORSE** (−0.0040 → −0.0111).
The trajectory is **non-monotonic** on every seed (s0: −0.018 → −0.009 →
−0.012 → −0.007). `disagree @ a_R*` (twin-Q) **did not shrink** — it grew on
s1 and s2 (0.008 → 0.014, 0.011 → 0.013).

## 5. CRITIC-ARGMAX TORQUE TABLE  `T_CE(a_Q*)`  (CORE mean, per seed)

| seed | N=0 | N=50k | N=150k | N=400k | **net move (0→400k)** |
|---|---|---|---|---|---|
| s0 | 24.57 | 50.20 | 34.21 | 42.26 | **+17.7 Nm** |
| s1 | 32.54 | 33.99 | 21.61 | 30.14 | **−2.4 Nm** |
| s2 | 58.81 | 11.03 | 36.14 | 38.90 | **−19.9 Nm** |

**Per-state, the critic argmax is BISTABLE** — it flips between the OFF lobe
(`T_CE ≈ 0`) and the deep-LPS lobe (`T_CE ≈ 55–78`) as N changes, with no
convergence. Examples (`report_data.json`):
* s2, 35–50 Nm, SoC 40 %: `T_CE(a_Q*)` = **78.5 → 1.9 → 72.5 → 77.7**
* s1, 25–30 Nm, SoC 50 %: **8.2 → 50.3 → 0.0 → 34.4**
* s0, 30–35 Nm, SoC 50 %: **61.8 → 54.0 → 7.0 → 64.7**
* s2, 30–35 Nm, SoC 37 %: **63.4 → 0.9 → 62.0 → 64.0**

## 6. REWARD-ARGMAX TORQUE  `T_CE(a_R*)`

`T_CE(a_R*)` is **N-independent** (physics only). CORE mean = **63.22 Nm**
(constant across all seeds and all N — it is `argmax_a r(s,a)` over the feasible
grid, i.e. the deepest-LPS / `U_MIN` clamp; per 11B it is the boundary value,
not a free interior optimum). Per-state `T_CE(a_R*)` ranges 45–78 Nm by demand.

## 7. ΔT = T_CE(a_R*) − T_CE(a_Q*)   (CORE mean, per seed)

| seed | N=0 | N=50k | N=150k | N=400k |
|---|---|---|---|---|
| s0 | 38.65 | **13.03** | 29.01 | 20.96 |
| s1 | 30.69 | 29.24 | 41.62 | 33.08 |
| s2 | **4.41** | 52.20 | 27.09 | 24.33 |

ΔT does **not** shrink toward 0 with refinement. It shrinks then re-grows (s0),
stays flat / grows (s1), or **grows from 4 → 24** (s2 — the critic argmax
started *near* the reward and refinement pushed it *away*).

## 8. TWIN-Q DISAGREEMENT  `|Q1 − Q2|`

At `a_R*` (deep-LPS), CORE mean: s0 0.0063→0.0082 ; s1 0.0077→**0.0137** ;
s2 0.0109→0.0131. **Refinement did not reduce twin-Q disagreement; it increased
it on 2/3 seeds** — an instability signature, not a convergence signature.
(For scale: Phase 9 measured 0.056–0.066 for genuinely infeasible far-OOD
actions; these deep-LPS values remain an order of magnitude below that.)

**Critic-loss trajectory (`run.log`) — non-convergent / unstable:** oscillates
between ~5e-3 and periodic blow-ups — s0 spikes to **18.6** (step 75k), 2.6
(300k), 1.4 (375k); s2 to 1.45 (300k). Classic deadly-triad divergence when
fitting a bootstrapped critic on a *static* buffer with no co-evolving policy;
plausibly aggravated by the ~0.7–2.4 % of transitions with inverted `eq_eff`
(11A, SoC > 61 %, reward magnitude up to 0.87× normal).

## 9. COVERAGE-STRATIFIED RESULTS

High-engine-load replay coverage (matched fraction with executed `T_CE ≥ 50 Nm`,
from Stage-1 `s1_11CDE_NEDC.json`), per diagnostic state:

| stratum | states | coverage | critic-argmax behaviour under refinement |
|---|---|---|---|
| **15–25 Nm** | 5 | **0 %** (`T_CE ≈ 58` **unreachable**) | excluded from deep-LPS metrics |
| **25–30 Nm** | 5 | **7–9 %** (thin) | argmax stuck near the higher of {OFF, ~50 Nm}; flips on refinement |
| **30–35 Nm** | 5 | **7 % (T≈31) — 23 % (T≈35, SoC 37)** | argmax bistable 0 ↔ 54–64; T≈31 states pinned near **OFF** at every N |
| **35–50 Nm (well-covered)** | 5 | **28–54 %** | argmax **still oscillates 0 ↔ 78** across N; no stable move to the reward |

Well-covered 35–50 Nm band, `resid @ a_R*` and `T_CE(a_Q*)`, per seed:

| seed | N=0 resid / argmax | N=400k resid / argmax |
|---|---|---|
| s0 | −0.0147 / 23.1 | −0.0020 / 68.6 |
| s1 | −0.0085 / 45.4 | −0.0086 / 44.3 |
| s2 | −0.0130 / 73.4 | −0.0048 / 64.1 |

**Only s0 shows the H-CRITIC signature in the well-covered band** (resid →
−0.002, argmax → 68.6), and it does so via a non-monotonic path
(23 → 55 → 52 → 69). **s1 shows no change** (resid −0.0085 → −0.0086, argmax
45 → 44). s2's argmax started near the reward (73) and drifted down (64). The
per-state 35–50 detail confirms wild oscillation even where coverage is 38–54 %
(s2 SoC 40 %: argmax 78 → 2 → 72 → 78).

**⇒ Refinement does not reliably repair the critic in ANY coverage stratum —
including the well-covered 35–50 Nm band.** In thin-coverage states the argmax
stays at OFF regardless of N. Where it "improves" (s0), it is one seed, one
band, non-monotone.

## 10. PER-SEED RESULTS (no averaging away failures)

| pre-registered criterion (N=0 → N=400k) | s0 | s1 | s2 |
|---|---|---|---|
| `|resid @ a_R*|` (CORE) → **< 0.004** | −0.0066 ✗ | −0.0111 ✗ (worse) | −0.0061 ✗ |
| critic argmax → reward by **≥ +10 Nm** (CORE) | +17.7 ✓ | −2.4 ✗ | −19.9 ✗ (away) |
| improvement in **well-covered 35–50 band** (resid→<0.004 & argmax→reward, stable) | partial (1 band, non-monotone) | ✗ (no change) | ✗ (argmax drifted down) |
| **reproduced** | — | — | — |

**Pre-registered H-CRITIC success requires all three on ≥ 2/3 seeds. Met on
at most 1/3 (s0), and even s0 fails criterion 1.**

**Pre-registered H-COVERAGE promotion** ("after 400k, deep-LPS residual remains
substantially negative AND/OR critic argmax moves < +3 Nm, on ≥ 2/3 seeds"):
* residual stays substantially negative (worse than −0.004) on **3/3 seeds**;
* critic argmax net move < +3 Nm on **2/3 seeds** (s1 −2.4, s2 −19.9).
* **Both conditions met.**

## 11. IS H-CRITIC CONFIRMED OR FALSIFIED?

> **H-CRITIC is FALSIFIED (as stated).** 400 000 additional critic-only
> gradient steps on the existing replay buffer, with the actor / reward / γ /
> network / target-construction all frozen, **do NOT** eliminate the systematic
> deep-LPS Bellman under-fit and **do NOT** move the critic's action ranking
> toward the reward's — on 2 of 3 seeds the argmax moves *away* from the reward
> or not at all, the deep-LPS residual stays negative on 3/3, twin-Q
> disagreement grows on 2/3, and the offline critic loss is non-convergent
> (spikes to 18). The one partial positive (s0, well-covered band) is a single
> seed via a non-monotone path and does not reproduce.
>
> A secondary structural finding: the critic's `Q(T_CE)` is **near-degenerate
> bimodal** (an OFF lobe ≈ a deep-LPS lobe in value); under refinement the
> per-state argmax **flips between the two lobes** with no stable resolution —
> the deep-LPS value is *not identifiable* from this data.

## 12. IS H-COVERAGE PROMOTED?

> **YES.** Per the pre-registered decision rule, H-COVERAGE is promoted: the
> existing replay data is **insufficient to pin down the value of the
> high-engine-load / deep-LPS region**, so additional fitting cannot repair the
> critic there (and de-stabilises it). This holds even in the nominally
> "well-covered" 35–50 Nm band (28–54 % matched coverage), indicating the
> *informative* coverage — transitions that actually constrain the deep-LPS
> Bellman target — is thinner than the raw occupancy suggests, likely because
> the frozen actor at every next-state reverts to part-load, so the
> bootstrapped `V(s')` never reflects a *sustained* deep-LPS strategy.
>
> **Caveat (not hidden):** the offline fit is itself unstable, so H-COVERAGE is
> concluded from a noisy estimator. The pre-registered rule was designed for
> exactly this outcome (residual stays negative AND argmax doesn't move), and
> both hold on ≥ 2/3 seeds; but a *stabilised* offline critic procedure is a
> distinct (multi-variable) question this experiment was not allowed to open.

## 13. THE SINGLE NEXT EXPERIMENT JUSTIFIED BY THE EVIDENCE

H-COVERAGE is now the promoted hypothesis. The justified next experiment is a
**single-variable targeted high-engine-load exploration run** — a training run
that *adds* replay coverage of the deep-LPS / high-engine-load region so the
critic has data to identify its value, then measures whether the deep-LPS
Bellman residual closes, the critic argmax stabilises toward the reward's
preference, and vehicle fuel / CS respond.

**Proposed (NOT executed here — §10 STOP applies):**

* **Objective:** test whether adding informative coverage of feasible
  high-engine-load actions (executed `T_CE` in `[1.3·demand, 0.9·T_CE_max]`) in
  the 15–50 Nm demand band repairs the deep-LPS critic under-fit that offline
  fitting could not.
* **Single independent variable:** a training-time exploration schedule — with
  probability `p = 0.25`, when the engine is commanded ON and a higher feasible
  engine load exists, replace the sampled action with a **uniform draw from the
  feasible high-engine-load `u`-interval** (a feasibility bracket; **no ECMS
  action, no benchmark trajectory, no imitation, no warm-start**).
  `predict(deterministic=True)` untouched — evaluation-safe by construction
  (identical safeguard to Phase 6).
* **Everything else frozen** at CONTROL: plant / env / reward / `eq_factor` /
  `k_fb` / γ 0.20 / n_step 1 / entropy auto / `modeaware_gated` map / net
  [256,256] / optimiser / `τ` / replay size / `gradient_steps 16` / lookahead
  5 / evaluator / ECMS / rule-based.
* **Seeds:** 3. **Budget:** 150 000 steps (matches every prior CONTROL
  comparison). **Control:** the existing CONTROL checkpoints/buffers.
* **Primary metric:** deep-LPS Bellman residual `|Q̂ − Q_target|` at `a_R*` in
  the 15–35 Nm bands + critic per-state `argmax T_CE` and its **stability**
  across the last 50k steps.
* **Secondary metrics:** replay coverage of `T_CE ≥ 50 Nm` per band; twin-Q
  disagreement there; `V_CE_equiv` mean ± SD + per-seed; charge-sustaining
  count; ΔSoC; regional fuel (Stage-1 §6.1 table); engine `T_CE|on`, BSFC,
  on-steps; actor `T_CE` vs critic argmax; training-stability class.
* **Success criterion (H-COVERAGE confirmed):** on ≥ 2/3 seeds, coverage of
  `T_CE ≥ 50 Nm` in 15–35 Nm rises to ≥ 40 %, the deep-LPS residual at `a_R*`
  falls below 0.004 **and stays stable** over the last 50k steps, and the
  critic argmax in 30–50 Nm rises by ≥ +10 Nm toward the reward — with the
  effect present in the previously thin-coverage 25–30 / 30–35 bands.
* **Failure criterion (H-COVERAGE also falsified → policy-structure / reward
  hypothesis):** coverage rises to ≥ 40 % but the residual stays worse than
  −0.008 and/or the critic argmax moves < +3 Nm and/or remains bistable, on
  ≥ 2/3 seeds — the deep-LPS value is unidentifiable even with dense data ⇒ the
  bimodal near-degenerate Q is intrinsic (policy-parameterisation / reward-shape
  question), and neither coverage nor critic-fitting is the lever.
* **Rollback condition:** training-time flag only; the CONTROL and all `src/`
  files are untouched; outputs to `results/phase12/`. No deployment.

---

## OBJECTIVE CAVEAT (§8 — kept strictly separate)

This experiment answers **only**: *"why does the trained SAC critic fail to
reproduce its own reward's action preference in 15–35 Nm?"* → **because the
replay data does not identify the value of the high-engine-load region, and
offline fitting on it is unstable/bistable.**

It does **not** address, and no conclusion here bears on: *"why is the SAC
reward's optimum (deep LPS, `T_CE ≈ 58–78`) different from ECMS's optimum (OFF)
at these low-demand operating points?"* — V1 established that separately (the
SAC reward is a stiffer-battery Hamiltonian than ECMS). **Repairing the critic
would move the actor toward the reward's deep-LPS point, which is NOT the
ECMS-optimal action here.** Closing the ECMS gap is a distinct question.

---

## STOP

EXP-P11-S1 complete. **H-CRITIC falsified. H-COVERAGE promoted.** One next
experiment specified (single-variable targeted high-engine-load exploration,
`results/phase12/`) with pre-registered success/failure/rollback criteria —
**not executed**.

No `src/` file modified. No CONTROL artefact modified. No actor trained. No
reward / γ / `k_fb` / `eq_factor` / entropy / action-map change. No targeted
exploration run. Awaiting human approval.
