# PHASE 12 — STAGE B2: DEEP-LPS COVERAGE FALSIFICATION

> **Testing whether reward-relevant deep-LPS replay coverage is the missing condition for stable critic learning in the 25–35 Nm demand region.**

**Date/time:** 2026-08-30 · **Code revision:** git `90af969` + Stage-A
`clip_eq_eff` flag (`src/env/ems_env.py`, +28 lines, default-OFF; the **only**
`src/` change in all of Phase 12). · **Training:** 3 seeds × 150 000 env steps
(no more). · **Reward:** unchanged except the approved Stage-12A `eq_eff`
safety correction (`clip_eq_eff=True`). · **γ = 0.20**, `n_step = 1`, net
`[256,256]`, optimiser / entropy / observation / physical model / reachable
action set all frozen. `predict(deterministic=True)` untouched.

**Machine-readable:** `results/phase12/phase12_stage_b2_summary.json`.
**Pre-flight:** `results/phase12/PHASE12_STAGE_B2_PREFLIGHT_REPORT.md` (all 6
acceptance criteria PASS). **Scripts:** `te_highload_b2.py`,
`stage_b2_preflight.py`, `stage_b2_train.py`, `stage_b2_diagnostics.py`.
**Artifacts:** `stage_b2/seed{0,1,2}/{sac_ems_{50k,100k,150k,best}.zip,
replay_buffer.pkl, te_stats.json, te_events.json, coverage_evolution.json,
eval_history.csv}`, `stage_b2/{config_frozen.json, train_summary.json}`.
Phase-12A / 12B reports not overwritten.

---

## 20. REPRODUCIBILITY

| item | value |
|---|---|
| random seeds | 0, 1, 2 |
| code revision | git `90af969`; `src/env/ems_env.py` +28 lines (`clip_eq_eff` flag, default `False`) |
| config file | `results/phase12/stage_b2/config_frozen.json` |
| environment | SB3 2.8.0 · torch 2.12 CPU · numpy 2.4.6 · scipy 1.17.1 · Python 3.13.2 · `EMSEnv` (`clip_eq_eff=True`) |
| checkpoints | `results/phase12/stage_b2/seed{s}/sac_ems_{50k,100k,150k,best}.zip` |
| training steps | 150 000 / seed |
| intervention probability | `p = 0.25` |
| eligibility | `15 ≤ T_MGB < 50 Nm` ∧ `w > 0` ∧ `SoC < 0.55` ∧ feasible interval `≥ 5 Nm` wide |
| **exact action interval** | `TCE_injected ~ U( max(TCE_min_feasible, TCE_max_feasible − 15), TCE_max_feasible )`, `a` = the action whose **executed** `T_CE` equals `TCE_injected` (monotone `a→T_CE` inverted; env `_action_to_torques` is the authoritative feasible-bound source) |
| SoC cap | `0.55` |
| reward safety setting | `clip_eq_eff = True` (Stage-12A) |
| env unit tests | `pytest tests/test_ems_env.py` → **7 passed** |

**No undocumented changes.** The one code change is the Stage-A flag; the
intervention lives entirely in `results/phase12/te_highload_b2.py` and is
training-time only.

---

## 21. FINAL DECISION TABLE

CORE = 25–50 Nm demand matched-state set (20 states, 5/band; Phase-11 §11CDE
methodology). `tce_R` (reward arg-max `T_CE`, N-independent) = **63.22 Nm**.
Q-argmax stability = std of the CORE `tce_Q` across the 4 checkpoints
(50k/100k/150k/best). Deep-LPS Bellman residual = mean `Q̂ − Q_target` at
executed `T_CE ≥ TCE_max_feasible − 15 Nm`.

| Metric | CONTROL (3-seed) | B2 Seed 0 | B2 Seed 1 | B2 Seed 2 |
|---|---|---|---|---|
| Deep-LPS coverage 25–35 Nm, `frac(ρ ≥ 0.75)` | 0.32 / 0.25 (μ **0.27–0.38**) | **0.446 / 0.373** | **0.487 / 0.397** | **0.516 / 0.375** |
| Deep-LPS coverage 25–35 Nm, absolute `frac(T_CE ≥ 50)` | 0.080 / 0.114 | **0.098 / 0.195** | 0.098 / 0.206 | 0.086 / 0.187 |
| `a_R*` → `T_CE` | 63.22 | 63.22 | 63.22 | 63.22 |
| Final `a_Q*` → `T_CE` (best ckpt, CORE) | 24.5 / 32.4 / 58.7 (μ **38.5**) | **33.8** | **40.5** | **30.5** |
| `dT(R − Q)` (best) | 38.7 / 30.8 / 4.6 (μ **24.7**) | 29.4 | 22.7 | 32.7 |
| **Q-argmax stability** (`tce_Q` std over 4 ckpts) | seed spread ≈ 17 Nm; per-seed osc. 8–20 (Phase-11) | **std 2.0** [28.8, 33.8] | std 4.6 [28.9, 40.5] | **std 13.4** [4.4, 32.8] |
| Deep-LPS Bellman residual (best) | −0.011 / −0.001 / −0.014 (μ **−0.009**) | −0.010 | −0.019 | −0.039 |
| resid @ `a_R*` (best) | −0.018 / −0.004 / −0.013 (μ **−0.012**) | −0.009 | −0.023 | −0.049 |
| `Q@deep%` — critic arg-max is a deep-LPS action (best) | 0.20 / 0.33 / 0.93 (μ **0.49**) | 0.47 | 0.60 | 0.40 |
| Fuel / `V_CE_equiv` (best ckpt) | 3.6862 / 3.8431 / 3.7704 (μ **3.7666 ± 0.079**) | **3.7077** | **3.7625** | **3.8497** |
| Terminal SoC ΔSoC (pp) | +0.28 / −0.72 / +0.23 | +0.38 | −0.11 | −0.68 |
| CS satisfied | 3/3 | ✓ | ✓ | ✓ |
| Constraint violations | 0 | 0 | 0 | 0 |

**B2 fuel mean = 3.7733 ± 0.0585 (3/3 CS).**

---

## COVERAGE (§13) — THE INTERVENTION WORKED THIS TIME

`ρ_executed = (T_CE_executed − TCE_min_feasible) / (TCE_max_feasible − TCE_min_feasible)`.

| demand band | CONTROL μ `frac(ρ≥0.75)` | **B2 μ `frac(ρ≥0.75)`** | Δ | CONTROL μ `frac(T_CE≥50)` | **B2 μ `frac(T_CE≥50)`** | Δ |
|---|---|---|---|---|---|---|
| **25–30 Nm** | 0.377 | **0.483** | **+0.11** | 0.080 | **0.094** | +0.01 |
| **30–35 Nm** | 0.267 | **0.382** | **+0.11** | 0.114 | **0.196** | **+0.08** |
| 35–50 Nm | 0.316 | **0.462** | +0.15 | 0.380 | **0.510** | +0.13 |

Injection audit (`te_stats.json`, 3 seeds): **11 733 / 11 770 / 11 770**
injections; **0 clamped**; **execution fidelity = 1.000** (executed `T_CE` ==
requested `T_CE` to machine precision on every event); mean executed ρ ≈
**0.826**; executed `T_CE` reached **90 Nm** at high-demand-within-band states.
Contrast Phase-12B: 30–35 Nm `frac(T_CE≥50)` moved 0.114 → **0.116** (no
change). **B2 nearly doubled it (0.114 → 0.196).**

**⇒ Q1 = YES, Q2 = YES.** The intervention produced substantial, high-fidelity,
state-normalized deep-LPS coverage specifically in the critical 25–35 Nm demand
region. This is **not** a Case-D outcome (unlike 12B).

---

## CRITIC RESPONSE (§14–§16) — THE CRITIC DID NOT LEARN THE HIGH-LOAD BRANCH

* **`a_Q*` did NOT move toward `a_R*`.** B2 best-checkpoint CORE `tce_Q` mean
  ≈ **34.9 Nm** — *below* CONTROL's ≈ 38.5 Nm; `dT(R−Q)` ≈ 28.3 Nm — *slightly
  worse* than CONTROL's 24.7 Nm. The critic arg-max stayed on the **part-load
  branch (~30–40 Nm)**, not the reward-preferred deep-LPS branch (~63 Nm).
* **`Q@deep%` unchanged** — the critic's arg-max is a deep-LPS action in ≈ 49 %
  of matched states for both CONTROL and B2. +11 pp of deep-LPS data did **not**
  shift the critic's per-state preference.
* **Deep-LPS Bellman residual did NOT improve** — B2 best ≈ **−0.022**
  (resid @ `a_R*` ≈ **−0.027**) vs CONTROL ≈ **−0.011 / −0.012**. Both *more
  negative*; never below the 0.004 diagnostic threshold; and non-monotone
  across checkpoints (seed 2: −0.062 → +0.022 → −0.010 → −0.049).
* **Q-argmax stability — mixed and on the wrong branch.** Seed 0's CORE `tce_Q`
  stabilised to **std 2.0 Nm** across all four checkpoints (28.8–33.8) — the
  first stable Q-argmax seen in this project — but it stabilised at **~32 Nm
  (part-load)**, not toward the reward's 63 Nm. Seed 1: std 4.6 (partial).
  Seed 2: **std 13.4**, still flipping between the OFF lobe (≈ 4–5 Nm) and
  part-load (≈ 33 Nm). So: 1–2/3 seeds gained stability, but on the wrong lobe;
  1/3 remained fully bistable.

**⇒ Q3 = NO** (arg-max did not move toward the reward optimum).
**Q4 = PARTIAL** (1–2/3 seeds stabilised, but on the part-load branch, not
toward the reward). **Q5 = NO** (deep-LPS residual did not improve; it
slightly worsened). **Q6 = N/A** — the critic did not improve toward the
reward, so there was nothing for the actor to follow; the actor tracked the
unchanged part-load critic (`tce_pi` ≈ 34 Nm, same as CONTROL).

---

## VEHICLE (§18) & SoC

| run | `V_CE_equiv` | ΔSoC pp | CS | viol | OFF % | LPS % | engine-on s |
|---|---|---|---|---|---|---|---|
| CONTROL s0 | 3.6862 | +0.28 | ✓ | 0 | 39.8 | 27.9 | 376 |
| CONTROL s1 | 3.8431 | −0.72 | ✓ | 0 | 38.2 | 32.9 | 390 |
| CONTROL s2 | 3.7704 | +0.23 | ✓ | 0 | 37.8 | 28.6 | 393 |
| **B2 seed 0** | **3.7077** | +0.38 | ✓ | 0 | 42.2 | 29.9 | 355 |
| **B2 seed 1** | **3.7625** | −0.11 | ✓ | 0 | 37.5 | 25.4 | 396 |
| **B2 seed 2** | **3.8497** | −0.68 | ✓ | 0 | 37.4 | 32.2 | 397 |
| **B2 mean ± SD** | **3.7733 ± 0.0585** | — | **3/3** | 0 | ≈ 39 | ≈ 29 | ≈ 383 |
| advanced rule-based | 3.5056 | — | — | — | 59.0 | 23.8 | 209 |
| ECMS | 3.1887 | — | — | — | 53.1 | 29.7 | — |

**B2 fuel is statistically tied with CONTROL** (3.7733 vs 3.7666; both within
the CONTROL seed SD of ±0.079). **3/3 charge-sustaining** (better than 12B's
2/3 — the `SoC < 0.55` cap helped the *deterministic* policy). Mode split and
engine-on time essentially unchanged. Still +7.6 % over rule-based, +18.4 %
over ECMS.

**⇒ Q7 = NEITHER** (no material improvement or worsening; +0.007, within
noise). **Q8 = YES** — charge-sustaining held 3/3.

---

## DUAL-REWARD AUDIT (§19) — STAGE-A CORRECTION NON-CONTAMINATING

| seed | SoC_max (eval) | % transitions above 60.868 % | cum `R_patched` | cum `R_original` | **ΔR** | n affected |
|---|---|---|---|---|---|---|
| 0 | 50.38 % | 0.00 % | −78.5512 | −78.5512 | **0.000000** | **0** |
| 1 | 49.99 % | 0.00 % | −63.7134 | −63.7134 | **0.000000** | **0** |
| 2 | 50.41 % | 0.00 % | −51.5649 | −51.5649 | **0.000000** | **0** |

On all 3 evaluated controllers: SoC_max ≤ 50.4 %, no transition above the old
inversion threshold, `ΔR = 0.0` to the bit. **The Stage-A correction did not
influence the experiment at evaluation.** During *training* it was load-bearing
(replay SoC > 60.868 % in **1.26 % / 0.66 % / 1.27 %** of transitions per seed;
max training SoC 71.8 % / 67.2 % / 80.9 % — the LPS/charging injections raise
SoC even though injection is gated at SoC < 0.55). Without `clip_eq_eff=True`
those ~1000–1900 transitions/seed would have fed the critic an inverted
`eq_eff` down to ≈ −0.6. **Case E is ruled out; the correction did its job.**

---

## 22. FINAL SCIENTIFIC CLASSIFICATION (§17 / §22)

| Q | answer |
|---|---|
| **Q1. Did the intervention actually produce the intended deep-LPS coverage?** | **YES.** 11.7k injections/seed, fidelity 1.000, 0 clamped, mean executed ρ ≈ 0.83, reaching `T_CE` 90 Nm. `frac(ρ ≥ 0.75)` in 25–35 Nm: +0.11; absolute `frac(T_CE ≥ 50)` in 30–35 Nm: 0.114 → 0.196 (nearly doubled). |
| **Q2. Was coverage achieved specifically in the critical 25–35 Nm demand region?** | **YES.** Both critical bands gained ≈ +11 pp state-normalized deep coverage and the 30–35 Nm band gained +8 pp absolute `T_CE ≥ 50`. |
| **Q3. Did the critic's arg-max move toward the known instantaneous reward optimum?** | **NO.** Best-checkpoint CORE `tce_Q` ≈ 34.9 Nm (part-load), *below* CONTROL's 38.5; `dT(R−Q)` unchanged/slightly worse (28.3 vs 24.7). `Q@deep%` unchanged at ≈ 49 %. |
| **Q4. Did the critic arg-max stabilize?** | **PARTIAL / MISLEADING.** Seed 0: `tce_Q` std 2.0 Nm across all 4 checkpoints (genuinely stable — a first for this project). Seed 1: std 4.6 (partial). Seed 2: std 13.4 (still bistable OFF ↔ part-load). And every stable value is on the **part-load** branch, not the reward branch. |
| **Q5. Did the deep-LPS Bellman residual improve?** | **NO.** B2 best ≈ −0.022 (resid @ `a_R*` ≈ −0.027) vs CONTROL ≈ −0.011/−0.012 — *more negative*; never < 0.004; non-monotone across checkpoints. |
| **Q6. Did the actor follow the improved critic?** | **N/A** — the critic did not improve toward the reward. The actor tracked the unchanged part-load critic (`tce_pi` ≈ 34 Nm). |
| **Q7. Did vehicle fuel economy improve or worsen?** | **NEITHER.** 3.7733 ± 0.0585 vs CONTROL 3.7666 ± 0.079 — statistically tied. |
| **Q8. Did charge-sustaining remain valid?** | **YES, 3/3** (better than 12B's 2/3; the SoC < 0.55 injection cap helped the deterministic policy). |
| **Q9. Does the evidence support or falsify H-COVERAGE?** | **FALSIFIES H-COVERAGE as the primary explanation.** Substantial, high-fidelity, state-normalized deep-LPS coverage was delivered in the critical 25–35 Nm region (Q1/Q2 = YES), and the critic **still** did not move its arg-max toward the reward optimum (Q3 = NO), did not close the deep-LPS Bellman residual (Q5 = NO, slightly worse), and produced no policy or vehicle response (Q6/Q7). This is **§17 Outcome 2 — COVERAGE REJECTED.** |
| **Q10. Single most justified next experiment** | **A critic/policy *representation* change that breaks the part-load ⇄ deep-LPS Q-degeneracy — NOT another coverage or fitting experiment.** See below. |

### Why H-COVERAGE is now falsified (the causal chain, §24)

```
deep-LPS coverage      ✓  (Q1/Q2: +11 pp in 25–35 Nm, fidelity 1.0, T_CE→90)
        │
critic learning        ✗  (Q3: a_Q* stayed at part-load ~35 Nm, not toward 63 Nm;
        │                   Q@deep% unchanged at 49%)
        │
Q stabilization        ~  (Q4: 1–2/3 seeds stabilized — but on the WRONG (part-load) lobe)
        │
policy response        ✗  (Q6: actor unchanged, ~34 Nm)
        │
vehicle outcome        ✗  (Q7: tied)
```

**The chain breaks at "critic learning".** Three independent interventions have
now each supplied the critic with data or fitting effort for the deep-LPS
region and **none moved the critic's preference toward it**:

1. **Phase-11 EXP-P11-S1** — 400k offline critic gradient steps on the frozen
   replay buffer → deep-LPS Bellman residual stayed ≈ −0.02, arg-max bistable.
2. **Phase-12B** — online mid-load coverage → 30–35 Nm `frac(T_CE ≥ 50)`
   0.114 → 0.116 (no change); critic unchanged.
3. **Phase-12B2** (this) — online, state-normalized, fidelity-1.0 deep-LPS
   coverage in the critical bands (+11 pp) → arg-max still ~35 Nm part-load,
   residual ≈ −0.022, `Q@deep%` unchanged.

The `Q(T_CE)` surface has a **near-degenerate bimodal structure** (a part-load
lobe ≈ a deep-LPS lobe in value); with `γ = 0.20` the future term is action-flat
(Phase-11 §11D), so the two lobes stay close, and adding data at the deep-LPS
lobe does not tip the critic to prefer it. This is a **critic
function-approximation / representation limitation**, not a data limitation.

### §22 Q10 — the single most justified next experiment (RECOMMENDATION ONLY; NOT run)

**A mode-conditioned value/policy representation, tested single-variable against
CONTROL, 3 seeds, 150k steps, everything else frozen (γ, reward, `eq_factor`,
`k_fb`, net width, optimiser, observation, action bounds).** Concretely: a
**discrete engine-mode head** (OFF / part-load / deep-LPS, from the sign and
magnitude of the split) with a continuous within-mode action, and a critic that
is **conditioned on the discrete mode** (or a small distributional / 2-head
critic), so the value function represents the two lobes as *separate* branches
that the policy commits to — instead of a single smooth `Q(s,a)` that averages a
near-degenerate bimodal surface. This directly targets the measured failure
(the critic cannot distinguish / prefer the deep-LPS lobe even with data there).

* **Success:** on ≥ 2/3 seeds the critic's mode-preference in the 25–35 Nm band
  shifts toward the deep-LPS branch, `tce_Q` rises ≥ +10 Nm toward the reward,
  the deep-LPS residual falls below 0.004 **and is stable** over the last 50k,
  and the actor follows — with CS held 3/3.
* **Failure:** the mode-conditioned critic *also* fails to prefer the deep-LPS
  branch given the data — then the limiter is upstream (the `γ = 0.20` /
  costate formulation making the two branches genuinely equal-value), and the
  next lever is the temporal/objective structure, not representation.

Phase-8C tried a 2-component *actor* mixture (it collapsed) — but that was with
the **unchanged smooth critic**; this recommendation changes the **critic
representation**, which is where B2 localises the failure.

---

## LIMITATIONS

1. **150k-step budget** (per §14; consistent with every prior comparison). Seed
   0's newly-stable Q-argmax (std 2.0) could conceivably drift with more steps;
   not tested (§23 forbids scaling).
2. **Single cycle (NEDC).**
3. **20-state critic-diagnostic set** (5/band) — the bistability and
   "no move toward the reward" signals are unambiguous across checkpoints and
   seeds, but per-band means carry small-sample noise.
4. **Coverage is *executed*-`T_CE` occupancy** with a state-normalized ρ
   criterion — a proxy for whether the transitions actually constrain the
   deep-LPS Bellman target. Phase-11 §11CDE argued the *informative* content is
   thinner than raw occupancy because the slowly-moving actor reverts to
   part-load at every next-state, so the bootstrapped `V(s')` never reflects a
   *sustained* deep-LPS strategy. B2 does not remove that confound (it changes
   only the sampled action, not the subsequent trajectory) — but it is the
   cleanest coverage test available without changing the policy structure, and
   3/3 fidelity-1.0 targeting in the critical bands is as strong a coverage
   intervention as this action space allows.
5. **`clip_eq_eff` was active** (Stage-A). It is a verified bitwise no-op on all
   evaluated transitions (ΔR = 0.0, 3/3) but was load-bearing in 0.7–1.3 % of
   training replay; the experiment therefore ran on the corrected reward, which
   is the intended, approved condition — not the pre-12A reward.

---

## §24 INTERPRETATION RULE — COMPLIANCE

* The claim **"the critic did not learn deep-LPS"** is made **only because** the
  experiment first demonstrated (Q1/Q2, fidelity 1.0, +11 pp) that the critic
  *was* supplied with substantial deep-LPS data in the critical region.
* No claim is made that "exploration solved the problem" — coverage increased
  and the downstream links (critic → stabilization → policy → vehicle) did
  **not** follow. The chain break is identified precisely: **critic learning**
  (Q3/Q5).

---

## CONCLUSION — PASS / FAIL / INCONCLUSIVE

**Stage B2: the coverage intervention PASSED (Q1/Q2 — substantial, high-fidelity
deep-LPS coverage delivered in the critical 25–35 Nm region), and H-COVERAGE is
consequently FALSIFIED as the primary explanation for the critic/policy failure
(§17 Outcome 2 — COVERAGE REJECTED).** The critic, given the data, did not move
its arg-max toward the known reward optimum (Q3), did not close the deep-LPS
Bellman residual (Q5, slightly worse), stabilised on the *wrong* (part-load)
branch on 1–2/3 seeds (Q4), and produced no policy or vehicle response
(Q6/Q7). Vehicle fuel is statistically tied with CONTROL (3.7733 ± 0.059 vs
3.7666 ± 0.079), charge-sustaining 3/3 (Q8). The Stage-A `eq_eff` correction was
non-contaminating at evaluation (ΔR = 0.0, 3/3) and load-bearing during
training (Case E ruled out).

Together with Phase-11 EXP-P11-S1 (offline) and Phase-12B (mid-load online),
**three independent coverage/fitting interventions now fail to move the
critic** — the near-degenerate bimodal `Q(T_CE)` surface is an **intrinsic
critic-representation limitation**, not a replay-coverage deficiency. The next
justified experiment is a **mode-conditioned critic/policy representation**
change (recommendation only — §23).

---

## §23 HARD STOP — OBSERVED

Final report produced. **No further experiment run. No 1M-step run, no γ change,
no action reparameterisation, no `eq_factor` / `k_fb` tuning, no hybrid policy
implemented, no additional exploration.** The next experiment (mode-conditioned
representation) is *recommended only* and requires human authorisation.

Awaiting human review.
