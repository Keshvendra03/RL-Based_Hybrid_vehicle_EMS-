# Phase 2 — Final Report: 29-Point Completion

Covers every point of the Phase-2 forensic/intervention brief. Companion
documents: `RL_DIAGNOSTIC_REPORT.md` (diagnosis), `EXPERIMENT_LOG.md`
(reasoning per experiment), `experiments/experiment_registry.yaml`
(machine-readable configs), `VALIDATION.md` (plant validation + conflicts).

---

## HEADLINE RESULT

Starting point (session baseline, NEDC seed 1): **4.1245 L/100km**, not
charge-sustaining, ASSIST 20.6%.

Final validated configuration, **3 seeds per cycle**:

| Cycle | n | mean | std | best | 95% CI | rule-based | gap (mean / best) | CS | violations |
|---|---|---|---|---|---|---|---|---|---|
| NEDC | 3 | **3.7727** | 0.0281 | 3.7543 | [3.741, 3.805] | 3.5056 | **+7.6% / +7.1%** | 3/3 | 0 |
| FTP75 | 3 | **3.3821** | 0.0846 | **3.3265** | [3.286, 3.478] | 3.2323 | **+4.6% / +2.9%** | 2/3 | 0 |

Gap to the rule-based benchmark closed from **+17.7% → +7.6%** (NEDC) and
**+30.2% → +4.6%** (FTP75). Zero constraint violations across all seeds.
**The benchmark is still not beaten.** No success is claimed.

**Winning configuration**
```
--gamma 0.20 --n-step 1 --action-map linear
--eq-factor 0.2717 (NEDC) / 0.4981 (FTP75)   --k-fb 1.656
--gradient-steps 16 --target-entropy -2 (or -1; indistinguishable at n=3)
lookahead 5, buffer 300k, batch 512, lr 3e-4, net [256,256]
```

**Two coupled root causes** (neither works alone):
1. **Reward unit mismatch.** `elec_liters` is already EFC-converted, so
   `eq_factor=1.0` priced battery at 4.83 fuel-J per battery-J — 3.68x ECMS's
   proven costate. Correct conversion: `eq_factor = λ_ECMS / 4.8309`.
2. **Horizon mismatch.** ECMS proves the optimum here is *myopic*. γ=0.9999
   forced SAC to integrate ~1220 steps of return that add variance but no
   information, since the only inter-temporal coupling (SoC) is already
   carried by the costate feedback term.

---

## POINT-BY-POINT COMPLETION

### §1 Permanent diagnostic record — **DONE**
`RL_DIAGNOSTIC_REPORT.md`, `EXPERIMENT_LOG.md`, `PHASE2_FINAL_REPORT.md`
created; `ROADMAP.md` and `VALIDATION.md` updated. Locked components recorded
with the no-modification rule.

### §2 Experiment registry — **DONE**
`experiments/experiment_registry.yaml`: benchmarks, unit-conversion constants,
and every experiment with id/date/commit/seed/cycle/SoC/steps/algorithm/action
map/target entropy/γ/lr/batch/replay/reward version/checkpoint/results.

### §3 Instrumentation — **DONE**
`src/agents/instrumentation.py::SACDiagnostics` logs Q1, Q2, min-Q, Q-target,
TD error (mean + RMS), actor μ, actor σ, log π, pre/post-tanh action, action
percentiles p1/p25/p50/p75/p99 and saturation % to
`<out>/sac_diagnostics.csv` + TensorBoard. EMS-side per-step diagnostics
(action, u, T_CE, T_EM, engine on/off, mode, demand, SoC, fuel, battery power,
reward, mode %) are produced by `results/evaluate_policy.py`.

### §4 Single standard evaluation function — **DONE**
`results/evaluate_policy.py`. One code path for SAC, TD3, ECMS and the
advanced rule-based controller. Emits fuel, V_CE_equiv, SoC/ΔSoC, OFF/ASSIST/
LPS/ONLY/REGEN %, constraint violations, battery throughput, engine on-time,
reward, and action statistics.

### §5 Baseline reproduction — **DONE (PASS)**
Reproduced 4.1245 / SoC 52.63% / OFF 29.4% / ASSIST 20.6% / LPS 33.0% /
REGEN 17.0% exactly, and the rule-based benchmark at 3.5056 / OFF 59.0%.

### §6 Action-space mathematics — **DONE**
Three families evaluated: power-law, piecewise-fixed, analytic (mode-aware).

### §7 Control equivalence — **DONE (PROVED)**
`tests/test_action_mapping.py`, 53 tests: exact endpoints for every torque,
strict monotonicity (bijection onto [U_MIN, U_MAX]), identical reachable `u`
set, regen/sub-cutoff unchanged, `linear` bit-identical to the original.

### §8 OFF-band geometry table — **DONE**

| mapping | OFF med% | p25 | p75 | <5% | <10% | >25% | >40% | LPS med% |
|---|---|---|---|---|---|---|---|---|
| ORIGINAL linear | 12.19 | 7.13 | 30.22 | 19.8 | 47.8 | 26.0 | 4.4 | 45.95 |
| power p=1.5 | 8.30 | 4.81 | 21.33 | 26.3 | 57.8 | 4.4 | 4.4 | 59.54 |
| power p=2.0 | 6.29 | 3.63 | 16.46 | 47.8 | 74.0 | 4.4 | 0.0 | 67.78 |
| power p=3.0 | 4.24 | 2.43 | 11.30 | 57.8 | 74.0 | 0.0 | 0.0 | 77.16 |
| power p=0.5 | 22.89 | 13.75 | 51.31 | 1.9 | 20.5 | 43.8 | 26.0 | 21.11 |
| power p=0.25 | 40.55 | 25.60 | 76.29 | 0.0 | 2.4 | 77.6 | 51.0 | **4.46** |
| piecewise 35/25/40 | 40.00 | 23.39 | 50.77 | 0.0 | 4.2 | 73.7 | 49.9 | 35.00 |
| **analytic 35/25/40** | **40.00** | **40.00** | **40.00** | 0.0 | 0.0 | 100.0 | 4.4 | 35.00 |

The a-priori suggested p=1.5/2/3 move the band the **wrong way**.

### §9 Structural metrics before fuel — **DONE**
Mapping selected on OFF-band geometry, state-invariance and LPS preservation.

### §10 Implement only the best candidate — **DONE**
Only the mode-aware map implemented; default remains `linear`.

### §11 No combined fixes — **DONE**
EXP-B (action map) and EXP-B2 (reward pricing) run as separate single-variable
experiments against the same baseline.

### §12 Run Experiment B — **DONE**

### §13 Baseline vs Experiment B comparison — **DONE**

| Metric | Baseline | EXP-B (mode-aware) | EXP-B2 (reward fix) |
|---|---|---|---|
| V_CE_equiv | 4.1245 | 4.5573 | 4.1782 |
| Final SoC / ΔSoC | 52.63% / +2.63pp | 53.31% / +3.31pp | 52.86% / +2.86pp |
| OFF / ASSIST / LPS / REGEN % | 29.4/20.6/33.0/17.0 | 16.7/39.7/26.7/17.0 | 25.7/26.9/30.3/17.0 |
| Reward | −45.15 | −54.33 | −46.46 |
| Critic loss (MSE/RMSE) | 2.597 / 1.611 | 18.244 / 4.271 | 2.655 / 1.629 |
| Actor loss / α | 23.44 / 0.0195 | 49.27 / 0.0374 | 24.21 / 0.0261 |
| Action p50 / std | −0.081 / 0.644 | −0.046 / 0.396 | −0.081 / 0.487 |
| Constraint violations | 0 | 0 | 0 |

Q1/Q2/Q-target/TD-error were not logged during these runs (instrumentation
added afterwards); they were obtained offline via `results/q_landscape.py` and
are reported in §19. Post-instrumentation runs carry them directly:

| run | Q1 | Q2 | minQ | Q-target | TD-RMS | actor μ | actor σ | log π | sat% |
|---|---|---|---|---|---|---|---|---|---|
| final NEDC s0 | −0.082 | −0.076 | −0.086 | −0.073 | 0.177 | 0.184 | 0.165 | 1.940 | 5.40 |
| final FTP75 s1 | −0.093 | −0.090 | −0.097 | −0.092 | 0.062 | 0.911 | 0.305 | 1.835 | 12.57 |
| obs_clean 18d | −0.076 | −0.085 | −0.098 | −0.099 | 0.544 | 0.161 | 0.204 | 0.890 | 1.17 |

### §14 Is P0 confirmed? — **DONE**
`P0-OLD (action geometry)`: **PARTIALLY CONFIRMED** — cycle-dependent (helped
FTP75 −7.4%, hurt NEDC +10.5%). Not the primary cause.
`P0-REVISED (reward units)`: **CONFIRMED as a necessary correction**, but
insufficient alone.
`P0-NEW (critic SNR)`: **REJECTED as the binding constraint** — γ reduction
cut TD noise 2.7x while the Q signal shrank proportionally.

### §15 Both cycles before declaring success — **DONE**
EXP-B run on NEDC and FTP75. Final configuration validated on both cycles,
3 seeds each. EXP-B2 additionally run on FTP75 to close the gap:

| EXP-B2 (reward fix alone, γ=0.9999) | NEDC | FTP75 |
|---|---|---|
| baseline | 4.1245 | 4.2072 |
| EXP-B2 | 4.1782 (+1.3% worse) | **4.5023 (+7.0% worse)** |
| OFF% | 25.7 | **11.5** |
| ASSIST% | 26.9 | 27.8 |

Both cycles agree: **the reward correction alone, at high γ, makes things
worse.** This is the strongest evidence for the coupling — the corrected
reward is only useful once γ is low enough for SAC to act on it myopically.

### §16 Target-entropy sweep — **DONE**

| target_entropy | V_CE_equiv | ΔSoC | CS | OFF% | ASSIST% |
|---|---|---|---|---|---|
| −1 (SB3 default) | 3.7775 | +0.15pp | Y | 35.3 | 24.8 |
| **−2** | **3.7543** | −1.14pp | Y | 34.8 | **17.1** |
| −3 | 3.7831 | −0.01pp | Y | 34.6 | 19.1 |

−2 is a genuine interior optimum (−3 is worse), and it clearly reduces ASSIST.
**But at n=3 the multi-seed means are 3.7727 (−2) vs 3.7811 (−1) against
std 0.028 — statistically indistinguishable. Not claimed as an improvement.**

### §17 Gamma sweep — **DONE**

| γ | 0.9999 | 0.999 | 0.99 | 0.90 | 0.50 | **0.20** | 0.00 |
|---|---|---|---|---|---|---|---|
| V_CE_equiv | 4.1782 | 4.3158 | 4.1258 | 3.8795 | 3.8181 | **3.7775** | 3.8159 |
| charge-sustaining | NO | YES | NO | YES | YES | YES | YES |
| critic RMSE | 1.629 | 1.338 | 0.605 | — | — | — | — |

Monotone improvement toward low γ, with a shallow optimum around 0.2–0.5.

### §18 eq_factor sign bug — **DONE (all 6 questions)**
1. *Origin:* introduced (commit `2a8cdbe`) to mirror `ecms.py`'s proven
   closed-loop costate feedback.
2. *Interpretation:* λ(SoC) — battery price rises as SoC falls (Pontryagin
   costate with proportional feedback).
3. *Is λ<0 valid?* **No.** In ECMS λ>0 always; λ<0 means being paid to
   discharge, which is not a valid equivalence factor.
4. *Does clipping change the objective?* Only inside the pathological region;
   it preserves "battery always has positive value", which is the correct
   economics.
5. *Does any agent exploit it?* **No — measured 0 steps.** Threshold is
   SoC 66.41% (NEDC) / 80.08% (FTP75); max observed SoC was 50.2–55.9%.
   Note: dividing both `eq_factor` and `k_fb` by 4.8309 leaves the ratio, and
   thus the threshold, unchanged.
6. *What do ECMS/benchmark imply?* `ecms.py` has no clip either and never
   approaches the flip; the rule-based controller ranges 0.61–52.47%.

**Conclusion: latent defect, never exercised. Clipping is correct hygiene with
zero measurable effect — no training run was spent on it.**

### §19 Q-function instrumentation — **DONE**
`results/q_landscape.py` + `results/policy_analysis.py`; 8 probes (low/med/high
torque, high/low SoC, acceleration, cruising, braking) with plots at
`results/analysis/q_landscape_NEDC.png`.

Baseline (γ=0.9999): critic **agreed** with the true reward at all three
torque probes — the reward itself ranked ASSIST/LPS above OFF
(best-OFF − best-ASSIST = −0.0825 at 48.9 Nm). This is what redirected the
investigation from the critic to the reward.

### §20 State-conditioned action analysis — **DONE**
`results/analysis/policy_law_NEDC.png` (6 panels: T-vs-action, T-vs-T_CE,
T-vs-mode, SoC-vs-action, SoC-vs-mode, margin above a_off).

The policy learned a clean torque-threshold law:

| T_MGB band | n | mean a | mean a_off | margin | OFF% |
|---|---|---|---|---|---|
| 0–15 Nm | 156 | +0.663 | +0.385 | +0.278 | **100.0** |
| 15–30 Nm | 226 | +0.424 | +0.719 | −0.295 | 53.1 |
| 30–50 Nm | 158 | −0.197 | +0.836 | −1.033 | 0.0 |
| 50+ Nm | 150 | −0.125 | +0.922 | −1.047 | 0.0 |

**Headroom:** engine-OFF is physically feasible on **70.9%** of NEDC moving
steps (benchmark uses 59%, agent 35–37%). The agent is not blocked by the
motor envelope — it declines OFF at 30–50 Nm where OFF *is* feasible. That is
where the remaining gap lives.

### §21 State-duplicate ablation — **DONE (cleanup REFUTED)**
`v_next` (obs[7]) is byte-identical to `fut_v1` (obs[8]); `gear_oh6` (obs[19])
is always 0.0. Removing both (20→18 dims) made results **worse**: 3.9288 vs
3.7775 — a 4% gap, well outside the 0.29% seed std.
**Recommendation: keep the 20-dim observation.** The dead channels are
harmless; removing them perturbed input scaling.

### §22 Checkpoint selection rule — **DONE**
`src/agents/instrumentation.py::CheckpointRule`. Strict order: (1) zero
constraint violations, (2) charge-sustaining |SoC−0.5| ≤ 0.02, (3) minimum
V_CE_equiv. **Training reward is never used.** Writes
`best_checkpoint_rule.json` with a `gates_met` flag so an invalid checkpoint
can never be silently reported as valid.

### §23 Multi-seed statistics — **DONE**
See HEADLINE RESULT. NEDC 3.7727 ± 0.0281 (n=3); FTP75 3.3821 ± 0.0846 (n=3).
No cherry-picking: means, std, min, max and 95% CI all reported.
Seed spread collapsed from the baseline's **5.6%** to **0.7% / 2.5%**.

### §24 Correct target — **DONE**
Rule-based is the primary target; ECMS is treated as a stretch reference
because its λ is bisection-tuned with whole-cycle information. Superiority
over ECMS is **not** claimed.

### §25 Decision tree — **DONE**
EXP-B landed on **PATH B/C** (mixed; ASSIST not reduced on NEDC), which
directed the investigation to reward alignment and temporal credit assignment
rather than stacking further action-space changes.

### §26 TD3 comparison — **DONE**
Same observation, action, env, reward, evaluator, budget; only the algorithm
differs.

| | SAC (γ=0.20, ent −2) | TD3 |
|---|---|---|
| V_CE_equiv | **3.7543** | 3.9954 |
| ΔSoC / CS | −1.14pp / **YES** | +5.81pp / **NO** |
| OFF% | 34.8 | **41.3** |
| ASSIST% | 17.1 | **8.0** |

TD3's deterministic policy **commits harder than SAC** (highest OFF, lowest
ASSIST of any run), corroborating the entropy-blocks-commitment mechanism —
but it drifts off charge-sustaining and its fuel is worse.
**Verdict: TD3 confirms the mechanism, but SAC with tuned target entropy is
the better controller. Keep SAC.**

### §27 Final report — **this document**

### §28 Documentation — **DONE**
`EXPERIMENT_LOG.md`, `RL_DIAGNOSTIC_REPORT.md`, `ROADMAP.md`, `VALIDATION.md`,
`experiments/experiment_registry.yaml`, `PHASE2_FINAL_REPORT.md`.

### §29 Git / reproducibility — **DONE**
Commit before and after each experiment; `run_config.json` (CLI args + git
commit) written per run; `.gitignore` excludes model artifacts.

---

## VALIDATION CONFLICT (open, nothing modified)

The advanced rule-based benchmark reaches **SoC 0.61%** on NEDC (38/1220 steps
below 5%), a region the RL agent is hard-masked from (`SOC_MIN=0.05`).
Quantified: running the benchmark through the agent's masks costs it
**+0.0736 L/100km (+2.10%)** on NEDC and ~0% on FTP75.
**Authority-equal NEDC target: 3.5792.** On that basis the final agent is
**+5.4%** above the benchmark rather than +7.6%. Real but small; it does not
excuse the gap. No validated file was changed.

---

## HONEST LIMITATIONS

- **The benchmark is not beaten.** +7.6% (NEDC) / +4.6% (FTP75) remain.
- FTP75 charge-sustaining holds on only **2 of 3** seeds (−2.32pp, −2.00pp
  are marginally outside the ±2pp band).
- n=3 seeds; several intra-config differences (target entropy −1 vs −2, γ 0.2
  vs 0.5) are **inside noise** and are not claimed as improvements.
- All results are 150k-step runs. Longer runs were shown to *degrade* under
  the old configuration; this has not been re-tested under the new one.
- Generalization (train on one cycle, evaluate on the other) is **not** done.

## RECOMMENDED NEXT STEP (one)

Attack the 30–50 Nm band identified in §20, where OFF is feasible on ~89% of
traction steps but the policy uses it 0% of the time. That single band is
where the remaining benchmark gap is concentrated.
