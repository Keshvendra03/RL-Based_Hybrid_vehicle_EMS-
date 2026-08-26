# Verified Facts Ledger

**Purpose:** a single lookup table of facts, numbers, and behaviors that
have already been confirmed. Before re-running a validation check,
re-deriving a constant, or re-measuring a baseline, **search this file
first.** If it's here with a date and a "how confirmed" line, trust it —
don't redo the work.

**Rules for keeping this file honest:**
- When you verify something new, **append** a row/entry with the date and
  the exact command or method used. Don't restate detail already logged in
  `VALIDATION.md` — link to it instead.
- Never silently delete or edit an entry because it turned out to be wrong.
  Mark it `SUPERSEDED — see <new entry>` and add the corrected entry below
  it. The audit trail matters more than tidiness.
- §A–§D are **permanent** facts (physics, benchmarks, infra behavior) —
  they don't change unless the underlying code/model changes.
- §E is **state-dependent** — snapshots of the current RL training runs.
  These will be superseded as training progresses; each entry is dated and
  tied to a specific step count / run directory, not a permanent truth.

---

## A. Physics constants & sign conventions (locked, Phase 1)

Full detail and how each was confirmed: `VALIDATION.md` ("What was
validated", items 1–9).

| Fact | Value | Source |
|---|---|---|
| Air density `rho` | 1.18 kg/m³ (not the common 1.2) | `README.md`, `VALIDATION.md` |
| Cold-start factor `k_cs` | 1.15, applied to final `V_liter` | `README.md`, `VALIDATION.md` |
| Engine idle power `P_CE_idle` | 8000 W when `0 < w_gear <= w_idle`; `0` at true standstill (`w_gear == 0`) — not the consumption-map value | `README.md` |
| Engine consumption map indexing | `(w_CE_row, T_CE_col)` = (speed, torque) directly — **no** mean-effective-pressure conversion | `README.md`, `VALIDATION.md` item 3 |
| Battery sign convention | Discharge (`p_bt > 0`) correctly decreases `q_bt` | `VALIDATION.md` item 1 |
| Controller torque-split identity | `T_CE + T_EM == T_MGB` holds exactly for all tested `(w_MGB, dw_MGB, T_MGB)` | `VALIDATION.md` item 5 |
| Energy conservation | Symmetric discharge/regen battery cycle nets to zero or slightly negative — never net-charges | `VALIDATION.md` item 6 |
| Gearbox torque conversion | `t_mgb` requires division by total gear ratio `i_gt` (wheel→flywheel side), plus efficiency/friction terms | `README.md` |
| `w_wheel` computation | From *averaged* speed `v_a = 0.5*(v[t]+v[t-1])`, not instantaneous `v[t]` | `README.md` |
| `dv` / `x_tot` | Both recomputed locally (backward diff / trapezoidal integration of `v_a`) — the CSV columns of the same name do **not** match and should not be used directly | `README.md` |
| Driving-cycle CSV sample count | Requires one extra trailing row (`time_s=length+1, v=0, dv=0, gear=0`) to match MATLAB — 1221 samples for NEDC vs. 1220 "real" rows | `README.md` |
| Gearbox / vehicle dynamics numeric match vs. MATLAB | `w_mgb`, `t_mgb`, `w_wheel`, `T_wheel` match at t=53,54,55,65,66,102,103 to ~0.01% | `VALIDATION.md` items 7–8 |
| Full end-to-end chain match vs. MATLAB | Every intermediate signal matches at t=53,54,102, no mismatches | `VALIDATION.md` item 9 |

**Do not re-run the deleted one-off diagnostic scripts to re-check any of
the above** — they're gone on purpose (see `VALIDATION.md` "Removed
scripts"); the numbers above are final.

---

## B. Benchmark numbers — the targets Phase 4 must beat (locked)

| Cycle | Controller | Fuel (L/100km) | Equivalent fuel (L/100km) | Source |
|---|---|---|---|---|
| NEDC | Simple rule-based, Python | 4.535 | 4.535 | `README.md` (MATLAB: 4.513, ~0.5% diff) |
| NEDC | Advanced rule-based, Python | **3.506** | — | `README.md`, `VALIDATION.md` (MATLAB: 3.348; 22.69% reduction vs. 25.81% MATLAB — offset is systematic, doesn't invalidate relative comparisons) |
| NEDC | ECMS (stretch target) | — | 3.1887 | `models/NEDC/eval_history.csv` (embedded reference column) |
| FTP75 | Advanced rule-based, Python | **3.232** | — | `models/FTP75/eval_history.csv`, `models/FTP75/run_config.json` context |
| FTP75 | ECMS (stretch target) | — | 2.8097 | `models/FTP75/eval_history.csv` |

Note: the FTP75 baseline (3.232 / 2.8097) is confirmed via its consistent
use as the reference column across the entire `eval_history.csv` (46 rows,
every training eval), but — unlike NEDC — it has never been written up
narratively in `VALIDATION.md`/`README.md` with a MATLAB cross-check. Treat
it as confirmed-by-pipeline-consistency, not MATLAB-cross-validated, until
someone runs `python -m src.evaluate_advanced --cycle FTP75` and logs it
properly.

Also confirmed: final battery charge `Q_BT` on NEDC baseline — Python
18,950 As vs. MATLAB 17,840 As (~6.2% diff, systematic, doesn't affect
strategy-vs-strategy comparisons). `README.md`.

---

## C. Pipeline / infrastructure behavior (confirmed, Phase 3 audit)

Full detail: `VALIDATION.md` ("Phase 3 — RL pipeline audit & fixes").

| Fact | How confirmed |
|---|---|
| `ems_env.py` (post-merge) reproduces old 16-dim behavior exactly at `lookahead=0` | Byte-diff of pre-merge `ems_env.py` vs. `ems_env_lookahead.py` (only lookahead block differed) + `test_ems_env.py` 4/4 pass |
| `evaluate_rl.py` / `mode_breakdown_rl.py` auto-infer lookahead window from checkpoint | Ran both against a real lookahead-trained checkpoint post-fix; produced output instead of crashing |
| `best_score.txt` matches its adjacent `.zip` | Fixed via per-cycle output dirs + atomic write + `run_config.json` provenance sidecar; re-evaluated all 3 pre-existing checkpoints directly to confirm the old files had drifted |
| `--per` (Prioritized Experience Replay) is wired into `train_sac.py` | Code present and selectable; smoke-tested for 3,000 steps without error |
| `--n-step 5` (default), `--per`, vanilla (`--n-step 1`) each run end-to-end | Each ran 3,000 steps, saved a checkpoint, wrote TensorBoard events |
| `--resume` reloads checkpoint + replay buffer and continues | Smoke-tested |
| `--cycles NEDC,FTP75` multi-cycle interleave runs and evaluates both | Smoke-tested, wrote per-cycle rows to `eval_history.csv` |
| `run_config.json` records CLI args + resolved git commit correctly | Smoke-tested |
| `eval_history.csv` persists across stop/crash (not just in-memory) | Code path added and used to generate `results/figures.py` output |

**Caveat as of 2026-08-26:** all of the above is verified against the
current **uncommitted working tree** (see `git status` — the entire
[3.1.0] changeset is unstaged). It has not yet been re-verified against a
clean commit. Commit first (per `ROADMAP.md` §5 step 1), then this caveat
can be removed.

---

## D. Test suite baseline

- **211/211 tests pass** in `tests/` — confirmed by running
  `python -m pytest tests/ -v`.
- Root cause of the 18 that were previously failing: stale hardcoded
  constants in the test files themselves (predating Phase 1 corrections),
  not physics bugs — cross-checked by re-running
  `python -m src.evaluate_advanced --cycle NEDC` and matching
  `VALIDATION.md`'s 3.506 L/100km exactly before touching anything.
- 2 of those 18 were genuine test-authoring bugs (unrelated to staleness):
  a `test_done_flag` off-by-one, and a wrong-gridpoint
  `test_overload_at_high_speed` assertion. Both fixed.
- Same uncommitted-tree caveat as §C applies.

---

## E. Current RL training state (dated snapshots — will be superseded)

### 2026-08-26 — `models/NEDC` and `models/FTP75`, ~28% through 1.5M-step runs

Confirmed via `python -m results.figures --run models/<cycle>` (this is
the canonical diagnostic command — re-run it to get an updated snapshot,
don't hand-derive verdicts from the raw CSV).

**`models/NEDC`** @ 420,900 / 1,500,000 steps:
- Best V_CE_equiv: 4.4692 @ step 280,600, SoC 72.3%
- Benchmark 3.506 / ECMS 3.1887 — **verdict: does NOT beat benchmark**
- Not charge-sustaining (`|SoC-50%| <= 2%` fails)
- Quartile fuel plateau: 4.903 → 4.879 → 4.793 → 4.916 (flat, ~15%+ above benchmark; 2-point endpoint trend reads "improving" but is noise — quartile view overrides it)
- Mode breakdown (best ckpt): `OFF=10.5% ASSIST=26.6% LPS=46.0% ONLY=0.0% REGEN=17.0%` — **"ASSIST BLOB" confirmed** (see §G for exact mode definitions): the diagnostic fields are `OFF` (engine off / pure electric) and `ASSIST`, not `ONLY` (pure engine, no motor — correctly near 0% same as both benchmarks). The real gap vs. ECMS (`off=53.1%, assist=0.2%` on NEDC) is `OFF` far too low and `ASSIST` far too high — the agent charges via LPS almost twice as much as ECMS (46.0% vs 29.7%) but then spends that energy in small ASSIST increments instead of committing to sustained OFF/EV stretches

**`models/FTP75`** @ 422,100 / 1,500,000 steps:
- Best V_CE_equiv: 4.3479 @ step 56,280, SoC 47.1%
- Benchmark 3.232 / ECMS 2.8097 — **verdict: does NOT beat benchmark**
- Not charge-sustaining
- Quartile fuel trend **worsening**: 4.468 → 4.488 → 4.450 → 4.517; SoC drifting 47.3%→55.9% (mild instability, not just slow convergence)
- Mode breakdown (best ckpt): `OFF=12.6% ASSIST=27.4% LPS=34.4% ONLY=0.0% REGEN=25.7%` — same ASSIST BLOB pattern

**Root-cause hypothesis #1 (unconfirmed, needs TensorBoard check):**
continuous `Box(-1,1)` action space (`src/env/ems_env.py:263`) + SAC
`ent_coef="auto"` (`src/agents/train_sac.py:392`) structurally discourages
the policy from committing to action-range boundaries.

**Root-cause hypothesis #2 (2026-08-26, concrete and directly evidenced —
see §F below): the reward under-prices battery energy relative to the
project's own proven ECMS optimum, and gives near-zero marginal signal on
battery use inside a wide SoC deadband.** This is cheaper to test than #1
(one CLI flag, no code change) and is grounded in numbers this repo has
already proven correct (`src/baselines/ecms.py`), not a new derivation.

See `ROADMAP.md` §4 for the investigation plan and which to test first.

### 2026-08-26 (later same day) — hypothesis #1 tested: static eq_factor REFUTED

Ran `models_trial_eqfix/{NEDC,FTP75}`, 150k steps each, `--eq-factor 1.3125`
(NEDC) / `2.4062` (FTP75) — the ECMS-proven charge-sustaining prices —
replacing the flat default of 1.0. Gated with `results/readiness_gate.py`
(added this session). **Both FAILED the gate:**

- **NEDC**: `OFF=15.5%` (ECMS 53.1%, gap -37.6pp) `ASSIST=34.1%` (ECMS
  0.2%, gap +33.9pp) — `ASSIST` got WORSE than the flat-1.0 baseline's
  26.6%. SoC quartile trend: 53.2%→60.9%→64.3%→66.4%, monotonic drift
  13.2pp (charging) — worse than the flat-1.0 baseline logged above.
- **FTP75**: `OFF=13.9%` (ECMS 40.4%, gap -26.5pp) `ASSIST=34.2%` (ECMS
  6.0%, gap +28.2pp) — also worse than baseline's 27.4%. SoC quartile
  trend: 49.8%→45.9%→42.8%→40.7%, monotonic drift 9.1pp (depleting) — a
  NEW failure direction not present in the flat-1.0 baseline.

**Why this is refuted-with-an-asterisk, not a dead end:** `ecms.py`'s own
docstring (`tune_lambda`) states outright that a CONSTANT lambda cannot
make this plant charge-sustaining even for the Hamiltonian-optimal
controller — closed-loop SoC feedback (`k_fb=8.0`) is required. A flat
`eq_factor` was never a fair test of "does ECMS-correct pricing fix this";
it tested the version ECMS's own authors already proved insufficient.

**Action taken:** implemented `k_fb` (ECMS-style closed-loop costate
feedback, `eq_factor_eff = eq_factor + k_fb*(SOC_TARGET - soc_before)`) in
`EMSEnv` and threaded it through `train_sac.py` (commit `2a8cdbe`). Default
`k_fb=0.0` is an exact no-op (verified by a new unit test,
`test_k_fb_costate_feedback`, asserting the reward shift matches the
formula algebraically to 1e-8 and that physics/mode are unchanged by
`k_fb` — only the reward's battery-price term differs). 212/212 tests
pass. Fresh smoke test launched: `models_trial_kfb/{NEDC,FTP75}`,
`--eq-factor 1.3125/2.4062 --k-fb 8.0` — the actual, correctly-implemented
test of hypothesis #1. Results pending.

### 2026-08-26 (later still) — hypothesis #1 round 2 (k_fb=8.0): improves but does not resolve

Ran `models_trial_kfb/{NEDC,FTP75}`, 150k steps each, `--eq-factor
1.3125/2.4062 --k-fb 8.0` (the correctly-implemented closed-loop pricing).
Gated with `results/readiness_gate.py`. **Both still FAIL the gate**, but
not uniformly worse than round 1 — a genuine, measurable partial effect:

- **NEDC**: `OFF=7.8%` (ECMS 53.1%, gap **-45.3pp**, WORSE than the flat-1.0
  baseline's 10.5% and round 1's 15.5%). `ASSIST=27.5%` (back near the
  flat-1.0 baseline's 26.6%, recovered from round 1's regression to
  34.1%). SoC quartile: 49.9%→53.2%→56.3%→60.1%, still monotonically
  drifting up (10.1pp) — improved starting point (49.9% vs round 1's
  53.2%) but same runaway-charging character.
- **FTP75**: `OFF=12.7%` (ECMS 40.4%, gap -27.7pp, essentially flat vs.
  baseline's 12.6%). `ASSIST=25.8%` (recovered from round 1's 34.2% back
  to near baseline's 27.4%). SoC quartile: 56.4%→62.4%→62.5%→**52.7%** —
  **NOT monotonic** (gate's SoC-trend check PASSED here) — the closed-loop
  feedback visibly pulled SoC back down in the last quartile instead of
  the pure runaway seen in both prior FTP75 attempts (baseline: monotonic
  up to 55.9%; round 1: monotonic down to 40.7%).

**Conclusion:** the closed-loop feedback is doing something real and
directionally correct (undoes round 1's ASSIST regression; introduces
self-correcting SoC dynamics on FTP75 that neither the flat baseline nor
the static-price round 1 showed) — but it does NOT close the actual
gate-failing gap. `OFF` (pure-EV time, the core of the ASSIST BLOB) is
still 28-45 percentage points below ECMS on both cycles. **Hypothesis #1
(reward battery-pricing), even correctly implemented per ECMS's own proven
formula, is not sufficient on its own to fix the ASSIST BLOB within a
150k-step smoke test.** Per `ROADMAP.md`'s pre-agreed decision rule, moving
to hypothesis #2 (SAC entropy) next — `k_fb=8.0` is kept as a real,
evidenced improvement to carry forward into that investigation rather than
reverted, since it is not neutral-or-harmful, just insufficient alone.

### 2026-08-26 (later still) — hypothesis #2 tested via TensorBoard: ORIGINAL FORM NOT SUPPORTED, refined to #3

Extracted actual scalar traces from `models/{NEDC,FTP75}/tb/` (existing
420k+/422k+-step runs, no new training) via
`tensorboard.backend.event_processing.event_accumulator.EventAccumulator`.

**`ent_coef` (entropy coefficient) actually DECREASED substantially on
both cycles** — NEDC: 0.157 → 0.023 (first-10% vs. last-10% average);
FTP75: 0.160 → 0.099. `ent_coef_loss` converged toward ~0 on both,
meaning SAC's automatic entropy tuning reached a stable point. **This
does NOT match the original hypothesis #2 ("entropy stays high, blocks
commitment to action-range boundaries")** — entropy pressure was
shrinking, not stuck.

**What the data actually shows: critic (value-function) instability,**
markedly worse on FTP75:

- **NEDC** `train/critic_loss` shape (10 samples across the run):
  3.39 (step 4,880) → rises to a peak **12.63 (step 278,160)** → recovers
  back down to **3.16 (step 419,680)**. A temporary instability that
  appears to be self-correcting by the end of the available data.
- **FTP75** `train/critic_loss` shape: 13.53 (step 7,504) → climbs
  almost monotonically to **53.53 (step 382,704)** → only a slight dip to
  45.61 by step 420,224. **Not stabilized** within the available data —
  this tracks with `results/figures.py`'s independent "WORSENING quartile
  trend" flag for FTP75 (not raised for NEDC), giving two independent
  signals pointing at the same cycle-specific instability.
- `train/actor_loss` also climbs substantially on both cycles despite
  `rollout/ep_rew_mean` improving over the same period (NEDC: -129.9→
  -101.7; FTP75: -347.1→-85.5) — the policy IS learning something, but
  the critic backing it is not reliably converged, especially on FTP75.

**Refined hypothesis #3 (new, better-evidenced than original #2):**
critic-loss divergence/instability, likely driven by `gamma=0.9999`
(deliberately extreme, see `train_sac.py` lines 32-36, chosen so the
terminal charge-sustaining signal reaches early-episode steps) combined
with aggressive updates (`train_freq=64, gradient_steps=64` — 64 gradient
steps per 64 env steps). This is a known SAC pathology, not specific to
this project. Next test: reduce update aggressiveness (`--gradient-steps`,
added to `train_sac.py`) as a single-variable smoke test, keeping
`k_fb=8.0` (kept from hypothesis #1's evidenced partial improvement).

### 2026-08-26 (later still) — hypothesis #3 CONFIRMED: gradient_steps=16 fixes critic instability

Ran `models_trial_gradstep/{NEDC,FTP75}`, 150k steps each, `--eq-factor
1.3125/2.4062 --k-fb 8.0 --gradient-steps 16` (kept k_fb from #1's
evidenced improvement; only new variable is gradient_steps 64→16).

**Direct TensorBoard confirmation (not inferred from downstream behavior):**
comparing `train/critic_loss` at the SAME step count (~150k) across the
two configs:

| | step ~150k critic_loss | trajectory |
|---|---|---|
| FTP75, `gradient_steps=64` (original) | **25.9** (step 157,584) | climbs to 53.5, never recovers |
| FTP75, `gradient_steps=16` (this run) | **4.3** (step 148,204, final) | bounded 1.8-8.0 throughout, no upward trend |
| NEDC, `gradient_steps=64` (original) | ~3-5 (interpolated) | climbs to 12.6 @ 278k before recovering |
| NEDC, `gradient_steps=16` (this run) | **2.7** (step 148,840, final) | bounded 1.7-2.8 throughout |

`ent_coef` converged lower and faster on both (NEDC final 0.025, FTP75
final 0.016) — consistent with healthier, more stable training, not a
symptom of a new problem. `actor_loss` still climbs steadily on both, but
combined with a bounded (not diverging) critic this reads as the Q-function
tracking genuinely improving returns, not instability.

**Readiness gate (still NOT READY, but real improvement on every axis):**
- NEDC: `OFF=22.9%` (best of ALL rounds so far: baseline 10.5%, static-#1
  15.5%, k_fb-#1 7.8%, this run 22.9%; still -30.2pp vs ECMS's 53.1%).
  `ASSIST=32.8%`. SoC quartile 58.2%→51.6%→55.0%→58.6% — **NOT monotonic**
  (gate's SoC-trend check PASSED).
- FTP75: `OFF=15.6%` (also best of all rounds: baseline 12.6%, static-#1
  13.9%, k_fb-#1 12.7%, this run 15.6%; still -24.8pp vs ECMS's 40.4%).
  `ASSIST=36.6%` (worst of all rounds — a real regression on this one
  axis). SoC quartile 66.2%→50.7%→**50.0%→49.3%** — technically flagged
  monotonic-drift by the gate (16.9pp, driven by the high Q1 start), but
  the second-half behavior (50.7%→50.0%→49.3%) is the closest any run has
  come to ECMS's own SoC_end target (50.13%) — worth noting as a
  gate-heuristic limitation (a high Q1 that then corrects sharply still
  trips the "monotonic across all 4 quartiles" check), not a real failure
  of the underlying behavior.

**Conclusion:** hypothesis #3 confirmed with direct, quantitative
evidence — `gradient_steps=16` measurably fixes the critic divergence.
Downstream mode-split (`OFF`/`ASSIST`) improved on both cycles but has not
yet closed the gate. Since the 150k-step smoke budget is now exhausted
with a CONFIRMED-healthy (not diverging) training process, the next test
is whether the remaining gap closes with more training time at this same,
now-stable config, before considering a further hyperparameter change.

*(Next snapshot goes here — append, don't overwrite the ones above.)*

---

## F. Reward-function scientific audit (2026-08-26)

**Question asked:** is the reward function in `src/env/ems_env.py`
(`EMSEnv.step`, lines ~483-551) scientifically correct?

**Method:** compared it against the project's own ECMS implementation
(`src/baselines/ecms.py`), which applies Pontryagin's Minimum Principle to
this exact plant and has *tested & proven* charge-sustaining results (see
its module docstring) — the ECMS Hamiltonian `H(u) = P_fuel(u) + lambda *
P_batt(u)` is the ground-truth-correct instantaneous cost for this MDP, not
a hypothesis.

**Finding — two separate questions, two separate verdicts:**

1. **Is it the right OBJECTIVE?** Yes, confirmed. The per-step reward
   (`fuel_liters + eq_factor * elec_liters`, scaled) is an exact
   decomposition of the true evaluation metric `v_ce_equiv` — the
   env's own docstring proves it telescopes exactly, and
   `tests/test_ems_env.py` verifies this numerically. Nothing wrong here.

2. **Is it a good REWARD SIGNAL for the agent to learn efficiently from?**
   **No — one concrete, evidenced flaw found:**

   - `EMSEnv`'s default `eq_factor = 1.0` (`src/env/ems_env.py:221`,
     also `train_sac.py --eq-factor` default) prices battery energy at
     its raw physical equivalence value.
   - `ecms.py`'s bisection search — run against the SAME powertrain
     blocks — proved that a **flat price of 1.0 does NOT produce
     charge-sustaining behavior** on either cycle. The price that
     actually achieves charge-sustaining (`SoC_end ≈ 50%`) is
     `lambda_0 = 1.3125` (NEDC) / `2.4062` (FTP75), applied *with*
     closed-loop SoC feedback `lambda_t = lambda_0 + k_fb*(SOC_TARGET -
     soc)`, `k_fb = 8.0` (§ecms.py `tune_lambda`).
   - `ems_env.py` does NOT apply this proven pricing. Instead it prices
     battery energy flatly at 1.0 (via the "economic" term) and bolts on
     a *separate*, weaker mechanism — a quadratic+linear SoC-restoring
     penalty that only activates once `|SoC - 50%| > 10%`
     (`SOC_DEADBAND = 0.10`, `src/env/ems_env.py:192`).
   - Net effect: for the entire 40%-60% SoC band — which is exactly
     where a charge-sustaining policy spends most of its time — the
     agent's marginal cost of discharging vs. charging the battery is a
     **flat, SoC-independent 1.0**, not the state-dependent price ECMS
     proved is needed. There is no gradient inside that band telling the
     agent "commit to EV now" vs. "blend a little" — the two look almost
     equally cheap per-step.

**Why this is a strong candidate for the ASSIST BLOB / FTP75 SoC-drift
findings in §E:** an underpriced, flat battery cost is consistent with
*both* symptoms simultaneously — it explains the lack of decisive
commitment (no marginal signal to commit on) and the FTP75 divergence
(battery genuinely is cheaper than it should be, so nothing resists
draining it beyond the reactive, delayed deadband penalty).

**Theoretical note (why this isn't automatically self-correcting):** in the
ideal limit, a converged SAC critic's SoC-conditioned value gradient *is*
the Pontryagin costate — i.e. a perfect critic would learn the correct
implicit ECMS lambda on its own, no explicit shaping required. In practice
that requires the credit-assignment problem to be solved over
~1200-1877-step episodes at `gamma=0.9999` (`train_sac.py` line 32-36) —
exactly the difficulty `n-step` returns were already added to help with
(see `train_sac.py` lines 43-50). This reward-pricing gap is a plausible
*additional* reason n-step alone hasn't cleared the ASSIST BLOB: n-step
sharpens propagation of a real reward difference, but if the instantaneous
cost gap between adjacent actions is itself tiny (because of the flat
underpriced deadband), there's a smaller real difference to propagate.

**Not yet tested — this is a hypothesis, evidenced but unconfirmed by an
actual training run.** See `ROADMAP.md` §4-5 for the proposed experiment
(cheapest first step: set `--eq-factor` to the ECMS-proven value per
cycle, no code change, smoke-test at 150k steps before any full run).

---

## G. Operating-mode taxonomy (permanent — how every controller is classified)

**Correction (2026-08-26): earlier discussion in this project conflated
`OFF` and `ONLY` when reading mode-breakdown output. This section is the
authoritative definition — refer back to it, don't re-derive from memory.**

Classification logic lives in `src/agents/mode_breakdown_rl.classify_rollout`
(reused by `results/figures.py`), applied identically to the rule-based,
ECMS, and RL controllers so all three are directly comparable. Every
"moving" step (`T_MGB != 0`) is tagged into exactly one of five modes,
using the ACTUALLY EXECUTED torques (post feasibility-masking), not the
raw action:

| Mode | Code condition | Physical meaning | Why it saves fuel |
|---|---|---|---|
| **OFF** (engine off / pure electric) | `T_CE_cmd <= T_CUTOFF` (5 Nm) during traction | Engine fuel flow is zero (fuel-cutoff active); motor alone carries ~all propulsion torque | The single largest per-step saving — zero fuel for that instant, paid for by battery energy accumulated elsewhere |
| **ASSIST** | engine on (`T_CE > cutoff`) AND `T_EM > 0` | Both engine and motor add positive (propulsive) torque simultaneously | Lets the engine avoid an overloaded/inefficient high-torque operating point by sharing the load with the battery — a genuine boost, but it drains the battery and should be used sparingly |
| **LPS** (load-point shift / charge) | engine on AND `T_EM < 0` | Engine produces MORE torque than the wheels need; the surplus drives the motor as a generator, charging the battery | Engine brake-specific fuel consumption (BSFC) is a strongly nonlinear function of (speed, torque) — pushing the engine's operating point toward a more efficient region and "banking" the surplus as electrical energy (to spend later in OFF/ASSIST) uses less total fuel than running the engine only at the immediate, often-inefficient demand |
| **ONLY** (engine only) | engine on AND `T_EM ≈ 0` | Engine alone covers 100% of demanded torque, no motor interaction at all | Rarely optimal — there's almost always a marginal benefit to either LPS or ASSIST — hence both benchmarks and a well-trained agent keep this near 0% |
| **REGEN** | `T_MGB < 0` (braking) | Motor recovers braking energy as electrical charge (with conversion losses, η<1) instead of it being dissipated as heat in friction brakes | The env hard-codes MAXIMUM feasible regen for every controller equally (`src/env/ems_env.py` docstring) — this is not a decision variable, so it should read ~identical across rule-based/ECMS/RL by construction; a large mismatch here would itself be a red flag |

**Reference percentages** (`src/agents/mode_breakdown_rl.py` `REF` dict,
measured on this exact plant):

| | OFF | ASSIST | LPS | ONLY | REGEN |
|---|---|---|---|---|---|
| NEDC rule-based | 59.0% | 0.0% | 23.8% | 0.2% | 17.0% |
| NEDC ECMS | 53.1% | 0.2% | 29.7% | 0.0% | 17.0% |
| FTP75 rule-based | 46.3% | 0.4% | 22.4% | 5.2% | 25.7% |
| FTP75 ECMS | 40.4% | 6.0% | 27.9% | 0.0% | 25.7% |

**How to read a mode breakdown against these:** the diagnostic fields are
`OFF` and `ASSIST` (both should track the ECMS row); `ONLY` near 0% is
CORRECT, not a failure signal; `REGEN` should closely match by construction
and a mismatch there would flag a masking bug, not a policy quality issue.
`results/figures.py`'s "ASSIST BLOB" flag fires specifically when
`ASSIST` is >10pp above ECMS's AND `OFF` is >10pp below it simultaneously.

## Cross-references

- `ROADMAP.md` — current phase, decision gates, ordered next steps.
- `VALIDATION.md` — detailed narrative log of how each fact above was
  checked (commands run, output inspected).
- `CHANGELOG.md` — what changed in each version, when.
- `results/readiness_gate.py` — automated pre-flight checklist
  (`python -m results.readiness_gate --run <smoke-test-dir>`) that checks
  unit tests, git-clean, run completion, mode-breakdown-vs-ECMS gap, and
  SoC quartile trend before a config is scaled up to a full-length run.
