# Powertrain Validation Log

Phase 1 (see commit `d6345dc`, "Phase 1 complete: pure-Python powertrain
environment validated against MATLAB baseline and advanced rule-based
controller") validated the pure-Python powertrain against MATLAB/Simulink
ground truth using a set of one-off diagnostic/fix scripts. Those scripts
have since been deleted — their job is done and the results below are
final.

**Do not re-run or re-create equivalent checks. The numbers below are
already confirmed correct.**

## What was validated

1. **Battery sign convention** — discharge (`p_bt > 0`) correctly decreases
   `q_bt`.
2. **Engine/motor map axes** — `w_CE_max_fine`/`T_CE_max` and
   `w_EM_max_row`/`T_EM_max_arr` confirmed against the MATLAB workspace.
3. **Engine consumption map orientation fix** — `data/maps/engine_maps.npz`
   was rebuilt so the lookup is indexed by (speed, torque) directly,
   matching the Simulink `Lookup2D` block (replacing an earlier incorrect
   pressure-based indexing).
4. **Motor max-torque curve fix** — `data/maps/motor_maps.npz`'s
   `w_EM_max_row`/`T_EM_max` arrays were corrected to match the MATLAB
   workspace values feeding the Controller's `interp1` block.
5. **Controller torque-split identity** — `T_CE + T_EM == T_MGB` holds
   exactly for all tested `(w_MGB, dw_MGB, T_MGB)` points.
6. **Energy conservation** — a symmetric discharge/regen battery cycle nets
   to zero or slightly negative (never net-charges), confirming correct
   efficiency-map sign convention in `electric_motor()`/`Battery`.
7. **Gearbox** — `gearbox()` output (`w_mgb`, `t_mgb`) matches MATLAB-logged
   values at t=53,54,55,65,66,102,103 to within ~0.01%.
8. **Vehicle dynamics** — `VehicleDynamics.step()` output (`w_wheel`,
   `T_wheel`) matches MATLAB-logged values at t=53,54.
9. **Full end-to-end chain** — every intermediate signal (v, w_wheel,
   T_wheel, w_MGB, T_MGB, u, T_CE, P_CE) through the entire pipeline
   (VehicleDynamics -> gearbox -> control unit -> combustion
   engine/electric motor -> tank/battery) matched MATLAB ground truth at
   t=53, 54, 102 with no mismatches.

## Result

All checks passed. The pure-Python powertrain environment
(`src/env/powertrain.py`, `src/baselines/rule_based.py`) is validated
against the MATLAB/Simulink reference model.

## Removed scripts

The following one-off scripts performed the checks above and have been
deleted (recoverable from git history at or before commit `d6345dc` if
ever needed again): `src/check_battery_sign.py`, `src/check_engine_maps.py`,
`src/check_motor_maps.py`, `src/diagnose_subsystems.py`,
`src/fix_engine_maps.py`, `src/fix_motor_maps.py`, `src/full_diagnostic.py`,
`src/validate_gearbox.py`, `src/validate_vehicle.py`, plus their one-time
CSV inputs `w_CE_row.csv`, `T_CE_col.csv`, `V_CE_map.csv` (exported from
MATLAB, already baked into `data/maps/engine_maps.npz`).

---

## Phase 3 — RL pipeline audit & fixes (2026-08-17)

**Do not re-run or re-create equivalent checks for the items below. Every
claim here was actually executed and its output inspected, not assumed —
see "how confirmed" on each line. Re-verifying these wastes time; if you
suspect regression, run the specific command listed, not a broader search.**

### What was found broken

- 18 of `tests/` (`test_powertrain.py`, `test_tank.py`, `test_driving_cycle.py`,
  `test_electric_motor.py`) were failing. Root cause: those test files hardcode
  their own local copies of physical constants (air density, cold-start
  factor, NEDC sample count, motor torque-curve values, a few formulas) that
  predate the Phase 1 corrections documented above. **The physics code was
  never wrong** — confirmed by re-running `python -m src.evaluate_advanced
  --cycle NEDC` before touching anything and matching this file's numbers
  exactly (3.506 L/100km). Only the test files were stale; all 18 were fixed
  to assert against the current, correct behavior (2 of the 18 turned out to
  be genuine test-authoring bugs unrelated to staleness — a `test_done_flag`
  off-by-one and a wrong-gridpoint `test_overload_at_high_speed` assertion —
  also fixed).
  **How confirmed:** `python -m pytest tests/ -v` → 211 passed, 0 failed.
- `src/env/ems_env.py` and `src/env/ems_env_lookahead.py` were two separate
  files that had drifted apart risk (any edit to one silently didn't apply to
  the other). A byte-level diff showed they were identical except for the
  lookahead-observation feature, so they were merged into one file
  (`ems_env.py` gained a `lookahead: int = 0` constructor arg; default 0
  reproduces the exact old 16-dim behavior). `ems_env_lookahead.py` deleted.
  **How confirmed:** `diff` of the two files before merging (only the
  lookahead block differed); `python -m pytest tests/test_ems_env.py -v`
  → 4/4 passed after the merge (env-wiring-matches-evaluate_advanced,
  reward-decomposition-exactness, and gym API checks all still hold at
  `lookahead=0`).
- Training was guided by default: `train_sac.py --prefill-mode benchmark`
  (the default) seeded the SAC replay buffer with the advanced rule-based
  controller's own rollouts before training started, and
  `pretrain_bc.py`/`finetune_bcreg.py`/`finetune_sac.py` did literal
  behaviour cloning. Per explicit instruction, this is removed from the
  default pipeline: the `prefill_buffer_benchmark` function is deleted from
  `train_sac.py` entirely (not just disabled), and the new default is
  `--prefill-mode none --prefill-eps 0` — a genuine blank-slate run. The BC
  scripts remain in the repo for reference but are marked GUIDED / LEGACY in
  their own docstrings and are not invoked by anything in this pipeline.
- `models/best_score.txt` (3.8935) did not match the fuel figure the
  adjacent `sac_ems_best.zip` actually produced (4.271 on NEDC, 4.180 on
  FTP75) — the file and the checkpoint had silently drifted apart because
  different cycles/configs were being written into one shared `models/`
  directory. Fixed: each run now writes to `models/<cycle>/` (or
  `models/multi_<cycles>/`), `best_score.txt` is written atomically
  (temp-file + replace) immediately next to the checkpoint it describes, and
  a `run_config.json` sidecar records the exact CLI args + git commit.
  **How confirmed:** re-evaluated all three pre-existing checkpoints
  (`models/sac_ems_best`, `models_ftp75/sac_ems_best`,
  `models_robust/sac_ems_best`) directly with `SAC.load(...).predict(...)`
  through `EMSEnv` on both cycles — none beat the rule-based benchmark, and
  the mode-by-mode breakdown (below) explained why.
- `src/agents/per.py` (Prioritized Experience Replay) was fully implemented
  but not imported by any training script. Wired into `train_sac.py` behind
  `--per` (mutually exclusive with `--n-step > 1` for now — combine only
  behind a verified merged buffer/algorithm class, not attempted here).
- `evaluate_rl.py` / `mode_breakdown_rl.py` hardcoded a 16-dim `EMSEnv()` and
  crashed (`ValueError: Unexpected observation shape (16,)`) on any
  lookahead-trained checkpoint (which is a 20-dim observation at the
  validated default `lookahead=5`). Fixed to infer the lookahead window from
  the loaded checkpoint's `observation_space.shape[0]` automatically.
  **How confirmed:** ran both scripts against a real lookahead-trained
  checkpoint after the fix; both produced output instead of crashing.
- `train_sac_fix2.py`, `train_sac_nstep.py`, `train_sac_lookahead.py` were
  three near-duplicate forks of `train_sac.py` (this is exactly how the
  best_score.txt / checkpoint mismatch above happened — no way to tell which
  config produced a given checkpoint). Deleted; every option they offered
  (lookahead, n-step, multi-cycle interleave) is a flag on the single
  `train_sac.py` now.
- `EvalAndCheckpoint`'s periodic-eval history was kept only in an in-memory
  Python list, lost if a run was stopped or crashed. Now appended to
  `<out_dir>/eval_history.csv` after every evaluation, read by the new
  `results/checkpoints.py` / `results/figures.py` (training-curve +
  mode-breakdown plots + plain-text trend/verdict diagnosis — see
  `python -m results.figures --run models/<cycle>`).

### Pipeline smoke tests actually run (not just code review)

- n-step SAC (`--n-step 5`, default), PER (`--per`), and vanilla SAC
  (`--n-step 1`) each run for 3,000 steps end-to-end without error, saving a
  checkpoint and writing TensorBoard events (`<out>/tb/SAC_1/events.out...`).
- `--resume` reloads a checkpoint + its replay buffer and continues training.
- `--cycles NEDC,FTP75` (multi-cycle interleave) runs and evaluates both
  cycles, writing per-cycle rows to `eval_history.csv`.
- `--prefill-mode constant --prefill-eps N` seeds the buffer and reports the
  transition count.
- `run_config.json` correctly records CLI args + resolved git commit.

None of the smoke tests above ran long enough to demonstrate the agent
*learning* to beat the benchmark — they only prove the pipeline is wired
correctly (no crashes, correct files written, correct shapes). Whether the
new n-step / lookahead / unguided setup actually reaches or beats the
rule-based benchmark is an open, in-progress question — see the training
runs' `eval_history.csv` / `python -m results.figures` output for the
current answer, not this file.
