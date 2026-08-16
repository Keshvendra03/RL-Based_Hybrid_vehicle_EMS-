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
