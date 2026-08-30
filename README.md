# RL-Based Hybrid Vehicle Energy Management System (EMS)

An advanced Reinforcement Learning (RL) framework designed to optimize the
power split between an Internal Combustion Engine (ICE) and an Electric
Motor (EM) in a parallel mild-hybrid passenger vehicle.

This repository implements a pure-Python powertrain simulation — ported
and validated against an award-winning MATLAB/Simulink model — as the
foundation for a custom **Gymnasium environment** to train Deep RL agents
(PPO/DQN), benchmarked against rule-based Energy Management Strategies
(EMS).

---

## 📈 Project Status

- [x] **Phase 0:** Project scoping, environment initialization, repo structure.
- [x] **Phase 1:** Pure-Python powertrain physics implementation, validated
      against the reference Simulink model (NEDC cycle).
- [x] **Phase 2:** Custom Gymnasium environment (`ems_env.py`) wrapping the
      Phase 1 powertrain for RL training; action/observation space design.
- [x] **Phase 3:** SAC training pipeline (PER, n-step, lookahead,
      checkpointing, TensorBoard logging) built and audited — 211/211 tests
      pass. See `CHANGELOG.md`.
- [x] **Phase 4:** Gated mode-aware action map; FTP75 reaches the rule-based
      benchmark, NEDC regresses (SoC runaway). See `PHASE4_FINAL_REPORT.md`.
- [x] **Phase 5 / 5B:** Costate gain `k_fb` identified; NEDC charge-sustaining
      solved (3/3 seeds at `k_fb=2.5`), fuel tied. `PHASE5*_*.md`.
- [x] **Phase 6:** Controlled conditional-exploration A/B — the
      replay-coverage hypothesis is **REFUTED**. `PHASE6_FINAL_REPORT.md`.
- [x] **Phase 7:** Economic / costate forensic (no training) — the
      "battery energy over-priced" hypothesis is **REFUTED**; `k_fb` is a flat
      plateau. `PHASE7_FINAL_REPORT.md`.
- [x] **Phase 8:** Actor-side breakthrough attempt. The Q-oracle test shows
      exploiting the trained critic is **worse** than the current actor and
      loses charge-sustaining; the 2-component mixture actor confirms it
      (NEDC 3.873 1/3 CS, FTP75 3.246 3/3 CS, neither beats RB) ⇒ the policy
      class is not the bottleneck. `PHASE8_REPORT.md`.
- [x] **Phase 9:** Critic value-fidelity forensics + CQL. On-distribution the
      critic is **not grossly wrong** (min-Q ranks HIGH_EFF ≥ ECMS_NBHD ≥ LOW
      ≥ OFF); the defect is a mild compounding low-load arg-max bias in a
      thinly-covered region. A **CQL conservative critic FAILED at every
      coefficient** (SoC runaway / 100+ violations). Physical BSFC decomposition:
      ~39% of the NEDC gap is part-load engine inefficiency. Next: targeted
      high-load training coverage. `PHASE9_FINAL_REPORT.md`.

> **Current best validated SAC:** NEDC **3.7666** L/100km (3/3 charge-sustaining,
> +7.4 % vs rule-based 3.5056, +18.1 % vs ECMS 3.1887); FTP75 **3.2889**
> (3/3 CS, +1.8 % vs rule-based 3.2323). The rule-based benchmark is **not yet
> beaten** on either cycle. Live status, diagnosis and next steps: **`ROADMAP.md`**.

### Phase 1 Validation Results (NEDC cycle)

| Metric | MATLAB (baseline) | Python (baseline) | Diff |
|---|---|---|---|
| Fuel consumption (L/100km) | 4.513 | 4.535 | ~0.5% |
| Equivalent fuel consumption (L/100km) | 4.520 | 4.535 | ~0.3% |
| Final battery charge Q_BT (As) | 17,840 | 18,950 | ~6.2% |

| Metric | MATLAB (advanced rule-based) | Python (advanced rule-based) |
|---|---|---|
| Fuel consumption (L/100km) | 3.348 | 3.506 |
| **Reduction vs. baseline** | **25.81%** | **22.69%** |

Both environments show a substantial fuel-consumption reduction from the
advanced rule-based controller relative to the simple baseline, confirming
the Python environment correctly captures the *relative* impact of
energy-management strategy changes. The small residual offsets above are
systematic and present across both controllers, so strategy-vs-strategy
comparisons within the Python environment (rule-based vs. RL) remain valid.

---

## 🗂️ Repository Structure (current)

```
data/
  drive_cycles/
    nedc.csv            # NEDC driving cycle (v, dv, gear, x_tot vs time)
    ftp75.csv           # FTP-75 driving cycle
  maps/
    engine_maps.npz     # Combustion engine consumption & max-torque maps
    motor_maps.npz      # Electric motor efficiency & max-torque maps
  params.json           # Physical constants (vehicle, gearbox, engine,
                         # motor, battery, equivalent-consumption)

src/
  env/
    driving_cycle.py     # DrivingCycle: loads/steps through a drive cycle
    powertrain.py         # Vehicle dynamics, gearbox, combustion engine,
                           # electric motor, battery, tank,
                           # equivalent fuel consumption
    vehicle_dynamics.py    # (placeholder -- reserved for Phase 2 refactor)
    ems_env.py             # (placeholder -- Gymnasium env, Phase 2)
  baselines/
    rule_based.py           # Simple baseline controller (state_CE=1
                             # constant, LPS/regen-only torque split)
    advanced_rule_based.py  # Gear-dependent advanced rule-based
                             # controller with dynamic engine on/off
  agents/                  # (reserved for Phase 3 RL agents)
  evaluate.py            # Run the baseline controller over a full cycle
  evaluate_advanced.py   # Run the advanced rule-based controller

tests/                   # Unit tests for each powertrain component
```

> **Note:** `vehicle_dynamics.py` and `ems_env.py` are currently empty
> placeholders created during Phase 0 scoping. All Phase 1 vehicle
> physics live in `src/env/powertrain.py`. These placeholders will be
> filled in during Phase 2 (Gymnasium environment wrapper).

---

## 🏎️ Subsystem Architecture (`src/env/powertrain.py`, `driving_cycle.py`)

* **Driving Cycle** (`driving_cycle.py`): loads standard driving-cycle
  schedules (NEDC, FTP-75) as time-series velocity/gear profiles.
* **Vehicle Dynamics**: tractive force (F_total) from aerodynamic
  drag, rolling resistance, and inertial mass scaling; converts to
  wheel speed/torque.
* **Manual Gearbox**: gear-ratio lookup, transmission efficiency, and
  wheel-side <-> flywheel-side speed/torque conversion.
* **Combustion Engine (ICE)**: 2-D consumption map (speed x torque) ->
  fuel mass flow rate; idle, cutoff, overload/overspeed detection.
* **Electric Motor (EM)**: 2-D efficiency map (speed x signed torque)
  for motoring and regenerative-braking modes.
* **Battery Pack (ESS)**: charge integrator (Q_BT), open-circuit/
  charging/discharging voltage curves, current, SoC, under-voltage and
  over-current supervision.
* **Fuel Tank**: trapezoidal integration of fuel power -> cumulative
  fuel mass -> L/100km, including cold-start factor.
* **Equivalent Fuel Consumption**: converts battery energy use into an
  equivalent fuel-consumption figure for overall comparison.

## 🎮 Control Strategies (`src/baselines/`)

* **`rule_based.py`** -- simple baseline: load-point-shifting / regen
  torque split only, engine always on (`state_CE=1`).
* **`advanced_rule_based.py`** -- gear-dependent rules with dynamic
  engine on/off (electric driving mode), SoC-aware thresholds, and
  speed/acceleration/torque-rate corrections. This is the strategy that
  the future RL agent (Phase 3) will be benchmarked against.

---

## 🛠️ Tech Stack & Dependencies

* **Core Language:** Python 3.11+
* **Scientific Computing:** NumPy, SciPy (current, Phase 1)
* **Deep Learning Framework (planned):** PyTorch
* **RL Framework (planned):** Gymnasium (OpenAI Gym standard API)
* **Visualization (planned):** Matplotlib, TensorBoard

---

## Running the Environment (Phase 1)

From the project root (with the virtual environment activated):

```bash
# Baseline rule-based controller
python -m src.evaluate --cycle NEDC
python -m src.evaluate --cycle FTP75

# Advanced rule-based controller
python -m src.evaluate_advanced --cycle NEDC
python -m src.evaluate_advanced --cycle FTP75
```

Each prints the final fuel consumption (L/100km), equivalent fuel
consumption, final battery charge, and state of charge for the full
driving cycle.

## Testing

```bash
python -m pytest tests/ -v
```

---

## Key Implementation Notes (Phase 1 validation findings)

A number of subtle discrepancies between the Simulink model and an initial
direct port were found and corrected during validation:

- **Gearbox torque conversion**: `t_mgb` requires division by the total
  gear ratio `i_gt` (wheel-side torque -> flywheel-side torque), in
  addition to efficiency/friction terms.
- **`w_wheel`** is computed from the *averaged* speed `v_a =
  0.5*(v[t]+v[t-1])`, not the instantaneous speed `v[t]`.
- **`dv`** is recomputed locally as a backward difference of `v`
  (`dv[t] = v[t] - v[t-1]`); the `dv` column in the driving-cycle CSVs is
  shifted by one sample relative to `v`.
- **`x_tot`** (total distance) is computed locally via trapezoidal
  integration of `v_a`; the CSV's `x_tot` column does not match MATLAB's
  internal distance integrator.
- **Engine idle power**: when `0 < w_gear <= w_idle` (includes the entire
  NEDC startup/low-speed phase), fuel power is held at the constant
  `P_CE_idle = 8000 W`, *not* the consumption-map value. At true
  standstill (`w_gear == 0`), fuel power is `0`.
- **Engine consumption map**: the 2-D lookup table is indexed by
  **(speed, torque)** = `(w_CE_row, T_CE_col)` directly -- there is *no*
  mean-effective-pressure conversion in the original model.
- **Air density**: `rho = 1.18 kg/m^3` (not the commonly-used `1.2`).
- **Cold-start factor**: `k_cs = 1.15` is applied to the final
  `V_liter` output.
- **Driving cycle CSVs** require one extra trailing row (`time_s =
  length+1, v=0, dv=0, gear=0, x_tot=<final x_tot>`) to match MATLAB's
  sample count (1221 samples for NEDC, vs. 1220 rows of "real" data).
