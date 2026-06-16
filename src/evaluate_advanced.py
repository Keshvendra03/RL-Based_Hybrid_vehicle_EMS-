"""
evaluate_advanced.py
=====================
Runs the full powertrain chain with the ADVANCED rule-based controller
(controller_Bharati_Ramawat_Sharma) instead of the simple baseline
(rule_based.control_unit_baseline).

Signal chain per timestep
---------------------------------------------------
    obs        = DrivingCycle.step()            -> v, dv, i, x_tot
    veh_out    = VehicleDynamics.step(v, dv_corrected)
                                                  -> w_wheel, dw_wheel, T_wheel
    gb_out     = gearbox(w_wheel, dw_wheel, T_wheel, i)
                                                  -> w_mgb, dw_mgb, t_mgb
    ctrl_out   = AdvancedController.step(w_mgb, dw_mgb, t_mgb, i, Q_BT, v)
                                                  -> u, state_CE, d_T_MGB
    cu_out     = control_unit_advanced(w_mgb, dw_mgb, t_mgb, u, state_CE)
                                                  -> w_gear, dw_gear, T_CE, T_EM
    eng_out    = combustion_engine(cu_out["w_gear"], cu_out["dw_gear"], cu_out["T_CE"])
                                                  -> p_ce
    mot_out    = electric_motor(cu_out["w_gear"], cu_out["dw_gear"], cu_out["T_EM"])
                                                  -> p_em
    tank_out   = Tank.step(p_ce, x_tot)          -> v_liter
    batt_out   = Battery.step(p_em, x_tot)       -> v_bt, q_bt
    Q_BT       = batt_out["q_bt"]                (fed back to controller next step)
    efc_out    = equivalent_fuel_consumption(tank_out["v_liter"], batt_out["v_bt"])
                                                  -> v_ce_equiv

NOTES (carried over from the validated baseline, evaluate.py)
---------------------------------------------------------------
  - x_tot is computed locally via trapezoidal integration of v_a
    (CSV's x_tot column was found to be inconsistent with MATLAB's).
  - dv is recomputed locally as a backward difference of v
    (CSV's dv column is shifted by one sample relative to v).
  - nedc.csv / ftp75.csv should each have one extra trailing row
    (time_s = length+1, v=0, dv=0, gear=0, x_tot=<final x_tot>) to
    match MATLAB's sample count.

Usage
-----
    python -m src.evaluate_advanced --cycle NEDC
    python -m src.evaluate_advanced --cycle FTP75

Place at: src/evaluate_advanced.py
"""

from __future__ import annotations

import argparse

from src.env.driving_cycle import DrivingCycle
from src.env.powertrain import (
    VehicleDynamics,
    gearbox,
    combustion_engine,
    electric_motor,
    Tank,
    Battery,
    equivalent_fuel_consumption,
    _Q_BT_IC,
)
from src.baselines.advanced_rule_based import AdvancedController, control_unit_advanced


def run_advanced(cycle_name: str = "NEDC", verbose: bool = True) -> dict:
    """
    Run the advanced rule-based EMS over one full driving cycle.

    Returns
    -------
    dict with final-step results matching the Simulink Display blocks:
        v_liter     [L/100km]   Tank display
        v_ce_equiv  [L/100km]   Equivalent Fuel Consumption display
        q_bt        [As]        Battery charge display
        soc         [-]         Final state of charge
        sim_time    [s]         Total simulated time
    """

    cycle = DrivingCycle(cycle_name)
    veh   = VehicleDynamics()
    ctrl  = AdvancedController(cycle_name=cycle_name)
    tank  = Tank()
    batt  = Battery()

    veh.reset()
    ctrl.reset()
    tank.reset()
    batt.reset()
    obs = cycle.reset()

    Q_BT: float = _Q_BT_IC

    v_liter    = 0.0
    v_ce_equiv = 0.0
    q_bt       = Q_BT
    soc        = 1.0
    t          = 0.0

    H_STEP: float = 1.0
    x_tot: float = 0.0
    v_prev_for_dv: float = 0.0

    while True:
        # ── Recompute dv as backward difference of v ────────────────────
        dv_corrected: float = (obs["v"] - v_prev_for_dv) / H_STEP
        v_prev_for_dv = obs["v"]

        # ── Vehicle ──────────────────────────────────────────────────────
        veh_out = veh.step(v=obs["v"], dv=dv_corrected)

        # ── Distance accumulator ────────────────────────────────────────
        x_tot += veh_out["v_a"] * H_STEP

        # ── Manual Gear Box ──────────────────────────────────────────────
        gb_out = gearbox(
            w_wheel  = veh_out["w_wheel"],
            dw_wheel = veh_out["dw_wheel"],
            t_wheel  = veh_out["T_wheel"],
            gear     = obs["i"],
        )

        # ── Controller (advanced rule-based) ────────────────────────────
        ctrl_out = ctrl.step(
            w_MGB  = gb_out["w_mgb"],
            dw_MGB = gb_out["dw_mgb"],
            T_MGB  = gb_out["t_mgb"],
            gear   = obs["i"],
            Q_BT   = Q_BT,
            v      = obs["v"],
        )

        # ── Control Unit (outer wiring) ─────────────────────────────────
        cu_out = control_unit_advanced(
            w_MGB    = gb_out["w_mgb"],
            dw_MGB   = gb_out["dw_mgb"],
            T_MGB    = gb_out["t_mgb"],
            u        = ctrl_out["u"],
            state_CE = ctrl_out["state_CE"],
        )

        # ── Combustion Engine ────────────────────────────────────────────
        eng_out = combustion_engine(
            w_gear  = cu_out["w_gear"],
            dw_gear = cu_out["dw_gear"],
            t_gear  = cu_out["T_CE"],
        )

        # ── Electric Motor ───────────────────────────────────────────────
        mot_out = electric_motor(
            w_gear  = cu_out["w_gear"],
            dw_gear = cu_out["dw_gear"],
            t_gear  = cu_out["T_EM"],
        )

        # ── Tank ─────────────────────────────────────────────────────────
        tank_out = tank.step(p_fuel=eng_out["p_ce"], x_tot=x_tot)

        # ── Battery ──────────────────────────────────────────────────────
        batt_out = batt.step(p_bt=mot_out["p_em"], x_tot=x_tot)
        Q_BT = batt_out["q_bt"]

        # ── Equivalent Fuel Consumption ─────────────────────────────────
        efc_out = equivalent_fuel_consumption(
            v_ce = tank_out["v_liter"],
            v_bt = batt_out["v_bt"],
        )

        v_liter    = tank_out["v_liter"]
        v_ce_equiv = efc_out["v_ce_equiv"]
        q_bt       = batt_out["q_bt"]
        soc        = batt_out["soc"]
        t          = obs["t"]

        obs, done = cycle.step()
        if done:
            break

    results = {
        "v_liter"   : v_liter,
        "v_ce_equiv": v_ce_equiv,
        "q_bt"      : q_bt,
        "soc"       : soc,
        "sim_time"  : t,
        "x_tot"     : x_tot,
        "m_fuel"    : tank.m_fuel,
    }

    if verbose:
        print(f"\n=== Advanced rule-based EMS — {cycle_name} ===")
        print(f"  Sim time       : {results['sim_time']:.0f} s")
        print(f"  V_liter (Tank) : {results['v_liter']:.3f} L/100km   (expected ~3.348)")
        print(f"  V_CE_equiv     : {results['v_ce_equiv']:.3f}        (expected ~3.342)")
        print(f"  Q_BT (final)   : {results['q_bt']:.3e} As           (expected ~1.815e+04)")
        print(f"  SoC (final)    : {results['soc']*100:.2f} %")
        print(f"  x_tot (final)  : {results['x_tot']:.4f} m")
        print(f"  m_fuel (final) : {results['m_fuel']:.6f} kg")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run advanced rule-based EMS")
    parser.add_argument("--cycle", default="NEDC", choices=["NEDC", "FTP75"],
                         help="Driving cycle to run (default: NEDC)")
    args = parser.parse_args()

    run_advanced(args.cycle)