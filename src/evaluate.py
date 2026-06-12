"""
evaluate.py
================
Runs the full baseline powertrain chain (Driving Cycle -> Vehicle -> Gearbox
-> Control Unit (baseline rule-based) -> Combustion Engine / Electric Motor
-> Tank / Battery -> Equivalent Fuel Consumption) over a full driving cycle,
exactly mirroring the Simulink model "qss_hybrid_electric_vehicle_example".

Signal chain per timestep (matches image 1 wiring)
---------------------------------------------------
    obs           = DrivingCycle.step()           -> v, dv, i, x_tot
    veh_out       = VehicleDynamics.step(v, dv)    -> w_wheel, dw_wheel, T_wheel
    gb_out        = gearbox(w_wheel, dw_wheel, T_wheel, i)
                                                    -> w_mgb, dw_mgb, t_mgb
    cu_out        = control_unit_baseline(w_mgb, dw_mgb, t_mgb, i, v, Q_BT)
                                                    -> w_gear, dw_gear, T_CE, T_EM
    eng_out       = combustion_engine(cu_out["w_gear"], cu_out["dw_gear"], cu_out["T_CE"])
                                                    -> p_ce
    mot_out       = electric_motor(cu_out["w_gear"], cu_out["dw_gear"], cu_out["T_EM"])
                                                    -> p_em
    tank_out      = Tank.step(p_ce, x_tot)         -> v_liter
    batt_out      = Battery.step(p_em, x_tot)      -> v_bt, q_bt
    Q_BT          = batt_out["q_bt"]               (fed back to Control Unit next step)
    efc_out       = equivalent_fuel_consumption(tank_out["v_liter"], batt_out["v_bt"])
                                                    -> v_ce_equiv

Final display values (image 1) correspond to the LAST timestep's:
    tank_out["v_liter"]   -> "4.513" display
    efc_out["v_ce_equiv"] -> "4.52"  display
    batt_out["q_bt"]      -> "1.784e+04" display

Usage
-----
    python -m src.evaluate --cycle NEDC
    python -m src.evaluate --cycle FTP75

Place at: src/evaluate.py
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
from src.baselines.rule_based import control_unit_baseline, _T_MGB_TH_NEDC, _T_MGB_TH_FTP75


def run_baseline(cycle_name: str = "NEDC", verbose: bool = True) -> dict:
    """
    Run the baseline rule-based EMS over one full driving cycle.

    Returns
    -------
    dict with final-step results matching the Simulink Display blocks:
        v_liter     [L/100km]   Tank display  (~4.513 for NEDC)
        v_ce_equiv  [L/100km]   Equivalent Fuel Consumption display (~4.52)
        q_bt        [As]        Battery charge display (~1.784e+04)
        soc         [-]         Final state of charge
        sim_time    [s]         Total simulated time (~1220 for NEDC)
    """

    # T_MGB_th depends on the cycle (Slide 3-8)
    t_mgb_th = _T_MGB_TH_NEDC if cycle_name.upper().startswith("NEDC") else _T_MGB_TH_FTP75

    cycle = DrivingCycle(cycle_name)
    veh   = VehicleDynamics()
    tank  = Tank()
    batt  = Battery()

    veh.reset()
    tank.reset()
    batt.reset()
    obs = cycle.reset()

    print(f"DEBUG: first obs t={obs['t']}, v={obs['v']}")

    Q_BT: float = _Q_BT_IC  # initial battery charge fed to Control Unit

    v_liter    = 0.0
    v_ce_equiv = 0.0
    q_bt       = Q_BT
    soc        = 1.0
    t          = 0.0

    # --- DEBUG accumulators ---
    sum_T_EM = 0.0
    sum_T_MGB_pos = 0.0
    sum_T_MGB_neg = 0.0
    sum_p_em = 0.0
    sum_i_bt = 0.0
    w_mgb_min, w_mgb_max = 1e9, -1e9
    # --------------------------

    # ── x_tot computed locally via trapezoidal integration of v_a ────────
    # (matches MATLAB's "Average speed" -> distance integrator chain;
    #  CSV's x_tot column was found to be off relative to MATLAB's logged
    #  x_tot, causing large early-cycle V_liter errors)
    H_STEP: float = 1.0  # s, matches DrivingCycle dt
    x_tot: float = 0.0

    # ── dv computed locally as backward difference of v ──────────────────
    # (CSV's dv column was found to be shifted by one step relative to v:
    #  CSV dv(t) actually equals v(t+1)-v(t), not v(t)-v(t-1). MATLAB's dv
    #  at time t = v(t)-v(t-1)/h. Recompute locally rather than trust the
    #  CSV column.)
    v_prev_for_dv: float = 0.0

    while True:

        # --- DEBUG: trace obs["t"] sequence at start/end of cycle ---
        if obs["t"] <= 5 or obs["t"] >= 1217:
            print(f"LOOP obs t={obs['t']}")
        # --------------------------------------------------------------

        # ── Recompute dv as backward difference of v (see note above) ──
        dv_corrected: float = (obs["v"] - v_prev_for_dv) / H_STEP
        v_prev_for_dv = obs["v"]
        # -----------------------------------------------------------------

        # ── Vehicle ────────────────────────────────────────────────────
        veh_out = veh.step(v=obs["v"], dv=dv_corrected)

        # ── Distance accumulator (trapezoidal integration of v_a) ──────
        x_tot += veh_out["v_a"] * H_STEP
        # -----------------------------------------------------------------

        # ── Manual Gear Box ────────────────────────────────────────────
        gb_out = gearbox(
            w_wheel  = veh_out["w_wheel"],
            dw_wheel = veh_out["dw_wheel"],
            t_wheel  = veh_out["T_wheel"],
            gear     = obs["i"],
        )

        # --- DEBUG w_MGB range ---
        w_mgb_min = min(w_mgb_min, gb_out["w_mgb"])
        w_mgb_max = max(w_mgb_max, gb_out["w_mgb"])
        # -------------------------

        # ── Control Unit (baseline rule-based) ────────────────────────
        cu_out = control_unit_baseline(
            w_MGB = gb_out["w_mgb"],
            dw_MGB= gb_out["dw_mgb"],
            T_MGB = gb_out["t_mgb"],
            i     = obs["i"],
            v     = obs["v"],
            Q_BT  = Q_BT,
            t_mgb_threshold = t_mgb_th,
        )

        # --- DEBUG accumulation ---
        sum_T_EM += cu_out["T_EM"]
        if gb_out["t_mgb"] > 0:
            sum_T_MGB_pos += gb_out["t_mgb"]
        else:
            sum_T_MGB_neg += gb_out["t_mgb"]
        # --------------------------

        # --- DEBUG: compare against MATLAB log at specific timesteps ---
        _DEBUG_TS = (53, 54, 102, 600, 900, 300, 450, 750, 1050, 1150)
        # -----------------------------------------------------------------

        # ── Combustion Engine ──────────────────────────────────────────
        eng_out = combustion_engine(
            w_gear  = cu_out["w_gear"],
            dw_gear = cu_out["dw_gear"],
            t_gear  = cu_out["T_CE"],
        )

        if obs["t"] in _DEBUG_TS:
            print(f"t={obs['t']:4d}  w_MGB={gb_out['w_mgb']:9.4f}  "
                  f"T_MGB={gb_out['t_mgb']:9.4f}  u={cu_out['u']:7.4f}  "
                  f"T_CE={cu_out['T_CE']:9.4f}  T_EM={cu_out['T_EM']:9.4f}  "
                  f"P_CE={eng_out['p_ce']:10.4f}")

        # ── Electric Motor ─────────────────────────────────────────────
        mot_out = electric_motor(
            w_gear  = cu_out["w_gear"],
            dw_gear = cu_out["dw_gear"],
            t_gear  = cu_out["T_EM"],
        )

        # ── Tank ───────────────────────────────────────────────────────
        tank_out = tank.step(p_fuel=eng_out["p_ce"], x_tot=x_tot)

        # --- DEBUG: x_tot and V_liter at comparison timesteps ---
        if obs["t"] in (53, 54, 55, 102, 199):
            print(f"t={obs['t']:4d}  x_tot={x_tot:9.6f}  "
                  f"m_fuel={tank_out['m_fuel']:.8f}  V_liter={tank_out['v_liter']:.6f}")
        # ---------------------------------------------------------

        # ── Battery ────────────────────────────────────────────────────
        batt_out = batt.step(p_bt=mot_out["p_em"], x_tot=x_tot)
        Q_BT = batt_out["q_bt"]  # feedback to Control Unit next step

        # --- DEBUG accumulation v2 ---
        sum_p_em += mot_out["p_em"]
        sum_i_bt += batt_out["i_bt"]
        # -----------------------------

        # ── Equivalent Fuel Consumption ───────────────────────────────
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

    print(f"\nFINAL: x_tot={x_tot:.6f}  (MATLAB expects 10932.666667)")
    print(f"FINAL: m_fuel={tank.m_fuel:.8f}")

    results = {
        "v_liter"   : v_liter,
        "v_ce_equiv": v_ce_equiv,
        "q_bt"      : q_bt,
        "soc"       : soc,
        "sim_time"  : t,
    }

    if verbose:
        print(f"\n=== Baseline rule-based EMS — {cycle_name} ===")
        print(f"  Sim time       : {results['sim_time']:.0f} s")
        print(f"  V_liter (Tank) : {results['v_liter']:.3f} L/100km   (expected ~4.513)")
        print(f"  V_CE_equiv     : {results['v_ce_equiv']:.3f}        (expected ~4.52)")
        print(f"  Q_BT (final)   : {results['q_bt']:.3e} As           (expected ~1.784e+04)")
        print(f"  SoC (final)    : {results['soc']*100:.2f} %")

        # --- DEBUG print ---
        print(f"\n  sum(T_EM)       = {sum_T_EM:.2f}")
        print(f"  sum(T_MGB, >0)  = {sum_T_MGB_pos:.2f}")
        print(f"  sum(T_MGB, <0)  = {sum_T_MGB_neg:.2f}")
        print(f"  sum(p_em)       = {sum_p_em:.2f}")
        print(f"  sum(i_bt)       = {sum_i_bt:.2f}")
        print(f"  -1*sum(i_bt)    = {-1.0*sum_i_bt:.2f}  (should equal q_bt_final - q_bt_initial = {results['q_bt']-_Q_BT_IC:.2f})")
        print(f"  w_MGB range     = [{w_mgb_min:.2f}, {w_mgb_max:.2f}]  (map range: [0, 600])")
        # -------------------

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline rule-based EMS")
    parser.add_argument("--cycle", default="NEDC", choices=["NEDC", "FTP75"],
                         help="Driving cycle to run (default: NEDC)")
    args = parser.parse_args()

    run_baseline(args.cycle)