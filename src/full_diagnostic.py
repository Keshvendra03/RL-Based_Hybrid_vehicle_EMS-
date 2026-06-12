"""
full_diagnostic.py
==================
ONE-SHOT diagnostic: runs the full powertrain chain and prints, at each
timestep where we have MATLAB ground truth, EVERY intermediate signal
side-by-side with the MATLAB value. The first row where Python and MATLAB
diverge tells you exactly which block has the bug.

Run from project root:
    python -m src.full_diagnostic

MATLAB GROUND TRUTH (collected over this session)
--------------------------------------------------
Indexing note: MATLAB arrays are 1-indexed and logged from k=1. The "t"
printed below is MATLAB's k-1 (so MATLAB row k = our t = k-1). The CSV /
DrivingCycle uses time_s starting at 1, so obs["t"]==N corresponds to
MATLAB row where time_s==N. We compare on obs["t"]==N vs MATLAB(time_s==N).

  t=53:  v=1.041667  dv=1.041667  gear=1
         w_wheel=1.808449  T_wheel=392.210298
         w_MGB=19.4558    T_MGB=42.4453   u=0.0000  T_CE=50.2276  T_EM=3.8912
         P_CE=8000.0  (idle: w_MGB=19.46 < w_idle=105)

  t=54:  v=2.083333  dv=1.041667  gear=1
         w_wheel=5.425347  T_wheel=392.473217
         w_MGB=58.3675    T_MGB=38.9737
         P_CE=8000.0  (idle: w_MGB=58.37 < 105)

  t=102: gear=2
         w_wheel=29.578173  T_wheel=299.507030
         w_MGB(=w_wheel*i_gt) should be 29.578173 * (1.92*3.29)=186.84
         T_MGB=48.9281    T_CE=52.1776
         P_CE=30779.0821

  x_tot(MATLAB):  t=53 -> 1.0,  t=54 -> 1.520833,  t=55 -> 3.083333

KEY OPEN QUESTIONS THIS SCRIPT ANSWERS
---------------------------------------
1. Does the live loop's (w_wheel, T_wheel) at obs["t"]==102 equal the
   MATLAB ground truth (29.578173, 299.507030)?  If NOT -> indexing bug
   between DrivingCycle obs and the rest of the chain.

2. Does T_MGB at obs["t"]==102 equal 48.9281 (it did in standalone
   validate_gearbox, but live loop printed 8.6058)?  If they differ ->
   the (v,dv) feeding the chain at obs["t"]==102 is wrong (indexing).

3. Does T_CE at t=102 equal MATLAB's 52.1776?  Python had 8.6058 ->
   if T_MGB is fixed to 48.9281, does T_CE follow?

4. With the corrected engine map (w_CE_row, T_CE_col, V_CE_map indexed
   by TORQUE t_ce directly, NOT p_me), does P_CE at t=102 match 30779?
   (This requires the powertrain.py combustion_engine fix described at
   the bottom of this file.)
"""

from __future__ import annotations

import numpy as np

from src.env.driving_cycle import DrivingCycle
from src.env.powertrain import (
    VehicleDynamics, gearbox, combustion_engine, electric_motor,
    Tank, Battery, equivalent_fuel_consumption, _Q_BT_IC,
    _GEAR_RATIOS, _DIFF_RATIO,
)
from src.baselines.rule_based import control_unit_baseline, _T_MGB_TH_NEDC


# MATLAB ground truth keyed by obs["t"]
GT = {
    53:  dict(v=1.041667, dv=1.041667, gear=1, w_wheel=1.808449, T_wheel=392.210298,
             w_MGB=19.4558, T_MGB=42.4453, u=0.0, T_CE=50.2276, P_CE=8000.0),
    54:  dict(v=2.083333, dv=1.041667, gear=1, w_wheel=5.425347, T_wheel=392.473217,
             w_MGB=58.3675, T_MGB=38.9737, u=0.0, T_CE=None, P_CE=8000.0),
    102: dict(v=None, dv=0.7407500000000002, gear=2, w_wheel=29.578173, T_wheel=299.507030,
             w_MGB=186.84, T_MGB=48.9281, u=None, T_CE=52.1776, P_CE=30779.0821),
}


def fmt(label, py, ml):
    if ml is None:
        return f"    {label:12s} py={py:12.4f}   ml=    --"
    err = "" if ml == 0 else f"  err={100*(py-ml)/ml:+7.2f}%"
    flag = "" if (ml is not None and abs(py-ml) < max(1e-3, abs(ml)*0.01)) else "   <<< MISMATCH"
    return f"    {label:12s} py={py:12.4f}   ml={ml:12.4f}{err}{flag}"


def main():
    cycle = DrivingCycle("NEDC")
    veh   = VehicleDynamics()
    tank  = Tank()
    batt  = Battery()
    veh.reset(); tank.reset(); batt.reset()
    obs = cycle.reset()

    Q_BT = _Q_BT_IC
    x_tot = 0.0
    H = 1.0
    v_prev_for_dv = 0.0

    while True:
        dv_corrected = (obs["v"] - v_prev_for_dv) / H
        v_prev_for_dv = obs["v"]

        veh_out = veh.step(v=obs["v"], dv=dv_corrected)
        x_tot += veh_out["v_a"] * H

        gb_out = gearbox(
            w_wheel=veh_out["w_wheel"], dw_wheel=veh_out["dw_wheel"],
            t_wheel=veh_out["T_wheel"], gear=obs["i"],
        )
        cu_out = control_unit_baseline(
            w_MGB=gb_out["w_mgb"], dw_MGB=gb_out["dw_mgb"], T_MGB=gb_out["t_mgb"],
            i=obs["i"], v=obs["v"], Q_BT=Q_BT, t_mgb_threshold=_T_MGB_TH_NEDC,
        )
        eng_out = combustion_engine(
            w_gear=cu_out["w_gear"], dw_gear=cu_out["dw_gear"], t_gear=cu_out["T_CE"],
        )
        mot_out = electric_motor(
            w_gear=cu_out["w_gear"], dw_gear=cu_out["dw_gear"], t_gear=cu_out["T_EM"],
        )
        tank_out = tank.step(p_fuel=eng_out["p_ce"], x_tot=x_tot)
        batt_out = batt.step(p_bt=mot_out["p_em"], x_tot=x_tot)
        Q_BT = batt_out["q_bt"]

        if obs["t"] in GT:
            g = GT[obs["t"]]
            i_gt = _GEAR_RATIOS[obs["i"]-1]*_DIFF_RATIO if obs["i"] >= 1 else 0.0
            print(f"\n===== obs['t'] = {obs['t']}  (gear={obs['i']}, i_gt={i_gt:.4f}) =====")
            print(f"  -- DrivingCycle obs --")
            print(f"    v            py={obs['v']:12.6f}" + (f"   ml={g['v']:12.6f}" if g['v'] else "   ml=    --"))
            print(f"    dv           py={dv_corrected:12.6f}" + (f"   ml={g['dv']:12.6f}" if g['dv'] else "   ml=    --"))
            print(f"    gear         py={obs['i']:12d}   ml={g['gear']:12d}")
            print(f"  -- VehicleDynamics --")
            print(fmt("w_wheel", veh_out["w_wheel"], g["w_wheel"]))
            print(fmt("T_wheel", veh_out["T_wheel"], g["T_wheel"]))
            print(f"    v_a          py={veh_out['v_a']:12.6f}   (x_tot={x_tot:.4f})")
            print(f"  -- Gearbox --")
            print(fmt("w_MGB", gb_out["w_mgb"], g["w_MGB"]))
            print(fmt("T_MGB", gb_out["t_mgb"], g["T_MGB"]))
            print(f"  -- Control Unit --")
            print(fmt("u", cu_out["u"], g["u"]))
            print(f"  -- Combustion Engine --")
            print(f"    t_gear(in)   py={cu_out['T_CE']:12.4f}")
            print(f"    dw_gear(in)  py={cu_out['dw_gear']:12.4f}")
            print(fmt("t_ce(after)", eng_out["t_ce"], g["T_CE"]))
            print(f"    w_ce         py={eng_out['w_ce']:12.4f}")
            print(f"    is_idle      py={str(eng_out['is_idle']):>12s}")
            print(f"    is_cutoff    py={str(eng_out['is_cutoff']):>12s}")
            print(fmt("P_CE", eng_out["p_ce"], g["P_CE"]))
            print(f"    v_dot        py={eng_out['v_dot']:12.8f}")

        obs, done = cycle.step()
        if done:
            break

    print("\n\nDONE. Read DOWN each block: the FIRST block with '<<< MISMATCH'")
    print("is where the bug is. Everything above it matches MATLAB.")


if __name__ == "__main__":
    main()