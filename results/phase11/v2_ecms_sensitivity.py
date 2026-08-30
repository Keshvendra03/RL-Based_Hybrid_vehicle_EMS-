"""
V2  --  ECMS BENCHMARK SENSITIVITY (NO TRAINING, NO RL CHANGE, NO ECMS-SOURCE CHANGE).

Re-implements ONLY the ECMS cycle loop here (mirroring src/baselines/ecms.py
run_ecms_fixed_lambda exactly: same plant blocks, same feasibility masks via
_hamiltonian_best_u, same per-step wiring as evaluate_advanced.py) so that extra
diagnostics (engine-on fraction, mean T_CE when on, regional fuel) can be
recorded WITHOUT editing the frozen ecms.py.

Sweeps:
  V2-A  lambda0 x {0.80..1.20}     (k_fb=8, grid=81)
  V2-B  k_fb in {0,4,8,16}         (lambda0 = tuned value, grid=81)
  V2-C  grid in {41,81,161,321}    (lambda0 tuned, k_fb=8)
  V2-D  CS tolerance 0.5% vs 2%    (labels applied to every V2-A point)

Outputs: results/phase11/data/v2_ecms_sensitivity.json + console tables.
"""
import json, math
import numpy as np
from pathlib import Path

from src.env.ems_env import enable_fast_interpolation
enable_fast_interpolation()

from src.env.driving_cycle import DrivingCycle
from src.env.powertrain import (VehicleDynamics, gearbox, combustion_engine,
                                electric_motor, Tank, Battery,
                                equivalent_fuel_consumption, _Q_BT_IC, _Q_BT_0,
                                _T_CUTOFF)
from src.baselines.ecms import _hamiltonian_best_u

TUNED_LAM0 = {"NEDC": 1.3125, "FTP75": 2.4062}
RULE_BASED = {"NEDC": 3.5056, "FTP75": 3.2323}
REPORTED_ECMS = {"NEDC": 3.1887, "FTP75": 2.8097}
SOC_TARGET = _Q_BT_IC / _Q_BT_0

BANDS = [("brake", -1e9, 0), ("0-15", 0, 15), ("15-30", 15, 30), ("30-35", 30, 35),
         ("35-50", 35, 50), ("50-75", 50, 75), (">75", 75, 1e9)]


def band_of(T):
    for n, lo, hi in BANDS:
        if lo <= T < hi:
            return n
    return ">75"


def run_ecms(cycle_name, lam0, k_fb=8.0, n_points=81):
    """Faithful copy of ecms.run_ecms_fixed_lambda + extra diagnostics."""
    cycle = DrivingCycle(cycle_name)
    veh, tank, batt = VehicleDynamics(), Tank(), Battery()
    veh.reset(); tank.reset(); batt.reset()
    obs = cycle.reset()
    Q_BT = _Q_BT_IC
    x_tot = 0.0
    v_prev_for_dv = 0.0
    v_liter = v_ce_equiv = 0.0
    eng_on = 0
    tce_on_sum = 0.0
    moving = 0
    band_fuel = {n: 0.0 for n, _, _ in BANDS}
    m_fuel_prev = 0.0
    while True:
        dv = (obs["v"] - v_prev_for_dv) / 1.0
        v_prev_for_dv = obs["v"]
        veh_out = veh.step(v=obs["v"], dv=dv)
        x_tot += veh_out["v_a"] * 1.0
        gb = gearbox(w_wheel=veh_out["w_wheel"], dw_wheel=veh_out["dw_wheel"],
                     t_wheel=veh_out["T_wheel"], gear=obs["i"])
        w, dwm, T = gb["w_mgb"], gb["dw_mgb"], gb["t_mgb"]
        soc = Q_BT / _Q_BT_0
        lam_eff = lam0 + k_fb * (SOC_TARGET - soc)
        u = _hamiltonian_best_u(w, dwm, T, soc, lam_eff, n_points)
        if obs["v"] == 0.0 or T == 0.0 or w <= 0.0:
            u = 0.0
        t_em = u * T
        t_ce = T - t_em
        eng = combustion_engine(w_gear=w, dw_gear=dwm, t_gear=t_ce)
        mot = electric_motor(w_gear=w, dw_gear=dwm, t_gear=t_em)
        tank_out = tank.step(p_fuel=eng["p_ce"], x_tot=x_tot)
        batt_out = batt.step(p_bt=mot["p_em"], x_tot=x_tot)
        Q_BT = batt_out["q_bt"]
        efc = equivalent_fuel_consumption(v_ce=tank_out["v_liter"], v_bt=batt_out["v_bt"])
        v_liter = tank_out["v_liter"]; v_ce_equiv = efc["v_ce_equiv"]
        if T != 0.0 and w > 0.0:
            moving += 1
            if t_ce > _T_CUTOFF:
                eng_on += 1
                tce_on_sum += t_ce
        dmf = tank.m_fuel - m_fuel_prev
        m_fuel_prev = tank.m_fuel
        band_fuel[band_of(T)] += dmf
        obs, done = cycle.step()
        if done:
            break
    soc_final = Q_BT / _Q_BT_0
    return dict(
        lam0=lam0, k_fb=k_fb, n_points=n_points,
        v_liter=round(v_liter, 4), v_ce_equiv=round(v_ce_equiv, 4),
        soc_final=round(soc_final, 4),
        d_soc_pp=round((soc_final - SOC_TARGET) * 100, 3),
        cs_05pct=bool(abs(soc_final - SOC_TARGET) <= 0.005),
        cs_2pct=bool(abs(soc_final - SOC_TARGET) <= 0.02),
        engine_on_frac=round(eng_on / max(moving, 1), 4),
        engine_on_steps=eng_on, moving_steps=moving,
        mean_tce_when_on=round(tce_on_sum / max(eng_on, 1), 2),
        band_fuel_kg={k: round(v, 6) for k, v in band_fuel.items()},
    )


def sweep(cycle):
    lam0 = TUNED_LAM0[cycle]
    res = {"cycle": cycle, "tuned_lam0": lam0, "reported": REPORTED_ECMS[cycle],
           "rule_based": RULE_BASED[cycle]}

    # --- reproduce the headline number -----------------------------------------
    base = run_ecms(cycle, lam0, 8.0, 81)
    res["reproduction"] = base
    res["reproduction_matches"] = bool(abs(base["v_ce_equiv"] - REPORTED_ECMS[cycle]) <= 0.01)

    # --- V2-A  lambda0 sensitivity ------------------------------------------------
    res["V2A_lambda0"] = []
    for mult in (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20):
        r = run_ecms(cycle, lam0 * mult, 8.0, 81)
        r["mult"] = mult
        res["V2A_lambda0"].append(r)

    # --- V2-B  k_fb sensitivity -------------------------------------------------
    res["V2B_kfb"] = []
    for kfb in (0.0, 4.0, 8.0, 16.0):
        r = run_ecms(cycle, lam0, kfb, 81)
        res["V2B_kfb"].append(r)

    # --- V2-C  grid resolution ------------------------------------------------
    res["V2C_grid"] = []
    for npt in (41, 81, 161, 321):
        r = run_ecms(cycle, lam0, 8.0, npt)
        res["V2C_grid"].append(r)

    return res


if __name__ == "__main__":
    Path("results/phase11/data").mkdir(parents=True, exist_ok=True)
    out = {}
    for cyc in ("NEDC", "FTP75"):
        print(f"\n################  {cyc}   (tuned lam0 = {TUNED_LAM0[cyc]}, "
              f"reported ECMS = {REPORTED_ECMS[cyc]}, rule-based = {RULE_BASED[cyc]})")
        r = sweep(cyc); out[cyc] = r
        b = r["reproduction"]
        print(f"  reproduce headline: v_ce_equiv={b['v_ce_equiv']}  (reported {r['reported']}, "
              f"match={r['reproduction_matches']})  SoC_end={b['soc_final']*100:.2f}%  "
              f"eng_on={b['engine_on_frac']*100:.1f}%  Tce|on={b['mean_tce_when_on']}")

        print("  -- V2-A  lambda0 x mult  (k_fb=8, grid=81)")
        print(f"     {'mult':>5} {'lam0':>8} {'v_ce':>8} {'v_liter':>8} {'SoC%':>7} {'dSoC_pp':>8} "
              f"{'CS0.5':>6} {'CS2%':>5} {'engOn%':>7} {'Tce|on':>7}")
        for x in r["V2A_lambda0"]:
            print(f"     {x['mult']:>5.2f} {x['lam0']:>8.4f} {x['v_ce_equiv']:>8.4f} "
                  f"{x['v_liter']:>8.4f} {x['soc_final']*100:>7.2f} {x['d_soc_pp']:>8.2f} "
                  f"{str(x['cs_05pct']):>6} {str(x['cs_2pct']):>5} "
                  f"{x['engine_on_frac']*100:>7.1f} {x['mean_tce_when_on']:>7.1f}")

        print("  -- V2-B  k_fb  (lam0 tuned, grid=81)")
        print(f"     {'k_fb':>5} {'v_ce':>8} {'v_liter':>8} {'SoC%':>7} {'dSoC_pp':>8} {'CS0.5':>6} {'CS2%':>5}")
        for x in r["V2B_kfb"]:
            print(f"     {x['k_fb']:>5.1f} {x['v_ce_equiv']:>8.4f} {x['v_liter']:>8.4f} "
                  f"{x['soc_final']*100:>7.2f} {x['d_soc_pp']:>8.2f} "
                  f"{str(x['cs_05pct']):>6} {str(x['cs_2pct']):>5}")

        print("  -- V2-C  action-grid resolution  (lam0 tuned, k_fb=8)")
        print(f"     {'npts':>5} {'v_ce':>8} {'v_liter':>8} {'SoC%':>7} {'dSoC_pp':>8} {'engOn%':>7} {'Tce|on':>7}")
        for x in r["V2C_grid"]:
            print(f"     {x['n_points']:>5} {x['v_ce_equiv']:>8.4f} {x['v_liter']:>8.4f} "
                  f"{x['soc_final']*100:>7.2f} {x['d_soc_pp']:>8.2f} "
                  f"{x['engine_on_frac']*100:>7.1f} {x['mean_tce_when_on']:>7.1f}")

    Path("results/phase11/data/v2_ecms_sensitivity.json").write_text(json.dumps(out, indent=2))
    print("\n[saved] results/phase11/data/v2_ecms_sensitivity.json")
