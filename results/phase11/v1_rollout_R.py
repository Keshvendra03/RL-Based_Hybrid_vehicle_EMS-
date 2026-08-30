"""
V1-C  --  rollout-level realised fuel<->battery exchange rate (DIAGNOSTIC ONLY).

R = sum_t d_fuel_L,t  /  sum_t (P_EM,t * dt)      [L/100km-equiv per Joule of P_EM]

Computed for the CONTROL SAC (3 seeds) and ECMS, over the real EMSEnv rollout,
split by: charging / motoring / regen ; engine-on / engine-off ; demand band.

Task section 5: this is a rollout-level diagnostic, NOT by itself a proof of the
costate. Reported alongside V1-B (analytical) and V1-D (decisive action ranking).

Sign convention (frozen source): p_em > 0 => discharge, p_em < 0 => charge.
Zero / near-zero P_EM steps are bucketed separately and excluded from ratios.

Outputs: results/phase11/data/v1_rollout_R.json + console summary.
"""
import copy, json
import numpy as np
from pathlib import Path

from stable_baselines3 import SAC
from src.env.ems_env import EMSEnv, SOC_TARGET, enable_fast_interpolation
from src.env.powertrain import (_Q_BT_0, _battery_energy, _T_CUTOFF,
                                combustion_engine, electric_motor)
from src.baselines.ecms import _hamiltonian_best_u

enable_fast_interpolation()

ECMS_LAM0 = {"NEDC": 1.3125, "FTP75": 2.4062}
ECMS_KFB = 8.0
CTRL = dict(action_map="modeaware_gated", k_fb=2.5,
            eq_factor={"NEDC": 0.2717, "FTP75": 0.4981}, lookahead=5)

CKPTS = {
    "NEDC": ["models_p5s0_k2.5/NEDC/sac_ems_best",
             "models_p5_k2.5/NEDC/sac_ems_best",
             "models_p5_k2.5_s2/NEDC/sac_ems_best"],
    "FTP75": ["models_p5f_k2.5_s0/FTP75/sac_ems_best",
              "models_p5f_k2.5_s1/FTP75/sac_ems_best",
              "models_p5f_k2.5_s2/FTP75/sac_ems_best"],
}
BANDS = [("brake", -1e9, 0), ("0-15", 0, 15), ("15-30", 15, 30), ("30-35", 30, 35),
         ("35-50", 35, 50), ("50-75", 50, 75), (">75", 75, 1e9)]


def band_of(T):
    for n, lo, hi in BANDS:
        if lo <= T < hi:
            return n
    return ">75"


def make_env(cycle):
    return EMSEnv(cycle, eq_factor=CTRL["eq_factor"][cycle], k_fb=CTRL["k_fb"],
                  action_map=CTRL["action_map"], lookahead=CTRL["lookahead"])


def rollout_records(cycle, kind, model=None):
    env = make_env(cycle); env.reset()
    lam0 = ECMS_LAM0[cycle]
    recs = []
    while True:
        d = env._demand
        w, dw, T = d["w_MGB"], d["dw_MGB"], d["T_MGB"]
        soc_pre = env._Q_BT / _Q_BT_0
        if kind == "sac":
            a, _ = model.predict(env._last_obs, deterministic=True)
            obs, r, term, _, info = env.step(a)
            fuelL, pem, tce = info["fuel_liters_step"], info["p_em"], info["T_CE_cmd"]
        else:  # ecms -- bypass action map, drive plant with ECMS torques
            if T == 0.0 or w <= 0.0:
                u = 0.0
            else:
                lam = lam0 + ECMS_KFB * (SOC_TARGET - soc_pre)
                u = _hamiltonian_best_u(w, dw, T, soc_pre, lam, 81)
            t_em = u * T; t_ce = T - t_em
            eng = combustion_engine(w_gear=w, dw_gear=dw, t_gear=t_ce)
            mot = electric_motor(w_gear=w, dw_gear=dw, t_gear=t_em)
            tank_out = env.tank.step(p_fuel=eng["p_ce"], x_tot=env._x_tot)
            E_prev = env._E_prev
            batt_out = env.batt.step(p_bt=mot["p_em"], x_tot=env._x_tot)
            env._Q_BT = batt_out["q_bt"]
            E_now = _battery_energy(env._Q_BT); env._E_prev = E_now
            dm_fuel = tank_out["m_dot_fuel"]
            from src.env.ems_env import K_FUEL_L_PER_KG
            fuelL = dm_fuel * K_FUEL_L_PER_KG
            pem = mot["p_em"]; tce = eng["p_ce"] and t_ce  # actual engine torque cmd
            tce = t_ce
            env._T_MGB_prev = d["T_MGB"]
            cyc_obs, term = env.cycle.step()
            if not term:
                env._precompute_demand(cyc_obs)
        recs.append(dict(band=band_of(T), soc=soc_pre, fuelL=float(fuelL),
                         pem=float(pem), tce=float(tce),
                         eng_on=bool(tce > _T_CUTOFF), moving=bool(T != 0.0 and w > 0.0)))
        if term:
            break
    return recs


def bucketise(recs):
    P_EPS = 1.0   # W ; below this |P_EM| is "idle", excluded from ratios
    def agg(sel):
        rs = [r for r in recs if sel(r)]
        sfuel = sum(r["fuelL"] for r in rs)
        spem = sum(r["pem"] for r in rs)                 # signed, J (dt=1)
        sabs = sum(abs(r["pem"]) for r in rs)
        n = len(rs)
        R_signed = sfuel / spem if abs(spem) > P_EPS else None
        R_absbatt = sfuel / sabs if sabs > P_EPS else None
        return dict(n=n, sum_fuel_L=sfuel, sum_pem_J_signed=spem,
                    sum_abs_pem_J=sabs, R_signed=R_signed, R_over_absbatt=R_absbatt)
    return dict(
        all_moving=agg(lambda r: r["moving"]),
        discharge=agg(lambda r: r["pem"] > P_EPS),
        charge=agg(lambda r: r["pem"] < -P_EPS),
        regen=agg(lambda r: r["band"] == "brake"),
        engine_on=agg(lambda r: r["eng_on"] and r["moving"]),
        engine_off=agg(lambda r: (not r["eng_on"]) and r["moving"]),
        by_band={n: agg(lambda r, n=n: r["band"] == n and r["moving"])
                 for n, _, _ in BANDS},
    )


if __name__ == "__main__":
    Path("results/phase11/data").mkdir(parents=True, exist_ok=True)
    out = {}
    for cyc in ("NEDC", "FTP75"):
        out[cyc] = {}
        # ECMS
        er = rollout_records(cyc, "ecms")
        out[cyc]["ECMS"] = bucketise(er)
        # SAC 3 seeds -> average the bucket sums
        seed_buckets = []
        for ck in CKPTS[cyc]:
            m = SAC.load(ck)
            seed_buckets.append(bucketise(rollout_records(cyc, "sac", m)))
        # merge: element-wise mean of numeric fields
        def mean_bucket(key, sub=None):
            vals = []
            for sb in seed_buckets:
                b = sb[key] if sub is None else sb[key][sub]
                vals.append(b)
            keys = vals[0].keys()
            m = {}
            for k in keys:
                xs = [v[k] for v in vals if v[k] is not None]
                m[k] = float(np.mean(xs)) if xs else None
            return m
        merged = {k: mean_bucket(k) for k in
                  ("all_moving", "discharge", "charge", "regen", "engine_on", "engine_off")}
        merged["by_band"] = {n: mean_bucket("by_band", n) for n, _, _ in BANDS}
        out[cyc]["SAC_ctrl_3seed"] = merged

    Path("results/phase11/data/v1_rollout_R.json").write_text(json.dumps(out, indent=2))

    for cyc in ("NEDC", "FTP75"):
        print(f"\n================ {cyc}   R = sum(dFuel_L) / sum(P_EM*dt)   "
              f"[larger |R| on the discharge side => stiffer effective battery price]")
        for who in ("ECMS", "SAC_ctrl_3seed"):
            b = out[cyc][who]
            print(f"  --- {who}")
            for bucket in ("all_moving", "discharge", "charge", "regen",
                           "engine_on", "engine_off"):
                x = b[bucket]
                rs = x["R_signed"]; ra = x["R_over_absbatt"]
                print(f"    {bucket:>11}: n={x['n']:>4}  sumFuel_L={x['sum_fuel_L']:>9.4f}  "
                      f"sumP_EM_J(signed)={x['sum_pem_J_signed']:>12.1f}  "
                      f"R_signed={('%.3e'%rs) if rs is not None else '  n/a  '}  "
                      f"R/|batt|={('%.3e'%ra) if ra is not None else '  n/a  '}")
        # side-by-side discharge-side comparison
        e = out[cyc]["ECMS"]["discharge"]["R_over_absbatt"]
        s = out[cyc]["SAC_ctrl_3seed"]["discharge"]["R_over_absbatt"]
        if e and s:
            print(f"  >>> discharge-side realised fuel/|battery| ratio:  SAC {s:.3e}  vs  ECMS {e:.3e}   "
                  f"(SAC/ECMS = {s/e:.3f})")
    print("\n[saved] results/phase11/data/v1_rollout_R.json")
