"""
V1-D  --  DECISIVE matched-state action-ranking test (NO TRAINING, NO CODE CHANGE).

Pass 1: run ECMS (frozen Hamiltonian, lam0 + 8*(0.5-SoC), 81-pt grid) through the
        real EMSEnv; record per traction step (w, dw, T_MGB, SoC_pre-decision).
Pass 2: drive a fresh EMSEnv along the cycle; at every PROBE_EVERY-th traction step
        overwrite _Q_BT / _E_prev to the ECMS SoC for that step, then sweep the
        action a in [-1,1] on a dense grid. For each a: deep-copy, env.step([a]),
        read the ACTUAL implemented per-step reward r(a) and the executed operating
        point (u, T_CE, T_EM, p_ce, p_em). Evaluate ECMS's Hamiltonian on the SAME
        executed points:  H(a) = p_ce(a) + lambda_ECMS(SoC)*p_em(a).

Compare argmax_a r(a)  (SAC instantaneous objective: MAX reward == MIN equiv fuel)
   vs   argmin_a H(a)   (ECMS instantaneous objective).

Outputs: results/phase11/data/v1_action_ranking_{CYCLE}.json + console summary.
"""
import copy, json
import numpy as np
from pathlib import Path

from src.env.ems_env import EMSEnv, SOC_TARGET, enable_fast_interpolation
from src.env.powertrain import _Q_BT_0, _battery_energy, _T_CUTOFF
from src.baselines.ecms import _hamiltonian_best_u

enable_fast_interpolation()   # provided API, proven exact <1e-12; runtime only

ECMS_LAM0 = {"NEDC": 1.3125, "FTP75": 2.4062}
ECMS_KFB = 8.0
CTRL = dict(action_map="modeaware_gated", k_fb=2.5,
            eq_factor={"NEDC": 0.2717, "FTP75": 0.4981}, lookahead=5)
N_A = 161
PROBE_EVERY = 6

BANDS = [("0-15", 0, 15), ("15-30", 15, 30), ("30-35", 30, 35),
         ("35-50", 35, 50), ("50-75", 50, 75), (">75", 75, 1e9)]


def band_of(T):
    for name, lo, hi in BANDS:
        if lo <= T < hi:
            return name
    return None


def make_env(cycle):
    return EMSEnv(cycle, eq_factor=CTRL["eq_factor"][cycle], k_fb=CTRL["k_fb"],
                  action_map=CTRL["action_map"], lookahead=CTRL["lookahead"])


def pass1_ecms_soc(cycle):
    """Run ECMS; return {step_index: soc_pre_decision} plus the executed u each step."""
    env = make_env(cycle); env.reset()
    lam0 = ECMS_LAM0[cycle]
    soc_by_step = {}
    step = 0
    while True:
        d = env._demand
        w, dw, T = d["w_MGB"], d["dw_MGB"], d["T_MGB"]
        soc = env._Q_BT / _Q_BT_0
        soc_by_step[step] = soc
        if T == 0.0 or w <= 0.0:
            u = 0.0
        else:
            lam = lam0 + ECMS_KFB * (SOC_TARGET - soc)
            u = _hamiltonian_best_u(w, dw, T, soc, lam, 81)
        # drive the plant with ECMS's exact (t_ce, t_em) via a temporary patch
        t_em = u * T
        t_ce = T - t_em
        env._demand_saved = None
        # replicate env.step's plant call path but with ECMS torques (bypass action map)
        from src.env.powertrain import combustion_engine, electric_motor
        eng = combustion_engine(w_gear=w, dw_gear=dw, t_gear=t_ce)
        mot = electric_motor(w_gear=w, dw_gear=dw, t_gear=t_em)
        tank_out = env.tank.step(p_fuel=eng["p_ce"], x_tot=env._x_tot)
        batt_out = env.batt.step(p_bt=mot["p_em"], x_tot=env._x_tot)
        env._Q_BT = batt_out["q_bt"]
        env._E_prev = _battery_energy(env._Q_BT)
        env._T_MGB_prev = d["T_MGB"]
        cyc_obs, done = env.cycle.step()
        step += 1
        if done:
            break
        env._precompute_demand(cyc_obs)
    return soc_by_step


def pass2_probe(cycle, soc_by_step):
    env = make_env(cycle); env.reset()
    lam0 = ECMS_LAM0[cycle]
    probes = []
    step = 0
    while True:
        d = env._demand
        w, dw, T = d["w_MGB"], d["dw_MGB"], d["T_MGB"]
        traction = (T > 0.0 and w > 0.0)
        if traction and (step % PROBE_EVERY == 0) and band_of(T) and step in soc_by_step:
            soc_pre = soc_by_step[step]
            # overwrite SoC to the ECMS trajectory value for this step
            env._Q_BT = soc_pre * _Q_BT_0
            env._E_prev = _battery_energy(env._Q_BT)
            lam_ecms = lam0 + ECMS_KFB * (SOC_TARGET - soc_pre)
            grid_a = np.linspace(-1.0, 1.0, N_A)
            r_l, fuelL_l, elecL_l, u_l, tce_l, tem_l, pce_l, pem_l, H_l = ([] for _ in range(9))
            for a in grid_a:
                cp = copy.deepcopy(env)
                _, r, _, _, info = cp.step(np.array([a], np.float32))
                pce, pem = info["p_ce"], info["p_em"]
                r_l.append(r); fuelL_l.append(info["fuel_liters_step"])
                elecL_l.append(info["elec_liters_step"]); u_l.append(info["u"])
                tce_l.append(info["T_CE_cmd"]); tem_l.append(info["T_EM_cmd"])
                pce_l.append(pce); pem_l.append(pem)
                H_l.append(pce + lam_ecms * pem)
            r_arr, H_arr = np.array(r_l), np.array(H_l)
            i_s, i_e = int(np.argmax(r_arr)), int(np.argmin(H_arr))
            probes.append(dict(
                step=step, band=band_of(T), T_MGB=float(T), w=float(w),
                soc_pre=float(soc_pre), lam_ecms=float(lam_ecms),
                a_sac=float(grid_a[i_s]), a_ecms=float(grid_a[i_e]),
                da=float(grid_a[i_s] - grid_a[i_e]),
                u_sac=float(u_l[i_s]), u_ecms=float(u_l[i_e]),
                du=float(u_l[i_s] - u_l[i_e]),
                tce_sac=float(tce_l[i_s]), tce_ecms=float(tce_l[i_e]),
                dtce=float(tce_l[i_s] - tce_l[i_e]),
                pem_sac=float(pem_l[i_s]), pem_ecms=float(pem_l[i_e]),
                sac_off=bool(tce_l[i_s] <= _T_CUTOFF), ecms_off=bool(tce_l[i_e] <= _T_CUTOFF),
                d_fuel_L=float(fuelL_l[i_s] - fuelL_l[i_e]),
                d_pem_J=float(pem_l[i_s] - pem_l[i_e]),
                # realised exchange rate at the two argmins on a common P_EM basis:
                # extra fuel (L) SAC accepts per extra Joule of battery it declines
                realised_lambda_LperkJ=(
                    float((fuelL_l[i_s] - fuelL_l[i_e]) / ((pem_l[i_e] - pem_l[i_s]) / 1000.0))
                    if abs(pem_l[i_e] - pem_l[i_s]) > 1.0 else None),
            ))
        # advance one step with a harmless action (demand is cycle-driven; SoC is
        # overwritten at probes, so the driving action does not affect the test)
        _, _, term, _, _ = env.step(np.array([0.0], np.float32))
        step += 1
        if term:
            break
    return probes


def summarise(cycle, probes):
    out = {"cycle": cycle, "n_probes": len(probes), "by_band": {}}
    for name, _, _ in BANDS:
        ps = [p for p in probes if p["band"] == name]
        if not ps:
            continue
        da = np.array([p["da"] for p in ps]); dtce = np.array([p["dtce"] for p in ps])
        rl = [p["realised_lambda_LperkJ"] for p in ps if p["realised_lambda_LperkJ"] is not None]
        out["by_band"][name] = dict(
            n=len(ps),
            da_mean=float(da.mean()), da_median=float(np.median(da)),
            dtce_mean=float(dtce.mean()), dtce_median=float(np.median(dtce)),
            frac_sac_more_engine=float(np.mean([p["tce_sac"] > p["tce_ecms"] + 1.0 for p in ps])),
            frac_sac_more_battery=float(np.mean([p["u_sac"] > p["u_ecms"] + 0.01 for p in ps])),
            frac_identical=float(np.mean([abs(p["da"]) < 2.5 * (2.0 / (N_A - 1)) for p in ps])),
            sac_off_pct=float(100 * np.mean([p["sac_off"] for p in ps])),
            ecms_off_pct=float(100 * np.mean([p["ecms_off"] for p in ps])),
            mean_tce_sac=float(np.mean([p["tce_sac"] for p in ps])),
            mean_tce_ecms=float(np.mean([p["tce_ecms"] for p in ps])),
        )
    allda = np.array([p["da"] for p in probes])
    out["overall"] = dict(
        n=len(probes), da_mean=float(allda.mean()), da_median=float(np.median(allda)),
        frac_sac_more_engine=float(np.mean([p["tce_sac"] > p["tce_ecms"] + 1.0 for p in probes])),
        frac_sac_more_battery=float(np.mean([p["u_sac"] > p["u_ecms"] + 0.01 for p in probes])),
        frac_identical=float(np.mean([abs(p["da"]) < 2.5 * (2.0/(N_A-1)) for p in probes])),
    )
    return out


if __name__ == "__main__":
    Path("results/phase11/data").mkdir(parents=True, exist_ok=True)
    for cyc in ("NEDC", "FTP75"):
        print(f"\n######## {cyc}  ({N_A}-pt grid, probe every {PROBE_EVERY} traction steps)")
        soc_by_step = pass1_ecms_soc(cyc)
        probes = pass2_probe(cyc, soc_by_step)
        summ = summarise(cyc, probes)
        Path(f"results/phase11/data/v1_action_ranking_{cyc}.json").write_text(
            json.dumps({"summary": summ, "probes": probes}, indent=2))
        print(f"  n_probes = {summ['n_probes']}")
        print(f"  {'band':>7} {'n':>4} {'da_med':>8} {'dTce_med':>9} {'SAC>eng%':>9} "
              f"{'SAC>batt%':>9} {'same%':>7} {'SACoff%':>8} {'ECMSoff%':>9} {'TceS':>7} {'TceE':>7}")
        for name, b in summ["by_band"].items():
            print(f"  {name:>7} {b['n']:>4} {b['da_median']:>8.3f} {b['dtce_median']:>9.2f} "
                  f"{100*b['frac_sac_more_engine']:>9.0f} {100*b['frac_sac_more_battery']:>9.0f} "
                  f"{100*b['frac_identical']:>7.0f} {b['sac_off_pct']:>8.0f} {b['ecms_off_pct']:>9.0f} "
                  f"{b['mean_tce_sac']:>7.1f} {b['mean_tce_ecms']:>7.1f}")
        o = summ["overall"]
        print(f"  OVERALL n={o['n']}  da_median={o['da_median']:+.3f}  "
              f"SAC-more-engine={100*o['frac_sac_more_engine']:.0f}%  "
              f"SAC-more-battery={100*o['frac_sac_more_battery']:.0f}%  "
              f"same-argmin={100*o['frac_identical']:.0f}%")
    print("\n[saved] results/phase11/data/v1_action_ranking_{NEDC,FTP75}.json")
