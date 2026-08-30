"""
11B -- ACTION-TRANSFORMATION FORENSIC (NO TRAINING, NO CODE CHANGE).

Trace the PRODUCTION path  actor output a in [-1,1]  ->  T_CE  exactly as
EMSEnv._action_to_torques implements it (modeaware_gated map + motor-envelope
clamp + SoC masks + engine over-torque guard). Compute, at representative
15-35 Nm demand states drawn from CONTROL rollouts:
  * T_CE(a) curve on a dense action grid
  * inverse: action a giving T_CE in {35,40,45,50,55,58,60} Nm
  * local sensitivity dT_CE/da around 35-60 Nm
  * continuity / monotonicity / clipping / saturation / mask boundaries
  * whether 58 Nm is reachable at all, and its policy-space neighbourhood width

Outputs: results/phase11/data/s1_11B_{CYCLE}.json + console.
"""
import copy, json, warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")

from stable_baselines3 import SAC
from src.env.ems_env import EMSEnv, U_MIN, U_MAX, map_action_to_u
from src.env.powertrain import _Q_BT_0, _T_CUTOFF

CTRL = dict(action_map="modeaware_gated", k_fb=2.5,
            eq_factor={"NEDC": 0.2717, "FTP75": 0.4981}, lookahead=5)
CKPT = {"NEDC": "models_p5s0_k2.5/NEDC", "FTP75": "models_p5f_k2.5_s0/FTP75"}
N_A = 4001
TARGET_TCE = [35, 40, 45, 50, 55, 58, 60]
BANDS = [("15-25", 15, 25), ("25-30", 25, 30), ("30-35", 30, 35)]


def collect_states(cycle, n_per_band=4):
    """Real CONTROL-rollout states, grouped by demand band."""
    m = SAC.load(f"{CKPT[cycle]}/sac_ems_best")
    env = EMSEnv(cycle, eq_factor=CTRL["eq_factor"][cycle], k_fb=CTRL["k_fb"],
                 action_map=CTRL["action_map"], lookahead=CTRL["lookahead"])
    obs, _ = env.reset()
    buckets = {n: [] for n, _, _ in BANDS}
    while True:
        d = env._demand
        w, dw, T = d["w_MGB"], d["dw_MGB"], d["T_MGB"]
        soc = env._Q_BT / _Q_BT_0
        for n, lo, hi in BANDS:
            if lo <= T < hi and w > 0 and len(buckets[n]) < 40:
                buckets[n].append(dict(w=w, dw=dw, T=T, soc=soc,
                                       demand=copy.deepcopy(d)))
        a, _ = m.predict(obs, deterministic=True)
        obs, r, term, _, info = env.step(a)
        if term:
            break
    # evenly subsample n_per_band across each bucket
    out = {}
    for n in buckets:
        b = buckets[n]
        if not b:
            out[n] = []
            continue
        idx = np.linspace(0, len(b) - 1, min(n_per_band, len(b))).astype(int)
        out[n] = [b[i] for i in idx]
    return out, env


def tce_curve(env, st, grid_a):
    """Executed T_CE for each action a, via the exact production path."""
    env._demand = st["demand"]
    env._Q_BT = st["soc"] * _Q_BT_0
    tce, tem, uu = [], [], []
    for a in grid_a:
        t_ce, t_em, u, mode = env._action_to_torques(np.array([a], np.float32))
        tce.append(t_ce); tem.append(t_em); uu.append(u)
    return np.array(tce), np.array(tem), np.array(uu)


def invert(grid_a, tce, target):
    """Smallest |a| region where the executed T_CE crosses `target` (monotone-ish)."""
    diff = tce - target
    sign = np.sign(diff)
    crossings = np.where(np.diff(sign) != 0)[0]
    if len(crossings) == 0:
        return None
    i = crossings[0]
    # linear interp
    a0, a1 = grid_a[i], grid_a[i + 1]
    t0, t1 = tce[i], tce[i + 1]
    if t1 == t0:
        return float(a0)
    return float(a0 + (target - t0) * (a1 - a0) / (t1 - t0))


def analyse_state(env, st, grid_a):
    tce, tem, uu = tce_curve(env, st, grid_a)
    T = st["T"]
    tce_max = float(tce.max()); tce_min = float(tce.min())
    a_at_max = float(grid_a[int(np.argmax(tce))])
    # monotonicity / continuity
    dtce = np.diff(tce)
    non_monotone_frac = float(np.mean(dtce > 1e-9))   # fraction of steps where T_CE INCREASES with a
    max_jump = float(np.max(np.abs(dtce)))            # largest single-grid-step jump (Nm)
    # clip / mask flats: consecutive identical tce
    flat = np.abs(dtce) < 1e-9
    longest_flat_a = float(flat.sum() * (grid_a[1] - grid_a[0])) if flat.any() else 0.0
    # inverse map for target torques
    inv = {}
    for tt in TARGET_TCE:
        a = invert(grid_a, tce, tt)
        inv[tt] = None if a is None else round(a, 4)
    reachable_58 = inv[58] is not None
    # local sensitivity dT_CE/da around the 35-60 region (where reachable)
    mask = (tce >= 33) & (tce <= 62)
    if mask.sum() > 3:
        sens = np.gradient(tce[mask], grid_a[mask])
        sens_med = float(np.median(sens)); sens_min = float(np.min(sens)); sens_max = float(np.max(sens))
    else:
        sens_med = sens_min = sens_max = None
    # policy-space widths
    def width(t_lo, t_hi):
        m = (tce >= t_lo) & (tce <= t_hi)
        return float(m.sum() * (grid_a[1] - grid_a[0])) if m.any() else 0.0
    w_50_60 = width(50, 60)
    w_58pm1 = width(57, 59)
    w_58pm5 = width(53, 63)
    a58 = inv[58]
    return dict(
        band=st.get("band"), T_MGB=round(T, 2), w=round(st["w"], 1),
        dw=round(st["dw"], 2), soc=round(st["soc"], 4),
        T_CE_reachable=[round(tce_min, 2), round(tce_max, 2)],
        a_at_T_CE_max=a_at_max,
        u_range=[round(float(uu.min()), 3), round(float(uu.max()), 3)],
        U_MIN=U_MIN, U_MAX=U_MAX,
        monotone_decreasing_in_a=bool(non_monotone_frac < 0.02),
        frac_grid_steps_TCE_increasing_with_a=round(non_monotone_frac, 4),
        max_single_step_jump_Nm=round(max_jump, 4),
        longest_flat_region_in_a=round(longest_flat_a, 4),
        inverse_action_for_T_CE=inv,
        T_CE_58_reachable=reachable_58,
        a_for_58=a58,
        dT_CE_da_around_35_60={"median": None if sens_med is None else round(sens_med, 3),
                               "min": None if sens_min is None else round(sens_min, 3),
                               "max": None if sens_max is None else round(sens_max, 3)},
        policy_space_width={"T_CE_50_60": round(w_50_60, 4),
                            "T_CE_58_pm1": round(w_58pm1, 4),
                            "T_CE_58_pm5": round(w_58pm5, 4)},
        note_58_vs_boundary=(
            "UNREACHABLE: 58 Nm exceeds max feasible T_CE (LPS clamped at U_MIN=-0.85 "
            "and/or motor-envelope cap)" if not reachable_58 else
            (f"near U_MIN boundary (a_for_58={a58:.3f}, action range starts at -1.0)"
             if a58 is not None and a58 < -0.75 else
             f"interior of the action range (a_for_58={a58:.3f})")),
    )


if __name__ == "__main__":
    grid_a = np.linspace(-1.0, 1.0, N_A)
    Path("results/phase11/data").mkdir(parents=True, exist_ok=True)
    for cyc in ("NEDC", "FTP75"):
        states, env = collect_states(cyc)
        rows = []
        for band, sts in states.items():
            for st in sts:
                st["band"] = band
                rows.append(analyse_state(env, st, grid_a))
        Path(f"results/phase11/data/s1_11B_{cyc}.json").write_text(
            json.dumps({"grid_points": N_A, "rows": rows}, indent=2))
        print(f"\n================ {cyc}   action->T_CE forensic  ({N_A}-pt action grid)")
        for r in rows:
            inv = r["inverse_action_for_T_CE"]
            print(f"  band {r['band']:>6}  T={r['T_MGB']:>5} Nm  w={r['w']:>5}  SoC={r['soc']*100:.1f}%  "
                  f"T_CE reachable {r['T_CE_reachable']}  (u in {r['u_range']}, U_MIN={U_MIN})")
            print(f"       a for T_CE: " + "  ".join(
                f"{tt}Nm->{('%.3f'%inv[tt]) if inv[tt] is not None else 'NONE'}" for tt in TARGET_TCE))
            s = r["dT_CE_da_around_35_60"]
            print(f"       monotone_dec_in_a={r['monotone_decreasing_in_a']}  "
                  f"max_step_jump={r['max_single_step_jump_Nm']}Nm  "
                  f"longest_flat(a)={r['longest_flat_region_in_a']}  "
                  f"dT_CE/da[med/min/max]=[{s['median']},{s['min']},{s['max']}]")
            w = r["policy_space_width"]
            print(f"       policy-space width: T_CE 50-60 -> {w['T_CE_50_60']}  "
                  f"58+-1Nm -> {w['T_CE_58_pm1']}  58+-5Nm -> {w['T_CE_58_pm5']}   "
                  f"|  58Nm: {r['note_58_vs_boundary']}")
    print("\n[saved] results/phase11/data/s1_11B_{NEDC,FTP75}.json")
