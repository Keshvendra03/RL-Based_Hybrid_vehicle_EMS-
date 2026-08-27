"""
phase6_ab_full.py
==================
CLOSES the Phase-6 gaps identified in the completion audit:
  - sections G/H/I/J run for BOTH cycles (was NEDC-only)
  - section J: explicit A/B/C/D actor-critic classification + P(ASSIST)/P(LPS)
  - section L: full per-seed metric list incl. SoC min/max, REGEN%, torques
  - section N: matched-state SAC-vs-ECMS for the ACTUAL Phase-6 checkpoints
  - section W: writes the exact requested file set under results/phase6/

    python -m results.phase6_ab_full --cycle NEDC
    python -m results.phase6_ab_full --cycle FTP75

No training. Uses existing checkpoints/replay buffers only.
"""
from __future__ import annotations

import argparse
import copy
import json
import types
from pathlib import Path

import numpy as np
import torch as th
from scipy.stats import norm

from src.env.ems_env import EMSEnv, U_MIN, U_MAX, ZB_MODEAWARE, SOC_TARGET, _EPS_T, map_action_to_u
from src.env.powertrain import (_T_CUTOFF, _interp1d_linear, _w_EM_max_row,
                                _T_EM_max_arr, _THETA_EM)
from src.agents.targeted_exploration import _off_reachable, _a_off

EQF = {"NEDC": 0.2717, "FTP75": 0.4981}
LAM0 = {"NEDC": 1.3125, "FTP75": 2.4062}
BENCH = {"NEDC": 3.5056, "FTP75": 3.2323}
AMAP = "modeaware_gated"
TB = [(0, 15, "0-15"), (15, 30, "15-30"), (30, 35, "30-35"),
      (35, 50, "35-50"), (50, 75, "50-75"), (75, 1e9, ">75")]
SB = [(0.0, 0.40, "<40"), (0.40, 0.50, "40-50"), (0.50, 0.55, "50-55"), (0.55, 1.0, ">55")]
CONFIG = {
    "NEDC": {"CONTROL": ["models_p5s0_k2.5", "models_p5_k2.5", "models_p5_k2.5_s2"],
             "TREATMENT": ["models_p6_trt_N0", "models_p6_trt_N1", "models_p6_trt_N2"]},
    "FTP75": {"CONTROL": ["models_p5f_k2.5_s0", "models_p5f_k2.5_s1", "models_p5f_k2.5_s2"],
              "TREATMENT": ["models_p6_trt_F0", "models_p6_trt_F1", "models_p6_trt_F2"]},
}


def mode_of(a, T, w, dw):
    if T <= 0:
        return "REGEN" if T < 0 else "stop"
    u = map_action_to_u(float(a), T, AMAP, w, dw)
    cap = max(_interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w) - abs(_THETA_EM * dw) - _EPS_T, 0.0)
    t_em = float(np.clip(u * T, -cap, cap))
    return "OFF" if (T - t_em) <= _T_CUTOFF else ("LPS" if t_em < 0 else ("ASSIST" if t_em > 0 else "ONLY"))


def classify_final(mode, t_ce):
    if mode == "stop":
        return "stop"
    if mode == "regen":
        return "REGEN"
    if t_ce <= _T_CUTOFF:
        return "OFF"
    return {"assist": "ASSIST", "lps_gen": "LPS"}.get(mode, "ONLY")


# ----------------------------- section G ----------------------------------- #
def coverage(model, P):
    rb = model.replay_buffer
    n = rb.size()
    obs = rb.observations[:n, 0, :]; act = rb.actions[:n, 0, 0]
    T = obs[:, 2] * 150.0; w = obs[:, 0] * 300.0
    dw = obs[:, 1] * 60.0; soc = (obs[:, 4] + 1.0) / 2.0
    md = np.array([mode_of(act[i], T[i], w[i], dw[i]) for i in range(n)])
    out = {"total_transitions": int(n)}
    rows = []
    P(f"  {'T band':>8}{'SoC':>7}{'count':>9}{'%buf':>7}{'OFF n':>8}{'OFF%':>7}{'ASST%':>7}{'LPS%':>7}{'feasOFF%':>10}")
    for lo, hi, tn in TB:
        for slo, shi, sn in SB:
            m = (T >= lo) & (T < hi) & (soc >= slo) & (soc < shi)
            c = int(m.sum())
            if c < 30:
                continue
            sub = md[m]
            idx = np.where(m)[0][:: max(1, c // 300)]
            feas = 100.0 * np.mean([_off_reachable(T[i], w[i], dw[i]) for i in idx])
            offn = int(np.sum(sub == "OFF"))
            cell = dict(t_band=tn, soc_band=sn, count=c, pct_of_buffer=100 * c / n,
                       off_n=offn, off_pct=100 * offn / c,
                       assist_pct=100 * np.mean(sub == "ASSIST"),
                       lps_pct=100 * np.mean(sub == "LPS"), feasible_off_pct=feas)
            rows.append(cell)
            P(f"  {tn:>8}{sn:>7}{c:>9,}{100*c/n:>6.1f}%{offn:>8,}{100*offn/c:>6.1f}%"
              f"{100*np.mean(sub=='ASSIST'):>6.1f}%{100*np.mean(sub=='LPS'):>6.1f}%{feas:>9.1f}%")
    out["cells"] = rows
    return out


# ----------------------------- matched states ------------------------------ #
def matched_states(cycle, kfb=2.5):
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=kfb, lookahead=5, action_map=AMAP)
    obs, _ = env.reset()
    S = []
    while True:
        d = env._demand
        if d["T_MGB"] > _T_CUTOFF and d["w_MGB"] > 0:
            S.append((obs.copy(), copy.deepcopy(env), d["T_MGB"], d["w_MGB"],
                      d["dw_MGB"], (obs[4] + 1) / 2, d["gear"], d["v"]))
        obs, r, t, _, i = env.step(np.zeros(1, np.float32))
        if t:
            return S


def q_at(model, ob, acts):
    ot = th.as_tensor(np.repeat(ob.reshape(1, -1), len(acts), 0)).float().to(model.device)
    at = th.as_tensor(np.asarray(acts).reshape(-1, 1)).float().to(model.device)
    with th.no_grad():
        q = model.critic(ot, at)
    return np.minimum(q[0].cpu().numpy().ravel(), q[1].cpu().numpy().ravel())


def actor_at(model, ob):
    ot = th.as_tensor(ob.reshape(1, -1)).float().to(model.device)
    with th.no_grad():
        mu, ls, _ = model.actor.get_action_dist_params(ot)
    return float(mu.cpu().numpy().ravel()[0]), float(np.exp(ls.cpu().numpy().ravel()[0]))


def stats(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), median=float(np.median(x)), std=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                p10=float(np.percentile(x, 10)), p25=float(np.percentile(x, 25)),
                p75=float(np.percentile(x, 75)), p90=float(np.percentile(x, 90)),
                pos_pct=float(100.0 * np.mean(x > 0)))


# ------------------------- sections H, I, J --------------------------------- #
def hij_forensics(model, S, region, cycle, P, grid):
    lo, hi = region
    sel = [s for s in S if lo <= s[2] < hi and 0.40 <= s[5] < 0.50]
    if len(sel) < 20:
        sel = [s for s in S if lo <= s[2] < hi]
    sel = sel[:: max(1, len(sel) // 120)][:120]
    if not sel:
        return None
    dqa, dql, dr, poff, passt, plps, disp, cls_ = [], [], [], [], [], [], [], []
    for ob, sn, T, w, dw, soc, gear, v in sel:
        aoff = _a_off(T, w, dw, AMAP)
        a_zero_lps_boundary = -0.30  # representative ASSIST/LPS probe points
        probe_off, probe_asst, probe_lps = min(1.0, aoff + 0.05), 0.40, -0.50
        q = q_at(model, ob, [probe_off, probe_asst, probe_lps])
        dqa.append(q[0] - q[1]); dql.append(q[0] - q[2])
        mu, sd = actor_at(model, ob)
        z_off = (np.arctanh(np.clip(aoff, -.999999, .999999)) - mu) / sd
        poff.append(norm.sf(z_off))
        # P(ASSIST): mass between a_zero (u=0) and a_off ; P(LPS): mass below a_zero
        a_zero = 2 * (0.0 - U_MIN) / (U_MAX - U_MIN) - 1
        z_zero = (np.arctanh(np.clip(a_zero, -.999999, .999999)) - mu) / sd
        passt.append(norm.cdf(z_off) - norm.cdf(z_zero))
        plps.append(norm.cdf(z_zero))
        qg = q_at(model, ob, grid)
        a_star = grid[qg.argmax()]
        d = abs(np.tanh(mu) - a_star) / 2.0
        disp.append(d)
        spread = qg.max() - qg.min()
        if spread < 0.005:
            cls_.append("D_flat")
        elif d < 0.10:
            cls_.append("A_aligned")
        else:
            cls_.append("B_displaced" if a_star >= aoff else "C_Q_prefers_non_OFF")
        e1 = copy.deepcopy(sn); _, r1, _, _, _ = e1.step(np.array([probe_off], np.float32))
        e2 = copy.deepcopy(sn); _, r2, _, _, _ = e2.step(np.array([probe_asst], np.float32))
        dr.append(r1 - r2)
    dqa, dr = np.array(dqa), np.array(dr)
    cls_ = np.array(cls_)
    c1 = 100 * np.mean((dr > 0) & (dqa > 0)); c2 = 100 * np.mean((dr > 0) & (dqa <= 0))
    c3 = 100 * np.mean((dr <= 0) & (dqa > 0)); c4 = 100 * np.mean((dr <= 0) & (dqa <= 0))
    out = dict(n=len(sel), dQ_off_assist=stats(dqa), dQ_off_lps=stats(dql), dr_off_assist=stats(dr),
              p_off=float(100 * np.mean(poff)), p_assist=float(100 * np.mean(np.clip(passt, 0, 1))),
              p_lps=float(100 * np.mean(plps)), mean_displacement=float(np.mean(disp)),
              A_aligned_pct=float(100 * np.mean(cls_ == "A_aligned")),
              B_displaced_pct=float(100 * np.mean(cls_ == "B_displaced")),
              C_Q_prefers_non_OFF_pct=float(100 * np.mean(cls_ == "C_Q_prefers_non_OFF")),
              D_flat_pct=float(100 * np.mean(cls_ == "D_flat")),
              reward_Q_classification=dict(r_OFF_Q_OFF=c1, r_OFF_Q_ASSIST=c2, r_ASSIST_Q_OFF=c3, r_ASSIST_Q_ASSIST=c4))
    P(f"    n={out['n']}")
    P(f"    dQ(OFF-ASSIST): mean={out['dQ_off_assist']['mean']:+.4f} median={out['dQ_off_assist']['median']:+.4f} "
      f"std={out['dQ_off_assist']['std']:.4f} p10={out['dQ_off_assist']['p10']:+.4f} p25={out['dQ_off_assist']['p25']:+.4f} "
      f"p75={out['dQ_off_assist']['p75']:+.4f} p90={out['dQ_off_assist']['p90']:+.4f} >0:{out['dQ_off_assist']['pos_pct']:.0f}%")
    P(f"    dQ(OFF-LPS):    mean={out['dQ_off_lps']['mean']:+.4f} >0:{out['dQ_off_lps']['pos_pct']:.0f}%")
    P(f"    dr(OFF-ASSIST): mean={out['dr_off_assist']['mean']:+.4f} >0:{out['dr_off_assist']['pos_pct']:.0f}%")
    P(f"    P(OFF)={out['p_off']:.1f}%  P(ASSIST)={out['p_assist']:.1f}%  P(LPS)={out['p_lps']:.1f}%")
    P(f"    actor-Q mean displacement={out['mean_displacement']:.3f}")
    P(f"    A_aligned={out['A_aligned_pct']:.1f}%  B_displaced={out['B_displaced_pct']:.1f}%  "
      f"C_Q_prefers_non_OFF={out['C_Q_prefers_non_OFF_pct']:.1f}%  D_flat={out['D_flat_pct']:.1f}%")
    P(f"    r/Q: [r=OFF,Q=OFF]={c1:.0f}%  [r=OFF,Q=ASSIST]={c2:.0f}%  [r=ASST,Q=OFF]={c3:.0f}%  [r=ASST,Q=ASST]={c4:.0f}%")
    return out


# ------------------------------- section L ---------------------------------- #
def vehicle_metrics(ckpt, cycle, kfb):
    from stable_baselines3 import SAC
    m = SAC.load(ckpt)
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=kfb, lookahead=5, action_map=AMAP)
    obs, _ = env.reset()
    socs, R = [], []
    while True:
        a, _ = m.predict(obs, deterministic=True)
        obs, r, t, _, i = env.step(a)
        socs.append(i["soc"])
        R.append(dict(cls=classify_final(i["mode"], i["T_CE_cmd"]), t_ce=i["T_CE_cmd"],
                      t_em=i["T_EM_cmd"], p_em=i["p_em"], fuel=i["fuel_liters_step"],
                      elec=i["elec_liters_step"]))
        if t:
            fin = i["episode_final"]; break
    mv = [x for x in R if x["cls"] != "stop"]
    f = lambda k: 100.0 * np.mean([x["cls"] == k for x in mv]) if mv else 0.0
    socs = np.array(socs)
    return dict(v_ce_equiv=fin["v_ce_equiv"], v_liter=fin["v_liter"],
               elec_equiv=fin["v_ce_equiv"] - fin["v_liter"],
               soc_final=fin["soc_final"], d_soc_pp=(fin["soc_final"] - 0.5) * 100,
               soc_min=float(socs.min() * 100), soc_max=float(socs.max() * 100),
               off_pct=f("OFF"), assist_pct=f("ASSIST"), lps_pct=f("LPS"), regen_pct=f("REGEN"),
               engine_on_s=sum(1 for x in R if x["t_ce"] > _T_CUTOFF),
               mean_abs_t_ce=float(np.mean([abs(x["t_ce"]) for x in mv])) if mv else 0.0,
               mean_abs_t_em=float(np.mean([abs(x["t_em"]) for x in mv])) if mv else 0.0,
               constraint_violations=sum(1 for s in socs if not (0.05 <= s <= 0.95)))


# ------------------------------- section N ----------------------------------- #
def ecms_rollout(cycle):
    from src.baselines.ecms import _hamiltonian_best_u
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=2.5, lookahead=0)
    lam0 = LAM0[cycle]

    def patched(self, action):
        d = self._demand
        w, dw, T = d["w_MGB"], d["dw_MGB"], d["T_MGB"]
        soc = self._Q_BT / 36000.0
        if T == 0.0 or w <= 0.0:
            return 0.0, 0.0, 0.0, "stop"
        u = _hamiltonian_best_u(w, dw, T, soc, lam0 + 8.0 * (SOC_TARGET - soc), 81)
        te = u * T
        m = "regen" if T < 0 else ("lps_gen" if te < 0 else ("assist" if te > 0 else "engine"))
        return T - te, te, u, m
    env._action_to_torques = types.MethodType(patched, env)
    obs, _ = env.reset()
    R = []
    while True:
        d = dict(env._demand)
        obs, r, t, _, i = env.step(np.zeros(1, np.float32))
        R.append(dict(T=d["T_MGB"], soc=i["soc"], u=i["u"], t_ce=i["T_CE_cmd"], t_em=i["T_EM_cmd"],
                      cls=classify_final(i["mode"], i["T_CE_cmd"]), fuel=i["fuel_liters_step"],
                      elec=i["elec_liters_step"]))
        if t:
            R[-1]["final"] = i["episode_final"]; return R


def sac_rollout_full(ckpt, cycle, kfb):
    from stable_baselines3 import SAC
    m = SAC.load(ckpt)
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=kfb, lookahead=5, action_map=AMAP)
    obs, _ = env.reset()
    R = []
    while True:
        d = dict(env._demand)
        a, _ = m.predict(obs, deterministic=True)
        obs, r, t, _, i = env.step(a)
        R.append(dict(T=d["T_MGB"], soc=i["soc"], u=i["u"], t_ce=i["T_CE_cmd"], t_em=i["T_EM_cmd"],
                      cls=classify_final(i["mode"], i["T_CE_cmd"]), fuel=i["fuel_liters_step"],
                      elec=i["elec_liters_step"]))
        if t:
            R[-1]["final"] = i["episode_final"]; return R


def ecms_comparison(arm_ckpt, cycle, kfb, P):
    SAC_R = sac_rollout_full(arm_ckpt, cycle, kfb)
    ECMS_R = ecms_rollout(cycle)
    dT = max(abs(a["T"] - b["T"]) for a, b in zip(SAC_R, ECMS_R))
    P(f"    demand alignment check: max|T_SAC - T_ECMS| = {dT:.2e}")
    K = 1e5 / ECMS_R[-1]["final"]["x_tot_m"]
    regs = [("0-15", lambda x: 0 <= x["T"] < 15), ("15-30", lambda x: 15 <= x["T"] < 30),
            ("30-50", lambda x: 30 <= x["T"] < 50), ("50-75", lambda x: 50 <= x["T"] < 75),
            (">75", lambda x: x["T"] >= 75)]
    out = {"summary": dict(sac_v_ce=SAC_R[-1]["final"]["v_ce_equiv"], ecms_v_ce=ECMS_R[-1]["final"]["v_ce_equiv"],
                           sac_soc=SAC_R[-1]["final"]["soc_final"], ecms_soc=ECMS_R[-1]["final"]["soc_final"]),
          "regions": {}}
    P(f"    SAC V_CE_equiv={SAC_R[-1]['final']['v_ce_equiv']:.4f}  ECMS={ECMS_R[-1]['final']['v_ce_equiv']:.4f}  "
      f"SAC SoC={SAC_R[-1]['final']['soc_final']*100:.2f}%  ECMS SoC={ECMS_R[-1]['final']['soc_final']*100:.2f}%")
    P(f"    {'region':>8}{'dFuel':>10}{'dElec':>10}{'SAC OFF%':>10}{'ECMS OFF%':>11}{'SAC eng|T|':>11}{'ECMS eng|T|':>12}")
    for nm, f in regs:
        ia = [i for i, x in enumerate(SAC_R) if f(x)]
        ie = [i for i, x in enumerate(ECMS_R) if f(x)]
        if not ia:
            continue
        df = (sum(SAC_R[i]["fuel"] for i in ia) - sum(ECMS_R[i]["fuel"] for i in ie)) * K
        de = (sum(SAC_R[i]["elec"] for i in ia) - sum(ECMS_R[i]["elec"] for i in ie)) * K
        so = 100 * np.mean([SAC_R[i]["cls"] == "OFF" for i in ia])
        eo = 100 * np.mean([ECMS_R[i]["cls"] == "OFF" for i in ie])
        sace = np.mean([abs(SAC_R[i]["t_ce"]) for i in ia if SAC_R[i]["cls"] != "OFF"] or [0])
        ecmse = np.mean([abs(ECMS_R[i]["t_ce"]) for i in ie if ECMS_R[i]["cls"] != "OFF"] or [0])
        out["regions"][nm] = dict(dfuel=df, delec=de, sac_off_pct=so, ecms_off_pct=eo)
        P(f"    {nm:>8}{df:>+10.4f}{de:>+10.4f}{so:>9.1f}%{eo:>10.1f}%{sace:>11.1f}{ecmse:>12.1f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, choices=["NEDC", "FTP75"])
    ap.add_argument("--out", default="results/phase6")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    from stable_baselines3 import SAC

    cyc = a.cycle
    dirs = CONFIG[cyc]
    fh = open(out / f"phase6_forensics_{cyc}_FULL.txt", "w", encoding="utf-8")
    P = lambda s: (print(s), fh.write(s + "\n"))
    P(f"PHASE 6 FULL A/B FORENSICS -- {cyc} (closes completion-audit gaps)")

    # ---- section G: coverage for BOTH arms ----
    cov = {}
    for arm, ds in dirs.items():
        d0 = ds[0]
        m = SAC.load(f"{d0}/{cyc}/sac_ems_best")
        m.load_replay_buffer(f"{d0}/{cyc}/replay_buffer.pkl")
        P(f"\n{'='*100}\nSECTION G -- REPLAY COVERAGE, {arm} ({cyc}, seed0 buffer)\n{'='*100}")
        cov[arm] = coverage(m, P)
        json.dump(cov[arm], open(out / f"replay_coverage_{arm.lower()}_{cyc}.json", "w"), indent=2)

    # ---- sections H/I/J for BOTH arms, 3 regions ----
    S = matched_states(cyc)
    grid = np.linspace(-1, 1, 61)
    models = {arm: SAC.load(f"{ds[0]}/{cyc}/sac_ems_best") for arm, ds in dirs.items()}
    qres = {}
    for region, rn in [((15, 30), "15-30"), ((30, 35), "30-35"), ((35, 50), "35-50")]:
        P(f"\n{'='*100}\nSECTIONS H/I/J -- {rn} Nm ({cyc})\n{'='*100}")
        qres[rn] = {}
        for arm, m in models.items():
            P(f"\n  --- {arm} ---")
            qres[rn][arm] = hij_forensics(m, S, region, cyc, P, grid)
    json.dump(qres, open(out / f"matched_q_{cyc}.json", "w"), indent=2)

    # ---- section L: full vehicle metrics, all seeds, both arms ----
    P(f"\n{'='*100}\nSECTION L -- FULL VEHICLE METRICS ({cyc})\n{'='*100}")
    kfb = 2.5
    vres = {}
    for arm, ds in dirs.items():
        vres[arm] = []
        for i, d in enumerate(ds):
            vm = vehicle_metrics(f"{d}/{cyc}/sac_ems_best", cyc, kfb)
            vm["seed"] = i
            vres[arm].append(vm)
            P(f"  {arm} seed{i}: V_CE={vm['v_ce_equiv']:.4f} V_l={vm['v_liter']:.4f} "
              f"elec_equiv={vm['elec_equiv']:.4f} SoC_final={vm['soc_final']*100:.2f}% "
              f"dSoC={vm['d_soc_pp']:+.2f}pp SoCmin={vm['soc_min']:.1f}% SoCmax={vm['soc_max']:.1f}% "
              f"OFF={vm['off_pct']:.1f}% ASST={vm['assist_pct']:.1f}% LPS={vm['lps_pct']:.1f}% "
              f"REGEN={vm['regen_pct']:.1f}% engON={vm['engine_on_s']}s "
              f"meanTce={vm['mean_abs_t_ce']:.1f} meanTem={vm['mean_abs_t_em']:.1f} "
              f"viol={vm['constraint_violations']}")
        v = np.array([r["v_ce_equiv"] for r in vres[arm]])
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        P(f"  {arm} summary: mean={v.mean():.4f} std={v.std(ddof=1):.4f} min={v.min():.4f} "
          f"max={v.max():.4f} 95%CI=[{v.mean()-ci:.4f},{v.mean()+ci:.4f}]")
    json.dump(vres, open(out / f"{'control' if False else 'vehicle'}_metrics_{cyc}.json", "w"), indent=2)
    json.dump(vres.get("CONTROL", []), open(out / f"control_metrics_{cyc}.json", "w"), indent=2)
    json.dump(vres.get("TREATMENT", []), open(out / f"treatment_metrics_{cyc}.json", "w"), indent=2)

    # ---- section N: matched-state SAC-vs-ECMS for THIS phase's actual checkpoints ----
    P(f"\n{'='*100}\nSECTION N -- MATCHED-STATE SAC vs ECMS ({cyc})\n{'='*100}")
    ecms_res = {}
    for arm, ds in dirs.items():
        P(f"\n  --- {arm} (seed0) vs ECMS ---")
        ecms_res[arm] = ecms_comparison(f"{ds[0]}/{cyc}/sac_ems_best", cyc, kfb, P)
    json.dump(ecms_res, open(out / f"ecms_comparison_{cyc}.json", "w"), indent=2)

    fh.close()
    print(f"\n[saved] {out}/phase6_forensics_{cyc}_FULL.txt and companion .json files")


if __name__ == "__main__":
    main()
