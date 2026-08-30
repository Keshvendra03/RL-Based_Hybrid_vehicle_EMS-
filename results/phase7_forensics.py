"""
phase7_forensics.py
===================
PHASE 7 -- ECONOMIC-VALUE / COSTATE FORENSIC EXPERIMENT.

Pure forensic calibration on the EXISTING validated checkpoints. NO training,
NO physics change, NO SAC-algorithm change, NO exploration change. Every number
here is reconstructed from checkpoints already on disk.

Covers brief sections:
  §1  baseline lock                          -> data/00_baseline_lock.json
  §2  effective battery price (state-cond.)  -> data/effective_price_<C>.json
  §3  matched-state central hypothesis       -> data/matched_states_<C>.csv (+_summary.json)
  §4  counterfactual dense action grid       -> data/counterfactual_<C>.json + figures/
  §5  economic-vs-temporal critic decomp.    -> inside _summary.json ["s5_*"]
  §6  k_fb 1.656 vs 2.5 vs 3.0 matched table -> data/kfb_compare_<C>.json
  §8  required-k_fb derivation               -> data/required_kfb_<C>.json
  §11 SAC vs ECMS gap decomposition          -> data/ecms_gap_<C>.json

    python -m results.phase7_forensics --cycle NEDC
    python -m results.phase7_forensics --cycle FTP75
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import SAC

from src.env.ems_env import (EMSEnv, U_MIN, U_MAX, ZB_MODEAWARE, SOC_TARGET,
                             _EPS_T, map_action_to_u,
                             K_FUEL_L_PER_KG, K_ELEC_L_PER_J)
from src.env.powertrain import (_T_CUTOFF, _interp1d_linear, _w_EM_max_row,
                                _T_EM_max_arr, _THETA_EM, _Q_BT_0)
from src.baselines.ecms import _hamiltonian_best_u
from src.baselines.advanced_rule_based import AdvancedController, control_unit_advanced

# ---------------------------------------------------------------------------
# Locked constants (validated in Phases 2/5; see RL_DIAGNOSTIC_REPORT.md §2)
# ---------------------------------------------------------------------------
ECMS_UNIT = 4.8309                       # eq_factor(liter-units) * this = ECMS lambda units
EQF = {"NEDC": 0.2717, "FTP75": 0.4981}  # = lambda0 / ECMS_UNIT
LAM0 = {"NEDC": 1.3125, "FTP75": 2.4062} # ECMS proven charge-sustaining costate
LAM_REF = 1.3125                         # the brief's fixed reference point (both cycles)
KFB_ECMS_EQUIV = 8.0 / ECMS_UNIT         # = 1.656  (ECMS's own k_fb, in env units)
BENCH = {"NEDC": 3.5056, "FTP75": 3.2323}
ECMS_V = {"NEDC": 3.1887, "FTP75": 2.8097}
AMAP = "modeaware_gated"
KFB_CONTROL = 2.5

# CONTROL + reference checkpoints (§1). 3 seeds each where available.
CKPTS = {
    "NEDC": {
        "control_k2.5_gated":  ["models_p5s0_k2.5", "models_p5_k2.5", "models_p5_k2.5_s2"],
        "ref_k1.656_gated":    ["models_p4_gated_g20", "models_p4g_N0", "models_p4g_N2"],
        "ref_k3.0_gated":      ["models_p5s0_k3.0", "models_p5_k3.0", "models_p5s2_k3.0"],
        "ref_k1.656_linear":   ["models_seed_NEDC_s0", "models_seed_NEDC_s2", "models_final_NEDC_s2"],
    },
    "FTP75": {
        "control_k2.5_gated":  ["models_p5f_k2.5_s0", "models_p5f_k2.5_s1", "models_p5f_k2.5_s2"],
        "ref_k1.656_gated":    ["models_p4g_F0", "models_p4g_F1", "models_p4g_F2"],
    },
}
KFB_OF = {"control_k2.5_gated": 2.5, "ref_k1.656_gated": 1.656, "ref_k3.0_gated": 3.0,
          "ref_k1.656_linear": 1.656}
AMAP_OF = {"control_k2.5_gated": AMAP, "ref_k1.656_gated": AMAP, "ref_k3.0_gated": AMAP,
           "ref_k1.656_linear": "linear"}

TB = [(0, 15, "0-15"), (15, 30, "15-30"), (30, 35, "30-35"),
      (35, 50, "35-50"), (50, 75, "50-75"), (75, 1e9, ">75")]
SB = [(0.0, 0.40, "<40"), (0.40, 0.45, "40-45"), (0.45, 0.50, "45-50"),
      (0.50, 0.55, "50-55"), (0.55, 1.0, ">55")]
REGIONS_FOCUS = [(15, 30, "15-30"), (30, 35, "30-35"), (35, 50, "35-50"), (50, 75, "50-75")]


# ======================================================================== #
# small helpers                                                            #
# ======================================================================== #
def eq_eff(kfb, soc, cycle):
    """effective per-step equivalence factor in reward (liter) units."""
    return EQF[cycle] + kfb * (SOC_TARGET - soc)


def eq_eff_ecms_units(kfb, soc, cycle):
    return eq_eff(kfb, soc, cycle) * ECMS_UNIT


def motor_cap(w, dw):
    return max(_interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w) - abs(_THETA_EM * dw) - _EPS_T, 0.0)


def torques_from_u(u, T, w, dw, soc):
    """Replicate EMSEnv._action_to_torques feasibility clamps (traction, T>cutoff)."""
    cap = motor_cap(w, dw)
    t_em = float(np.clip(u * T, -cap, cap))
    if soc <= 0.05:
        t_em = min(t_em, 0.0)
    if soc >= 0.95:
        t_em = max(t_em, 0.0)
    t_ce = T - t_em
    return t_ce, t_em


def mode_of_u(u, T, w, dw, soc):
    if T <= 0:
        return "REGEN" if T < 0 else "STOP"
    t_ce, t_em = torques_from_u(u, T, w, dw, soc)
    if t_ce <= _T_CUTOFF:
        return "OFF"
    return "LPS" if t_em < 0 else ("ASSIST" if t_em > 0 else "ONLY")


def mode_of_a(a, T, w, dw, soc, amap=AMAP):
    u = map_action_to_u(float(a), T, amap, w, dw)
    return mode_of_u(u, T, w, dw, soc)


def classify_final(mode, t_ce):
    if mode in ("stop", "STOP"):
        return "stop"
    if mode in ("regen", "REGEN"):
        return "REGEN"
    if t_ce <= _T_CUTOFF:
        return "OFF"
    return {"assist": "ASSIST", "lps_gen": "LPS"}.get(mode, mode.upper() if mode in ("assist", "lps_gen") else "ONLY")


def a_for_u(u_target, T, w, dw, amap=AMAP, grid=None):
    """Invert the action map numerically: action whose mapped u is closest to u_target."""
    if grid is None:
        grid = np.linspace(-1, 1, 401)
    us = np.array([map_action_to_u(float(a), T, amap, w, dw) for a in grid])
    return float(grid[np.argmin(np.abs(us - u_target))])


def q_at(model, ob, acts):
    ot = th.as_tensor(np.repeat(ob.reshape(1, -1), len(acts), 0)).float().to(model.device)
    at = th.as_tensor(np.asarray(acts, dtype=np.float32).reshape(-1, 1)).float().to(model.device)
    with th.no_grad():
        q = model.critic(ot, at)
    return np.minimum(q[0].cpu().numpy().ravel(), q[1].cpu().numpy().ravel())


def actor_at(model, ob):
    ot = th.as_tensor(ob.reshape(1, -1)).float().to(model.device)
    with th.no_grad():
        mu, ls, _ = model.actor.get_action_dist_params(ot)
    return float(mu.cpu().numpy().ravel()[0]), float(np.exp(ls.cpu().numpy().ravel()[0]))


def pct(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return dict(n=0)
    return dict(n=int(x.size), min=float(x.min()), p5=float(np.percentile(x, 5)),
               p25=float(np.percentile(x, 25)), median=float(np.median(x)),
               p75=float(np.percentile(x, 75)), p95=float(np.percentile(x, 95)),
               max=float(x.max()), mean=float(x.mean()),
               std=float(x.std(ddof=1)) if x.size > 1 else 0.0)


# ======================================================================== #
# rollouts                                                                 #
# ======================================================================== #
def sac_rollout(ckpt, cycle, kfb, amap=AMAP):
    m = SAC.load(ckpt)
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=kfb, lookahead=5, action_map=amap)
    obs, _ = env.reset()
    R = []
    while True:
        d = dict(env._demand)
        soc_before = env._Q_BT / _Q_BT_0
        a, _ = m.predict(obs, deterministic=True)
        obs, r, t, _, i = env.step(a)
        R.append(dict(T=d["T_MGB"], w=d["w_MGB"], dw=d["dw_MGB"], v=d["v"], gear=d["gear"],
                      soc_before=soc_before, soc=i["soc"], a=float(np.asarray(a).ravel()[0]),
                      u=i["u"], t_ce=i["T_CE_cmd"], t_em=i["T_EM_cmd"], p_em=i["p_em"],
                      cls=classify_final(i["mode"], i["T_CE_cmd"]),
                      fuel=i["fuel_liters_step"], elec=i["elec_liters_step"], r=float(r)))
        if t:
            R[-1]["final"] = i["episode_final"]
            return R


def ecms_rollout(cycle):
    lam0 = LAM0[cycle]
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=KFB_CONTROL, lookahead=0)

    def patched(self, action):
        d = self._demand
        w, dw, T = d["w_MGB"], d["dw_MGB"], d["T_MGB"]
        soc = self._Q_BT / _Q_BT_0
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
        soc_before = env._Q_BT / _Q_BT_0
        obs, r, t, _, i = env.step(np.zeros(1, np.float32))
        R.append(dict(T=d["T_MGB"], w=d["w_MGB"], dw=d["dw_MGB"], v=d["v"], gear=d["gear"],
                      soc_before=soc_before, soc=i["soc"], u=i["u"], t_ce=i["T_CE_cmd"],
                      t_em=i["T_EM_cmd"], p_em=i["p_em"],
                      cls=classify_final(i["mode"], i["T_CE_cmd"]),
                      fuel=i["fuel_liters_step"], elec=i["elec_liters_step"], r=float(r)))
        if t:
            R[-1]["final"] = i["episode_final"]
            return R


def rule_rollout(cycle):
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=KFB_CONTROL, lookahead=0)
    ctrl = AdvancedController(cycle_name=cycle)
    ctrl.reset()

    def patched(self, action):
        d = self._demand
        c = ctrl.step(d["w_MGB"], d["dw_MGB"], d["T_MGB"], d["gear"], self._Q_BT, d["v"])
        cu = control_unit_advanced(d["w_MGB"], d["dw_MGB"], d["T_MGB"], c["u"], c["state_CE"])
        m = ("regen" if d["T_MGB"] < 0 else
             ("stop" if d["T_MGB"] == 0 or d["w_MGB"] <= 0 else
              ("lps_gen" if cu["T_EM"] < 0 else ("assist" if cu["T_EM"] > 0 else "engine"))))
        return cu["T_CE"], cu["T_EM"], c["u"], m
    env._action_to_torques = types.MethodType(patched, env)
    obs, _ = env.reset()
    R = []
    while True:
        d = dict(env._demand)
        obs, r, t, _, i = env.step(np.zeros(1, np.float32))
        R.append(dict(T=d["T_MGB"], w=d["w_MGB"], dw=d["dw_MGB"], soc=i["soc"], u=i["u"],
                      t_ce=i["T_CE_cmd"], t_em=i["T_EM_cmd"],
                      cls=classify_final(i["mode"], i["T_CE_cmd"]),
                      fuel=i["fuel_liters_step"], elec=i["elec_liters_step"]))
        if t:
            R[-1]["final"] = i["episode_final"]
            return R


# ======================================================================== #
# §2  effective battery price, state-conditioned                            #
# ======================================================================== #
def section2_price(cycle, P, out):
    P(f"\n{'='*90}\n§2  EFFECTIVE BATTERY PRICE (state-conditioned) -- {cycle}\n{'='*90}")
    kfb = KFB_CONTROL
    R = sac_rollout(f"{CKPTS[cycle]['control_k2.5_gated'][0]}/{cycle}/sac_ems_best", cycle, kfb)
    E = ecms_rollout(cycle)
    mov = [x for x in R if x["T"] > _T_CUTOFF and x["w"] > 0]
    price = np.array([eq_eff_ecms_units(kfb, x["soc_before"], cycle) for x in mov])
    price_raw = np.array([eq_eff(kfb, x["soc_before"], cycle) for x in mov])
    socs = np.array([x["soc_before"] for x in mov])
    # ECMS effective lambda over ITS OWN visited SoC (reference law: lam0 + 8*(0.5-soc))
    emov = [x for x in E if x["T"] > _T_CUTOFF and x["w"] > 0]
    eprice = np.array([LAM0[cycle] + 8.0 * (SOC_TARGET - x["soc_before"]) for x in emov])

    res = {"cycle": cycle, "k_fb": kfb, "eq_factor_base": EQF[cycle],
           "lambda0_cycle": LAM0[cycle], "lambda_ref": LAM_REF, "ECMS_UNIT": ECMS_UNIT,
           "conversion": "lambda_ECMS = eq_factor_eff * 4.8309 ; eq_factor_eff = eq_factor + k_fb*(0.5-soc_before)"}
    res["sac_price_ecms_units"] = pct(price)
    res["sac_price_reward_units"] = pct(price_raw)
    res["sac_soc_before_visited"] = pct(socs)
    res["ecms_effective_lambda_own_rollout"] = pct(eprice)
    res["frac_episode_priced_above_lambda_ref"] = float(np.mean(price > LAM_REF))
    res["frac_episode_priced_above_lambda0_cycle"] = float(np.mean(price > LAM0[cycle]))
    res["frac_episode_priced_above_ecms_effective_median"] = float(
        np.mean(price > np.median(eprice)))

    P(f"  SAC effective price (ECMS units)  vs lambda_ref={LAM_REF}  (cycle lambda0={LAM0[cycle]})")
    s = res["sac_price_ecms_units"]
    P(f"    min={s['min']:.3f} p5={s['p5']:.3f} p25={s['p25']:.3f} median={s['median']:.3f} "
      f"p75={s['p75']:.3f} p95={s['p95']:.3f} max={s['max']:.3f} mean={s['mean']:.3f} sd={s['std']:.3f}")
    P(f"  fraction of moving-episode priced ABOVE lambda_ref (1.3125) : {res['frac_episode_priced_above_lambda_ref']*100:.1f}%")
    P(f"  fraction priced ABOVE cycle lambda0 ({LAM0[cycle]})          : {res['frac_episode_priced_above_lambda0_cycle']*100:.1f}%")
    e = res["ecms_effective_lambda_own_rollout"]
    P(f"  ECMS effective lambda over its own rollout: median={e['median']:.3f} "
      f"p25={e['p25']:.3f} p75={e['p75']:.3f} mean={e['mean']:.3f}")

    # ---- by SoC band ----
    P(f"\n  {'SoC band':>10}{'n':>7}{'price median':>14}{'price mean':>12}{'vs 1.3125':>11}{'vs lam0':>9}")
    res["by_soc_band"] = []
    for lo, hi, nm in SB:
        m = (socs >= lo) & (socs < hi)
        if m.sum() < 5:
            res["by_soc_band"].append(dict(band=nm, n=int(m.sum())))
            P(f"  {nm:>10}{int(m.sum()):>7}   (too few)")
            continue
        pm, pmn = float(np.median(price[m])), float(price[m].mean())
        row = dict(band=nm, n=int(m.sum()), price_median=pm, price_mean=pmn,
                   ratio_vs_lambda_ref=pm / LAM_REF, ratio_vs_lambda0=pm / LAM0[cycle])
        res["by_soc_band"].append(row)
        P(f"  {nm:>10}{int(m.sum()):>7}{pm:>14.3f}{pmn:>12.3f}{pm/LAM_REF:>10.2f}x{pm/LAM0[cycle]:>8.2f}x")

    # ---- by torque band (price experienced while deciding in that band) ----
    Tv = np.array([x["T"] for x in mov])
    P(f"\n  {'T band':>10}{'n':>7}{'price median':>14}{'price mean':>12}{'SoC median':>12}{'vs 1.3125':>11}")
    res["by_torque_band"] = []
    for lo, hi, nm in TB:
        m = (Tv >= lo) & (Tv < hi)
        if m.sum() < 5:
            res["by_torque_band"].append(dict(band=nm, n=int(m.sum())))
            P(f"  {nm:>10}{int(m.sum()):>7}   (too few)")
            continue
        pm = float(np.median(price[m]))
        row = dict(band=nm, n=int(m.sum()), price_median=pm, price_mean=float(price[m].mean()),
                   soc_median=float(np.median(socs[m])), ratio_vs_lambda_ref=pm / LAM_REF)
        res["by_torque_band"].append(row)
        P(f"  {nm:>10}{int(m.sum()):>7}{pm:>14.3f}{float(price[m].mean()):>12.3f}"
          f"{float(np.median(socs[m])):>12.3f}{pm/LAM_REF:>10.2f}x")

    json.dump(res, open(out / f"data/effective_price_{cycle}.json", "w"), indent=2)
    return res


# ======================================================================== #
# matched states (env deep-copy at every traction step)                     #
# ======================================================================== #
def matched_states(cycle, kfb=KFB_CONTROL, amap=AMAP, trajectory="ecms"):
    """Snapshot (obs, deep-copied clean env) at every traction step along a
    reference trajectory. trajectory='ecms' follows the charge-sustaining ECMS
    SoC path (realistic operating SoC); 'zero' follows the constant a=0 policy
    (legacy; drains to SOC_MIN and is NOT representative)."""
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=kfb, lookahead=5, action_map=amap)
    obs, _ = env.reset()
    lam0 = LAM0[cycle]
    S = []
    while True:
        d = env._demand
        T, w, dw = d["T_MGB"], d["dw_MGB"] * 0 + d["dw_MGB"], d["dw_MGB"]
        w = d["w_MGB"]
        soc = env._Q_BT / _Q_BT_0
        if T > _T_CUTOFF and w > 0:
            S.append(dict(obs=obs.copy(), env=copy.deepcopy(env), T=T, w=w,
                          dw=dw, soc=soc, gear=d["gear"], v=d["v"],
                          v_next=d.get("v_next", d["v"])))
        if trajectory == "ecms" and T > 0 and w > 0:
            u_e = _hamiltonian_best_u(w, dw, T, soc, lam0 + 8.0 * (SOC_TARGET - soc), 81)
            a_step = a_for_u(u_e, T, w, dw, amap)
        else:
            a_step = 0.0
        obs, r, t, _, i = env.step(np.array([a_step], np.float32))
        if t:
            return S


def reward_of_action(saved_env, a):
    e = copy.deepcopy(saved_env)
    _, r, _, _, _ = e.step(np.array([a], np.float32))
    return float(r)


def grid_probe(model, st, cycle, kfb, grid, amap=AMAP):
    """Full per-state counterfactual over a dense action grid."""
    ob, T, w, dw, soc = st["obs"], st["T"], st["w"], st["dw"], st["soc"]
    Q = q_at(model, ob, grid)
    rr, us, modes, tce, tem, pbatt, nsoc = [], [], [], [], [], [], []
    for a in grid:
        u = map_action_to_u(float(a), T, amap, w, dw)
        us.append(u)
        t_ce, t_em = torques_from_u(u, T, w, dw, soc)
        tce.append(t_ce); tem.append(t_em)
        modes.append(mode_of_u(u, T, w, dw, soc))
        e = copy.deepcopy(st["env"])
        _, r, _, _, ii = e.step(np.array([a], np.float32))
        rr.append(float(r)); pbatt.append(ii["p_em"]); nsoc.append(ii["soc"])
    rr = np.array(rr); Q = np.array(Q); us = np.array(us)
    # reference actions
    mu, sd = actor_at(model, ob)
    a_sac = float(np.tanh(mu))
    a_q = float(grid[int(Q.argmax())])
    a_r = float(grid[int(rr.argmax())])
    u_ecms = _hamiltonian_best_u(w, dw, T, soc, LAM0[cycle] + 8.0 * (SOC_TARGET - soc), 81)
    a_ecms = a_for_u(u_ecms, T, w, dw, amap)
    return dict(grid=grid.tolist(), Q=Q.tolist(), r=rr.tolist(), u=us.tolist(),
                mode=modes, t_ce=tce, t_em=tem, p_batt=pbatt, next_soc=nsoc,
                mu=mu, sigma=sd, a_sac=a_sac, a_argmaxQ=a_q, a_argmaxR=a_r,
                u_ecms=float(u_ecms), a_ecms=a_ecms,
                eq_eff_ecms_units=float(eq_eff_ecms_units(kfb, soc, cycle)))


# ======================================================================== #
# §3 + §5  matched-state central hypothesis + economic/temporal decomp      #
# ======================================================================== #
def section35(cycle, P, out):
    P(f"\n{'='*90}\n§3 + §5  MATCHED-STATE CENTRAL HYPOTHESIS + ECONOMIC/TEMPORAL DECOMPOSITION -- {cycle}\n{'='*90}")
    kfb = KFB_CONTROL
    ck = f"{CKPTS[cycle]['control_k2.5_gated'][0]}/{cycle}/sac_ems_best"
    model = SAC.load(ck)
    S = matched_states(cycle, kfb)
    grid = np.linspace(-1, 1, 81)
    rows = []
    per_region = {}
    for lo, hi, nm in REGIONS_FOCUS:
        sel = [s for s in S if lo <= s["T"] < hi]
        if not sel:
            continue
        sel = sel[:: max(1, len(sel) // 90)][:90]
        rec = []
        for st in sel:
            g = grid_probe(model, st, cycle, kfb, grid)
            # candidate probe actions
            a_off = 2.0 * ZB_MODEAWARE - 1.0
            # representative in-mode probes
            def nearest(mode_name, default):
                idx = [k for k, mm in enumerate(g["mode"]) if mm == mode_name]
                return float(grid[idx[len(idx) // 2]]) if idx else default
            pa_off = nearest("OFF", min(1.0, a_off + 0.05))
            pa_ass = nearest("ASSIST", 0.30)
            pa_lps = nearest("LPS", -0.50)
            r_off = reward_of_action(st["env"], pa_off)
            r_ass = reward_of_action(st["env"], pa_ass)
            r_lps = reward_of_action(st["env"], pa_lps)
            q_off, q_ass, q_lps = q_at(model, st["obs"], [pa_off, pa_ass, pa_lps])
            # ECMS / SAC action-level errors
            q_ecms, q_sac = q_at(model, st["obs"], [g["a_ecms"], g["a_sac"]])
            r_ecms = reward_of_action(st["env"], g["a_ecms"])
            r_sac = reward_of_action(st["env"], g["a_sac"])
            # SoC / discharge consequences
            i_ecms = g["grid"].index(min(g["grid"], key=lambda x: abs(x - g["a_ecms"])))
            i_sac = g["grid"].index(min(g["grid"], key=lambda x: abs(x - g["a_sac"])))
            row = dict(
                region=nm, SoC=st["soc"], T_MGB=st["T"], w_MGB=st["w"], dw_MGB=st["dw"],
                v=st["v"], v_next=st["v_next"], gear=int(st["gear"]),
                actor_mu=g["mu"], actor_sigma=g["sigma"], a_sac=g["a_sac"],
                a_argmaxQ=g["a_argmaxQ"], a_argmaxR=g["a_argmaxR"],
                a_ecms=g["a_ecms"], u_ecms=g["u_ecms"],
                mode_sac=mode_of_a(g["a_sac"], st["T"], st["w"], st["dw"], st["soc"]),
                mode_argmaxQ=mode_of_a(g["a_argmaxQ"], st["T"], st["w"], st["dw"], st["soc"]),
                mode_ecms=mode_of_a(g["a_ecms"], st["T"], st["w"], st["dw"], st["soc"]),
                r_OFF=r_off, r_ASSIST=r_ass, r_LPS=r_lps,
                Q_OFF=float(q_off), Q_ASSIST=float(q_ass), Q_LPS=float(q_lps),
                dr_OFF_ASSIST=r_off - r_ass, dQ_OFF_ASSIST=float(q_off - q_ass),
                dr_OFF_LPS=r_off - r_lps, dQ_OFF_LPS=float(q_off - q_lps),
                eq_eff_ecms_units=g["eq_eff_ecms_units"],
                t_ce_sac=g["t_ce"][i_sac], t_em_sac=g["t_em"][i_sac],
                p_batt_sac=g["p_batt"][i_sac], next_soc_sac=g["next_soc"][i_sac],
                t_ce_ecms=g["t_ce"][i_ecms], t_em_ecms=g["t_em"][i_ecms],
                p_batt_ecms=g["p_batt"][i_ecms], next_soc_ecms=g["next_soc"][i_ecms],
                ERROR_critic=float(q_ecms - q_sac), ERROR_reward=float(r_ecms - r_sac),
                dSoC_ecms_minus_sac=g["next_soc"][i_ecms] - g["next_soc"][i_sac],
                discharge_ecms_minus_sac=g["p_batt"][i_ecms] - g["p_batt"][i_sac],
            )
            rec.append(row)
            rows.append(row)
        A = lambda k: np.array([r[k] for r in rec])
        drA, dqA = A("dr_OFF_ASSIST"), A("dQ_OFF_ASSIST")
        ec, er = A("ERROR_critic"), A("ERROR_reward")
        per_region[nm] = dict(
            n=len(rec),
            dr_OFF_ASSIST=pct(drA), dQ_OFF_ASSIST=pct(dqA),
            dr_OFF_ASSIST_pos_pct=float(100 * np.mean(drA > 0)),
            dQ_OFF_ASSIST_pos_pct=float(100 * np.mean(dqA > 0)),
            dQ_OFF_LPS=pct(A("dQ_OFF_LPS")), dr_OFF_LPS=pct(A("dr_OFF_LPS")),
            ERROR_critic=pct(ec), ERROR_reward=pct(er),
            # §5 decomposition: does the critic reject ECMS because the REWARD does,
            # or in spite of the reward?
            s5_reward_rejects_ecms_pct=float(100 * np.mean(er < 0)),
            s5_critic_rejects_ecms_pct=float(100 * np.mean(ec < 0)),
            s5_critic_rejects_while_reward_accepts_pct=float(100 * np.mean((ec < 0) & (er >= 0))),
            s5_both_reject_pct=float(100 * np.mean((ec < 0) & (er < 0))),
            s5_corr_ERRcritic_ERRreward=float(np.corrcoef(ec, er)[0, 1]) if len(ec) > 2 else 0.0,
            s5_corr_ERRcritic_dSoC=float(np.corrcoef(ec, A("dSoC_ecms_minus_sac"))[0, 1]) if len(ec) > 2 else 0.0,
            s5_corr_ERRcritic_eqprice=float(np.corrcoef(ec, A("eq_eff_ecms_units"))[0, 1]) if len(ec) > 2 else 0.0,
            mode_sac_OFF_pct=float(100 * np.mean([r["mode_sac"] == "OFF" for r in rec])),
            mode_ecms_OFF_pct=float(100 * np.mean([r["mode_ecms"] == "OFF" for r in rec])),
            mode_argmaxQ_OFF_pct=float(100 * np.mean([r["mode_argmaxQ"] == "OFF" for r in rec])),
        )
        pr = per_region[nm]
        P(f"\n  --- {nm} Nm  (n={pr['n']}) ---")
        P(f"    dr(OFF-ASSIST): mean={pr['dr_OFF_ASSIST']['mean']:+.4f} median={pr['dr_OFF_ASSIST']['median']:+.4f} "
          f">0:{pr['dr_OFF_ASSIST_pos_pct']:.0f}%")
        P(f"    dQ(OFF-ASSIST): mean={pr['dQ_OFF_ASSIST']['mean']:+.4f} median={pr['dQ_OFF_ASSIST']['median']:+.4f} "
          f">0:{pr['dQ_OFF_ASSIST_pos_pct']:.0f}%")
        P(f"    dQ(OFF-LPS):    mean={pr['dQ_OFF_LPS']['mean']:+.4f}  dr(OFF-LPS): mean={pr['dr_OFF_LPS']['mean']:+.4f}")
        P(f"    ERROR_reward = r(a_ECMS)-r(a_SAC): mean={pr['ERROR_reward']['mean']:+.4f} median={pr['ERROR_reward']['median']:+.4f}")
        P(f"    ERROR_critic = Q(a_ECMS)-Q(a_SAC): mean={pr['ERROR_critic']['mean']:+.4f} median={pr['ERROR_critic']['median']:+.4f}")
        P(f"    §5: reward rejects ECMS in {pr['s5_reward_rejects_ecms_pct']:.0f}%  |  "
          f"critic rejects ECMS in {pr['s5_critic_rejects_ecms_pct']:.0f}%  |  "
          f"critic rejects WHILE reward accepts in {pr['s5_critic_rejects_while_reward_accepts_pct']:.0f}%")
        P(f"    §5: corr(ERR_critic,ERR_reward)={pr['s5_corr_ERRcritic_ERRreward']:+.2f}  "
          f"corr(ERR_critic,dSoC)={pr['s5_corr_ERRcritic_dSoC']:+.2f}  "
          f"corr(ERR_critic,eq_price)={pr['s5_corr_ERRcritic_eqprice']:+.2f}")
        P(f"    OFF share:  SAC={pr['mode_sac_OFF_pct']:.0f}%  argmaxQ={pr['mode_argmaxQ_OFF_pct']:.0f}%  ECMS={pr['mode_ecms_OFF_pct']:.0f}%")

    # write CSV
    import csv
    if rows:
        keys = list(rows[0].keys())
        with open(out / f"data/matched_states_{cycle}.csv", "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys)
            wtr.writeheader()
            for r in rows:
                wtr.writerow(r)
    json.dump({"cycle": cycle, "k_fb": kfb, "per_region": per_region},
              open(out / f"data/matched_states_{cycle}_summary.json", "w"), indent=2)
    return per_region


# ======================================================================== #
# §4  counterfactual dense-grid figures                                     #
# ======================================================================== #
def section4_figs(cycle, P, out):
    P(f"\n{'='*90}\n§4  COUNTERFACTUAL DENSE ACTION GRID (figures) -- {cycle}\n{'='*90}")
    kfb = KFB_CONTROL
    model = SAC.load(f"{CKPTS[cycle]['control_k2.5_gated'][0]}/{cycle}/sac_ems_best")
    S = matched_states(cycle, kfb)
    grid = np.linspace(-1, 1, 121)
    dump = {"cycle": cycle, "regions": {}}
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for col, (lo, hi, nm) in enumerate(REGIONS_FOCUS):
        sel = [s for s in S if lo <= s["T"] < hi]
        if not sel:
            continue
        st = sel[len(sel) // 2]     # median-index representative state
        g = grid_probe(model, st, cycle, kfb, grid)
        Q = np.array(g["Q"]); rr = np.array(g["r"])
        dump["regions"][nm] = dict(state=dict(SoC=st["soc"], T=st["T"], w=st["w"], v=st["v"]),
                                   grid=g["grid"], Q=g["Q"], r=g["r"], mode=g["mode"],
                                   a_sac=g["a_sac"], a_argmaxQ=g["a_argmaxQ"],
                                   a_argmaxR=g["a_argmaxR"], a_ecms=g["a_ecms"])
        for row, (Y, lab) in enumerate([(rr, "immediate reward r(a)"), (Q, "SAC min-Q(a)")]):
            ax = axes[row, col]
            ax.plot(g["grid"], Y, "-", lw=1.6, color="#1f77b4")
            # shade modes
            modes = np.array(g["mode"])
            for mm, cc in [("OFF", "#d6f5d6"), ("ASSIST", "#ffe9cc"), ("LPS", "#e6e6fa")]:
                idx = np.where(modes == mm)[0]
                if idx.size:
                    ax.axvspan(g["grid"][idx.min()], g["grid"][idx.max()], color=cc, alpha=.6, zorder=0)
            for a_, c_, l_ in [(g["a_sac"], "k", "SAC"), (g["a_argmaxQ"], "r", "argmaxQ"),
                               (g["a_argmaxR"], "g", "argmaxR"), (g["a_ecms"], "m", "ECMS")]:
                ax.axvline(a_, color=c_, ls="--", lw=1.3, label=l_)
            ax.set_title(f"{nm} Nm | SoC {st['soc']*100:.1f}% | {lab}", fontsize=9)
            ax.set_xlabel("action a")
            if col == 0:
                ax.set_ylabel(lab)
            if row == 0 and col == 3:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"§4 Counterfactual Q(a) / r(a) -- {cycle}  (CONTROL k_fb=2.5, gated)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / f"figures/counterfactual_{cycle}.png", dpi=110)
    plt.close(fig)
    json.dump(dump, open(out / f"data/counterfactual_{cycle}.json", "w"), indent=2)
    P(f"  [saved] figures/counterfactual_{cycle}.png")
    return dump


# ======================================================================== #
# §6  k_fb 1.656 vs 2.5 vs 3.0 matched-state table                          #
# ======================================================================== #
def section6_kfb(cycle, P, out):
    P(f"\n{'='*90}\n§6  k_fb 1.656 vs 2.5 vs 3.0 -- matched-state economic table -- {cycle}\n{'='*90}")
    variants = [k for k in ("ref_k1.656_gated", "control_k2.5_gated", "ref_k3.0_gated")
                if k in CKPTS[cycle]]
    S = matched_states(cycle, KFB_CONTROL)     # same states for all variants
    grid = np.linspace(-1, 1, 81)
    res = {"cycle": cycle, "variants": {}}
    P(f"  {'variant':>20}{'k_fb':>6}{'region':>9}{'SoC band':>10}{'eq med(ECMSu)':>15}"
      f"{'dr(OFF-ASST)':>14}{'dQ(OFF-ASST)':>14}{'P(OFF)%':>9}{'fuel L/step':>13}")
    for v in variants:
        kfb = KFB_OF[v]
        ck = f"{CKPTS[cycle][v][0]}/{cycle}/sac_ems_best"
        model = SAC.load(ck)
        rows = []
        _sbands = ([(0.32, 0.42, "32-42"), (0.42, 0.52, "42-52")] if cycle == "NEDC"
                   else [(0.40, 0.48, "40-48"), (0.46, 0.54, "46-54")])
        for lo, hi, rn in REGIONS_FOCUS:
            for slo, shi, sn in _sbands:
                sel = [s for s in S if lo <= s["T"] < hi and slo <= s["soc"] < shi]
                if len(sel) < 8:
                    continue
                sel = sel[:: max(1, len(sel) // 40)][:40]
                drl, dql, poffl, eql, fuell = [], [], [], [], []
                for st in sel:
                    a_off = 2.0 * ZB_MODEAWARE - 1.0
                    g_mode = [mode_of_a(a, st["T"], st["w"], st["dw"], st["soc"]) for a in grid]
                    def nn(mn, dfl):
                        ix = [k for k, mm in enumerate(g_mode) if mm == mn]
                        return float(grid[ix[len(ix) // 2]]) if ix else dfl
                    pa_off, pa_ass = nn("OFF", min(1.0, a_off + .05)), nn("ASSIST", .30)
                    r_off = reward_of_action(st["env"], pa_off)
                    r_ass = reward_of_action(st["env"], pa_ass)
                    q_off, q_ass = q_at(model, st["obs"], [pa_off, pa_ass])
                    mu, sd = actor_at(model, st["obs"])
                    z = (np.arctanh(np.clip(a_off, -.999999, .999999)) - mu) / sd
                    poffl.append(float(norm.sf(z)))
                    drl.append(r_off - r_ass); dql.append(float(q_off - q_ass))
                    eql.append(eq_eff_ecms_units(kfb, st["soc"], cycle))
                    # fuel of the SAC deterministic action at this state
                    fuell.append(reward_of_action(st["env"], float(np.tanh(mu))))
                row = dict(region=rn, soc_band=sn, n=len(sel),
                           eq_median_ecms_units=float(np.median(eql)),
                           dr_OFF_ASSIST_mean=float(np.mean(drl)),
                           dQ_OFF_ASSIST_mean=float(np.mean(dql)),
                           P_OFF_pct=float(100 * np.mean(poffl)),
                           reward_proxy_mean=float(np.mean(fuell)))
                rows.append(row)
                P(f"  {v:>20}{kfb:>6.3f}{rn:>9}{sn:>10}{row['eq_median_ecms_units']:>15.3f}"
                  f"{row['dr_OFF_ASSIST_mean']:>+14.4f}{row['dQ_OFF_ASSIST_mean']:>+14.4f}"
                  f"{row['P_OFF_pct']:>9.1f}{row['reward_proxy_mean']:>13.4f}")
        res["variants"][v] = dict(k_fb=kfb, checkpoint=ck, rows=rows)
    json.dump(res, open(out / f"data/kfb_compare_{cycle}.json", "w"), indent=2)
    return res


# ======================================================================== #
# §8  required-k_fb derivation from the measured SoC distribution           #
# ======================================================================== #
def section8_required_kfb(cycle, price_res, P, out):
    P(f"\n{'='*90}\n§8  REQUIRED-k_fb DERIVATION (from measured visited-SoC distribution) -- {cycle}\n{'='*90}")
    socs_stats = price_res["sac_soc_before_visited"]
    med_soc = socs_stats["median"]
    p25_soc, p75_soc = socs_stats["p25"], socs_stats["p75"]
    # eq_eff_ecms_units(median_soc) = 1.3125 + k_fb*ECMS_UNIT*(0.5 - median_soc)
    # target A: median effective price == lambda_ref (1.3125)  -> k_fb = 0 unless 0.5-med_soc != 0
    d = SOC_TARGET - med_soc
    kfb_median_eq_lambdaref = 0.0 if abs(d) < 1e-6 else 0.0  # by construction the base already = lambda_ref at soc=0.5
    # Because base eq_factor already equals lambda_ref/ECMS_UNIT, the ONLY way the
    # MEDIAN price equals lambda_ref is k_fb*(0.5-med_soc)=0.  If med_soc<0.5 that
    # needs k_fb<=0 (rejected: reopens SoC drift, Phase 4/5).  The meaningful
    # calibration target is therefore to REPRODUCE ECMS's OWN costate law slope.
    kfb_match_ecms_law = KFB_ECMS_EQUIV           # 1.656
    # target C: make the SAC median effective price equal ECMS's median effective
    # lambda over its own rollout (a like-for-like closed-loop match)
    ecms_med = price_res["ecms_effective_lambda_own_rollout"]["median"]
    kfb_match_ecms_median = (ecms_med - LAM0[cycle]) / (ECMS_UNIT * d) if abs(d) > 1e-6 else float("nan")
    res = dict(cycle=cycle, median_visited_soc=med_soc, p25_soc=p25_soc, p75_soc=p75_soc,
               current_kfb=KFB_CONTROL,
               kfb_to_match_ecms_costate_law=kfb_match_ecms_law,
               kfb_to_match_ecms_median_effective_lambda=kfb_match_ecms_median,
               note=("Base eq_factor already == lambda_ref/4.8309, so 'median price == 1.3125' "
                     "is only achievable at k_fb<=0, which reopens the SoC drift refuted in "
                     "Phase 4/5. The defensible calibration targets are the ECMS costate-law "
                     "slope (k_fb=1.656) and the ECMS median-effective-lambda match."))
    # scientifically-justified sweep bracket
    lo = min(kfb_match_ecms_law, kfb_match_ecms_median if np.isfinite(kfb_match_ecms_median) else kfb_match_ecms_law)
    hi = KFB_CONTROL
    mid = 0.5 * (lo + hi)
    res["proposed_sweep"] = sorted({round(float(x), 3) for x in (lo, mid, hi)})
    res["existing_checkpoints_on_this_axis"] = {
        "1.656": "models_p4g_* (gated, NEDC 1/3 CS ~3.88 ; FTP75 best 3.246)",
        "2.0": "models_p5_k2.0 (seed1 only)",
        "2.5": "models_p5*_k2.5 (3 seeds, NEDC 3.7666 3/3 CS) = CONTROL",
        "3.0": "models_p5*_k3.0 (3 seeds, NEDC 3.784 3/3 CS)",
        "4.0/5.0": "models_p5_k4.0 / k5.0",
    }
    P(f"  median visited SoC (pre-decision, moving) = {med_soc*100:.2f}%   (p25={p25_soc*100:.2f}%, p75={p75_soc*100:.2f}%)")
    P(f"  k_fb to reproduce ECMS costate-law slope           : {kfb_match_ecms_law:.3f}")
    P(f"  k_fb to match ECMS median effective lambda          : {kfb_match_ecms_median:.3f}")
    P(f"  proposed minimal sweep (one variable)              : {res['proposed_sweep']}")
    P(f"  NOTE: k_fb in [2.0,3.0] is ALREADY a measured flat plateau (3.766->3.784, both 3/3 CS);")
    P(f"        k_fb=1.656 is ALREADY measured (NEDC 1/3 CS, ~3.88). See existing checkpoints.")
    json.dump(res, open(out / f"data/required_kfb_{cycle}.json", "w"), indent=2)
    return res


# ======================================================================== #
# §11  SAC vs ECMS gap decomposition                                        #
# ======================================================================== #
def section11_gap(cycle, P, out):
    P(f"\n{'='*90}\n§11  SAC vs ECMS GAP DECOMPOSITION (matched demand) -- {cycle}\n{'='*90}")
    kfb = KFB_CONTROL
    sac = sac_rollout(f"{CKPTS[cycle]['control_k2.5_gated'][0]}/{cycle}/sac_ems_best", cycle, kfb)
    ecms = ecms_rollout(cycle)
    rule = rule_rollout(cycle)
    n = min(len(sac), len(ecms), len(rule))
    sac, ecms, rule = sac[:n], ecms[:n], rule[:n]
    dT = max(abs(sac[i]["T"] - ecms[i]["T"]) for i in range(n))
    P(f"  demand alignment: max|T_SAC - T_ECMS| = {dT:.2e}")
    K = 1e5 / ecms[-1]["final"]["x_tot_m"]
    regs = [("brake", lambda x: x["T"] < 0), ("0-15", lambda x: 0 <= x["T"] < 15),
            ("15-30", lambda x: 15 <= x["T"] < 30), ("30-35", lambda x: 30 <= x["T"] < 35),
            ("35-50", lambda x: 35 <= x["T"] < 50), ("50-75", lambda x: 50 <= x["T"] < 75),
            (">75", lambda x: x["T"] >= 75)]
    out_d = {"cycle": cycle,
             "summary": dict(sac_v_ce=sac[-1]["final"]["v_ce_equiv"],
                             ecms_v_ce=ecms[-1]["final"]["v_ce_equiv"],
                             rule_v_ce=rule[-1]["final"]["v_ce_equiv"],
                             sac_soc=sac[-1]["final"]["soc_final"],
                             ecms_soc=ecms[-1]["final"]["soc_final"]),
             "regions": {}}
    P(f"\n  {'region':>7}{'dFuel':>10}{'dElec':>10}{'dTotal':>10}{'SAC OFF%':>10}{'ECMS OFF%':>11}"
      f"{'SAC eng|Tce|':>13}{'ECMS eng|Tce|':>14}{'dEngPt*':>10}")
    tot_fuel = tot_elec = 0.0
    for nm, f in regs:
        ia = [i for i in range(n) if f(sac[i])]
        ie = [i for i in range(n) if f(ecms[i])]
        if not ia:
            continue
        df = (sum(sac[i]["fuel"] for i in ia) - sum(ecms[i]["fuel"] for i in ie)) * K
        de = (sum(sac[i]["elec"] for i in ia) - sum(ecms[i]["elec"] for i in ie)) * K
        so = 100 * np.mean([sac[i]["cls"] == "OFF" for i in ia])
        eo = 100 * np.mean([ecms[i]["cls"] == "OFF" for i in ie])
        s_eng = [abs(sac[i]["t_ce"]) for i in ia if sac[i]["cls"] not in ("OFF", "REGEN", "stop")]
        e_eng = [abs(ecms[i]["t_ce"]) for i in ie if ecms[i]["cls"] not in ("OFF", "REGEN", "stop")]
        s_engpt = float(np.mean(s_eng)) if s_eng else 0.0
        e_engpt = float(np.mean(e_eng)) if e_eng else 0.0
        # engine operating-point proxy: fuel per engine-on second when engine is on
        s_fpos = sum(sac[i]["fuel"] for i in ia if sac[i]["cls"] not in ("OFF", "REGEN")) / max(len(s_eng), 1)
        e_fpos = sum(ecms[i]["fuel"] for i in ie if ecms[i]["cls"] not in ("OFF", "REGEN")) / max(len(e_eng), 1)
        d_engpt = (s_fpos - e_fpos) * K * max(len(s_eng), 1)
        tot_fuel += df; tot_elec += de
        out_d["regions"][nm] = dict(dfuel=df, delec=de, dtotal=df + de, sac_off_pct=so,
                                    ecms_off_pct=eo, sac_eng_tce=s_engpt, ecms_eng_tce=e_engpt,
                                    engpt_gap_proxy=d_engpt,
                                    n_sac=len(ia), n_ecms=len(ie))
        P(f"  {nm:>7}{df:>+10.4f}{de:>+10.4f}{df+de:>+10.4f}{so:>9.1f}%{eo:>10.1f}%"
          f"{s_engpt:>13.1f}{e_engpt:>14.1f}{d_engpt:>+10.4f}")
    # 4-way split of the total gap
    gap_total = out_d["summary"]["sac_v_ce"] - out_d["summary"]["ecms_v_ce"]
    # mode-selection component: fuel delta explained where SAC OFF% < ECMS OFF% (SAC burns where ECMS doesn't)
    mode_sel = sum(max(v["dfuel"], 0.0) for k, v in out_d["regions"].items()
                   if v["sac_off_pct"] < v["ecms_off_pct"] - 3)
    engpt = sum(v["engpt_gap_proxy"] for k, v in out_d["regions"].items()
                if abs(v["sac_off_pct"] - v["ecms_off_pct"]) <= 15 and v["engpt_gap_proxy"] > 0)
    batt_mgmt = tot_elec
    soc_equiv = gap_total - (mode_sel + engpt + batt_mgmt)
    out_d["gap_split"] = dict(gap_total=gap_total, mode_selection=mode_sel,
                              engine_operating_point=engpt, battery_energy_mgmt=batt_mgmt,
                              soc_equivalence_residual=soc_equiv)
    P(f"\n  GAP (SAC-ECMS) total V_CE = {gap_total:+.4f} L/100km")
    P(f"    mode-selection component      ~ {mode_sel:+.4f}")
    P(f"    engine operating-point comp.  ~ {engpt:+.4f}")
    P(f"    battery-energy-mgmt component ~ {batt_mgmt:+.4f}")
    P(f"    SoC-equivalence residual      ~ {soc_equiv:+.4f}")
    json.dump(out_d, open(out / f"data/ecms_gap_{cycle}.json", "w"), indent=2)
    return out_d


# ======================================================================== #
# §1  baseline lock                                                         #
# ======================================================================== #
def section1_lock(out, P):
    P(f"\n{'='*90}\n§1  BASELINE LOCK\n{'='*90}")
    import subprocess
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    lock = {
        "phase": 7, "git_commit_before_experiment": commit,
        "date": "2026-08-27",
        "CONTROL": {
            "id": "gated k_fb=2.5 (Phase-5 validated candidate)",
            "checkpoints_NEDC": ["models_p5s0_k2.5/NEDC/sac_ems_best (seed0, V_CE 3.6862)",
                                 "models_p5_k2.5/NEDC/sac_ems_best (seed1, 3.8431)",
                                 "models_p5_k2.5_s2/NEDC/sac_ems_best (seed2, 3.7704)"],
            "checkpoints_FTP75": ["models_p5f_k2.5_s0/FTP75/sac_ems_best (3.2699)",
                                  "models_p5f_k2.5_s1/FTP75/sac_ems_best (3.3041)",
                                  "models_p5f_k2.5_s2/FTP75/sac_ems_best (3.2926)"],
            "seed": [0, 1, 2], "gamma": 0.20, "n_step": 1,
            "action_representation": "modeaware_gated",
            "eq_factor": {"NEDC": 0.2717, "FTP75": 0.4981},
            "k_fb": 2.5, "target_entropy": "auto (= -1.0, action_dim=1)",
            "learning_rate": 3e-4, "batch_size": 512, "buffer_size": 300000,
            "gradient_steps": 16, "train_freq": 64, "tau": 0.005,
            "lookahead": 5, "lambda_soc": 2.0, "soc_deadband": 0.10,
            "net_arch": [256, 256], "training_steps": 150000,
            "learning_starts": "2 * episode_len",
            "training_git_commit": "9a125adc7577ec5a1d66962ef32ebb91ce5d5497",
            "V_CE_NEDC_mean_3seed": 3.7666, "V_CE_FTP75_mean_3seed": 3.2889,
        },
        "COMPARISON_REFERENCES_not_training_targets": {
            "linear_k_fb_1.656": {"ckpts": ["models_seed_NEDC_s0", "models_seed_NEDC_s2",
                                            "models_final_NEDC_s2"], "V_CE_NEDC": 3.7727},
            "gated_k_fb_1.656": {"ckpts": ["models_p4_gated_g20", "models_p4g_N0", "models_p4g_N2"],
                                 "V_CE_NEDC": 3.8824, "CS": "1/3",
                                 "ckpts_FTP75": ["models_p4g_F0", "models_p4g_F1", "models_p4g_F2"],
                                 "V_CE_FTP75": 3.2460},
            "gated_k_fb_2.5": {"= CONTROL": True},
            "gated_k_fb_3.0": {"ckpts": ["models_p5s0_k3.0", "models_p5_k3.0", "models_p5s2_k3.0"],
                               "V_CE_NEDC": 3.7840, "CS": "3/3"},
            "ECMS": {"NEDC": 3.1887, "FTP75": 2.8097, "lambda0": {"NEDC": 1.3125, "FTP75": 2.4062},
                     "k_fb_ecms": 8.0},
            "advanced_rule_based": {"NEDC": 3.5056, "FTP75": 3.2323},
        },
        "conversion_validated_phase2_phase5": {
            "ECMS_UNIT": 4.8309,
            "formula": "lambda_ECMS = eq_factor_eff * 4.8309",
            "eq_factor_eff": "eq_factor + k_fb * (0.5 - soc_before)",
            "k_fb_ecms_equivalent_in_env_units": 1.656,
        },
        "LOCKED_do_not_modify": ["powertrain.py", "driving_cycle.py", "advanced_rule_based.py",
                                 "ecms.py", "env<->plant wiring", "feasibility masks",
                                 "drive-cycle data", "evaluate_policy.py evaluator"],
    }
    json.dump(lock, open(out / "data/00_baseline_lock.json", "w"), indent=2)
    P(f"  git commit before experiment : {commit}")
    P(f"  CONTROL: gated k_fb=2.5, gamma=0.20, n_step=1, eq_factor(NEDC)=0.2717, tgt_entropy=auto,")
    P(f"           lr=3e-4, batch=512, buffer=300k, grad_steps=16, lookahead=5, 150k steps, 3 seeds")
    P(f"  [saved] data/00_baseline_lock.json")
    return lock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, choices=["NEDC", "FTP75"])
    ap.add_argument("--out", default="results/phase7")
    ap.add_argument("--skip", default="", help="comma list of sections to skip e.g. 4,6")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    skip = set(s.strip() for s in a.skip.split(",") if s.strip())

    fh = open(out / f"raw/phase7_forensics_{a.cycle}.txt", "w", encoding="utf-8")
    P = lambda s: (print(s), fh.write(str(s) + "\n"))
    P(f"PHASE 7 FORENSIC CALIBRATION -- {a.cycle}")
    P(f"(pure analysis on existing checkpoints; no training, no physics/algorithm change)")

    if "1" not in skip:
        section1_lock(out, P)
    price_res = None
    if "2" not in skip:
        price_res = section2_price(a.cycle, P, out)
    else:
        price_res = json.load(open(out / f"data/effective_price_{a.cycle}.json"))
    if "3" not in skip:
        section35(a.cycle, P, out)
    if "4" not in skip:
        section4_figs(a.cycle, P, out)
    if "6" not in skip:
        section6_kfb(a.cycle, P, out)
    if "8" not in skip:
        section8_required_kfb(a.cycle, price_res, P, out)
    if "11" not in skip:
        section11_gap(a.cycle, P, out)

    fh.close()
    print(f"\n[done] {a.cycle}: raw -> {out}/raw/phase7_forensics_{a.cycle}.txt  + data/*.json")


if __name__ == "__main__":
    main()
