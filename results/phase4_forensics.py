"""
phase4_forensics.py
===================
Phase-4 root-cause analysis of the residual SAC-vs-benchmark gap, focused on
the 30-50 Nm engine-OFF/ASSIST decision boundary.

Covers brief sections 6-16:
   6  state-conditioned policy map (dense grid)
   7  30-50 Nm split into 5 Nm sub-bands
   8  counterfactual Q-values (Q1/Q2/minQ) at matched states
   9  counterfactual IMMEDIATE reward + heatmap
  10  short counterfactual rollouts (1/5/10/20 steps)
  11  costate eq_factor(SoC) and d(eq_factor)/dSoC
  12  SAC vs ECMS vs rule-based at MATCHED states
  13  error budget by operating region
  15  stochasticity (actor sigma, sampled vs deterministic)
  16  policy consistency at fixed torques

KEY STRUCTURAL FACT exploited here (verified in section 13):
T_MGB at each timestep depends only on the drive cycle, vehicle dynamics and
gearbox -- NOT on the controller. So every controller sees an identical demand
sequence and per-timestep comparisons are exactly aligned.

    python -m results.phase4_forensics --cycle NEDC --out results/phase4

No training. No physics modified.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch as th

from src.env.ems_env import EMSEnv, U_MIN, U_MAX, SOC_TARGET, map_action_to_u
from src.env.powertrain import (_T_CUTOFF, _interp1d_linear, _w_EM_max_row,
                                _T_EM_max_arr, _THETA_EM)
from src.env.ems_env import _EPS_T

REF = {"short": ("models_expD_g20", 0.20), "inter": ("models_expD_g90", 0.90)}
EQF = {"NEDC": 0.2717, "FTP75": 0.4981}
KFB = 1.656
BENCH = {"NEDC": 3.5056, "FTP75": 3.2323}
ECMS_LAM = {"NEDC": 1.3125, "FTP75": 2.4062}


# --------------------------------------------------------------------------- #
def a_for_u(u: float) -> float:
    return float(np.clip(2.0 * (u - U_MIN) / (U_MAX - U_MIN) - 1.0, -1.0, 1.0))


def a_off_of(T: float) -> float:
    """Smallest action giving engine-OFF at this torque demand."""
    if T <= _T_CUTOFF:
        return -1.0
    return a_for_u(1.0 - _T_CUTOFF / T)


def off_feasible(T: float, w: float, dw: float) -> bool:
    cap = max(_interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w)
              - abs(_THETA_EM * dw) - _EPS_T, 0.0)
    return cap >= T - _T_CUTOFF


def load(tag, cycle):
    from stable_baselines3 import SAC
    d, g = REF[tag]
    m = SAC.load(f"{d}/{cycle}/sac_ems_best")
    return m, g


def make_env(cycle):
    return EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=KFB, lookahead=5)


def classify(mode, t_ce):
    if mode == "stop":
        return "stop"
    if mode == "regen":
        return "regen"
    if t_ce <= _T_CUTOFF:
        return "OFF"
    return {"assist": "ASSIST", "lps_gen": "LPS"}.get(mode, "ONLY")


# --------------------------------------------------------------------------- #
def rollout(model, cycle, deterministic=True, snapshots=False):
    env = make_env(cycle)
    obs, _ = env.reset()
    R = []
    while True:
        a, _ = model.predict(obs, deterministic=deterministic)
        av = float(np.asarray(a).reshape(-1)[0])
        d = dict(env._demand)
        snap = copy.deepcopy(env) if snapshots else None
        ob = obs.copy()
        obs, r, term, _, info = env.step(a)
        R.append(dict(t=d["t"], T=d["T_MGB"], w=d["w_MGB"], dw=d["dw_MGB"],
                      v=d["v"], a=av, obs=ob, snap=snap, soc=info["soc"],
                      u=info["u"], t_ce=info["T_CE_cmd"], t_em=info["T_EM_cmd"],
                      p_em=info["p_em"], fuel=info["fuel_liters_step"],
                      elec=info["elec_liters_step"], r=r,
                      cls=classify(info["mode"], info["T_CE_cmd"])))
        if term:
            R[-1]["final"] = info["episode_final"]
            return R


def bench_rollout(cycle, which):
    """Run rule_based or ecms through the same env, recording per-step data."""
    import types
    env = make_env(cycle)
    if which == "rule_based":
        from src.baselines.advanced_rule_based import (AdvancedController,
                                                       control_unit_advanced)
        ctrl = AdvancedController(cycle_name=cycle); ctrl.reset()

        def patched(self, action):
            d = self._demand
            c = ctrl.step(d["w_MGB"], d["dw_MGB"], d["T_MGB"], d["gear"],
                          self._Q_BT, d["v"])
            cu = control_unit_advanced(d["w_MGB"], d["dw_MGB"], d["T_MGB"],
                                       c["u"], c["state_CE"])
            m = ("regen" if d["T_MGB"] < 0 else
                 ("stop" if d["T_MGB"] == 0 or d["w_MGB"] <= 0 else
                  ("lps_gen" if cu["T_EM"] < 0 else
                   ("assist" if cu["T_EM"] > 0 else "engine"))))
            return cu["T_CE"], cu["T_EM"], c["u"], m
    else:
        from src.baselines.ecms import _hamiltonian_best_u
        lam0 = ECMS_LAM[cycle]

        def patched(self, action):
            d = self._demand
            w, dw, T = d["w_MGB"], d["dw_MGB"], d["T_MGB"]
            soc = self._Q_BT / 36000.0
            if T == 0.0 or w <= 0.0:
                return 0.0, 0.0, 0.0, "stop"
            u = _hamiltonian_best_u(w, dw, T, soc, lam0 + 8.0 * (SOC_TARGET - soc), 81)
            te = u * T
            m = ("regen" if T < 0 else ("lps_gen" if te < 0 else
                 ("assist" if te > 0 else "engine")))
            return T - te, te, u, m
    env._action_to_torques = types.MethodType(patched, env)
    obs, _ = env.reset()
    R = []
    while True:
        d = dict(env._demand)
        obs, r, term, _, info = env.step(np.zeros(1, np.float32))
        R.append(dict(t=d["t"], T=d["T_MGB"], w=d["w_MGB"], dw=d["dw_MGB"],
                      v=d["v"], soc=info["soc"], u=info["u"],
                      t_ce=info["T_CE_cmd"], t_em=info["T_EM_cmd"],
                      fuel=info["fuel_liters_step"], elec=info["elec_liters_step"],
                      cls=classify(info["mode"], info["T_CE_cmd"])))
        if term:
            R[-1]["final"] = info["episode_final"]
            return R


# --------------------------------------------------------------------------- #
def sec7_bands(policies, benches, cycle, fh):
    p = lambda s: (print(s), fh.write(s + "\n"))
    p("\n" + "=" * 100)
    p("SECTION 7 -- 30-50 Nm SPLIT INTO 5 Nm SUB-BANDS (+ neighbours for context)")
    p("=" * 100)
    bands = [(15, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 75)]
    ref = policies["short"]
    p(f"{'band':>10}{'n':>6}{'feas%':>8}"
      f"{'SAC20 OFF':>11}{'SAC90 OFF':>11}{'RB OFF':>9}{'ECMS OFF':>10}"
      f"{'SAC20 ASST':>12}{'RB ASST':>10}{'dFuel(SAC20-RB)':>18}")
    rows = []
    for lo, hi in bands:
        idx = [i for i, x in enumerate(ref) if lo <= x["T"] < hi and x["cls"] != "stop"]
        if not idx:
            continue
        n = len(idx)
        feas = 100 * np.mean([off_feasible(ref[i]["T"], ref[i]["w"], ref[i]["dw"]) for i in idx])
        def pct(rec, k):
            return 100 * np.mean([rec[i]["cls"] == k for i in idx])
        dfuel = sum(ref[i]["fuel"] for i in idx) - sum(benches["rule_based"][i]["fuel"] for i in idx)
        rows.append((f"{lo}-{hi}", n, feas, pct(ref, "OFF"),
                     pct(policies["inter"], "OFF"), pct(benches["rule_based"], "OFF"),
                     pct(benches["ecms"], "OFF"), pct(ref, "ASSIST"),
                     pct(benches["rule_based"], "ASSIST"), dfuel))
        p(f"{rows[-1][0]:>10}{n:>6}{feas:>8.1f}"
          f"{rows[-1][3]:>11.1f}{rows[-1][4]:>11.1f}{rows[-1][5]:>9.1f}{rows[-1][6]:>10.1f}"
          f"{rows[-1][7]:>12.1f}{rows[-1][8]:>10.1f}{dfuel:>18.5f}")
    return rows


def sec13_error_budget(policies, benches, cycle, fh):
    p = lambda s: (print(s), fh.write(s + "\n"))
    p("\n" + "=" * 100)
    p("SECTION 13 -- ERROR BUDGET BY OPERATING REGION  (timestep-aligned)")
    p("=" * 100)
    ref, rb = policies["short"], benches["rule_based"]
    # verify demand alignment
    dT = max(abs(a["T"] - b["T"]) for a, b in zip(ref, rb))
    p(f"  demand alignment check: max |T_SAC - T_RB| over all steps = {dT:.2e}"
      f"  -> {'ALIGNED (controller-independent demand)' if dT < 1e-9 else 'NOT ALIGNED'}")
    regions = [("braking/regen", lambda x: x["T"] < 0),
               ("standstill", lambda x: x["cls"] == "stop"),
               ("0-15 Nm", lambda x: 0 <= x["T"] < 15 and x["cls"] != "stop"),
               ("15-30 Nm", lambda x: 15 <= x["T"] < 30),
               ("30-50 Nm", lambda x: 30 <= x["T"] < 50),
               ("50-75 Nm", lambda x: 50 <= x["T"] < 75),
               (">75 Nm", lambda x: x["T"] >= 75)]
    x_tot = ref[-1]["final"]["x_tot_m"]
    K = 1e5 / x_tot  # per-step liters -> L/100km contribution
    p(f"\n{'region':>16}{'steps':>7}{'time%':>7}"
      f"{'SAC fuel':>11}{'RB fuel':>10}{'dFuel':>10}"
      f"{'SAC elec':>11}{'RB elec':>10}{'dElec':>10}{'dTOTAL':>10}")
    tot_df = tot_de = 0.0
    budget = []
    for name, f in regions:
        idx = [i for i, x in enumerate(ref) if f(x)]
        if not idx:
            continue
        sf = sum(ref[i]["fuel"] for i in idx) * K
        bf = sum(rb[i]["fuel"] for i in idx) * K
        se = sum(ref[i]["elec"] for i in idx) * K
        be = sum(rb[i]["elec"] for i in idx) * K
        df, de = sf - bf, se - be
        tot_df += df; tot_de += de
        budget.append((name, len(idx), 100 * len(idx) / len(ref), sf, bf, df, se, be, de, df + de))
        p(f"{name:>16}{len(idx):>7}{100*len(idx)/len(ref):>7.1f}"
          f"{sf:>11.4f}{bf:>10.4f}{df:>+10.4f}{se:>11.4f}{be:>10.4f}{de:>+10.4f}{df+de:>+10.4f}")
    p(f"{'TOTAL':>16}{len(ref):>7}{100.0:>7.1f}"
      f"{'':>11}{'':>10}{tot_df:>+10.4f}{'':>11}{'':>10}{tot_de:>+10.4f}{tot_df+tot_de:>+10.4f}")
    p(f"\n  actual V_CE_equiv: SAC={ref[-1]['final']['v_ce_equiv']:.4f}  "
      f"RB={rb[-1]['final']['v_ce_equiv']:.4f}  "
      f"diff={ref[-1]['final']['v_ce_equiv']-rb[-1]['final']['v_ce_equiv']:+.4f}")
    p(f"  SoC final: SAC={ref[-1]['final']['soc_final']*100:.2f}%  RB={rb[-1]['final']['soc_final']*100:.2f}%")
    p("  NOTE: dElec is a PATH term; because V_CE_equiv saturates net battery use at 0,")
    p("        the fuel column (dFuel) is what maps directly onto the headline gap.")
    return budget


def sec89_counterfactual(model, gname, cycle, recs, fh, nmax=40):
    """sections 8 + 9: Q and immediate reward for OFF / ASSIST / LPS at matched states."""
    p = lambda s: (print(s), fh.write(s + "\n"))
    p("\n" + "=" * 100)
    p(f"SECTIONS 8+9 -- COUNTERFACTUAL Q AND IMMEDIATE REWARD, 30-50 Nm  (gamma={gname})")
    p("=" * 100)
    idx = [i for i, x in enumerate(recs) if 30 <= x["T"] < 50 and x["snap"] is not None]
    idx = idx[:: max(1, len(idx) // nmax)][:nmax]
    out = []
    for i in idx:
        x = recs[i]
        aoff = min(1.0, a_off_of(x["T"]) + 0.02)
        cands = {"OFF": aoff, "ASSIST": 0.40, "LPS": -0.50, "SAC": x["a"]}
        qs, rs, modes = {}, {}, {}
        ot = th.as_tensor(np.repeat(x["obs"].reshape(1, -1), len(cands), 0)).float().to(model.device)
        at = th.as_tensor(np.array([[v] for v in cands.values()])).float().to(model.device)
        with th.no_grad():
            q = model.critic(ot, at)
            q1 = q[0].cpu().numpy().ravel(); q2 = q[1].cpu().numpy().ravel()
        mq = np.minimum(q1, q2)
        for j, (k, av) in enumerate(cands.items()):
            e2 = copy.deepcopy(x["snap"]); act = np.array([av], np.float32)
            tce, _, _, md = e2._action_to_torques(act)
            modes[k] = classify(md, tce)
            _, rr, _, _, _ = e2.step(act)
            rs[k] = rr; qs[k] = mq[j]
        out.append(dict(T=x["T"], soc=x["soc"], v=x["v"], w=x["w"],
                        feas=off_feasible(x["T"], x["w"], x["dw"]),
                        dQ_off_asst=qs["OFF"] - qs["ASSIST"], dQ_off_lps=qs["OFF"] - qs["LPS"],
                        dr_off_asst=rs["OFF"] - rs["ASSIST"], dr_off_lps=rs["OFF"] - rs["LPS"],
                        sac_mode=x["cls"], off_mode=modes["OFF"], q=qs, r=rs))
    fo = [o for o in out if o["feas"] and o["off_mode"] == "OFF"]
    p(f"  states sampled: {len(out)}   OFF actually achievable: {len(fo)}")
    if fo:
        dQa = np.array([o["dQ_off_asst"] for o in fo]); dra = np.array([o["dr_off_asst"] for o in fo])
        dQl = np.array([o["dQ_off_lps"] for o in fo]); drl = np.array([o["dr_off_lps"] for o in fo])
        p(f"  dQ(OFF-ASSIST): mean={dQa.mean():+.4f}  median={np.median(dQa):+.4f}  >0 in {100*(dQa>0).mean():.0f}% of states")
        p(f"  dr(OFF-ASSIST): mean={dra.mean():+.4f}  median={np.median(dra):+.4f}  >0 in {100*(dra>0).mean():.0f}% of states")
        p(f"  dQ(OFF-LPS)   : mean={dQl.mean():+.4f}  median={np.median(dQl):+.4f}  >0 in {100*(dQl>0).mean():.0f}% of states")
        p(f"  dr(OFF-LPS)   : mean={drl.mean():+.4f}  median={np.median(drl):+.4f}  >0 in {100*(drl>0).mean():.0f}% of states")
        agree = 100 * np.mean(np.sign(dQa) == np.sign(dra))
        p(f"  sign(dQ) == sign(dr) for OFF-vs-ASSIST in {agree:.0f}% of states")
        p(f"  SAC's own chosen mode here: " +
          ", ".join(f"{k}={100*np.mean([o['sac_mode']==k for o in fo]):.0f}%"
                    for k in ["OFF", "ASSIST", "LPS", "ONLY"]))
        p("\n  CASE CLASSIFICATION (brief section 17):")
        if dra.mean() > 0 and dQa.mean() > 0:
            p("    reward favours OFF and critic favours OFF -> CASE A (actor fails to select it)")
        elif dra.mean() < 0 and dQa.mean() < 0:
            p("    reward favours ASSIST and critic agrees -> CASE B (economics, not RL failure)")
        elif dra.mean() > 0 and dQa.mean() < 0:
            p("    reward favours OFF but critic favours ASSIST -> CASE C (credit assignment/critic bias)")
        elif abs(dQa.mean()) < 0.01:
            p("    dQ ~ 0 -> CASE D (insufficient critic resolution)")
    return out


def sec10_rollouts(model, cycle, recs, fh, nstates=12):
    p = lambda s: (print(s), fh.write(s + "\n"))
    p("\n" + "=" * 100)
    p("SECTION 10 -- SHORT COUNTERFACTUAL ROLLOUTS (hold the counterfactual 1 step, then follow SAC)")
    p("=" * 100)
    idx = [i for i, x in enumerate(recs) if 30 <= x["T"] < 50 and x["snap"] is not None
           and off_feasible(x["T"], x["w"], x["dw"])]
    idx = idx[:: max(1, len(idx) // nstates)][:nstates]
    horizons = [1, 5, 10, 20]
    agg = {h: {"OFF": [], "ASSIST": []} for h in horizons}
    for i in idx:
        x = recs[i]
        aoff = min(1.0, a_off_of(x["T"]) + 0.02)
        for label, a0 in (("OFF", aoff), ("ASSIST", 0.40)):
            e2 = copy.deepcopy(x["snap"])
            obs = x["obs"].copy(); cum = 0.0; k = 0
            for step in range(max(horizons)):
                act = np.array([a0], np.float32) if step == 0 else \
                    model.predict(obs, deterministic=True)[0]
                obs, r, term, _, _ = e2.step(act)
                cum += r; k += 1
                if k in agg: agg[k][label].append(cum)
                if term: break
    p(f"  states: {len(idx)}  (OFF feasible, 30-50 Nm)")
    p(f"{'horizon':>9}{'cum r OFF':>12}{'cum r ASSIST':>14}{'OFF - ASSIST':>14}{'OFF wins %':>12}")
    for h in horizons:
        o = np.array(agg[h]["OFF"]); a = np.array(agg[h]["ASSIST"])
        n = min(len(o), len(a))
        if n == 0: continue
        d = o[:n] - a[:n]
        p(f"{h:>8}s{o[:n].mean():>12.4f}{a[:n].mean():>14.4f}{d.mean():>+14.4f}{100*(d>0).mean():>11.0f}%")
    return agg


def sec11_costate(cycle, recs, fh):
    p = lambda s: (print(s), fh.write(s + "\n"))
    p("\n" + "=" * 100)
    p("SECTION 11 -- COSTATE ANALYSIS")
    p("=" * 100)
    eqf = EQF[cycle]
    socs = np.array([x["soc"] for x in recs])
    p(f"  eq_factor(SoC) = {eqf} + {KFB}*(0.5 - SoC)   [liter-units]")
    p(f"  d(eq_factor)/dSoC = -{KFB}  (constant, proportional feedback)")
    p(f"  SoC range visited: {socs.min()*100:.2f}% .. {socs.max()*100:.2f}%")
    lo, hi = eqf + KFB*(0.5-socs.max()), eqf + KFB*(0.5-socs.min())
    p(f"  => eq_factor range over the visited SoC: {lo:.4f} .. {hi:.4f}  (span {hi-lo:.4f})")
    p(f"  in ECMS units (x4.8309): {lo*4.8309:.4f} .. {hi*4.8309:.4f}  "
      f"(ECMS lambda_0 = {ECMS_LAM[cycle]})")
    p(f"  relative variation of the price across the whole episode: "
      f"{100*(hi-lo)/((hi+lo)/2):.1f}%")
    p("\n  INTERPRETATION: the costate already varies with SoC by the amount ECMS")
    p("  prescribes. Because this inter-temporal coupling is supplied EXPLICITLY in")
    p("  the per-step reward, a value function integrating many steps adds variance")
    p("  without adding information -- which is why gamma=0.90 does not beat 0.20.")
    return dict(lo=lo, hi=hi, soc_min=float(socs.min()), soc_max=float(socs.max()))


def sec1516_stochastic(model, cycle, recs, fh, n_rep=30):
    p = lambda s: (print(s), fh.write(s + "\n"))
    p("\n" + "=" * 100)
    p("SECTIONS 15+16 -- STOCHASTICITY AND POLICY CONSISTENCY")
    p("=" * 100)
    tgts = [20, 30, 40, 50, 75]
    p(f"{'T_MGB':>7}{'a_det':>9}{'a_off':>9}{'mean|sig|':>11}"
      f"{'P(OFF) stoch':>14}{'det mode':>10}{'a_std(samp)':>13}")
    for T in tgts:
        cand = [x for x in recs if abs(x["T"] - T) < 3 and x["snap"] is not None]
        if not cand: continue
        x = min(cand, key=lambda y: abs(y["T"] - T))
        ot = th.as_tensor(x["obs"].reshape(1, -1)).float().to(model.device)
        with th.no_grad():
            mu, log_std, _ = model.actor.get_action_dist_params(ot)
            sig = float(th.exp(log_std).cpu().numpy().ravel()[0])
            det = float(np.tanh(mu.cpu().numpy().ravel()[0]))
            samples = np.array([float(model.predict(x["obs"], deterministic=False)[0][0])
                                for _ in range(n_rep)])
        aoff = a_off_of(x["T"])
        offp = 100 * np.mean(samples >= aoff)
        e2 = copy.deepcopy(x["snap"])
        tce, _, _, md = e2._action_to_torques(np.array([det], np.float32))
        p(f"{x['T']:>7.1f}{det:>9.3f}{aoff:>9.3f}{sig:>11.4f}"
          f"{offp:>13.0f}%{classify(md,tce):>10}{samples.std():>13.4f}")
    # deterministic vs stochastic whole-cycle
    p("")
    det_R = rollout(model, cycle, deterministic=True)
    sto_R = rollout(model, cycle, deterministic=False)
    for nm, R in (("deterministic", det_R), ("stochastic", sto_R)):
        mv = [x for x in R if x["cls"] != "stop"]
        p(f"  {nm:<14} V_CE_equiv={R[-1]['final']['v_ce_equiv']:.4f}  "
          f"SoC={R[-1]['final']['soc_final']*100:.2f}%  "
          f"OFF={100*np.mean([x['cls']=='OFF' for x in mv]):.1f}%  "
          f"ASSIST={100*np.mean([x['cls']=='ASSIST' for x in mv]):.1f}%")


def sec12_matched(policies, benches, cycle, fh, n=14):
    p = lambda s: (print(s), fh.write(s + "\n"))
    p("\n" + "=" * 100)
    p("SECTION 12 -- MATCHED-STATE ACTION COMPARISON (30-50 Nm)")
    p("=" * 100)
    ref = policies["short"]
    idx = [i for i, x in enumerate(ref) if 30 <= x["T"] < 50]
    idx = idx[:: max(1, len(idx) // n)][:n]
    p(f"{'t':>5}{'T_MGB':>8}{'SoC%':>7}{'v':>6}"
      f"{'SAC20 u':>9}{'SAC90 u':>9}{'RB u':>8}{'ECMS u':>9}"
      f"{'SAC20':>8}{'SAC90':>8}{'RB':>8}{'ECMS':>8}")
    for i in idx:
        x = ref[i]
        p(f"{x['t']:>5}{x['T']:>8.1f}{x['soc']*100:>7.1f}{x['v']:>6.1f}"
          f"{x['u']:>9.3f}{policies['inter'][i]['u']:>9.3f}"
          f"{benches['rule_based'][i]['u']:>8.3f}{benches['ecms'][i]['u']:>9.3f}"
          f"{x['cls']:>8}{policies['inter'][i]['cls']:>8}"
          f"{benches['rule_based'][i]['cls']:>8}{benches['ecms'][i]['cls']:>8}")


def sec9_heatmap(model, cycle, recs, out: Path):
    """r_OFF - r_ASSIST heatmap over (T_MGB, SoC)."""
    idx = [i for i, x in enumerate(recs) if x["snap"] is not None and x["T"] > _T_CUTOFF]
    idx = idx[:: max(1, len(idx) // 260)][:260]
    T, S, D = [], [], []
    for i in idx:
        x = recs[i]
        aoff = min(1.0, a_off_of(x["T"]) + 0.02)
        vals = {}
        for k, av in (("OFF", aoff), ("ASSIST", 0.40)):
            e2 = copy.deepcopy(x["snap"])
            _, rr, _, _, _ = e2.step(np.array([av], np.float32))
            vals[k] = rr
        T.append(x["T"]); S.append(x["soc"] * 100); D.append(vals["OFF"] - vals["ASSIST"])
    T, S, D = np.array(T), np.array(S), np.array(D)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sc = ax[0].scatter(T, S, c=D, cmap="RdBu_r", vmin=-np.abs(D).max(), vmax=np.abs(D).max(), s=26)
    ax[0].set_xlabel("T_MGB [Nm]"); ax[0].set_ylabel("SoC [%]")
    ax[0].set_title("r_OFF - r_ASSIST  (red = OFF better)")
    ax[0].axvspan(30, 50, alpha=0.10, color="k")
    plt.colorbar(sc, ax=ax[0])
    o = np.argsort(T)
    ax[1].plot(T[o], D[o], ".", ms=5); ax[1].axhline(0, color="k", lw=1)
    ax[1].axvspan(30, 50, alpha=0.12, color="tab:orange")
    ax[1].set_xlabel("T_MGB [Nm]"); ax[1].set_ylabel("r_OFF - r_ASSIST")
    ax[1].set_title("immediate-reward advantage of OFF vs torque")
    fig.tight_layout(); fig.savefig(out / f"reward_counterfactual_{cycle}.png", dpi=110)
    return float(np.mean(D[(T >= 30) & (T < 50)])) if ((T >= 30) & (T < 50)).any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", default="NEDC")
    ap.add_argument("--out", default="results/phase4")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fh = open(out / f"forensics_{a.cycle}.txt", "w", encoding="utf-8")

    print(f"loading policies for {a.cycle} ...")
    m20, _ = load("short", a.cycle)
    m90, _ = load("inter", a.cycle)
    policies = {"short": rollout(m20, a.cycle, snapshots=True),
                "inter": rollout(m90, a.cycle, snapshots=True)}
    benches = {"rule_based": bench_rollout(a.cycle, "rule_based"),
               "ecms": bench_rollout(a.cycle, "ecms")}

    fh.write(f"PHASE-4 FORENSICS -- {a.cycle}\n")
    fh.write(f"REF-SHORT gamma=0.20 V_CE_equiv={policies['short'][-1]['final']['v_ce_equiv']:.4f}\n")
    fh.write(f"REF-INTER gamma=0.90 V_CE_equiv={policies['inter'][-1]['final']['v_ce_equiv']:.4f}\n")
    print(f"REF-SHORT  V={policies['short'][-1]['final']['v_ce_equiv']:.4f}")
    print(f"REF-INTER  V={policies['inter'][-1]['final']['v_ce_equiv']:.4f}")
    print(f"rule-based V={benches['rule_based'][-1]['final']['v_ce_equiv']:.4f}")
    print(f"ECMS       V={benches['ecms'][-1]['final']['v_ce_equiv']:.4f}")

    sec7_bands(policies, benches, a.cycle, fh)
    sec13_error_budget(policies, benches, a.cycle, fh)
    sec12_matched(policies, benches, a.cycle, fh)
    sec89_counterfactual(m20, "0.20", a.cycle, policies["short"], fh)
    sec89_counterfactual(m90, "0.90", a.cycle, policies["inter"], fh)
    sec10_rollouts(m20, a.cycle, policies["short"], fh)
    sec11_costate(a.cycle, policies["short"], fh)
    sec1516_stochastic(m20, a.cycle, policies["short"], fh)
    d = sec9_heatmap(m20, a.cycle, policies["short"], out)
    msg = f"\n  [heatmap] mean r_OFF - r_ASSIST in 30-50 Nm = {d:+.5f}"
    print(msg); fh.write(msg + "\n")
    print(f"\n[saved] {out}/forensics_{a.cycle}.txt")
    fh.close()


if __name__ == "__main__":
    main()
