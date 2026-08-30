"""
phase8_qoracle.py
=================
PHASE 8B + 8G -- the pivotal diagnostic.

Builds three diagnostic policies on the EXISTING validated env / plant / critic
(NO training, NO physics change, NO benchmark change, NO reward change):

  Policy A  -- current SAC deterministic actor            (models_p5*_k2.5)
  Policy B  -- SAC-Q ORACLE: at every step pick the feasible action that
               maximizes the trained SAC min(Q1,Q2). The critic is used exactly
               as trained; only the actor is replaced by arg-max over a dense
               feasible action grid.
  Policy C  -- ECMS (reference only)
  + advanced rule-based (reference only)

If B ~ C while A << B  -> the ACTOR is the bottleneck (build the mode-aware actor).
If B << C              -> the CRITIC / reward / state is the bottleneck.

Also (8G / section 15): engine operating-point counterfactual on matched ECMS
states -- does Q prefer the high-engine-load point ECMS uses, or the low-load
point the current actor uses?

    python -m results.phase8_qoracle --cycle NEDC
    python -m results.phase8_qoracle --cycle FTP75
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch as th

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import SAC

from src.env.ems_env import EMSEnv, ZB_MODEAWARE, SOC_TARGET, map_action_to_u
from src.env.powertrain import _T_CUTOFF, _Q_BT_0
from src.baselines.ecms import _hamiltonian_best_u

from results.phase7_forensics import (
    EQF, LAM0, ECMS_V, BENCH, AMAP, KFB_CONTROL, CKPTS, ECMS_UNIT,
    q_at, actor_at, torques_from_u, mode_of_u, classify_final,
    ecms_rollout, rule_rollout, matched_states, pct,
)

TB = [(0, 15, "0-15"), (15, 30, "15-30"), (30, 35, "30-35"),
      (35, 50, "35-50"), (50, 75, "50-75"), (75, 1e9, ">75")]
NGRID = 121


# --------------------------------------------------------------------------- #
# Policy B -- SAC-Q oracle rollout                                            #
# --------------------------------------------------------------------------- #
def q_oracle_rollout(model, cycle, kfb=KFB_CONTROL, amap=AMAP, ngrid=NGRID):
    grid = np.linspace(-1.0, 1.0, ngrid).astype(np.float32)
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=kfb, lookahead=5, action_map=amap)
    obs, _ = env.reset()
    R = []
    while True:
        d = dict(env._demand)
        soc_b = env._Q_BT / _Q_BT_0
        Q = q_at(model, obs, grid)
        a = float(grid[int(np.argmax(Q))])
        obs, r, t, _, i = env.step(np.array([a], np.float32))
        R.append(dict(T=d["T_MGB"], w=d["w_MGB"], dw=d["dw_MGB"], v=d["v"], gear=d["gear"],
                      soc_before=soc_b, soc=i["soc"], a=a, u=i["u"],
                      t_ce=i["T_CE_cmd"], t_em=i["T_EM_cmd"], p_em=i["p_em"],
                      cls=classify_final(i["mode"], i["T_CE_cmd"]),
                      fuel=i["fuel_liters_step"], elec=i["elec_liters_step"], r=float(r),
                      q_spread=float(Q.max() - Q.min())))
        if t:
            R[-1]["final"] = i["episode_final"]
            return R


def sac_actor_rollout(model, cycle, kfb=KFB_CONTROL, amap=AMAP):
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=kfb, lookahead=5, action_map=amap)
    obs, _ = env.reset()
    R = []
    while True:
        d = dict(env._demand)
        soc_b = env._Q_BT / _Q_BT_0
        a, _ = model.predict(obs, deterministic=True)
        obs, r, t, _, i = env.step(a)
        R.append(dict(T=d["T_MGB"], w=d["w_MGB"], dw=d["dw_MGB"], v=d["v"], gear=d["gear"],
                      soc_before=soc_b, soc=i["soc"], a=float(np.asarray(a).ravel()[0]),
                      u=i["u"], t_ce=i["T_CE_cmd"], t_em=i["T_EM_cmd"], p_em=i["p_em"],
                      cls=classify_final(i["mode"], i["T_CE_cmd"]),
                      fuel=i["fuel_liters_step"], elec=i["elec_liters_step"], r=float(r)))
        if t:
            R[-1]["final"] = i["episode_final"]
            return R


# --------------------------------------------------------------------------- #
# scorecard from a rollout record list                                       #
# --------------------------------------------------------------------------- #
def scorecard(R, label):
    fin = R[-1]["final"]
    mov = [x for x in R if x["cls"] != "stop"]
    n = max(len(mov), 1)
    f = lambda k: 100.0 * sum(1 for x in mov if x["cls"] == k) / n
    socs = np.array([x["soc"] for x in R])
    eng_on = [x for x in R if x["t_ce"] > _T_CUTOFF]
    return dict(
        label=label, v_ce=fin["v_ce_equiv"], v_liter=fin["v_liter"],
        soc_final=fin["soc_final"], d_soc_pp=(fin["soc_final"] - 0.5) * 100.0,
        soc_min=float(socs.min() * 100), soc_max=float(socs.max() * 100),
        charge_sustaining=bool(abs(fin["soc_final"] - 0.5) <= 0.02),
        off_pct=f("OFF"), assist_pct=f("ASSIST"), lps_pct=f("LPS"),
        regen_pct=f("REGEN"), only_pct=f("ONLY"),
        engine_on_s=len(eng_on),
        mean_eng_tce_when_on=float(np.mean([x["t_ce"] for x in eng_on])) if eng_on else 0.0,
        violations=int(sum(1 for s in socs if not (0.05 <= s <= 0.95))),
    )


def regional(R_sac, R_orc, R_ecms, K):
    """per-torque-band OFF% + engine|Tce| for A / B / C, and B-vs-C dFuel/dElec."""
    rows = {}
    for lo, hi, nm in TB:
        ia = [i for i, x in enumerate(R_sac) if lo <= x["T"] < hi]
        ib = [i for i, x in enumerate(R_orc) if lo <= x["T"] < hi]
        ic = [i for i, x in enumerate(R_ecms) if lo <= x["T"] < hi]
        if not ib:
            continue
        def off(idx, RR): return 100.0 * np.mean([RR[i]["cls"] == "OFF" for i in idx]) if idx else 0.0
        def engt(idx, RR):
            v = [RR[i]["t_ce"] for i in idx if RR[i]["cls"] not in ("OFF", "REGEN", "stop")]
            return float(np.mean(v)) if v else 0.0
        rows[nm] = dict(
            n_orc=len(ib),
            off_sac=off(ia, R_sac), off_orc=off(ib, R_orc), off_ecms=off(ic, R_ecms),
            engt_sac=engt(ia, R_sac), engt_orc=engt(ib, R_orc), engt_ecms=engt(ic, R_ecms),
            dfuel_orc_ecms=(sum(R_orc[i]["fuel"] for i in ib) - sum(R_ecms[i]["fuel"] for i in ic)) * K,
            delec_orc_ecms=(sum(R_orc[i]["elec"] for i in ib) - sum(R_ecms[i]["elec"] for i in ic)) * K,
            dfuel_sac_ecms=(sum(R_sac[i]["fuel"] for i in ia) - sum(R_ecms[i]["fuel"] for i in ic)) * K,
        )
    return rows


# --------------------------------------------------------------------------- #
# 8G -- engine operating-point counterfactual on matched ECMS states         #
# --------------------------------------------------------------------------- #
def engine_op_counterfactual(model, cycle, P, out, kfb=KFB_CONTROL, amap=AMAP):
    P(f"\n{'='*92}\n8G  ENGINE OPERATING-POINT COUNTERFACTUAL (matched ECMS states) -- {cycle}\n{'='*92}")
    S = matched_states(cycle, kfb, amap, trajectory="ecms")
    grid = np.linspace(-1, 1, 161).astype(np.float32)
    res = {"cycle": cycle, "regions": {}}
    P(f"  {'region':>8}{'n':>5}{'Q@ECMS-load':>12}{'Q@actor-load':>13}{'Q@max-load':>12}"
      f"{'ecms Tce':>10}{'actor Tce':>11}{'argmaxQ Tce':>12}{'verdict':>26}")
    for lo, hi, nm in TB:
        sel = [s for s in S if lo <= s["T"] < hi]
        if len(sel) < 8:
            continue
        sel = sel[:: max(1, len(sel) // 60)][:60]
        q_at_ecms, q_at_actor, q_at_maxload = [], [], []
        tce_ecms, tce_actor, tce_argmaxq = [], [], []
        crit_prefers_highload = []
        for st in sel:
            T, w, dw, soc = st["T"], st["w"], st["dw"], st["soc"]
            Q = q_at(model, st["obs"], grid)
            us = np.array([map_action_to_u(float(a), T, amap, w, dw) for a in grid])
            tce = np.array([torques_from_u(u, T, w, dw, soc)[0] for u in us])
            modes = np.array([mode_of_u(u, T, w, dw, soc) for u in us])
            on = (modes != "OFF") & (modes != "REGEN") & (modes != "STOP")
            # ECMS engine load at this state
            u_e = _hamiltonian_best_u(w, dw, T, soc, LAM0[cycle] + 8.0 * (SOC_TARGET - soc), 81)
            tce_e = torques_from_u(u_e, T, w, dw, soc)[0]
            # actor engine load
            mu, _ = actor_at(model, st["obs"])
            a_sac = float(np.tanh(mu))
            u_s = map_action_to_u(a_sac, T, amap, w, dw)
            tce_s = torques_from_u(u_s, T, w, dw, soc)[0]
            # nearest grid pts
            i_e = int(np.argmin(np.abs(tce - tce_e)))
            i_s = int(np.argmin(np.abs(grid - a_sac)))
            i_argmax = int(np.argmax(Q))
            q_at_ecms.append(float(Q[i_e])); q_at_actor.append(float(Q[i_s]))
            tce_ecms.append(float(tce_e)); tce_actor.append(float(tce_s))
            tce_argmaxq.append(float(tce[i_argmax]))
            if on.any():
                i_maxload = int(np.where(on)[0][np.argmax(tce[on])])
                q_at_maxload.append(float(Q[i_maxload]))
                # among ON actions only: does Q rank higher-load above lower-load?
                on_idx = np.where(on)[0]
                if on_idx.size >= 3:
                    order = np.argsort(tce[on_idx])
                    lo_q = np.mean(Q[on_idx[order[:max(1, len(order)//3)]]])
                    hi_q = np.mean(Q[on_idx[order[-max(1, len(order)//3):]]])
                    crit_prefers_highload.append(hi_q > lo_q)
        verdict = ("Q PREFERS high load -> ACTOR problem"
                   if np.mean(crit_prefers_highload) > 0.55
                   else ("Q PREFERS low load -> CRITIC/REWARD problem"
                         if np.mean(crit_prefers_highload) < 0.45 else "mixed"))
        row = dict(n=len(sel),
                   q_at_ecms_load=float(np.mean(q_at_ecms)),
                   q_at_actor_load=float(np.mean(q_at_actor)),
                   q_at_maxload=float(np.mean(q_at_maxload)) if q_at_maxload else None,
                   tce_ecms=float(np.mean(tce_ecms)), tce_actor=float(np.mean(tce_actor)),
                   tce_argmaxq=float(np.mean(tce_argmaxq)),
                   frac_states_Q_prefers_highload=float(np.mean(crit_prefers_highload)) if crit_prefers_highload else None,
                   verdict=verdict)
        res["regions"][nm] = row
        P(f"  {nm:>8}{len(sel):>5}{row['q_at_ecms_load']:>12.4f}{row['q_at_actor_load']:>13.4f}"
          f"{(row['q_at_maxload'] if row['q_at_maxload'] is not None else float('nan')):>12.4f}"
          f"{row['tce_ecms']:>10.1f}{row['tce_actor']:>11.1f}{row['tce_argmaxq']:>12.1f}{verdict:>26}")
    json.dump(res, open(out / f"data/engine_op_counterfactual_{cycle}.json", "w"), indent=2)
    return res


# --------------------------------------------------------------------------- #
# main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, choices=["NEDC", "FTP75"])
    ap.add_argument("--out", default="results/phase8")
    a = ap.parse_args()
    out = Path(a.out); (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    cyc = a.cycle
    fh = open(out / f"logs/phase8_qoracle_{cyc}.txt", "w", encoding="utf-8")
    P = lambda s: (print(s), fh.write(str(s) + "\n"))
    P(f"PHASE 8B/8G -- Q-ORACLE CEILING + ENGINE-OP COUNTERFACTUAL -- {cyc}")
    P("(no training, no physics change, critic used exactly as trained)")

    seeds = CKPTS[cyc]["control_k2.5_gated"]
    R_ecms = ecms_rollout(cyc)
    R_rb = rule_rollout(cyc)
    K = 1e5 / R_ecms[-1]["final"]["x_tot_m"]

    sc_A, sc_B = [], []
    R_A0 = R_B0 = None
    for si, d in enumerate(seeds):
        m = SAC.load(f"{d}/{cyc}/sac_ems_best")
        RA = sac_actor_rollout(m, cyc)
        RB = q_oracle_rollout(m, cyc)
        sc_A.append(scorecard(RA, f"A_sac_seed{si}"))
        sc_B.append(scorecard(RB, f"B_qoracle_seed{si}"))
        if si == 0:
            R_A0, R_B0 = RA, RB
        P(f"  seed{si}:  A(actor) V_CE={sc_A[-1]['v_ce']:.4f} OFF={sc_A[-1]['off_pct']:.1f}% "
          f"dSoC={sc_A[-1]['d_soc_pp']:+.2f}pp CS={sc_A[-1]['charge_sustaining']} viol={sc_A[-1]['violations']}"
          f"  ||  B(Q-oracle) V_CE={sc_B[-1]['v_ce']:.4f} OFF={sc_B[-1]['off_pct']:.1f}% "
          f"dSoC={sc_B[-1]['d_soc_pp']:+.2f}pp CS={sc_B[-1]['charge_sustaining']} viol={sc_B[-1]['violations']}")

    def agg(scs, key):
        v = np.array([s[key] for s in scs], float)
        return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    min=float(v.min()), max=float(v.max()),
                    ci95=[float(v.mean() - 1.96 * v.std(ddof=1) / np.sqrt(len(v))),
                          float(v.mean() + 1.96 * v.std(ddof=1) / np.sqrt(len(v)))] if len(v) > 1 else [float(v.mean())] * 2)

    sc_C = scorecard(R_ecms, "C_ecms")
    sc_RB = scorecard(R_rb, "advanced_rule_based")

    summary = dict(
        cycle=cyc, benchmark_rb=BENCH[cyc], benchmark_ecms=ECMS_V[cyc],
        A_current_sac=dict(scorecards=sc_A, v_ce=agg(sc_A, "v_ce"),
                           off_pct=agg(sc_A, "off_pct"), d_soc_pp=agg(sc_A, "d_soc_pp"),
                           mean_eng_tce_when_on=agg(sc_A, "mean_eng_tce_when_on"),
                           charge_sustaining=[s["charge_sustaining"] for s in sc_A],
                           violations=[s["violations"] for s in sc_A]),
        B_q_oracle=dict(scorecards=sc_B, v_ce=agg(sc_B, "v_ce"),
                        off_pct=agg(sc_B, "off_pct"), d_soc_pp=agg(sc_B, "d_soc_pp"),
                        mean_eng_tce_when_on=agg(sc_B, "mean_eng_tce_when_on"),
                        charge_sustaining=[s["charge_sustaining"] for s in sc_B],
                        violations=[s["violations"] for s in sc_B]),
        C_ecms=sc_C, advanced_rule_based=sc_RB,
        regional_A_B_C=regional(R_A0, R_B0, R_ecms, K),
    )

    # ceiling interpretation
    vA, vB, vC = summary["A_current_sac"]["v_ce"]["mean"], summary["B_q_oracle"]["v_ce"]["mean"], ECMS_V[cyc]
    gap_AC = vA - vC
    closed_by_oracle = (vA - vB)
    frac = closed_by_oracle / gap_AC if gap_AC else float("nan")
    summary["ceiling"] = dict(
        v_A=vA, v_B=vB, v_ECMS=vC, v_RB=BENCH[cyc],
        gap_A_to_ECMS=gap_AC, oracle_closes=closed_by_oracle,
        frac_of_A_ECMS_gap_closed_by_oracle=frac,
        oracle_beats_RB=bool(vB < BENCH[cyc]),
        oracle_vs_ECMS_pct=100.0 * (vB - vC) / vC,
        interpretation=("ACTOR is the dominant bottleneck (Q-oracle closes most of the gap "
                        "and reaches/relieves RB)"
                        if frac > 0.5 and vB < BENCH[cyc] * 1.02 else
                        ("ACTOR is a major but partial bottleneck" if frac > 0.3 else
                         "CRITIC/REWARD/STATE limits performance -- Q-oracle stays far from ECMS")))
    P(f"\n  {'-'*70}")
    P(f"  CEILING:  A(actor) {vA:.4f}  ->  B(Q-oracle) {vB:.4f}  ->  ECMS {vC:.4f}   (RB {BENCH[cyc]:.4f})")
    P(f"  Q-oracle closes {closed_by_oracle:+.4f} of the {gap_AC:+.4f} A->ECMS gap  ({frac*100:.0f}%)")
    P(f"  Q-oracle beats advanced rule-based? {summary['ceiling']['oracle_beats_RB']}   "
      f"(Q-oracle vs ECMS: {summary['ceiling']['oracle_vs_ECMS_pct']:+.1f}%)")
    P(f"  => {summary['ceiling']['interpretation']}")

    P(f"\n  regional (seed0)  {'band':>7}{'OFF% A/B/C':>18}{'eng|Tce| A/B/C':>20}{'dFuel B-C':>11}{'dFuel A-C':>11}")
    for nm, r in summary["regional_A_B_C"].items():
        P(f"  {'':7}{nm:>7}{r['off_sac']:>6.0f}/{r['off_orc']:>4.0f}/{r['off_ecms']:>4.0f}"
          f"{r['engt_sac']:>8.0f}/{r['engt_orc']:>4.0f}/{r['engt_ecms']:>4.0f}"
          f"{r['dfuel_orc_ecms']:>+11.4f}{r['dfuel_sac_ecms']:>+11.4f}")

    json.dump(summary, open(out / f"data/qoracle_ceiling_{cyc}.json", "w"), indent=2)

    # 8G
    m0 = SAC.load(f"{seeds[0]}/{cyc}/sac_ems_best")
    engine_op_counterfactual(m0, cyc, P, out)

    # figure: A vs B vs C
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    bands = list(summary["regional_A_B_C"].keys())
    x = np.arange(len(bands)); w = 0.26
    ax[0].bar(x - w, [summary["regional_A_B_C"][b]["off_sac"] for b in bands], w, label="A actor", color="#d62728")
    ax[0].bar(x, [summary["regional_A_B_C"][b]["off_orc"] for b in bands], w, label="B Q-oracle", color="#1f77b4")
    ax[0].bar(x + w, [summary["regional_A_B_C"][b]["off_ecms"] for b in bands], w, label="C ECMS", color="#2ca02c")
    ax[0].set_xticks(x); ax[0].set_xticklabels(bands, rotation=30); ax[0].set_title(f"{cyc} OFF% by torque band"); ax[0].legend()
    ax[1].bar(x - w, [summary["regional_A_B_C"][b]["engt_sac"] for b in bands], w, label="A actor", color="#d62728")
    ax[1].bar(x, [summary["regional_A_B_C"][b]["engt_orc"] for b in bands], w, label="B Q-oracle", color="#1f77b4")
    ax[1].bar(x + w, [summary["regional_A_B_C"][b]["engt_ecms"] for b in bands], w, label="C ECMS", color="#2ca02c")
    ax[1].set_xticks(x); ax[1].set_xticklabels(bands, rotation=30); ax[1].set_title(f"{cyc} engine |T_CE| when ON"); ax[1].legend()
    labels = ["A actor", "B Q-oracle", "RB", "ECMS"]
    vals = [vA, vB, BENCH[cyc], ECMS_V[cyc]]
    ax[2].bar(labels, vals, color=["#d62728", "#1f77b4", "#7f7f7f", "#2ca02c"])
    ax[2].axhline(BENCH[cyc], color="#7f7f7f", ls="--", lw=1)
    ax[2].set_title(f"{cyc} V_CE_equiv  (Q-oracle ceiling)"); ax[2].set_ylabel("L/100km")
    for i, v in enumerate(vals):
        ax[2].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.suptitle(f"PHASE 8B -- Q-oracle ceiling -- {cyc}", fontsize=13)
    fig.tight_layout(); fig.savefig(out / f"figures/qoracle_ceiling_{cyc}.png", dpi=110); plt.close(fig)
    P(f"\n[saved] {out}/data/qoracle_ceiling_{cyc}.json  + figures/qoracle_ceiling_{cyc}.png")
    fh.close()


if __name__ == "__main__":
    main()
