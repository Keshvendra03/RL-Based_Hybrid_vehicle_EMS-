"""
phase8_reward_state.py
======================
PHASE 8 -- section 16 (reward sufficiency) + section 21 state-sufficiency note
        + section 3 / 8A supplement (actor-to-argmaxQ distance per band, 3-seed
          vehicle scorecard).

NO training. NO physics / reward / benchmark change. Matched-state methodology
identical to Phase 7 (ECMS-trajectory deep-copied clean envs).

Central question (section 16): is the CURRENT reward capable of producing the
ECMS engine operating point? I.e. does the *immediate reward* r(a) prefer the
ECMS (harder) engine load, or the actor's (softer) load?

    python -m results.phase8_reward_state --cycle NEDC
    python -m results.phase8_reward_state --cycle FTP75
"""
from __future__ import annotations
import argparse, copy, csv, json
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC

from src.env.ems_env import EMSEnv, map_action_to_u, SOC_TARGET
from src.env.powertrain import _T_CUTOFF
from src.baselines.ecms import _hamiltonian_best_u
from results.phase7_forensics import (
    EQF, LAM0, AMAP, KFB_CONTROL, CKPTS, q_at, actor_at,
    torques_from_u, mode_of_u, matched_states, reward_of_action, pct,
)
from results.phase8_qoracle import scorecard, sac_actor_rollout, TB


def reward_counterfactual(cycle, P, out):
    P(f"\n{'='*92}\n8H-pre  REWARD SUFFICIENCY COUNTERFACTUAL (matched ECMS states) -- {cycle}\n{'='*92}")
    m = SAC.load(f"{CKPTS[cycle]['control_k2.5_gated'][0]}/{cycle}/sac_ems_best")
    S = matched_states(cycle, KFB_CONTROL, AMAP, trajectory="ecms")
    grid = np.linspace(-1, 1, 161).astype(np.float32)
    res = {"cycle": cycle, "regions": {}}
    P(f"  {'region':>8}{'n':>5}{'r@ECMS-load':>13}{'r@actor-load':>14}{'r@argmax-r':>12}"
      f"{'ecms Tce':>10}{'actor Tce':>11}{'argmaxR Tce':>13}{'verdict':>34}")
    for lo, hi, nm in TB:
        sel = [s for s in S if lo <= s["T"] < hi]
        if len(sel) < 8:
            continue
        sel = sel[:: max(1, len(sel) // 60)][:60]
        r_ecms_l, r_actor_l, r_argmax = [], [], []
        tce_ecms, tce_actor, tce_argmaxr = [], [], []
        reward_prefers_ecms_over_actor = []
        for st in sel:
            T, w, dw, soc = st["T"], st["w"], st["dw"], st["soc"]
            rr = np.array([reward_of_action(st["env"], float(a)) for a in grid])
            us = np.array([map_action_to_u(float(a), T, AMAP, w, dw) for a in grid])
            tce = np.array([torques_from_u(u, T, w, dw, soc)[0] for u in us])
            u_e = _hamiltonian_best_u(w, dw, T, soc, LAM0[cycle] + 8.0 * (SOC_TARGET - soc), 81)
            tce_e = torques_from_u(u_e, T, w, dw, soc)[0]
            mu, _ = actor_at(m, st["obs"])
            a_sac = float(np.tanh(mu))
            u_s = map_action_to_u(a_sac, T, AMAP, w, dw)
            tce_s = torques_from_u(u_s, T, w, dw, soc)[0]
            i_e = int(np.argmin(np.abs(tce - tce_e)))
            i_s = int(np.argmin(np.abs(grid - a_sac)))
            i_r = int(np.argmax(rr))
            r_ecms_l.append(float(rr[i_e])); r_actor_l.append(float(rr[i_s]))
            r_argmax.append(float(rr[i_r]))
            tce_ecms.append(float(tce_e)); tce_actor.append(float(tce_s))
            tce_argmaxr.append(float(tce[i_r]))
            reward_prefers_ecms_over_actor.append(rr[i_e] >= rr[i_s])
        fr = float(np.mean(reward_prefers_ecms_over_actor))
        verdict = ("REWARD prefers ECMS load -> reward OK, critic/actor problem" if fr > 0.55
                   else ("REWARD prefers actor(soft) load -> REWARD problem" if fr < 0.45
                         else "reward ~indifferent"))
        row = dict(n=len(sel), r_at_ecms_load=float(np.mean(r_ecms_l)),
                   r_at_actor_load=float(np.mean(r_actor_l)),
                   r_at_argmaxr=float(np.mean(r_argmax)),
                   tce_ecms=float(np.mean(tce_ecms)), tce_actor=float(np.mean(tce_actor)),
                   tce_argmaxr=float(np.mean(tce_argmaxr)),
                   frac_reward_prefers_ecms_over_actor=fr, verdict=verdict)
        res["regions"][nm] = row
        P(f"  {nm:>8}{row['n']:>5}{row['r_at_ecms_load']:>13.4f}{row['r_at_actor_load']:>14.4f}"
          f"{row['r_at_argmaxr']:>12.4f}{row['tce_ecms']:>10.1f}{row['tce_actor']:>11.1f}"
          f"{row['tce_argmaxr']:>13.1f}{verdict:>34}")
    json.dump(res, open(out / f"data/reward_counterfactual_{cycle}.json", "w"), indent=2)
    return res


def actor_alignment_supplement(cycle, P, out):
    P(f"\n{'='*92}\n8A-supp  ACTOR<->argmaxQ DISTANCE + P(mode) by torque band -- {cycle}\n{'='*92}")
    csvp = Path("results/phase8/baseline/data") / f"matched_states_{cycle}.csv"
    rows = list(csv.DictReader(open(csvp)))
    bands = {}
    P(f"  {'region':>8}{'n':>5}{'|a_sac-argmaxQ|/2':>20}{'mode_sac OFF%':>15}{'argmaxQ OFF%':>14}{'ECMS OFF%':>12}")
    for r in rows:
        bands.setdefault(r["region"], []).append(r)
    res = {}
    for reg, rs in bands.items():
        d = [abs(float(x["a_sac"]) - float(x["a_argmaxQ"])) / 2.0 for x in rs]
        f_off = lambda k: 100.0 * np.mean([x[k] == "OFF" for x in rs])
        res[reg] = dict(n=len(rs), actor_argmaxQ_dist_mean=float(np.mean(d)),
                        actor_argmaxQ_dist_median=float(np.median(d)),
                        mode_sac_OFF_pct=f_off("mode_sac"),
                        mode_argmaxQ_OFF_pct=f_off("mode_argmaxQ"),
                        mode_ecms_OFF_pct=f_off("mode_ecms"))
        P(f"  {reg:>8}{len(rs):>5}{np.mean(d):>20.3f}{f_off('mode_sac'):>14.0f}%"
          f"{f_off('mode_argmaxQ'):>13.0f}%{f_off('mode_ecms'):>11.0f}%")
    json.dump(res, open(out / f"data/actor_alignment_{cycle}.json", "w"), indent=2)
    return res


def vehicle_scorecard_3seed(cycle, P, out):
    P(f"\n{'='*92}\n8A-supp  3-SEED VEHICLE SCORECARD (CONTROL actor) -- {cycle}\n{'='*92}")
    scs = []
    for si, d in enumerate(CKPTS[cycle]["control_k2.5_gated"]):
        m = SAC.load(f"{d}/{cycle}/sac_ems_best")
        sc = scorecard(sac_actor_rollout(m, cycle), f"seed{si}")
        scs.append(sc)
        P(f"  seed{si}: V_CE={sc['v_ce']:.4f} V_l={sc['v_liter']:.4f} SoC_f={sc['soc_final']*100:.2f}% "
          f"dSoC={sc['d_soc_pp']:+.2f}pp SoC[min,max]=[{sc['soc_min']:.1f},{sc['soc_max']:.1f}] "
          f"OFF={sc['off_pct']:.1f}% ASST={sc['assist_pct']:.1f}% LPS={sc['lps_pct']:.1f}% "
          f"engON={sc['engine_on_s']}s meanTce_on={sc['mean_eng_tce_when_on']:.1f} viol={sc['violations']} CS={sc['charge_sustaining']}")
    v = np.array([s["v_ce"] for s in scs])
    agg = dict(mean=float(v.mean()), std=float(v.std(ddof=1)), min=float(v.min()), max=float(v.max()),
               ci95=[float(v.mean() - 1.96 * v.std(ddof=1) / np.sqrt(3)),
                     float(v.mean() + 1.96 * v.std(ddof=1) / np.sqrt(3))],
               cs=[s["charge_sustaining"] for s in scs], viol=[s["violations"] for s in scs])
    P(f"  MEAN V_CE = {agg['mean']:.4f} +/- {agg['std']:.4f}  95%CI {agg['ci95']}  CS {agg['cs']}  viol {agg['viol']}")
    json.dump(dict(scorecards=scs, agg=agg), open(out / f"data/baseline_scorecard_{cycle}.json", "w"), indent=2)
    return scs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, choices=["NEDC", "FTP75"])
    ap.add_argument("--out", default="results/phase8")
    a = ap.parse_args()
    out = Path(a.out); (out / "data").mkdir(parents=True, exist_ok=True)
    fh = open(out / f"logs/phase8_reward_state_{a.cycle}.txt", "w", encoding="utf-8")
    P = lambda s: (print(s), fh.write(str(s) + "\n"))
    P(f"PHASE 8 -- reward sufficiency + state/alignment supplement -- {a.cycle}")
    actor_alignment_supplement(a.cycle, P, out)
    vehicle_scorecard_3seed(a.cycle, P, out)
    reward_counterfactual(a.cycle, P, out)
    fh.close()
    print(f"[done] {a.cycle}")


if __name__ == "__main__":
    main()
