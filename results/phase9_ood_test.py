"""
phase9_ood_test.py
==================
PHASE 9 -- the decisive test for the REFINED H9-A.

Phase-9 §3/§4 (critic_error_map) showed the critic ranks HIGH_EFF >= ECMS_NBHD
>= LOW >= OFF ON the ECMS-trajectory (good-SoC) states -- i.e. NO clean OFF
overvaluation or high-load undervaluation there. Yet the Phase-8 Q-oracle
collapses SoC. Hypothesis: the value error is a DISTRIBUTION-SHIFT error -- the
critic is only reliable near the CONTROL actor's visited states; greedy
exploitation drives the state off that manifold where Q is unconstrained.

Test: run the region-Q analysis at TWO state distributions and compare:
  (a) ECMS-trajectory states  (in-distribution, good SoC)   [= §3/§4]
  (b) the Q-oracle's OWN visited traction states            (where it actually goes)

If OFF becomes the argmax-Q region at (b) but not (a), and replay support at
(b) is far lower -> distribution shift confirmed.

    python -m results.phase9_ood_test --cycle NEDC
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import torch as th
from stable_baselines3 import SAC

from src.env.ems_env import EMSEnv, map_action_to_u, SOC_TARGET
from src.env.powertrain import _T_CUTOFF, _Q_BT_0
from src.baselines.ecms import _hamiltonian_best_u
from results.phase7_forensics import (EQF, LAM0, AMAP, KFB_CONTROL, CKPTS,
                                      torques_from_u, actor_at, matched_states)
from results.phase9_critic_diag import (twin_q, tce_max_feasible, classify_region,
                                        ReplaySupport, REGION_NAMES, TB, NGRID)
from results.phase8_qoracle import q_oracle_rollout


def qoracle_states(model, cycle, kfb=KFB_CONTROL):
    """Snapshot (obs, clean-env) at every traction step of the Q-oracle rollout."""
    grid = np.linspace(-1, 1, NGRID).astype(np.float32)
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=kfb, lookahead=5, action_map=AMAP)
    obs, _ = env.reset()
    S = []
    while True:
        d = env._demand
        T, w, dw = d["T_MGB"], d["w_MGB"], d["dw_MGB"]
        soc = env._Q_BT / _Q_BT_0
        if T > _T_CUTOFF and w > 0:
            S.append(dict(obs=obs.copy(), env=copy.deepcopy(env), T=T, w=w, dw=dw, soc=soc))
        q1, q2 = twin_q(model, obs, grid)
        a = float(grid[int(np.argmin(np.stack([q1, q2]), axis=0).argmax()
                           if False else np.minimum(q1, q2).argmax())])
        obs, r, t, _, i = env.step(np.array([a], np.float32))
        if t:
            return S


def region_analysis(models, states, rs, cycle, label, P):
    grid = np.linspace(-1, 1, NGRID).astype(np.float32)
    out = {}
    for lo, hi, nm in TB:
        sel = [s for s in states if lo <= s["T"] < hi]
        if len(sel) < 8:
            continue
        sel = sel[:: max(1, len(sel) // 40)][:40]
        acc = {r: dict(minq=[], supp=[], nsoc=[]) for r in REGION_NAMES}
        argmax_region = {r: 0 for r in REGION_NAMES}
        soc_here = []
        for st in sel:
            T, w, dw, soc = st["T"], st["w"], st["dw"], st["soc"]
            soc_here.append(soc)
            tce_max = tce_max_feasible(w, dw)
            u_e = _hamiltonian_best_u(w, dw, T, soc, LAM0[cycle] + 8.0 * (SOC_TARGET - soc), 81)
            tce_ecms = torques_from_u(u_e, T, w, dw, soc)[0]
            us = np.array([map_action_to_u(float(a), T, AMAP, w, dw) for a in grid])
            tces = np.array([torques_from_u(u, T, w, dw, soc)[0] for u in us])
            regs = [classify_region(tc, tce_ecms, tce_max) for tc in tces]
            q1s = np.zeros(NGRID); q2s = np.zeros(NGRID)
            for m in models:
                a1, a2 = twin_q(m, st["obs"], grid)
                q1s += a1 / len(models); q2s += a2 / len(models)
            minqs = np.minimum(q1s, q2s)
            argmax_region[regs[int(minqs.argmax())]] += 1
            supp, supp_soc = rs.region_support(st["obs"], T, w, dw, soc, tce_ecms, tce_max)
            for r in REGION_NAMES:
                ri = [i for i, rr in enumerate(regs) if rr == r]
                if not ri:
                    continue
                acc[r]["minq"].append(float(minqs[ri].mean()))
                acc[r]["supp"].append(supp[r])
        band = dict(n=len(sel), soc_median=float(np.median(soc_here)),
                    argmax_region={r: argmax_region[r] / len(sel) for r in REGION_NAMES})
        for r in REGION_NAMES:
            if acc[r]["minq"]:
                band[r] = dict(minQ=float(np.mean(acc[r]["minq"])),
                               support=float(np.mean(acc[r]["supp"])))
        out[nm] = band
        am = band["argmax_region"]
        P(f"  [{label}] {nm:>6}Nm n={band['n']:>2} SoC_med={band['soc_median']*100:4.1f}%  "
          f"argmaxQ region: " + " ".join(f"{r}={am[r]*100:.0f}%" for r in REGION_NAMES if am[r] > 0)
          + f"   | minQ OFF={band.get('OFF',{}).get('minQ',float('nan')):+.3f} "
          f"ECMS={band.get('ECMS_NBHD',{}).get('minQ',float('nan')):+.3f} "
          f"HIGH={band.get('HIGH_EFF',{}).get('minQ',float('nan')):+.3f}"
          f"   | supp OFF={band.get('OFF',{}).get('support',0)*100:.0f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, choices=["NEDC", "FTP75"])
    ap.add_argument("--out", default="results/phase9")
    a = ap.parse_args()
    out = Path(a.out); (out / "data").mkdir(parents=True, exist_ok=True)
    fh = open(out / f"logs/phase9_ood_test_{a.cycle}.txt", "w", encoding="utf-8")
    P = lambda s: (print(s), fh.write(str(s) + "\n"))
    P(f"PHASE 9 OOD TEST -- {a.cycle}  (critic reliability: ECMS-traj states vs Q-oracle's own states)")

    ck = CKPTS[a.cycle]["control_k2.5_gated"]
    models = [SAC.load(f"{d}/{a.cycle}/sac_ems_best") for d in ck]
    rs = ReplaySupport(a.cycle)

    S_ecms = matched_states(a.cycle, KFB_CONTROL, AMAP, trajectory="ecms")
    S_orc = qoracle_states(models[0], a.cycle)
    P(f"\n  ECMS-trajectory traction states: {len(S_ecms)}   Q-oracle traction states: {len(S_orc)}")
    P(f"  ECMS-traj SoC range: [{min(s['soc'] for s in S_ecms)*100:.1f}, {max(s['soc'] for s in S_ecms)*100:.1f}]%")
    P(f"  Q-oracle  SoC range: [{min(s['soc'] for s in S_orc)*100:.1f}, {max(s['soc'] for s in S_orc)*100:.1f}]%")

    P(f"\n--- region-Q analysis @ ECMS-trajectory states (in-distribution) ---")
    a_ecms = region_analysis(models, S_ecms, rs, a.cycle, "ECMS", P)
    P(f"\n--- region-Q analysis @ Q-oracle's OWN visited states ---")
    a_orc = region_analysis(models, S_orc, rs, a.cycle, "ORACLE", P)

    # verdict
    def off_argmax_frac(A):
        v = [b["argmax_region"]["OFF"] for b in A.values() if b["n"]]
        return float(np.mean(v)) if v else 0.0
    def mean_off_support(A):
        v = [b.get("OFF", {}).get("support", 0.0) for b in A.values() if b["n"]]
        return float(np.mean(v)) if v else 0.0
    verdict = dict(
        cycle=a.cycle,
        off_argmaxQ_frac_ECMS_states=off_argmax_frac(a_ecms),
        off_argmaxQ_frac_ORACLE_states=off_argmax_frac(a_orc),
        mean_OFF_replay_support_ECMS_states=mean_off_support(a_ecms),
        mean_OFF_replay_support_ORACLE_states=mean_off_support(a_orc),
        distribution_shift_confirmed=bool(
            off_argmax_frac(a_orc) > off_argmax_frac(a_ecms) + 0.15),
    )
    P(f"\n{'='*90}")
    P(f"  OFF is the argmax-Q region in: {verdict['off_argmaxQ_frac_ECMS_states']*100:.0f}% of ECMS-traj states "
      f"vs {verdict['off_argmaxQ_frac_ORACLE_states']*100:.0f}% of Q-oracle's own states")
    P(f"  mean OFF replay support: {verdict['mean_OFF_replay_support_ECMS_states']*100:.0f}% (ECMS) "
      f"vs {verdict['mean_OFF_replay_support_ORACLE_states']*100:.0f}% (oracle)")
    P(f"  DISTRIBUTION-SHIFT (refined H9-A) confirmed: {verdict['distribution_shift_confirmed']}")
    json.dump(dict(verdict=verdict, ecms_states=a_ecms, oracle_states=a_orc),
              open(out / f"data/ood_test_{a.cycle}.json", "w"), indent=2)
    fh.close()


if __name__ == "__main__":
    main()
