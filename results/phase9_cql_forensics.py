"""
phase9_cql_forensics.py
=======================
PHASE 9 sections 6/7/8 -- forensics on the trained CQL(H) conservative-critic
checkpoints.

  §8  3-seed deterministic-eval scorecard of the normal SAC actor trained with
      the conservative critic
  §7  Q-oracle on the repaired critic (Phase-8 Q-oracle definition, 3 seeds)
  §6  did pessimism fix the RIGHT thing? -- re-run the region-min-Q ordering /
      arg-max region / Q1-Q2 disagreement and compare vs CONTROL

    python -m results.phase9_cql_forensics --cycle NEDC --dirs models_p9a_N0,models_p9a_N1,models_p9a_N2
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC

from results.phase9_cql import CQLSAC  # noqa: F401  (registers class for SAC.load)
from results.phase7_forensics import BENCH, ECMS_V, AMAP, KFB_CONTROL, CKPTS, matched_states
from results.phase8_qoracle import (scorecard, sac_actor_rollout, q_oracle_rollout,
                                    ecms_rollout, rule_rollout, regional, TB)
from results.phase9_critic_diag import (twin_q, tce_max_feasible, classify_region,
                                        ReplaySupport, REGION_NAMES, NGRID)
from src.env.ems_env import map_action_to_u, SOC_TARGET
from src.env.powertrain import _Q_BT_0
from src.baselines.ecms import _hamiltonian_best_u
from results.phase7_forensics import LAM0, torques_from_u
import numpy as _np


def load(p):
    try:
        return SAC.load(p)
    except Exception:
        return CQLSAC.load(p)


def region_ordering(models, cycle, P, label):
    S = matched_states(cycle, KFB_CONTROL, AMAP, trajectory="ecms")
    grid = _np.linspace(-1, 1, NGRID).astype(_np.float32)
    rs = ReplaySupport(cycle)
    res = {}
    for lo, hi, nm in [(15, 30, "15-30"), (30, 35, "30-35"), (35, 50, "35-50"), (50, 75, "50-75")]:
        sel = [s for s in S if lo <= s["T"] < hi]
        if len(sel) < 8:
            continue
        sel = sel[:: max(1, len(sel) // 40)][:40]
        acc = {r: [] for r in REGION_NAMES}
        disag = {r: [] for r in REGION_NAMES}
        argmax_reg = {r: 0 for r in REGION_NAMES}
        for st in sel:
            T, w, dw, soc = st["T"], st["w"], st["dw"], st["soc"]
            tce_max = tce_max_feasible(w, dw)
            u_e = _hamiltonian_best_u(w, dw, T, soc, LAM0[cycle] + 8.0 * (SOC_TARGET - soc), 81)
            tce_ecms = torques_from_u(u_e, T, w, dw, soc)[0]
            us = _np.array([map_action_to_u(float(a), T, AMAP, w, dw) for a in grid])
            tces = _np.array([torques_from_u(u, T, w, dw, soc)[0] for u in us])
            regs = [classify_region(tc, tce_ecms, tce_max) for tc in tces]
            q1 = _np.zeros(NGRID); q2 = _np.zeros(NGRID)
            for m in models:
                a1, a2 = twin_q(m, st["obs"], grid); q1 += a1 / len(models); q2 += a2 / len(models)
            mq = _np.minimum(q1, q2); dq = _np.abs(q1 - q2)
            argmax_reg[regs[int(mq.argmax())]] += 1
            for r in REGION_NAMES:
                ri = [i for i, rr in enumerate(regs) if rr == r]
                if ri:
                    acc[r].append(float(mq[ri].mean())); disag[r].append(float(dq[ri].mean()))
        res[nm] = dict(
            minQ={r: (float(_np.mean(acc[r])) if acc[r] else None) for r in REGION_NAMES},
            disagree={r: (float(_np.mean(disag[r])) if disag[r] else None) for r in REGION_NAMES},
            argmax_region={r: argmax_reg[r] / len(sel) for r in REGION_NAMES},
        )
        am = res[nm]["argmax_region"]; mq = res[nm]["minQ"]
        P(f"  [{label}] {nm:>6}Nm  argmaxQ: " +
          " ".join(f"{r}={am[r]*100:.0f}%" for r in REGION_NAMES if am[r] > 0) +
          f"  | minQ OFF={mq['OFF'] and round(mq['OFF'],3)} ECMS={mq['ECMS_NBHD'] and round(mq['ECMS_NBHD'],3)} "
          f"HIGH={mq['HIGH_EFF'] and round(mq['HIGH_EFF'],3)}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, choices=["NEDC", "FTP75"])
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", default="results/phase9")
    a = ap.parse_args()
    out = Path(a.out); (out / "data").mkdir(parents=True, exist_ok=True)
    cyc = a.cycle
    dirs = [d.strip() for d in a.dirs.split(",")]
    fh = open(out / f"logs/phase9_cql_forensics_{cyc}.txt", "w", encoding="utf-8")
    P = lambda s: (print(s), fh.write(str(s) + "\n"))
    P(f"PHASE 9 CQL FORENSICS -- {cyc}  dirs={dirs}")

    models = [load(f"{d}/{cyc}/sac_ems_best") for d in dirs]

    # ---- §8 normal-actor scorecard ----
    P(f"\n{'='*92}\n§8  CQL normal-actor 3-seed scorecard -- {cyc}\n{'='*92}")
    scs = []
    for i, m in enumerate(models):
        sc = scorecard(sac_actor_rollout(m, cyc), f"cql_s{i}")
        scs.append(sc)
        P(f"  s{i}: V_CE={sc['v_ce']:.4f} SoC_f={sc['soc_final']*100:.2f}% dSoC={sc['d_soc_pp']:+.2f}pp "
          f"OFF={sc['off_pct']:.1f}% engON={sc['engine_on_s']}s meanTce_on={sc['mean_eng_tce_when_on']:.1f} "
          f"viol={sc['violations']} CS={sc['charge_sustaining']}")
    v = _np.array([s["v_ce"] for s in scs])
    agg_actor = dict(mean=float(v.mean()), std=float(v.std(ddof=1)), min=float(v.min()), max=float(v.max()),
                     cs=[s["charge_sustaining"] for s in scs], viol=[s["violations"] for s in scs])
    P(f"  CQL ACTOR MEAN V_CE = {agg_actor['mean']:.4f} +/- {agg_actor['std']:.4f}  "
      f"CS {agg_actor['cs']}  | CONTROL {'3.7666' if cyc=='NEDC' else '3.2889'}  RB {BENCH[cyc]}  ECMS {ECMS_V[cyc]}")

    # ---- §7 Q-oracle on the repaired critic ----
    P(f"\n{'='*92}\n§7  Q-oracle on the CQL critic -- {cyc}\n{'='*92}")
    orc = []
    for i, m in enumerate(models):
        sc = scorecard(q_oracle_rollout(m, cyc), f"cql_qoracle_s{i}")
        orc.append(sc)
        P(f"  s{i}: V_CE={sc['v_ce']:.4f} dSoC={sc['d_soc_pp']:+.2f}pp OFF={sc['off_pct']:.1f}% "
          f"meanTce_on={sc['mean_eng_tce_when_on']:.1f} viol={sc['violations']} CS={sc['charge_sustaining']}")
    vo = _np.array([s["v_ce"] for s in orc])
    agg_orc = dict(mean=float(vo.mean()), std=float(vo.std(ddof=1)), min=float(vo.min()), max=float(vo.max()),
                   cs=[s["charge_sustaining"] for s in orc], dsoc=[round(s["d_soc_pp"], 2) for s in orc])
    P(f"  CQL Q-ORACLE MEAN V_CE = {agg_orc['mean']:.4f} +/- {agg_orc['std']:.4f}  CS {agg_orc['cs']}")
    P(f"  vs Phase-8 Q-oracle {'3.9404 (1/3 CS)' if cyc=='NEDC' else '3.3545 (0/3 CS)'}")

    # ---- §6 region ordering: CQL vs CONTROL ----
    P(f"\n{'='*92}\n§6  region min-Q ordering + argmax region: CONTROL vs CQL -- {cyc}\n{'='*92}")
    ctrl = [SAC.load(f"{d}/{cyc}/sac_ems_best") for d in CKPTS[cyc]["control_k2.5_gated"]]
    ord_ctrl = region_ordering(ctrl, cyc, P, "CONTROL")
    ord_cql = region_ordering(models, cyc, P, "CQL    ")

    # ---- §11 re-test physical decomposition for best CQL seed ----
    bi = int(_np.argmin(v))
    P(f"\n  best CQL seed = {bi} (V_CE {v[bi]:.4f})")

    verdict = dict(
        cycle=cyc, cql_alpha=1.0,
        control_v_ce=3.7666 if cyc == "NEDC" else 3.2889,
        cql_actor=agg_actor, cql_qoracle=agg_orc,
        phase8_qoracle_v_ce=3.9404 if cyc == "NEDC" else 3.3545,
        phase8_qoracle_cs="1/3" if cyc == "NEDC" else "0/3",
        rule_based=BENCH[cyc], ecms=ECMS_V[cyc],
        gap_closed_vs_control=((3.7666 if cyc == "NEDC" else 3.2889) - agg_actor["mean"]) /
                              ((3.7666 if cyc == "NEDC" else 3.2889) - ECMS_V[cyc]),
        qoracle_became_CS=bool(sum(agg_orc["cs"]) >= 2),
        qoracle_improved_over_phase8=bool(agg_orc["mean"] < (3.9404 if cyc == "NEDC" else 3.3545)),
        actor_beats_RB=bool(agg_actor["mean"] < BENCH[cyc]),
        region_ordering_control=ord_ctrl, region_ordering_cql=ord_cql,
    )
    P(f"\n{'='*92}")
    P(f"  gap closed vs CONTROL: {verdict['gap_closed_vs_control']*100:+.1f}%")
    P(f"  Q-oracle became CS (>=2/3): {verdict['qoracle_became_CS']}   improved over Phase-8: {verdict['qoracle_improved_over_phase8']}")
    P(f"  CQL actor beats rule-based: {verdict['actor_beats_RB']}")
    json.dump(verdict, open(out / f"data/cql_forensics_{cyc}.json", "w"), indent=2)
    fh.close()


if __name__ == "__main__":
    main()
