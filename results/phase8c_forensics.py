"""
phase8c_forensics.py
====================
PHASE 8C/8D/8F -- forensics on the trained mixture-actor checkpoints.

  - 3-seed vehicle scorecard (V_CE, dSoC, OFF/ASSIST/LPS, engine Tce, violations, CS)
  - matched-state (ECMS trajectory) actor<->argmaxQ alignment, P(OFF) by band,
    BEFORE (CONTROL) vs AFTER (mixture)
  - SAC-vs-ECMS gap decomposition for the best mixture seed
  - mixture-specific: component weights / means / separation by torque band

    python -m results.phase8c_forensics --cycle NEDC --dirs models_p8c_N0,models_p8c_N1,models_p8c_N2
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch as th
from stable_baselines3 import SAC

from src.env.ems_env import map_action_to_u
from results.phase8_mixture_policy import MixtureSACPolicy, mixture_readout  # noqa: F401  (import registers class for load)
from results.phase7_forensics import (
    EQF, LAM0, BENCH, ECMS_V, AMAP, KFB_CONTROL, CKPTS,
    q_at, actor_at, mode_of_a, matched_states,
)
from results.phase8_qoracle import scorecard, sac_actor_rollout, ecms_rollout, rule_rollout, regional, TB


def load_any(path):
    """Load a checkpoint whether it is a plain SAC or a MixtureSACPolicy SAC."""
    return SAC.load(path)


def alignment_by_band(model, cycle, P, tag):
    S = matched_states(cycle, KFB_CONTROL, AMAP, trajectory="ecms")
    grid = np.linspace(-1, 1, 81).astype(np.float32)
    out = {}
    for lo, hi, nm in [(15, 30, "15-30"), (30, 35, "30-35"), (35, 50, "35-50"), (50, 75, "50-75")]:
        sel = [s for s in S if lo <= s["T"] < hi]
        if len(sel) < 8:
            continue
        sel = sel[:: max(1, len(sel) // 90)][:90]
        dist, off_sac, off_q, off_ecms = [], [], [], []
        for st in sel:
            Q = q_at(model, st["obs"], grid)
            a_q = float(grid[int(Q.argmax())])
            mu, sd = actor_at(model, st["obs"])
            a_sac = float(np.tanh(mu))
            dist.append(abs(a_sac - a_q) / 2.0)
            off_sac.append(mode_of_a(a_sac, st["T"], st["w"], st["dw"], st["soc"]) == "OFF")
            off_q.append(mode_of_a(a_q, st["T"], st["w"], st["dw"], st["soc"]) == "OFF")
            u_e = None
        out[nm] = dict(n=len(sel), actor_argmaxQ_dist=float(np.mean(dist)),
                       actor_OFF_pct=float(100 * np.mean(off_sac)),
                       argmaxQ_OFF_pct=float(100 * np.mean(off_q)))
        P(f"    {tag} {nm:>6}Nm  n={out[nm]['n']:>3}  |a_sac-argmaxQ|/2={out[nm]['actor_argmaxQ_dist']:.3f}  "
          f"actor OFF%={out[nm]['actor_OFF_pct']:.0f}  argmaxQ OFF%={out[nm]['argmaxQ_OFF_pct']:.0f}")
    return out


def mixture_structure(model, cycle, P):
    S = matched_states(cycle, KFB_CONTROL, AMAP, trajectory="ecms")
    out = {}
    for lo, hi, nm in TB:
        sel = [s for s in S if lo <= s["T"] < hi]
        if len(sel) < 8:
            continue
        sel = sel[:: max(1, len(sel) // 60)][:60]
        seps, wmax, k_off = [], [], []
        for st in sel:
            comps = mixture_readout(model, st["obs"])
            comps = sorted(comps, key=lambda c: -c["weight"])
            seps.append(abs(comps[0]["mean"] - comps[1]["mean"]))
            wmax.append(comps[0]["weight"])
            # does either component sit in the OFF region?
            a_off = 2.0 * 0.60 - 1.0
            k_off.append(any(c["mean"] >= a_off for c in comps))
        out[nm] = dict(n=len(sel), mean_component_separation=float(np.mean(seps)),
                       mean_dominant_weight=float(np.mean(wmax)),
                       frac_states_a_component_in_OFF=float(np.mean(k_off)))
        P(f"    {nm:>6}Nm  comp-sep={out[nm]['mean_component_separation']:.2f}  "
          f"dom-w={out[nm]['mean_dominant_weight']:.2f}  a-comp-in-OFF={out[nm]['frac_states_a_component_in_OFF']*100:.0f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, choices=["NEDC", "FTP75"])
    ap.add_argument("--dirs", required=True, help="comma list of mixture model dirs")
    ap.add_argument("--out", default="results/phase8")
    a = ap.parse_args()
    out = Path(a.out); (out / "data").mkdir(parents=True, exist_ok=True)
    cyc = a.cycle
    dirs = [d.strip() for d in a.dirs.split(",")]
    fh = open(out / f"logs/phase8c_forensics_{cyc}.txt", "w", encoding="utf-8")
    P = lambda s: (print(s), fh.write(str(s) + "\n"))
    P(f"PHASE 8C/8D/8F FORENSICS -- mixture actor -- {cyc}")

    # ---- 8D: 3-seed scorecard ----
    P(f"\n{'='*92}\n8D  3-SEED VEHICLE SCORECARD -- mixture actor -- {cyc}\n{'='*92}")
    scs = []
    for si, d in enumerate(dirs):
        m = load_any(f"{d}/{cyc}/sac_ems_best")
        sc = scorecard(sac_actor_rollout(m, cyc), f"mix_seed{si}")
        scs.append(sc)
        P(f"  seed{si}: V_CE={sc['v_ce']:.4f} SoC_f={sc['soc_final']*100:.2f}% dSoC={sc['d_soc_pp']:+.2f}pp "
          f"SoC[min,max]=[{sc['soc_min']:.1f},{sc['soc_max']:.1f}] OFF={sc['off_pct']:.1f}% ASST={sc['assist_pct']:.1f}% "
          f"LPS={sc['lps_pct']:.1f}% engON={sc['engine_on_s']}s meanTce_on={sc['mean_eng_tce_when_on']:.1f} "
          f"viol={sc['violations']} CS={sc['charge_sustaining']}")
    v = np.array([s["v_ce"] for s in scs])
    agg = dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
               min=float(v.min()), max=float(v.max()),
               ci95=[float(v.mean() - 1.96 * v.std(ddof=1) / np.sqrt(len(v))),
                     float(v.mean() + 1.96 * v.std(ddof=1) / np.sqrt(len(v)))] if len(v) > 1 else [float(v.mean())] * 2,
               cs=[s["charge_sustaining"] for s in scs], viol=[s["violations"] for s in scs])
    P(f"  MIXTURE MEAN V_CE = {agg['mean']:.4f} +/- {agg['std']:.4f}  95%CI {agg['ci95']}  "
      f"CS {agg['cs']}  viol {agg['viol']}")
    P(f"  CONTROL reference   = {ECMS_V[cyc]:.4f} ECMS | {BENCH[cyc]:.4f} rule-based | "
      f"{'3.7666' if cyc=='NEDC' else '3.2889'} CONTROL")

    # ---- 8C: alignment before/after ----
    P(f"\n{'='*92}\n8C  ACTOR<->argmaxQ ALIGNMENT  BEFORE (CONTROL) vs AFTER (mixture) -- {cyc}\n{'='*92}")
    ctrl = load_any(f"{CKPTS[cyc]['control_k2.5_gated'][0]}/{cyc}/sac_ems_best")
    P("  BEFORE (CONTROL unimodal):")
    a_before = alignment_by_band(ctrl, cyc, P, "  before")
    best_i = int(np.argmin(v))
    mbest = load_any(f"{dirs[best_i]}/{cyc}/sac_ems_best")
    P(f"  AFTER (mixture, best seed {best_i} V_CE={v[best_i]:.4f}):")
    a_after = alignment_by_band(mbest, cyc, P, "  after ")

    P(f"\n  mixture component structure (best seed) by torque band:")
    mix_struct = mixture_structure(mbest, cyc, P)

    # ---- 8F: SAC-vs-ECMS gap decomposition for best mixture seed ----
    P(f"\n{'='*92}\n8F  SAC(mixture best) vs ECMS gap decomposition -- {cyc}\n{'='*92}")
    RA = sac_actor_rollout(mbest, cyc)
    RE = ecms_rollout(cyc)
    RR = rule_rollout(cyc)
    n = min(len(RA), len(RE)); RA, RE = RA[:n], RE[:n]
    K = 1e5 / RE[-1]["final"]["x_tot_m"]
    reg = regional(RA, RA, RE, K)  # A==B here; use dfuel_sac_ecms column
    P(f"  {'band':>7}{'dFuel(mix-ECMS)':>17}{'OFF% mix/ECMS':>16}{'engTce mix/ECMS':>17}")
    gapF = 0.0
    for nm, r in reg.items():
        gapF += r["dfuel_sac_ecms"]
        P(f"  {nm:>7}{r['dfuel_sac_ecms']:>+17.4f}{r['off_sac']:>9.0f}/{r['off_ecms']:>4.0f}"
          f"{r['engt_sac']:>11.0f}/{r['engt_ecms']:>4.0f}")
    P(f"  total dFuel(mix-ECMS) approx {gapF:+.4f}  |  V_CE(mix best) {v[best_i]:.4f} vs ECMS {ECMS_V[cyc]:.4f} "
      f"= {v[best_i]-ECMS_V[cyc]:+.4f}")

    json.dump(dict(cycle=cyc, dirs=dirs, scorecards=scs, agg=agg,
                   alignment_before=a_before, alignment_after=a_after,
                   mixture_structure=mix_struct,
                   gap_decomp_best={nm: reg[nm] for nm in reg}),
              open(out / f"data/phase8c_forensics_{cyc}.json", "w"), indent=2)
    fh.close()
    print(f"[done] {cyc} -> {out}/data/phase8c_forensics_{cyc}.json")


if __name__ == "__main__":
    main()
