"""
phase9_critic_diag.py
=====================
PHASE 9 sections 3 + 4 -- map the critic value-fidelity error.

For matched states (Phase-7/8 ECMS-trajectory methodology), classify the dense
action grid into 5 engine-load regions and, per region, report:
  replay support (density of nearby (obs,action) pairs in the CONTROL buffer),
  Q1, Q2, min(Q1,Q2), |Q1-Q2| disagreement, immediate reward, predicted next SoC.

Then classify the critic error:
  Type 1  OFF / sustained-discharge OVERVALUATION
  Type 2  high-efficiency engine-load UNDERVALUATION
  Type 3  both.

NO training. NO physics/reward/critic/actor/env change. Critic used exactly as
trained (CONTROL, gated k_fb=2.5, 3 seeds).

    python -m results.phase9_critic_diag --cycle NEDC
    python -m results.phase9_critic_diag --cycle FTP75
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import torch as th
from stable_baselines3 import SAC

from src.env.ems_env import EMSEnv, map_action_to_u, SOC_TARGET
from src.env.powertrain import (_T_CUTOFF, _Q_BT_0, combustion_engine,
                                _interp1d_linear, _w_CE_max_fine, _T_CE_max, _THETA)
from src.baselines.ecms import _hamiltonian_best_u
from results.phase7_forensics import (EQF, LAM0, BENCH, ECMS_V, AMAP, KFB_CONTROL,
                                      CKPTS, actor_at, torques_from_u, mode_of_u,
                                      matched_states, reward_of_action)

TB = [(0, 15, "0-15"), (15, 30, "15-30"), (30, 35, "30-35"),
      (35, 50, "35-50"), (50, 75, "50-75"), (75, 1e9, ">75")]
REGION_NAMES = ["OFF", "LOW", "ECMS_NBHD", "HIGH_EFF", "MAX"]
NGRID = 161


# --------------------------------------------------------------------------- #
def twin_q(model, ob, acts):
    ot = th.as_tensor(np.repeat(ob.reshape(1, -1), len(acts), 0)).float().to(model.device)
    at = th.as_tensor(np.asarray(acts, np.float32).reshape(-1, 1)).float().to(model.device)
    with th.no_grad():
        qs = model.critic(ot, at)
    q1 = qs[0].cpu().numpy().ravel()
    q2 = qs[1].cpu().numpy().ravel()
    return q1, q2


def tce_max_feasible(w, dw):
    w_ce = max(w, 105.0)
    return _interp1d_linear(_w_CE_max_fine, _T_CE_max, w_ce) - abs(_THETA * dw) - 0.01


def classify_region(t_ce, tce_ecms, tce_max):
    """5-region engine-load classifier, state-conditioned on the ECMS load."""
    if t_ce <= _T_CUTOFF:
        return "OFF"
    if tce_ecms <= _T_CUTOFF:
        # ECMS is OFF here -> everything above cutoff is 'above ECMS'
        if t_ce <= 0.5 * tce_max:
            return "LOW"
        return "HIGH_EFF" if t_ce <= 0.9 * tce_max else "MAX"
    lo, hi = 0.75 * tce_ecms, 1.30 * tce_ecms
    if t_ce < lo:
        return "LOW"
    if t_ce <= hi:
        return "ECMS_NBHD"
    if t_ce <= 0.9 * tce_max:
        return "HIGH_EFF"
    return "MAX"


def engine_metrics(w, dw, t_ce):
    """Instantaneous engine efficiency + BSFC at this operating point."""
    if t_ce <= _T_CUTOFF:
        return dict(bsfc_g_per_kWh=None, eff=0.0, fuel_rate_g_s=0.0, w_ce_rpm=w * 60 / (2 * np.pi))
    eng = combustion_engine(w_gear=w, dw_gear=dw, t_gear=t_ce)
    mech_w = eng["t_ce"] * eng["w_ce"]                       # W
    fuel_w = max(eng["p_ce_fuel"], 1e-9)
    eff = mech_w / fuel_w if mech_w > 0 else 0.0
    bsfc = (eng["v_dot"] / mech_w) * 3.6e9 if mech_w > 0 else None   # g/kWh
    return dict(bsfc_g_per_kWh=bsfc, eff=float(eff), fuel_rate_g_s=eng["v_dot"] * 1000.0,
                w_ce_rpm=eng["w_ce"] * 60 / (2 * np.pi))


# --------------------------------------------------------------------------- #
class ReplaySupport:
    """density of nearby (obs, action) transitions in the CONTROL buffer."""

    def __init__(self, cycle, n_sample=40000):
        d0 = CKPTS[cycle]["control_k2.5_gated"][0]
        m = SAC.load(f"{d0}/{cycle}/sac_ems_best")
        m.load_replay_buffer(f"{d0}/{cycle}/replay_buffer.pkl")
        rb = m.replay_buffer
        n = rb.size()
        idx = np.random.default_rng(0).choice(n, size=min(n_sample, n), replace=False)
        self.obs = rb.observations[idx, 0, :].astype(np.float32)          # [N,20]
        self.act = rb.actions[idx, 0, 0].astype(np.float32)               # [N]
        nobs = rb.next_observations[idx, 0, :].astype(np.float32)
        self.next_soc = (nobs[:, 4] + 1.0) / 2.0                          # SoC channel
        self.soc = (self.obs[:, 4] + 1.0) / 2.0
        # per-channel scale for distance (std over the sample, guard zeros)
        self.scale = self.obs.std(axis=0) + 1e-6

    def neighbors(self, ob, k=400):
        d = np.sqrt((((self.obs - ob.reshape(1, -1)) / self.scale) ** 2).sum(axis=1))
        nn = np.argpartition(d, k)[:k]
        return nn

    def region_support(self, ob, T, w, dw, soc, tce_ecms, tce_max, k=400):
        nn = self.neighbors(ob, k)
        acts = self.act[nn]
        counts = {r: 0 for r in REGION_NAMES}
        soc_by_region = {r: [] for r in REGION_NAMES}
        for a, i in zip(acts, nn):
            u = map_action_to_u(float(a), T, AMAP, w, dw)
            t_ce, _ = torques_from_u(u, T, w, dw, soc)
            reg = classify_region(t_ce, tce_ecms, tce_max)
            counts[reg] += 1
            soc_by_region[reg].append(float(self.next_soc[i]))
        tot = max(sum(counts.values()), 1)
        return ({r: counts[r] / tot for r in REGION_NAMES},
                {r: (float(np.mean(soc_by_region[r])) if soc_by_region[r] else None)
                 for r in REGION_NAMES})


# --------------------------------------------------------------------------- #
def run(cycle, P, out):
    P(f"\n{'='*100}\nPHASE 9 §3/§4  CRITIC ERROR MAP -- {cycle}\n{'='*100}")
    ck_dirs = CKPTS[cycle]["control_k2.5_gated"]
    models = [SAC.load(f"{d}/{cycle}/sac_ems_best") for d in ck_dirs]
    rs = ReplaySupport(cycle)
    S = matched_states(cycle, KFB_CONTROL, AMAP, trajectory="ecms")
    grid = np.linspace(-1, 1, NGRID).astype(np.float32)

    raw_rows = []
    band_agg = {}
    for lo, hi, nm in TB:
        sel = [s for s in S if lo <= s["T"] < hi]
        if len(sel) < 8:
            continue
        sel = sel[:: max(1, len(sel) // 50)][:50]
        # accumulators per region
        acc = {r: dict(minq=[], q1=[], q2=[], disag=[], rew=[], nsoc=[], support=[],
                       nsoc_buf=[], bsfc=[], eff=[]) for r in REGION_NAMES}
        for st in sel:
            T, w, dw, soc = st["T"], st["w"], st["dw"], st["soc"]
            tce_max = tce_max_feasible(w, dw)
            u_e = _hamiltonian_best_u(w, dw, T, soc, LAM0[cycle] + 8.0 * (SOC_TARGET - soc), 81)
            tce_ecms = torques_from_u(u_e, T, w, dw, soc)[0]
            # dense grid physical quantities
            us = np.array([map_action_to_u(float(a), T, AMAP, w, dw) for a in grid])
            tces = np.array([torques_from_u(u, T, w, dw, soc)[0] for u in us])
            regs = [classify_region(tc, tce_ecms, tce_max) for tc in tces]
            # twin-Q for the whole grid, averaged over the 3 critics
            q1s = np.zeros(NGRID); q2s = np.zeros(NGRID)
            for m in models:
                a1, a2 = twin_q(m, st["obs"], grid)
                q1s += a1 / len(models); q2s += a2 / len(models)
            minqs = np.minimum(q1s, q2s)
            disag = np.abs(q1s - q2s)
            # per-region replay support + buffer next-SoC
            supp, supp_soc = rs.region_support(st["obs"], T, w, dw, soc, tce_ecms, tce_max)
            for r in REGION_NAMES:
                ri = [i for i, rr in enumerate(regs) if rr == r]
                if not ri:
                    continue
                # representative action = grid point closest to region-centre torque
                mid = ri[len(ri) // 2]
                acc[r]["minq"].append(float(minqs[ri].mean()))
                acc[r]["q1"].append(float(q1s[ri].mean()))
                acc[r]["q2"].append(float(q2s[ri].mean()))
                acc[r]["disag"].append(float(disag[ri].mean()))
                rew = reward_of_action(st["env"], float(grid[mid]))
                acc[r]["rew"].append(rew)
                e = copy.deepcopy(st["env"]); _, _, _, _, ii = e.step(np.array([grid[mid]], np.float32))
                acc[r]["nsoc"].append(float(ii["soc"]))
                acc[r]["support"].append(supp[r])
                if supp_soc[r] is not None:
                    acc[r]["nsoc_buf"].append(supp_soc[r])
                em = engine_metrics(w, dw, float(tces[mid]))
                if em["bsfc_g_per_kWh"] is not None:
                    acc[r]["bsfc"].append(em["bsfc_g_per_kWh"]); acc[r]["eff"].append(em["eff"])
            raw_rows.append(dict(cycle=cycle, band=nm, SoC=soc, T=T, w=w,
                                 tce_ecms=float(tce_ecms), tce_max=float(tce_max),
                                 actor_a=float(np.tanh(actor_at(models[0], st["obs"])[0]))))
        band = {}
        for r in REGION_NAMES:
            a = acc[r]
            if not a["minq"]:
                band[r] = dict(n=0)
                continue
            band[r] = dict(
                n=len(a["minq"]),
                minQ=float(np.mean(a["minq"])), Q1=float(np.mean(a["q1"])), Q2=float(np.mean(a["q2"])),
                Q_disagree=float(np.mean(a["disag"])),
                reward=float(np.mean(a["rew"])),
                next_soc_env=float(np.mean(a["nsoc"])),
                next_soc_buffer=float(np.mean(a["nsoc_buf"])) if a["nsoc_buf"] else None,
                replay_support=float(np.mean(a["support"])),
                bsfc_g_per_kWh=float(np.mean(a["bsfc"])) if a["bsfc"] else None,
                engine_eff=float(np.mean(a["eff"])) if a["eff"] else None,
            )
        band_agg[nm] = band
        # print
        P(f"\n  --- {nm} Nm  (n_states={len(sel)}) ---")
        P(f"    {'region':>10}{'supp%':>8}{'minQ':>10}{'|Q1-Q2|':>10}{'reward':>10}"
          f"{'nSoC_env':>10}{'nSoC_buf':>10}{'BSFC':>9}{'eff':>7}")
        for r in REGION_NAMES:
            b = band[r]
            if b.get("n", 0) == 0:
                P(f"    {r:>10}   (no grid actions in region)")
                continue
            P(f"    {r:>10}{b['replay_support']*100:>7.1f}%{b['minQ']:>10.4f}{b['Q_disagree']:>10.4f}"
              f"{b['reward']:>10.4f}{b['next_soc_env']*100:>9.1f}%"
              f"{(b['next_soc_buffer']*100 if b['next_soc_buffer'] is not None else float('nan')):>9.1f}%"
              f"{(b['bsfc_g_per_kWh'] if b['bsfc_g_per_kWh'] is not None else float('nan')):>9.0f}"
              f"{(b['engine_eff'] if b['engine_eff'] is not None else float('nan')):>7.2f}")

    # -------- Type classification (focus bands 15-35, both cycles) --------
    P(f"\n{'='*100}\n  CRITIC ERROR TYPE CLASSIFICATION -- {cycle}\n{'='*100}")
    verdict = {}
    for nm in ("15-30", "30-35", "35-50", "50-75"):
        if nm not in band_agg:
            continue
        b = band_agg[nm]
        off, ecmsn, hieff = b.get("OFF", {}), b.get("ECMS_NBHD", {}), b.get("HIGH_EFF", {})
        low = b.get("LOW", {})
        t1 = t2 = None
        if off.get("n") and ecmsn.get("n"):
            # Type 1: minQ(OFF) >= minQ(ECMS) while OFF has worse SoC consequence and thin support
            t1 = dict(
                minQ_OFF_minus_ECMS=off["minQ"] - ecmsn["minQ"],
                nSoC_OFF_minus_ECMS_env=off["next_soc_env"] - ecmsn["next_soc_env"],
                support_OFF=off["replay_support"], support_ECMS=ecmsn["replay_support"],
                OFF_overvalued=bool(off["minQ"] >= ecmsn["minQ"] - 1e-4
                                    and off["next_soc_env"] < ecmsn["next_soc_env"] - 0.002),
            )
        ref = ecmsn if ecmsn.get("n") else low
        if hieff.get("n") and ref.get("n"):
            # Type 2: minQ(HIGH_EFF) < minQ(ref) despite reward(HIGH_EFF) >= reward(ref)
            t2 = dict(
                minQ_HIGH_minus_ref=hieff["minQ"] - ref["minQ"],
                reward_HIGH_minus_ref=hieff["reward"] - ref["reward"],
                bsfc_HIGH=hieff.get("bsfc_g_per_kWh"), bsfc_ref=ref.get("bsfc_g_per_kWh"),
                support_HIGH=hieff["replay_support"],
                HIGH_undervalued=bool(hieff["minQ"] < ref["minQ"] - 1e-4
                                      and hieff["reward"] >= ref["reward"] - 1e-3),
            )
        typ = ("Type 3 (both)" if (t1 and t1["OFF_overvalued"] and t2 and t2["HIGH_undervalued"])
               else "Type 1 (OFF overvaluation)" if (t1 and t1["OFF_overvalued"])
               else "Type 2 (high-load undervaluation)" if (t2 and t2["HIGH_undervalued"])
               else "neither cleanly triggered")
        verdict[nm] = dict(type=typ, type1=t1, type2=t2)
        P(f"  {nm} Nm : {typ}")
        if t1:
            P(f"      T1: minQ(OFF)-minQ(ECMS)={t1['minQ_OFF_minus_ECMS']:+.4f}  "
              f"nSoC(OFF)-nSoC(ECMS)={t1['nSoC_OFF_minus_ECMS_env']*100:+.2f}pp  "
              f"support OFF/ECMS={t1['support_OFF']*100:.0f}%/{t1['support_ECMS']*100:.0f}%  "
              f"-> OFF overvalued: {t1['OFF_overvalued']}")
        if t2:
            P(f"      T2: minQ(HIGH)-minQ(ref)={t2['minQ_HIGH_minus_ref']:+.4f}  "
              f"reward(HIGH)-reward(ref)={t2['reward_HIGH_minus_ref']:+.4f}  "
              f"BSFC HIGH/ref={t2['bsfc_HIGH'] and round(t2['bsfc_HIGH'])}/{t2['bsfc_ref'] and round(t2['bsfc_ref'])}  "
              f"support HIGH={t2['support_HIGH']*100:.0f}%  -> HIGH undervalued: {t2['HIGH_undervalued']}")

    json.dump(dict(cycle=cycle, band_regions=band_agg, type_classification=verdict),
              open(out / f"data/critic_error_map_{cycle}.json", "w"), indent=2)
    P(f"\n[saved] {out}/data/critic_error_map_{cycle}.json")
    return band_agg, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, choices=["NEDC", "FTP75"])
    ap.add_argument("--out", default="results/phase9")
    a = ap.parse_args()
    out = Path(a.out); (out / "data").mkdir(parents=True, exist_ok=True); (out / "logs").mkdir(exist_ok=True)
    fh = open(out / f"logs/phase9_critic_diag_{a.cycle}.txt", "w", encoding="utf-8")
    P = lambda s: (print(s), fh.write(str(s) + "\n"))
    run(a.cycle, P, out)
    fh.close()


if __name__ == "__main__":
    main()
