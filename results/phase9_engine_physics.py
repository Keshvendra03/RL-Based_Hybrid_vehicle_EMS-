"""
phase9_engine_physics.py
========================
PHASE 9 section 10 -- physically explain why ECMS uses more engine torque but
less fuel. Matched-demand rollout of CONTROL SAC vs ECMS through the validated
env; at every step record engine torque/speed/BSFC/efficiency/fuel-rate,
battery & motor power, SoC, regen. Then decompose the SAC-ECMS fuel gap:

  A  excess fuel from inefficient engine OPERATING POINTS
     (steps where BOTH run the engine: (BSFC_SAC - BSFC_ECMS) * mech_energy)
  B  fuel from different engine ON/OFF DECISIONS
     (steps where exactly one runs the engine)
  C  battery-energy timing / SoC management  (net electrical-equivalent term)
  D  residual

    python -m results.phase9_engine_physics --cycle NEDC
"""
from __future__ import annotations
import argparse, json, types
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from src.env.ems_env import EMSEnv, SOC_TARGET, K_ELEC_L_PER_J, K_FUEL_L_PER_KG
from src.env import powertrain as pt
from src.env.powertrain import (_T_CUTOFF, combustion_engine, electric_motor, _Q_BT_0,
                                _V_CE_map, _w_CE_row, _T_CE_col, _H_u)
from src.baselines.ecms import _hamiltonian_best_u
from results.phase7_forensics import EQF, LAM0, BENCH, ECMS_V, AMAP, KFB_CONTROL, CKPTS

TB = [(-1e9, 0, "brake"), (0, 15, "0-15"), (15, 30, "15-30"), (30, 35, "30-35"),
      (35, 50, "35-50"), (50, 75, "50-75"), (75, 1e9, ">75")]


def _engine_point(w, dw, t_ce):
    if t_ce <= _T_CUTOFF:
        return dict(on=False, bsfc=None, eff=0.0, fuel_g_s=0.0, mech_W=0.0,
                    w_ce_rpm=w * 60 / (2 * np.pi), t_ce=t_ce, fuel_W=0.0)
    eng = combustion_engine(w_gear=w, dw_gear=dw, t_gear=t_ce)
    mech = eng["t_ce"] * eng["w_ce"]
    fuel_W = max(eng["p_ce_fuel"], 1e-9)
    _eff = (mech / fuel_W) if mech > 2e3 else np.nan
    _eff = _eff if (0.0 <= _eff <= 0.60) else np.nan        # guard idle/near-cutoff blow-ups
    _bsfc = (eng["v_dot"] / mech * 3.6e9) if mech > 2e3 else None
    if _bsfc is not None and not (150.0 <= _bsfc <= 900.0):
        _bsfc = None
    return dict(on=True, bsfc=_bsfc,
                eff=_eff, fuel_g_s=eng["v_dot"] * 1000.0,
                mech_W=mech, w_ce_rpm=eng["w_ce"] * 60 / (2 * np.pi), t_ce=eng["t_ce"],
                fuel_W=fuel_W)


def rollout(cycle, which, kfb=KFB_CONTROL):
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=kfb, lookahead=5,
                 action_map=AMAP if which == "sac" else "linear")
    if which == "sac":
        m = SAC.load(f"{CKPTS[cycle]['control_k2.5_gated'][0]}/{cycle}/sac_ems_best")
    else:
        lam0 = LAM0[cycle]

        def patched(self, action):
            d = self._demand
            w, dw, T = d["w_MGB"], d["dw_MGB"], d["T_MGB"]
            soc = self._Q_BT / _Q_BT_0
            if T == 0.0 or w <= 0.0:
                return 0.0, 0.0, 0.0, "stop"
            u = _hamiltonian_best_u(w, dw, T, soc, lam0 + 8.0 * (SOC_TARGET - soc), 81)
            te = u * T
            return T - te, te, u, ("regen" if T < 0 else ("lps_gen" if te < 0 else
                                   ("assist" if te > 0 else "engine")))
        env._action_to_torques = types.MethodType(patched, env)

    obs, _ = env.reset()
    R = []
    while True:
        d = dict(env._demand)
        if which == "sac":
            a, _ = m.predict(obs, deterministic=True)
        else:
            a = np.zeros(1, np.float32)
        obs, r, t, _, i = env.step(a)
        ep = _engine_point(d["w_MGB"], d["dw_MGB"], i["T_CE_cmd"])
        R.append(dict(T=d["T_MGB"], w=d["w_MGB"], dw=d["dw_MGB"], soc=i["soc"],
                      t_ce_cmd=i["T_CE_cmd"], t_em=i["T_EM_cmd"], p_em=i["p_em"],
                      fuel_step=i["fuel_liters_step"], elec_step=i["elec_liters_step"], **ep))
        if t:
            R[-1]["final"] = i["episode_final"]
            return R


def decompose(cycle, P, out):
    P(f"\n{'='*96}\nPHASE 9 §10  PHYSICAL SAC-vs-ECMS DECOMPOSITION -- {cycle}\n{'='*96}")
    S = rollout(cycle, "sac")
    E = rollout(cycle, "ecms")
    n = min(len(S), len(E)); S, E = S[:n], E[:n]
    dT = max(abs(S[i]["T"] - E[i]["T"]) for i in range(n))
    P(f"  demand alignment: max|T_SAC - T_ECMS| = {dT:.2e}   steps={n}")
    K = 1e5 / E[-1]["final"]["x_tot_m"]                        # step-liters -> L/100km

    both_on = one_on = 0
    A_bsfc = 0.0          # L/100km from operating-point BSFC (both engines on)
    B_onoff = 0.0         # L/100km from different ON/OFF decisions
    engine_on_S = engine_on_E = 0
    bsfc_S, bsfc_E, wce_S, wce_E, tce_S, tce_E, eff_S, eff_E = ([] for _ in range(8))
    for i in range(n):
        s, e = S[i], E[i]
        if s["on"]:
            engine_on_S += 1; bsfc_S.append(s["bsfc"]); wce_S.append(s["w_ce_rpm"])
            tce_S.append(s["t_ce"]); eff_S.append(s["eff"])
        if e["on"]:
            engine_on_E += 1; bsfc_E.append(e["bsfc"]); wce_E.append(e["w_ce_rpm"])
            tce_E.append(e["t_ce"]); eff_E.append(e["eff"])
        if s["on"] and e["on"]:
            both_on += 1
            # fuel to deliver ECMS's mech work at SAC's BSFC vs ECMS's BSFC
            if s["bsfc"] and e["bsfc"]:
                d_fuel_kg = (s["bsfc"] - e["bsfc"]) / 3.6e9 * e["mech_W"]   # kg this step
                A_bsfc += d_fuel_kg * K_FUEL_L_PER_KG * K
        elif s["on"] != e["on"]:
            one_on += 1
            # signed: SAC fuel this step minus ECMS fuel this step (one of them ~0)
            B_onoff += (s["fuel_step"] - e["fuel_step"]) * K

    fuelS = S[-1]["final"]["v_liter"]; fuelE = E[-1]["final"]["v_liter"]
    vceS = S[-1]["final"]["v_ce_equiv"]; vceE = E[-1]["final"]["v_ce_equiv"]
    # C: battery-energy timing / SoC-equivalence -- the electrical-equivalent term
    #    (difference of the equivalent-fuel electrical component)
    C_batt = (vceS - fuelS) - (vceE - fuelE)
    gap_vce = vceS - vceE
    gap_fuel = fuelS - fuelE
    D_resid = gap_fuel - (A_bsfc + B_onoff)

    P(f"\n  raw engine fuel  : SAC {fuelS:.4f}  ECMS {fuelE:.4f}  (SAC-ECMS {gap_fuel:+.4f} L/100km)")
    P(f"  V_CE_equiv       : SAC {vceS:.4f}  ECMS {vceE:.4f}  (SAC-ECMS {gap_vce:+.4f} L/100km)")
    P(f"  engine-on steps  : SAC {engine_on_S}  ECMS {engine_on_E}   (both-on {both_on}, exactly-one-on {one_on})")
    P(f"  mean BSFC [g/kWh]: SAC {np.nanmedian([b for b in bsfc_S if b]):.0f}  ECMS {np.nanmedian([b for b in bsfc_E if b]):.0f}")
    P(f"  mean engine rpm  : SAC {np.nanmean(wce_S):.0f}  ECMS {np.nanmean(wce_E):.0f}")
    P(f"  mean engine T_CE : SAC {np.nanmean(tce_S):.1f}  ECMS {np.nanmean(tce_E):.1f}")
    P(f"  mean engine eff  : SAC {np.nanmean(eff_S):.3f}  ECMS {np.nanmean(eff_E):.3f}")
    P(f"\n  DECOMPOSITION of the SAC-ECMS gap (raw fuel basis, +C for the battery term):")
    P(f"    A  operating-point BSFC (both engines on) : {A_bsfc:+.4f} L/100km")
    P(f"    B  different ON/OFF decisions             : {B_onoff:+.4f} L/100km")
    P(f"    C  battery-energy timing / SoC-equivalence: {C_batt:+.4f} L/100km")
    P(f"    D  residual (transients, interaction)     : {D_resid:+.4f} L/100km")
    P(f"    ---------------------------------------------------------------")
    P(f"    A+B+D (raw fuel gap)      = {A_bsfc + B_onoff + D_resid:+.4f}  (measured {gap_fuel:+.4f})")
    P(f"    A+B+C+D (V_CE gap)        = {A_bsfc + B_onoff + C_batt + D_resid:+.4f}  (measured {gap_vce:+.4f})")

    # per-band A/B
    P(f"\n  {'band':>7}{'A_bsfc':>10}{'B_onoff':>10}{'SAC on%':>9}{'ECMS on%':>10}{'dBSFC':>9}")
    band_rows = {}
    for lo, hi, nm in TB:
        idx = [i for i in range(n) if lo <= S[i]["T"] < hi]
        if not idx:
            continue
        a_b = b_b = 0.0
        for i in idx:
            s, e = S[i], E[i]
            if s["on"] and e["on"] and s["bsfc"] and e["bsfc"]:
                a_b += (s["bsfc"] - e["bsfc"]) / 3.6e9 * e["mech_W"] * K_FUEL_L_PER_KG * K
            elif s["on"] != e["on"]:
                b_b += (s["fuel_step"] - e["fuel_step"]) * K
        son = 100 * np.mean([S[i]["on"] for i in idx]); eon = 100 * np.mean([E[i]["on"] for i in idx])
        sb = np.nanmean([S[i]["bsfc"] for i in idx if S[i]["on"]]) if any(S[i]["on"] for i in idx) else np.nan
        eb = np.nanmean([E[i]["bsfc"] for i in idx if E[i]["on"]]) if any(E[i]["on"] for i in idx) else np.nan
        band_rows[nm] = dict(A_bsfc=a_b, B_onoff=b_b, sac_on_pct=son, ecms_on_pct=eon,
                             bsfc_sac=float(sb) if sb == sb else None,
                             bsfc_ecms=float(eb) if eb == eb else None)
        P(f"  {nm:>7}{a_b:>+10.4f}{b_b:>+10.4f}{son:>8.0f}%{eon:>9.0f}%{(sb-eb) if sb==sb and eb==eb else float('nan'):>9.0f}")

    res = dict(cycle=cycle, gap_vce=gap_vce, gap_fuel=gap_fuel,
               A_operating_point_bsfc=A_bsfc, B_onoff_decisions=B_onoff,
               C_battery_soc_equiv=C_batt, D_residual=D_resid,
               A_pct=100 * A_bsfc / gap_vce, B_pct=100 * B_onoff / gap_vce,
               C_pct=100 * C_batt / gap_vce, D_pct=100 * D_resid / gap_vce,
               engine_on_sac=engine_on_S, engine_on_ecms=engine_on_E,
               mean_bsfc_sac=float(np.nanmedian([b for b in bsfc_S if b])), mean_bsfc_ecms=float(np.nanmedian([b for b in bsfc_E if b])),
               mean_rpm_sac=float(np.nanmean(wce_S)), mean_rpm_ecms=float(np.nanmean(wce_E)),
               mean_eff_sac=float(np.nanmean(eff_S)), mean_eff_ecms=float(np.nanmean(eff_E)),
               per_band=band_rows)
    json.dump(res, open(out / f"data/engine_physics_{cycle}.json", "w"), indent=2)

    # ---- BSFC map figure with SAC vs ECMS operating points ----
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    bsfc_map = np.full_like(_V_CE_map, np.nan, dtype=float)
    for r, w in enumerate(_w_CE_row):
        for c, tq in enumerate(_T_CE_col):
            mech = tq * w
            if mech > 1e3 and _V_CE_map[r, c] > 0:
                bsfc_map[r, c] = _V_CE_map[r, c] / mech * 3.6e9
    cs = ax.contourf(_w_CE_row * 60 / (2 * np.pi), _T_CE_col, np.clip(bsfc_map.T, 180, 500),
                     levels=20, cmap="viridis_r")
    fig.colorbar(cs, label="BSFC [g/kWh]")
    on_S = [(s["w_ce_rpm"], s["t_ce"]) for s in S if s["on"]]
    on_E = [(e["w_ce_rpm"], e["t_ce"]) for e in E if e["on"]]
    ax.scatter(*zip(*on_S), s=6, c="red", alpha=0.4, label=f"SAC engine-on (n={len(on_S)})")
    ax.scatter(*zip(*on_E), s=6, c="white", alpha=0.5, label=f"ECMS engine-on (n={len(on_E)})", edgecolors="k", linewidths=0.2)
    ax.set_xlabel("engine speed [rpm]"); ax.set_ylabel("engine torque [Nm]")
    ax.set_title(f"{cycle}: engine operating points on the BSFC map -- SAC vs ECMS")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out / f"figures/bsfc_map_{cycle}.png", dpi=110); plt.close(fig)
    P(f"  [saved] figures/bsfc_map_{cycle}.png")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, choices=["NEDC", "FTP75"])
    ap.add_argument("--out", default="results/phase9")
    a = ap.parse_args()
    out = Path(a.out); (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True); (out / "logs").mkdir(exist_ok=True)
    fh = open(out / f"logs/phase9_engine_physics_{a.cycle}.txt", "w", encoding="utf-8")
    P = lambda s: (print(s), fh.write(str(s) + "\n"))
    decompose(a.cycle, P, out)
    fh.close()


if __name__ == "__main__":
    main()
