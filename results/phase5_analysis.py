"""
phase5_analysis.py
==================
Phase-5 costate-gain (k_fb) identification and SoC-stability analysis.

Covers brief sections 4, 5, 6, 8, 9, 11, 24, 25:
   4  full metric record per candidate (never fuel alone)
   5  SoC TRAJECTORY analysis, not just terminal SoC
   6  control law vs torque band for each k_fb
   8  eq_factor(SoC) over the ACTUALLY VISITED SoC range, in ECMS units
   9  monotonic vs U-shaped check on k_fb
  11  Pareto table NEDC vs FTP75
  24  timestep-aligned error budget per candidate
  25  explicit high-torque (>50 Nm) offset monitoring

    python -m results.phase5_analysis --cycle NEDC --out results/phase5

No training. Physics untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.env.ems_env import EMSEnv, SOC_TARGET
from src.env.powertrain import _T_CUTOFF

EQF = {"NEDC": 0.2717, "FTP75": 0.4981}
BENCH = {"NEDC": 3.5056, "FTP75": 3.2323}
AUTH_EQ = {"NEDC": 3.5792, "FTP75": 3.2318}
ECMS = {"NEDC": 3.1887, "FTP75": 2.8097}
LIN_REF = {"NEDC": (3.7727, 0.0281), "FTP75": (3.3821, 0.0846)}
ECMS_UNIT = 4.8309          # eq_factor(liter-units) * this = ECMS lambda units
BANDS = [(0, 15, "0-15"), (15, 30, "15-30"), (30, 35, "30-35"),
         (35, 50, "35-50"), (50, 75, "50-75"), (75, 1e9, ">75")]


def cls(mode, t_ce):
    if mode == "stop":
        return "stop"
    if mode == "regen":
        return "regen"
    if t_ce <= _T_CUTOFF:
        return "OFF"
    return {"assist": "ASSIST", "lps_gen": "LPS"}.get(mode, "ONLY")


def roll_rl(ckpt, cycle, k_fb, amap):
    from stable_baselines3 import SAC
    m = SAC.load(ckpt)
    od = int(m.observation_space.shape[0])
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=k_fb, action_map=amap,
                 lookahead=0 if od <= 16 else od - 15)
    obs, _ = env.reset()
    R = []
    while True:
        d = dict(env._demand)
        a, _ = m.predict(obs, deterministic=True)
        obs, r, term, _, i = env.step(a)
        R.append(dict(T=d["T_MGB"], soc=i["soc"], cls=cls(i["mode"], i["T_CE_cmd"]),
                      fuel=i["fuel_liters_step"], elec=i["elec_liters_step"],
                      t_ce=i["T_CE_cmd"], t_em=i["T_EM_cmd"],
                      a=float(np.asarray(a).reshape(-1)[0])))
        if term:
            R[-1]["final"] = i["episode_final"]
            return R


def roll_bench(cycle, which):
    import types
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=1.656, lookahead=0)
    if which == "rule_based":
        from src.baselines.advanced_rule_based import (AdvancedController,
                                                       control_unit_advanced)
        c = AdvancedController(cycle_name=cycle); c.reset()

        def patched(self, action):
            d = self._demand
            o = c.step(d["w_MGB"], d["dw_MGB"], d["T_MGB"], d["gear"], self._Q_BT, d["v"])
            cu = control_unit_advanced(d["w_MGB"], d["dw_MGB"], d["T_MGB"], o["u"], o["state_CE"])
            m = ("regen" if d["T_MGB"] < 0 else
                 ("stop" if d["T_MGB"] == 0 or d["w_MGB"] <= 0 else
                  ("lps_gen" if cu["T_EM"] < 0 else ("assist" if cu["T_EM"] > 0 else "engine"))))
            return cu["T_CE"], cu["T_EM"], o["u"], m
    else:
        from src.baselines.ecms import _hamiltonian_best_u
        lam0 = {"NEDC": 1.3125, "FTP75": 2.4062}[cycle]

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
        obs, r, term, _, i = env.step(np.zeros(1, np.float32))
        R.append(dict(T=d["T_MGB"], soc=i["soc"], cls=cls(i["mode"], i["T_CE_cmd"]),
                      fuel=i["fuel_liters_step"], elec=i["elec_liters_step"],
                      t_ce=i["T_CE_cmd"], t_em=i["T_EM_cmd"], a=np.nan))
        if term:
            R[-1]["final"] = i["episode_final"]
            return R


# ---------------- section 5: SoC trajectory metrics ------------------------ #
def soc_metrics(R):
    s = np.array([x["soc"] for x in R])
    n = len(s)
    t = np.arange(n)
    slope = float(np.polyfit(t, s, 1)[0]) * 3600.0   # pp-equivalent per hour *100 later
    return dict(soc_min=float(s.min()), soc_max=float(s.max()),
                soc_final=float(s[-1]), d_soc_pp=float((s[-1] - 0.5) * 100),
                max_excursion_pp=float(np.max(np.abs(s - 0.5)) * 100),
                frac_above=float(np.mean(s > 0.5)), frac_below=float(np.mean(s < 0.5)),
                integral_abs_dev=float(np.mean(np.abs(s - 0.5)) * 100),
                drift_slope_pp_per_1000s=float(slope * 100 / 3.6),
                charge_sustaining=bool(abs(s[-1] - 0.5) <= 0.02))


# ---------------- section 8: costate over the visited SoC ------------------ #
def costate_metrics(R, cycle, k_fb):
    s = np.array([x["soc"] for x in R])
    eq = EQF[cycle] + k_fb * (SOC_TARGET - s)
    return dict(eq_min=float(eq.min()), eq_max=float(eq.max()),
                eq_mean=float(eq.mean()), eq_median=float(np.median(eq)),
                ecms_min=float(eq.min() * ECMS_UNIT), ecms_max=float(eq.max() * ECMS_UNIT),
                ecms_mean=float(eq.mean() * ECMS_UNIT),
                frac_eq_negative=float(np.mean(eq < 0)),
                soc_flip=float(SOC_TARGET + EQF[cycle] / k_fb) if k_fb > 0 else np.inf)


# ---------------- section 6: control law by band --------------------------- #
def band_law(R):
    out = {}
    for lo, hi, nm in BANDS:
        idx = [i for i, x in enumerate(R) if lo <= x["T"] < hi and x["cls"] != "stop"]
        if not idx:
            continue
        f = lambda k: 100.0 * np.mean([R[i]["cls"] == k for i in idx])
        out[nm] = dict(n=len(idx), OFF=f("OFF"), ASSIST=f("ASSIST"), LPS=f("LPS"),
                       t_ce=float(np.mean([R[i]["t_ce"] for i in idx])),
                       t_em=float(np.mean([R[i]["t_em"] for i in idx])))
    return out


# ---------------- sections 24/25: error budget ----------------------------- #
def error_budget(R, RB):
    K = 1e5 / R[-1]["final"]["x_tot_m"]
    regions = [("braking/regen", lambda x: x["T"] < 0),
               ("standstill", lambda x: x["cls"] == "stop"),
               ("0-15", lambda x: 0 <= x["T"] < 15 and x["cls"] != "stop"),
               ("15-30", lambda x: 15 <= x["T"] < 30),
               ("30-50", lambda x: 30 <= x["T"] < 50),
               ("50-75", lambda x: 50 <= x["T"] < 75),
               (">75", lambda x: x["T"] >= 75)]
    out = {}
    for nm, f in regions:
        ia = [i for i, x in enumerate(R) if f(x)]
        ib = [i for i, x in enumerate(RB) if f(x)]
        if not ia:
            continue
        out[nm] = dict(dfuel=(sum(R[i]["fuel"] for i in ia) - sum(RB[i]["fuel"] for i in ib)) * K,
                       delec=(sum(R[i]["elec"] for i in ia) - sum(RB[i]["elec"] for i in ib)) * K,
                       steps=len(ia))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", default="NEDC")
    ap.add_argument("--out", default="results/phase5")
    ap.add_argument("--candidates", default="", help="comma list dir:k_fb:amap")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    cands = []
    if a.candidates:
        for c in a.candidates.split(","):
            d, k, m = c.split(":")
            cands.append((f"k_fb={k}", d, float(k), m))
    else:
        cands = [("linear ref k=1.656", f"models_expD_g20", 1.656, "linear"),
                 ("gated k=1.656", f"models_p4_gated_g20", 1.656, "modeaware_gated")]
        for k in ["2.0", "2.5", "3.0", "4.0", "5.0"]:
            cands.append((f"gated k={k}", f"models_p5_k{k}", float(k), "modeaware_gated"))

    rb = roll_bench(a.cycle, "rule_based")
    ec = roll_bench(a.cycle, "ecms")
    rows, traj = [], {}
    fh = open(out / f"phase5_{a.cycle}.txt", "w", encoding="utf-8")
    P = lambda s: (print(s), fh.write(s + "\n"))

    P("=" * 118)
    P(f"PHASE 5 -- k_fb SWEEP, {a.cycle}   (all else FROZEN: gamma=0.20, n_step=1,"
      f" gated map, seed 1, 150k)")
    P(f"linear reference: {LIN_REF[a.cycle][0]:.4f} +/- {LIN_REF[a.cycle][1]:.4f}  |  "
      f"rule-based {BENCH[a.cycle]}  |  authority-equal {AUTH_EQ[a.cycle]}  |  ECMS {ECMS[a.cycle]}")
    P("=" * 118)

    for label, d, k, amap in cands:
        ck = Path(d) / a.cycle / "sac_ems_best"
        if not ck.with_suffix(".zip").exists():
            P(f"  {label:<20} MISSING ({ck})"); continue
        R = roll_rl(str(ck), a.cycle, k, amap)
        sm = soc_metrics(R); cm = costate_metrics(R, a.cycle, k)
        bl = band_law(R); eb = error_budget(R, rb)
        mv = [x for x in R if x["cls"] != "stop"]
        rows.append(dict(label=label, k=k, amap=amap,
                         v=R[-1]["final"]["v_ce_equiv"], vl=R[-1]["final"]["v_liter"],
                         off=100 * np.mean([x["cls"] == "OFF" for x in mv]),
                         asst=100 * np.mean([x["cls"] == "ASSIST" for x in mv]),
                         lps=100 * np.mean([x["cls"] == "LPS" for x in mv]),
                         regen=100 * np.mean([x["cls"] == "regen" for x in mv]),
                         **sm, **cm, bands=bl, budget=eb))
        traj[label] = np.array([x["soc"] for x in R])

    # ---- section 4/11 headline table ----
    P(f"\n{'candidate':<20}{'V_CE_eq':>9}{'V_liter':>9}{'dSoC':>8}{'CS':>4}"
      f"{'SoCmin':>8}{'SoCmax':>8}{'|dev|':>7}{'drift':>8}"
      f"{'OFF':>7}{'ASST':>7}{'LPS':>7}{'REGEN':>7}")
    for r in rows:
        P(f"{r['label']:<20}{r['v']:>9.4f}{r['vl']:>9.4f}{r['d_soc_pp']:>+8.2f}"
          f"{('Y' if r['charge_sustaining'] else 'N'):>4}"
          f"{r['soc_min']*100:>8.1f}{r['soc_max']*100:>8.1f}{r['integral_abs_dev']:>7.2f}"
          f"{r['drift_slope_pp_per_1000s']:>+8.2f}"
          f"{r['off']:>7.1f}{r['asst']:>7.1f}{r['lps']:>7.1f}{r['regen']:>7.1f}")

    # ---- section 8 costate ----
    P(f"\n--- SECTION 8: eq_factor over the VISITED SoC (ECMS lambda_0 = "
      f"{1.3125 if a.cycle=='NEDC' else 2.4062}) ---")
    P(f"{'candidate':<20}{'eq_min':>9}{'eq_max':>9}{'eq_mean':>9}"
      f"{'ECMS_min':>10}{'ECMS_max':>10}{'ECMS_mean':>11}{'SoC_flip%':>11}{'eq<0':>7}")
    for r in rows:
        P(f"{r['label']:<20}{r['eq_min']:>9.4f}{r['eq_max']:>9.4f}{r['eq_mean']:>9.4f}"
          f"{r['ecms_min']:>10.3f}{r['ecms_max']:>10.3f}{r['ecms_mean']:>11.3f}"
          f"{r['soc_flip']*100:>11.1f}{100*r['frac_eq_negative']:>6.1f}%")

    # ---- section 6 control law ----
    P(f"\n--- SECTION 6: OFF% by torque band (benchmark row for reference) ---")
    hdr = f"{'candidate':<20}" + "".join(f"{nm:>9}" for _, _, nm in BANDS)
    P(hdr)
    for r in rows:
        P(f"{r['label']:<20}" + "".join(
            f"{r['bands'].get(nm,{}).get('OFF',float('nan')):>9.1f}" for _, _, nm in BANDS))
    blrb = band_law(rb); blec = band_law(ec)
    P(f"{'RULE-BASED':<20}" + "".join(f"{blrb.get(nm,{}).get('OFF',float('nan')):>9.1f}" for _,_,nm in BANDS))
    P(f"{'ECMS':<20}" + "".join(f"{blec.get(nm,{}).get('OFF',float('nan')):>9.1f}" for _,_,nm in BANDS))
    P(f"\n--- LPS% by torque band (watch for over-charging) ---")
    P(hdr)
    for r in rows:
        P(f"{r['label']:<20}" + "".join(
            f"{r['bands'].get(nm,{}).get('LPS',float('nan')):>9.1f}" for _, _, nm in BANDS))
    P(f"{'RULE-BASED':<20}" + "".join(f"{blrb.get(nm,{}).get('LPS',float('nan')):>9.1f}" for _,_,nm in BANDS))

    # ---- sections 24/25 error budget ----
    P(f"\n--- SECTIONS 24/25: dFuel vs rule-based by region (negative = SAC better) ---")
    regs = ["braking/regen", "0-15", "15-30", "30-50", "50-75", ">75"]
    P(f"{'candidate':<20}" + "".join(f"{r_:>14}" for r_ in regs) + f"{'TOTAL':>11}")
    for r in rows:
        tot = sum(v["dfuel"] for v in r["budget"].values())
        P(f"{r['label']:<20}" + "".join(
            f"{r['budget'].get(rg,{}).get('dfuel',0.0):>+14.4f}" for rg in regs) + f"{tot:>+11.4f}")

    # ---- section 9 monotonic vs U-shaped ----
    g = [r for r in rows if r["amap"] == "modeaware_gated"]
    g.sort(key=lambda r: r["k"])
    if len(g) >= 3:
        P(f"\n--- SECTION 9: is fuel monotonic in k_fb? ---")
        ks = [r["k"] for r in g]; vs = [r["v"] for r in g]
        P("  k_fb : " + "  ".join(f"{k:>6.3f}" for k in ks))
        P("  V    : " + "  ".join(f"{v:>6.4f}" for v in vs))
        d = np.diff(vs)
        shape = ("MONOTONIC DECREASING" if all(x < 0 for x in d) else
                 "MONOTONIC INCREASING" if all(x > 0 for x in d) else
                 "U-SHAPED / NON-MONOTONIC")
        P(f"  -> {shape};  best k_fb = {ks[int(np.argmin(vs))]}  (V={min(vs):.4f})")
        cs = [r for r in g if r["charge_sustaining"]]
        P("  -> charge-sustaining candidates: "
          + (", ".join(f"k_fb={r['k']}" for r in cs) if cs else "NONE"))

    # ---- section 5 SoC trajectory plot ----
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    for lab, s in traj.items():
        ax[0].plot(s * 100, lw=1.2, label=lab)
    ax[0].plot(np.array([x["soc"] for x in rb]) * 100, "k--", lw=1.6, label="rule-based")
    ax[0].axhline(50, color="gray", lw=0.8)
    ax[0].axhspan(48, 52, alpha=0.12, color="green")
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("SoC [%]")
    ax[0].set_title(f"SECTION 5: SoC trajectory ({a.cycle})"); ax[0].legend(fontsize=7)
    if len(g) >= 3:
        ax[1].plot([r["k"] for r in g], [r["v"] for r in g], "o-", label="V_CE_equiv")
        ax[1].axhline(LIN_REF[a.cycle][0], color="tab:red", ls="--", label="linear ref")
        ax[1].axhline(BENCH[a.cycle], color="k", ls=":", label="rule-based")
        for r in g:
            ax[1].annotate("CS" if r["charge_sustaining"] else "no-CS",
                           (r["k"], r["v"]), fontsize=7,
                           color="green" if r["charge_sustaining"] else "red")
        ax[1].set_xlabel("k_fb"); ax[1].set_ylabel("V_CE_equiv"); ax[1].legend(fontsize=8)
        ax[1].set_title("SECTION 9: fuel vs costate gain")
    fig.tight_layout(); fig.savefig(out / f"phase5_{a.cycle}.png", dpi=110)
    P(f"\n[saved] {out}/phase5_{a.cycle}.png")
    json.dump([{k: v for k, v in r.items() if k != "bands"} for r in rows],
              open(out / f"phase5_{a.cycle}.json", "w"), indent=2, default=str)
    fh.close()


if __name__ == "__main__":
    main()
