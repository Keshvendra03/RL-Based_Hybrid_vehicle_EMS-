"""
PHASE 12 STAGE A -- reward-domain mathematical verification (NO TRAINING).

A2  derive eq_eff(SoC) zero-crossing directly from the current source + params
A5  CONTROL non-regression: R_original vs R_corrected on every transition of the
    3 NEDC CONTROL deterministic trajectories (+ 3 FTP75)
A6  synthetic SoC sweep 0.30..0.95, both eq_eff and both rewards, with a plot

R_original  = EMSEnv(..., clip_eq_eff=False)   [pre-Phase-12 behaviour, default]
R_corrected = EMSEnv(..., clip_eq_eff=True)    [eq_factor_eff = max(eq_factor_eff, 0)]

Outputs: results/phase12/data/stage_a_*.json , results/phase12/figures/stage_a_eq_eff_sweep.png
"""
import json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from src.env.ems_env import (EMSEnv, SOC_TARGET, REWARD_SCALE,
                             K_FUEL_L_PER_KG, K_ELEC_L_PER_J)
from src.env.powertrain import _Q_BT_0

DATA = Path("results/phase12/data"); DATA.mkdir(parents=True, exist_ok=True)
FIG = Path("results/phase12/figures"); FIG.mkdir(parents=True, exist_ok=True)

CFG = {
    "NEDC":  dict(eq_factor=0.2717, k_fb=2.5, action_map="modeaware_gated", lookahead=5,
                  ckpts=["models_p5s0_k2.5/NEDC", "models_p5_k2.5/NEDC", "models_p5_k2.5_s2/NEDC"]),
    "FTP75": dict(eq_factor=0.4981, k_fb=2.5, action_map="modeaware_gated", lookahead=5,
                  ckpts=["models_p5f_k2.5_s0/FTP75", "models_p5f_k2.5_s1/FTP75", "models_p5f_k2.5_s2/FTP75"]),
}


# --------------------------------------------------------------------- A2
def threshold(eq_factor, k_fb):
    """eq_eff(SoC) = eq_factor + k_fb*(SOC_TARGET - SoC) ; = 0 at SoC = SOC_TARGET + eq_factor/k_fb."""
    return SOC_TARGET + eq_factor / k_fb


# --------------------------------------------------------------------- A5
def control_nonregression(cycle):
    c = CFG[cycle]
    per_ckpt = []
    for ck in c["ckpts"]:
        m = SAC.load(f"{ck}/sac_ems_best")
        env_o = EMSEnv(cycle, eq_factor=c["eq_factor"], k_fb=c["k_fb"],
                       action_map=c["action_map"], lookahead=c["lookahead"], clip_eq_eff=False)
        env_c = EMSEnv(cycle, eq_factor=c["eq_factor"], k_fb=c["k_fb"],
                       action_map=c["action_map"], lookahead=c["lookahead"], clip_eq_eff=True)
        o_o, _ = env_o.reset(); o_c, _ = env_c.reset()
        thr = threshold(c["eq_factor"], c["k_fb"])
        r_o, r_c, socs, eqeff = [], [], [], []
        while True:
            soc_before = env_o._Q_BT / _Q_BT_0
            eqe = c["eq_factor"] + c["k_fb"] * (SOC_TARGET - soc_before)
            socs.append(soc_before); eqeff.append(eqe)
            a, _ = m.predict(o_o, deterministic=True)
            o_o, ro, to, _, _ = env_o.step(a)
            o_c, rc, tc, _, _ = env_c.step(a)      # identical action -> identical physical trajectory
            r_o.append(ro); r_c.append(rc)
            if to:
                break
        r_o = np.array(r_o); r_c = np.array(r_c)
        socs = np.array(socs); eqeff = np.array(eqeff)
        below = socs <= thr
        affected = ~below
        d = np.abs(r_o - r_c)
        rel = np.abs(r_o - r_c) / (np.abs(r_o) + 1e-12)
        per_ckpt.append(dict(
            ckpt=ck, n_transitions=int(len(r_o)),
            soc_min=float(socs.min()), soc_max=float(socs.max()), threshold=float(thr),
            n_below_threshold=int(below.sum()), n_affected=int(affected.sum()),
            max_abs_diff_below=float(d[below].max()) if below.any() else 0.0,
            max_rel_diff_below=float(rel[below].max()) if below.any() else 0.0,
            max_abs_diff_affected=float(d[affected].max()) if affected.any() else 0.0,
            eqeff_min=float(eqeff.min()), eqeff_max=float(eqeff.max()),
            cum_R_original=float(r_o.sum()), cum_R_corrected=float(r_c.sum()),
            dR_total=float(r_c.sum() - r_o.sum()),
        ))
    return per_ckpt


# --------------------------------------------------------------------- A6
def synthetic_sweep(cycle):
    """Identical physical conditions (fixed fuel_liters, fixed |elec_liters|),
    sweep SoC 0.30..0.95; evaluate original vs corrected eq_eff and the resulting
    per-step economic reward term R_econ = -REWARD_SCALE*(fuel_L + eq_eff*elec_L)
    for a DISCHARGE step (elec_L>0) and a CHARGE step (elec_L<0).

    The economic term is the ONLY place eq_eff enters the reward (ems_env.py:643),
    so isolating it with fixed fuel_L/elec_L is the exact, operating-point-free
    demonstration of where the two formulations diverge.

    Representative magnitudes taken from a real mid-demand CONTROL step
    (T_MGB ~ 60 Nm): fuel_L ~ 3.0e-3 L, |elec_L| ~ 6.0e-3 equiv-L per step.
    """
    c = CFG[cycle]
    thr = threshold(c["eq_factor"], c["k_fb"])
    socs = np.linspace(0.30, 0.95, 66)
    FUEL_L = 3.0e-3
    ELEC_L = 6.0e-3          # discharge = +ELEC_L ; charge = -ELEC_L

    def R_econ(fuel_L, elec_L, eq_eff):
        return -REWARD_SCALE * (fuel_L + eq_eff * elec_L)

    rows = []
    for soc in socs:
        eqe_o = c["eq_factor"] + c["k_fb"] * (SOC_TARGET - soc)
        eqe_c = max(eqe_o, 0.0)
        rows.append(dict(
            soc=float(soc), eq_eff_orig=float(eqe_o), eq_eff_corr=float(eqe_c),
            R_disch_orig=float(R_econ(FUEL_L, +ELEC_L, eqe_o)),
            R_disch_corr=float(R_econ(FUEL_L, +ELEC_L, eqe_c)),
            R_charge_orig=float(R_econ(FUEL_L, -ELEC_L, eqe_o)),
            R_charge_corr=float(R_econ(FUEL_L, -ELEC_L, eqe_c)),
            elec_L_discharge=+ELEC_L, elec_L_charge=-ELEC_L))
    # plot
    S = np.array([r["soc"] for r in rows]) * 100
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].axhline(0, color="k", lw=0.8); ax[0].axvline(thr * 100, color="r", ls="--", lw=1,
                 label=f"zero-crossing {thr*100:.2f}%")
    ax[0].plot(S, [r["eq_eff_orig"] for r in rows], label="eq_eff ORIGINAL (unbounded)")
    ax[0].plot(S, [r["eq_eff_corr"] for r in rows], label="eq_eff CORRECTED (max(.,0))", lw=2)
    ax[0].set_xlabel("SoC [%]"); ax[0].set_ylabel("eq_factor_eff  [fuel-J per battery-J]")
    ax[0].set_title(f"{cycle}: equivalence factor vs SoC"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].axvline(thr * 100, color="r", ls="--", lw=1)
    ax[1].plot(S, [r["R_disch_orig"] for r in rows], label="R original")
    ax[1].plot(S, [r["R_disch_corr"] for r in rows], label="R corrected", lw=2)
    ax[1].set_xlabel("SoC [%]"); ax[1].set_ylabel("per-step reward")
    ax[1].set_title(f"{cycle}: reward, DISCHARGE action (a=+0.6)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].axvline(thr * 100, color="r", ls="--", lw=1)
    ax[2].plot(S, [r["R_charge_orig"] for r in rows], label="R original")
    ax[2].plot(S, [r["R_charge_corr"] for r in rows], label="R corrected", lw=2)
    ax[2].set_xlabel("SoC [%]"); ax[2].set_ylabel("per-step reward")
    ax[2].set_title(f"{cycle}: reward, CHARGE action (a=-0.7)"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.tight_layout()
    p = FIG / f"stage_a_eq_eff_sweep_{cycle}.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return dict(threshold_pct=float(thr * 100), fixed_fuel_L=FUEL_L, fixed_abs_elec_L=ELEC_L,
                rows=rows, figure=str(p))


if __name__ == "__main__":
    out = {"note": "eq_eff(SoC) = eq_factor + k_fb*(0.5 - SoC); zero at SoC = 0.5 + eq_factor/k_fb",
           "params": {c: {"eq_factor": CFG[c]["eq_factor"], "k_fb": CFG[c]["k_fb"],
                          "SOC_TARGET": SOC_TARGET, "REWARD_SCALE": REWARD_SCALE,
                          "K_FUEL_L_PER_KG": K_FUEL_L_PER_KG, "K_ELEC_L_PER_J": K_ELEC_L_PER_J,
                          "zero_crossing_SoC": threshold(CFG[c]["eq_factor"], CFG[c]["k_fb"]),
                          "zero_crossing_pct": threshold(CFG[c]["eq_factor"], CFG[c]["k_fb"]) * 100,
                          "eq_eff_at_soc_0.95": CFG[c]["eq_factor"] + CFG[c]["k_fb"] * (SOC_TARGET - 0.95)}
                      for c in CFG},
           "A5_control_nonregression": {}, "A6_synthetic_sweep": {}}
    for cyc in ("NEDC", "FTP75"):
        print(f"\n===== {cyc}  zero-crossing SoC = {threshold(CFG[cyc]['eq_factor'], CFG[cyc]['k_fb'])*100:.4f}%")
        nr = control_nonregression(cyc)
        out["A5_control_nonregression"][cyc] = nr
        for r in nr:
            print(f"  {r['ckpt']:<26} n={r['n_transitions']}  SoC[min,max]=[{r['soc_min']*100:.1f},{r['soc_max']*100:.1f}]%  "
                  f"n_below_thr={r['n_below_threshold']}  n_affected={r['n_affected']}  "
                  f"max|dR|_below={r['max_abs_diff_below']:.3e}  max_rel_below={r['max_rel_diff_below']:.3e}  "
                  f"dR_total={r['dR_total']:.3e}")
        sw = synthetic_sweep(cyc)
        out["A6_synthetic_sweep"][cyc] = sw
        print(f"  synthetic sweep -> {sw['figure']}")

    (DATA / "stage_a_reward_domain.json").write_text(json.dumps(out, indent=2))
    print("\n[saved] results/phase12/data/stage_a_reward_domain.json")
