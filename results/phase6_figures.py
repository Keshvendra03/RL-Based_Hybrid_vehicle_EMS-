"""
phase6_figures.py
=================
Phase-6 section K: Q(a) + actor-density figures, CONTROL vs TREATMENT.

Benchmark and ECMS actions are plotted as DIAGNOSTIC MARKERS ONLY -- they are
never training targets (section K explicit requirement).

    python -m results.phase6_figures --cycle NEDC --out results/phase6/figures
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from src.env.ems_env import EMSEnv, U_MIN, U_MAX, ZA_MODEAWARE, ZB_MODEAWARE
from src.env.powertrain import _T_CUTOFF
from src.agents.targeted_exploration import _a_off

EQF = {"NEDC": 0.2717, "FTP75": 0.4981}
AMAP = "modeaware_gated"
REGIONS = [(15, 30, "15-30 Nm"), (30, 35, "30-35 Nm"), (35, 50, "35-50 Nm")]


def u_to_a(u):
    return float(np.clip(2 * (u - U_MIN) / (U_MAX - U_MIN) - 1, -1, 1))


def collect(cycle, kfb):
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=kfb, lookahead=5, action_map=AMAP)
    obs, _ = env.reset()
    S = []
    while True:
        d = env._demand
        if d["T_MGB"] > _T_CUTOFF and d["w_MGB"] > 0:
            S.append((obs.copy(), copy.deepcopy(env), d["T_MGB"], d["w_MGB"],
                      d["dw_MGB"], (obs[4] + 1) / 2, d["gear"], d["v"]))
        obs, r, t, _, i = env.step(np.zeros(1, np.float32))
        if t:
            return S


def bench_u(cycle, st):
    """Diagnostic marker only -- benchmark + ECMS action at this state."""
    ob, sn, T, w, dw, soc, gear, v = st
    from src.baselines.advanced_rule_based import AdvancedController
    from src.baselines.ecms import _hamiltonian_best_u
    from src.env.ems_env import SOC_TARGET
    c = AdvancedController(cycle_name=cycle); c.reset()
    try:
        rb = c.step(w, dw, T, gear, soc * 36000.0, v)["u"]
    except Exception:
        rb = None
    lam0 = {"NEDC": 1.3125, "FTP75": 2.4062}[cycle]
    ec = _hamiltonian_best_u(w, dw, T, soc, lam0 + 8.0 * (SOC_TARGET - soc), 41)
    return rb, ec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", default="NEDC")
    ap.add_argument("--control", default="models_p5s0_k2.5")
    ap.add_argument("--treatment", default="models_p6_trt_N0")
    ap.add_argument("--out", default="results/phase6/figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from stable_baselines3 import SAC

    S = collect(a.cycle, 2.5)
    models = {"CONTROL": SAC.load(f"{a.control}/{a.cycle}/sac_ems_best"),
              "TREATMENT": SAC.load(f"{a.treatment}/{a.cycle}/sac_ems_best")}
    grid = np.linspace(-1, 1, 161)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for col, (lo, hi, nm) in enumerate(REGIONS):
        sel = [s for s in S if lo <= s[2] < hi and 0.40 <= s[5] < 0.50] or \
              [s for s in S if lo <= s[2] < hi]
        if not sel:
            continue
        st = sel[len(sel) // 2]
        ob, sn, T, w, dw, soc, gear, v = st
        aoff = _a_off(T, w, dw, AMAP)
        a_zero = 2 * ZA_MODEAWARE - 1
        rbu, ecu = bench_u(a.cycle, st)
        for row, (arm, m) in enumerate(models.items()):
            ax = axes[row, col]
            ot = th.as_tensor(np.repeat(ob.reshape(1, -1), len(grid), 0)).float()
            at = th.as_tensor(grid.reshape(-1, 1)).float()
            with th.no_grad():
                q = m.critic(ot, at)
                mq = np.minimum(q[0].numpy().ravel(), q[1].numpy().ravel())
                mu, ls, _ = m.actor.get_action_dist_params(
                    th.as_tensor(ob.reshape(1, -1)).float())
            mu = float(mu.numpy().ravel()[0]); sd = float(np.exp(ls.numpy().ravel()[0]))
            ax.axvspan(-1, a_zero, alpha=.09, color="tab:blue")
            ax.axvspan(a_zero, aoff, alpha=.09, color="orange")
            ax.axvspan(aoff, 1, alpha=.13, color="green")
            ax.plot(grid, mq, lw=2.2, color="tab:blue", label="min Q(a)")
            ax.axvline(np.tanh(mu), color="red", lw=2.2, label=f"actor mean")
            for k, st_ in ((1, "--"), (2, ":")):
                for sgn in (1, -1):
                    ax.axvline(np.tanh(mu + sgn * k * sd), color="red", ls=st_, lw=1,
                               label=(f"actor +/-{k}sigma" if sgn == 1 else None))
            ax.axvline(grid[mq.argmax()], color="k", ls="-.", lw=1.8, label="argmax Q")
            if rbu is not None:
                ax.axvline(u_to_a(rbu), color="purple", lw=1.6, alpha=.8,
                           label="rule-based (marker only)")
            ax.axvline(u_to_a(ecu), color="darkgreen", lw=1.6, alpha=.8,
                       label="ECMS (marker only)")
            ax2 = ax.twinx()
            ax2.fill_between(grid, norm.pdf(np.arctanh(np.clip(grid, -.999, .999)), mu, sd),
                             alpha=.18, color="red")
            ax2.set_yticks([])
            ax.set_title(f"{arm} - {nm} (T={T:.1f} Nm, SoC={soc*100:.1f}%)", fontsize=10)
            ax.set_xlabel("action a"); ax.set_ylabel("min Q")
            if col == 0 and row == 0:
                ax.legend(fontsize=6.5, loc="lower left")
    fig.suptitle(f"Phase 6 section K -- Q(a) + actor density, CONTROL vs TREATMENT ({a.cycle})\n"
                 f"blue=LPS  orange=ASSIST  green=OFF   (benchmark/ECMS lines are DIAGNOSTIC MARKERS, never targets)",
                 fontsize=12)
    fig.tight_layout()
    p = out / f"q_landscape_ab_{a.cycle}.png"
    fig.savefig(p, dpi=110)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
