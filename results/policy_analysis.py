"""
policy_analysis.py
==================
Phase-2 sections 19 + 20.

  section 19  Q-landscape over the FULL action range at EXTENDED operating
              points -- low/med/high torque, high/low SoC, acceleration,
              cruising, braking -- with plots.
  section 20  State-conditioned action analysis: the OFF threshold a_off(T_MGB)
              versus the policy's ACTUAL learned law a_policy(state), plus
              T_MGB-vs-action, T_MGB-vs-engine-torque, T_MGB-vs-mode,
              SoC-vs-action and SoC-vs-mode plots.

    python -m results.policy_analysis --checkpoint models_expD_g20/NEDC/sac_ems_best \
        --cycle NEDC --eq-factor 0.2717 --k-fb 1.656 --out results/analysis

No training. Physics untouched.
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

from src.env.ems_env import EMSEnv, U_MIN, U_MAX
from src.env.powertrain import _T_CUTOFF


def rollout(model, env):
    """Deterministic rollout; record the full state->action->mode law."""
    obs, _ = env.reset()
    R = []
    while True:
        a, _ = model.predict(obs, deterministic=True)
        av = float(np.asarray(a).reshape(-1)[0])
        d = dict(env._demand)
        snap = copy.deepcopy(env)
        obs, r, term, _, info = env.step(a)
        cls = ("stop" if info["mode"] == "stop" else
               "regen" if info["mode"] == "regen" else
               "OFF" if info["T_CE_cmd"] <= _T_CUTOFF else
               "assist" if info["mode"] == "assist" else
               "lps" if info["mode"] == "lps_gen" else "only")
        R.append(dict(a=av, T=d["T_MGB"], w=d["w_MGB"], v=d["v"], dw=d["dw_MGB"],
                      soc=info["soc"], u=info["u"], t_ce=info["T_CE_cmd"],
                      t_em=info["T_EM_cmd"], cls=cls, r=r, snap=snap, obs=None))
        if term:
            return R


def q_sweep(model, snap, grid):
    obs = snap._last_obs
    ot = th.as_tensor(np.repeat(obs.reshape(1, -1), len(grid), 0)).float().to(model.device)
    at = th.as_tensor(grid.reshape(-1, 1)).float().to(model.device)
    with th.no_grad():
        qs = model.critic(ot, at)
    q1 = qs[0].cpu().numpy().ravel(); q2 = qs[1].cpu().numpy().ravel()
    rew, modes = [], []
    for av in grid:
        e2 = copy.deepcopy(snap)
        act = np.array([av], np.float32)
        t_ce, _, _, md = e2._action_to_torques(act)
        modes.append("OFF" if (md != "regen" and t_ce <= _T_CUTOFF) else md)
        _, rr, _, _, _ = e2.step(act)
        rew.append(rr)
    return q1, q2, np.minimum(q1, q2), np.array(rew), modes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cycle", default="NEDC")
    p.add_argument("--eq-factor", type=float, default=0.2717)
    p.add_argument("--k-fb", type=float, default=1.656)
    p.add_argument("--action-map", default="linear")
    p.add_argument("--out", default="results/analysis")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import SAC
    model = SAC.load(a.checkpoint)
    od = int(model.observation_space.shape[0]); la = 0 if od <= 16 else od - 15
    env = EMSEnv(a.cycle, eq_factor=a.eq_factor, k_fb=a.k_fb,
                 action_map=a.action_map, lookahead=la)
    R = rollout(model, env)
    mv = [x for x in R if x["cls"] != "stop"]
    tr = [x for x in mv if x["T"] > _T_CUTOFF]

    # ---------------- section 19: extended probes ----------------
    socs = np.array([x["soc"] for x in tr])
    probes = {
        "low_torque":   min(tr, key=lambda x: abs(x["T"] - 10)),
        "med_torque":   min(tr, key=lambda x: abs(x["T"] - 25)),
        "high_torque":  min(tr, key=lambda x: abs(x["T"] - 55)),
        "high_soc":     max(tr, key=lambda x: x["soc"]),
        "low_soc":      min(tr, key=lambda x: x["soc"]),
        "acceleration": max(tr, key=lambda x: x["dw"]),
        "cruising":     min(tr, key=lambda x: abs(x["dw"]) + abs(x["v"] - 22)),
    }
    brk = [x for x in mv if x["cls"] == "regen"]
    if brk:
        probes["braking"] = min(brk, key=lambda x: x["dw"])

    grid = np.linspace(-1, 1, 41)
    print(f"\n{'='*100}\nSECTION 19 -- Q-LANDSCAPE, EXTENDED PROBES  ({a.cycle})\n{'='*100}")
    print(f"{'probe':<14}{'T_MGB':>8}{'SoC%':>7}{'dw':>8}{'v':>7}"
          f"{'argmaxQ':>9}{'argmaxR':>9}{'OFF-ASSIST(R)':>15}{'(minQ)':>9}{'agree':>7}{'Qspan':>8}")
    fig1, axes = plt.subplots(2, 4, figsize=(22, 9))
    for i, (nm, st) in enumerate(probes.items()):
        q1, q2, mq, rew, modes = q_sweep(model, st["snap"], grid)
        offm = np.array([m == "OFF" for m in modes])
        asm = np.array([m == "assist" for m in modes])
        dR = (rew[offm].max() - rew[asm].max()) if (offm.any() and asm.any()) else np.nan
        dQ = (mq[offm].max() - mq[asm].max()) if (offm.any() and asm.any()) else np.nan
        agree = "YES" if (not np.isnan(dR) and np.sign(dR) == np.sign(dQ)) else "NO"
        print(f"{nm:<14}{st['T']:>8.1f}{st['soc']*100:>7.1f}{st['dw']:>8.1f}{st['v']:>7.1f}"
              f"{grid[mq.argmax()]:>9.2f}{grid[rew.argmax()]:>9.2f}"
              f"{dR:>15.4f}{dQ:>9.3f}{agree:>7}{mq.max()-mq.min():>8.3f}")
        ax = axes.ravel()[i]
        ax.plot(grid, mq, label="min Q", lw=2)
        ax2 = ax.twinx(); ax2.plot(grid, rew, color="tab:red", ls="--", label="true reward")
        if offm.any():
            ax.axvspan(grid[offm].min(), 1.0, alpha=0.12, color="green")
        ax.set_title(f"{nm}\nT={st['T']:.0f}Nm SoC={st['soc']*100:.0f}%", fontsize=9)
        ax.set_xlabel("action a"); ax.set_ylabel("min Q"); ax2.set_ylabel("reward", color="tab:red")
    for j in range(len(probes), 8):
        axes.ravel()[j].axis("off")
    fig1.suptitle(f"Q-landscape vs true reward, green = engine-OFF region ({a.cycle})")
    fig1.tight_layout(); fig1.savefig(out / f"q_landscape_{a.cycle}.png", dpi=110)
    print(f"[saved] {out / f'q_landscape_{a.cycle}.png'}")

    # ---------------- section 20: state-conditioned law ----------------
    T = np.array([x["T"] for x in tr]); A = np.array([x["a"] for x in tr])
    S = np.array([x["soc"] for x in tr]); TCE = np.array([x["t_ce"] for x in tr])
    C = [x["cls"] for x in tr]
    a_off = 2 * ((1 - _T_CUTOFF / T) - U_MIN) / (U_MAX - U_MIN) - 1
    cols = {"OFF": "tab:green", "assist": "tab:orange", "lps": "tab:blue",
            "only": "tab:red", "regen": "tab:gray"}
    cvec = [cols.get(c, "k") for c in C]

    print(f"\n{'='*100}\nSECTION 20 -- STATE-CONDITIONED ACTION LAW\n{'='*100}")
    above = A >= a_off
    print(f"  steps where policy action >= a_off (i.e. commands OFF): "
          f"{above.sum()}/{len(A)} ({100*above.mean():.1f}%)")
    print(f"  mean margin (a_policy - a_off): {np.mean(A - a_off):+.4f}  "
          f"(negative => policy sits BELOW the OFF threshold)")
    for lo, hi in [(0, 15), (15, 30), (30, 50), (50, 200)]:
        m = (T >= lo) & (T < hi)
        if m.sum():
            print(f"    T in [{lo:3d},{hi:3d}) n={m.sum():4d}: mean a={A[m].mean():+.3f} "
                  f"mean a_off={a_off[m].mean():+.3f} margin={np.mean(A[m]-a_off[m]):+.3f} "
                  f"OFF%={100*np.mean([c=='OFF' for c,mm in zip(C,m) if mm]):.1f}")

    fig2, ax = plt.subplots(2, 3, figsize=(18, 9))
    ax[0, 0].scatter(T, A, c=cvec, s=8); ax[0, 0].plot(np.sort(T), a_off[np.argsort(T)],
                     "k--", lw=2, label="a_off(T) threshold")
    ax[0, 0].set_xlabel("T_MGB [Nm]"); ax[0, 0].set_ylabel("action a"); ax[0, 0].legend()
    ax[0, 0].set_title("T_MGB vs action (colour = mode)")
    ax[0, 1].scatter(T, TCE, c=cvec, s=8); ax[0, 1].axhline(_T_CUTOFF, color="k", ls="--")
    ax[0, 1].set_xlabel("T_MGB [Nm]"); ax[0, 1].set_ylabel("T_CE commanded [Nm]")
    ax[0, 1].set_title("T_MGB vs engine torque (dashed = cutoff)")
    for c in set(C):
        m = [cc == c for cc in C]
        ax[0, 2].scatter(T[m], np.full(sum(m), c), s=6, c=cols.get(c, "k"))
    ax[0, 2].set_xlabel("T_MGB [Nm]"); ax[0, 2].set_title("T_MGB vs mode")
    ax[1, 0].scatter(S * 100, A, c=cvec, s=8)
    ax[1, 0].set_xlabel("SoC [%]"); ax[1, 0].set_ylabel("action a"); ax[1, 0].set_title("SoC vs action")
    for c in set(C):
        m = [cc == c for cc in C]
        ax[1, 1].scatter(S[m] * 100, np.full(sum(m), c), s=6, c=cols.get(c, "k"))
    ax[1, 1].set_xlabel("SoC [%]"); ax[1, 1].set_title("SoC vs mode")
    ax[1, 2].scatter(T, A - a_off, c=cvec, s=8); ax[1, 2].axhline(0, color="k", ls="--")
    ax[1, 2].set_xlabel("T_MGB [Nm]"); ax[1, 2].set_ylabel("a_policy - a_off")
    ax[1, 2].set_title("margin above the OFF threshold (>0 => engine off)")
    fig2.suptitle(f"State-conditioned policy law ({a.cycle})")
    fig2.tight_layout(); fig2.savefig(out / f"policy_law_{a.cycle}.png", dpi=110)
    print(f"[saved] {out / f'policy_law_{a.cycle}.png'}")


if __name__ == "__main__":
    main()
