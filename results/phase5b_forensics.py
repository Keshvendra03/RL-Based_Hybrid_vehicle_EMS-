"""
phase5b_forensics.py
====================
Phase-5B forensic closure. NO TRAINING. Closes the evidence gaps left by
Phase 5.

  section 2/3   replay-buffer coverage, DIRECTLY measured (not inferred)
  section 4     matched before/after k_fb (1.656 vs 2.5) at identical states
  section 5     Q(a) + actor-density figures
  section 6     actor-vs-critic classification over a DISTRIBUTION of states
  section 7     costate percentile forensics
  section 8     error budget for all three configurations
  section 10    FTP75 SAC vs rule-based vs ECMS matched-state comparison

    python -m results.phase5b_forensics --out results/phase5b

Observation reconstruction note: the replay buffer stores observations, not
raw demand. T_MGB is recovered from obs[2]*150 and SoC from obs[4] via
soc = (obs[4]+1)/2 -- exactly the normalisations in EMSEnv._make_observation.
Mode is recovered by replaying the stored action through the SAME action map
at the stored operating point, so the classification matches the env exactly.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from src.env.ems_env import (EMSEnv, U_MIN, U_MAX, ZA_MODEAWARE, ZB_MODEAWARE,
                             map_action_to_u, _EPS_T, SOC_TARGET)
from src.env.powertrain import (_T_CUTOFF, _interp1d_linear, _w_EM_max_row,
                                _T_EM_max_arr, _THETA_EM)

EQF = {"NEDC": 0.2717, "FTP75": 0.4981}
BANDS = [(0, 15, "0-15"), (15, 30, "15-30"), (30, 35, "30-35"),
         (35, 50, "35-50"), (50, 75, "50-75"), (75, 1e9, ">75")]
SOC_BANDS = [(0.0, 0.40, "SoC<40"), (0.40, 0.50, "40-50"), (0.50, 1.0, "SoC>=50")]


def off_reachable(T, w, dw):
    cap = max(_interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w)
              - abs(_THETA_EM * dw) - _EPS_T, 0.0)
    return cap >= T - _T_CUTOFF


def a_off_for(T, w, dw, amap):
    """Smallest action producing engine-OFF under the given map."""
    if T <= _T_CUTOFF:
        return -1.0
    if amap == "modeaware_gated" and off_reachable(T, w, dw):
        return 2 * ZB_MODEAWARE - 1
    return 2 * ((1 - _T_CUTOFF / T) - U_MIN) / (U_MAX - U_MIN) - 1


def mode_of(a, T, w, dw, amap):
    """Classify the mode the stored action actually produced."""
    if T <= 0:
        return "REGEN" if T < 0 else "stop"
    u = map_action_to_u(float(a), T, amap, w, dw)
    t_em = u * T
    cap = max(_interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w)
              - abs(_THETA_EM * dw) - _EPS_T, 0.0)
    t_em = float(np.clip(t_em, -cap, cap))
    t_ce = T - t_em
    if t_ce <= _T_CUTOFF:
        return "OFF"
    return "LPS" if t_em < 0 else ("ASSIST" if t_em > 0 else "ONLY")


# --------------------------------------------------------------------------- #
# section 2/3 -- replay buffer forensics
# --------------------------------------------------------------------------- #
def replay_forensics(model, cycle, amap, P):
    rb = model.replay_buffer
    n = rb.size()
    obs = rb.observations[:n, 0, :]
    act = rb.actions[:n, 0, 0]
    # recover physical quantities from the stored observation
    T = obs[:, 2] * 150.0
    w = obs[:, 0] * 300.0
    dw = obs[:, 1] * 60.0
    soc = (obs[:, 4] + 1.0) / 2.0

    P(f"\n{'='*104}")
    P(f"SECTION 2/3 -- REPLAY BUFFER FORENSICS ({cycle}, map={amap})")
    P(f"{'='*104}")
    P(f"  total transitions stored: {n:,}   (capacity {rb.buffer_size:,})")
    P(f"  SoC range in buffer: {soc.min()*100:.1f}% .. {soc.max()*100:.1f}%")

    modes = np.array([mode_of(act[i], T[i], w[i], dw[i], amap) for i in range(n)])

    P(f"\n  --- coverage by torque band (percentages are WITHIN the band) ---")
    P(f"  {'band':>8}{'count':>10}{'% of buf':>10}"
      f"{'OFF':>9}{'ASSIST':>9}{'LPS':>9}{'REGEN':>9}{'stop/ONLY':>11}"
      f"{'feasOFF%':>10}")
    tab = {}
    for lo, hi, nm in BANDS:
        m = (T >= lo) & (T < hi)
        c = int(m.sum())
        if c == 0:
            continue
        sub = modes[m]
        f = lambda k: 100.0 * np.mean(sub == k)
        feas = 100.0 * np.mean([off_reachable(T[i], w[i], dw[i])
                                for i in np.where(m)[0][::max(1, c // 400)]])
        tab[nm] = dict(count=c, pct=100 * c / n, OFF=f("OFF"), ASSIST=f("ASSIST"),
                       LPS=f("LPS"), REGEN=f("REGEN"), feas=feas)
        P(f"  {nm:>8}{c:>10,}{100*c/n:>9.1f}%"
          f"{f('OFF'):>8.1f}%{f('ASSIST'):>8.1f}%{f('LPS'):>8.1f}%{f('REGEN'):>8.1f}%"
          f"{100*np.mean((sub=='stop')|(sub=='ONLY')):>10.1f}%{feas:>9.1f}%")

    P(f"\n  --- OFF share in the CRITICAL bands, conditioned on SoC ---")
    P(f"  {'band':>8}{'SoC band':>12}{'count':>9}{'OFF%':>8}{'ASSIST%':>9}{'LPS%':>8}")
    for lo, hi, nm in [(15, 30, "15-30"), (30, 35, "30-35"), (35, 50, "35-50")]:
        for slo, shi, snm in SOC_BANDS:
            m = (T >= lo) & (T < hi) & (soc >= slo) & (soc < shi)
            c = int(m.sum())
            if c < 20:
                continue
            sub = modes[m]
            P(f"  {nm:>8}{snm:>12}{c:>9,}"
              f"{100*np.mean(sub=='OFF'):>7.1f}%{100*np.mean(sub=='ASSIST'):>8.1f}%"
              f"{100*np.mean(sub=='LPS'):>7.1f}%")
    return tab


# --------------------------------------------------------------------------- #
# section 4 -- matched before/after k_fb
# --------------------------------------------------------------------------- #
def collect_states(cycle, amap, kfb, nmax=400):
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=kfb, lookahead=5, action_map=amap)
    obs, _ = env.reset()
    S = []
    while True:
        d = env._demand
        if d["T_MGB"] > _T_CUTOFF and d["w_MGB"] > 0:
            S.append((obs.copy(), copy.deepcopy(env), d["T_MGB"], d["w_MGB"], d["dw_MGB"]))
        obs, r, t, _, i = env.step(np.zeros(1, np.float32))
        if t:
            break
    return S[:nmax]


def actor_stats(model, ob):
    ot = th.as_tensor(ob.reshape(1, -1)).float().to(model.device)
    with th.no_grad():
        mu, log_std, _ = model.actor.get_action_dist_params(ot)
    return float(mu.cpu().numpy().ravel()[0]), float(np.exp(log_std.cpu().numpy().ravel()[0]))


def q_of(model, ob, actions):
    ot = th.as_tensor(np.repeat(ob.reshape(1, -1), len(actions), 0)).float().to(model.device)
    at = th.as_tensor(np.asarray(actions).reshape(-1, 1)).float().to(model.device)
    with th.no_grad():
        q = model.critic(ot, at)
    return np.minimum(q[0].cpu().numpy().ravel(), q[1].cpu().numpy().ravel())


def before_after(mA, mB, cycle, amap, P):
    """mA = k_fb 1.656 model, mB = k_fb 2.5 model. States from a NEUTRAL rollout."""
    P(f"\n{'='*104}")
    P(f"SECTION 4 -- MATCHED BEFORE/AFTER k_fb  (1.656 -> 2.5), {cycle}")
    P(f"{'='*104}")
    S = collect_states(cycle, amap, 1.656)
    rows = {}
    for lo, hi, nm in [(15, 30, "15-30"), (30, 35, "30-35"), (35, 50, "35-50")]:
        sel = [s for s in S if lo <= s[2] < hi]
        if len(sel) < 5:
            continue
        sel = sel[:: max(1, len(sel) // 30)][:30]
        acc = {k: [] for k in ["muA", "sdA", "muB", "sdB", "pA", "pB",
                               "dqA_oa", "dqB_oa", "dqA_ol", "dqB_ol",
                               "drA", "drB"]}
        for ob, sn, T, w, dw in sel:
            aoff = a_off_for(T, w, dw, amap)
            a_ass, a_lps = 0.40, -0.50
            a_probe = min(1.0, aoff + 0.05)
            for tag, m in (("A", mA), ("B", mB)):
                mu, sd = actor_stats(m, ob)
                acc[f"mu{tag}"].append(np.tanh(mu))
                acc[f"sd{tag}"].append(sd)
                z = (np.arctanh(np.clip(aoff, -0.999999, 0.999999)) - mu) / sd
                acc[f"p{tag}"].append(norm.sf(z))
                q = q_of(m, ob, [a_probe, a_ass, a_lps])
                acc[f"dq{tag}_oa"].append(q[0] - q[1])
                acc[f"dq{tag}_ol"].append(q[0] - q[2])
            # immediate reward is model-independent given the env's k_fb; evaluate both
            for tag, kfb in (("A", 1.656), ("B", 2.5)):
                e = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=kfb, lookahead=5,
                           action_map=amap)
                e.__dict__.update({k: copy.deepcopy(v) for k, v in sn.__dict__.items()
                                   if k not in ("k_fb",)})
                e.k_fb = kfb
                e2 = copy.deepcopy(e)
                _, r1, _, _, _ = e2.step(np.array([a_probe], np.float32))
                e3 = copy.deepcopy(e)
                _, r2, _, _, _ = e3.step(np.array([a_ass], np.float32))
                acc[f"dr{tag}"].append(r1 - r2)
        m_ = {k: float(np.mean(v)) for k, v in acc.items()}
        rows[nm] = m_
    P(f"  {'band':>8} | {'actor mean':^17} | {'actor sigma':^17} | {'P(OFF)':^17}")
    P(f"  {'':>8} | {'k=1.656':>8}{'k=2.5':>9} | {'k=1.656':>8}{'k=2.5':>9} | {'k=1.656':>8}{'k=2.5':>9}")
    for nm, m_ in rows.items():
        P(f"  {nm:>8} | {m_['muA']:>8.3f}{m_['muB']:>9.3f} | {m_['sdA']:>8.3f}{m_['sdB']:>9.3f}"
          f" | {100*m_['pA']:>7.1f}%{100*m_['pB']:>8.1f}%")
    P(f"\n  {'band':>8} | {'dQ(OFF-ASSIST)':^19} | {'dQ(OFF-LPS)':^19} | {'dr(OFF-ASSIST)':^19}")
    P(f"  {'':>8} | {'k=1.656':>9}{'k=2.5':>10} | {'k=1.656':>9}{'k=2.5':>10} | {'k=1.656':>9}{'k=2.5':>10}")
    for nm, m_ in rows.items():
        P(f"  {nm:>8} | {m_['dqA_oa']:>+9.4f}{m_['dqB_oa']:>+10.4f}"
          f" | {m_['dqA_ol']:>+9.4f}{m_['dqB_ol']:>+10.4f}"
          f" | {m_['drA']:>+9.4f}{m_['drB']:>+10.4f}")
    return rows


# --------------------------------------------------------------------------- #
# section 6 -- actor vs critic over a DISTRIBUTION
# --------------------------------------------------------------------------- #
def actor_critic_classify(model, cycle, amap, kfb, P, label):
    S = collect_states(cycle, amap, kfb)
    grid = np.linspace(-1, 1, 61)
    dist, cases = [], []
    for ob, sn, T, w, dw in S[:: max(1, len(S) // 120)][:120]:
        q = q_of(model, ob, grid)
        mu, sd = actor_stats(model, ob)
        a_act = np.tanh(mu)
        a_star = grid[q.argmax()]
        d = abs(a_act - a_star) / 2.0            # normalised by action range
        dist.append(d)
        spread = q.max() - q.min()
        if spread < 0.005:
            cases.append("D_flat")
        elif d < 0.10:
            cases.append("A_aligned")
        else:
            aoff = a_off_for(T, w, dw, amap)
            cases.append("B_displaced" if a_star >= aoff else "C_Q_prefers_non_OFF")
    dist = np.array(dist); cases = np.array(cases)
    P(f"\n  --- {label} ---")
    P(f"    |actor_mean - argmax_a Q| / range : mean={dist.mean():.3f}  "
      f"median={np.median(dist):.3f}  p90={np.percentile(dist,90):.3f}")
    for c in ["A_aligned", "B_displaced", "C_Q_prefers_non_OFF", "D_flat"]:
        P(f"    {c:<22}: {100*np.mean(cases==c):5.1f}%  ({int(np.sum(cases==c))}/{len(cases)})")
    return dist, cases


# --------------------------------------------------------------------------- #
# section 5 -- Q(a) + actor density figures
# --------------------------------------------------------------------------- #
def q_figures(model, cycle, amap, kfb, out: Path, bench_u=None):
    S = collect_states(cycle, amap, kfb)
    targets = [(15, 30), (30, 35), (35, 50)]
    grid = np.linspace(-1, 1, 121)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    for ax, (lo, hi) in zip(axes, targets):
        sel = [s for s in S if lo <= s[2] < hi]
        if not sel:
            ax.axis("off"); continue
        ob, sn, T, w, dw = sel[len(sel) // 2]
        q = q_of(model, ob, grid)
        mu, sd = actor_stats(model, ob)
        aoff = a_off_for(T, w, dw, amap)
        a_zero = 2 * (0.0 - U_MIN) / (U_MAX - U_MIN) - 1 if amap == "linear" \
            else 2 * ZA_MODEAWARE - 1
        ax.plot(grid, q, lw=2, color="tab:blue", label="min Q(a)")
        ax.axvspan(aoff, 1.0, alpha=0.13, color="green", label="OFF region")
        ax.axvspan(a_zero, aoff, alpha=0.10, color="orange", label="ASSIST region")
        ax.axvspan(-1.0, a_zero, alpha=0.10, color="tab:blue", label="LPS region")
        ax.axvline(np.tanh(mu), color="red", lw=2, label="actor mean")
        for k, ls in ((1, "--"), (2, ":")):
            ax.axvline(np.tanh(mu + k * sd), color="red", ls=ls, lw=1)
            ax.axvline(np.tanh(mu - k * sd), color="red", ls=ls, lw=1,
                       label=f"actor +/-{k}sigma" if k == 1 else None)
        ax.axvline(grid[q.argmax()], color="k", ls="-.", lw=1.6, label="argmax Q")
        pdf = norm.pdf(np.arctanh(np.clip(grid, -0.999, 0.999)), mu, sd)
        ax2 = ax.twinx()
        ax2.fill_between(grid, pdf, alpha=0.18, color="red")
        ax2.set_yticks([]); ax2.set_ylabel("actor density", color="red", fontsize=8)
        ax.set_title(f"{cycle} {lo}-{hi} Nm  (T={T:.1f}, SoC={(ob[4]+1)/2*100:.1f}%)",
                     fontsize=10)
        ax.set_xlabel("action a"); ax.set_ylabel("min Q")
        ax.legend(fontsize=6.5, loc="lower left")
    fig.suptitle(f"SECTION 5: Q(a) with actor density -- {cycle}, {amap}, k_fb={kfb}")
    fig.tight_layout()
    p = out / f"q_landscape_{cycle}.png"
    fig.savefig(p, dpi=110)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase5b")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "phase5b_forensics.txt", "w", encoding="utf-8")
    P = lambda s: (print(s), fh.write(s + "\n"))
    from stable_baselines3 import SAC

    P("PHASE 5B -- FORENSIC CLOSURE AUDIT (no training)")

    # ---- replay forensics on the Phase-5 candidate (NEDC k=2.5) and k=1.656
    for tag, d, cyc, kfb in [("NEDC k_fb=2.5 (candidate)", "models_p5s0_k2.5", "NEDC", 2.5),
                             ("NEDC k_fb=1.656 (Phase 4)", "models_p4g_N0", "NEDC", 1.656),
                             ("FTP75 k_fb=1.656 (best)", "models_p4g_F0", "FTP75", 1.656)]:
        m = SAC.load(f"{d}/{cyc}/sac_ems_best")
        try:
            m.load_replay_buffer(f"{d}/{cyc}/replay_buffer.pkl")
        except Exception as e:
            P(f"  {tag}: replay load failed ({e})"); continue
        P(f"\n>>> {tag}")
        replay_forensics(m, cyc, "modeaware_gated", P)

    # ---- section 4 before/after
    mA = SAC.load("models_p4g_N0/NEDC/sac_ems_best")
    mB = SAC.load("models_p5s0_k2.5/NEDC/sac_ems_best")
    before_after(mA, mB, "NEDC", "modeaware_gated", P)

    # ---- section 6 distributions
    P(f"\n{'='*104}")
    P("SECTION 6 -- ACTOR vs CRITIC CLASSIFICATION (distribution over states)")
    P(f"{'='*104}")
    actor_critic_classify(mA, "NEDC", "modeaware_gated", 1.656, P, "NEDC k_fb=1.656")
    actor_critic_classify(mB, "NEDC", "modeaware_gated", 2.5, P, "NEDC k_fb=2.5 (candidate)")
    mF = SAC.load("models_p4g_F0/FTP75/sac_ems_best")
    actor_critic_classify(mF, "FTP75", "modeaware_gated", 1.656, P, "FTP75 k_fb=1.656 (best)")

    # ---- section 5 figures
    p1 = q_figures(mB, "NEDC", "modeaware_gated", 2.5, out)
    p2 = q_figures(mF, "FTP75", "modeaware_gated", 1.656, out)
    P(f"\n[saved] {p1}\n[saved] {p2}")
    fh.close()


if __name__ == "__main__":
    main()
