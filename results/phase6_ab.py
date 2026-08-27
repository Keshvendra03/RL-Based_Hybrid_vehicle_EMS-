"""
phase6_ab.py
============
Phase-6 CONTROL vs TREATMENT causal analysis (sections G, H, I, J, M).

Measures the full hypothesised chain:
    conditional coverage -> Q(OFF) -> actor P(OFF) -> behaviour -> SoC -> fuel

    python -m results.phase6_ab --cycle NEDC --out results/phase6

NO TRAINING. Matched states are identical between arms (same env, same
deterministic demand sequence), so ONLY the policy/critic differ.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch as th
from scipy.stats import norm

from src.env.ems_env import EMSEnv, U_MIN, U_MAX, ZB_MODEAWARE, _EPS_T
from src.env.powertrain import (_T_CUTOFF, _interp1d_linear, _w_EM_max_row,
                                _T_EM_max_arr, _THETA_EM)
from src.agents.targeted_exploration import _off_reachable, _a_off, decode_obs

EQF = {"NEDC": 0.2717, "FTP75": 0.4981}
TB = [(0, 15, "0-15"), (15, 30, "15-30"), (30, 35, "30-35"),
      (35, 50, "35-50"), (50, 75, "50-75"), (75, 1e9, ">75")]
SB = [(0.0, 0.40, "<40"), (0.40, 0.50, "40-50"), (0.50, 0.55, "50-55"), (0.55, 1.0, ">55")]
AMAP = "modeaware_gated"


def mode_of(a, T, w, dw):
    if T <= 0:
        return "REGEN" if T < 0 else "stop"
    from src.env.ems_env import map_action_to_u
    u = map_action_to_u(float(a), T, AMAP, w, dw)
    cap = max(_interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w)
              - abs(_THETA_EM * dw) - _EPS_T, 0.0)
    t_em = float(np.clip(u * T, -cap, cap))
    return "OFF" if (T - t_em) <= _T_CUTOFF else ("LPS" if t_em < 0 else
                                                 ("ASSIST" if t_em > 0 else "ONLY"))


# ---------------- section G: replay coverage ------------------------------- #
def coverage(model, P, label):
    rb = model.replay_buffer
    n = rb.size()
    obs = rb.observations[:n, 0, :]
    act = rb.actions[:n, 0, 0]
    T = obs[:, 2] * 150.0; w = obs[:, 0] * 300.0
    dw = obs[:, 1] * 60.0; soc = (obs[:, 4] + 1.0) / 2.0
    md = np.array([mode_of(act[i], T[i], w[i], dw[i]) for i in range(n)])
    P(f"\n  --- {label}: torque x SoC coverage (n={n:,}) ---")
    P(f"  {'T band':>8}{'SoC':>7}{'count':>9}{'%buf':>7}"
      f"{'OFF n':>8}{'OFF%':>7}{'ASST%':>7}{'LPS%':>7}{'feasOFF%':>10}")
    out = {}
    for lo, hi, tn in TB:
        for slo, shi, sn in SB:
            m = (T >= lo) & (T < hi) & (soc >= slo) & (soc < shi)
            c = int(m.sum())
            if c < 50:
                continue
            sub = md[m]
            idx = np.where(m)[0][:: max(1, c // 300)]
            feas = 100.0 * np.mean([_off_reachable(T[i], w[i], dw[i]) for i in idx])
            offn = int(np.sum(sub == "OFF"))
            out[f"{tn}|{sn}"] = dict(count=c, off_n=offn, off=100 * offn / c,
                                     assist=100 * np.mean(sub == "ASSIST"),
                                     lps=100 * np.mean(sub == "LPS"), feas=feas)
            P(f"  {tn:>8}{sn:>7}{c:>9,}{100*c/n:>6.1f}%{offn:>8,}"
              f"{100*offn/c:>6.1f}%{100*np.mean(sub=='ASSIST'):>6.1f}%"
              f"{100*np.mean(sub=='LPS'):>6.1f}%{feas:>9.1f}%")
    return out


# ---------------- matched states ------------------------------------------- #
def matched_states(cycle, kfb=2.5):
    env = EMSEnv(cycle, eq_factor=EQF[cycle], k_fb=kfb, lookahead=5, action_map=AMAP)
    obs, _ = env.reset()
    S = []
    while True:
        d = env._demand
        if d["T_MGB"] > _T_CUTOFF and d["w_MGB"] > 0:
            S.append((obs.copy(), copy.deepcopy(env), d["T_MGB"], d["w_MGB"], d["dw_MGB"],
                      (obs[4] + 1) / 2))
        obs, r, t, _, i = env.step(np.zeros(1, np.float32))
        if t:
            return S


def q_at(model, ob, acts):
    ot = th.as_tensor(np.repeat(ob.reshape(1, -1), len(acts), 0)).float().to(model.device)
    at = th.as_tensor(np.asarray(acts).reshape(-1, 1)).float().to(model.device)
    with th.no_grad():
        q = model.critic(ot, at)
    return np.minimum(q[0].cpu().numpy().ravel(), q[1].cpu().numpy().ravel())


def actor_at(model, ob):
    ot = th.as_tensor(ob.reshape(1, -1)).float().to(model.device)
    with th.no_grad():
        mu, ls, _ = model.actor.get_action_dist_params(ot)
    return float(mu.cpu().numpy().ravel()[0]), float(np.exp(ls.cpu().numpy().ravel()[0]))


def stats(x):
    x = np.asarray(x)
    return dict(mean=x.mean(), median=float(np.median(x)), std=x.std(ddof=1),
                p10=float(np.percentile(x, 10)), p25=float(np.percentile(x, 25)),
                p75=float(np.percentile(x, 75)), p90=float(np.percentile(x, 90)),
                pos=100.0 * np.mean(x > 0))


# ---------------- sections H, I, J ----------------------------------------- #
def qforensics(models, S, cycle, P):
    grid = np.linspace(-1, 1, 61)
    regions = [(15, 30, "15-30"), (30, 35, "30-35"), (35, 50, "35-50")]
    RES = {}
    for lo, hi, nm in regions:
        sel = [s for s in S if lo <= s[2] < hi and 0.40 <= s[5] < 0.50]
        if len(sel) < 10:
            sel = [s for s in S if lo <= s[2] < hi]
        step = max(1, len(sel) // 120)
        sel = sel[::step][:120]
        if not sel:
            continue
        RES[nm] = {}
        P(f"\n  === {nm} Nm (SoC 40-50 where available), n={len(sel)} matched states ===")
        for arm, m in models.items():
            dqa, dql, dr, poff, disp, cls_ = [], [], [], [], [], []
            for ob, sn, T, w, dw, soc in sel:
                aoff = _a_off(T, w, dw, AMAP)
                probe = min(1.0, aoff + 0.05)
                q = q_at(m, ob, [probe, 0.40, -0.50])
                dqa.append(q[0] - q[1]); dql.append(q[0] - q[2])
                mu, sd = actor_at(m, ob)
                poff.append(norm.sf((np.arctanh(np.clip(aoff, -.999999, .999999)) - mu) / sd))
                qg = q_at(m, ob, grid)
                disp.append(abs(np.tanh(mu) - grid[qg.argmax()]) / 2.0)
                e1 = copy.deepcopy(sn); _, r1, _, _, _ = e1.step(np.array([probe], np.float32))
                e2 = copy.deepcopy(sn); _, r2, _, _, _ = e2.step(np.array([0.40], np.float32))
                dr.append(r1 - r2)
            sa, sl, sr = stats(dqa), stats(dql), stats(dr)
            RES[nm][arm] = dict(dq_oa=sa, dq_ol=sl, dr=sr,
                                p_off=float(np.mean(poff)), disp=float(np.mean(disp)))
            P(f"    {arm:<10} dQ(OFF-ASSIST) mean={sa['mean']:+.4f} med={sa['median']:+.4f} "
              f"sd={sa['std']:.4f} p10={sa['p10']:+.4f} p90={sa['p90']:+.4f} >0:{sa['pos']:.0f}%")
            P(f"    {'':<10} dQ(OFF-LPS)    mean={sl['mean']:+.4f} med={sl['median']:+.4f} >0:{sl['pos']:.0f}%")
            P(f"    {'':<10} dr(OFF-ASSIST) mean={sr['mean']:+.4f} >0:{sr['pos']:.0f}%   "
              f"P(OFF)={100*np.mean(poff):.1f}%   actor-Q disp={np.mean(disp):.3f}")
            # section I classification
            dqa_a, dr_a = np.array(dqa), np.array(dr)
            c1 = 100 * np.mean((dr_a > 0) & (dqa_a > 0)); c2 = 100 * np.mean((dr_a > 0) & (dqa_a <= 0))
            c3 = 100 * np.mean((dr_a <= 0) & (dqa_a > 0)); c4 = 100 * np.mean((dr_a <= 0) & (dqa_a <= 0))
            P(f"    {'':<10} r/Q agree: [r=OFF,Q=OFF]={c1:.0f}%  "
              f"**[r=OFF,Q=ASSIST]={c2:.0f}%**  [r=ASST,Q=OFF]={c3:.0f}%  [r=ASST,Q=ASST]={c4:.0f}%")
            RES[nm][arm]["conflict_r_off_q_assist"] = float(c2)
    return RES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", default="NEDC")
    ap.add_argument("--out", default="results/phase6")
    ap.add_argument("--control", default="models_p5s0_k2.5")
    ap.add_argument("--treatment", default="models_p6_trt_N0")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    fh = open(out / f"phase6_forensics_{a.cycle}.txt", "w", encoding="utf-8")
    P = lambda s: (print(s), fh.write(s + "\n"))
    from stable_baselines3 import SAC

    P(f"PHASE 6 A/B FORENSICS -- {a.cycle}")
    P(f"CONTROL  : {a.control}")
    P(f"TREATMENT: {a.treatment}")

    models, cov = {}, {}
    for arm, d in (("CONTROL", a.control), ("TREATMENT", a.treatment)):
        m = SAC.load(f"{d}/{a.cycle}/sac_ems_best")
        models[arm] = m
        try:
            m2 = SAC.load(f"{d}/{a.cycle}/sac_ems_best")
            m2.load_replay_buffer(f"{d}/{a.cycle}/replay_buffer.pkl")
            P(f"\n{'='*104}\nSECTION G -- REPLAY COVERAGE, {arm}\n{'='*104}")
            cov[arm] = coverage(m2, P, arm)
            del m2
        except Exception as e:
            P(f"  {arm}: replay unavailable ({e})")

    if "CONTROL" in cov and "TREATMENT" in cov:
        P(f"\n  *** KEY CELLS: OFF coverage change ***")
        P(f"  {'cell':>16}{'CTL n':>9}{'CTL OFF%':>10}{'TRT n':>9}{'TRT OFF%':>10}{'delta pp':>10}")
        for k in ["15-30|40-50", "30-35|40-50", "30-35|50-55", "15-30|50-55"]:
            c, t = cov["CONTROL"].get(k), cov["TREATMENT"].get(k)
            if c and t:
                P(f"  {k:>16}{c['count']:>9,}{c['off']:>9.1f}%{t['count']:>9,}"
                  f"{t['off']:>9.1f}%{t['off']-c['off']:>+10.1f}")

    P(f"\n{'='*104}\nSECTIONS H/I/J -- MATCHED-STATE Q, REWARD-vs-Q, ACTOR\n{'='*104}")
    S = matched_states(a.cycle)
    R = qforensics(models, S, a.cycle, P)
    json.dump(dict(coverage=cov, q=R), open(out / f"phase6_{a.cycle}.json", "w"),
              indent=2, default=float)
    P(f"\n[saved] {out}/phase6_{a.cycle}.json")
    fh.close()


if __name__ == "__main__":
    main()
