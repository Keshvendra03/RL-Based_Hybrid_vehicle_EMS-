"""
PHASE 13 STAGE A  --  gamma-sweep diagnostics (NO TRAINING).

For every (gamma, seed): critic response, Bellman-decomposition AT THAT GAMMA,
actor response, vehicle response, effective-horizon table, Q-argmax stability.

Reads results/phase13/stage_a/gamma/g{XX}/seed{s}/sac_ems_{50k,100k,150k,best}.zip
Writes results/phase13/stage_a/data/p13a_gamma_diagnostics.json + console.
"""
import copy, json, warnings
from pathlib import Path
import numpy as np
import torch as th
warnings.filterwarnings("ignore")

from stable_baselines3 import SAC
from src.env.ems_env import EMSEnv
from src.env.powertrain import _Q_BT_0
from results.evaluate_policy import evaluate

G = Path("results/phase13/stage_a/gamma")
OUT = Path("results/phase13/stage_a/data"); OUT.mkdir(parents=True, exist_ok=True)
GAMMAS = [0.20, 0.50, 0.90, 0.98]
SEEDS = [0, 1, 2]
CKPTS = ["sac_ems_50k", "sac_ems_100k", "sac_ems_150k", "sac_ems_best"]
CONTROL0 = "models_p5s0_k2.5/NEDC"
EQF, KFB, AMAP, LA = 0.2717, 2.5, "modeaware_gated", 5
N_A = 161
GRID_A = np.linspace(-1.0, 1.0, N_A)
BANDS = [("25-30", 25, 30), ("30-35", 30, 35), ("35-50", 35, 50)]
N_PER_BAND = 8
K = 24
DEEP_WIN = 15.0


def band_of(T):
    for n, lo, hi in BANDS:
        if lo <= T < hi:
            return n
    return None


def make_env():
    return EMSEnv("NEDC", eq_factor=EQF, k_fb=KFB, action_map=AMAP, lookahead=LA, clip_eq_eff=True)


def collect_states(m0):
    env = make_env(); obs, _ = env.reset()
    b = {n: [] for n, _, _ in BANDS}
    while True:
        d = env._demand; w, T = d["w_MGB"], d["T_MGB"]; bb = band_of(T)
        if bb and w > 0 and len(b[bb]) < 80:
            b[bb].append(dict(env=copy.deepcopy(env), obs=obs.copy(), band=bb, T=T,
                              soc=env._Q_BT / _Q_BT_0))
        a, _ = m0.predict(obs, deterministic=True)
        obs, r, term, _, info = env.step(a)
        if term:
            break
    out = []
    for n in b:
        if b[n]:
            idx = np.linspace(0, len(b[n]) - 1, min(N_PER_BAND, len(b[n]))).astype(int)
            out += [b[n][i] for i in idx]
    return out


def cache_physics(states):
    for st in states:
        E = st["env"]; rr, tce, ng = [], [], []
        for a in GRID_A:
            cp = copy.deepcopy(E)
            _, r, term, _, info = cp.step(np.array([a], np.float32))
            rr.append(r); tce.append(info["T_CE_cmd"]); ng.append((cp._last_obs.copy(), bool(term)))
        st["r_grid"] = np.array(rr); st["tce_grid"] = np.array(tce); st["ng"] = ng
        st["iR"] = int(np.argmax(st["r_grid"]))
        st["tce_max"] = float(st["tce_grid"].max())
        st["deep_mask"] = st["tce_grid"] >= (st["tce_max"] - DEEP_WIN)
    return states


@th.no_grad()
def Vt(model, s_next, alpha, k=K):
    O = th.as_tensor(np.repeat(np.asarray(s_next)[None, :], k, 0), dtype=th.float32)
    a_s, logp = model.actor.action_log_prob(O)
    q1, q2 = model.critic_target(O, a_s)
    v = th.minimum(q1, q2).squeeze(1) - alpha * logp.reshape(-1)
    return float(v.mean().item())


@th.no_grad()
def critic_and_bellman(model, states, gamma):
    alpha = float(th.exp(model.log_ent_coef.detach()))
    dr_l, dfut_l, dtgt_l, dQ_l, residR_l, residDeep_l, qdeep_l = ([] for _ in range(7))
    tq_l, tr_l, dTRQ_l, tpi_l, dTQpi_l = ([] for _ in range(5))
    for st in states:
        O = th.as_tensor(np.repeat(st["obs"][None, :], N_A, 0), dtype=th.float32)
        q1, q2 = model.critic(O, th.as_tensor(GRID_A[:, None], dtype=th.float32))
        minq = np.minimum(q1.squeeze(1).numpy(), q2.squeeze(1).numpy())
        iQ = int(np.argmax(minq)); iR = st["iR"]
        a_pi = float(model.actor(th.as_tensor(st["obs"][None, :], dtype=th.float32),
                                 deterministic=True).squeeze().item())
        i_pi = int(np.argmin(np.abs(GRID_A - a_pi)))
        # Bellman targets at a_part(=i_pi) and a_R*(=iR)
        sP, dP = st["ng"][i_pi]; sR, dR = st["ng"][iR]
        VP = 0.0 if dP else Vt(model, sP, alpha)
        VR = 0.0 if dR else Vt(model, sR, alpha)
        yP = st["r_grid"][i_pi] + gamma * VP
        yR = st["r_grid"][iR] + gamma * VR
        dr = float(st["r_grid"][iR] - st["r_grid"][i_pi])
        dfut = float(gamma * VR - gamma * VP)
        dtgt = float(yR - yP)
        dQ = float(minq[iR] - minq[i_pi])
        residR = float(minq[iR] - yR)
        dm = st["deep_mask"]
        # residual at deep-LPS actions: y for each deep action needs its own s'
        rd = []
        for j in np.where(dm)[0][::3]:
            sj, dj = st["ng"][j]
            yj = st["r_grid"][j] + (0.0 if dj else gamma * Vt(model, sj, alpha, k=8))
            rd.append(float(minq[j] - yj))
        residDeep = float(np.mean(rd)) if rd else None
        dr_l.append(dr); dfut_l.append(dfut); dtgt_l.append(dtgt); dQ_l.append(dQ)
        residR_l.append(residR); residDeep_l.append(residDeep); qdeep_l.append(bool(dm[iQ]))
        tq_l.append(float(st["tce_grid"][iQ])); tr_l.append(float(st["tce_grid"][iR]))
        dTRQ_l.append(float(st["tce_grid"][iR] - st["tce_grid"][iQ]))
        tpi_l.append(float(st["tce_grid"][i_pi]))
        dTQpi_l.append(float(st["tce_grid"][iQ] - st["tce_grid"][i_pi]))
    rd_ok = [x for x in residDeep_l if x is not None]
    return dict(
        alpha=alpha,
        mean_dr=round(float(np.mean(dr_l)), 5), mean_dfuture=round(float(np.mean(dfut_l)), 5),
        mean_dtarget=round(float(np.mean(dtgt_l)), 5), mean_dQ_online=round(float(np.mean(dQ_l)), 5),
        frac_dtarget_gt0=round(float(np.mean(np.array(dtgt_l) > 0)), 3),
        frac_dQ_online_gt0=round(float(np.mean(np.array(dQ_l) > 0)), 3),
        frac_Q_argmax_deep=round(float(np.mean(qdeep_l)), 3),
        mean_resid_aR=round(float(np.mean(residR_l)), 5),
        mean_resid_deep=round(float(np.mean(rd_ok)), 5) if rd_ok else None,
        mean_tce_Q=round(float(np.mean(tq_l)), 2), mean_tce_R=round(float(np.mean(tr_l)), 2),
        mean_dT_R_Q=round(float(np.mean(dTRQ_l)), 2),
        mean_tce_pi=round(float(np.mean(tpi_l)), 2), mean_dT_Q_pi=round(float(np.mean(dTQpi_l)), 2))


def vehicle(run_dir):
    r = evaluate(checkpoint=f"{run_dir}/sac_ems_best", cycle="NEDC", controller="rl",
                 eq_factor=EQF, k_fb=KFB, action_map=AMAP, lookahead=LA)
    return {k: (bool(r[k]) if k == "charge_sustaining" else r[k]) for k in
            ("v_ce_equiv", "soc_final", "d_soc_pp", "charge_sustaining", "constraint_violations",
             "off_pct", "lps_pct", "engine_on_time_s")}


def horizon(gamma):
    return dict(H_eff=round(1.0 / (1 - gamma), 2),
                **{f"g^{k}": round(gamma ** k, 6) for k in (1, 2, 5, 10, 20, 50)})


def _js(o):
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


if __name__ == "__main__":
    states = cache_physics(collect_states(SAC.load(f"{CONTROL0}/sac_ems_best")))
    out = dict(horizon={f"{g}": horizon(g) for g in GAMMAS}, per_gamma={})
    for g in GAMMAS:
        tag = f"g{int(round(g*100)):02d}"
        out["per_gamma"][f"{g}"] = {}
        for s in SEEDS:
            rd = str(G / tag / f"seed{s}")
            perck = {}
            for ck in CKPTS:
                p = G / tag / f"seed{s}" / f"{ck}.zip"
                if not p.exists():
                    continue
                m = SAC.load(str(p)[:-4])
                perck[ck] = critic_and_bellman(m, states, g)
            tq = [perck[c]["mean_tce_Q"] for c in perck]
            out["per_gamma"][f"{g}"][f"seed{s}"] = dict(
                per_checkpoint=perck,
                tce_Q_stability=dict(mean=round(float(np.mean(tq)), 2), std=round(float(np.std(tq)), 2),
                                     min=round(float(np.min(tq)), 2), max=round(float(np.max(tq)), 2)) if tq else None,
                best=perck.get("sac_ems_best"),
                vehicle=vehicle(rd))
    OUTJSON = OUT / "p13a_gamma_diagnostics.json"
    OUTJSON.write_text(json.dumps(out, indent=2, default=_js))

    print("=== effective horizon ===")
    for g in GAMMAS:
        h = out["horizon"][f"{g}"]
        print(f"  g={g}: H_eff={h['H_eff']:>6}  g^1={h['g^1']} g^2={h['g^2']} g^5={h['g^5']} "
              f"g^10={h['g^10']} g^20={h['g^20']} g^50={h['g^50']}")
    print("\n=== per (gamma, seed): BEST-checkpoint critic + Bellman-decomp + vehicle ===")
    print(f"{'g/seed':>10} {'tce_Q':>7} {'tce_R':>7} {'dT(R-Q)':>8} {'dr':>9} {'dfuture':>9} {'dtarget':>9} "
          f"{'f(dtgt>0)':>10} {'f(dQ>0)':>8} {'Q@deep%':>8} {'residR':>9} {'residDp':>9} {'tceQ_std':>9} "
          f"{'V_CE':>8} {'CS':>4}")
    for g in GAMMAS:
        for s in SEEDS:
            d = out["per_gamma"][f"{g}"][f"seed{s}"]; b = d["best"]; v = d["vehicle"]
            st = d["tce_Q_stability"]
            print(f"{('g'+str(g)+' s'+str(s)):>10} {b['mean_tce_Q']:>7} {b['mean_tce_R']:>7} {b['mean_dT_R_Q']:>8} "
                  f"{b['mean_dr']:>9.5f} {b['mean_dfuture']:>9.5f} {b['mean_dtarget']:>9.5f} "
                  f"{b['frac_dtarget_gt0']:>10} {b['frac_dQ_online_gt0']:>8} {b['frac_Q_argmax_deep']:>8} "
                  f"{b['mean_resid_aR']:>9.5f} {str(b['mean_resid_deep']):>9} {st['std']:>9} "
                  f"{v['v_ce_equiv']:>8.4f} {str(v['charge_sustaining']):>4}")
    print(f"\n[saved] {OUTJSON}")
