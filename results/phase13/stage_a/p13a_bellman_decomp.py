"""
PHASE 13 STAGE A  --  Bellman-target decomposition (NO TRAINING).

Distinguishes H1 (Bellman target itself prefers part-load) from H2 (target
prefers deep-LPS but the learned critic does not fit it).

Exact SB3 2.8.0 SAC target (n_step=1, plain ReplayBuffer -> discounts=None -> gamma):
    y(s,a) = r(s,a) + (1 - done) * gamma * [ min_j Qbar_j(s', a') - alpha * log pi(a'|s') ]
             a' ~ pi(.|s')   (stochastic sample)
    Qbar = target critic (polyak tau=0.005) ; alpha = exp(log_ent_coef)

For CORE matched states (25-30 / 30-35 / 35-50 Nm demand) and the two actions
  a_part = actor deterministic action at s   (~30-40 Nm, current preference)
  a_R*   = argmax_a r(s,a)                    (~63 Nm, instantaneous reward optimum)
compute r, s', V_target(s'), gamma*V, y, Q_online, Q_target, residual, and the
four Deltas.  Averaged over the 3 CONTROL seeds.

Outputs: results/phase13/stage_a/data/p13a_bellman_decomp.json + console.
"""
import copy, json, warnings
from pathlib import Path
import numpy as np
import torch as th
warnings.filterwarnings("ignore")

from stable_baselines3 import SAC
from src.env.ems_env import EMSEnv, SOC_TARGET
from src.env.powertrain import _Q_BT_0, _T_CUTOFF

OUT = Path("results/phase13/stage_a/data"); OUT.mkdir(parents=True, exist_ok=True)
CONTROL = ["models_p5s0_k2.5/NEDC", "models_p5_k2.5/NEDC", "models_p5_k2.5_s2/NEDC"]
EQF, KFB, AMAP, LA = 0.2717, 2.5, "modeaware_gated", 5
GAMMA = 0.20
N_A = 161
GRID_A = np.linspace(-1.0, 1.0, N_A)
BANDS = [("25-30", 25, 30), ("30-35", 30, 35), ("35-50", 35, 50)]
N_PER_BAND = 8
K_SAMPLES = 32           # samples of a'~pi(.|s') to reduce V(s') estimator variance


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
            b[bb].append(dict(env=copy.deepcopy(env), obs=obs.copy(), band=bb,
                              w=w, T=T, soc=env._Q_BT / _Q_BT_0))
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
    """For each state: reward grid, executed T_CE grid, and (r, s', done) for
    a_part (=actor det) and a_R* (=reward argmax)."""
    tmpl_actor = SAC.load(f"{CONTROL[0]}/sac_ems_best")
    for st in states:
        E = st["env"]
        rr, tce = [], []
        nobs_grid = []
        for a in GRID_A:
            cp = copy.deepcopy(E)
            _, r, term, _, info = cp.step(np.array([a], np.float32))
            rr.append(r); tce.append(info["T_CE_cmd"]); nobs_grid.append((cp._last_obs.copy(), bool(term)))
        st["r_grid"] = np.array(rr); st["tce_grid"] = np.array(tce)
        st["iR"] = int(np.argmax(st["r_grid"]))
        # a_R* details
        cpR = copy.deepcopy(E)
        _, rR, tR, _, infoR = cpR.step(np.array([GRID_A[st["iR"]]], np.float32))
        st["aR"] = float(GRID_A[st["iR"]]); st["rR"] = float(rR); st["sR"] = cpR._last_obs.copy()
        st["doneR"] = bool(tR); st["tceR"] = float(infoR["T_CE_cmd"])
        st["nobs_grid"] = nobs_grid
    return states


@th.no_grad()
def V_target(model, s_next, alpha, k=K_SAMPLES):
    """SAC next-state value: min_j Qbar_j(s', a') - alpha*log pi(a'|s'), a'~pi.
    Returns (mean over k samples, single-sample value, std)."""
    O = th.as_tensor(np.repeat(np.asarray(s_next)[None, :], k, 0), dtype=th.float32)
    a_s, logp = model.actor.action_log_prob(O)          # stochastic samples
    q1, q2 = model.critic_target(O, a_s)
    v = th.minimum(q1, q2).squeeze(1) - alpha * logp.reshape(-1)
    v = v.numpy()
    return float(v.mean()), float(v[0]), float(v.std())


@th.no_grad()
def V_greedy(model, s_next):
    """max_a' min_j Q_online(s', a') over the action grid -- 'optimistic' next-state
    value the target WOULD use if the policy were greedy w.r.t. the online critic."""
    O = th.as_tensor(np.repeat(np.asarray(s_next)[None, :], N_A, 0), dtype=th.float32)
    A = th.as_tensor(GRID_A[:, None], dtype=th.float32)
    q1, q2 = model.critic(O, A)
    return float(np.max(np.minimum(q1.squeeze(1).numpy(), q2.squeeze(1).numpy())))


@th.no_grad()
def Q_of(model, obs, a, target=False):
    O = th.as_tensor(np.asarray(obs).reshape(1, -1), dtype=th.float32)
    A = th.as_tensor(np.array([[a]]), dtype=th.float32)
    q1, q2 = (model.critic_target if target else model.critic)(O, A)
    return float(min(float(q1), float(q2)))


def analyse():
    models = [SAC.load(f"{c}/sac_ems_best") for c in CONTROL]
    alphas = [float(th.exp(m.log_ent_coef.detach())) for m in models]
    states = cache_physics(collect_states(models[0]))

    rows = []
    for st in states:
        # a_part = actor deterministic action (per seed); use seed-0 for the state def,
        # but evaluate the decomposition per seed and average.
        per_seed = []
        for m, alpha in zip(models, alphas):
            a_pi = float(m.actor(th.as_tensor(st["obs"][None, :], dtype=th.float32),
                                 deterministic=True).squeeze().item())
            i_pi = int(np.argmin(np.abs(GRID_A - a_pi)))
            s_part, done_part = st["nobs_grid"][i_pi]
            r_part = float(st["r_grid"][i_pi]); tce_part = float(st["tce_grid"][i_pi])
            # V(s') for part-load and for a_R*
            Vp_m, Vp_1, Vp_s = V_target(m, s_part, alpha)
            VR_m, VR_1, VR_s = V_target(m, st["sR"], alpha)
            Vp_g = V_greedy(m, s_part); VR_g = V_greedy(m, st["sR"])
            gVp = (0.0 if done_part else GAMMA * Vp_m)
            gVR = (0.0 if st["doneR"] else GAMMA * VR_m)
            gVp_g = (0.0 if done_part else GAMMA * Vp_g)
            gVR_g = (0.0 if st["doneR"] else GAMMA * VR_g)
            y_part = r_part + gVp
            y_R = st["rR"] + gVR
            Qo_part = Q_of(m, st["obs"], GRID_A[i_pi], target=False)
            Qo_R = Q_of(m, st["obs"], st["aR"], target=False)
            Qt_part = Q_of(m, st["obs"], GRID_A[i_pi], target=True)
            Qt_R = Q_of(m, st["obs"], st["aR"], target=True)
            per_seed.append(dict(
                a_pi=a_pi, tce_part=tce_part, r_part=r_part,
                gV_part=gVp, gV_R=gVR, gV_part_greedy=gVp_g, gV_R_greedy=gVR_g,
                y_part=y_part, y_R=y_R, Qo_part=Qo_part, Qo_R=Qo_R, Qt_part=Qt_part, Qt_R=Qt_R,
                dr=st["rR"] - r_part,
                dfuture=gVR - gVp,
                dfuture_greedy=gVR_g - gVp_g,
                dtarget=y_R - y_part,
                dtarget_greedy=(st["rR"] + gVR_g) - (r_part + gVp_g),
                dQ_online=Qo_R - Qo_part,
                dQ_target=Qt_R - Qt_part,
                resid_part=Qo_part - y_part, resid_R=Qo_R - y_R,
                V_R_sample_std=VR_s, V_part_sample_std=Vp_s))
        def mean(k): return float(np.mean([p[k] for p in per_seed]))
        rows.append(dict(
            band=st["band"], T=round(float(st["T"]), 2), soc=round(float(st["soc"]), 4),
            tce_part=round(mean("tce_part"), 2), tce_R=round(float(st["tceR"]), 2),
            dr=round(mean("dr"), 5),
            dfuture=round(mean("dfuture"), 5),
            dfuture_greedy=round(mean("dfuture_greedy"), 5),
            dtarget=round(mean("dtarget"), 5),
            dtarget_greedy=round(mean("dtarget_greedy"), 5),
            dQ_online=round(mean("dQ_online"), 5),
            dQ_target=round(mean("dQ_target"), 5),
            resid_R=round(mean("resid_R"), 5), resid_part=round(mean("resid_part"), 5),
            gV_part=round(mean("gV_part"), 5), gV_R=round(mean("gV_R"), 5),
            V_R_sample_std=round(mean("V_R_sample_std"), 5),
            per_seed=per_seed))
    return rows, alphas


def summarise(rows):
    core = rows
    def frac(key, pred): return round(float(np.mean([pred(r[key]) for r in core])), 4)
    S = dict(
        n_states=len(core),
        mean_dr=round(float(np.mean([r["dr"] for r in core])), 5),
        mean_dfuture=round(float(np.mean([r["dfuture"] for r in core])), 5),
        mean_dfuture_greedy=round(float(np.mean([r["dfuture_greedy"] for r in core])), 5),
        mean_dtarget=round(float(np.mean([r["dtarget"] for r in core])), 5),
        mean_dtarget_greedy=round(float(np.mean([r["dtarget_greedy"] for r in core])), 5),
        mean_dQ_online=round(float(np.mean([r["dQ_online"] for r in core])), 5),
        mean_dQ_target=round(float(np.mean([r["dQ_target"] for r in core])), 5),
        frac_dr_gt0=frac("dr", lambda x: x > 0),
        frac_dfuture_lt0=frac("dfuture", lambda x: x < 0),
        frac_dtarget_gt0=frac("dtarget", lambda x: x > 0),
        frac_dtarget_greedy_gt0=frac("dtarget_greedy", lambda x: x > 0),
        frac_dQ_online_gt0=frac("dQ_online", lambda x: x > 0),
        by_band={},
    )
    for bn, _, _ in BANDS:
        b = [r for r in core if r["band"] == bn]
        if not b:
            continue
        S["by_band"][bn] = dict(
            n=len(b),
            mean_dr=round(float(np.mean([r["dr"] for r in b])), 5),
            mean_dfuture=round(float(np.mean([r["dfuture"] for r in b])), 5),
            mean_dtarget=round(float(np.mean([r["dtarget"] for r in b])), 5),
            mean_dtarget_greedy=round(float(np.mean([r["dtarget_greedy"] for r in b])), 5),
            mean_dQ_online=round(float(np.mean([r["dQ_online"] for r in b])), 5),
            frac_dtarget_gt0=round(float(np.mean([r["dtarget"] > 0 for r in b])), 3),
            frac_dtarget_greedy_gt0=round(float(np.mean([r["dtarget_greedy"] > 0 for r in b])), 3),
            frac_dQ_online_gt0=round(float(np.mean([r["dQ_online"] > 0 for r in b])), 3))
    # pre-training classification
    if S["frac_dr_gt0"] >= 0.6 and S["frac_dtarget_gt0"] <= 0.4:
        S["pretraining_case"] = "CASE T (temporal/objective): reward prefers deep-LPS, Bellman target prefers part-load"
    elif S["frac_dtarget_gt0"] >= 0.6 and S["frac_dQ_online_gt0"] <= 0.4:
        S["pretraining_case"] = "CASE C (critic approximation): Bellman target prefers deep-LPS, learned critic prefers part-load"
    else:
        S["pretraining_case"] = "CASE M (mixed/state-dependent): see by_band"
    return S


if __name__ == "__main__":
    rows, alphas = analyse()
    S = summarise(rows)
    (OUT / "p13a_bellman_decomp.json").write_text(json.dumps(
        dict(gamma=GAMMA, K_samples=K_SAMPLES, alphas_control=alphas,
             sac_target="y = r + (1-done)*gamma*[ min_j Qbar_j(s',a') - alpha*log pi(a'|s') ], a'~pi ; n_step=1 -> discounts=gamma",
             summary=S, rows=rows), indent=2))

    print(f"alphas (3 CONTROL) = {[round(a,5) for a in alphas]}   gamma={GAMMA}  K_samples={K_SAMPLES}")
    print(f"\n{'band':>6} {'T':>6} {'SoC%':>5} {'tceP':>6} {'tceR':>6} {'dr':>9} {'dfuture':>9} "
          f"{'dtarget':>9} {'dtgt_grdy':>10} {'dQ_onl':>9} {'dQ_tgt':>9} {'resid_R':>9}")
    for r in rows:
        print(f"{r['band']:>6} {r['T']:>6} {r['soc']*100:>5.0f} {r['tce_part']:>6.1f} {r['tce_R']:>6.1f} "
              f"{r['dr']:>9.5f} {r['dfuture']:>9.5f} {r['dtarget']:>9.5f} {r['dtarget_greedy']:>10.5f} "
              f"{r['dQ_online']:>9.5f} {r['dQ_target']:>9.5f} {r['resid_R']:>9.5f}")
    print("\n=== SUMMARY (mean over CORE states, averaged over 3 seeds) ===")
    for k in ("mean_dr", "mean_dfuture", "mean_dfuture_greedy", "mean_dtarget", "mean_dtarget_greedy",
              "mean_dQ_online", "mean_dQ_target"):
        print(f"  {k:<24} {S[k]:+.5f}")
    print(f"  frac dr>0                {S['frac_dr_gt0']}")
    print(f"  frac dfuture<0           {S['frac_dfuture_lt0']}")
    print(f"  frac dtarget>0           {S['frac_dtarget_gt0']}   (greedy: {S['frac_dtarget_greedy_gt0']})")
    print(f"  frac dQ_online>0         {S['frac_dQ_online_gt0']}")
    print("\n  by band (mean dr / dfuture / dtarget / dtarget_greedy / dQ_online ; frac dtarget>0):")
    for bn, b in S["by_band"].items():
        print(f"    {bn:>6}: {b['mean_dr']:+.4f} / {b['mean_dfuture']:+.4f} / {b['mean_dtarget']:+.4f} / "
              f"{b['mean_dtarget_greedy']:+.4f} / {b['mean_dQ_online']:+.4f}   f(dtgt>0)={b['frac_dtarget_gt0']} "
              f"f(dtgt_grdy>0)={b['frac_dtarget_greedy_gt0']}")
    print(f"\n  >>> PRE-TRAINING CLASSIFICATION: {S['pretraining_case']}")
    print("\n[saved] results/phase13/stage_a/data/p13a_bellman_decomp.json")
