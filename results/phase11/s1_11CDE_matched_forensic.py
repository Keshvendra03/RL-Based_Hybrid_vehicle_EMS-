"""
11C / 11D / 11E -- STATE-MATCHED REWARD / CRITIC / ACTOR / REPLAY / BELLMAN
FORENSIC  (NO TRAINING, NO CODE CHANGE).

For real CONTROL-rollout states in demand bands {15-25, 25-30, 30-35, 35-50} Nm:
  11C-A  r(s,a) curve over a dense feasible action grid  -> a_R*  (+ full local curve)
  11C-B  Q1,Q2, min-Q, disagreement over the same (s,a)  -> a_Q*
  11C-C  actor deterministic action a_pi and its policy-space output
  11C-D  replay-buffer action occupancy near the state, bucketed by executed T_CE
  11D    Q(s,a) = r(s,a) + gamma*(1-done)*V(s'(a)) decomposition
         V(s') = min_i Q(s', a'_det) - ent_coef * log pi(a'_det | s')   [ent_coef ~ 1.7e-3 => term ~ 0]
         -> is the critic's ~35 Nm preference driven by immediate term, future term,
            approximation error, or coverage/extrapolation?
  11E    actor-tracking: Q@a_pi vs Q@a_Q*, policy-space distance, local dQ/da at a_pi.

Uses the 3 CONTROL NEDC seeds (critic averaged where noted) + FTP75 seed0 for cross-check.
Outputs: results/phase11/data/s1_11CDE_{CYCLE}.json + console summary.
"""
import copy, json, warnings
import numpy as np
import torch as th
from pathlib import Path
warnings.filterwarnings("ignore")

from stable_baselines3 import SAC
from src.env.ems_env import EMSEnv, U_MIN, U_MAX
from src.env.powertrain import _Q_BT_0, _T_CUTOFF

CTRL = dict(action_map="modeaware_gated", k_fb=2.5,
            eq_factor={"NEDC": 0.2717, "FTP75": 0.4981}, lookahead=5)
SEEDS = {
    "NEDC": ["models_p5s0_k2.5/NEDC", "models_p5_k2.5/NEDC", "models_p5_k2.5_s2/NEDC"],
    "FTP75": ["models_p5f_k2.5_s0/FTP75"],
}
BANDS = [("15-25", 15, 25), ("25-30", 25, 30), ("30-35", 30, 35), ("35-50", 35, 50)]
N_A = 121
N_PER_BAND = 5
GAMMA = 0.20


def band_of(T):
    for n, lo, hi in BANDS:
        if lo <= T < hi:
            return n
    return None


def make_env(cycle):
    return EMSEnv(cycle, eq_factor=CTRL["eq_factor"][cycle], k_fb=CTRL["k_fb"],
                  action_map=CTRL["action_map"], lookahead=CTRL["lookahead"])


# ---------------------------------------------------------------- critic/actor
def q_minmax(models, obs_np, a_np):
    """min-Q averaged over the given models; also per-model + disagreement."""
    ot = th.as_tensor(obs_np.reshape(1, -1), dtype=th.float32)
    at = th.as_tensor(np.asarray(a_np).reshape(1, -1), dtype=th.float32)
    q1s, q2s = [], []
    with th.no_grad():
        for m in models:
            q1, q2 = m.critic(ot, at)
            q1s.append(float(q1)); q2s.append(float(q2))
    q1s, q2s = np.array(q1s), np.array(q2s)
    return dict(q1=q1s.mean(), q2=q2s.mean(),
                minq_permodel=[min(a, b) for a, b in zip(q1s, q2s)],
                minq_meanacross=float(np.mean([min(a, b) for a, b in zip(q1s, q2s)])),
                disagree_within=float(np.mean(np.abs(q1s - q2s))),
                disagree_across=float(np.std([min(a, b) for a, b in zip(q1s, q2s)])))


def actor_det(m, obs_np):
    ot = th.as_tensor(obs_np.reshape(1, -1), dtype=th.float32)
    with th.no_grad():
        a = m.actor(ot, deterministic=True)
    return float(a.reshape(-1)[0])


def actor_logp(m, obs_np, a_val):
    ot = th.as_tensor(obs_np.reshape(1, -1), dtype=th.float32)
    at = th.as_tensor(np.array([[a_val]]), dtype=th.float32)
    with th.no_grad():
        mean, log_std, _ = m.actor.get_action_dist_params(ot)
        lp = m.actor.action_dist.proba_distribution(mean, log_std).log_prob(at)
    return float(lp.reshape(-1)[0]), float(th.exp(m.log_ent_coef).item()), float(log_std.reshape(-1)[0])


# ---------------------------------------------------------------- states
def collect_states(cycle, model0):
    env = make_env(cycle); obs, _ = env.reset()
    buckets = {n: [] for n, _, _ in BANDS}
    while True:
        d = env._demand
        w, T = d["w_MGB"], d["T_MGB"]
        b = band_of(T)
        if b and w > 0 and len(buckets[b]) < 60:
            buckets[b].append(dict(env=copy.deepcopy(env), obs=obs.copy(),
                                   w=w, dw=d["dw_MGB"], T=T, dT=d["d_T_MGB"],
                                   soc=env._Q_BT / _Q_BT_0))
        a, _ = model0.predict(obs, deterministic=True)
        obs, r, term, _, info = env.step(a)
        if term:
            break
    out = {}
    for n, b in buckets.items():
        if not b:
            out[n] = []; continue
        idx = np.linspace(0, len(b) - 1, min(N_PER_BAND, len(b))).astype(int)
        out[n] = [b[i] for i in idx]
    return out


# ---------------------------------------------------------------- replay coverage
def replay_tce_coverage(cycle):
    """Executed-T_CE occupancy of replay actions, near each state-cluster."""
    tmpl = make_env(cycle)
    per_seed = []
    for rd in SEEDS[cycle]:
        m = SAC.load(f"{rd}/sac_ems_best")
        try:
            m.load_replay_buffer(f"{rd}/replay_buffer.pkl")
        except Exception as e:
            per_seed.append({"error": str(e)}); continue
        rb = m.replay_buffer
        n = rb.buffer_size if rb.full else rb.pos
        O = rb.observations[:n, 0, :]
        A = rb.actions[:n, 0, :].reshape(-1)
        w = O[:, 0] * 300.0
        dw = O[:, 1] * 60.0
        T = O[:, 2] * 150.0
        soc = (O[:, 4] + 1.0) / 2.0
        keep = (T >= 12) & (T <= 55) & (w > 0)
        idx = np.where(keep)[0]
        tce = np.full(len(idx), np.nan)
        for k, i in enumerate(idx):
            tmpl._demand = dict(w_MGB=float(w[i]), dw_MGB=float(dw[i]), T_MGB=float(T[i]),
                                d_T_MGB=0.0)
            tmpl._Q_BT = float(soc[i]) * _Q_BT_0
            t_ce, t_em, u, mode = tmpl._action_to_torques(np.array([A[i]], np.float32))
            tce[k] = t_ce
        per_seed.append(dict(rd=rd, n_total=int(n), n_in_1250=int(len(idx)),
                             T=T[idx], w=w[idx], soc=soc[idx], tce=tce))
    return per_seed


def coverage_for_state(per_seed, st, band):
    """Counts + fractions of replay actions in T_CE buckets, (a) band-wide, (b) state-matched."""
    lo, hi = [(l, h) for n, l, h in BANDS if n == band][0]
    tce_buckets = [("<30", -1e9, 30), ("30-40", 30, 40), ("40-50", 40, 50),
                   ("50-55", 50, 55), ("55-60", 55, 60), (">60", 60, 1e9)]
    agg_band = {k: 0 for k, _, _ in tce_buckets}
    agg_match = {k: 0 for k, _, _ in tce_buckets}
    n_band = n_match = 0
    for ps in per_seed:
        if "error" in ps:
            continue
        inb = (ps["T"] >= lo) & (ps["T"] < hi)
        match = inb & (np.abs(ps["w"] - st["w"]) <= max(0.15 * st["w"], 20)) & \
                (np.abs(ps["soc"] - st["soc"]) <= 0.05) & (np.abs(ps["T"] - st["T"]) <= 3.0)
        for k, blo, bhi in tce_buckets:
            m1 = inb & (ps["tce"] >= blo) & (ps["tce"] < bhi)
            m2 = match & (ps["tce"] >= blo) & (ps["tce"] < bhi)
            agg_band[k] += int(np.nansum(m1)); agg_match[k] += int(np.nansum(m2))
        n_band += int(inb.sum()); n_match += int(match.sum())
    def frac(d, N):
        return {k: (round(v / N, 4) if N else None) for k, v in d.items()}
    return dict(
        state_match_criterion="|w-w0|<=max(15%,20) AND |soc-soc0|<=0.05 AND |T-T0|<=3Nm, T in band",
        n_band=n_band, band_counts=agg_band, band_fracs=frac(agg_band, n_band),
        n_state_matched=n_match, matched_counts=agg_match, matched_fracs=frac(agg_match, n_match),
    )


# ---------------------------------------------------------------- main per state
def analyse_state(models, st, grid_a):
    E = st["env"]; o = st["obs"]
    r_l, tce_l, u_l, pem_l, minq_l, dis_w_l, dis_a_l = ([] for _ in range(7))
    qt_l, imm_l, fut_l, done_l = [], [], [], []
    for a in grid_a:
        cp = copy.deepcopy(E)
        _, r, term, _, info = cp.step(np.array([a], np.float32))
        r_l.append(r); tce_l.append(info["T_CE_cmd"]); u_l.append(info["u"]); pem_l.append(info["p_em"])
        qm = q_minmax(models, o, a)
        minq_l.append(qm["minq_meanacross"]); dis_w_l.append(qm["disagree_within"]); dis_a_l.append(qm["disagree_across"])
        # ---- Bellman decomposition (11D) ----
        sp = cp._last_obs
        if term:
            V = 0.0
        else:
            ap = actor_det(models[0], sp)             # next action from seed-0 actor (on-policy-ish)
            lp, ec, _ = actor_logp(models[0], sp, ap)
            qmp = q_minmax(models, sp, ap)
            V = qmp["minq_meanacross"] - ec * lp
        qt = r + GAMMA * (0.0 if term else 1.0) * V
        qt_l.append(qt); imm_l.append(r); fut_l.append(GAMMA * (0.0 if term else 1.0) * V); done_l.append(bool(term))
    r_a = np.array(r_l); q_a = np.array(minq_l); qt_a = np.array(qt_l)
    tce_a = np.array(tce_l)
    i_R = int(np.argmax(r_a)); i_Q = int(np.argmax(q_a)); i_QT = int(np.argmax(qt_a))
    a_pi = actor_det(models[0], o)
    # nearest grid index to a_pi
    i_pi = int(np.argmin(np.abs(grid_a - a_pi)))
    # interp helpers at target T_CE (35 vs the reachable max, and 58 where possible)
    def val_at_tce(arr, tgt):
        # tce_a is monotone decreasing in a; find bracketing
        d = tce_a - tgt
        cr = np.where(np.diff(np.sign(d)) != 0)[0]
        if len(cr) == 0:
            return None
        j = cr[0]
        f = (tgt - tce_a[j]) / (tce_a[j + 1] - tce_a[j] + 1e-12)
        return float(arr[j] + f * (arr[j + 1] - arr[j]))
    r35, r58 = val_at_tce(r_a, 35.0), val_at_tce(r_a, 58.0)
    q35, q58 = val_at_tce(q_a, 35.0), val_at_tce(q_a, 58.0)
    # local critic gradient dQ/da at a_pi (finite diff on the critic, obs fixed)
    h = 0.02
    qp = q_minmax(models, o, min(a_pi + h, 1.0))["minq_meanacross"]
    qm = q_minmax(models, o, max(a_pi - h, -1.0))["minq_meanacross"]
    dQda_at_pi = (qp - qm) / (min(a_pi + h, 1.0) - max(a_pi - h, -1.0))
    # per-model minq at a_pi and a_Q*
    qm_pi = q_minmax(models, o, a_pi)
    qm_Q = q_minmax(models, o, grid_a[i_Q])
    return dict(
        band=st["band"], T_MGB=round(st["T"], 2), w=round(st["w"], 1), soc=round(st["soc"], 4),
        dT_MGB=round(st["dT"], 2),
        reachable_TCE=[round(float(tce_a.min()), 2), round(float(tce_a.max()), 2)],
        a_R_star=round(float(grid_a[i_R]), 4), TCE_at_a_R=round(float(tce_a[i_R]), 2),
        a_Q_star=round(float(grid_a[i_Q]), 4), TCE_at_a_Q=round(float(tce_a[i_Q]), 2),
        a_QT_star=round(float(grid_a[i_QT]), 4), TCE_at_a_QT=round(float(tce_a[i_QT]), 2),
        a_pi=round(a_pi, 4), TCE_at_a_pi=round(float(tce_a[i_pi]), 2),
        actor_logstd=round(actor_logp(models[0], o, a_pi)[2], 3),
        ent_coef=round(actor_logp(models[0], o, a_pi)[1], 6),
        # 11C-A strength of 58 vs 35 preference in the immediate reward
        r_at_TCE35=None if r35 is None else round(r35, 5),
        r_at_TCE58=None if r58 is None else round(r58, 5),
        dR_58_minus_35=None if (r35 is None or r58 is None) else round(r58 - r35, 5),
        # 11C-B same for critic
        minQ_at_TCE35=None if q35 is None else round(q35, 5),
        minQ_at_TCE58=None if q58 is None else round(q58, 5),
        dQ_58_minus_35=None if (q35 is None or q58 is None) else round(q58 - q35, 5),
        # 11C-B critic disagreement near the reward-optimum vs actor
        disagree_within_at_a_R=round(dis_w_l[i_R], 5),
        disagree_within_at_a_pi=round(dis_w_l[i_pi], 5),
        disagree_across_at_a_R=round(dis_a_l[i_R], 5),
        # 11D Bellman decomposition at a_R*, a_Q*, a_pi
        bellman_at_a_R=dict(r=round(imm_l[i_R], 5), gammaV=round(fut_l[i_R], 5),
                            Qtarget=round(qt_l[i_R], 5), Qhat=round(q_a[i_R], 5),
                            resid_Qhat_minus_Qtarget=round(q_a[i_R] - qt_l[i_R], 5)),
        bellman_at_a_Q=dict(r=round(imm_l[i_Q], 5), gammaV=round(fut_l[i_Q], 5),
                            Qtarget=round(qt_l[i_Q], 5), Qhat=round(q_a[i_Q], 5),
                            resid_Qhat_minus_Qtarget=round(q_a[i_Q] - qt_l[i_Q], 5)),
        bellman_at_a_pi=dict(r=round(imm_l[i_pi], 5), gammaV=round(fut_l[i_pi], 5),
                             Qtarget=round(qt_l[i_pi], 5), Qhat=round(q_a[i_pi], 5),
                             resid_Qhat_minus_Qtarget=round(q_a[i_pi] - qt_l[i_pi], 5)),
        # does the future term flip the preference?  compare argmax over r vs over (r+gammaV)
        argmax_immediate_TCE=round(float(tce_a[i_R]), 2),
        argmax_reward_plus_future_TCE=round(float(tce_a[i_QT]), 2),
        argmax_learned_Q_TCE=round(float(tce_a[i_Q]), 2),
        # 11E actor tracking
        minQ_at_a_pi=round(qm_pi["minq_meanacross"], 5),
        minQ_at_a_Q=round(qm_Q["minq_meanacross"], 5),
        Q_loss_actor_vs_criticopt=round(qm_Q["minq_meanacross"] - qm_pi["minq_meanacross"], 5),
        policyspace_dist_pi_to_Qstar=round(abs(a_pi - float(grid_a[i_Q])), 4),
        dQda_at_a_pi=round(dQda_at_pi, 5),
        # curve dumps (coarse) for the report
        curve_a=[round(float(x), 3) for x in grid_a[::8]],
        curve_TCE=[round(float(x), 2) for x in tce_a[::8]],
        curve_r=[round(float(x), 5) for x in r_a[::8]],
        curve_minQ=[round(float(x), 5) for x in q_a[::8]],
        curve_rplusfut=[round(float(x), 5) for x in qt_a[::8]],
    )


if __name__ == "__main__":
    Path("results/phase11/data").mkdir(parents=True, exist_ok=True)
    grid_a = np.linspace(-1.0, 1.0, N_A)
    for cyc in ("NEDC", "FTP75"):
        models = [SAC.load(f"{rd}/sac_ems_best") for rd in SEEDS[cyc]]
        states = collect_states(cyc, models[0])
        per_seed_cov = replay_tce_coverage(cyc)
        rows = []
        for band, sts in states.items():
            for st in sts:
                st["band"] = band
                row = analyse_state(models, st, grid_a)
                row["replay_coverage"] = coverage_for_state(per_seed_cov, st, band)
                rows.append(row)
        Path(f"results/phase11/data/s1_11CDE_{cyc}.json").write_text(
            json.dumps({"gamma": GAMMA, "n_action_grid": N_A, "rows": rows}, indent=2))

        print(f"\n############################## {cyc}")
        print(f"{'band':>6} {'T':>6} {'SoC%':>6} {'reach_TCE':>14} "
              f"{'aR*':>6} {'TCE_R':>6} {'aQ*':>6} {'TCE_Q':>6} {'a_pi':>6} {'TCE_pi':>7} "
              f"{'dR58-35':>9} {'dQ58-35':>9} {'Qloss':>8} {'BellRes@pi':>11} {'lstd':>6}")
        for r in rows:
            print(f"{r['band']:>6} {r['T_MGB']:>6} {r['soc']*100:>6.1f} "
                  f"{str(r['reachable_TCE']):>14} {r['a_R_star']:>6} {r['TCE_at_a_R']:>6} "
                  f"{r['a_Q_star']:>6} {r['TCE_at_a_Q']:>6} {r['a_pi']:>6} {r['TCE_at_a_pi']:>7} "
                  f"{str(r['dR_58_minus_35']):>9} {str(r['dQ_58_minus_35']):>9} "
                  f"{r['Q_loss_actor_vs_criticopt']:>8} "
                  f"{r['bellman_at_a_pi']['resid_Qhat_minus_Qtarget']:>11} {r['actor_logstd']:>6.2f}")
        # aggregate replay coverage over 30-35 band
        b3035 = [r for r in rows if r["band"] == "30-35"]
        if b3035:
            print("  -- replay T_CE occupancy, 30-35 Nm band (aggregate over 3 seeds):")
            cov = b3035[0]["replay_coverage"]
            print(f"     band-wide  n={cov['n_band']:>6}  fracs={cov['band_fracs']}")
            print(f"     state-matched n={cov['n_state_matched']:>5}  fracs={cov['matched_fracs']}")
    print("\n[saved] results/phase11/data/s1_11CDE_{NEDC,FTP75}.json")
