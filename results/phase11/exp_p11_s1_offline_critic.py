"""
EXP-P11-S1  --  OFFLINE CRITIC FALSIFICATION (H-CRITIC vs H-COVERAGE).

Strictly offline critic-only refinement on the FROZEN CONTROL replay buffers.
NO env interaction for training, NO new rollouts, NO actor update, NO reward /
gamma / k_fb / eq_factor / entropy / net-arch / optimizer / replay / target
changes. Only critic parameters move, via the exact SB3 SAC critic loss.

The env is instantiated ONLY to compute the diagnostic r(s,a) landscape and the
next-state transitions for the Bellman residual -- identical to every prior
Phase-11 diagnostic. No transition produced this way ever enters the replay
buffer or the critic update.

N in {0, 50k, 150k, 400k} additional critic gradient steps, per each of the 3
NEDC CONTROL checkpoints. Snapshots taken at each N.

Outputs: results/phase11/EXP_P11_S1/{report_data.json, freeze_verification.json}
"""
import copy, json, hashlib, time, warnings
from pathlib import Path
import numpy as np
import torch as th
import torch.nn.functional as F
warnings.filterwarnings("ignore")

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from src.env.ems_env import EMSEnv
from src.env.powertrain import _Q_BT_0, _T_CUTOFF

OUT = Path("results/phase11/EXP_P11_S1"); OUT.mkdir(parents=True, exist_ok=True)
CTRL = dict(action_map="modeaware_gated", k_fb=2.5, lookahead=5,
            eq_factor={"NEDC": 0.2717}, gamma=0.20)
SEEDS = ["models_p5s0_k2.5/NEDC", "models_p5_k2.5/NEDC", "models_p5_k2.5_s2/NEDC"]
N_SCHEDULE = [0, 50_000, 150_000, 400_000]
BANDS = [("15-25", 15, 25), ("25-30", 25, 30), ("30-35", 30, 35), ("35-50", 35, 50)]
N_A = 121
N_PER_BAND = 5
GRID_A = np.linspace(-1.0, 1.0, N_A)


def band_of(T):
    for n, lo, hi in BANDS:
        if lo <= T < hi:
            return n
    return None


def sd_hash(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- diagnostic states
def collect_states(model0):
    env = EMSEnv("NEDC", eq_factor=CTRL["eq_factor"]["NEDC"], k_fb=CTRL["k_fb"],
                 action_map=CTRL["action_map"], lookahead=CTRL["lookahead"])
    obs, _ = env.reset()
    buckets = {n: [] for n, _, _ in BANDS}
    while True:
        d = env._demand
        w, T = d["w_MGB"], d["T_MGB"]
        b = band_of(T)
        if b and w > 0 and len(buckets[b]) < 60:
            buckets[b].append(dict(env=copy.deepcopy(env), obs=obs.copy(),
                                   band=b, w=w, T=T, soc=env._Q_BT / _Q_BT_0))
        a, _ = model0.predict(obs, deterministic=True)
        obs, r, term, _, info = env.step(a)
        if term:
            break
    out = []
    for n, b in buckets.items():
        if not b:
            continue
        idx = np.linspace(0, len(b) - 1, min(N_PER_BAND, len(b))).astype(int)
        out += [b[i] for i in idx]
    return out


def cache_physics(states):
    """N-independent: r(a), T_CE(a), next_obs(a), done(a) for every diagnostic state."""
    for st in states:
        E = st["env"]
        rr, tce, uu, pem, nobs, dn = [], [], [], [], [], []
        for a in GRID_A:
            cp = copy.deepcopy(E)
            _, r, term, _, info = cp.step(np.array([a], np.float32))
            rr.append(r); tce.append(info["T_CE_cmd"]); uu.append(info["u"]); pem.append(info["p_em"])
            nobs.append(cp._last_obs.copy()); dn.append(bool(term))
        st["r"] = np.array(rr); st["tce"] = np.array(tce); st["u"] = np.array(uu)
        st["pem"] = np.array(pem); st["next_obs"] = np.array(nobs); st["done"] = np.array(dn)
        st["a_R_star_idx"] = int(np.argmax(st["r"]))
    return states


# ---------------------------------------------------------------- critic eval
@th.no_grad()
def eval_critic_on_states(model, states, ent_coef, actor_next_cache):
    """For each state: a_Q*, residual at a_R* and a_Q*, twin-Q disagreement."""
    rows = []
    for si, st in enumerate(states):
        O = th.as_tensor(np.repeat(st["obs"][None, :], N_A, axis=0), dtype=th.float32)
        A = th.as_tensor(GRID_A[:, None], dtype=th.float32)
        q1, q2 = model.critic(O, A)
        q1 = q1.squeeze(1).numpy(); q2 = q2.squeeze(1).numpy()
        minq = np.minimum(q1, q2)
        disagree = np.abs(q1 - q2)
        i_R = st["a_R_star_idx"]; i_Q = int(np.argmax(minq))
        # V(s'(a)) with the FROZEN actor's deterministic next action (cached per seed)
        a_next, logp_next = actor_next_cache[si]      # (N_A,), (N_A,)
        NO = th.as_tensor(st["next_obs"], dtype=th.float32)
        AN = th.as_tensor(a_next[:, None], dtype=th.float32)
        nq1, nq2 = model.critic(NO, AN)
        Vnext = np.minimum(nq1.squeeze(1).numpy(), nq2.squeeze(1).numpy()) - ent_coef * logp_next
        Qtarget = st["r"] + (1.0 - st["done"].astype(float)) * CTRL["gamma"] * Vnext
        resid = minq - Qtarget
        # deep-LPS region = executed T_CE > 50 Nm (where reachable)
        deep = st["tce"] > 50.0
        rows.append(dict(
            band=st["band"], T_MGB=round(float(st["T"]), 2), soc=round(float(st["soc"]), 4),
            tce_reach_max=round(float(st["tce"].max()), 2),
            TCE_a_R=round(float(st["tce"][i_R]), 2), TCE_a_Q=round(float(st["tce"][i_Q]), 2),
            dT_R_minus_Q=round(float(st["tce"][i_R] - st["tce"][i_Q]), 2),
            resid_at_a_R=round(float(resid[i_R]), 5), resid_at_a_Q=round(float(resid[i_Q]), 5),
            resid_deepLPS_mean=round(float(resid[deep].mean()), 5) if deep.any() else None,
            disagree_at_a_R=round(float(disagree[i_R]), 5),
            disagree_at_a_Q=round(float(disagree[i_Q]), 5),
            disagree_deepLPS_mean=round(float(disagree[deep].mean()), 5) if deep.any() else None,
            minQ_at_a_R=round(float(minq[i_R]), 5), minQ_at_a_Q=round(float(minq[i_Q]), 5),
        ))
    return rows


def build_actor_next_cache(model, states):
    """FROZEN actor deterministic next-action + its log-prob, per state's next_obs grid."""
    cache = []
    with th.no_grad():
        for st in states:
            NO = th.as_tensor(st["next_obs"], dtype=th.float32)
            a_det = model.actor(NO, deterministic=True).squeeze(1).numpy()
            mean, log_std, _ = model.actor.get_action_dist_params(NO)
            lp = model.actor.action_dist.proba_distribution(
                mean, log_std).log_prob(th.as_tensor(a_det[:, None], dtype=th.float32)).numpy()
            cache.append((a_det, lp))
    return cache


# ---------------------------------------------------------------- critic refine
def refine_one_seed(run_dir, states, freeze_log):
    m = SAC.load(f"{run_dir}/sac_ems_best")
    m.load_replay_buffer(f"{run_dir}/replay_buffer.pkl")
    rb = m.replay_buffer
    n_tr = rb.buffer_size if rb.full else rb.pos
    ent_coef = float(th.exp(m.log_ent_coef.detach()))

    # ---- freeze verification (pre) ----
    actor_hash_pre = sd_hash(m.actor)
    for p in m.actor.parameters():
        p.requires_grad_(False)               # defensive: critic loss must not touch the actor
    cfg_pre = dict(gamma=m.gamma, tau=m.tau, batch=m.batch_size,
                   critic_lr=m.critic.optimizer.param_groups[0]["lr"],
                   ent_coef=ent_coef, n_replay=int(n_tr),
                   target_update_interval=m.target_update_interval,
                   net_arch=str(m.policy_kwargs.get("net_arch")))
    replay_hash_pre = hashlib.sha256(
        rb.observations[:n_tr].tobytes() + rb.actions[:n_tr].tobytes()
        + rb.rewards[:n_tr].tobytes() + rb.dones[:n_tr].tobytes()).hexdigest()

    actor_next_cache = build_actor_next_cache(m, states)

    snaps = {}   # N -> (critic_sd, critic_target_sd)
    def snapshot(N):
        snaps[N] = ({k: v.detach().clone() for k, v in m.critic.state_dict().items()},
                    {k: v.detach().clone() for k, v in m.critic_target.state_dict().items()})
    snapshot(0)

    t0 = time.time()
    total = max(N_SCHEDULE)
    log_every = 25_000
    for step in range(1, total + 1):
        data = rb.sample(m.batch_size)
        with th.no_grad():
            next_actions, next_log_prob = m.actor.action_log_prob(data.next_observations)
            nq = th.cat(m.critic_target(data.next_observations, next_actions), dim=1)
            nq, _ = th.min(nq, dim=1, keepdim=True)
            nq = nq - ent_coef * next_log_prob.reshape(-1, 1)
            target_q = data.rewards + (1 - data.dones) * m.gamma * nq
        cur_q = m.critic(data.observations, data.actions)
        critic_loss = 0.5 * sum(F.mse_loss(cq, target_q) for cq in cur_q)
        m.critic.optimizer.zero_grad()
        critic_loss.backward()
        m.critic.optimizer.step()
        polyak_update(m.critic.parameters(), m.critic_target.parameters(), m.tau)
        if step in N_SCHEDULE:
            snapshot(step)
        if step % log_every == 0:
            print(f"    [{run_dir}] step {step:>7}  critic_loss={float(critic_loss):.6e}  "
                  f"({time.time()-t0:.0f}s)")

    # ---- freeze verification (post) ----
    actor_hash_post = sd_hash(m.actor)
    replay_hash_post = hashlib.sha256(
        rb.observations[:n_tr].tobytes() + rb.actions[:n_tr].tobytes()
        + rb.rewards[:n_tr].tobytes() + rb.dones[:n_tr].tobytes()).hexdigest()
    freeze_log[run_dir] = dict(
        actor_unchanged=bool(actor_hash_pre == actor_hash_post),
        actor_hash=actor_hash_pre,
        replay_unchanged=bool(replay_hash_pre == replay_hash_post),
        replay_hash=replay_hash_pre[:16], n_replay=int(n_tr),
        cfg=cfg_pre,
        critic_changed=bool(sd_hash(m.critic) != _sd_hash_from_snap(snaps[0][0])),
    )

    # ---- diagnostics at each N ----
    results = {}
    for N in N_SCHEDULE:
        csd, ctsd = snaps[N]
        m.critic.load_state_dict(csd); m.critic_target.load_state_dict(ctsd)
        results[str(N)] = eval_critic_on_states(m, states, ent_coef, actor_next_cache)
    return results, freeze_log


def _sd_hash_from_snap(sd):
    h = hashlib.sha256()
    for k, v in sorted(sd.items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- aggregation
def stratify_by_coverage(states):
    """high-load replay coverage (matched 50-55 + 55-60 + >60 fraction) from Stage-1 data."""
    s1 = json.load(open("results/phase11/data/s1_11CDE_NEDC.json"))["rows"]
    cov = {}
    for st in states:
        best = None; bd = 1e9
        for r in s1:
            if r["band"] != st["band"]:
                continue
            dd = abs(r["T_MGB"] - st["T"]) + 100 * abs(r["soc"] - st["soc"])
            if dd < bd:
                bd = dd; best = r
        m = best["replay_coverage"]["matched_fracs"] if best else {}
        f = sum((m.get(k) or 0) for k in ("50-55", "55-60", ">60"))
        cov[(st["band"], round(st["T"], 2), round(st["soc"], 4))] = f
    return cov


if __name__ == "__main__":
    print("[EXP-P11-S1] collecting diagnostic states + caching r(s,a) physics ...")
    m0 = SAC.load(f"{SEEDS[0]}/sac_ems_best")
    states = cache_physics(collect_states(m0))
    print(f"  {len(states)} diagnostic states  ({[st['band'] for st in states].count('35-50')} in 35-50 band)")
    coverage = stratify_by_coverage(states)

    freeze_log = {}
    per_seed = {}
    for rd in SEEDS:
        print(f"[EXP-P11-S1] refining critic: {rd}")
        res, freeze_log = refine_one_seed(rd, states, freeze_log)
        per_seed[rd] = res

    # ---------- assemble tables ----------
    def agg(seed_rows, key, filt=lambda r: True):
        vals = [r[key] for r in seed_rows if filt(r) and r[key] is not None]
        vals = np.array(vals, float)
        return dict(mean=float(vals.mean()), median=float(np.median(vals)),
                    std=float(vals.std()), min=float(vals.min()), max=float(vals.max()),
                    n=len(vals))

    summary = {"N_schedule": N_SCHEDULE, "n_states": len(states),
               "objective_caveat": ("This experiment answers ONLY 'why does the trained SAC "
                    "critic fail to reproduce its own reward preference'. It does NOT address "
                    "'why is the SAC reward optimum different from ECMS' (V1: the reward is a "
                    "stiffer-battery Hamiltonian than ECMS; ECMS runs OFF at 30-35 Nm demand)."),
               "freeze_verification": freeze_log, "per_seed": per_seed,
               "coverage": {f"{k[0]}|T={k[1]}|soc={k[2]}": round(v, 4) for k, v in coverage.items()},
               "tables": {}}

    for N in N_SCHEDULE:
        Nk = str(N)
        # exclude 15-25 band (58 Nm unreachable) from the core deep-LPS metrics
        core = lambda r: r["band"] in ("25-30", "30-35", "35-50")
        wellcov = lambda r: r["band"] == "35-50"
        d = {}
        for rd in SEEDS:
            rows = per_seed[rd][Nk]
            d[rd] = dict(
                resid_at_a_R=agg(rows, "resid_at_a_R", core),
                resid_at_a_Q=agg(rows, "resid_at_a_Q", core),
                resid_deepLPS=agg(rows, "resid_deepLPS_mean", core),
                dT_R_minus_Q=agg(rows, "dT_R_minus_Q", core),
                TCE_a_Q=agg(rows, "TCE_a_Q", core),
                TCE_a_R=agg(rows, "TCE_a_R", core),
                disagree_at_a_R=agg(rows, "disagree_at_a_R", core),
                resid_at_a_R_wellcov=agg(rows, "resid_at_a_R", wellcov),
                dT_R_minus_Q_wellcov=agg(rows, "dT_R_minus_Q", wellcov),
                TCE_a_Q_wellcov=agg(rows, "TCE_a_Q", wellcov),
            )
        summary["tables"][Nk] = d

    (OUT / "report_data.json").write_text(json.dumps(summary, indent=2))
    (OUT / "freeze_verification.json").write_text(json.dumps(freeze_log, indent=2))

    # ---------- console ----------
    print("\n================ FREEZE VERIFICATION")
    for rd, fl in freeze_log.items():
        print(f"  {rd}: actor_unchanged={fl['actor_unchanged']}  replay_unchanged={fl['replay_unchanged']}  "
              f"critic_changed={fl['critic_changed']}  n_replay={fl['n_replay']}  cfg={fl['cfg']}")

    print("\n================ CORE (25-50 Nm demand)  mean over states, per seed")
    print(f"{'seed':<26} {'N':>8} {'resid@aR*':>11} {'resid@aQ*':>11} {'TCE@aQ*':>9} {'TCE@aR*':>9} "
          f"{'dT(R-Q)':>9} {'disag@aR':>9}")
    for rd in SEEDS:
        for N in N_SCHEDULE:
            t = summary["tables"][str(N)][rd]
            print(f"{rd:<26} {N:>8} {t['resid_at_a_R']['mean']:>11.5f} {t['resid_at_a_Q']['mean']:>11.5f} "
                  f"{t['TCE_a_Q']['mean']:>9.2f} {t['TCE_a_R']['mean']:>9.2f} "
                  f"{t['dT_R_minus_Q']['mean']:>9.2f} {t['disagree_at_a_R']['mean']:>9.5f}")

    print("\n================ WELL-COVERED 35-50 Nm band (38-54% replay coverage), per seed")
    print(f"{'seed':<26} {'N':>8} {'resid@aR*':>11} {'TCE@aQ*':>9} {'dT(R-Q)':>9}")
    for rd in SEEDS:
        for N in N_SCHEDULE:
            t = summary["tables"][str(N)][rd]
            print(f"{rd:<26} {N:>8} {t['resid_at_a_R_wellcov']['mean']:>11.5f} "
                  f"{t['TCE_a_Q_wellcov']['mean']:>9.2f} {t['dT_R_minus_Q_wellcov']['mean']:>9.2f}")
    print("\n[saved] results/phase11/EXP_P11_S1/report_data.json")
