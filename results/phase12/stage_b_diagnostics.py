"""
PHASE 12 STAGE B diagnostics: B5 coverage · B6 critic argmax · B7 critic stability
· B8 dual-reward audit · B9 vehicle-level eval.  NO TRAINING.

Reads results/phase12/stage_b/seed{0,1,2}/* and the 3 NEDC CONTROL checkpoints.
Writes results/phase12/stage_b/diagnostics.json + console tables.
"""
import copy, json, warnings
from pathlib import Path
import numpy as np
import torch as th
warnings.filterwarnings("ignore")

from stable_baselines3 import SAC
from src.env.ems_env import EMSEnv, SOC_TARGET
from src.env.powertrain import _Q_BT_0, _T_CUTOFF
from results.evaluate_policy import evaluate

SB = Path("results/phase12/stage_b")
SEEDS = [0, 1, 2]
CKPTS = ["sac_ems_50k", "sac_ems_100k", "sac_ems_150k", "sac_ems_best"]
CONTROL = ["models_p5s0_k2.5/NEDC", "models_p5_k2.5/NEDC", "models_p5_k2.5_s2/NEDC"]
EQF, KFB, AMAP, LA = 0.2717, 2.5, "modeaware_gated", 5
GAMMA = 0.20
NEDC_THR = SOC_TARGET + EQF / KFB          # 0.60868
N_A = 121
GRID_A = np.linspace(-1.0, 1.0, N_A)
BANDS = [("15-25", 15, 25), ("25-30", 25, 30), ("30-35", 30, 35), ("35-50", 35, 50)]
TCE_BUCKETS = [("<20", -1e9, 20), ("20-30", 20, 30), ("30-40", 30, 40), ("40-50", 40, 50),
               ("50-60", 50, 60), ("60-75", 60, 75), (">75", 75, 1e9)]


def band_of(T):
    for n, lo, hi in BANDS:
        if lo <= T < hi:
            return n
    return None


def make_env(clip=False):
    return EMSEnv("NEDC", eq_factor=EQF, k_fb=KFB, action_map=AMAP, lookahead=LA, clip_eq_eff=clip)


# ---------------------------------------------------------------- B5 coverage
def reconstruct_tce(buf_obs, buf_act):
    e = make_env()
    w = buf_obs[:, 0] * 300.0; dw = buf_obs[:, 1] * 60.0
    T = buf_obs[:, 2] * 150.0; soc = (buf_obs[:, 4] + 1.0) / 2.0
    keep = np.where((T >= 12) & (T <= 55) & (w > 0))[0]
    tce = np.full(len(keep), np.nan)
    for i, k in enumerate(keep):
        e._demand = dict(w_MGB=float(w[k]), dw_MGB=float(dw[k]), T_MGB=float(T[k]), d_T_MGB=0.0)
        e._Q_BT = float(soc[k]) * _Q_BT_0
        t_ce, _, _, _ = e._action_to_torques(np.array([buf_act[k]], np.float32))
        tce[i] = t_ce
    return T[keep], soc[keep], tce, len(keep)


def coverage_for_buffer(run_dir):
    m = SAC.load(f"{run_dir}/sac_ems_best")
    m.load_replay_buffer(f"{run_dir}/replay_buffer.pkl")
    rb = m.replay_buffer
    n = rb.buffer_size if rb.full else rb.pos
    O = rb.observations[:n, 0, :]; A = rb.actions[:n, 0, :].reshape(-1)
    T, soc, tce, n_moving_1250 = reconstruct_tce(O, A)
    out = dict(n_total=int(n), n_in_12_55_Nm=int(n_moving_1250))
    for bn, lo, hi in BANDS:
        inb = (T >= lo) & (T < hi)
        nb = int(inb.sum())
        buckets = {tn: int(((tce >= tl) & (tce < th_) & inb).sum()) for tn, tl, th_ in TCE_BUCKETS}
        fr = {tn: (round(v / nb, 4) if nb else None) for tn, v in buckets.items()}
        hi_load = int(((tce >= 50) & inb).sum())
        out[bn] = dict(n=nb, tce_counts=buckets, tce_fracs=fr,
                       frac_TCE_ge_50=round(hi_load / nb, 4) if nb else None)
    return out


# ---------------------------------------------------------------- states + physics
def collect_states(model0):
    env = make_env(); obs, _ = env.reset()
    buckets = {n: [] for n, _, _ in BANDS}
    while True:
        d = env._demand
        w, T = d["w_MGB"], d["T_MGB"]
        b = band_of(T)
        if b and w > 0 and len(buckets[b]) < 60:
            buckets[b].append(dict(env=copy.deepcopy(env), obs=obs.copy(), band=b,
                                   w=w, dw=d["dw_MGB"], T=T, soc=env._Q_BT / _Q_BT_0))
        a, _ = model0.predict(obs, deterministic=True)
        obs, r, term, _, info = env.step(a)
        if term:
            break
    out = []
    for n, b in buckets.items():
        if b:
            idx = np.linspace(0, len(b) - 1, min(5, len(b))).astype(int)
            out += [b[i] for i in idx]
    return out


def cache_physics(states):
    for st in states:
        E = st["env"]; rr, tce, nobs, dn = [], [], [], []
        for a in GRID_A:
            cp = copy.deepcopy(E)
            _, r, term, _, info = cp.step(np.array([a], np.float32))
            rr.append(r); tce.append(info["T_CE_cmd"]); nobs.append(cp._last_obs.copy()); dn.append(bool(term))
        st["r"] = np.array(rr); st["tce"] = np.array(tce)
        st["next_obs"] = np.array(nobs); st["done"] = np.array(dn)
        st["iR"] = int(np.argmax(st["r"]))
    return states


@th.no_grad()
def critic_eval(model, states, ent_coef):
    a_det_cache = []
    for st in states:
        NO = th.as_tensor(st["next_obs"], dtype=th.float32)
        ad = model.actor(NO, deterministic=True).squeeze(1).numpy()
        mean, ls, _ = model.actor.get_action_dist_params(NO)
        lp = model.actor.action_dist.proba_distribution(mean, ls).log_prob(
            th.as_tensor(ad[:, None], dtype=th.float32)).numpy()
        a_det_cache.append((ad, lp))
    rows = []
    for si, st in enumerate(states):
        O = th.as_tensor(np.repeat(st["obs"][None, :], N_A, 0), dtype=th.float32)
        Agrid = th.as_tensor(GRID_A[:, None], dtype=th.float32)
        q1, q2 = model.critic(O, Agrid)
        minq = np.minimum(q1.squeeze(1).numpy(), q2.squeeze(1).numpy())
        iQ = int(np.argmax(minq)); iR = st["iR"]
        ad, lp = a_det_cache[si]
        NO = th.as_tensor(st["next_obs"], dtype=th.float32)
        nq1, nq2 = model.critic(NO, th.as_tensor(ad[:, None], dtype=th.float32))
        Vn = np.minimum(nq1.squeeze(1).numpy(), nq2.squeeze(1).numpy()) - ent_coef * lp
        Qt = st["r"] + (1 - st["done"].astype(float)) * GAMMA * Vn
        a_pi = float(model.actor(th.as_tensor(st["obs"][None, :], dtype=th.float32),
                                 deterministic=True).squeeze().item())
        i_pi = int(np.argmin(np.abs(GRID_A - a_pi)))
        rows.append(dict(band=st["band"], T=round(float(st["T"]), 1), soc=round(float(st["soc"]), 3),
                         tce_R=round(float(st["tce"][iR]), 1), tce_Q=round(float(st["tce"][iQ]), 1),
                         tce_pi=round(float(st["tce"][i_pi]), 1),
                         dT_R_Q=round(float(st["tce"][iR] - st["tce"][iQ]), 1),
                         dT_Q_pi=round(float(st["tce"][iQ] - st["tce"][i_pi]), 1),
                         resid_aR=round(float(minq[iR] - Qt[iR]), 5),
                         resid_aQ=round(float(minq[iQ] - Qt[iQ]), 5),
                         minq_mag=round(float(np.abs(minq).mean()), 4)))
    return rows


def agg(rows, keyfilter=lambda r: r["band"] in ("25-30", "30-35", "35-50")):
    R = [r for r in rows if keyfilter(r)]
    def m(k): return round(float(np.mean([r[k] for r in R])), 4)
    return dict(n=len(R), tce_Q=m("tce_Q"), tce_R=m("tce_R"), tce_pi=m("tce_pi"),
               dT_R_Q=m("dT_R_Q"), dT_Q_pi=m("dT_Q_pi"), resid_aR=m("resid_aR"),
               resid_aQ=m("resid_aQ"), minq_mag=m("minq_mag"))


# ---------------------------------------------------------------- B8 dual reward
def dual_reward_audit(run_dir):
    m = SAC.load(f"{run_dir}/sac_ems_best")
    env_c = make_env(clip=True); env_o = make_env(clip=False)
    oc, _ = env_c.reset(); oo, _ = env_o.reset()
    Rc = Ro = 0.0; socs = []; n_above = 0; n = 0
    while True:
        soc_before = env_c._Q_BT / _Q_BT_0
        socs.append(soc_before)
        if soc_before > NEDC_THR:
            n_above += 1
        a, _ = m.predict(oc, deterministic=True)
        oc, rc, tc, _, _ = env_c.step(a)
        oo, ro, to, _, _ = env_o.step(a)
        Rc += rc; Ro += ro; n += 1
        if tc:
            break
    return dict(run=run_dir, n_transitions=n, soc_max=float(max(socs)),
               pct_above_threshold=round(100 * n_above / n, 4),
               cum_R_corrected=round(Rc, 4), cum_R_original=round(Ro, 4),
               dR=round(Rc - Ro, 6), n_affected=int(n_above),
               threshold_pct=round(NEDC_THR * 100, 3))


# ---------------------------------------------------------------- B9 vehicle
def vehicle_eval(run_dir):
    r = evaluate(checkpoint=f"{run_dir}/sac_ems_best", cycle="NEDC", controller="rl",
                 eq_factor=EQF, k_fb=KFB, action_map=AMAP, lookahead=LA)
    return {k: r[k] for k in ("v_ce_equiv", "v_liter", "soc_init", "soc_final", "d_soc_pp",
                              "soc_min", "soc_max", "charge_sustaining", "off_pct", "assist_pct",
                              "lps_pct", "only_pct", "regen_pct", "constraint_violations",
                              "battery_throughput_kJ", "engine_on_time_s")}


if __name__ == "__main__":
    out = {"threshold_pct_NEDC": round(NEDC_THR * 100, 4), "B5_coverage": {}, "B6_critic_argmax": {},
           "B7_stability": {}, "B8_dual_reward": {}, "B9_vehicle": {},
           "control_reference": {}}

    # fixed diagnostic states from a CONTROL rollout (band-driven, model-independent)
    m0 = SAC.load(f"{CONTROL[0]}/sac_ems_best")
    states = cache_physics(collect_states(m0))

    # ---- CONTROL reference (B6/B7/B8/B9) ----
    print("== CONTROL reference ==")
    ctrl_rows = {}
    for cd in CONTROL:
        m = SAC.load(f"{cd}/sac_ems_best"); ec = float(th.exp(m.log_ent_coef.detach()))
        ctrl_rows[cd] = agg(critic_eval(m, states, ec))
    out["control_reference"]["B6_agg"] = ctrl_rows
    out["control_reference"]["B5_coverage"] = {cd: coverage_for_buffer(cd) for cd in CONTROL}
    out["control_reference"]["B9_vehicle"] = {cd: vehicle_eval(cd) for cd in CONTROL}

    # ---- Stage B seeds ----
    for s in SEEDS:
        rd = str(SB / f"seed{s}")
        print(f"== seed {s} ==")
        out["B5_coverage"][f"seed{s}"] = coverage_for_buffer(rd)
        # B6/B7 across checkpoints
        perck = {}
        for ck in CKPTS:
            p = SB / f"seed{s}" / f"{ck}.zip"
            if not p.exists():
                continue
            m = SAC.load(str(p)[:-4]); ec = float(th.exp(m.log_ent_coef.detach()))
            perck[ck] = agg(critic_eval(m, states, ec))
        out["B6_critic_argmax"][f"seed{s}"] = perck
        # B7 stability = argmax movement across 100k->150k->best
        keys = [k for k in ("sac_ems_100k", "sac_ems_150k", "sac_ems_best") if k in perck]
        tq = [perck[k]["tce_Q"] for k in keys]
        ra = [perck[k]["resid_aR"] for k in keys]
        out["B7_stability"][f"seed{s}"] = dict(
            tce_Q_trajectory={k: perck[k]["tce_Q"] for k in perck},
            resid_aR_trajectory={k: perck[k]["resid_aR"] for k in perck},
            tce_Q_std_last=round(float(np.std(tq)), 3) if len(tq) > 1 else None,
            resid_aR_last=perck.get("sac_ems_best", {}).get("resid_aR"),
            minq_mag_best=perck.get("sac_ems_best", {}).get("minq_mag"),
        )
        out["B8_dual_reward"][f"seed{s}"] = dual_reward_audit(rd)
        out["B9_vehicle"][f"seed{s}"] = vehicle_eval(rd)

    def _js(o):
        import numpy as _np
        if isinstance(o, (_np.bool_,)):
            return bool(o)
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
        return str(o)
    (SB / "diagnostics.json").write_text(json.dumps(out, indent=2, default=_js))

    # -------- console --------
    print("\n================ B5  replay coverage: frac(executed T_CE >= 50 Nm) by demand band")
    print(f"{'run':<26} {'15-25':>8} {'25-30':>8} {'30-35':>8} {'35-50':>8}")
    for cd in CONTROL:
        c = out['control_reference']['B5_coverage'][cd]
        print(f"{'CTRL '+cd.split('/')[0]:<26} " + " ".join(f"{c[b]['frac_TCE_ge_50']:>8}" for b, _, _ in BANDS))
    for s in SEEDS:
        c = out['B5_coverage'][f'seed{s}']
        print(f"{'12B seed'+str(s):<26} " + " ".join(f"{c[b]['frac_TCE_ge_50']:>8}" for b, _, _ in BANDS))

    print("\n================ B6/B7  critic argmax T_CE (CORE 25-50 Nm) + Bellman residual @ a_R*")
    print(f"{'run/ckpt':<34} {'tce_Q':>7} {'tce_R':>7} {'tce_pi':>7} {'dT(R-Q)':>8} {'dT(Q-pi)':>9} {'resid_aR':>9}")
    for cd in CONTROL:
        a = out['control_reference']['B6_agg'][cd]
        print(f"{'CTRL '+cd.split('/')[0]:<34} {a['tce_Q']:>7} {a['tce_R']:>7} {a['tce_pi']:>7} "
              f"{a['dT_R_Q']:>8} {a['dT_Q_pi']:>9} {a['resid_aR']:>9}")
    for s in SEEDS:
        for ck, a in out['B6_critic_argmax'][f'seed{s}'].items():
            print(f"{'12B s'+str(s)+' '+ck:<34} {a['tce_Q']:>7} {a['tce_R']:>7} {a['tce_pi']:>7} "
                  f"{a['dT_R_Q']:>8} {a['dT_Q_pi']:>9} {a['resid_aR']:>9}")

    print("\n================ B8  dual-reward audit (best ckpt eval)")
    for s in SEEDS:
        d = out['B8_dual_reward'][f'seed{s}']
        print(f"  seed{s}: SoC_max={d['soc_max']*100:.1f}%  %>thr={d['pct_above_threshold']}  "
              f"cumR_corr={d['cum_R_corrected']}  cumR_orig={d['cum_R_original']}  dR={d['dR']}  n_affected={d['n_affected']}")

    print("\n================ B9  vehicle-level (best ckpt, NEDC)   [CONTROL mean 3.7666, RB 3.5056, ECMS 3.1887]")
    print(f"{'run':<14} {'v_ce_equiv':>11} {'dSoC_pp':>8} {'CS':>4} {'viol':>5} {'OFF%':>6} {'ASSIST%':>8} {'LPS%':>6} {'eng_on_s':>9}")
    for cd in CONTROL:
        v = out['control_reference']['B9_vehicle'][cd]
        print(f"{'CTRL '+cd.split('/')[0][:8]:<14} {v['v_ce_equiv']:>11.4f} {v['d_soc_pp']:>8.2f} "
              f"{str(v['charge_sustaining']):>4} {v['constraint_violations']:>5} {v['off_pct']:>6.1f} "
              f"{v['assist_pct']:>8.1f} {v['lps_pct']:>6.1f} {v['engine_on_time_s']:>9}")
    vs = []
    for s in SEEDS:
        v = out['B9_vehicle'][f'seed{s}']; vs.append(v['v_ce_equiv'])
        print(f"{'12B seed'+str(s):<14} {v['v_ce_equiv']:>11.4f} {v['d_soc_pp']:>8.2f} "
              f"{str(v['charge_sustaining']):>4} {v['constraint_violations']:>5} {v['off_pct']:>6.1f} "
              f"{v['assist_pct']:>8.1f} {v['lps_pct']:>6.1f} {v['engine_on_time_s']:>9}")
    print(f"  12B mean V_CE = {np.mean(vs):.4f} +/- {np.std(vs):.4f}   (CONTROL 3.7666)")
    print("\n[saved] results/phase12/stage_b/diagnostics.json")
