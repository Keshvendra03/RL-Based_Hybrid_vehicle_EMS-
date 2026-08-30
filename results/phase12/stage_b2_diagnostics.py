"""
PHASE 12 STAGE B2 diagnostics  --  NO TRAINING.

§13 state-normalized deep-LPS coverage (rho>=0.75 AND T_CE>=TCE_max_feas-15) by demand band
§14 critic argmax across the physical action grid at fixed states, per checkpoint
§15 Q-argmax stability (50k/100k/150k/best; mean/std/min/max)
§16 Bellman residual: overall / 25-35 Nm demand / deep-LPS actions / evolution / final
§17 a_R* vs a_Q* vs a_pi + deep-LPS coverage
§18 vehicle metrics
§19 dual-reward audit (R_patched vs R_original)

Reads results/phase12/stage_b2/seed{0,1,2}/* + the 3 NEDC CONTROL checkpoints.
Writes results/phase12/phase12_stage_b2_summary.json  (+ console).
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

SB = Path("results/phase12/stage_b2")
OUTJSON = Path("results/phase12/phase12_stage_b2_summary.json")
SEEDS = [0, 1, 2]
CKPTS = ["sac_ems_50k", "sac_ems_100k", "sac_ems_150k", "sac_ems_best"]
CONTROL = ["models_p5s0_k2.5/NEDC", "models_p5_k2.5/NEDC", "models_p5_k2.5_s2/NEDC"]
EQF, KFB, AMAP, LA = 0.2717, 2.5, "modeaware_gated", 5
GAMMA = 0.20
NEDC_THR = SOC_TARGET + EQF / KFB           # 0.60868  (old eq_eff<0 threshold)
N_A = 161
GRID_A = np.linspace(-1.0, 1.0, N_A)
BANDS = [("15-25", 15, 25), ("25-30", 25, 30), ("30-35", 30, 35), ("35-50", 35, 50)]
DEEP_WINDOW = 15.0


def band_of(T):
    for n, lo, hi in BANDS:
        if lo <= T < hi:
            return n
    return None


def make_env(clip=False):
    return EMSEnv("NEDC", eq_factor=EQF, k_fb=KFB, action_map=AMAP, lookahead=LA, clip_eq_eff=clip)


# ---------------------------------------------------------------- B5/§13 coverage
def coverage(run_dir):
    m = SAC.load(f"{run_dir}/sac_ems_best")
    m.load_replay_buffer(f"{run_dir}/replay_buffer.pkl")
    rb = m.replay_buffer
    n = rb.buffer_size if rb.full else rb.pos
    O = rb.observations[:n, 0, :]; A = rb.actions[:n, 0, :].reshape(-1)
    w = O[:, 0] * 300.0; dw = O[:, 1] * 60.0; T = O[:, 2] * 150.0; soc = (O[:, 4] + 1) / 2
    keep = np.where((T >= 12) & (T < 55) & (w > 0))[0]
    e = make_env()
    rho = np.full(len(keep), np.nan); tce_x = np.full(len(keep), np.nan)
    within15 = np.zeros(len(keep), bool); tmax_arr = np.full(len(keep), np.nan)
    for i, k in enumerate(keep):
        e._demand = dict(w_MGB=float(w[k]), dw_MGB=float(dw[k]), T_MGB=float(T[k]), d_T_MGB=0.0)
        e._Q_BT = float(soc[k]) * _Q_BT_0
        tce_grid = np.array([e._action_to_torques(np.array([a], np.float32))[0] for a in GRID_A])
        tmin, tmax = float(tce_grid.min()), float(tce_grid.max())
        t_ce_exec = e._action_to_torques(np.array([A[k]], np.float32))[0]
        rho[i] = (t_ce_exec - tmin) / max(tmax - tmin, 1e-9)
        tce_x[i] = t_ce_exec; tmax_arr[i] = tmax
        within15[i] = t_ce_exec >= tmax - DEEP_WINDOW
    Tk = T[keep]
    out = {}
    for bn, lo, hi in BANDS:
        inb = (Tk >= lo) & (Tk < hi); nb = int(inb.sum())
        deep_rho = int(((rho >= 0.75) & inb).sum())
        deep_15 = int((within15 & inb).sum())
        out[bn] = dict(n=nb,
                       frac_deepLPS_rho_ge_0p75=round(deep_rho / nb, 4) if nb else None,
                       frac_within_15Nm_of_max=round(deep_15 / nb, 4) if nb else None,
                       mean_executed_tce=round(float(np.mean(tce_x[inb])), 2) if nb else None,
                       mean_tce_max_feasible=round(float(np.mean(tmax_arr[inb])), 2) if nb else None,
                       frac_TCE_ge_50=round(float(np.mean((tce_x >= 50) & inb)) / (nb / len(keep)), 4) if nb else None)
    return out


# ---------------------------------------------------------------- states + physics
def collect_states(m0):
    env = make_env(); obs, _ = env.reset()
    b = {n: [] for n, _, _ in BANDS}
    while True:
        d = env._demand; w, T = d["w_MGB"], d["T_MGB"]; bb = band_of(T)
        if bb and w > 0 and len(b[bb]) < 60:
            b[bb].append(dict(env=copy.deepcopy(env), obs=obs.copy(), band=bb,
                              w=w, dw=d["dw_MGB"], T=T, soc=env._Q_BT / _Q_BT_0))
        a, _ = m0.predict(obs, deterministic=True)
        obs, r, term, _, info = env.step(a)
        if term:
            break
    out = []
    for n in b:
        if b[n]:
            idx = np.linspace(0, len(b[n]) - 1, min(5, len(b[n]))).astype(int)
            out += [b[n][i] for i in idx]
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
        st["tce_max"] = float(st["tce"].max()); st["tce_min"] = float(st["tce"].min())
        st["deep_mask"] = st["tce"] >= (st["tce_max"] - DEEP_WINDOW)
    return states


@th.no_grad()
def critic_eval(model, states, ent_coef):
    a_nc = []
    for st in states:
        NO = th.as_tensor(st["next_obs"], dtype=th.float32)
        ad = model.actor(NO, deterministic=True).squeeze(1).numpy()
        mean, ls, _ = model.actor.get_action_dist_params(NO)
        lp = model.actor.action_dist.proba_distribution(mean, ls).log_prob(
            th.as_tensor(ad[:, None], dtype=th.float32)).numpy()
        a_nc.append((ad, lp))
    rows = []
    for si, st in enumerate(states):
        O = th.as_tensor(np.repeat(st["obs"][None, :], N_A, 0), dtype=th.float32)
        q1, q2 = model.critic(O, th.as_tensor(GRID_A[:, None], dtype=th.float32))
        minq = np.minimum(q1.squeeze(1).numpy(), q2.squeeze(1).numpy())
        iQ = int(np.argmax(minq)); iR = st["iR"]
        ad, lp = a_nc[si]
        NO = th.as_tensor(st["next_obs"], dtype=th.float32)
        nq1, nq2 = model.critic(NO, th.as_tensor(ad[:, None], dtype=th.float32))
        Vn = np.minimum(nq1.squeeze(1).numpy(), nq2.squeeze(1).numpy()) - ent_coef * lp
        Qt = st["r"] + (1 - st["done"].astype(float)) * GAMMA * Vn
        resid = minq - Qt
        a_pi = float(model.actor(th.as_tensor(st["obs"][None, :], dtype=th.float32),
                                 deterministic=True).squeeze().item())
        i_pi = int(np.argmin(np.abs(GRID_A - a_pi)))
        dm = st["deep_mask"]
        rows.append(dict(band=st["band"], T=round(float(st["T"]), 1), soc=round(float(st["soc"]), 3),
                         a_R=round(float(GRID_A[iR]), 4), a_Q=round(float(GRID_A[iQ]), 4), a_pi=round(a_pi, 4),
                         tce_R=round(float(st["tce"][iR]), 1), tce_Q=round(float(st["tce"][iQ]), 1),
                         tce_pi=round(float(st["tce"][i_pi]), 1),
                         dT_R_Q=round(float(st["tce"][iR] - st["tce"][iQ]), 1),
                         dT_Q_pi=round(float(st["tce"][iQ] - st["tce"][i_pi]), 1),
                         resid_aR=round(float(resid[iR]), 5),
                         resid_deep=round(float(resid[dm].mean()), 5) if dm.any() else None,
                         q_argmax_is_deep=bool(dm[iQ])))
    return rows


def agg(rows, f=lambda r: r["band"] in ("25-30", "30-35", "35-50")):
    R = [r for r in rows if f(r)]
    def m(k):
        v = [r[k] for r in R if r[k] is not None]
        return round(float(np.mean(v)), 4) if v else None
    return dict(n=len(R), tce_Q=m("tce_Q"), tce_R=m("tce_R"), tce_pi=m("tce_pi"),
               dT_R_Q=m("dT_R_Q"), dT_Q_pi=m("dT_Q_pi"), resid_aR=m("resid_aR"),
               resid_deep=m("resid_deep"),
               frac_Q_argmax_deep=round(float(np.mean([r["q_argmax_is_deep"] for r in R])), 3) if R else None)


def agg_2535(rows):
    return agg(rows, f=lambda r: r["band"] in ("25-30", "30-35"))


# ---------------------------------------------------------------- §19 dual reward
def dual_reward(run_dir):
    m = SAC.load(f"{run_dir}/sac_ems_best")
    ec, eo = make_env(clip=True), make_env(clip=False)
    oc, _ = ec.reset(); oo, _ = eo.reset()
    Rc = Ro = 0.0; socs = []; nab = 0; n = 0
    while True:
        sb = ec._Q_BT / _Q_BT_0; socs.append(sb); nab += int(sb > NEDC_THR)
        a, _ = m.predict(oc, deterministic=True)
        oc, rc, tc, _, _ = ec.step(a); oo, ro, to, _, _ = eo.step(a)
        Rc += rc; Ro += ro; n += 1
        if tc:
            break
    return dict(n=n, soc_max=round(float(max(socs)), 4),
                pct_above_old_threshold=round(100 * nab / n, 4),
                cum_R_patched=round(Rc, 4), cum_R_original=round(Ro, 4),
                dR=round(Rc - Ro, 6), n_affected=int(nab), old_threshold_pct=round(NEDC_THR * 100, 3))


def vehicle(run_dir):
    r = evaluate(checkpoint=f"{run_dir}/sac_ems_best", cycle="NEDC", controller="rl",
                 eq_factor=EQF, k_fb=KFB, action_map=AMAP, lookahead=LA)
    return {k: (bool(r[k]) if k == "charge_sustaining" else r[k]) for k in
            ("v_ce_equiv", "v_liter", "soc_init", "soc_final", "d_soc_pp", "soc_min", "soc_max",
             "charge_sustaining", "off_pct", "assist_pct", "lps_pct", "only_pct", "regen_pct",
             "constraint_violations", "battery_throughput_kJ", "engine_on_time_s")}


def _js(o):
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


if __name__ == "__main__":
    out = dict(old_eq_eff_threshold_pct=round(NEDC_THR * 100, 4),
               train_summary=json.loads((SB / "train_summary.json").read_text()),
               B5_coverage={}, B6_B7_critic={}, B8_dual_reward={}, B9_vehicle={},
               control_reference={})

    m0 = SAC.load(f"{CONTROL[0]}/sac_ems_best")
    states = cache_physics(collect_states(m0))

    print("== CONTROL reference ==")
    out["control_reference"]["B5_coverage"] = {cd: coverage(cd) for cd in CONTROL}
    out["control_reference"]["B6_agg"] = {}
    out["control_reference"]["B6_agg_2535"] = {}
    out["control_reference"]["B9_vehicle"] = {}
    for cd in CONTROL:
        m = SAC.load(f"{cd}/sac_ems_best"); ec = float(th.exp(m.log_ent_coef.detach()))
        rows = critic_eval(m, states, ec)
        out["control_reference"]["B6_agg"][cd] = agg(rows)
        out["control_reference"]["B6_agg_2535"][cd] = agg_2535(rows)
        out["control_reference"]["B9_vehicle"][cd] = vehicle(cd)

    for s in SEEDS:
        rd = str(SB / f"seed{s}")
        print(f"== seed {s} ==")
        out["B5_coverage"][f"seed{s}"] = coverage(rd)
        perck = {}
        for ck in CKPTS:
            p = SB / f"seed{s}" / f"{ck}.zip"
            if not p.exists():
                continue
            m = SAC.load(str(p)[:-4]); ec = float(th.exp(m.log_ent_coef.detach()))
            rows = critic_eval(m, states, ec)
            perck[ck] = dict(core=agg(rows), crit2535=agg_2535(rows))
        # stability across 50k/100k/150k/best
        tq = [perck[k]["core"]["tce_Q"] for k in perck if perck[k]["core"]["tce_Q"] is not None]
        ra = [perck[k]["core"]["resid_aR"] for k in perck if perck[k]["core"]["resid_aR"] is not None]
        out["B6_B7_critic"][f"seed{s}"] = dict(
            per_checkpoint=perck,
            tce_Q_trajectory={k: perck[k]["core"]["tce_Q"] for k in perck},
            resid_aR_trajectory={k: perck[k]["core"]["resid_aR"] for k in perck},
            tce_Q_stability=dict(mean=round(float(np.mean(tq)), 2), std=round(float(np.std(tq)), 2),
                                 min=round(float(np.min(tq)), 2), max=round(float(np.max(tq)), 2)) if tq else None,
            resid_aR_best=perck.get("sac_ems_best", {}).get("core", {}).get("resid_aR"),
            resid_deep_best=perck.get("sac_ems_best", {}).get("core", {}).get("resid_deep"),
            frac_Q_argmax_deep_best=perck.get("sac_ems_best", {}).get("core", {}).get("frac_Q_argmax_deep"),
        )
        out["B8_dual_reward"][f"seed{s}"] = dual_reward(rd)
        out["B9_vehicle"][f"seed{s}"] = vehicle(rd)

    OUTJSON.write_text(json.dumps(out, indent=2, default=_js))

    # ---- console ----
    print("\n================ §13  state-normalized DEEP-LPS coverage  frac(rho>=0.75) / frac(within 15Nm of TCE_max)")
    print(f"{'run':<22} {'25-30':>16} {'30-35':>16} {'35-50':>16}")
    def cov_cell(c, b):
        return f"{c[b]['frac_deepLPS_rho_ge_0p75']}/{c[b]['frac_within_15Nm_of_max']}"
    for cd in CONTROL:
        c = out['control_reference']['B5_coverage'][cd]
        print(f"{'CTRL '+cd.split('/')[0][-3:]:<22} " +
              " ".join(f"{cov_cell(c, b):>16}" for b in ('25-30', '30-35', '35-50')))
    for s in SEEDS:
        c = out['B5_coverage'][f'seed{s}']
        print(f"{'B2 seed'+str(s):<22} " +
              " ".join(f"{cov_cell(c, b):>16}" for b in ('25-30', '30-35', '35-50')))

    print("\n================ §14/§15/§16  critic argmax T_CE (CORE 25-50) + Bellman residual")
    print(f"{'run/ckpt':<30} {'tce_Q':>7} {'tce_R':>7} {'tce_pi':>7} {'dT(R-Q)':>8} {'resid_aR':>9} {'resid_deep':>11} {'Q@deep%':>8}")
    for cd in CONTROL:
        a = out['control_reference']['B6_agg'][cd]
        print(f"{'CTRL '+cd.split('/')[0][-3:]:<30} {a['tce_Q']:>7} {a['tce_R']:>7} {a['tce_pi']:>7} {a['dT_R_Q']:>8} "
              f"{a['resid_aR']:>9} {str(a['resid_deep']):>11} {str(a['frac_Q_argmax_deep']):>8}")
    for s in SEEDS:
        for ck, a in ((k, v['core']) for k, v in out['B6_B7_critic'][f'seed{s}']['per_checkpoint'].items()):
            print(f"{'B2 s'+str(s)+' '+ck:<30} {a['tce_Q']:>7} {a['tce_R']:>7} {a['tce_pi']:>7} {a['dT_R_Q']:>8} "
                  f"{a['resid_aR']:>9} {str(a['resid_deep']):>11} {str(a['frac_Q_argmax_deep']):>8}")
        st = out['B6_B7_critic'][f'seed{s}']['tce_Q_stability']
        print(f"     -> seed{s} tce_Q stability: mean={st['mean']} std={st['std']} [{st['min']},{st['max']}]")

    print("\n================ §19  dual-reward audit (best ckpt)")
    for s in SEEDS:
        d = out['B8_dual_reward'][f'seed{s}']
        print(f"  seed{s}: SoC_max={d['soc_max']*100:.1f}%  %>old_thr={d['pct_above_old_threshold']}  "
              f"cumR_patched={d['cum_R_patched']}  cumR_original={d['cum_R_original']}  dR={d['dR']}  n_affected={d['n_affected']}")

    print("\n================ §18  vehicle (best ckpt)  [CONTROL mean 3.7666, RB 3.5056, ECMS 3.1887]")
    print(f"{'run':<14} {'v_ce_equiv':>11} {'dSoC_pp':>8} {'CS':>6} {'viol':>5} {'OFF%':>6} {'LPS%':>6} {'eng_on_s':>9}")
    for cd in CONTROL:
        v = out['control_reference']['B9_vehicle'][cd]
        print(f"{'CTRL '+cd.split('/')[0][-3:]:<14} {v['v_ce_equiv']:>11.4f} {v['d_soc_pp']:>8.2f} {str(v['charge_sustaining']):>6} "
              f"{v['constraint_violations']:>5} {v['off_pct']:>6.1f} {v['lps_pct']:>6.1f} {v['engine_on_time_s']:>9}")
    vs = []
    for s in SEEDS:
        v = out['B9_vehicle'][f'seed{s}']; vs.append(v['v_ce_equiv'])
        print(f"{'B2 seed'+str(s):<14} {v['v_ce_equiv']:>11.4f} {v['d_soc_pp']:>8.2f} {str(v['charge_sustaining']):>6} "
              f"{v['constraint_violations']:>5} {v['off_pct']:>6.1f} {v['lps_pct']:>6.1f} {v['engine_on_time_s']:>9}")
    print(f"  B2 mean V_CE = {np.mean(vs):.4f} +/- {np.std(vs):.4f}")
    print("\n[saved] results/phase12/phase12_stage_b2_summary.json")
