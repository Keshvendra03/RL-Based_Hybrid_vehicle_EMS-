"""
PHASE 12 STAGE B  --  targeted informative-coverage experiment.
B1 config freeze  ·  B3 action-path integrity dry-run  ·  B4 three-seed training.

Frozen exactly at the CONTROL config (train_sac.py CONTROL flag set), the ONLY
changes being:
  * reward: clip_eq_eff=True  (Stage-A verified safety correction, default-OFF
    flag; provable bitwise no-op on all CONTROL transitions)
  * training-time exploration: high-engine-load informative-coverage injection
    (results/phase12/te_highload.py), p=0.25, evaluation untouched.

Outputs under results/phase12/stage_b/:
  config_frozen.json, dry_run_checks.json,
  seed{0,1,2}/{sac_ems_best.zip, sac_ems_50k.zip, sac_ems_100k.zip,
              sac_ems_150k.zip, replay_buffer.pkl, te_stats.json,
              coverage_evolution.json, eval_history.csv}
"""
import json, csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from src.env.ems_env import EMSEnv, SOC_TARGET, TERM_TOL
from src.env.powertrain import _Q_BT_0, _T_CUTOFF
from results.phase12.te_highload import make_highload_targeted, decode_obs, HighLoadInterval

OUT = Path("results/phase12/stage_b"); OUT.mkdir(parents=True, exist_ok=True)

CYCLE = "NEDC"
FROZEN = dict(
    cycle=CYCLE, eq_factor=0.2717, k_fb=2.5, gamma=0.20, n_step=1,
    action_map="modeaware_gated", lookahead=5, lambda_soc=2.0, soc_deadband=0.10,
    target_entropy="auto", learning_rate=3e-4, buffer_size=300_000,
    batch_size=512, tau=0.005, train_freq=64, gradient_steps=16,
    net_arch=[256, 256], timesteps=150_000, seeds=[0, 1, 2],
    clip_eq_eff=True, te_prob=0.25,
    te_activation="15<=T_MGB<50 Nm AND w>0 AND soc<0.70 AND feasible high-load interval >=5 Nm wide",
    te_injection="a ~ Uniform over feasible T_CE in [1.3*demand, 0.9*T_CE_max_feasible] intersect [-1,1]",
    reward_change_vs_control="clip_eq_eff=True ONLY (Stage A); no eq_factor/k_fb/gamma/net/action/obs/opt change",
)


def make_env(seed=None):
    e = EMSEnv(CYCLE, eq_factor=FROZEN["eq_factor"], lambda_soc=FROZEN["lambda_soc"],
              soc_deadband=FROZEN["soc_deadband"], lookahead=FROZEN["lookahead"],
              k_fb=FROZEN["k_fb"], action_map=FROZEN["action_map"], clip_eq_eff=True)
    return e


def rollout_deterministic(model):
    env = make_env()
    obs, _ = env.reset()
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, _, info = env.step(a)
        if term:
            return info["episode_final"]


def score(fin):
    miss = max(abs(fin["soc_final"] - SOC_TARGET) - TERM_TOL, 0.0)
    return fin["v_ce_equiv"] + 10.0 * miss


class EvalCkpt(BaseCallback):
    def __init__(self, out_dir, every_steps):
        super().__init__(0)
        self.out = out_dir; self.every = every_steps
        self.best = np.inf; self.rows = []
        self.cov_evo = []      # (step, injected, feasible, in_region, steps, mean_inj_tce)

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            fin = rollout_deterministic(self.model)
            s = score(fin)
            is_best = s < self.best
            if is_best:
                self.best = s
                self.model.save(self.out / "sac_ems_best")
            self.rows.append(dict(t=self.num_timesteps, v_ce_equiv=fin["v_ce_equiv"],
                                  soc_final=fin["soc_final"], score=s, is_best=is_best))
            st = getattr(self.model, "te_stats", {})
            inj = st.get("injected", 0)
            self.cov_evo.append(dict(step=self.num_timesteps, **{k: st.get(k, 0) for k in
                                 ("steps", "in_region", "feasible", "injected")},
                                 mean_injected_tce=(st.get("injected_tce_sum", 0.0) / inj) if inj else None))
        for milestone in (50_000, 100_000, 150_000):
            if self.num_timesteps == milestone:
                self.model.save(self.out / f"sac_ems_{milestone // 1000}k")
        return True


def build_model(seed, te_enabled):
    cls = make_highload_targeted(SAC)
    env = make_env(seed)
    ep_len = env.cycle.length - 1
    m = cls(
        "MlpPolicy", env,
        learning_rate=FROZEN["learning_rate"], buffer_size=FROZEN["buffer_size"],
        learning_starts=max(2 * ep_len, 0), batch_size=FROZEN["batch_size"],
        tau=FROZEN["tau"], gamma=FROZEN["gamma"], train_freq=FROZEN["train_freq"],
        gradient_steps=FROZEN["gradient_steps"], ent_coef="auto", target_entropy="auto",
        policy_kwargs=dict(net_arch=FROZEN["net_arch"]), seed=seed, verbose=0,
        te_enabled=te_enabled, te_prob=FROZEN["te_prob"], te_cycle=CYCLE,
        te_action_map=FROZEN["action_map"], te_lookahead=FROZEN["lookahead"],
    )
    return m, ep_len


# --------------------------------------------------------------- B3 dry-run
def dry_run():
    checks = {}
    m, ep_len = build_model(seed=0, te_enabled=True)

    # (1) predict(deterministic=True) unchanged  &  (2)/(3) no intervention at eval
    st0 = dict(m.te_stats)
    env = make_env(); obs, _ = env.reset()
    for _ in range(500):
        a, _ = m.predict(obs, deterministic=True)
        assert -1.0 <= float(a[0]) <= 1.0, "predict produced OOB action"
        obs, _, term, _, _ = env.step(a)
        if term:
            obs, _ = env.reset()
    checks["predict_does_not_trigger_intervention"] = (m.te_stats == st0)
    checks["predict_class_is_stock_SAC"] = (type(m).predict is SAC.predict)

    # (4)/(5)/(6) intervention path: exercise _sample_action on real obs, verify feasibility
    probe = HighLoadInterval(CYCLE, FROZEN["action_map"], FROZEN["lookahead"])
    fresh = make_env()
    env2 = make_env(); obs2, _ = env2.reset()
    inj_actions, inj_tce, oob, infeasible = [], [], 0, 0
    m2, _ = build_model(seed=0, te_enabled=True)
    for _ in range(4000):
        m2._last_obs = obs2.reshape(1, -1).astype(np.float32)
        before = m2.te_stats["injected"]
        act, buf = m2._sample_action(0)
        if not (-1.0 <= float(act[0, 0]) <= 1.0):
            oob += 1
        if m2.te_stats["injected"] > before:
            a = float(act[0, 0]); inj_actions.append(a)
            T, w, dw, soc = decode_obs(obs2)
            fresh._demand = dict(w_MGB=w, dw_MGB=dw, T_MGB=T, d_T_MGB=0.0)
            fresh._Q_BT = soc * _Q_BT_0
            t_ce, t_em, u, mode = fresh._action_to_torques(np.array([a], np.float32))
            inj_tce.append(t_ce)
            # feasibility: engine ON at moderate/high load, within the intended band
            lo_nom, hi_nom = 1.3 * T, 0.9 * probe.interval(T, w, dw, soc)[1] if probe.interval(T, w, dw, soc) else (0, 0)
            if not (t_ce > _T_CUTOFF):
                infeasible += 1
        obs2, _, term, _, _ = env2.step(np.zeros(1, np.float32))
        if term:
            obs2, _ = env2.reset()
    checks["n_injected_in_dry_run"] = len(inj_actions)
    checks["all_injected_actions_in_bounds"] = (oob == 0)
    checks["all_injected_engine_ON"] = (infeasible == 0)
    checks["injected_action_range"] = [float(np.min(inj_actions)), float(np.max(inj_actions))] if inj_actions else None
    checks["injected_executed_TCE_range"] = [float(np.min(inj_tce)), float(np.max(inj_tce))] if inj_tce else None
    checks["injected_executed_TCE_mean"] = float(np.mean(inj_tce)) if inj_tce else None
    checks["action_space"] = str(m.action_space)
    checks["reachable_set_unchanged"] = (str(m.action_space) == "Box(-1.0, 1.0, (1,), float32)")

    # te_enabled=False must be a stock no-op
    m_off, _ = build_model(seed=0, te_enabled=False)
    env3 = make_env(); o3, _ = env3.reset()
    for _ in range(1000):
        m_off._last_obs = o3.reshape(1, -1).astype(np.float32)
        m_off._sample_action(0)
        o3, _, term, _, _ = env3.step(np.zeros(1, np.float32))
        if term:
            o3, _ = env3.reset()
    checks["te_disabled_never_injects"] = (m_off.te_stats["injected"] == 0)

    checks["PASS"] = bool(checks["predict_does_not_trigger_intervention"]
                          and checks["predict_class_is_stock_SAC"]
                          and checks["all_injected_actions_in_bounds"]
                          and checks["all_injected_engine_ON"]
                          and checks["reachable_set_unchanged"]
                          and checks["te_disabled_never_injects"]
                          and checks["n_injected_in_dry_run"] > 0)
    return checks


# --------------------------------------------------------------- B4 training
def train_seed(seed):
    sd = OUT / f"seed{seed}"; sd.mkdir(parents=True, exist_ok=True)
    m, ep_len = build_model(seed, te_enabled=True)
    cb = EvalCkpt(sd, every_steps=2 * ep_len)
    m.learn(total_timesteps=FROZEN["timesteps"], callback=CallbackList([cb]),
            progress_bar=False)
    m.save(sd / "sac_ems_last")
    m.save_replay_buffer(sd / "replay_buffer.pkl")
    (sd / "te_stats.json").write_text(json.dumps(m.te_stats, indent=2))
    (sd / "coverage_evolution.json").write_text(json.dumps(cb.cov_evo, indent=2))
    with open(sd / "eval_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "v_ce_equiv", "soc_final", "score", "is_best"])
        w.writeheader(); w.writerows(cb.rows)
    fin = rollout_deterministic(SAC.load(sd / "sac_ems_best"))
    return dict(seed=seed, best_score=cb.best, final_eval_v_ce=fin["v_ce_equiv"],
               final_eval_soc=fin["soc_final"], te_stats=m.te_stats,
               n_evals=len(cb.rows))


if __name__ == "__main__":
    (OUT / "config_frozen.json").write_text(json.dumps(FROZEN, indent=2))
    print("[B1] frozen config written -> results/phase12/stage_b/config_frozen.json")

    print("[B3] action-path integrity dry-run ...")
    checks = dry_run()
    (OUT / "dry_run_checks.json").write_text(json.dumps(checks, indent=2))
    for k, v in checks.items():
        print(f"     {k}: {v}")
    if not checks["PASS"]:
        raise SystemExit("[B3] DRY-RUN FAILED -- not training.")
    print("[B3] PASS")

    print("[B4] three-seed training (150k each) ...")
    results = []
    for s in FROZEN["seeds"]:
        print(f"  --- seed {s}")
        results.append(train_seed(s))
        print(f"      {results[-1]}")
    (OUT / "train_summary.json").write_text(json.dumps(results, indent=2))
    print("[B4] done -> results/phase12/stage_b/train_summary.json")
