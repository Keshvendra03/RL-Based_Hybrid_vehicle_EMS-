"""
PHASE 12 STAGE B2  --  deep-LPS coverage falsification: three-seed training.

Runs ONLY after the pre-flight report passes. Identical to Phase-12B training
except the exploration intervention is the DEEP-LPS one (te_highload_b2.py):
inject uniformly from the TOP 15 Nm of the state-dependent feasible engine-torque
range.  clip_eq_eff=True (Stage-A correction).  Everything else frozen.

Outputs under results/phase12/stage_b2/:
  config_frozen.json,
  seed{0,1,2}/{sac_ems_best.zip, sac_ems_{50k,100k,150k}.zip, sac_ems_last.zip,
              replay_buffer.pkl, te_stats.json, te_events.json,
              coverage_evolution.json, eval_history.csv}
  train_summary.json
"""
import json, csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from src.env.ems_env import EMSEnv, SOC_TARGET, TERM_TOL
from results.phase12.te_highload_b2 import make_deeplps_targeted

OUT = Path("results/phase12/stage_b2"); OUT.mkdir(parents=True, exist_ok=True)
CYCLE = "NEDC"
FROZEN = dict(
    cycle=CYCLE, eq_factor=0.2717, k_fb=2.5, gamma=0.20, n_step=1,
    action_map="modeaware_gated", lookahead=5, lambda_soc=2.0, soc_deadband=0.10,
    target_entropy="auto", learning_rate=3e-4, buffer_size=300_000, batch_size=512,
    tau=0.005, train_freq=64, gradient_steps=16, net_arch=[256, 256],
    timesteps=150_000, seeds=[0, 1, 2], clip_eq_eff=True, te_prob=0.25,
    te_soc_cap=0.55, te_deep_window_Nm=15.0,
    te_interval="[max(TCE_min_feasible, TCE_max_feasible - 15), TCE_max_feasible]  (env-authoritative)",
    reward_change_vs_control="clip_eq_eff=True ONLY (Phase 12A); nothing else",
    intervention_vs_12B="lower bound = TCE_max_feasible-15 (not 1.3*demand); SoC cap 0.55 (not 0.70)",
)


def make_env():
    return EMSEnv(CYCLE, eq_factor=FROZEN["eq_factor"], lambda_soc=FROZEN["lambda_soc"],
                 soc_deadband=FROZEN["soc_deadband"], lookahead=FROZEN["lookahead"],
                 k_fb=FROZEN["k_fb"], action_map=FROZEN["action_map"], clip_eq_eff=True)


def rollout_det(model):
    env = make_env(); obs, _ = env.reset()
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, _, info = env.step(a)
        if term:
            return info["episode_final"]


def score(fin):
    return fin["v_ce_equiv"] + 10.0 * max(abs(fin["soc_final"] - SOC_TARGET) - TERM_TOL, 0.0)


class EvalCkpt(BaseCallback):
    def __init__(self, out_dir, every):
        super().__init__(0)
        self.out, self.every, self.best = out_dir, every, np.inf
        self.rows, self.cov = [], []

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            fin = rollout_det(self.model); s = score(fin)
            best = s < self.best
            if best:
                self.best = s; self.model.save(self.out / "sac_ems_best")
            self.rows.append(dict(t=self.num_timesteps, v_ce_equiv=fin["v_ce_equiv"],
                                  soc_final=fin["soc_final"], score=s, is_best=best))
            st = getattr(self.model, "te_stats", {})
            inj = st.get("injected", 0)
            self.cov.append(dict(step=self.num_timesteps,
                **{k: st.get(k, 0) for k in ("steps", "in_region", "feasible", "injected", "n_clamped")},
                mean_exec_tce=(st.get("tce_exec_sum", 0.0) / inj) if inj else None,
                mean_rho=(st.get("rho_sum", 0.0) / inj) if inj else None,
                mean_fidelity=(st.get("fidelity_sum", 0.0) / inj) if inj else None))
        for ms in (50_000, 100_000, 150_000):
            if self.num_timesteps == ms:
                self.model.save(self.out / f"sac_ems_{ms // 1000}k")
        return True


def build(seed):
    cls = make_deeplps_targeted(SAC)
    env = make_env(); ep_len = env.cycle.length - 1
    m = cls("MlpPolicy", env, learning_rate=FROZEN["learning_rate"],
            buffer_size=FROZEN["buffer_size"], learning_starts=2 * ep_len,
            batch_size=FROZEN["batch_size"], tau=FROZEN["tau"], gamma=FROZEN["gamma"],
            train_freq=FROZEN["train_freq"], gradient_steps=FROZEN["gradient_steps"],
            ent_coef="auto", target_entropy="auto",
            policy_kwargs=dict(net_arch=FROZEN["net_arch"]), seed=seed, verbose=0,
            te_enabled=True, te_prob=FROZEN["te_prob"], te_cycle=CYCLE,
            te_action_map=FROZEN["action_map"], te_lookahead=FROZEN["lookahead"])
    return m, ep_len


def train_seed(seed):
    sd = OUT / f"seed{seed}"; sd.mkdir(parents=True, exist_ok=True)
    m, ep_len = build(seed)
    cb = EvalCkpt(sd, 2 * ep_len)
    m.learn(total_timesteps=FROZEN["timesteps"], callback=CallbackList([cb]), progress_bar=False)
    m.save(sd / "sac_ems_last")
    m.save_replay_buffer(sd / "replay_buffer.pkl")
    (sd / "te_stats.json").write_text(json.dumps(m.te_stats, indent=2))
    (sd / "te_events.json").write_text(json.dumps(m.te_events[:8000], indent=2))
    (sd / "coverage_evolution.json").write_text(json.dumps(cb.cov, indent=2))
    with open(sd / "eval_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "v_ce_equiv", "soc_final", "score", "is_best"])
        w.writeheader(); w.writerows(cb.rows)
    fin = rollout_det(SAC.load(sd / "sac_ems_best"))
    return dict(seed=seed, best_score=cb.best, final_eval_v_ce=fin["v_ce_equiv"],
               final_eval_soc=fin["soc_final"], te_stats=m.te_stats, n_evals=len(cb.rows))


if __name__ == "__main__":
    (OUT / "config_frozen.json").write_text(json.dumps(FROZEN, indent=2))
    pf = Path("results/phase12/data/stage_b2_preflight.json")
    assert pf.exists() and json.loads(pf.read_text())["acceptance_criteria"]["ALL_PASS"], \
        "PRE-FLIGHT NOT PASSED -- refusing to train."
    print("[B2] pre-flight PASS confirmed; training 3 seeds x 150k ...")
    res = []
    for s in FROZEN["seeds"]:
        print(f"  --- seed {s}")
        res.append(train_seed(s)); print(f"      {res[-1]}")
    (OUT / "train_summary.json").write_text(json.dumps(res, indent=2, default=str))
    print("[B2] done -> results/phase12/stage_b2/train_summary.json")
