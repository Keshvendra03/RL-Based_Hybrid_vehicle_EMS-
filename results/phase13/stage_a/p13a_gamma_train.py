"""
PHASE 13 STAGE A  --  controlled gamma experiment (the ONLY training variable is gamma).

gamma in {0.20, 0.50, 0.90, 0.98} x seeds {0,1,2} x 150000 steps.
Frozen at the CONTROL config: modeaware_gated, eq_factor 0.2717, k_fb 2.5, n_step 1,
net [256,256], lr 3e-4, buffer 300k, batch 512, tau 0.005, train_freq 64,
gradient_steps 16, ent_coef auto, lookahead 5, clip_eq_eff=True (Stage-12A).
NO B2 exploration, NO replay injection, plain SAC + ReplayBuffer.

Per-run frozen-config + git-revision purity check (aborts if anything but gamma differs).

Outputs under results/phase13/stage_a/gamma/g{XX}/seed{s}/:
  sac_ems_{50k,100k,150k,best}.zip, sac_ems_last.zip, replay_buffer.pkl, eval_history.csv
  + g{XX}/seed{s}/frozen_config.json  and  results/phase13/stage_a/gamma/purity.json
"""
import json, csv, subprocess, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from src.env.ems_env import EMSEnv, SOC_TARGET, TERM_TOL

ROOT = Path("results/phase13/stage_a/gamma"); ROOT.mkdir(parents=True, exist_ok=True)
GAMMAS = [0.20, 0.50, 0.90, 0.98]
SEEDS = [0, 1, 2]
BASE = dict(cycle="NEDC", eq_factor=0.2717, k_fb=2.5, n_step=1,
            action_map="modeaware_gated", lookahead=5, lambda_soc=2.0, soc_deadband=0.10,
            target_entropy="auto", learning_rate=3e-4, buffer_size=300_000, batch_size=512,
            tau=0.005, train_freq=64, gradient_steps=16, net_arch=[256, 256],
            timesteps=150_000, clip_eq_eff=True,
            b2_exploration=False, replay_injection=False)


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def make_env():
    return EMSEnv(BASE["cycle"], eq_factor=BASE["eq_factor"], lambda_soc=BASE["lambda_soc"],
                 soc_deadband=BASE["soc_deadband"], lookahead=BASE["lookahead"],
                 k_fb=BASE["k_fb"], action_map=BASE["action_map"], clip_eq_eff=True)


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
        self.out, self.every, self.best, self.rows = out_dir, every, np.inf, []

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            fin = rollout_det(self.model); s = score(fin); best = s < self.best
            if best:
                self.best = s; self.model.save(self.out / "sac_ems_best")
            self.rows.append(dict(t=self.num_timesteps, v_ce_equiv=fin["v_ce_equiv"],
                                  soc_final=fin["soc_final"], score=s, is_best=best))
        for ms in (50_000, 100_000, 150_000):
            if self.num_timesteps == ms:
                self.model.save(self.out / f"sac_ems_{ms // 1000}k")
        return True


def build(gamma, seed):
    env = make_env(); ep_len = env.cycle.length - 1
    m = SAC("MlpPolicy", env, learning_rate=BASE["learning_rate"], buffer_size=BASE["buffer_size"],
            learning_starts=2 * ep_len, batch_size=BASE["batch_size"], tau=BASE["tau"],
            gamma=gamma, train_freq=BASE["train_freq"], gradient_steps=BASE["gradient_steps"],
            ent_coef="auto", target_entropy="auto",
            policy_kwargs=dict(net_arch=BASE["net_arch"]), seed=seed, verbose=0)
    return m, ep_len


def purity_check(gamma, seed, model):
    """Verify EVERYTHING except gamma matches BASE; abort otherwise."""
    c = dict(
        gamma=float(model.gamma), lr=model.lr_schedule(1.0), buffer_size=model.buffer_size,
        batch_size=model.batch_size, tau=model.tau, train_freq=str(model.train_freq),
        gradient_steps=model.gradient_steps, net_arch=str(model.policy_kwargs.get("net_arch")),
        ent_coef=str(model.ent_coef), target_entropy=str(model.target_entropy),
        seed=seed, clip_eq_eff=make_env().clip_eq_eff, action_map=BASE["action_map"],
        eq_factor=BASE["eq_factor"], k_fb=BASE["k_fb"], lookahead=BASE["lookahead"],
        obs_dim=int(model.observation_space.shape[0]), act_space=str(model.action_space),
        git=git_rev())
    ref = dict(lr=BASE["learning_rate"], buffer_size=BASE["buffer_size"], batch_size=BASE["batch_size"],
              tau=BASE["tau"], gradient_steps=BASE["gradient_steps"],
              net_arch=str(BASE["net_arch"]), seed=seed, clip_eq_eff=True,
              action_map=BASE["action_map"], eq_factor=BASE["eq_factor"], k_fb=BASE["k_fb"],
              lookahead=BASE["lookahead"], obs_dim=20, act_space="Box(-1.0, 1.0, (1,), float32)")
    diffs = {k: (c[k], ref[k]) for k in ref if str(c[k]) != str(ref[k])}
    ok = (len(diffs) == 0)
    return dict(config=c, ref=ref, diffs_other_than_gamma=diffs, PASS=ok)


def train_one(gamma, seed):
    tag = f"g{int(round(gamma*100)):02d}"
    sd = ROOT / tag / f"seed{seed}"; sd.mkdir(parents=True, exist_ok=True)
    m, ep_len = build(gamma, seed)
    pc = purity_check(gamma, seed, m)
    (sd / "frozen_config.json").write_text(json.dumps(pc, indent=2, default=str))
    if not pc["PASS"]:
        raise SystemExit(f"[PURITY FAIL] g={gamma} seed={seed}: {pc['diffs_other_than_gamma']}")
    cb = EvalCkpt(sd, 2 * ep_len)
    m.learn(total_timesteps=BASE["timesteps"], callback=CallbackList([cb]), progress_bar=False)
    m.save(sd / "sac_ems_last"); m.save_replay_buffer(sd / "replay_buffer.pkl")
    with open(sd / "eval_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "v_ce_equiv", "soc_final", "score", "is_best"])
        w.writeheader(); w.writerows(cb.rows)
    fin = rollout_det(SAC.load(sd / "sac_ems_best"))
    return dict(gamma=gamma, seed=seed, best_score=cb.best,
               final_eval_v_ce=fin["v_ce_equiv"], final_eval_soc=fin["soc_final"], n_evals=len(cb.rows))


if __name__ == "__main__":
    print(f"git = {git_rev()}")
    purity = {"git": git_rev(), "base_config": BASE, "runs": {}}
    summary = []
    for g in GAMMAS:
        for s in SEEDS:
            tag = f"g{int(round(g*100)):02d}_s{s}"
            print(f"--- {tag}")
            r = train_one(g, s)
            summary.append(r)
            purity["runs"][tag] = json.loads(
                (ROOT / f"g{int(round(g*100)):02d}" / f"seed{s}" / "frozen_config.json").read_text())
            print(f"    {r}")
    (ROOT / "purity.json").write_text(json.dumps(purity, indent=2, default=str))
    (ROOT / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n[done] results/phase13/stage_a/gamma/{purity.json, train_summary.json}")
