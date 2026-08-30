"""
phase8_train_mixture.py
=======================
PHASE 8C -- train SAC with the 2-component mixture actor. EVERYTHING except the
actor policy class is frozen at the CONTROL values (Phase-8 brief section 4/8):

  gamma 0.20 | n_step 1 | modeaware_gated | k_fb 2.5 | eq_factor 0.2717/0.4981
  target_entropy auto | lr 3e-4 | batch 512 | buffer 300k | grad_steps 16
  train_freq 64 | tau 0.005 | net_arch [256,256] | lookahead 5 | 150k steps

    python -m results.phase8_train_mixture --cycle NEDC --seed 0 --out models_p8c_N0
"""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList

from src.env.ems_env import EMSEnv
from src.agents.train_sac import EvalAndCheckpoint, SACDiagnostics, rollout_deterministic, score
from results.phase8_mixture_policy import MixtureSACPolicy

EQF = {"NEDC": 0.2717, "FTP75": 0.4981}
RB = {"NEDC": 3.5056, "FTP75": 3.2323}
ECMSV = {"NEDC": 3.1887, "FTP75": 2.8097}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, choices=["NEDC", "FTP75"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-entropy", default="auto")
    ap.add_argument("--n-components", type=int, default=2)
    a = ap.parse_args()

    out_dir = Path(a.out) / a.cycle
    out_dir.mkdir(parents=True, exist_ok=True)

    env = EMSEnv(a.cycle, eq_factor=EQF[a.cycle], lambda_soc=2.0, soc_deadband=0.10,
                 lookahead=5, k_fb=2.5, action_map="modeaware_gated")
    ep_len = env.cycle.length - 1

    model = SAC(
        MixtureSACPolicy, env,
        learning_rate=3e-4, buffer_size=300_000,
        learning_starts=2 * ep_len, batch_size=512, tau=0.005, gamma=0.20,
        train_freq=64, gradient_steps=16,
        ent_coef="auto",
        target_entropy=("auto" if a.target_entropy == "auto" else float(a.target_entropy)),
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=a.seed, verbose=0,
    )

    cb = EvalAndCheckpoint([a.cycle], every_steps=2 * ep_len, out_dir=out_dir,
                           eq_factor=EQF[a.cycle], soc_deadband=0.10, lookahead=5,
                           k_fb=2.5, action_map="modeaware_gated", verbose=1)
    diag = SACDiagnostics(out_dir, batch_size=512, log_every=5000)

    (out_dir / "run_config.json").write_text(json.dumps({
        "cycle": a.cycle, "seed": a.seed, "timesteps": a.timesteps,
        "policy": "MixtureSACPolicy", "n_components": a.n_components,
        "gamma": 0.20, "n_step": 1, "action_map": "modeaware_gated", "k_fb": 2.5,
        "eq_factor": EQF[a.cycle], "target_entropy": a.target_entropy,
        "lr": 3e-4, "batch": 512, "buffer": 300000, "gradient_steps": 16,
        "lookahead": 5, "phase": "8C",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    }, indent=2))

    print(f"[8C] {a.cycle} seed{a.seed}  mixture K={a.n_components}  {a.timesteps} steps  "
          f"(RB {RB[a.cycle]}  ECMS {ECMSV[a.cycle]})")
    model.learn(total_timesteps=a.timesteps, callback=CallbackList([cb, diag]),
                progress_bar=False)
    model.save(out_dir / "sac_ems_last")
    model.save_replay_buffer(out_dir / "replay_buffer.pkl")

    fin = rollout_deterministic(model, a.cycle, EQF[a.cycle], 0.10, 5, 2.5, "modeaware_gated")
    print(f"[8C final] {a.cycle} seed{a.seed}  V_CE={fin['v_ce_equiv']:.4f}  "
          f"V_liter={fin['v_liter']:.4f}  SoC={fin['soc_final']*100:.2f}%  "
          f"score={score(fin):.4f}  best_score={cb.best:.4f}")
    json.dump({"final_last": fin, "best_score": cb.best},
              open(out_dir / "phase8c_final.json", "w"), indent=2)


if __name__ == "__main__":
    main()
