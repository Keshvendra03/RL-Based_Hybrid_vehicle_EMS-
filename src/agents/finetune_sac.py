"""
finetune_sac.py
===============
GUIDED / LEGACY — not part of the default unguided pipeline (see
train_sac.py's module docstring). Kept only for reference/comparison; not
invoked by any current workflow. NOTE: its call to EvalAndCheckpoint(...) is
missing the now-required eq_factor/soc_deadband arguments (this predates the
train_sac.py consolidation and was never updated) -- treat this script as
broken until that call site is fixed, if it's ever needed again.

Fine-tune the behaviourally-cloned SAC policy WITHOUT destroying it.

Naive SAC fine-tuning of a cloned actor collapses immediately: the critic
is random, so its policy-gradient is noise, and SAC's automatic entropy
target drives the actor back toward a wide (random) policy. Both pull the
good clone apart in the first few thousand steps.

This script fixes that with three guards:
  1. CRITIC WARMUP - run gradient steps on the critics only (actor frozen)
     against the buffer of real benchmark transitions, so the value
     function is meaningful before the actor is allowed to move.
  2. LOW, FIXED ENTROPY - ent_coef is a small constant, not "auto", so the
     policy stays close to the deterministic clone and only explores
     locally.
  3. SMALL ACTOR LR + the BC policy kept as the best-so-far floor, so a bad
     update can never lose the warm-started result.

    python -m src.agents.finetune_sac --cycle NEDC --in models --out models \
        --timesteps 120000 --warmup 4000 --ent-coef 0.01 --actor-lr 5e-5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

from src.env.ems_env import EMSEnv, SOC_TARGET, TERM_TOL
from src.agents.train_sac import rollout_deterministic, score, EvalAndCheckpoint


def critic_warmup(model: SAC, steps: int, batch: int = 512):
    """Train critics only (actor & alpha frozen) on the seeded buffer."""
    if model.replay_buffer.size() == 0 or steps <= 0:
        return
    for p in model.actor.parameters():
        p.requires_grad_(False)
    for i in range(steps):
        data = model.replay_buffer.sample(batch)
        with torch.no_grad():
            next_a, next_logp = model.actor.action_log_prob(data.next_observations)
            next_q = torch.cat(model.critic_target(data.next_observations, next_a), dim=1)
            next_q, _ = torch.min(next_q, dim=1, keepdim=True)
            # fixed-entropy target backup
            ent = model.ent_coef_tensor
            next_q = next_q - ent * next_logp.reshape(-1, 1)
            target_q = data.rewards + (1 - data.dones) * model.gamma * next_q
        cur_q = model.critic(data.observations, data.actions)
        loss = 0.5 * sum(((q - target_q) ** 2).mean() for q in cur_q)
        model.critic.optimizer.zero_grad()
        loss.backward()
        model.critic.optimizer.step()
        polyak_update(model.critic.parameters(),
                      model.critic_target.parameters(), model.tau)
        if (i + 1) % 1000 == 0:
            print(f"  critic warmup {i+1}/{steps}: q_loss={loss.item():.4f}")
    for p in model.actor.parameters():
        p.requires_grad_(True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycle", default="NEDC", choices=["NEDC", "FTP75"])
    p.add_argument("--in", dest="in_dir", default="models")
    p.add_argument("--out", default="models")
    p.add_argument("--timesteps", type=int, default=120_000)
    p.add_argument("--warmup", type=int, default=4000)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--actor-lr", type=float, default=5e-5)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lambda-soc", type=float, default=8.0)
    p.add_argument("--soc-deadband", type=float, default=0.03)
    args = p.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = EMSEnv(args.cycle, lambda_soc=args.lambda_soc,
                 soc_deadband=args.soc_deadband)
    ep_len = env.cycle.length - 1

    # load BC seed with a FIXED low entropy coefficient and small actor LR
    model = SAC.load(
        in_dir / "sac_ems_last", env=env,
        custom_objects=dict(
            learning_rate=args.critic_lr,
            learning_starts=0,
        ),
    )
    model.load_replay_buffer(in_dir / "replay_buffer.pkl")
    # honor the fine-tune entropy coefficient (seed was saved fixed)
    model.ent_coef_tensor = torch.tensor(
        float(args.ent_coef), device=model.device)
    # separate, smaller LR for the actor so the clone moves slowly
    for g in model.actor.optimizer.param_groups:
        g["lr"] = args.actor_lr
    print(f"loaded BC seed (buffer {model.replay_buffer.size()}, "
          f"ent_coef={args.ent_coef}, actor_lr={args.actor_lr})")

    # establish the BC policy as the best-so-far floor
    cb = EvalAndCheckpoint(args.cycle, every_steps=5 * ep_len, out_dir=out_dir)
    bc_final = rollout_deterministic(model, args.cycle)
    cb.best = score(bc_final)
    model.save(out_dir / "sac_ems_best")
    (out_dir / "best_score.txt").write_text(str(cb.best))
    print(f"[BC seed] V_liter={bc_final['v_liter']:.3f}  "
          f"V_CE_equiv={bc_final['v_ce_equiv']:.3f}  "
          f"SoC={bc_final['soc_final']*100:.1f}%  score={cb.best:.3f}")

    # 1) critic warmup
    critic_warmup(model, args.warmup)
    wf = rollout_deterministic(model, args.cycle)
    print(f"[after warmup] V_liter={wf['v_liter']:.3f}  "
          f"SoC={wf['soc_final']*100:.1f}%  (actor unchanged by warmup)")

    # 2) gentle SAC fine-tuning
    model.learn(total_timesteps=args.timesteps, callback=cb,
                progress_bar=False, reset_num_timesteps=True)
    model.save(out_dir / "sac_ems_last")
    model.save_replay_buffer(out_dir / "replay_buffer.pkl")

    final = rollout_deterministic(model, args.cycle)
    s = score(final)
    if s < cb.best:
        cb.best = s
        model.save(out_dir / "sac_ems_best")
        (out_dir / "best_score.txt").write_text(str(cb.best))
    print(f"\n[final] V_liter={final['v_liter']:.3f}  "
          f"V_CE_equiv={final['v_ce_equiv']:.3f}  "
          f"SoC={final['soc_final']*100:.1f}%  score={s:.3f}")
    print("best score:", cb.best)


if __name__ == "__main__":
    main()
