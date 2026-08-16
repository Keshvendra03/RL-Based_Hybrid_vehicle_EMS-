"""
finetune_bcreg.py
=================
BC-anchored online fine-tuning of the cloned SAC policy (TD3+BC style).

Plain SAC fine-tuning of a behaviourally-cloned actor collapses: the critic
starts random, so its policy-gradient is noise, and the actor drifts away
from the good clone (observed: 3.90 -> 4.8 L/100km, SoC -> 95%).

The fix (Fujimoto & Gu, "A Minimalist Approach to Offline RL", 2021) adds a
behavioural-cloning term to the actor loss:

    actor_loss = -lambda * Q(s, pi(s))  +  ( pi(s) - a_dataset )^2
    lambda     = alpha / mean|Q(s, pi(s))|          # normalises Q scale

The BC term anchors the policy to the actions already in the replay buffer
(dominated by the benchmark rollout we seeded), so the actor can only depart
from the clone where the critic shows a clear, consistent Q improvement.
A small online exploration noise feeds fresh transitions in so the critic
learns real action-gradients around the clone.

    python -m src.agents.finetune_bcreg --cycle NEDC --in models --out models \
        --steps 120000 --alpha 2.5 --explore-std 0.1

The deterministic policy is evaluated periodically on the TRUE metric
(v_ce_equiv + charge-sustaining penalty); the best checkpoint is kept and
the cloned seed is the floor, so fine-tuning can never lose ground.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

from src.env.ems_env import EMSEnv, SOC_TARGET, TERM_TOL
from src.agents.train_sac import rollout_deterministic, score


def det_action(actor, obs_t):
    """Deterministic (tanh-mean) action for a batch of observations."""
    mean, _log_std, _ = actor.get_action_dist_params(obs_t)
    return torch.tanh(mean)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycle", default="NEDC", choices=["NEDC", "FTP75"])
    p.add_argument("--in", dest="in_dir", default="models")
    p.add_argument("--out", default="models")
    p.add_argument("--steps", type=int, default=120_000)
    p.add_argument("--alpha", type=float, default=2.5,
                   help="Q weight in the actor loss; larger = trust the "
                        "critic more / BC anchor less")
    p.add_argument("--explore-std", type=float, default=0.1,
                   help="Gaussian exploration noise std (action space)")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--policy-delay", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=6000)
    p.add_argument("--lambda-soc", type=float, default=8.0)
    p.add_argument("--soc-deadband", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    env = EMSEnv(args.cycle, lambda_soc=args.lambda_soc,
                 soc_deadband=args.soc_deadband)

    model = SAC.load(in_dir / "sac_ems_last", env=env,
                     custom_objects=dict(learning_starts=0))
    model.load_replay_buffer(in_dir / "replay_buffer.pkl")
    for g in model.actor.optimizer.param_groups:
        g["lr"] = args.actor_lr
    for g in model.critic.optimizer.param_groups:
        g["lr"] = args.critic_lr

    actor, critic, critic_t = model.actor, model.critic, model.critic_target
    dev, gamma, tau = model.device, model.gamma, model.tau
    rb = model.replay_buffer

    # ---- floor: the cloned seed is the best until beaten ----------------
    best = score(rollout_deterministic(model, args.cycle))
    bc_final = rollout_deterministic(model, args.cycle)
    model.save(out_dir / "sac_ems_best")
    (out_dir / "best_score.txt").write_text(str(best))
    print(f"[BC floor] V_liter={bc_final['v_liter']:.3f}  "
          f"SoC={bc_final['soc_final']*100:.1f}%  score={best:.3f}")

    obs, _ = env.reset()
    for t in range(1, args.steps + 1):
        # --- act with exploration noise around the deterministic policy --
        with torch.no_grad():
            o_t = torch.as_tensor(obs, device=dev).float().unsqueeze(0)
            a = det_action(actor, o_t).cpu().numpy()[0]
        a = np.clip(a + rng.normal(0, args.explore_std, size=a.shape), -1, 1)
        a = a.astype(np.float32)
        nobs, r, term, trunc, info = env.step(a)
        rb.add(obs, nobs, a, np.array([r], np.float32),
               np.array([term]), [info])
        obs = nobs if not term else env.reset()[0]

        # --- critic update ----------------------------------------------
        data = rb.sample(args.batch)
        with torch.no_grad():
            na = det_action(actor, data.next_observations)
            na = (na + torch.randn_like(na).mul_(0.1).clamp_(-0.2, 0.2)).clamp_(-1, 1)
            q1t, q2t = critic_t(data.next_observations, na)
            qt = torch.min(q1t, q2t)
            target = data.rewards + (1 - data.dones) * gamma * qt
        q1, q2 = critic(data.observations, data.actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        critic.optimizer.zero_grad()
        critic_loss.backward()
        critic.optimizer.step()

        # --- delayed BC-anchored actor update ---------------------------
        if t % args.policy_delay == 0:
            pi = det_action(actor, data.observations)
            q_pi = critic(data.observations, pi)[0]
            lam = args.alpha / q_pi.abs().mean().detach()
            bc = F.mse_loss(pi, data.actions)
            actor_loss = -lam * q_pi.mean() + bc
            actor.optimizer.zero_grad()
            actor_loss.backward()
            actor.optimizer.step()
            polyak_update(critic.parameters(), critic_t.parameters(), tau)

        # --- periodic evaluation on the true metric ---------------------
        if t % args.eval_every == 0:
            fin = rollout_deterministic(model, args.cycle)
            s = score(fin)
            tag = ""
            if s < best:
                best = s
                model.save(out_dir / "sac_ems_best")
                (out_dir / "best_score.txt").write_text(str(best))
                tag = "  <-- new best"
            print(f"[ft @ {t:>7}] V_liter={fin['v_liter']:.3f}  "
                  f"V_CE_equiv={fin['v_ce_equiv']:.3f}  "
                  f"SoC={fin['soc_final']*100:5.1f}%  bc={bc.item():.3f}  "
                  f"score={s:.3f}{tag}")

    model.save(out_dir / "sac_ems_last")
    model.save_replay_buffer(out_dir / "replay_buffer.pkl")
    print(f"\nbest score: {best:.3f}  (saved {out_dir/'sac_ems_best.zip'})")


if __name__ == "__main__":
    main()
