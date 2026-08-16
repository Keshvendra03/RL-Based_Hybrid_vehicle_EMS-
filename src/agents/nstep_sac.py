"""
nstep_sac.py
============
FIX 4 — n-step returns for SAC, to sharpen long-horizon credit assignment.

WHY (confirmed by the mode breakdown, not hypothesized)
-------------------------------------------------------
On NEDC the trained agent uses engine-OFF on 28.4% of moving steps where the
ECMS optimum uses 53.1%, and ASSIST on 24.1% where ECMS uses 0.2%. The agent
sits in "engine on + small motor assist" exactly where the optimum commits to
"engine off, pure electric". Those two actions are ADJACENT in action space and
differ by a TINY per-step fuel amount, so under a single-step-TD critic at
gamma=0.9999 over a 1220-step episode their Q-values are nearly tied and the
policy has no sharp gradient to prefer OFF. Integrated over ~24% of the cycle
that tiny per-step difference IS the ~0.5 L/100km gap.

n-step returns inject n steps of REAL reward into the critic target instead of
relying on one bootstrap, which directly sharpens the critic's ability to rank
these small-but-persistent differences. This is a pure ESTIMATION-quality change:

  * It does NOT touch the env, plant, reward, action interface, or observation.
    (The action-snap fix was rejected precisely because it distorted the optimum
     from 3.189 to 3.538; n-step cannot distort the plant — it only changes how
     the existing reward is propagated into the critic target.)
  * It injects NO ECMS / rule-based knowledge -> reward independence preserved.
  * It is cycle-agnostic -> generalization preserved.

n-step target used (standard):
    G_t^(n) = sum_{k=0}^{n-1} gamma^k r_{t+k}     (truncated at episode end)
    y_t     = G_t^(n) + gamma^n * (1-done) * [ min_i Q_i(s_{t+n}, a') - alpha*logpi ]
The only change SAC needs is to discount the bootstrap by gamma**n_eff (where
n_eff is the actual, possibly-truncated step count) instead of gamma**1.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


class NStepSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor      # the n-step-ahead observation
    dones: th.Tensor                  # done within the n-step window
    rewards: th.Tensor                # discounted n-step reward sum G^(n)
    discounts: th.Tensor              # gamma ** n_eff  (per-sample bootstrap discount)


class NStepReplayBuffer(ReplayBuffer):
    """ReplayBuffer that returns n-step transitions.

    Stores 1-step transitions exactly like the base buffer (so prefill and
    model.replay_buffer.add() are unchanged), but at SAMPLE time accumulates
    the discounted n-step reward and the n-step-ahead next obs, truncating at
    episode boundaries.
    """

    def __init__(self, *args, n_step: int = 5, gamma: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_step = int(n_step)
        self.gamma = float(gamma)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> NStepSamples:  # type: ignore[override]
        # Avoid sampling the in-progress transition at self.pos (same rule as base).
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_nstep_samples(batch_inds, env)

    def _get_nstep_samples(self, batch_inds: np.ndarray, env=None) -> NStepSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        bs = len(batch_inds)

        obs = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        actions = self.actions[batch_inds, env_indices, :]

        g_return = np.zeros(bs, dtype=np.float32)        # discounted n-step reward sum
        disc = np.ones(bs, dtype=np.float32)             # gamma ** n_eff
        done_in_window = np.zeros(bs, dtype=np.float32)
        # next obs pointer; starts one ahead, walks forward up to n steps
        nstep_idx = batch_inds.copy()

        active = np.ones(bs, dtype=bool)                 # still accumulating (no done yet)
        cur = batch_inds.copy()
        gpow = np.ones(bs, dtype=np.float32)             # gamma**k accumulator

        for k in range(self.n_step):
            r = self.rewards[cur, env_indices]
            # add reward only for still-active (not yet terminated) samples
            g_return += np.where(active, gpow * r, 0.0)
            # real (non-timeout) done at this step
            d = self.dones[cur, env_indices] * (1.0 - self.timeouts[cur, env_indices])
            # the next-state index after this step
            nxt = (cur + 1) % self.buffer_size
            # for active samples, advance the bootstrap pointer & discount
            newly_done = active & (d > 0.5)
            # samples that terminate here: bootstrap is masked, pointer frozen at nxt
            done_in_window = np.where(newly_done, 1.0, done_in_window)
            # update pointer to nxt for samples that are still active
            nstep_idx = np.where(active, nxt, nstep_idx)
            # discount for bootstrap grows by one gamma for each active step taken
            disc = np.where(active & (~newly_done), disc * self.gamma, disc)
            # advance
            gpow = gpow * self.gamma
            cur = nxt
            # deactivate samples that just terminated, and stop if we wrapped to self.pos
            active = active & (~newly_done)
            # also stop accumulating if we reached the write head (data boundary)
            at_head = (cur == self.pos)
            active = active & (~at_head)

        next_obs = self._normalize_obs(self.next_observations[
            (nstep_idx) % self.buffer_size, env_indices, :], env) \
            if not self.optimize_memory_usage else \
            self._normalize_obs(self.observations[(nstep_idx + 1) % self.buffer_size, env_indices, :], env)

        data = (
            obs,
            actions,
            next_obs,
            done_in_window.reshape(-1, 1),
            self._normalize_reward(g_return.reshape(-1, 1), env),
            disc.reshape(-1, 1),
        )
        return NStepSamples(*tuple(map(self.to_torch, data)))


class NStepSAC(SAC):
    """SAC whose critic target discounts the bootstrap by gamma**n_eff.

    Only train() differs from stock SAC: it reads the per-sample n-step discount
    from the buffer and uses it in the TD target. Everything else (actor loss,
    entropy tuning, target smoothing) is identical to SB3 SAC.
    """

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            replay = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore

            if self.use_sde:
                self.actor.reset_noise()

            # --- entropy coefficient (unchanged from SB3) ---
            actions_pi, log_prob = self.actor.action_log_prob(replay.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # --- critic target with n-step discount ---
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay.next_observations)
                next_q = th.cat(self.critic_target(replay.next_observations, next_actions), dim=1)
                next_q, _ = th.min(next_q, dim=1, keepdim=True)
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                # KEY CHANGE: discount bootstrap by gamma**n_eff (replay.discounts),
                # and reward term is already the discounted n-step sum G^(n).
                target_q = replay.rewards + (1 - replay.dones) * replay.discounts * next_q

            current_q = self.critic(replay.observations, replay.actions)
            critic_loss = 0.5 * sum(th.nn.functional.mse_loss(cq, target_q) for cq in current_q)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # --- actor loss (unchanged) ---
            q_pi = th.cat(self.critic(replay.observations, actions_pi), dim=1)
            min_q_pi, _ = th.min(q_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_q_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if self._n_updates % self.target_update_interval == 0:
                from stable_baselines3.common.utils import polyak_update
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
