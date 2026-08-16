"""
per.py
======
Prioritized Experience Replay (PER) for SAC.

Standard SAC samples the replay buffer uniformly. PER (Schaul et al., 2016)
samples transitions in proportion to their temporal-difference (TD) error,
so the agent revisits the transitions it predicts worst. On HEV-EMS this
improves sample efficiency markedly — valuable when training on CPU.

Two pieces:
  * PrioritizedReplayBuffer - a drop-in SB3 ReplayBuffer that keeps a
    per-transition priority, samples proportionally to priority**alpha, and
    returns importance-sampling (IS) weights to correct the induced bias.
  * SACPER - SAC with train() overridden to (a) weight the critic loss by the
    IS weights and (b) write fresh |TD error| back as the new priorities.

Usage (see train_sac.py --per):
    model = SACPER("MlpPolicy", env,
                   replay_buffer_class=PrioritizedReplayBuffer, ...)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional PER buffer compatible with SB3's off-policy algorithms.

    sample() selects indices proportional to priority**alpha and stashes the
    chosen indices and IS weights on the instance (`_last_indices`,
    `_last_weights`) so the algorithm can read them after sampling without
    changing SB3's sample() return signature. beta is annealed linearly from
    `beta0` to 1.0 over `beta_steps` samples.
    """

    def __init__(self, *args, alpha: float = 0.6, beta0: float = 0.4,
                 beta_steps: int = 200_000, eps: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.beta0 = float(beta0)
        self.beta_steps = int(beta_steps)
        self.eps = float(eps)
        self._priorities = np.zeros(self.buffer_size, dtype=np.float64)
        self._max_priority = 1.0
        self._sample_count = 0
        self._last_indices: Optional[np.ndarray] = None
        self._last_weights: Optional[th.Tensor] = None

    def add(self, *args, **kwargs) -> None:
        # the slot about to be written is the current self.pos
        idx = self.pos
        super().add(*args, **kwargs)
        # new transitions enter at max priority so they are seen at least once
        self._priorities[idx] = self._max_priority

    def _beta(self) -> float:
        frac = min(1.0, self._sample_count / max(1, self.beta_steps))
        return self.beta0 + frac * (1.0 - self.beta0)

    def sample(self, batch_size: int, env=None):
        upper = self.buffer_size if self.full else self.pos
        prio = self._priorities[:upper] ** self.alpha
        total = prio.sum()
        if total <= 0:
            probs = np.full(upper, 1.0 / upper)
        else:
            probs = prio / total
        indices = np.random.choice(upper, size=batch_size, p=probs)

        beta = self._beta()
        weights = (upper * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        self._sample_count += 1
        self._last_indices = indices
        self._last_weights = th.as_tensor(
            weights, dtype=th.float32, device=self.device).reshape(-1, 1)

        return self._get_samples(indices, env=env)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        p = np.abs(td_errors).reshape(-1) + self.eps
        self._priorities[indices] = p
        self._max_priority = max(self._max_priority, float(p.max()))


class SACPER(SAC):
    """SAC whose critic update uses PER importance-sampling weights and writes
    |TD error| back as priorities. Falls back to plain SAC behaviour if the
    buffer is not a PrioritizedReplayBuffer."""

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        per = isinstance(self.replay_buffer, PrioritizedReplayBuffer)
        ent_coef_losses, ent_coefs, actor_losses, critic_losses = [], [], [], []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = getattr(replay_data, "discounts", None)
            discounts = discounts if discounts is not None else self.gamma
            if per:
                idx = self.replay_buffer._last_indices
                weights = self.replay_buffer._last_weights
            else:
                weights = 1.0

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
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

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q, _ = th.min(next_q, dim=1, keepdim=True)
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q

            current_q = self.critic(replay_data.observations, replay_data.actions)

            # IS-weighted critic loss (weights == 1.0 reduces to standard SAC)
            critic_loss = 0.5 * sum(
                (weights * (cq - target_q) ** 2).mean() for cq in current_q)
            critic_losses.append(float(critic_loss.item()))

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # update priorities from the mean |TD error| across critics
            if per:
                with th.no_grad():
                    td = th.stack([(cq - target_q).abs() for cq in current_q], dim=0).mean(0)
                    self.replay_buffer.update_priorities(
                        idx, td.squeeze(-1).cpu().numpy())

            q_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_q_pi, _ = th.min(q_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_q_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            self._n_updates += 1

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
