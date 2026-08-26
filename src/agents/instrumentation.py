"""
instrumentation.py
==================
Phase-2 section 3 + section 22.

section 3 -- SAC/TD3 internals that SB3 does NOT log by default. On every
`log_every` gradient batch this records, to TensorBoard and to
`<out_dir>/sac_diagnostics.csv`:

    Q1, Q2, min(Q1,Q2), Q-target, TD error (mean and RMS),
    actor mean (pre-tanh mu), actor std, log pi(a|s),
    action before tanh, action after tanh, final env action,
    action percentiles p1/p25/p50/p75/p99, action saturation %

Why this exists: the Phase-1 audit could not inspect raw Q values or TD error
because SB3 logs only `train/critic_loss`. Without them the SNR argument
(critic fit noise vs action-value spread) could not be made from training logs
alone -- it needed an offline Q-sweep. This closes that gap.

section 22 -- CheckpointRule: a deterministic, auditable checkpoint policy.
NEVER selects on training reward. Selection is, in strict order:
    1. zero constraint violations           (hard gate)
    2. charge-sustaining |SoC-0.5| <= tol   (hard gate)
    3. minimum V_CE_equiv                   (primary metric)
A run that never satisfies the gates falls back to best V_CE_equiv but is
flagged `gates_met=False` so it can never be silently reported as valid.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

_FIELDS = ["timesteps", "q1_mean", "q2_mean", "minq_mean", "qtarget_mean",
           "td_error_mean", "td_error_rms", "actor_mu_mean", "actor_std_mean",
           "log_pi_mean", "pre_tanh_mean", "post_tanh_mean",
           "act_p1", "act_p25", "act_p50", "act_p75", "act_p99", "act_sat_pct"]


class SACDiagnostics(BaseCallback):
    """Logs SAC internals that SB3 omits (section 3)."""

    def __init__(self, out_dir: Path, batch_size: int = 512,
                 log_every: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.out_dir = Path(out_dir)
        self.batch_size = batch_size
        self.log_every = log_every
        self.csv_path = self.out_dir / "sac_diagnostics.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=_FIELDS).writeheader()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_every != 0:
            return True
        rb = self.model.replay_buffer
        if rb is None or rb.size() < self.batch_size:
            return True
        try:
            d = rb.sample(self.batch_size, env=self.model._vec_normalize_env)
            with th.no_grad():
                obs, act = d.observations, d.actions
                qs = self.model.critic(obs, act)
                q1 = qs[0].cpu().numpy().ravel()
                q2 = qs[1].cpu().numpy().ravel() if len(qs) > 1 else q1
                minq = np.minimum(q1, q2)

                actor = self.model.actor
                if hasattr(actor, "action_log_prob"):          # SAC
                    a_pi, logp = actor.action_log_prob(obs)
                    mu, log_std, _ = actor.get_action_dist_params(obs)
                    std = th.exp(log_std)
                    logp_np = logp.cpu().numpy().ravel()
                    mu_np = mu.cpu().numpy().ravel()
                    std_np = std.cpu().numpy().ravel()
                    pre = mu_np                                # pre-tanh mean
                    post = np.tanh(mu_np)                      # post-tanh
                    a_np = a_pi.cpu().numpy().ravel()
                else:                                          # TD3 (deterministic)
                    a_pi = actor(obs)
                    a_np = a_pi.cpu().numpy().ravel()
                    logp_np = np.full_like(a_np, np.nan)
                    mu_np = a_np; std_np = np.zeros_like(a_np)
                    pre = np.arctanh(np.clip(a_np, -0.999999, 0.999999)); post = a_np

                # Q-target (mirrors the algorithm's own Bellman target)
                nxt = d.next_observations
                if hasattr(actor, "action_log_prob"):
                    na, nlogp = actor.action_log_prob(nxt)
                    ent = (th.exp(self.model.log_ent_coef.detach())
                           if getattr(self.model, "log_ent_coef", None) is not None
                           else self.model.ent_coef_tensor)
                    nq = th.cat(self.model.critic_target(nxt, na), dim=1)
                    nq, _ = th.min(nq, dim=1, keepdim=True)
                    nq = nq - ent * nlogp.reshape(-1, 1)
                else:
                    na = self.model.actor_target(nxt)
                    nq = th.cat(self.model.critic_target(nxt, na), dim=1)
                    nq, _ = th.min(nq, dim=1, keepdim=True)
                disc = d.discounts if getattr(d, "discounts", None) is not None \
                    else self.model.gamma
                qt = (d.rewards + (1 - d.dones) * disc * nq).cpu().numpy().ravel()

            td = qt - minq
            row = dict(
                timesteps=self.num_timesteps,
                q1_mean=float(q1.mean()), q2_mean=float(q2.mean()),
                minq_mean=float(minq.mean()), qtarget_mean=float(qt.mean()),
                td_error_mean=float(td.mean()),
                td_error_rms=float(np.sqrt((td ** 2).mean())),
                actor_mu_mean=float(mu_np.mean()), actor_std_mean=float(std_np.mean()),
                log_pi_mean=float(np.nanmean(logp_np)),
                pre_tanh_mean=float(pre.mean()), post_tanh_mean=float(post.mean()),
                act_p1=float(np.percentile(a_np, 1)), act_p25=float(np.percentile(a_np, 25)),
                act_p50=float(np.percentile(a_np, 50)), act_p75=float(np.percentile(a_np, 75)),
                act_p99=float(np.percentile(a_np, 99)),
                act_sat_pct=float(100.0 * (np.abs(a_np) > 0.99).mean()),
            )
            with open(self.csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=_FIELDS).writerow(row)
            for k, v in row.items():
                if k != "timesteps":
                    self.logger.record(f"diag/{k}", v)
        except Exception as e:                       # never kill a run for logging
            if self.verbose:
                print(f"[diag] skipped @ {self.num_timesteps}: {e}")
        return True


class CheckpointRule:
    """Deterministic checkpoint selection (section 22).

    Ordering: (gates_met desc, v_ce_equiv asc). Training reward is NEVER used.
    """

    def __init__(self, soc_tol: float = 0.02):
        self.soc_tol = soc_tol
        self.best = None      # dict

    @staticmethod
    def _gates(v_ce_equiv, soc_final, violations, soc_tol):
        return (violations == 0) and (abs(soc_final - 0.5) <= soc_tol)

    def offer(self, *, step, cycle, seed, v_ce_equiv, soc_final,
              violations=0) -> bool:
        """Return True if this candidate becomes the new best."""
        g = self._gates(v_ce_equiv, soc_final, violations, self.soc_tol)
        cand = dict(step=int(step), cycle=cycle, seed=seed,
                    v_ce_equiv=float(v_ce_equiv), soc_final=float(soc_final),
                    violations=int(violations), gates_met=bool(g))
        if self.best is None:
            self.best = cand; return True
        b = self.best
        better = ((g, -v_ce_equiv) > (b["gates_met"], -b["v_ce_equiv"]))
        if better:
            self.best = cand; return True
        return False

    def save(self, out_dir: Path) -> None:
        if self.best is not None:
            (Path(out_dir) / "best_checkpoint_rule.json").write_text(
                json.dumps(self.best, indent=2))
