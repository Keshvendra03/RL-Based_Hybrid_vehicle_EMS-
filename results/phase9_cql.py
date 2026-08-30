"""
phase9_cql.py
=============
PHASE 9 EXPERIMENT A -- CQL-style conservative critic.

The ONLY change vs the Phase-8 CONTROL is the CRITIC LOSS. Everything else is
frozen: reward, state, actor class (unimodal squashed Gaussian), gamma 0.20,
n_step 1, k_fb 2.5, env, action map, ECMS, rule-based, evaluator, 150k budget,
batch 512, buffer 300k, actor lr 3e-4, seeds {0,1,2}, net_arch [256,256].

----------------------------------------------------------------------------
MODIFIED CRITIC LOSS  (CQL(H), continuous, Kumar et al. 2020 eq. 4 + appendix)
----------------------------------------------------------------------------
Standard SB3 SAC critic loss (unchanged):
    L_TD(theta_i) = 0.5 * MSE( Q_i(s, a_data),  y )
    y = r + (1-done) * gamma * [ min_j Q_target_j(s', a') - alpha * log pi(a'|s') ]

Added conservative term, per critic i:
    L_CQL(theta_i) = cql_alpha * ( logsumexp_{a in A_s} Q_i(s, a)  -  Q_i(s, a_data) )

with the importance-corrected action set A_s (CQL(H)):
    * n_act actions  a_rand ~ Uniform[-1, 1]        , weight  -log(0.5^{dim_A})
    * n_act actions  a_pi   ~ pi(.|s)               , weight  -log pi(a_pi|s).detach()
    * n_act actions  a_pi'  ~ pi(.|s')              , weight  -log pi(a_pi'|s').detach()
    logsumexp is taken over the concatenation of (Q_i(s,a) + weight).

Total critic loss:  L_TD + L_CQL   (actor loss and everything else untouched.)

Intuition: L_CQL pushes Q DOWN on actions the current policy / a uniform prior
would pick but that are NOT in the data, and pushes Q UP on the logged action.
This lowers the value of out-of-distribution actions so that a Q-greedy policy
cannot exploit unsupported Q spikes (the Phase-8 Q-oracle failure mode).

    python -m results.phase9_cql --cycle NEDC --seed 0 --cql-alpha 1.0 --out models_p9a_N0
"""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.callbacks import CallbackList

from src.env.ems_env import EMSEnv
from src.agents.train_sac import EvalAndCheckpoint, SACDiagnostics, rollout_deterministic, score

EQF = {"NEDC": 0.2717, "FTP75": 0.4981}
RB = {"NEDC": 3.5056, "FTP75": 3.2323}
ECMSV = {"NEDC": 3.1887, "FTP75": 2.8097}


class CQLSAC(SAC):
    """SAC with a CQL(H) conservative term added to the critic loss only."""

    def __init__(self, *a, cql_alpha: float = 1.0, cql_n_actions: int = 10, **kw):
        self.cql_alpha = float(cql_alpha)
        self.cql_n_actions = int(cql_n_actions)
        super().__init__(*a, **kw)

    def _cql_q(self, obs, actions):
        """Q for [B, N, A] action tensor -> list of [B, N] per critic."""
        B, N, A = actions.shape
        obs_rep = obs.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        act_rep = actions.reshape(B * N, A)
        qs = self.critic(obs_rep, act_rep)                 # tuple of [B*N, 1]
        return [q.view(B, N) for q in qs]

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, cql_losses = [], [], []
        A = int(self.action_space.shape[0])
        unif_logprob = -np.log(0.5) * A       # -log( (1/2)^A ) for U[-1,1]

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

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
                self.ent_coef_optimizer.zero_grad(); ent_coef_loss.backward(); self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q, _ = th.min(next_q, dim=1, keepdim=True)
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q

            current_q = self.critic(replay_data.observations, replay_data.actions)
            td_loss = 0.5 * sum(F.mse_loss(cq, target_q) for cq in current_q)

            # ---------------- CQL(H) conservative term --------------------
            B = replay_data.observations.shape[0]
            n = self.cql_n_actions
            rand_a = (th.rand(B, n, A, device=self.device) * 2.0 - 1.0)
            with th.no_grad():
                pa_s, lp_s = self.actor.action_log_prob(
                    replay_data.observations.unsqueeze(1).expand(-1, n, -1).reshape(B * n, -1))
                pa_sp, lp_sp = self.actor.action_log_prob(
                    replay_data.next_observations.unsqueeze(1).expand(-1, n, -1).reshape(B * n, -1))
            pa_s = pa_s.view(B, n, A); lp_s = lp_s.view(B, n)
            pa_sp = pa_sp.view(B, n, A); lp_sp = lp_sp.view(B, n)

            q_rand = self._cql_q(replay_data.observations, rand_a)          # list [B,n]
            q_pi_s = self._cql_q(replay_data.observations, pa_s)
            q_pi_sp = self._cql_q(replay_data.observations, pa_sp)

            cql_term = 0.0
            for i, cq_data in enumerate(current_q):                        # per critic
                cat = th.cat([q_rand[i] - unif_logprob,
                              q_pi_s[i] - lp_s.detach(),
                              q_pi_sp[i] - lp_sp.detach()], dim=1)          # [B, 3n]
                logsumexp = th.logsumexp(cat, dim=1, keepdim=True)         # [B,1]
                cql_term = cql_term + (logsumexp - cq_data).mean()
            cql_term = self.cql_alpha * cql_term
            cql_losses.append(float(cql_term.detach()))

            critic_loss = td_loss + cql_term
            critic_losses.append(float(td_loss.detach()))
            self.critic.optimizer.zero_grad(); critic_loss.backward(); self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())
            self.actor.optimizer.zero_grad(); actor_loss.backward(); self.actor.optimizer.step()

            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            self._n_updates += 1

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/cql_term", np.mean(cql_losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, choices=["NEDC", "FTP75"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--cql-alpha", type=float, default=1.0)
    ap.add_argument("--cql-n", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out_dir = Path(a.out) / a.cycle
    out_dir.mkdir(parents=True, exist_ok=True)
    env = EMSEnv(a.cycle, eq_factor=EQF[a.cycle], lambda_soc=2.0, soc_deadband=0.10,
                 lookahead=5, k_fb=2.5, action_map="modeaware_gated")
    ep_len = env.cycle.length - 1

    model = CQLSAC(
        "MlpPolicy", env, cql_alpha=a.cql_alpha, cql_n_actions=a.cql_n,
        learning_rate=3e-4, buffer_size=300_000, learning_starts=2 * ep_len,
        batch_size=512, tau=0.005, gamma=0.20, train_freq=64, gradient_steps=16,
        ent_coef="auto", target_entropy="auto",
        policy_kwargs=dict(net_arch=[256, 256]), seed=a.seed, verbose=0,
    )
    cb = EvalAndCheckpoint([a.cycle], every_steps=2 * ep_len, out_dir=out_dir,
                           eq_factor=EQF[a.cycle], soc_deadband=0.10, lookahead=5,
                           k_fb=2.5, action_map="modeaware_gated", verbose=1)
    diag = SACDiagnostics(out_dir, batch_size=512, log_every=5000)
    (out_dir / "run_config.json").write_text(json.dumps({
        "cycle": a.cycle, "seed": a.seed, "timesteps": a.timesteps,
        "phase": "9A", "critic": "CQL(H)", "cql_alpha": a.cql_alpha, "cql_n": a.cql_n,
        "gamma": 0.20, "n_step": 1, "action_map": "modeaware_gated", "k_fb": 2.5,
        "eq_factor": EQF[a.cycle], "target_entropy": "auto", "lr": 3e-4, "batch": 512,
        "buffer": 300000, "gradient_steps": 16, "lookahead": 5,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    }, indent=2))

    print(f"[9A CQL] {a.cycle} seed{a.seed} cql_alpha={a.cql_alpha} n={a.cql_n}  "
          f"{a.timesteps} steps  (RB {RB[a.cycle]} ECMS {ECMSV[a.cycle]})")
    model.learn(total_timesteps=a.timesteps, callback=CallbackList([cb, diag]), progress_bar=False)
    model.save(out_dir / "sac_ems_last")
    model.save_replay_buffer(out_dir / "replay_buffer.pkl")
    fin = rollout_deterministic(model, a.cycle, EQF[a.cycle], 0.10, 5, 2.5, "modeaware_gated")
    print(f"[9A final] {a.cycle} s{a.seed} V_CE={fin['v_ce_equiv']:.4f} SoC={fin['soc_final']*100:.2f}% "
          f"score={score(fin):.4f} best={cb.best:.4f}")
    json.dump({"final_last": fin, "best_score": cb.best, "cql_alpha": a.cql_alpha},
              open(out_dir / "phase9a_final.json", "w"), indent=2)


if __name__ == "__main__":
    main()
