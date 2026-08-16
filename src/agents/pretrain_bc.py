"""
pretrain_bc.py
==============
Warm-start a SAC agent from the advanced rule-based benchmark.

Beating a well-tuned rule-based EMS from a blank policy is a hard
exploration problem: vanilla SAC descends slowly and plateaus well above
the benchmark. Instead we (1) clone the benchmark's torque-split decision
into the SAC actor by supervised regression, and (2) pre-fill the SAC
replay buffer with real benchmark transitions (true env rewards) so the
critic immediately has the benchmark's value function to bootstrap from.
SAC then *fine-tunes* this policy, only ever moving away from the
benchmark where doing so lowers (equivalent) fuel.

    python -m src.agents.pretrain_bc --cycle NEDC --out models --epochs 40

Produces <out>/sac_ems_last(.zip) + <out>/replay_buffer.pkl, ready for:

    python -m src.agents.train_sac --cycle NEDC --resume --timesteps 120000

The cloning target is exact: the env's action->u map is u = 0.8*a + 0.2,
so a* = (u_benchmark - 0.2) / 0.8. The env reproduces the benchmark's fuel
figure to 1e-9 when fed the benchmark u (env validation test [2]), so a
perfectly cloned actor starts at ~3.50 L/100km.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from src.env.ems_env import EMSEnv, U_MIN, U_MAX
from src.baselines.advanced_rule_based import (
    AdvancedController, control_unit_advanced,
)

U_SCALE = 0.5 * (U_MAX - U_MIN)   # 0.8
U_OFF = U_MIN + U_SCALE            # 0.2  ->  u = U_SCALE * a + U_OFF


def u_to_action(u: float) -> float:
    return float(np.clip((u - U_OFF) / U_SCALE, -1.0, 1.0))


class _MirrorEnv(EMSEnv):
    """EMSEnv whose action layer is the advanced controller; also records
    the (observation, benchmark-u) pair seen at every step."""

    def __init__(self, cycle_name="NEDC"):
        super().__init__(cycle_name=cycle_name, fast_interp=True)
        self.ctrl = AdvancedController(cycle_name=cycle_name)

    def reset(self, **kw):
        self.ctrl.reset()
        return super().reset(**kw)

    def _action_to_torques(self, action):
        d = self._demand
        c = self.ctrl.step(d["w_MGB"], d["dw_MGB"], d["T_MGB"],
                           d["gear"], self._Q_BT, d["v"])
        cu = control_unit_advanced(d["w_MGB"], d["dw_MGB"], d["T_MGB"],
                                   c["u"], c["state_CE"])
        self._last_u = c["u"]
        return cu["T_CE"], cu["T_EM"], c["u"], "mirror"


def collect_benchmark(cycle: str):
    """Roll the benchmark through the env; return obs, target_action, and
    full SARS transitions (obs, action, reward, next_obs, done)."""
    env = _MirrorEnv(cycle)
    obs, _ = env.reset()
    obs_list, act_list = [], []
    transitions = []
    while True:
        nxt, r, term, trunc, info = env.step(np.zeros(1, dtype=np.float32))
        a = u_to_action(info["u"])
        obs_list.append(obs.copy())
        act_list.append([a])
        transitions.append((obs.copy(), np.array([a], dtype=np.float32),
                            float(r), nxt.copy(), bool(term)))
        obs = nxt
        if term:
            break
    return (np.asarray(obs_list, np.float32),
            np.asarray(act_list, np.float32),
            transitions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycle", default="NEDC", choices=["NEDC", "FTP75"])
    p.add_argument("--out", default="models")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--buffer-copies", type=int, default=8,
                   help="how many noisy copies of the benchmark rollout to "
                        "seed the replay buffer with (exploration around it)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs, target_a, transitions = collect_benchmark(args.cycle)
    print(f"collected {len(obs)} benchmark steps")

    env = EMSEnv(args.cycle)
    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4, buffer_size=400_000, batch_size=512,
        tau=0.005, gamma=0.999, ent_coef=0.01,   # fixed; fine-tuner matches
        policy_kwargs=dict(net_arch=[256, 256]), seed=args.seed, verbose=0,
    )

    # ---- 1. behavioural cloning of the actor ----------------------------
    dev = model.device
    obs_t = torch.as_tensor(obs, device=dev)
    tgt_t = torch.as_tensor(target_a, device=dev)
    opt = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    n = obs_t.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            mean, log_std, _ = model.actor.get_action_dist_params(obs_t[idx])
            pred = torch.tanh(mean)
            loss = ((pred - tgt_t[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  bc epoch {ep+1:>3}: mse={tot/n:.5f}")

    # ---- 2. seed the replay buffer with real benchmark transitions ------
    rng = np.random.default_rng(args.seed)
    for c in range(args.buffer_copies):
        for (o, a, r, no, d) in transitions:
            aa = a if c == 0 else np.clip(
                a + rng.normal(0, 0.1, size=a.shape).astype(np.float32),
                -1.0, 1.0)
            model.replay_buffer.add(o, no, aa, np.array([r], np.float32),
                                    np.array([d]), [{}])
    print(f"replay buffer seeded: {model.replay_buffer.size()} transitions")

    model.save(out_dir / "sac_ems_last")
    model.save_replay_buffer(out_dir / "replay_buffer.pkl")

    # ---- 3. report the cloned policy's deterministic fuel ---------------
    from src.agents.evaluate_rl import evaluate
    evaluate(str(out_dir / "sac_ems_last"), args.cycle)


if __name__ == "__main__":
    main()
