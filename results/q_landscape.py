"""
q_landscape.py
==============
Q-function forensics (Phase-2 §19). Sweeps the FULL action range at
representative operating points and reports Q1/Q2/min-Q alongside the TRUE
one-step reward for the same action, to answer one question:

    Does the critic actually resolve the OFF/ASSIST distinction, or does it
    smooth the fuel-cutoff discontinuity into a ramp the actor can't exploit?

    python -m results.q_landscape --checkpoint models_x/NEDC/sac_ems_best \
        --cycle NEDC --eq-factor 1.3125 --k-fb 8.0 --action-map linear

The true reward for each candidate action is obtained by deep-copying the
environment at the probe state and stepping it once, so every action is
evaluated from EXACTLY the same physical state (same SoC, same demand).
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch as th

from src.env.ems_env import EMSEnv
from src.env.powertrain import _T_CUTOFF


def probe_states(env: EMSEnv, bands: dict):
    """Roll the cycle; snapshot the env the first time each torque band is hit."""
    found = {}
    obs, _ = env.reset()
    while True:
        d = env._demand
        T, w = d["T_MGB"], d["w_MGB"]
        for name, (lo, hi) in bands.items():
            if name not in found and lo <= T < hi and w > 50.0:
                found[name] = (obs.copy(), copy.deepcopy(env), T, w,
                               env._Q_BT / 36000.0)
        obs, _, term, _, _ = env.step(np.array([0.0], np.float32))
        if term or len(found) == len(bands):
            break
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cycle", default="NEDC")
    p.add_argument("--eq-factor", type=float, default=1.3125)
    p.add_argument("--k-fb", type=float, default=8.0)
    p.add_argument("--action-map", default="linear")
    p.add_argument("--n-a", type=int, default=41)
    a = p.parse_args()

    from stable_baselines3 import SAC
    model = SAC.load(a.checkpoint)
    od = int(model.observation_space.shape[0])
    la = 0 if od <= 16 else od - 15

    env = EMSEnv(a.cycle, eq_factor=a.eq_factor, k_fb=a.k_fb,
                 action_map=a.action_map, lookahead=la)
    bands = {"low_torque": (5.0, 15.0),
             "med_torque": (20.0, 30.0),
             "high_torque": (45.0, 80.0)}
    states = probe_states(env, bands)
    grid = np.linspace(-1.0, 1.0, a.n_a)

    print(f"\n{'='*104}")
    print(f"Q-LANDSCAPE  ckpt={a.checkpoint}  cycle={a.cycle}  action_map={a.action_map}")
    print(f"{'='*104}")

    for name, (obs, snap, T, w, soc) in states.items():
        u_thr = 1.0 - _T_CUTOFF / T if T > _T_CUTOFF else float("-inf")
        print(f"\n--- {name}:  T_MGB={T:.2f} Nm  w_MGB={w:.1f} rad/s  SoC={soc*100:.2f}%"
              f"   u_thr(OFF)={u_thr:+.4f} ---")

        ot = th.as_tensor(np.repeat(obs.reshape(1, -1), len(grid), 0)).float().to(model.device)
        at = th.as_tensor(grid.reshape(-1, 1)).float().to(model.device)
        with th.no_grad():
            qs = model.critic(ot, at)
            q1 = qs[0].cpu().numpy().ravel()
            q2 = qs[1].cpu().numpy().ravel()
        minq = np.minimum(q1, q2)

        rewards, us, tces, modes = [], [], [], []
        for av in grid:
            e2 = copy.deepcopy(snap)           # exact same physical state
            act = np.array([av], np.float32)
            t_ce, t_em, u, mode = e2._action_to_torques(act)
            us.append(u); tces.append(t_ce)
            modes.append("OFF" if (mode != "regen" and t_ce <= _T_CUTOFF) else mode)
            _, r, _, _, _ = e2.step(act)
            rewards.append(r)
        rewards = np.array(rewards)

        print(f"   {'a':>7}{'u':>9}{'T_CE':>9}  {'mode':<9}{'reward':>10}"
              f"{'Q1':>11}{'Q2':>11}{'minQ':>11}")
        for i in range(0, len(grid), max(len(grid) // 14, 1)):
            print(f"   {grid[i]:>7.3f}{us[i]:>9.3f}{tces[i]:>9.2f}  {modes[i]:<9}"
                  f"{rewards[i]:>10.4f}{q1[i]:>11.2f}{q2[i]:>11.2f}{minq[i]:>11.2f}")

        off_m = np.array([m == "OFF" for m in modes])
        asst_m = np.array([m == "assist" for m in modes])
        print(f"   -> argmax minQ  at a={grid[minq.argmax()]:+.3f} (mode={modes[minq.argmax()]})")
        print(f"   -> argmax trueR at a={grid[rewards.argmax()]:+.3f} (mode={modes[rewards.argmax()]})")
        if off_m.any() and asst_m.any():
            dq = minq[off_m].max() - minq[asst_m].max()
            dr = rewards[off_m].max() - rewards[asst_m].max()
            agree = "CRITIC AGREES" if np.sign(dq) == np.sign(dr) else \
                    "*** CRITIC DISAGREES WITH REWARD ***"
            print(f"   -> best-OFF minus best-ASSIST:  trueR {dr:+.4f}   minQ {dq:+.3f}   {agree}")
        print(f"   -> OFF actions available: {off_m.sum()}/{len(grid)} "
              f"({100 * off_m.mean():.1f}% of action range)")
        # curvature / resolution of the critic across the OFF boundary
        if off_m.any() and asst_m.any():
            print(f"   -> minQ range across full action sweep: "
                  f"{minq.min():.2f} .. {minq.max():.2f}  (span {minq.max()-minq.min():.2f})")


if __name__ == "__main__":
    main()
