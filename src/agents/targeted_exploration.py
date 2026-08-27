"""
targeted_exploration.py
=======================
Phase-6 CONTROLLED CONDITIONAL-EXPLORATION intervention.

WHAT THIS IS
------------
A training-time-only exploration mechanism that raises the *data coverage* of
feasible engine-OFF actions in a specific conditional region of state space:

    torque demand  15 <= T_MGB < 35 Nm      (diagnostic focus 30-35 Nm)
    state of charge 0.40 <= SoC <= 0.55
    AND engine-OFF must be PHYSICALLY REACHABLE at that operating point

Phase-5B measured that in exactly this region the replay buffer holds only
~4.5% OFF transitions (~276 of 6,132 at 30-35 Nm / SoC 40-50), so the critic
fits Q(OFF) there from very few samples while the true reward prefers OFF
(dr(OFF-ASSIST) = +0.0011) and the critic disagrees (dQ = -0.0062..-0.0222).

WHAT THIS IS NOT
----------------
  * NOT imitation learning. No benchmark or ECMS action is ever used as a
    label, target, or demonstration. The injected action is drawn UNIFORMLY
    from the feasible OFF interval -- it carries no information about what a
    good controller would do beyond "engine off is physically possible here".
  * NOT a control rule. The trained policy is never forced to use OFF; only
    the *data distribution the critic learns from* is altered.
  * NOT active at evaluation. This overrides `_sample_action`, which SB3 calls
    ONLY from `collect_rollouts` (verified: SAC does not override it, and
    `predict()` never calls it). Every evaluation path in this project uses
    `model.predict(..., deterministic=True)` and is therefore unaffected.
  * NOT a reward, critic-target, architecture, gamma or k_fb change.

FEASIBILITY GUARANTEE
---------------------
Injection only happens when `_off_reachable()` is true, i.e. the motor
envelope can carry (T_MGB - T_CUTOFF) at the current speed/acceleration. The
injected action lies in [a_off, 1.0], which by construction of the action map
produces engine torque <= T_CUTOFF. The env's own feasibility masks still run
afterwards, unchanged.

USAGE
-----
    model = make_targeted(SAC)(..., te_enabled=True, te_prob=0.30)

CONTROL runs use `te_enabled=False`, which restores the stock SB3 behaviour
byte-for-byte (the override returns `super()._sample_action(...)` untouched).
"""
from __future__ import annotations

import numpy as np

from src.env.ems_env import (U_MIN, U_MAX, ZB_MODEAWARE, _EPS_T)
from src.env.powertrain import (_T_CUTOFF, _interp1d_linear, _w_EM_max_row,
                                _T_EM_max_arr, _THETA_EM)

# activation region (Phase-5B diagnostic region)
TE_T_LO, TE_T_HI = 15.0, 35.0
TE_SOC_LO, TE_SOC_HI = 0.40, 0.55


def _off_reachable(T: float, w: float, dw: float) -> bool:
    cap = max(_interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w)
              - abs(_THETA_EM * dw) - _EPS_T, 0.0)
    return cap >= T - _T_CUTOFF


def _a_off(T: float, w: float, dw: float, action_map: str) -> float:
    """Smallest action producing engine-OFF under the active action map."""
    if T <= _T_CUTOFF:
        return -1.0
    if action_map == "modeaware_gated" and _off_reachable(T, w, dw):
        return 2.0 * ZB_MODEAWARE - 1.0
    if action_map == "modeaware":
        return 2.0 * ZB_MODEAWARE - 1.0
    return 2.0 * ((1.0 - _T_CUTOFF / T) - U_MIN) / (U_MAX - U_MIN) - 1.0


def decode_obs(obs_row: np.ndarray):
    """Recover physical quantities from the normalised observation.

    Mirrors EMSEnv._make_observation exactly:
        obs[0] = w_MGB/300, obs[1] = dw_MGB/60, obs[2] = T_MGB/150,
        obs[4] = 2*SoC - 1
    """
    w = float(obs_row[0]) * 300.0
    dw = float(obs_row[1]) * 60.0
    T = float(obs_row[2]) * 150.0
    soc = (float(obs_row[4]) + 1.0) / 2.0
    return T, w, dw, soc


def make_targeted(base_cls):
    """Return a subclass of `base_cls` with targeted conditional exploration."""

    class TargetedExploration(base_cls):
        def __init__(self, *args, te_enabled: bool = False, te_prob: float = 0.30,
                     te_action_map: str = "linear", **kwargs):
            super().__init__(*args, **kwargs)
            self.te_enabled = bool(te_enabled)
            self.te_prob = float(te_prob)
            self.te_action_map = te_action_map
            # bookkeeping for the report (never used by learning)
            self.te_stats = dict(steps=0, in_region=0, feasible=0, injected=0)

        def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
            action, buffer_action = super()._sample_action(
                learning_starts, action_noise, n_envs)
            if not self.te_enabled or self._last_obs is None:
                return action, buffer_action

            obs = np.asarray(self._last_obs)
            for i in range(obs.shape[0]):
                self.te_stats["steps"] += 1
                T, w, dw, soc = decode_obs(obs[i])
                if not (TE_T_LO <= T < TE_T_HI and TE_SOC_LO <= soc <= TE_SOC_HI):
                    continue
                self.te_stats["in_region"] += 1
                if w <= 0.0 or not _off_reachable(T, w, dw):
                    continue
                self.te_stats["feasible"] += 1
                if np.random.rand() >= self.te_prob:
                    continue
                # uniform draw from the FEASIBLE OFF interval -- carries no
                # information about the benchmark/ECMS action
                lo = _a_off(T, w, dw, self.te_action_map)
                a = float(np.random.uniform(lo, 1.0))
                buffer_action[i, 0] = a
                action[i, 0] = a
                self.te_stats["injected"] += 1
            return action, buffer_action

    TargetedExploration.__name__ = f"Targeted{base_cls.__name__}"
    return TargetedExploration
