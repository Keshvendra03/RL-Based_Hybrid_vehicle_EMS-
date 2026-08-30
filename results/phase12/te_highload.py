"""
PHASE 12B  --  targeted INFORMATIVE-COVERAGE intervention for the
HIGH-ENGINE-LOAD region (training-time only).

Parallel to src/agents/targeted_exploration.py (Phase-6 OFF injection), but here
the injected action is a uniform draw from the ACTUAL FEASIBLE high-engine-load
interval, computed from the current state via the env's own _action_to_torques
(no ECMS action, no expert torque, no benchmark label).

ACTIVATION (all must hold):
  * 15 <= T_MGB < 50 Nm            (Phase-11 low/mid-demand problem region)
  * w_MGB > 0                       (moving traction step)
  * soc < 0.70                      (SAFETY CAP: never inject charging actions
                                     into the eq_eff<0 zone; well above the
                                     ~37-45% operating band, below the FTP75
                                     60.9%/69.9% zero-crossing -- see Stage A)
  * a non-degenerate feasible high-load interval exists:
        T_CE_hi_nom = 0.9 * T_CE_max_feasible(state)     [true env clamp]
        T_CE_lo_nom = 1.3 * T_MGB
        require  T_CE_hi_nom - T_CE_lo_nom >= 5 Nm  after intersecting with the
        env's true feasible action set; otherwise SKIP (no injection).

INJECTION: with probability p (default 0.25), replace the sampled exploratory
action with  a ~ Uniform(a_lo, a_hi)  where [a_lo, a_hi] maps (via the monotone
action->T_CE relation, verified continuous/monotone in Phase-11 11B) to the
feasible T_CE sub-interval [T_CE_lo_nom, T_CE_hi_nom] intersected with [-1, 1].
The env's own feasibility masks then run unchanged on top.

NOT active at evaluation: this overrides `_sample_action`, which SB3 calls ONLY
from `collect_rollouts`. `predict(deterministic=True)` never calls it.
te_enabled=False restores stock SB3 byte-for-byte.
"""
from __future__ import annotations
import numpy as np

from src.env.ems_env import EMSEnv
from src.env.powertrain import _Q_BT_0, _T_CUTOFF

TE_T_LO, TE_T_HI = 15.0, 50.0
TE_SOC_CAP = 0.70
TCE_LO_MULT = 1.3          # nominal lower edge  = 1.3 * demand
TCE_HI_MULT = 0.9          # nominal upper edge  = 0.9 * max-feasible T_CE
MIN_INTERVAL_NM = 5.0      # "multiple feasible engine-load choices exist"
N_SCAN = 81


def decode_obs(obs_row: np.ndarray):
    """Mirror EMSEnv._make_observation: obs[0]=w/300, obs[1]=dw/60, obs[2]=T/150, obs[4]=2*soc-1."""
    w = float(obs_row[0]) * 300.0
    dw = float(obs_row[1]) * 60.0
    T = float(obs_row[2]) * 150.0
    soc = (float(obs_row[4]) + 1.0) / 2.0
    return T, w, dw, soc


class HighLoadInterval:
    """Computes the feasible high-engine-load action interval via the exact
    production path EMSEnv._action_to_torques (one persistent throwaway env)."""

    def __init__(self, cycle_name: str, action_map: str, lookahead: int):
        # a bare env used ONLY to call _action_to_torques with an injected _demand;
        # never stepped, never part of training, never sees a reward.
        self._e = EMSEnv(cycle_name, action_map=action_map, lookahead=lookahead)
        self._grid = np.linspace(-1.0, 1.0, N_SCAN)

    def interval(self, T, w, dw, soc):
        if not (w > 0.0 and T > _T_CUTOFF):
            return None
        self._e._demand = dict(w_MGB=float(w), dw_MGB=float(dw), T_MGB=float(T), d_T_MGB=0.0)
        self._e._Q_BT = float(soc) * _Q_BT_0
        tce = np.empty(N_SCAN)
        for i, a in enumerate(self._grid):
            t_ce, _, _, _ = self._e._action_to_torques(np.array([a], np.float32))
            tce[i] = t_ce
        tce_max = float(tce.max())
        lo_nom = TCE_LO_MULT * T
        hi_nom = TCE_HI_MULT * tce_max
        if hi_nom - lo_nom < MIN_INTERVAL_NM:
            return None
        # tce is monotone DECREASING in a (Phase-11 11B). Invert by interpolation.
        order = np.argsort(tce)                 # ascending tce
        tce_s, a_s = tce[order], self._grid[order]
        a_hi = float(np.interp(lo_nom, tce_s, a_s))   # lower T_CE  -> higher a
        a_lo = float(np.interp(hi_nom, tce_s, a_s))   # higher T_CE -> lower  a
        a_lo = max(a_lo, -1.0); a_hi = min(a_hi, 1.0)
        if a_hi - a_lo < 1e-4:
            return None
        return a_lo, a_hi


def make_highload_targeted(base_cls):
    """Return a subclass of `base_cls` with high-engine-load informative-coverage exploration."""

    class HighLoadTargeted(base_cls):
        def __init__(self, *args, te_enabled: bool = False, te_prob: float = 0.25,
                     te_cycle: str = "NEDC", te_action_map: str = "modeaware_gated",
                     te_lookahead: int = 5, **kwargs):
            super().__init__(*args, **kwargs)
            self.te_enabled = bool(te_enabled)
            self.te_prob = float(te_prob)
            self._hi = HighLoadInterval(te_cycle, te_action_map, te_lookahead) if self.te_enabled else None
            self.te_stats = dict(steps=0, in_region=0, feasible=0, injected=0,
                                 injected_tce_sum=0.0, injected_tce_min=1e9, injected_tce_max=-1e9)
            self._te_last_probe = self._hi  # for reuse in dry-run

        def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
            action, buffer_action = super()._sample_action(learning_starts, action_noise, n_envs)
            if not self.te_enabled or self._last_obs is None:
                return action, buffer_action
            obs = np.asarray(self._last_obs)
            for i in range(obs.shape[0]):
                self.te_stats["steps"] += 1
                T, w, dw, soc = decode_obs(obs[i])
                if not (TE_T_LO <= T < TE_T_HI and w > 0.0 and soc < TE_SOC_CAP):
                    continue
                self.te_stats["in_region"] += 1
                iv = self._hi.interval(T, w, dw, soc)
                if iv is None:
                    continue
                self.te_stats["feasible"] += 1
                if np.random.rand() >= self.te_prob:
                    continue
                a_lo, a_hi = iv
                a = float(np.random.uniform(a_lo, a_hi))
                a = float(np.clip(a, -1.0, 1.0))
                buffer_action[i, 0] = a
                action[i, 0] = a
                self.te_stats["injected"] += 1
                # record the executed engine load for the report (audit only)
                self._hi._e._demand = dict(w_MGB=float(w), dw_MGB=float(dw), T_MGB=float(T), d_T_MGB=0.0)
                self._hi._e._Q_BT = float(soc) * _Q_BT_0
                t_ce, _, _, _ = self._hi._e._action_to_torques(np.array([a], np.float32))
                self.te_stats["injected_tce_sum"] += float(t_ce)
                self.te_stats["injected_tce_min"] = min(self.te_stats["injected_tce_min"], float(t_ce))
                self.te_stats["injected_tce_max"] = max(self.te_stats["injected_tce_max"], float(t_ce))
            return action, buffer_action

    HighLoadTargeted.__name__ = f"HighLoadTargeted{base_cls.__name__}"
    return HighLoadTargeted
