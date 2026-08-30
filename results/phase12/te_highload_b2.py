"""
PHASE 12B2  --  DEEP-LPS informative-coverage intervention (training-time only).

Only difference vs Phase-12B (te_highload.py): the injection interval is the
TOP 15 Nm of the STATE-DEPENDENT feasible engine-torque range, computed from the
environment's OWN authoritative _action_to_torques clamp -- NOT [1.3*demand,
0.9*T_CE_max] and NOT a fixed [55,75] Nm rule.

    TCE_max_feasible = max_{a in [-1,1]} _action_to_torques(a).T_CE
    TCE_min_feasible = min_{a in [-1,1]} _action_to_torques(a).T_CE
    TCE_deep_low  = max(TCE_min_feasible, TCE_max_feasible - 15.0)
    TCE_deep_high = TCE_max_feasible
    TCE_injected  ~ Uniform(TCE_deep_low, TCE_deep_high)
    a_injected    = the action whose EXECUTED T_CE == TCE_injected (monotone invert)

ELIGIBILITY (all must hold):
  15 <= T_MGB < 50 Nm ; w_MGB > 0 ; soc < 0.55 ;
  feasible interval non-empty and TCE_max_feasible - TCE_min_feasible >= 5 Nm ;
  p = 0.25.

NOT active at evaluation (overrides _sample_action, which SB3 calls only from
collect_rollouts). predict(deterministic=True) is untouched.
te_enabled=False restores stock SB3 byte-for-byte.

PURITY: the injected action is a uniform draw from a feasibility-defined
interval -- no ECMS action, no reward optimisation, no critic/actor query, no
expert. It carries no information beyond "the deepest feasible engine load is
reachable here".
"""
from __future__ import annotations
import numpy as np

from src.env.ems_env import EMSEnv
from src.env.powertrain import _Q_BT_0, _T_CUTOFF

TE_T_LO, TE_T_HI = 15.0, 50.0
TE_SOC_CAP = 0.55
DEEP_WINDOW_NM = 15.0
MIN_INTERVAL_NM = 5.0
N_SCAN = 161


def decode_obs(o):
    return (float(o[2]) * 150.0, float(o[0]) * 300.0, float(o[1]) * 60.0, (float(o[4]) + 1.0) / 2.0)


class DeepLPSInterval:
    """Feasible-engine-torque range + top-15-Nm injection, via EMSEnv._action_to_torques."""

    def __init__(self, cycle_name, action_map, lookahead):
        self._e = EMSEnv(cycle_name, action_map=action_map, lookahead=lookahead)
        self._grid = np.linspace(-1.0, 1.0, N_SCAN)

    def _scan(self, T, w, dw, soc):
        self._e._demand = dict(w_MGB=float(w), dw_MGB=float(dw), T_MGB=float(T), d_T_MGB=0.0)
        self._e._Q_BT = float(soc) * _Q_BT_0
        tce = np.empty(N_SCAN)
        for i, a in enumerate(self._grid):
            t_ce, _, _, _ = self._e._action_to_torques(np.array([a], np.float32))
            tce[i] = t_ce
        return tce

    def bounds(self, T, w, dw, soc):
        """Return (TCE_min_feasible, TCE_max_feasible, TCE_deep_low, TCE_deep_high) or None."""
        if not (w > 0.0 and T > _T_CUTOFF):
            return None
        tce = self._scan(T, w, dw, soc)
        tmin, tmax = float(tce.min()), float(tce.max())
        if tmax - tmin < MIN_INTERVAL_NM:
            return None
        lo = max(tmin, tmax - DEEP_WINDOW_NM)
        hi = tmax
        if hi - lo < 1e-6:
            return None
        return tmin, tmax, lo, hi

    def inject(self, T, w, dw, soc, rng):
        """Draw TCE ~ U(deep_low, deep_high); invert to action a; return
        dict(a, tce_req, tce_exec, tmin, tmax, clamped) or None."""
        b = self.bounds(T, w, dw, soc)
        if b is None:
            return None
        tmin, tmax, lo, hi = b
        tce_req = float(rng.uniform(lo, hi))
        # invert executed T_CE(a) -- monotone DECREASING in a (Phase-11 11B)
        tce = self._scan(T, w, dw, soc)          # re-scan (state already set by bounds->_scan)
        order = np.argsort(tce)
        a = float(np.interp(tce_req, tce[order], self._grid[order]))
        a = float(np.clip(a, -1.0, 1.0))
        t_ce_exec, _, _, _ = self._e._action_to_torques(np.array([a], np.float32))
        return dict(a=a, tce_req=tce_req, tce_exec=float(t_ce_exec), tmin=tmin, tmax=tmax,
                    clamped=bool(abs(float(t_ce_exec) - tce_req) > 1.0))


def make_deeplps_targeted(base_cls):
    class DeepLPSTargeted(base_cls):
        def __init__(self, *args, te_enabled=False, te_prob=0.25, te_cycle="NEDC",
                     te_action_map="modeaware_gated", te_lookahead=5,
                     te_event_log_cap=8000, **kwargs):
            super().__init__(*args, **kwargs)
            self.te_enabled = bool(te_enabled)
            self.te_prob = float(te_prob)
            self._rng = np.random.default_rng(kwargs.get("seed", 0) or 0)
            self._iv = DeepLPSInterval(te_cycle, te_action_map, te_lookahead) if self.te_enabled else None
            self._cap = int(te_event_log_cap)
            self.te_stats = dict(steps=0, in_region=0, feasible=0, injected=0,
                                 tce_exec_sum=0.0, tce_exec_min=1e9, tce_exec_max=-1e9,
                                 rho_sum=0.0, n_clamped=0, fidelity_sum=0.0)
            self.te_events = []     # capped reservoir of intervention events (audit)

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
                b = self._iv.bounds(T, w, dw, soc)
                if b is None:
                    continue
                self.te_stats["feasible"] += 1
                if self._rng.random() >= self.te_prob:
                    continue
                ev = self._iv.inject(T, w, dw, soc, self._rng)
                if ev is None:
                    continue
                a = ev["a"]
                buffer_action[i, 0] = a
                action[i, 0] = a
                s = self.te_stats
                s["injected"] += 1
                s["tce_exec_sum"] += ev["tce_exec"]
                s["tce_exec_min"] = min(s["tce_exec_min"], ev["tce_exec"])
                s["tce_exec_max"] = max(s["tce_exec_max"], ev["tce_exec"])
                rho = (ev["tce_exec"] - ev["tmin"]) / max(ev["tmax"] - ev["tmin"], 1e-9)
                s["rho_sum"] += rho
                s["n_clamped"] += int(ev["clamped"])
                s["fidelity_sum"] += (ev["tce_exec"] / ev["tce_req"]) if ev["tce_req"] > 1e-6 else 1.0
                if len(self.te_events) < self._cap:
                    self.te_events.append(dict(step=int(self.num_timesteps), T=round(T, 2),
                        soc=round(soc, 4), tce_req=round(ev["tce_req"], 3),
                        tce_exec=round(ev["tce_exec"], 3), tmin=round(ev["tmin"], 3),
                        tmax=round(ev["tmax"], 3), rho=round(float(rho), 4),
                        clamped=ev["clamped"], a=round(a, 5)))
            return action, buffer_action

    DeepLPSTargeted.__name__ = f"DeepLPSTargeted{base_cls.__name__}"
    return DeepLPSTargeted
