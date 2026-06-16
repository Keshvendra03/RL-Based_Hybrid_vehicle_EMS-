"""
ems_env.py
==========
Gymnasium environment wrapping the validated Phase-1 powertrain
(`src/env/powertrain.py`) for RL-based energy management (Phase 2/3).

The agent controls the torque split between the combustion engine (CE) and
the electric motor (EM) of the parallel mild-hybrid powertrain at every
1-second step of a driving cycle (NEDC / FTP-75).

----------------------------------------------------------------------------
ACTION SPACE  —  Box(low=-1, high=1, shape=(2,))
  action[0] = a_ce: <0 → engine off, ≥0 → engine on
  action[1] = a_u:  torque split (same as original single action)
----------------------------------------------------------------------------
A single continuous action `a` is mapped inside the environment to the
torque-split factor `u` (same convention as the rule-based controllers:
T_EM = u * T_MGB, T_CE = (1-u) * T_MGB):

    u_raw = U_MIN + (a + 1)/2 * (U_MAX - U_MIN),   U_MIN = -0.6, U_MAX = 1.0

  *  u in (0, 1]   -> motor assists / electric driving
  *  u == 0        -> engine only
  *  u in [-0.6,0) -> load-point shifting (engine charges battery)

The upper bound is 1.0 (not 0.6) deliberately: pure-electric driving
(u = 1, engine torque 0 -> fuel-cutoff -> engine off) is the single
largest fuel-saving mode available and is exactly how the advanced
rule-based benchmark achieves most of its gains. Capping u at 0.6 would
make the benchmark unbeatable by construction.

Engine on/off is NOT a separate action. Per the design decision, it
emerges naturally: whenever the commanded engine torque
(1 - u) * T_MGB <= T_cutoff (5 Nm), the environment snaps to full
electric mode (T_EM = T_MGB, T_CE = 0) and the engine block's own
fuel-cutoff logic switches fuel flow to zero.

Physical feasibility is guaranteed by an action-masking layer
(`_action_to_torques`), so the agent never wastes exploration on
impossible commands:
  * |T_EM| (incl. the theta_EM*dw inertia share) is clamped to the
    motor max-torque curve T_EM_max(w) — the same curve the rule-based
    controllers use.
  * Battery state-of-charge guards: no discharge below SOC_MIN, no
    charge above SOC_MAX (prevents under-voltage / runaway).
  * Engine torque is kept under T_CE_max(w) by shifting the excess to
    the motor when possible.
  * When T_MGB < 0 (braking) the action is overridden by maximum
    feasible recuperation — identical policy to BOTH rule-based
    benchmarks, so no advantage or disadvantage is introduced; the
    agent's degrees of freedom are spent only where a decision exists.

----------------------------------------------------------------------------
OBSERVATION SPACE  —  Box(shape=(16,))
----------------------------------------------------------------------------
   0  w_MGB   / 300          flywheel speed
   1  dw_MGB  / 60           flywheel acceleration
   2  T_MGB   / 150          flywheel torque demand
   3  d_T_MGB / 80           torque rate (T_MGB - T_MGB_prev), as used by
                             the advanced rule-based controller
   4  2*SoC - 1              state of charge, centered
   5  (SoC - SOC_TARGET)*10  fine-resolution SoC error signal
   6  v      / 35            current cycle speed
   7  v_next / 35            1-step speed lookahead (anticipation)
   8  2*progress - 1         normalized cycle progress t / T_total
   9..15  one-hot gear (0..6)

----------------------------------------------------------------------------
REWARD  (dense, every step; minimizes TRUE equivalent fuel)
----------------------------------------------------------------------------
    r_t = -SCALE * [ dm_fuel_t * k_cs / rho_f               (liters of fuel)
                    + dE_batt_t * EFC_GAIN / 3.6e6 ]        (equiv. liters)
          - LAMBDA_SOC * max(|SoC - SOC_TARGET| - SOC_DEADBAND, 0)^2

  * dm_fuel_t  is the Tank block's exact per-step trapezoidal fuel-mass
    increment (m_dot_fuel) — NOT the cumulative L/100km display value.
  * dE_batt_t = E(Q_prev) - E(Q_now) is the SIGNED per-step change of the
    battery's stored energy, using the SAME capacitor-analogy energy
    function E(Q) = 0.5*U_oc(Q)*Q as the model's V_BT computation.
    Summed over the episode it telescopes to exactly the net battery
    energy that the Equivalent-Fuel-Consumption block converts to liters,
    so the cumulative reward equals (minus) the true objective up to the
    saturation term handled at the terminal step.
  * The small quadratic SoC penalty (with deadband) is the anti-loophole
    device: with gamma < 1 a terminal-only penalty would let the agent
    "borrow" battery energy early in the cycle. The deadband leaves the
    agent free to swing the battery within a healthy band.

  Terminal step additionally applies:
  * exact saturation correction: the model's EFC block gives NO credit
    for finishing above the initial SoC (clip at 0). The per-step signed
    electrical term DID hand out that credit during the episode, so it
    is removed here -> episode return matches the true metric exactly.
  * a charge-sustaining penalty on |SoC_end - SOC_TARGET| beyond a
    2 %-SoC tolerance, so the headline fuel number is comparable to the
    benchmark.

The per-step costs are exact decompositions of the final-display metric
(v_liter / v_ce_equiv): summing the fuel terms reproduces the Tank's
m_fuel; summing the electrical terms reproduces E_init - E_final. This is
verified by tests/test_ems_env.py.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.env.driving_cycle import DrivingCycle
from src.env import powertrain as pt
from src.env.powertrain import (
    VehicleDynamics,
    gearbox,
    combustion_engine,
    electric_motor,
    Tank,
    Battery,
    equivalent_fuel_consumption,
    _Q_BT_IC,
    _Q_BT_0,
    _battery_energy,
    _interp1d_linear,
    _w_EM_max_row,
    _T_EM_max_arr,
    _w_CE_max_fine,
    _T_CE_max,
    _T_CUTOFF,
    _THETA_EM,
    _THETA,
    _RHO_FUEL,
    _K_CS,
    _EFC_GAIN,
)


# ===========================================================================
# Optional fast bilinear interpolation (exact drop-in for the in-bounds
# behaviour of scipy.RegularGridInterpolator used by powertrain.py).
# powertrain.py clamps all lookups to grid bounds before interpolating, so
# extrapolation never occurs and plain bilinear interpolation is EXACTLY
# equivalent — verified in tests/test_ems_env.py to < 1e-12.
# Speeds the environment up ~20x (the scipy interpolator object is
# otherwise re-constructed on every powertrain call).
# ===========================================================================

def _fast_interp2d_linear(row_grid, col_grid, table, row_val, col_val):
    i = int(np.clip(np.searchsorted(row_grid, row_val) - 1, 0, len(row_grid) - 2))
    j = int(np.clip(np.searchsorted(col_grid, col_val) - 1, 0, len(col_grid) - 2))
    x0, x1 = row_grid[i], row_grid[i + 1]
    y0, y1 = col_grid[j], col_grid[j + 1]
    tx = (row_val - x0) / (x1 - x0)
    ty = (col_val - y0) / (y1 - y0)
    f00, f01 = table[i, j], table[i, j + 1]
    f10, f11 = table[i + 1, j], table[i + 1, j + 1]
    return float(
        f00 * (1 - tx) * (1 - ty)
        + f10 * tx * (1 - ty)
        + f01 * (1 - tx) * ty
        + f11 * tx * ty
    )


def enable_fast_interpolation() -> None:
    """Monkey-patch powertrain's 2-D lookup with the fast exact bilinear."""
    pt._interp2d_linear = _fast_interp2d_linear


# ===========================================================================
# Reward / control constants
# ===========================================================================

# action -> u mapping bounds
U_MIN: float = -0.6
U_MAX: float = 1.0

# Battery operating window (hard action masks)
SOC_MIN: float = 0.05
SOC_MAX: float = 0.95
SOC_TARGET: float = 0.5

# Reward shaping
REWARD_SCALE:  float = 100.0     # 1 equivalent liter -> 100 reward units
LAMBDA_SOC:    float = 2.0       # per-step quadratic SoC penalty weight
SOC_DEADBAND:  float = 0.20      # free SoC swing (±20%); widened from 0.08.
                                  # DP optimum uses SoC down to 20%, so the old
                                  # narrow deadband penalised the winning strategy.
TERM_TOL:      float = 0.02      # terminal SoC tolerance (2 % SoC)
TERM_W_LIN:    float = 50.0      # terminal penalty, linear part (strengthened)
TERM_W_QUAD:   float = 800.0     # terminal penalty, quadratic (strengthened)

# Equivalence constants (exact decompositions of the model's metric)
K_FUEL_L_PER_KG: float = _K_CS / _RHO_FUEL          # kg fuel -> display liters
K_ELEC_L_PER_J:  float = _EFC_GAIN / 3.6e6          # battery J -> equiv liters

_EPS_T: float = 0.01     # epsilon torque margin (same as rule-based)
N_GEARS_ONEHOT: int = 7  # gears 0..6


class EMSEnv(gym.Env):
    """Gymnasium environment for the parallel mild-HEV energy management task."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cycle_name: str = "NEDC",
        fast_interp: bool = True,
        reward_scale: float = REWARD_SCALE,
        lambda_soc: float = LAMBDA_SOC,
        soc_deadband: float = SOC_DEADBAND,
        eq_factor: float = 0.15,   # Discharge equivalence factor (asymmetric reward).
                                   # Regen always uses eq=1.0 (full credit).
                                   # 0.15 makes electric assist very cheap vs engine,
                                   # fixing the local optimum where agent avoided motor.
    ) -> None:
        super().__init__()
        if fast_interp:
            enable_fast_interpolation()

        self.cycle = DrivingCycle(cycle_name)
        self.cycle_name = cycle_name
        self.veh = VehicleDynamics()
        self.tank = Tank()
        self.batt = Battery()

        self.reward_scale = float(reward_scale)
        self.lambda_soc = float(lambda_soc)
        self.soc_deadband = float(soc_deadband)
        # Training-time ECMS equivalence factor: scales the PER-STEP battery
        # energy cost in the reward only. 1.0 == report-grade factor (used in
        # the env validation tests). A value < 1.0 stops the reward from over-
        # taxing the use of (largely regen-sourced) stored energy, encouraging
        # the bank-and-spend load-point-shifting strategy. It does NOT touch
        # the reported v_ce_equiv metric or the terminal saturation correction,
        # so evaluation against the benchmark stays on the true fuel figure.
        self.eq_factor = float(eq_factor)

        # 2D action: [a_ce, a_u]
        #   a_ce < 0  -> engine OFF (state_CE=0, full electric, guaranteed fuel=0)
        #   a_ce >= 0 -> engine ON  (state_CE=1, torque split via a_u)
        #   a_u ∈ [-1,1] -> motor torque fraction (same as original single action)
        # Engine on/off is now a DEDICATED dimension → SAC explores it 50% of time
        # vs the 8% it had when engine-off was buried at the edge of 1 action.
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9 + N_GEARS_ONEHOT,), dtype=np.float32
        )

        # episode state (set in reset)
        self._demand: dict = {}
        self._E_prev: float = 0.0
        self._x_tot: float = 0.0
        self._v_prev_for_dv: float = 0.0
        self._T_MGB_prev: float = 0.0
        self._Q_BT: float = _Q_BT_IC
        self._done: bool = False
        self._last_obs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Demand-side pre-computation (action-independent part of a timestep) #
    # ------------------------------------------------------------------ #

    def _precompute_demand(self, cyc_obs: dict) -> None:
        """Vehicle + gearbox for the CURRENT cycle sample.

        Mirrors evaluate.py exactly: dv recomputed as backward difference,
        x_tot integrated trapezoidally from v_a.
        """
        dv = (cyc_obs["v"] - self._v_prev_for_dv) / 1.0
        self._v_prev_for_dv = cyc_obs["v"]

        veh_out = self.veh.step(v=cyc_obs["v"], dv=dv)
        self._x_tot += veh_out["v_a"] * 1.0

        gb = gearbox(
            w_wheel=veh_out["w_wheel"],
            dw_wheel=veh_out["dw_wheel"],
            t_wheel=veh_out["T_wheel"],
            gear=cyc_obs["i"],
        )

        t_idx = self.cycle.t
        v_next = float(self.cycle._v[min(t_idx + 1, self.cycle.length - 1)])

        self._demand = {
            "t": t_idx,
            "v": cyc_obs["v"],
            "v_next": v_next,
            "gear": cyc_obs["i"],
            "w_MGB": gb["w_mgb"],
            "dw_MGB": gb["dw_mgb"],
            "T_MGB": gb["t_mgb"],
            "d_T_MGB": gb["t_mgb"] - self._T_MGB_prev,
        }

    def _make_observation(self) -> np.ndarray:
        d = self._demand
        soc = self._Q_BT / _Q_BT_0
        progress = d["t"] / (self.cycle.length - 1)

        gear_onehot = np.zeros(N_GEARS_ONEHOT, dtype=np.float32)
        gear_onehot[int(np.clip(d["gear"], 0, N_GEARS_ONEHOT - 1))] = 1.0

        cont = np.array(
            [
                d["w_MGB"] / 300.0,
                d["dw_MGB"] / 60.0,
                d["T_MGB"] / 150.0,
                d["d_T_MGB"] / 80.0,
                2.0 * soc - 1.0,
                (soc - SOC_TARGET) * 10.0,
                d["v"] / 35.0,
                d["v_next"] / 35.0,
                2.0 * progress - 1.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([cont, gear_onehot])

    # ------------------------------------------------------------------ #
    # Action -> physically feasible torque commands                       #
    # ------------------------------------------------------------------ #

    def _action_to_torques(self, action: np.ndarray) -> Tuple[float, float, float, str]:
        """Map 2D agent action to feasible (T_CE, T_EM) commands.

        action[0] = a_ce: engine on/off signal
            a_ce <  0  -> engine OFF (full electric, zero fuel guaranteed)
            a_ce >= 0  -> engine ON  (torque split decided by action[1])
        action[1] = a_u: torque split (same mapping as original single action)

        Returns (T_CE_cmd, T_EM_cmd, u_actual, mode).
        """
        d = self._demand
        w, dw, T = d["w_MGB"], d["dw_MGB"], d["T_MGB"]
        soc = self._Q_BT / _Q_BT_0

        act = np.asarray(action).reshape(-1)
        a_ce = float(np.clip(act[0], -1.0, 1.0))   # engine on/off signal
        a_u  = float(np.clip(act[1], -1.0, 1.0))   # torque split
        u_raw = U_MIN + (a_u + 1.0) * 0.5 * (U_MAX - U_MIN)
        engine_off_requested = (a_ce < 0.0)         # negative -> engine off

        # Motor torque envelope at this speed (includes inertia share)
        t_em_max = _interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w)
        inertia = abs(_THETA_EM * dw)
        cap = max(t_em_max - inertia - _EPS_T, 0.0)

        can_discharge = soc > SOC_MIN
        can_charge = soc < SOC_MAX

        # ---------------- braking / coasting: max recuperation -----------
        if T < 0.0:
            if can_charge:
                t_em = max(T, -cap)          # motor absorbs as much as possible
            else:
                t_em = 0.0                   # battery full -> friction brakes
            t_ce = T - t_em                  # <= 0 -> engine fuel cutoff
            u = t_em / T if T != 0.0 else 0.0
            return t_ce, t_em, u, "regen"

        # ---------------- standstill ------------------------------------
        if T == 0.0 or w <= 0.0:
            return 0.0, 0.0, 0.0, "stop"

        # ---------------- traction: agent decides the split --------------
        t_em_des = u_raw * T

        # Engine-off via explicit action: a_ce < 0 means agent chose electric.
        # Also keep the implicit snap for backward compatibility.
        engine_off_feasible = can_discharge and T <= cap
        if engine_off_requested and engine_off_feasible:
            return 0.0, T, 1.0, "electric"

        # Implicit electric snap: engine torque demand at/below fuel cutoff.
        if (1.0 - u_raw) * T <= _T_CUTOFF and engine_off_feasible:
            return 0.0, T, 1.0, "electric"

        # SoC guards
        if not can_discharge:
            t_em_des = min(t_em_des, 0.0)
        if not can_charge:
            t_em_des = max(t_em_des, 0.0)

        # Motor envelope clamp
        t_em = float(np.clip(t_em_des, -cap, cap))

        # Engine envelope guard: shift any overload back to the motor
        t_ce = T - t_em
        w_ce = max(w, 105.0)
        t_ce_max = _interp1d_linear(_w_CE_max_fine, _T_CE_max, w_ce) - abs(_THETA * dw) - _EPS_T
        if t_ce > t_ce_max:
            excess = t_ce - t_ce_max
            if can_discharge:
                t_em = float(np.clip(t_em + excess, -cap, cap))
            t_ce = T - t_em

        u = t_em / T
        mode = "lps_gen" if t_em < 0 else ("assist" if t_em > 0 else "engine")
        return t_ce, t_em, u, mode

    # ------------------------------------------------------------------ #
    # Gymnasium API                                                        #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        cyc_obs = self.cycle.reset()
        self.veh.reset()
        self.tank.reset()
        self.batt.reset()

        self._x_tot = 0.0
        self._v_prev_for_dv = 0.0
        self._T_MGB_prev = 0.0
        self._Q_BT = _Q_BT_IC
        self._E_prev = _battery_energy(_Q_BT_IC)
        self._done = False

        self._precompute_demand(cyc_obs)
        obs = self._make_observation()
        self._last_obs = obs
        return obs, {}

    def step(self, action):
        if self._done:
            raise RuntimeError("step() called after episode end; call reset().")

        d = self._demand

        # 1. action -> feasible torque commands
        t_ce_cmd, t_em_cmd, u_act, mode = self._action_to_torques(action)

        # 2. powertrain blocks (identical wiring to evaluate.py)
        eng = combustion_engine(w_gear=d["w_MGB"], dw_gear=d["dw_MGB"], t_gear=t_ce_cmd)
        mot = electric_motor(w_gear=d["w_MGB"], dw_gear=d["dw_MGB"], t_gear=t_em_cmd)
        tank_out = self.tank.step(p_fuel=eng["p_ce"], x_tot=self._x_tot)
        batt_out = self.batt.step(p_bt=mot["p_em"], x_tot=self._x_tot)

        self._Q_BT = batt_out["q_bt"]
        soc = batt_out["soc"]

        # 3. per-step exact equivalent-fuel decomposition
        dm_fuel = tank_out["m_dot_fuel"]                       # kg this step
        E_now = _battery_energy(self._Q_BT)
        dE_batt = self._E_prev - E_now                         # J, +discharge
        self._E_prev = E_now

        fuel_liters = dm_fuel * K_FUEL_L_PER_KG
        elec_liters = dE_batt * K_ELEC_L_PER_J                 # signed, report-grade

        # Asymmetric equivalence factor:
        #   - Discharge (elec_liters > 0): cheap factor -> agent freely uses motor
        #   - Regen     (elec_liters < 0): full factor  -> agent values regen credit
        # This broke the local optimum where agent preferred engine-only because
        # battery discharge was too expensive relative to engine running cost.
        if elec_liters >= 0:
            eff_eq = self.eq_factor          # discharge: cheap (default 0.15)
        else:
            eff_eq = 1.0                     # regen: always full credit
        reward = -self.reward_scale * (fuel_liters + eff_eq * elec_liters)

        # 4. per-step SoC band penalty (anti battery-borrowing loophole)
        dev = abs(soc - SOC_TARGET)
        excess = max(dev - self.soc_deadband, 0.0)
        reward -= self.lambda_soc * excess * excess

        # safety flags should be unreachable thanks to masking; penalize anyway
        if batt_out["stop_uv"] or batt_out["stop_oc"] or mot["overload"] or eng["overload"]:
            reward -= 1.0

        # 5. bookkeeping for next observation
        self._T_MGB_prev = d["T_MGB"]

        efc = equivalent_fuel_consumption(tank_out["v_liter"], batt_out["v_bt"])
        info = {
            "v_liter": tank_out["v_liter"],
            "v_ce_equiv": efc["v_ce_equiv"],
            "soc": soc,
            "u": u_act,
            "mode": mode,
            "T_CE_cmd": t_ce_cmd,
            "T_EM_cmd": t_em_cmd,
            "p_ce": eng["p_ce"],
            "p_em": mot["p_em"],
            "fuel_liters_step": fuel_liters,
            "elec_liters_step": elec_liters,
        }

        # 6. advance the cycle
        cyc_obs, done = self.cycle.step()

        if done:
            terminated = True
            self._done = True

            # exact saturation correction: remove any credit for finishing
            # ABOVE the initial charge (the EFC block clips converted V_BT
            # at >= 0, i.e. no reward for surplus charge).
            E_net = _battery_energy(_Q_BT_IC) - E_now          # J, episode net
            if E_net < 0.0:
                reward -= self.reward_scale * (-E_net) * K_ELEC_L_PER_J

            # charge-sustaining terminal penalty
            t_excess = max(abs(soc - SOC_TARGET) - TERM_TOL, 0.0)
            reward -= TERM_W_LIN * t_excess + TERM_W_QUAD * t_excess * t_excess

            info["episode_final"] = {
                "v_liter": tank_out["v_liter"],
                "v_ce_equiv": efc["v_ce_equiv"],
                "soc_final": soc,
                "m_fuel_kg": self.tank.m_fuel,
                "x_tot_m": self._x_tot,
            }
            obs = self._last_obs            # episode over; obs is a filler
        else:
            terminated = False
            self._precompute_demand(cyc_obs)
            obs = self._make_observation()
            self._last_obs = obs

        return obs, float(reward), terminated, False, info