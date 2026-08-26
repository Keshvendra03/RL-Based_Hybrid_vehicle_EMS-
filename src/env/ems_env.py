"""
ems_env.py
==========
Gymnasium environment wrapping the validated Phase-1 powertrain
(`src/env/powertrain.py`) for RL-based energy management (Phase 2/3).

The agent controls the torque split between the combustion engine (CE) and
the electric motor (EM) of the parallel mild-hybrid powertrain at every
1-second step of a driving cycle (NEDC / FTP-75).

----------------------------------------------------------------------------
ACTION SPACE  —  Box(low=-1, high=1, shape=(1,))
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
verified by tests/test_ems_env.py. This exact-telescoping property holds
at k_fb=0 (default); see the k_fb costate-feedback option below.

OPTIONAL: ECMS-STYLE COSTATE FEEDBACK ON eq_factor  (k_fb, default 0.0)
------------------------------------------------------------------------
    eq_factor_eff = eq_factor + k_fb * (SOC_TARGET - soc_before)

Mirrors src/baselines/ecms.py's PROVEN closed-loop pricing (k_fb=8.0
there): battery energy gets more expensive as SoC falls below target and
cheaper as it rises above. ecms.py's own docstring shows a CONSTANT price
cannot make this plant charge-sustaining even for the Hamiltonian-optimal
controller -- k_fb=0 (flat eq_factor) inherits that same limitation as a
training signal. Non-zero k_fb intentionally breaks the exact-telescoping
property above (this becomes a shaped training signal); v_ce_equiv itself
is computed independently from the fixed-efficiency EFC block and is
unaffected either way.
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
U_MIN: float = -0.85   # was -0.6. Benchmark charges to u=-0.749 (gear 1);
                       # -0.85 lets the agent bank at least as deep as the benchmark.
U_MAX: float = 1.0

# Battery operating window (hard action masks)
SOC_MIN: float = 0.05
SOC_MAX: float = 0.95
SOC_TARGET: float = 0.5

# Reward shaping
REWARD_SCALE:  float = 100.0     # 1 equivalent liter -> 100 reward units
LAMBDA_SOC:    float = 2.0       # per-step QUADRATIC SoC penalty weight
LAMBDA_SOC_LIN: float = 1.0      # per-step LINEAR SoC restoring weight.
                                  # Was 3.0 — that pinned SoC at 50% but OVER-taxed
                                  # the legitimate bank-and-spend swings the policy
                                  # needs to beat the benchmark, plateauing fuel at
                                  # ~3.9 (above benchmark). 1.0 keeps SoC bounded
                                  # without suppressing regen timing.
SOC_DEADBAND:  float = 0.10      # free SoC swing (±10%). Middle ground:
                                  # 0.30 (old) gave NO control -> SoC locked at 67%;
                                  # 0.05 gave too MUCH control -> fuel stuck at 3.9.
                                  # 0.10 bounds SoC yet frees the ~45-55% band the
                                  # bank-and-spend strategy uses to capture regen.
TERM_TOL:      float = 0.02      # terminal SoC tolerance (2 % SoC)
TERM_W_LIN:    float = 50.0      # terminal penalty, linear part (strengthened)
TERM_W_QUAD:   float = 800.0     # terminal penalty, quadratic (strengthened)

# Equivalence constants (exact decompositions of the model's metric)
K_FUEL_L_PER_KG: float = _K_CS / _RHO_FUEL          # kg fuel -> display liters
K_ELEC_L_PER_J:  float = _EFC_GAIN / 3.6e6          # battery J -> equiv liters

_EPS_T: float = 0.01     # epsilon torque margin (same as rule-based)
N_GEARS_ONEHOT: int = 7  # gears 0..6

# ===========================================================================
# ACTION -> u MAPPING  (RL reparameterization layer -- NOT vehicle physics)
# ===========================================================================
# "linear" (default) is the original mapping, bit-for-bit unchanged.
#
# "modeaware" is a pure REPARAMETERIZATION motivated by a measured defect:
# engine-OFF requires (1-u)*T_MGB <= T_CUTOFF, i.e. u >= 1 - T_CUTOFF/T_MGB.
# Under the linear map that OFF region occupies a median of only 12.2% (NEDC)
# / 9.3% (FTP75) of the action range, is <10% wide on ~half of all traction
# steps, and -- worse -- its boundary MOVES with torque demand (a_off spans
# +0.40..+0.89). The benchmarks spend 40-59% of moving time in OFF, so the
# policy must hold a narrow, state-varying sliver at the action boundary,
# which SAC's entropy objective actively resists. See RL_DIAGNOSTIC_REPORT.md.
#
# The mode-aware map assigns FIXED fractions of the action range to the three
# traction modes by anchoring the breakpoints on the true OFF boundary:
#     z in [0,   ZA] -> u in [U_MIN, 0    ]   (LPS / charging)
#     z in [ZA,  ZB] -> u in [0,     u_thr]   (ASSIST)
#     z in [ZB,  1 ] -> u in [u_thr, U_MAX]   (engine OFF / pure electric)
# with z = (a+1)/2 and u_thr = 1 - T_CUTOFF/T_MGB.
#
# CONTROL EQUIVALENCE (proved in tests/test_action_mapping.py):
#   * u(a=-1) == U_MIN and u(a=+1) == U_MAX exactly, for every T.
#   * strictly monotonic in a  ->  bijective onto [U_MIN, U_MAX].
#   * the REACHABLE set of u -- hence of (T_CE, T_EM) -- is IDENTICAL.
# Only the *density* of the action coordinate changes. No plant equation,
# torque limit, feasibility mask, reward, or benchmark is touched.
# "modeaware_gated": identical to "modeaware" WHERE ENGINE-OFF IS PHYSICALLY
# REACHABLE, and falls back to "linear" where it is not. Phase-4 measurement:
# the ungated map spends 40% of the action range on OFF even above ~50 Nm where
# the motor envelope makes OFF impossible, which compresses ASSIST and drove the
# policy to LPS 100% at >75 Nm (+0.1464 L/100km). Gating keeps the Phase-4 gain
# in 30-35 Nm (-0.1129 L/100km) without the high-torque distortion.
ACTION_MAPS = ("linear", "modeaware", "modeaware_gated")
ZA_MODEAWARE: float = 0.35   # share of action range for LPS   (u < 0)
ZB_MODEAWARE: float = 0.60   # ZB-ZA = 25% for ASSIST; 1-ZB = 40% for OFF


def _off_reachable(t_mgb: float, w: float, dw: float) -> bool:
    """True if the motor envelope can carry the whole demand (minus the engine
    cutoff torque), i.e. engine-OFF is physically achievable at this point."""
    cap = max(_interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w)
              - abs(_THETA_EM * dw) - _EPS_T, 0.0)
    return cap >= t_mgb - _T_CUTOFF


def map_action_to_u(a: float, t_mgb: float, action_map: str = "linear",
                    w: float = 0.0, dw: float = 0.0) -> float:
    """Map agent action a in [-1,1] to the torque-split factor u.

    Both maps are monotonic bijections [-1,1] -> [U_MIN, U_MAX], so they
    expose exactly the same physical control authority (see module note).
    """
    z = (a + 1.0) * 0.5                      # [0,1]
    if action_map == "linear":
        return U_MIN + z * (U_MAX - U_MIN)

    if action_map == "modeaware_gated" and not _off_reachable(t_mgb, w, dw):
        # OFF unreachable here -> don't waste action range on it
        return U_MIN + z * (U_MAX - U_MIN)

    # --- mode-aware ---------------------------------------------------
    # Braking / sub-cutoff demand has no meaningful OFF boundary (the engine
    # is already below cutoff for any u < 1), so fall back to the linear map
    # -- keeps regen behaviour bit-identical to the original environment.
    if t_mgb <= _T_CUTOFF:
        return U_MIN + z * (U_MAX - U_MIN)

    u_thr = 1.0 - _T_CUTOFF / t_mgb
    # keep the three segments strictly ordered and non-degenerate
    u_thr = float(np.clip(u_thr, 1e-3, U_MAX - 1e-3))

    if z <= ZA_MODEAWARE:                    # LPS: U_MIN -> 0
        return U_MIN + (0.0 - U_MIN) * (z / ZA_MODEAWARE)
    if z <= ZB_MODEAWARE:                    # ASSIST: 0 -> u_thr
        return u_thr * (z - ZA_MODEAWARE) / (ZB_MODEAWARE - ZA_MODEAWARE)
    # OFF: u_thr -> U_MAX
    return u_thr + (U_MAX - u_thr) * (z - ZB_MODEAWARE) / (1.0 - ZB_MODEAWARE)


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
        eq_factor: float = 1.0,    # Equivalence factor for battery energy in the
                                   # reward. 1.0 == report-grade: the per-step reward
                                   # then telescopes EXACTLY to (minus) v_ce_equiv, so
                                   # the training objective == the true objective.
                                   # (The old asymmetric 0.15 discharge / 1.0 regen
                                   # scheme rewarded battery THROUGHPUT and misranked
                                   # policies — removed.)
        k_fb: float = 0.0,        # ECMS-style closed-loop costate feedback on
                                   # eq_factor: eq_factor_eff = eq_factor + k_fb *
                                   # (SOC_TARGET - soc), evaluated at the PRE-decision
                                   # SoC each step -- exactly mirrors the proven
                                   # formula in src/baselines/ecms.py (k_fb=8.0 there).
                                   # Default 0.0 reproduces the old static-price
                                   # behavior exactly (eq_factor_eff == eq_factor).
                                   # WHY THIS EXISTS: ecms.py's own docstring proves
                                   # a CONSTANT lambda cannot make this plant charge-
                                   # sustaining (the Hamiltonian has a near-vertical
                                   # SoC-vs-lambda cliff); k_fb closes that loop for
                                   # the optimal Hamiltonian-minimizing controller,
                                   # and there is no reason a flat price would work
                                   # any better as a REWARD signal for an RL policy
                                   # trying to solve the same charge-sustaining
                                   # problem. This does NOT change the evaluation
                                   # metric (v_ce_equiv is still computed from the
                                   # fixed physical EFC block) -- only the training
                                   # signal, same as the SoC penalty already does.
        obs_clean: bool = False,   # §21 ablation: drop the two dead observation
                                   # channels -- v_next (byte-identical to
                                   # fut_v1 when lookahead>0) and gear_oh6
                                   # (always 0.0 on both cycles). 20 -> 18 dims.
        action_map: str = "linear",  # "linear" (original, default) or "modeaware".
                                     # Pure action-coordinate reparameterization;
                                     # identical reachable u/torque set either way.
                                     # See ACTION_MAPS note above.
        lookahead: int = 0,        # Number of upcoming prescribed speeds (causal,
                                   # from the drive cycle's own future demand — NOT
                                   # controller knowledge) appended to the
                                   # observation, replacing the absolute cycle-progress
                                   # scalar. 0 (default) = original 16-dim observation,
                                   # unchanged from the pre-lookahead environment.
                                   # 5 is the validated generalization-safe window
                                   # (long enough to inform the engine-off/on decision,
                                   # short enough not to fingerprint the cycle).
    ) -> None:
        super().__init__()
        if fast_interp:
            enable_fast_interpolation()

        self.cycle = DrivingCycle(cycle_name)
        self.cycle_name = cycle_name
        self.veh = VehicleDynamics()
        self.tank = Tank()
        self.batt = Battery()

        self.lookahead = int(lookahead)

        self.reward_scale = float(reward_scale)
        self.lambda_soc = float(lambda_soc)
        self.lambda_soc_lin = float(LAMBDA_SOC_LIN)
        self.soc_deadband = float(soc_deadband)
        # Training-time ECMS equivalence factor: scales the PER-STEP battery
        # energy cost in the reward only. 1.0 == report-grade factor (used in
        # the env validation tests). A value < 1.0 stops the reward from over-
        # taxing the use of (largely regen-sourced) stored energy, encouraging
        # the bank-and-spend load-point-shifting strategy. It does NOT touch
        # the reported v_ce_equiv metric or the terminal saturation correction,
        # so evaluation against the benchmark stays on the true fuel figure.
        self.eq_factor = float(eq_factor)
        self.k_fb = float(k_fb)
        if action_map not in ACTION_MAPS:
            raise ValueError(f"action_map must be one of {ACTION_MAPS}, got {action_map!r}")
        self.action_map = action_map
        self.obs_clean = bool(obs_clean)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Base obs = 9 continuous + 7 gear one-hot. Lookahead appends `lookahead`
        # normalized upcoming speeds and REPLACES the absolute progress scalar
        # (a cycle fingerprint) with local future context -> net dim:
        #   (9 - 1 progress) + 7 gears + lookahead = 15 + lookahead
        obs_dim = (9 - 1) + N_GEARS_ONEHOT + self.lookahead if self.lookahead > 0 \
            else 9 + N_GEARS_ONEHOT
        if self.obs_clean:
            # §21 ablation: drop v_next (byte-identical to fut_v1 when
            # lookahead>0) and gear_oh6 (always 0.0 on both cycles).
            obs_dim -= 2 if self.lookahead > 0 else 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
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

        if self.lookahead > 0:
            # Drop absolute `progress` (cycle fingerprint -> hurts generalization);
            # replace with LOCAL future demand: next `lookahead` prescribed speeds,
            # normalized like the other speed channels. Causal & cycle-agnostic.
            fut = self.cycle.future_speeds(self.lookahead) / 35.0
            base = [
                d["w_MGB"] / 300.0,
                d["dw_MGB"] / 60.0,
                d["T_MGB"] / 150.0,
                d["d_T_MGB"] / 80.0,
                2.0 * soc - 1.0,
                (soc - SOC_TARGET) * 10.0,
                d["v"] / 35.0,
            ]
            if not self.obs_clean:
                base.append(d["v_next"] / 35.0)   # duplicate of fut[0]
            cont = np.array(base, dtype=np.float32)
            go = gear_onehot[:-1] if self.obs_clean else gear_onehot
            return np.concatenate([cont, fut, go])

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
        # At lookahead=0 there is no fut_v1, so v_next is NOT a duplicate and is
        # kept; only gear_oh6 (always 0.0) is dropped under obs_clean.
        go = gear_onehot[:-1] if self.obs_clean else gear_onehot
        return np.concatenate([cont, go])

    # ------------------------------------------------------------------ #
    # Action -> physically feasible torque commands                       #
    # ------------------------------------------------------------------ #

    def _action_to_torques(self, action: np.ndarray) -> Tuple[float, float, float, str]:
        """Map agent action to feasible (T_CE, T_EM) commands.

        Returns (T_CE_cmd, T_EM_cmd, u_actual, mode).
        """
        d = self._demand
        w, dw, T = d["w_MGB"], d["dw_MGB"], d["T_MGB"]
        soc = self._Q_BT / _Q_BT_0

        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        u_raw = map_action_to_u(a, T, self.action_map, w, dw)

        # ---------------- standstill ------------------------------------
        if T == 0.0 or w <= 0.0:
            return 0.0, 0.0, 0.0, "stop"

        # ================================================================
        # CONTROL-EQUIVALENT linear split (matches the benchmark wiring
        # control_unit_advanced: T_EM = u*T_MGB, T_CE = (1-u)*T_MGB), with
        # ONLY feasibility clamps that the benchmark itself also respects.
        #
        # Why this changed: the previous version added (a) an "electric snap"
        # that REDIRECTED the sub-cutoff engine torque onto the motor whenever
        # (1-u)*T <= T_CUTOFF, and (b) a FORCED maximum-regen override on every
        # braking step. Both make the plant non-equivalent to the benchmark:
        # the benchmark's own optimal u-trajectory scored 3.815 (SoC drift to
        # 55%) through the old interface vs 3.506 through the faithful wiring.
        # That 0.31 L/100km was an interface artefact handicapping the agent.
        #
        # Engine-off still emerges for free: when (1-u)*T <= T_CUTOFF (5 Nm),
        # combustion_engine's own fuel-cutoff zeroes the fuel WITHOUT moving
        # torque onto the motor — exactly as in the benchmark.
        # ================================================================

        # Motor torque envelope at this speed (includes inertia share). This is
        # the SAME limit the benchmark's _u_motor / _u_regen helpers enforce in
        # u-space, so it never bites on a feasible benchmark-equivalent command.
        t_em_max = _interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w)
        inertia = abs(_THETA_EM * dw)
        cap = max(t_em_max - inertia - _EPS_T, 0.0)

        # 1) linear split (no snap, no forced regen — agent owns u on every step)
        t_em = u_raw * T

        # 2) motor envelope clamp (feasibility only)
        t_em = float(np.clip(t_em, -cap, cap))

        # 3) battery SoC hard limits (safety masks; rarely bind in-band)
        #    t_em > 0 => discharge (assist / electric);  t_em < 0 => charge (LPS / regen)
        if soc <= SOC_MIN:
            t_em = min(t_em, 0.0)            # empty -> cannot discharge
        if soc >= SOC_MAX:
            t_em = max(t_em, 0.0)            # full  -> cannot charge / regen

        # 4) engine torque envelope guard: shift any over-torque back to the
        #    motor when the battery allows (engine cannot exceed its max curve).
        t_ce = T - t_em
        if T > 0.0:
            w_ce = max(w, 105.0)
            t_ce_max = _interp1d_linear(_w_CE_max_fine, _T_CE_max, w_ce) \
                - abs(_THETA * dw) - _EPS_T
            if t_ce > t_ce_max and soc > SOC_MIN:
                t_em = float(np.clip(t_em + (t_ce - t_ce_max), -cap, cap))
                t_ce = T - t_em

        u = t_em / T if T != 0.0 else 0.0
        if T < 0.0:
            mode = "regen"
        else:
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
        soc_before = self._Q_BT / _Q_BT_0   # pre-decision SoC, for costate feedback

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

        # ALIGNED reward: price battery energy symmetrically with eq_factor
        # (default 1.0 == report-grade). With k_fb=0, eq_factor_eff == eq_factor
        # and the per-step electrical cost telescopes over the episode to exactly
        # the net battery depletion that the EFC block converts to liters, so the
        # cumulative reward equals (minus) the true v_ce_equiv objective (up to
        # the terminal saturation correction below) -- same as before. This
        # removes the spurious "battery throughput" bonus produced by the old
        # discharge/regen asymmetry.
        #
        # With k_fb != 0, eq_factor_eff is the ECMS-style costate: battery energy
        # gets MORE expensive as SoC falls below target and CHEAPER as it rises
        # above, mirroring src/baselines/ecms.py's proven closed-loop formula.
        # This intentionally breaks the exact-telescoping property above (the
        # reward is now a SHAPED training signal, not a running total of the
        # true metric) -- v_ce_equiv itself is unaffected, since it's computed
        # separately from the fixed-efficiency EFC block, not from this reward.
        eq_factor_eff = self.eq_factor + self.k_fb * (SOC_TARGET - soc_before)
        reward = -self.reward_scale * (fuel_liters + eq_factor_eff * elec_liters)

        # 4. per-step SoC restoring penalty (charge-sustaining feedback).
        #    ROOT-CAUSE FIX: quadratic ALONE with a wide deadband left a large
        #    dead zone where charging drift went unpunished. Now: small deadband
        #    + LINEAR + quadratic, so there is a nonzero per-step force pulling
        #    SoC back to target from ~55% onward (and symmetric below 45%),
        #    competing directly with the per-step charging reward that caused the
        #    runaway to ~67%. The 45-55% band stays ~free for bank-and-spend.
        dev = abs(soc - SOC_TARGET)
        excess = max(dev - self.soc_deadband, 0.0)
        reward -= self.lambda_soc * excess * excess
        reward -= self.lambda_soc_lin * excess

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
