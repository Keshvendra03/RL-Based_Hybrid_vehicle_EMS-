"""
powertrain.py
=============
Pure Python/NumPy equivalents of every Simulink powertrain block.

Each function is stateless (no class, no hidden memory) EXCEPT vehicle_dynamics
which carries one state variable: v_prev (previous speed for average-speed calc).

Block mapping  (Simulink  →  Python function)
---------------------------------------------
  Vehicle          →  vehicle_dynamics()
  Manual Gear Box  →  gearbox()          [next block]
  Engine           →  engine_power()     [next block]
  Motor            →  motor_power()      [next block]
  Battery          →  battery_step()     [next block]
  Tank             →  tank_step()        [next block]

All variable names match the Simulink wire labels exactly.
Units are SI throughout (m, s, kg, N, Nm, W, rad).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Load parameters once at import time
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    candidate = Path(__file__).resolve()
    for _ in range(10):
        candidate = candidate.parent
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Cannot find project root from " + str(Path(__file__).resolve()))


_PARAMS_PATH = _find_project_root() / "data" / "params.json"

with open(_PARAMS_PATH, "r") as _f:
    _P = json.load(_f)

# Shorthand dicts
_VP = _P["vehicle"]
_GP = _P["gearbox"]


# ---------------------------------------------------------------------------
# Helper: derived constants (computed once, not recomputed every timestep)
# ---------------------------------------------------------------------------

# Wheel radius [m]
R_WHEEL: float = _VP["wheel_diameter_m"] / 2.0          # 0.288 m

# Effective inertia factor  (1 + rotating_mass_pct/100)
INERTIA_FACTOR: float = 1.0 + _VP["rotating_mass_pct"] / 100.0   # 1.05

# Rolling resistance force at v > 0  [N]
#   F_roll = m * g * mu     (constant, switched on/off by speed)
F_ROLL_CONST: float = (
    _VP["mass_kg"]
    * _VP["gravity_ms2"]
    * _VP["rolling_friction"]
)                                                        # 142.24 N

# Aerodynamic drag prefactor  0.5 * rho * cw * A_f
AERO_FACTOR: float = (
    0.5
    * _VP["air_density_kgm3"]
    * _VP["drag_coeff"]
    * _VP["frontal_area_m2"]
)                                                        # 0.4278  N/(m/s)^2


# ===========================================================================
# BLOCK 1 — Vehicle
# Simulink subsystems reproduced:
#   Average speed       → v_a = 0.5 * (v[t] + v[t-1])
#   Rolling resistance  → F_roll  (switch: 0 at standstill)
#   Aerodynamic resist. → F_aero  = 0.5*rho*cw*A_f * v_a^2
#   Inertia             → F_iner  = m*(1+r_rot)*dv
#   Total resistance    → F_total = F_roll + F_aero + F_iner
#   Wheel radius gain   → T_wheel = F_total * r_wheel
#   Speed to omega      → w_wheel = v / r_wheel
#   Accel to alpha      → dw_wheel = dv / r_wheel
# ===========================================================================

class VehicleDynamics:
    """
    Stateful wrapper around the Vehicle Simulink subsystem.

    State
    -----
    v_prev : float
        Speed at the previous timestep [m/s].
        Required to compute v_a = 0.5*(v[t] + v[t-1])  (Average speed block).

    Usage
    -----
        veh = VehicleDynamics()
        veh.reset()
        out = veh.step(v=0.0, dv=0.0)   # returns dict of all outputs
    """

    def __init__(self) -> None:
        self.v_prev: float = 0.0

    def reset(self) -> None:
        """Call at the start of every episode (mirrors Simulink simulation reset)."""
        self.v_prev = 0.0

    def step(self, v: float, dv: float) -> dict:
        """
        Compute one timestep of vehicle dynamics.

        Parameters
        ----------
        v  : float  Target speed        [m/s]   — Driving Cycle Port 1
        dv : float  Target acceleration [m/s^2] — Driving Cycle Port 2

        Returns
        -------
        dict with keys matching Simulink output port names:
            w_wheel   [rad/s]   Port 1 → Gearbox
            dw_wheel  [rad/s^2] Port 2 → Gearbox
            T_wheel   [Nm]      Port 3 → Gearbox
            P_wheel   [W]       diagnostic
            P_roll    [W]       diagnostic
            P_aero    [W]       diagnostic
            P_iner    [W]       diagnostic
            v_a       [m/s]     diagnostic (average speed used internally)
        """

        # ── Average speed subsystem (Image 2) ────────────────────────────────
        # v_a = 0.5 * (v[t] + v[t-1])
        v_a: float = 0.5 * (v + self.v_prev)

        # ── Rolling resistance subsystem (Image 3) ───────────────────────────
        # F_roll = m*g*mu  if v_a > 0  else  0
        # Uses a switch block: output = F_roll_const when v > 0, else 0
        F_roll: float = F_ROLL_CONST if v_a > 0.0 else 0.0

        # ── Aerodynamic resistance subsystem (Image 5) ───────────────────────
        # F_aero = 0.5 * rho * cw * A_f * v_a^2
        F_aero: float = AERO_FACTOR * v_a ** 2

        # ── Inertia subsystem (Image 4) ──────────────────────────────────────
        # F_iner = m * (1 + rotating_mass_pct/100) * dv
        F_iner: float = _VP["mass_kg"] * INERTIA_FACTOR * dv

        # ── Total resistance (sum block in Image 1) ──────────────────────────
        F_total: float = F_roll + F_aero + F_iner

        # w_wheel  = v_a / r_wheel        (top gain block, Image 1 -- wired
        #                                   from "Average speed" output v_a,
        #                                   NOT raw v)
        # dw_wheel = dv  / r_wheel        (second gain block, Image 1 --
        #                                   wired from raw dv input)
        # T_wheel  = F_total * r_wheel    (Wheel radius gain block, Image 1)
        w_wheel: float = v_a / R_WHEEL
        dw_wheel: float = dv / R_WHEEL
        T_wheel: float = F_total * R_WHEEL

        # ── Diagnostic power signals (Image 1 — multiplier outputs) ──────────
        # P = F * v_a   (force × average speed = power)
        P_wheel: float = F_total * v_a
        P_roll:  float = F_roll  * v_a
        P_aero:  float = F_aero  * v_a
        P_iner:  float = F_iner  * v_a

        # ── Update state ─────────────────────────────────────────────────────
        self.v_prev = v

        return {
            "w_wheel" : w_wheel,    # → Gearbox Port 1
            "dw_wheel": dw_wheel,   # → Gearbox Port 2
            "T_wheel" : T_wheel,    # → Gearbox Port 3
            "P_wheel" : P_wheel,    # diagnostic
            "P_roll"  : P_roll,     # diagnostic
            "P_aero"  : P_aero,     # diagnostic
            "P_iner"  : P_iner,     # diagnostic
            "v_a"     : v_a,        # diagnostic
            "F_roll"  : F_roll,     # diagnostic
            "F_aero"  : F_aero,     # diagnostic
            "F_iner"  : F_iner,     # diagnostic
        }


# ---------------------------------------------------------------------------
# Module-level convenience function (for users who manage state themselves)
# ---------------------------------------------------------------------------

def vehicle_dynamics(
    v: float,
    dv: float,
    v_prev: float = 0.0,
) -> Tuple[dict, float]:
    """
    Stateless version of VehicleDynamics.step().
    Returns (outputs_dict, v_now) so the caller can store v_now as v_prev
    for the next call.

    Example
    -------
        v_prev = 0.0
        for obs in cycle:
            out, v_prev = vehicle_dynamics(obs["v"], obs["dv"], v_prev)
    """
    veh = VehicleDynamics()
    veh.v_prev = v_prev
    out = veh.step(v, dv)
    return out, v


# ===========================================================================
# BLOCK 2 — Manual Gear Box
#
# Simulink subsystems reproduced:
#   Manual Gear Box (MATLAB fn) → i_gt = gear_ratio[i] * i_diff
#   Switch                      → protect against neutral (gear=0)
#   w_MGB  = w_wheel  * i_gt
#   dw_MGB = dw_wheel * i_gt
#   Power flow subsystem:
#     Engine->Wheel (driving)  → T_MGB+ = (T_wheel + P_idle/w_wheel) / eta
#     Wheel->Engine (braking)  → T_MGB- = (T_wheel + eta*P_idle/w_wheel) * eta
#     Switch selects T_MGB+ when T_wheel >= 0, else T_MGB-
#   P_MGB  = T_MGB * w_MGB  (diagnostic)
# ===========================================================================

# Module-level gearbox constants (loaded once from params.json)
_GEAR_RATIOS: list  = _GP["gear_ratios"]       # [3.27, 1.92, 1.26, 0.88, 0.70]
_DIFF_RATIO:  float = _GP["diff_ratio"]         # 3.29
_ETA:         float = _GP["efficiency"]         # 0.98
_P_FRICTION:  float = _GP["idle_friction_W"]    # 100.0 W  (gearbox friction loss)
_W_MIN:       float = _GP["w_min_loss_rads"]    # 1.0 rad/s


def gearbox(
    w_wheel:  float,
    dw_wheel: float,
    t_wheel:  float,
    gear:     int,
) -> dict:
    """
    Manual Gear Box block — translates wheel-side signals to flywheel side.

    Mirrors the Simulink block exactly, including:
      - gear ratio lookup  (i_1..i_5 × i_diff)
      - neutral protection (gear=0 → i_gt treated as inf, outputs zeroed)
      - asymmetric efficiency model in the Power flow subsystem:
          driving  (t_wheel >= 0): friction loss ADDED   before dividing by eta
          braking  (t_wheel <  0): friction loss weighted by eta before multiply

    Parameters
    ----------
    w_wheel  : float  Wheel angular speed        [rad/s]  — from Vehicle Port 1
    dw_wheel : float  Wheel angular acceleration [rad/s²] — from Vehicle Port 2
    t_wheel  : float  Wheel torque               [Nm]     — from Vehicle Port 3
    gear     : int    Gear index  0=neutral, 1-5=gear     — from Driving Cycle

    Returns
    -------
    dict with keys matching Simulink output port names:
        w_mgb   [rad/s]   Port 1 → Control Unit
        dw_mgb  [rad/s²]  Port 2 → Control Unit
        t_mgb   [Nm]      Port 3 → Control Unit
        p_mgb   [W]       diagnostic
        i_gt    [-]       diagnostic (total gear ratio used)
    """

    # ── Manual Gear Box MATLAB function (Image 1) ─────────────────────────
    # i_gt = gear_ratio[gear] * i_diff   when gear in {1..5}
    # gear = 0 means neutral → i_gt = inf (no mechanical connection)
    if gear == 0 or gear > len(_GEAR_RATIOS):
        # Neutral — no torque or speed transmitted
        return {
            "w_mgb" : 0.0,
            "dw_mgb": 0.0,
            "t_mgb" : 0.0,
            "p_mgb" : 0.0,
            "i_gt"  : 0.0,
        }

    i_gt: float = _GEAR_RATIOS[gear - 1] * _DIFF_RATIO

    # ── Speed and acceleration (gain blocks, Image 1) ─────────────────────
    # w_MGB  = w_wheel  × i_gt
    # dw_MGB = dw_wheel × i_gt
    w_mgb:  float = w_wheel  * i_gt
    dw_mgb: float = dw_wheel * i_gt

    # ── Power flow subsystem (Images 2, 3, 4) ─────────────────────────────
    # Friction loss term: P_idle / w_wheel
    # Protected against division by zero (switch with inf in Simulink)
    w_protected: float = abs(w_wheel) if abs(w_wheel) > _W_MIN else _W_MIN
    friction_term: float = _P_FRICTION / w_protected

    if t_wheel >= 0.0:
        t_mgb: float = (t_wheel + friction_term) / _ETA / i_gt
    else:
        t_mgb = (t_wheel + _ETA * friction_term) * _ETA / i_gt

    # ── Diagnostic power (multiplier block, Image 1) ──────────────────────
    p_mgb: float = t_mgb * w_mgb

    return {
        "w_mgb" : w_mgb,
        "dw_mgb": dw_mgb,
        "t_mgb" : t_mgb,
        "p_mgb" : p_mgb,
        "i_gt"  : i_gt,
    }


# ===========================================================================
# BLOCK 3 — Combustion Engine
#
# Simulink subsystems reproduced (Images 1-4):
#
#   Image 1 — Top level:
#     Engine inertia  → T_inertia = theta_CE * dw_gear
#     Total torque    → T_total   = T_gear + T_inertia
#     Lower limit (speed at idle)   → w_CE = max(w_gear, w_idle)
#     Lower limit (fuel cutoff)     → T_CE = max(T_total, T_cutoff)
#     Consumption map → V_dot = interp2D(w_CE, p_me)  [kg/s]
#     Fuel power      → P_CE_fuel = V_dot * H_u        [W]
#     Detect idle     → replaces P_CE_fuel with P_idle when idling
#     Detect cutoff   → sets P_CE = 0 when T_CE <= T_cutoff
#     P_aux added     → P_CE = P_CE_fuel + P_aux
#
#   Image 2 — Detect overload and overspeed:
#     Max torque lookup: T_max = interp1D(w_CE, T_CE_max_curve)
#     Overload  flag: |T_CE| > T_max(w_CE)
#     Overspeed flag: w_CE  > w_CE_upper
#
#   Image 3 — Detect idle:
#     If w_CE <= w_idle OR T_CE <= 0 → use P_CE_idle, else use P_CE_fuel
#
#   Image 4 — Detect fuel cutoff:
#     If T_CE <= T_cutoff AND ENABLE_FUEL_CUTOFF → P_CE = P_cutoff (=0)
#     else P_CE = P_CE_fuel
#
# Map axes:
#   V_CE_map (15×16): rows = w_CE_row [rad/s], cols = p_me_col [Pa]
#   p_me = T_CE * 4*pi / V_d   (mean effective pressure, 4-stroke)
#   V_d  = 1698e-6 m³
#   H_u  = 42 936 780 J/kg
# ===========================================================================

# ── Load engine map data (once at import time) ────────────────────────────
_MAP_PATH = _find_project_root() / "data" / "maps" / "engine_maps.npz"
_maps     = np.load(str(_MAP_PATH))

# Speed and pressure grids
_w_CE_row = _maps["w_CE_row"]  # (31,) rad/s
_T_CE_col = _maps["T_CE_col"]  # (28,) Nm
_V_CE_map = _maps["V_CE_map"]  # (31,28) kg/s  rows=speed, cols=torque
_w_CE_max_fine = _maps["w_CE_max_fine"]  # (31,) rad/s  — grid for T_CE_max
_T_CE_max      = _maps["T_CE_max"]       # (31,) Nm     — max torque curve

# Scalar engine constants
_H_u     = float(_maps["H_u"][0])        # 42 936 780 J/kg
_V_d     = 1698e-6                        # m³  engine displacement (1698 cm³)
_W_IDLE  = float(_P["engine"]["idle_speed_rads"])   # 105 rad/s
_P_CE_IDLE = float(_P["engine"]["idle_power_W"])     # 8 000 W  (engine idle power)
_THETA   = float(_P["engine"]["inertia_kgm2"])       # 0.2  kg·m²
_T_CUTOFF = float(_P["engine"]["fuel_cutoff_Nm"])    # 5    Nm
_P_CUTOFF = float(_P["engine"]["fuel_cutoff_W"])     # 0    W
_P_AUX   = float(_P["engine"]["aux_power_W"])        # 0    W
_W_UPPER = float(_w_CE_row.max())                    # 850  rad/s


def _interp1d_linear(x_grid: np.ndarray, y_grid: np.ndarray, x: float) -> float:
    """
    1-D linear interpolation with extrapolation — matches Simulink
    'Interpolation-Extrapolation' lookup method.
    """
    return float(np.interp(x, x_grid, y_grid))


def _interp2d_linear(
    row_grid: np.ndarray,
    col_grid: np.ndarray,
    table:    np.ndarray,
    row_val:  float,
    col_val:  float,
) -> float:
    """
    2-D bilinear interpolation with extrapolation — matches Simulink
    Lookup2D 'Interpolation-Extrapolation' method.

    table shape: (len(row_grid), len(col_grid))
    """
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (row_grid, col_grid),
        table,
        method      = "linear",
        bounds_error= False,
        fill_value  = None,    # extrapolate outside bounds
    )
    return float(interp(np.array([[row_val, col_val]]))[0])


def combustion_engine(
    w_gear: float,
    dw_gear: float,
    t_gear: float,
) -> dict:
    """
    Combustion Engine block — computes fuel power P_CE from demanded
    speed, acceleration, and torque.

    Mirrors the Simulink block exactly including:
      - Engine inertia contribution to total torque
      - Idle speed lower limit
      - Fuel cutoff lower limit on torque
      - 2-D consumption map lookup (speed × mean effective pressure)
      - Idle detection (switches to constant idle power)
      - Fuel cutoff detection (sets power to zero)
      - Overload and overspeed flags (logged, not hard-stop)

    Parameters
    ----------
    w_gear  : float  Flywheel angular speed        [rad/s] — from Control Unit
    dw_gear : float  Flywheel angular acceleration [rad/s²]— from Control Unit
    t_gear  : float  Torque demand on engine       [Nm]    — T_CE from Control Unit

    Returns
    -------
    dict with keys:
        p_ce         [W]   Total fuel power output (main output → Tank)
        p_ce_fuel    [W]   Raw map fuel power (before idle/cutoff logic)
        w_ce         [rad/s]  Actual engine speed (after idle clamp)
        t_ce         [Nm]     Actual engine torque (after cutoff clamp)
        v_dot        [kg/s]   Fuel mass flow rate
        overload     [bool]   True if torque exceeds max torque curve
        overspeed    [bool]   True if speed exceeds upper limit
        is_idle      [bool]   True if engine is idling
        is_cutoff    [bool]   True if fuel cutoff is active
    """

    # ── Engine inertia (Image 1) ──────────────────────────────────────────
    # T_inertia = theta_CE * dw_gear
    t_inertia: float = _THETA * dw_gear

    # ── Total torque demand (Image 1) ─────────────────────────────────────
    t_total: float = t_gear + t_inertia

    # ── Lower limit: speed at idle (Image 1) ─────────────────────────────
    # w_CE = max(w_gear, w_idle)
    w_ce: float = max(w_gear, _W_IDLE)

    # ── Lower limit: fuel cutoff torque (Image 1) ─────────────────────────
    # T_CE = max(T_total, T_cutoff)
    t_ce: float = max(t_total, _T_CUTOFF)

    # ── Consumption map lookup: V_dot = f(w_CE, T_CE)  [kg/s] ─────────────
    # The Simulink Lookup2D is indexed by SPEED (w_CE_row) and TORQUE
    # (T_CE_col) directly -- no mean-effective-pressure conversion.
    # Lookup method is Interpolation-Extrapolation in Simulink; here we
    # clamp to grid bounds to avoid wild extrapolation below w_idle etc.
    w_ce_clamped: float = float(np.clip(w_ce, _w_CE_row.min(), _w_CE_row.max()))
    t_ce_clamped: float = float(np.clip(t_ce, _T_CE_col.min(), _T_CE_col.max()))

    v_dot: float = _interp2d_linear(
        _w_CE_row, _T_CE_col, _V_CE_map,
        w_ce_clamped, t_ce_clamped,
    )
    v_dot = max(v_dot, 0.0)  # fuel flow cannot be negative

    # ── Fuel power (Image 1) ─────────────────────────────────────────────
    # P_CE_fuel = V_dot * H_u   [W]
    p_ce_fuel: float = v_dot * _H_u

    # ── Detect overload and overspeed (Image 2) ───────────────────────────
    t_max_at_w: float = _interp1d_linear(_w_CE_max_fine, _T_CE_max, w_ce)
    overload:  bool   = abs(t_ce) > t_max_at_w
    overspeed: bool   = w_ce > _W_UPPER

    # ── Detect idle (Image 3) ─────────────────────────────────────────────
    # Idle condition: engine is at minimum speed and torque is not positive
    is_idle: bool = (0.0 < w_gear) and (w_gear <= _W_IDLE)
    if is_idle:
        p_ce_fuel = _P_CE_IDLE

    # ── Detect fuel cutoff (Image 4) ──────────────────────────────────────
    # Cutoff: T_CE <= T_cutoff AND ENABLE_FUEL_CUTOFF is checked
    is_cutoff: bool = (t_ce <= _T_CUTOFF)
    if is_cutoff and not is_idle:
        p_ce_fuel = _P_CUTOFF   # = 0 W

    # ── Add auxiliary power (Image 1) ─────────────────────────────────────
    p_ce: float = p_ce_fuel + _P_AUX

    return {
        "p_ce"     : p_ce,       # → Tank block (main output)
        "p_ce_fuel": p_ce_fuel,  # diagnostic
        "w_ce"     : w_ce,       # diagnostic
        "t_ce"     : t_ce,       # diagnostic
        "v_dot"    : v_dot,      # diagnostic [kg/s]
        "overload" : overload,   # diagnostic flag
        "overspeed": overspeed,  # diagnostic flag
        "is_idle"  : is_idle,    # diagnostic flag
        "is_cutoff": is_cutoff,  # diagnostic flag
    }


# ===========================================================================
# BLOCK 4 — Electric Motor
#
# Simulink subsystems reproduced (from shared screenshots):
#
#   Image 1 — Top level:
#     Motor inertia   → T_inertia = theta_EM * dw_gear
#     Total torque    → T_EM      = T_gear + T_inertia
#     w_EM            = w_gear   (direct wire — no idle speed clamp)
#     Efficiency map  → eta_factor = f(w_EM, T_EM)   [2-D lookup on SIGNED torque]
#     Power calc      → P_EM = T_EM * w_EM * eta_factor
#     P_aux added     → P_EM_total = P_EM + P_aux
#
#   Image 2 — Maximum torque curve (from motor_maps.mat):
#     w_EM_max [rad/s]: [  0, 100, 200, 300, 400, 500, 600, 700, 800]
#     T_EM_max [Nm]   : [ 60,  60,  60,  44.5714, 32.5714, 24, 16.2857, 11.1429, 7.7143]
#
#   Image 3 — Detect overload and overspeed:
#     Overload  check: T_max(w_EM) − |T_EM| < 0 → flag
#     Overspeed check: w_EM_upper  − w_EM   < 0 → flag
#
# Map structure (motor_maps.mat → motor_maps.npz):
# ─────────────────────────────────────────────────
#   The efficiency map uses a SIGNED torque axis (T_EM_col, 30 values from
#   −74.73 to +74.73 Nm) and stores a combined eta_factor:
#
#     T < 0  (generating / regen braking):
#       eta_factor = η  ∈ (0, 1)
#       P_EM = T_EM * w_EM * η          → |P_elec| < |P_mech|  (losses reduce return)
#
#     T > 0  (motoring):
#       eta_factor = 1/η  > 1
#       P_EM = T_EM * w_EM * (1/η)      → P_elec > P_mech  (losses increase draw)
#
#   Single unified formula for both modes:
#       P_EM = T_EM × w_EM × eta_map(w_EM, T_EM)
#
#   Sign convention (matches Battery block):
#       P_EM > 0  → motor draws power → battery discharges
#       P_EM < 0  → motor returns power → battery charges
#
#   Sentinel values in the raw map:
#       0.0  — zero cells on generating side at low speeds (T beyond T_max):
#              filled with edge row value before saving .npz
#       10.0 — on motoring side at T beyond T_max (1/η → ∞ → clipped to 10):
#              overload flag is set before these cells are ever used
#
# ===========================================================================

# ── Load motor map data (once at import time) ─────────────────────────────
_MOTOR_MAP_PATH = _find_project_root() / "data" / "maps" / "motor_maps.npz"
_motor_maps     = np.load(str(_MOTOR_MAP_PATH))

# Efficiency map — rows = speed, cols = signed torque
_w_EM_row   = _motor_maps["w_EM_row"].astype(np.float64)    # (9,)  rad/s
_T_EM_col   = _motor_maps["T_EM_col"].astype(np.float64)    # (30,) Nm  (signed)
_eta_EM_map = _motor_maps["eta_EM_map"].astype(np.float64)  # (9,30)

# Maximum torque curve (from Image 2 / motor_maps.mat)
_w_EM_max_row = _motor_maps["w_EM_max_row"].astype(np.float64)  # (9,) rad/s
_T_EM_max_arr = _motor_maps["T_EM_max"].astype(np.float64)      # (9,) Nm

# Scalar motor constants
_THETA_EM   = float(_P["motor"]["inertia_kgm2"])   # 0.1 kg·m²
_P_AUX_EM   = float(_P["motor"]["aux_power_W"])    # 0.0 W
_W_EM_UPPER = float(_motor_maps["w_EM_upper"][0])  # 600 rad/s


def electric_motor(
    w_gear:  float,
    dw_gear: float,
    t_gear:  float,
) -> dict:
    """
    Electric Motor block — computes electrical power P_EM from commanded
    speed, acceleration, and torque.

    Mirrors the Simulink block exactly (three provided screenshots) using
    the combined signed-torque efficiency map from motor_maps.npz.

    Unified formula (no if/else on torque sign):
        P_EM = T_EM × w_EM × eta_map(w_EM, T_EM)
    where the map stores:
        η       for T < 0  (generating/regen)  →  |P_elec| < |P_mech|
        1/η     for T > 0  (motoring)          →   P_elec  > P_mech

    Parameters
    ----------
    w_gear  : float  Flywheel angular speed        [rad/s]  — from Control Unit
    dw_gear : float  Flywheel angular acceleration [rad/s²] — from Control Unit
    t_gear  : float  Torque command to motor       [Nm]     — T_EM from Control Unit
                     Positive = motoring  (propels vehicle, draws battery power)
                     Negative = generating (regen braking, returns power to battery)

    Returns
    -------
    dict with keys matching Simulink wire / signal names:
        p_em          [W]     Electrical power → Battery block (main output)
                              Positive = battery discharging (motoring)
                              Negative = battery charging    (generating)
        p_mech        [W]     Mechanical shaft power T_EM × w_EM  (diagnostic)
        w_em          [rad/s] Motor speed  (= w_gear, no idle clamp)
        t_em          [Nm]    Motor torque after inertia addition
        eta_factor    [-]     Raw map value at operating point (η or 1/η)
        overload      [bool]  True if |T_EM| > T_EM_max(w_EM)
        overspeed     [bool]  True if w_EM > w_EM_upper
        is_motoring   [bool]  True when t_em > 0
        is_generating [bool]  True when t_em ≤ 0
    """

    # ── Motor inertia (Image 1 — theta_EM gain block) ─────────────────────
    # T_inertia = theta_EM × dw_gear
    t_inertia: float = _THETA_EM * dw_gear

    # ── Total torque (Image 1 — sum block) ────────────────────────────────
    # T_EM = T_gear + T_inertia
    t_em: float = t_gear + t_inertia

    # ── Motor speed (Image 1) ─────────────────────────────────────────────
    # w_EM = w_gear  (direct wire, unlike engine there is no lower-limit clamp)
    w_em: float = w_gear

    # ── Mechanical shaft power ────────────────────────────────────────────
    # P_mech = T_EM × w_EM
    p_mech: float = t_em * w_em

    # ── Detect overload and overspeed (Image 3) ───────────────────────────
    t_max_at_w: float = _interp1d_linear(_w_EM_max_row, _T_EM_max_arr, abs(w_em))
    overload:  bool   = abs(t_em) > t_max_at_w
    overspeed: bool   = abs(w_em) > _W_EM_UPPER

    # ── Efficiency map lookup (Image 1 — "Efficiency eta = f(w,T)" block) ─
    #
    # The map is indexed by (w_EM, T_EM) — both signed, but w is always ≥ 0
    # in normal forward driving.  Clamp to map boundaries for interpolation.
    #
    # The map stores:
    #   η       in the generating half (T_EM_col < 0) — values 0.57–0.90
    #   1/η     in the motoring half   (T_EM_col > 0) — values 1.11–1.75
    #
    # This means P_EM = P_mech × eta_factor works as a single expression:
    #   motoring  : P_mech > 0, eta_factor > 1  → P_EM > P_mech  (more drawn from battery)
    #   generating: P_mech < 0, eta_factor < 1  → |P_EM| < |P_mech| (less returned)
    w_clamped: float = float(np.clip(abs(w_em), _w_EM_row.min(), _w_EM_row.max()))
    t_clamped: float = float(np.clip(t_em,      _T_EM_col.min(), _T_EM_col.max()))

    eta_factor: float = _interp2d_linear(
        _w_EM_row, _T_EM_col, _eta_EM_map,
        w_clamped, t_clamped,
    )
    # Guard: sentinel 10.0 only appears in overload territory; clamp to 10
    # Guard: negative or zero should never occur after zero-fill, but defend anyway
    eta_factor = float(np.clip(eta_factor, 1e-4, 10.0))

    # ── Electrical power (Image 1 — multiplier and sum block) ─────────────
    # P_EM = T_EM × w_EM × eta_factor
    p_em: float = p_mech * eta_factor

    # ── Add auxiliary power (Image 1 — P_aux constant block) ──────────────
    p_em_total: float = p_em + _P_AUX_EM

    # ── Mode flags (diagnostic) ───────────────────────────────────────────
    is_motoring:   bool = t_em > 0.0
    is_generating: bool = not is_motoring

    return {
        "p_em"         : p_em_total,    # → Battery block (main output)
        "p_mech"       : p_mech,        # diagnostic [W]
        "w_em"         : w_em,          # diagnostic [rad/s]
        "t_em"         : t_em,          # diagnostic [Nm]
        "eta_factor"   : eta_factor,    # diagnostic (η or 1/η from map)
        "overload"     : overload,      # diagnostic flag
        "overspeed"    : overspeed,     # diagnostic flag
        "is_motoring"  : is_motoring,   # diagnostic flag
        "is_generating": is_generating, # diagnostic flag
    }

# ===========================================================================
# BLOCK 5 — Tank
#
# Simulink subsystems reproduced (Images 1 and 2):
#
#   Image 2 — Integration subsystem (inside Tank):
#     Trapezoidal integration of P_fuel to get accumulated fuel mass m_fuel.
#
#     Signal chain step by step:
#       1. P_fuel[t]  enters at Port 1
#       2. 1/z  (unit delay, sample time h=1 s) → P_fuel[t-1]
#       3. Sum  → P_fuel[t] + P_fuel[t-1]
#       4. Gain 0.5*h  → (P_fuel[t] + P_fuel[t-1]) * 0.5 * h
#                        = trapezoidal area under P_fuel curve [J/step]
#       5. Gain 1/H_u  → energy / H_u  = m_dot_fuel [kg/step]  (diagnostic)
#                        multiplier uses BOTH outputs: 1/H_u constant × step 4
#       6. Sum  → m_dot_fuel + m_fuel[t-1]
#       7. 1/z  accumulator → m_fuel[t]   (state, fed back into step 6)
#       Output: m_fuel [kg]  (total accumulated fuel mass)
#
#     This is standard trapezoidal (bilinear) integration:
#       m_fuel[t] = m_fuel[t-1] + (P_fuel[t] + P_fuel[t-1]) / (2 * H_u) * h
#
#   Image 1 — Tank top-level block:
#     Signal chain step by step:
#       1. m_fuel [kg]  from Integration subsystem
#       2. Gain 1/rho_f  → volume  = m_fuel / rho_f   [L]
#       3. ÷ x_tot [m]   → volume per metre           [L/m]
#       4. Gain k_cs     → cold-start correction       [L/m]
#       5. Gain 1e5      → convert to [L / 100 km]
#                          (1e5 m = 100 km, so [L/m] × 1e5 = [L/100km])
#       Output Port 1:   V_liter  [L/100km]   → Display + EFC block
#
# Constants:
#   H_u    = 42 936 779.9 J/kg   (diesel lower heating value — shared with engine)
#   rho_f  = 0.843 kg/L          (diesel density, QSS toolbox standard)
#   k_cs   = 1.0                 (cold-start factor for diesel; petrol would be > 1)
#   h      = 1.0 s               (simulation timestep — matches DrivingCycle dt)
#
# File placement decision:
#   Stays in powertrain.py, NOT a separate file.
#   Rationale: the project folder structure document lists powertrain.py as
#   "component functions (vehicle, gearbox, engine, motor, battery)".
#   Tank is the next component in that same list.  Splitting it to tank.py
#   would create a 30-line file and break the import symmetry used throughout
#   the test suite (all tests import from env.powertrain).
# ===========================================================================

# ── Tank constants (loaded once at import time) ───────────────────────────
# H_u is already defined above as _H_u from the engine map file.
# We reuse it here to keep a single source of truth.
_RHO_FUEL: float = 0.843          # kg/L   diesel density (QSS toolbox default)
_K_CS:     float = 1.15           # [-]    cold-start correction factor (confirmed from
                                   #        Simulink "Factor for cold start" Gain block)
_H_TANK:   float = 1.0            # s      simulation timestep (matches DrivingCycle dt)
_M_TO_100KM: float = 1e5          # m/100km  unit conversion constant


class Tank:
    """
    Stateful Tank block — integrates P_fuel to produce fuel consumption
    in liter/100 km.

    Mirrors the Simulink block exactly, including:
      - Trapezoidal (bilinear) integration of P_fuel  (Image 2)
      - Division by x_tot to get volumetric rate      (Image 1)
      - Cold-start correction factor k_cs             (Image 1)
      - Unit conversion ×1e5  [L/m → L/100km]        (Image 1)

    State
    -----
    m_fuel     : float   Accumulated fuel mass [kg].
                         Corresponds to the 1/z accumulator in Image 2.
    p_fuel_prev: float   P_fuel from the previous timestep [W].
                         Corresponds to the unit delay 1/z in Image 2.

    Usage
    -----
        tank = Tank()
        tank.reset()
        out = tank.step(p_fuel=8000.0, x_tot=150.0)
        # out["v_liter"]   → L/100km
        # out["m_fuel"]    → accumulated kg
        # out["m_dot_fuel"]→ kg/step (this step's trapezoidal contribution)
    """

    def __init__(self) -> None:
        self.m_fuel:      float = 0.0
        self.p_fuel_prev: float = 0.0

    def reset(self) -> None:
        """Reset to simulation start.  Call at the beginning of every episode."""
        self.m_fuel      = 0.0
        self.p_fuel_prev = 0.0

    def step(self, p_fuel: float, x_tot: float) -> dict:
        """
        Advance one timestep.

        Parameters
        ----------
        p_fuel : float  Fuel power from the Combustion Engine block  [W]
                        = P_CE output  (= v_dot * H_u from combustion_engine())
        x_tot  : float  Total distance traveled so far               [m]
                        = cumulative distance from the Driving Cycle

        Returns
        -------
        dict with keys:
            v_liter    [L/100km]  Fuel consumption rate (main output → Display)
            m_fuel     [kg]       Accumulated fuel mass (state, diagnostic)
            m_dot_fuel [kg/step]  Trapezoidal fuel mass added this step (diagnostic)
        """

        # ── Integration subsystem (Image 2) ──────────────────────────────
        #
        # Step 3–5: trapezoidal area under P_fuel curve, converted to kg
        #   m_dot_fuel = (P_fuel[t] + P_fuel[t-1]) * 0.5 * h / H_u
        #
        # Using _H_u from the engine block (shared constant, same J/kg value).
        m_dot_fuel: float = (
            (p_fuel + self.p_fuel_prev) * 0.5 * _H_TANK / _H_u
        )

        # Step 6–7: accumulate
        #   m_fuel[t] = m_fuel[t-1] + m_dot_fuel
        self.m_fuel += m_dot_fuel

        # ── Top-level Tank block (Image 1) ────────────────────────────────
        #
        # Convert accumulated mass → volume → L/100km
        #   volume  [L]       = m_fuel / rho_f
        #   rate    [L/m]     = volume / x_tot        (guard against x_tot = 0)
        #   v_liter [L/100km] = rate * k_cs * 1e5
        if x_tot > 0.0:
            v_liter: float = (
                (self.m_fuel / _RHO_FUEL)   # [L]
                / x_tot                      # [L/m]
                * _K_CS                      # cold-start correction [-]
                * _M_TO_100KM               # [L/100km]
            )
        else:
            # Before the vehicle has moved, consumption is undefined.
            # Output 0 to match Simulink's behaviour (no division by zero).
            v_liter = 0.0

        # ── Update state ──────────────────────────────────────────────────
        self.p_fuel_prev = p_fuel

        return {
            "v_liter"    : v_liter,      # → Port 1 output  [L/100km]
            "m_fuel"     : self.m_fuel,  # diagnostic [kg]
            "m_dot_fuel" : m_dot_fuel,   # diagnostic [kg/step]
        }


# ---------------------------------------------------------------------------
# Module-level convenience function (stateless, for users who carry state)
# ---------------------------------------------------------------------------

def tank_step(
    p_fuel:      float,
    x_tot:       float,
    m_fuel_prev: float = 0.0,
    p_fuel_prev: float = 0.0,
) -> tuple:
    """
    Stateless version of Tank.step().

    Returns
    -------
    (outputs_dict, m_fuel_new, p_fuel_new)
      Call the next step with m_fuel_prev=m_fuel_new, p_fuel_prev=p_fuel_new.

    Example
    -------
        m_fuel = p_fuel_p = 0.0
        for obs in cycle:
            out, m_fuel, p_fuel_p = tank_step(
                p_fuel=eng["p_ce"], x_tot=obs["x_tot"],
                m_fuel_prev=m_fuel, p_fuel_prev=p_fuel_p
            )
    """
    tank = Tank()
    tank.m_fuel      = m_fuel_prev
    tank.p_fuel_prev = p_fuel_prev
    out = tank.step(p_fuel, x_tot)
    return out, tank.m_fuel, p_fuel


# ===========================================================================
# BLOCK 6 — Battery
#
# Simulink subsystems reproduced (Image: Battery top-level, plus
# Battery voltage -> Charging / Discharging / Idle / Supervision battery):
#
#   "Battery charge" subsystem (1/z integrator):
#     Q_BT[k] = Q_BT[k-1] - h * I_BT[k-1]
#     h = 1.0 s  (simulation timestep)
#
#   Battery voltage (mode switch on sign of P_BT):
#     P_BT < 0 (charging):    U_BT = U_BT_L  (Charging subsystem)
#     P_BT > 0 (discharging): U_BT = U_BT_E  (Discharging subsystem)
#     P_BT == 0 (idle):       U_BT = U_BT_E  (Idle subsystem)
#
#   Charging   (Fcn block, exact MATLAB expression from model):
#     u1 = Q_BT/Q_BT_0  (SoC),  u2 = P_BT/I_0
#     a  = c_BT_L3*u1 + c_BT_L1
#     b  = c_BT_L4*u1 + c_BT_L2
#     U_BT_L = (a + sqrt(a^2 + 4*b*(-u2))) / 2
#
#   Discharging (Fcn block, exact MATLAB expression from model):
#     u1 = Q_BT/Q_BT_0  (SoC),  u2 = P_BT/I_0
#     a  = c_BT_E3*u1 + c_BT_E1
#     b  = c_BT_E4*u1 + c_BT_E2
#     U_BT_E = (a + sqrt(a^2 + 4*b*(-u2))) / 2
#
#   Idle (Gain c_BT_L3/Q_BT_0 on Q_BT, summed '++' with Constant c_BT_L1):
#     U_BT_E = (c_BT_L3/Q_BT_0)*Q_BT + c_BT_L1
#     (This is exactly the open-circuit voltage curve U_oc(Q_BT), and is
#      also the P_BT=0 limit of both Charging and Discharging formulas —
#      verified numerically to match exactly.)
#
#   Current:
#     I_BT = P_BT / U_BT
#
#   V_BT (energy consumption, top-level "Battery" diagram):
#     E(Q) = 0.5 * U_oc(Q) * Q          [capacitor-analogy stored energy, J]
#     -C-  (Simulink constant block)    = U_oc(Q_BT_IC)   [see note below]
#     V_BT = [0.5*(-C-)*Q_BT_IC - 0.5*f(Q_BT)*Q_BT] / x_tot / 36
#          = [E(Q_BT_IC) - E(Q_BT)] / x_tot / 36
#     The /36 gain is the exact J -> kWh/100km unit conversion:
#        1e5 [m/100km] / 3.6e6 [J/kWh] = 1/36
#
#   NOTE on "-C-": "-C-" is Simulink's GENERIC constant-block icon (it
#   appears for ANY Constant block regardless of value, not a variable
#   named C). Its value was derived as U_oc(Q_BT_IC) = 46.8 V — this
#   makes (-C-)*Q_BT_IC the "initial stored energy" term, paired with
#   f(Q_BT)*Q_BT = U_oc(Q_BT)*Q_BT as the "current stored energy" term.
#   If your f(u) block in Simulink returns something other than
#   U_oc(Q_BT)*Q_BT, update _battery_energy() below accordingly.
#
#   Supervision battery (Stop Simulation triggers when input < 0,
#   confirmed via Sum block port-sign extraction: Sum1='-+', Sum2='+-',
#   Sum3='++'):
#     threshold = (c_BT_E3/Q_BT_0)*Q_BT + c_BT_E1
#     stop_uv = (2*U_BT - threshold) < 0       # undervoltage
#     stop_oc = (I_BT_max - |I_BT|) < 0        # overcurrent
#
# Constants (extracted from Simulink base workspace / mask):
#   Q_BT_0   = capacity_Ah * 3600 = 10 * 3600 = 36000  [As]
#   Q_BT_IC  = 18000  [As]  (= 50% SoC, matches params.json soc_init_pct)
#   I_0      = 10.0   [A]   (from Gain "1/I_0" = 0.1)
#   I_BT_max = 300.0  [A]
#   c_BT_L1 = 39.0,  c_BT_L2 = 0.013,  c_BT_L3 = 15.6,  c_BT_L4 = 0.0
#   c_BT_E1 = 39.0,  c_BT_E2 = 0.013,  c_BT_E3 = 15.6,  c_BT_E4 = 0.0
#   h (battery charge integrator timestep) = 1.0 s
# ===========================================================================

# ── Battery constants (from Simulink base workspace) ──────────────────────
_BP = _P.get("battery", {})

_Q_BT_0:  float = float(_BP.get("capacity_Ah", 10.0)) * 3600.0   # 36000 As
_Q_BT_IC: float = _Q_BT_0 * (float(_BP.get("soc_init_pct", 50.0)) / 100.0)  # 18000 As

_I_0:      float = 10.0    # A   (1/I_0 = 0.1 gain in Charging/Discharging)
_I_BT_MAX: float = 300.0   # A   (Constant7 in Supervision battery)
_H_BATT:   float = 1.0     # s   (Battery charge integrator timestep)

# Charging coefficients (Fcn block, Charging subsystem)
_C_BT_L1: float = 39.0
_C_BT_L2: float = 0.013
_C_BT_L3: float = 15.6
_C_BT_L4: float = 0.0

# Discharging coefficients (Fcn block, Discharging subsystem)
_C_BT_E1: float = 39.0
_C_BT_E2: float = 0.013
_C_BT_E3: float = 15.6
_C_BT_E4: float = 0.0

# "-C-" constant in the V_BT energy path (top-level Battery diagram).
# Derived as U_oc(Q_BT_IC) — see note in the docstring above.
_V_BT_C_CONST: float = (_C_BT_L3 / _Q_BT_0) * _Q_BT_IC + _C_BT_L1   # = 46.8 V


def _u_oc(q_bt: float) -> float:
    """
    Open-circuit voltage U_oc(Q_BT) — the Idle formula from the Idle
    subsystem, also the P_BT=0 limit of Charging/Discharging:
        U_oc(Q) = (c_BT_L3/Q_BT_0) * Q + c_BT_L1
    """
    return (_C_BT_L3 / _Q_BT_0) * q_bt + _C_BT_L1


def _battery_energy(q_bt: float) -> float:
    """
    Stored-energy term E(Q) = 0.5 * U_oc(Q) * Q   [J]
    (capacitor-analogy energy used in the V_BT computation).
    """
    return 0.5 * _u_oc(q_bt) * q_bt


# Pre-computed "initial energy" term: 0.5 * (-C-) * Q_BT_IC
# Equal to _battery_energy(_Q_BT_IC) since _V_BT_C_CONST == _u_oc(_Q_BT_IC).
_E_INITIAL: float = 0.5 * _V_BT_C_CONST * _Q_BT_IC


class Battery:
    """
    Stateful Battery block.

    Mirrors the Simulink "Battery" subsystem exactly:
      - Charge integrator (1/z):  Q_BT[k] = Q_BT[k-1] - h*I_BT[k-1]
      - Battery voltage switch:   Charging / Discharging / Idle
      - V_BT energy-consumption metric (capacitor-analogy, /36 unit gain)
      - Supervision flags: stop_uv (undervoltage), stop_oc (overcurrent)

    State
    -----
    q_bt : float
        Battery charge [As/Coulombs]. Initialised to Q_BT_IC (50% SoC).

    Usage
    -----
        batt = Battery()
        batt.reset()
        out = batt.step(p_bt=2000.0, x_tot=150.0)
        # out["v_bt"]  -> kWh/100km
        # out["q_bt"]  -> current charge [As]
        # out["soc"]   -> state of charge [0,1]
    """

    def __init__(self) -> None:
        self.q_bt: float = _Q_BT_IC

    def reset(self) -> None:
        """Reset to simulation start (50% SoC). Call at the start of every episode."""
        self.q_bt = _Q_BT_IC

    def step(self, p_bt: float, x_tot: float) -> dict:
        """
        Advance one timestep.

        Parameters
        ----------
        p_bt  : float  Electrical power request [W] — P_EM from the Electric
                       Motor block.
                       Positive = discharging (motor draws power)
                       Negative = charging    (motor returns power / regen)
        x_tot : float  Total distance traveled so far [m]

        Returns
        -------
        dict with keys:
            v_bt    [kWh/100km]  Electrical energy consumption (main output)
            q_bt    [As]         Current battery charge (diagnostic, also
                                  fed back to Control Unit as Q_BT)
            u_bt    [V]          Battery terminal voltage (diagnostic)
            i_bt    [A]          Battery current (diagnostic)
            soc     [-]          State of charge, Q_BT/Q_BT_0 in [0,1]
            stop_uv [bool]       True if undervoltage detected
            stop_oc [bool]       True if overcurrent detected
        """

        q_bt_prev: float = self.q_bt
        soc: float = q_bt_prev / _Q_BT_0

        # ── Battery voltage subsystem (mode switch on sign of P_BT) ───────
        if p_bt < 0.0:
            # ── Charging (Fcn block) ───────────────────────────────────────
            u1 = soc
            u2 = p_bt / _I_0
            a  = _C_BT_L3 * u1 + _C_BT_L1
            b  = _C_BT_L4 * u1 + _C_BT_L2
            discriminant = a * a + 4.0 * b * (-u2)
            u_bt: float = (a + np.sqrt(max(discriminant, 0.0))) / 2.0

        elif p_bt > 0.0:
            # ── Discharging (Fcn block) ─────────────────────────────────────
            u1 = soc
            u2 = p_bt / _I_0
            a  = _C_BT_E3 * u1 + _C_BT_E1
            b  = _C_BT_E4 * u1 + _C_BT_E2
            discriminant = a * a + 4.0 * b * (-u2)
            u_bt = (a + np.sqrt(max(discriminant, 0.0))) / 2.0

        else:
            # ── Idle (Gain c_BT_L3/Q_BT_0 summed with Constant c_BT_L1) ─────
            u_bt = _u_oc(q_bt_prev)

        # ── Current: I_BT = P_BT / U_BT ────────────────────────────────────
        u_bt_safe: float = u_bt if abs(u_bt) > 1e-6 else 1e-6
        i_bt: float = p_bt / u_bt_safe

        # ── Battery charge integrator (1/z) ────────────────────────────────
        # Q_BT[k] = Q_BT[k-1] - h * I_BT[k-1]
        self.q_bt = q_bt_prev - _H_BATT * i_bt

        # ── V_BT energy consumption (top-level Battery diagram) ────────────
        # V_BT = [E(Q_BT_IC) - E(Q_BT)] / x_tot / 36
        e_now: float = _battery_energy(self.q_bt)
        if x_tot > 0.0:
            v_bt: float = (_E_INITIAL - e_now) / x_tot / 36.0
        else:
            v_bt = 0.0

        # ── Supervision battery ────────────────────────────────────────────
        # threshold = (c_BT_E3/Q_BT_0)*Q_BT + c_BT_E1
        # stop_uv = (2*U_BT - threshold) < 0   (Stop Simulation fires if <0)
        threshold: float = (_C_BT_E3 / _Q_BT_0) * q_bt_prev + _C_BT_E1
        stop_uv: bool = bool((2.0 * u_bt - threshold) < 0.0)

        # stop_oc = (I_BT_max - |I_BT|) < 0
        stop_oc: bool = bool((_I_BT_MAX - abs(i_bt)) < 0.0)

        return {
            "v_bt"   : v_bt,           # → main output [kWh/100km]
            "q_bt"   : self.q_bt,      # diagnostic / feedback to Control Unit [As]
            "u_bt"   : u_bt,           # diagnostic [V]
            "i_bt"   : i_bt,           # diagnostic [A]
            "soc"    : self.q_bt / _Q_BT_0,  # diagnostic [-]
            "stop_uv": stop_uv,        # diagnostic flag
            "stop_oc": stop_oc,        # diagnostic flag
        }


# ---------------------------------------------------------------------------
# Module-level convenience function (stateless, for users who carry state)
# ---------------------------------------------------------------------------

def battery_step(p_bt: float, x_tot: float, q_bt_prev: float = _Q_BT_IC) -> tuple:
    """
    Stateless version of Battery.step().

    Returns
    -------
    (outputs_dict, q_bt_new)
      Call the next step with q_bt_prev=q_bt_new.
    """
    batt = Battery()
    batt.q_bt = q_bt_prev
    out = batt.step(p_bt, x_tot)
    return out, batt.q_bt


# ===========================================================================
# BLOCK 7 — Equivalent Fuel Consumption
#
# Simulink subsystem (Equivalent Fuel Consumption):
#   gain  = 1 / (eta_BT * eta_EM * eta_CE * (H_u / (1000*3600)) * rho_f)
#   converted_v_bt = V_BT * gain
#   saturated_v_bt = saturate(converted_v_bt, lower, upper)
#   V_CE_equiv = V_CE + saturated_v_bt
#
# Constants (from params.json "equivalent_consumption" section):
#   eta_BT = 0.9   (battery efficiency)
#   eta_EM = 0.8   (motor efficiency)
#   eta_CE = 0.25  (engine efficiency)
#   H_u, rho_f shared with Engine/Tank blocks
#
# Saturation bounds: CONFIRMED from the Simulink Saturation block inside
# the Equivalent Fuel Consumption subsystem — Upper=inf, Lower=0.
#
# Physical meaning: converted_v_bt is clamped to >= 0. When V_BT < 0
# (net battery charging / regen), converted_v_bt would be negative —
# the saturation zeroes this out, so regenerated electrical energy gives
# NO credit toward V_CE_equiv (a conservative, no-credit convention).
# When V_BT > 0 (net discharging), converted_v_bt passes through
# unchanged (Upper=inf is a no-op on the positive side).
# ===========================================================================

_EFC_P = _P.get("equivalent_consumption", {})

_ETA_BT: float = float(_EFC_P.get("battery_efficiency", 0.9))
_ETA_EM: float = float(_EFC_P.get("motor_efficiency", 0.8))
_ETA_CE: float = float(_EFC_P.get("engine_efficiency", 0.25))

# Saturation bounds — CONFIRMED from Simulink Saturation block
_SAT_LOWER: float = 0.0
_SAT_UPPER: float = np.inf

# Pre-compute the thermodynamic equivalence gain:
#   gain = 1 / (eta_BT * eta_EM * eta_CE * (H_u/3.6e6) * rho_f)
_EFC_DENOM: float = _ETA_BT * _ETA_EM * _ETA_CE * (_H_u / 3.6e6) * _RHO_FUEL
_EFC_GAIN: float = 1.0 / _EFC_DENOM if _EFC_DENOM > 0.0 else 0.0


def equivalent_fuel_consumption(v_ce: float, v_bt: float) -> dict:
    """
    Equivalent Fuel Consumption block — converts electrical consumption
    (V_BT, kWh/100km) into an equivalent liquid-fuel consumption and adds
    it to the engine's fuel consumption (V_CE, L/100km).

    Mirrors the Simulink block exactly:
        gain = 1 / (eta_BT*eta_EM*eta_CE*(H_u/3.6e6)*rho_f)
        converted_v_bt = V_BT * gain
        saturated_v_bt = clip(converted_v_bt, SAT_LOWER, SAT_UPPER)
        V_CE_equiv     = V_CE + saturated_v_bt

    Parameters
    ----------
    v_ce : float  Engine fuel consumption [L/100km] — from Tank block
    v_bt : float  Battery energy consumption [kWh/100km] — from Battery block

    Returns
    -------
    dict with keys:
        v_ce_equiv     [L/100km]  Total equivalent consumption (main output)
        v_bt_converted [L/100km]  Saturated electrical-to-fuel equivalent
                                   (diagnostic)
    """

    # ── Constant gain block ────────────────────────────────────────────────
    converted_v_bt: float = v_bt * _EFC_GAIN

    # ── Saturation block ────────────────────────────────────────────────────
    saturated_v_bt: float = float(np.clip(converted_v_bt, _SAT_LOWER, _SAT_UPPER))

    # ── Sum block ────────────────────────────────────────────────────────────
    v_ce_equiv: float = v_ce + saturated_v_bt

    return {
        "v_ce_equiv"    : v_ce_equiv,      # → main output [L/100km]
        "v_bt_converted": saturated_v_bt,  # diagnostic [L/100km]
    }
