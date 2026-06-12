"""
advanced_rule_based.py
=======================
Advanced rule-based Energy Management Strategy (EMS) — the redesigned
"Controller" subsystem from the model "qss_hybrid_electric_vehicle_
Bharati_Ramawat_Sharma".

Unlike the simple baseline in `rule_based.py` (LPS/regen only,
`state_CE=1` constant), this controller:
  - Uses gear-dependent torque-split rules (different T_MGB_th,
    u_Motor_max, u_Generator_max per gear, plus several SoC-, speed-,
    acceleration-, and torque-rate-dependent corrections).
  - Computes `state_CE` dynamically via the Supervision-Engine-Stop-like
    logic in `state_Bharati_Ramawat_Sharma` (port to `controller_state`
    below), based on the sign/value of `u`.
  - Requires one piece of extra state: `T_MGB_prev`, used to compute
    `d_T_MGB = T_MGB - T_MGB_prev` (the `[A]` signal / `1/z` delay block
    in the Controller subsystem, image 2).

Direct port of:
    function u = controller_Bharati_Ramawat_Sharma(input)
    function s = State_Bharati_Ramawat_Sharma(input)

Two cycle-dependent rule sets are supported, selected by `cycle_name`:
    "NEDC"  (stop_time == 1220)
    "FTP75" (stop_time == 1877)

Usage
-----
    ctrl = AdvancedController(cycle_name="NEDC")
    ctrl.reset()
    out = ctrl.step(w_MGB, dw_MGB, T_MGB, gear, Q_BT, v)
    # out["u"], out["state_CE"], out["d_T_MGB"]

Then feed out["u"] / out["state_CE"] into the SAME outer Control Unit
wiring as the baseline (control_unit_baseline's torque-split formula):
    T_EM = u * T_MGB
    T_CE = T_MGB * (1 - u)   if state_CE == 1
         = -inf               if state_CE == 0  (engine off; not
                               currently produced by this controller
                               since state_CE only takes values 0 or 1
                               per state_Bharati_Ramawat_Sharma, with
                               0 meaning "electric driving" -- engine
                               off -- and the outer Switch would need
                               to handle this branch. See
                               `control_unit_advanced()` below for the
                               handling used here.)
"""

from __future__ import annotations

import numpy as np

from src.env.powertrain import _w_EM_max_row, _T_EM_max_arr, _interp1d_linear
from src.env.powertrain import _Q_BT_IC


# ---------------------------------------------------------------------------
# Shared constants (from the MATLAB function)
# ---------------------------------------------------------------------------

_THETA_EM: float = 0.1   # motor inertia [kg*m^2]
_EPSILON:  float = 0.01  # epsilon margin


# ---------------------------------------------------------------------------
# state_Bharati_Ramawat_Sharma -> controller_state()
# ---------------------------------------------------------------------------

def controller_state(u: float) -> int:
    """
    Direct port of:

        function s = State_Bharati_Ramawat_Sharma(input)
        u = input(1);
        if (u == 1)
            s = 0;          % Electric driving
        elseif u == 0
            s = 1;          % Conventional driving
        else
            s = 1;          % Hybrid driving
        end

    Returns
    -------
    state_CE : int
        0 -> engine off  (pure electric driving, u == 1)
        1 -> engine on   (conventional or hybrid driving, u != 1)
    """
    if u == 1.0:
        return 0
    return 1


# ---------------------------------------------------------------------------
# controller_Bharati_Ramawat_Sharma -> _controller_u_nedc / _controller_u_ftp75
# ---------------------------------------------------------------------------

def _u_generator(w_MGB: float, dw_MGB: float, T_MGB: float, floor: float) -> float:
    """u = max((interp1(w_EM_max,-T_EM_max,w_MGB)+|theta_EM*dw_MGB|+eps)/T_MGB, floor)"""
    t_em_max_neg = -_interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w_MGB)
    val = (t_em_max_neg + abs(_THETA_EM * dw_MGB) + _EPSILON) / T_MGB
    return max(val, floor)


def _u_motor(w_MGB: float, dw_MGB: float, T_MGB: float, cap: float) -> float:
    """u = min((interp1(w_EM_max,T_EM_max,w_MGB)-|theta_EM*dw_MGB|-eps)/T_MGB, cap)"""
    t_em_max_pos = _interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w_MGB)
    val = (t_em_max_pos - abs(_THETA_EM * dw_MGB) - _EPSILON) / T_MGB
    return min(val, cap)


def _u_regen(w_MGB: float, dw_MGB: float, T_MGB: float, cap: float = 1.0) -> float:
    """u = min((interp1(w_EM_max,-T_EM_max,w_MGB)+|theta_EM*dw_MGB|+eps)/T_MGB, cap)"""
    t_em_max_neg = -_interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w_MGB)
    val = (t_em_max_neg + abs(_THETA_EM * dw_MGB) + _EPSILON) / T_MGB
    return min(val, cap)


def _controller_u_nedc(
    w_MGB: float, dw_MGB: float, T_MGB: float,
    gear: int, Q_BT: float, v: float, d_T_MGB: float,
) -> float:
    """
    Direct, per-gear port of the stop_time==1220 branch of
    controller_Bharati_Ramawat_Sharma. Each gear's if/elseif chain is
    written out explicitly to match the MATLAB source line-for-line
    (no generalization across gears).
    """

    if gear == 1:
        u_motor_max, u_gen_max, t_th = 0.5, -0.52, 29.0

        if (T_MGB > 0.0) and (T_MGB <= t_th):
            if Q_BT >= 0.02 * _Q_BT_IC:
                u = 1.0
            else:
                u = _u_generator(w_MGB, dw_MGB, T_MGB, -3.0)

        elif (T_MGB > t_th) and (T_MGB < 40.0) and (Q_BT <= 1.0 * _Q_BT_IC):
            u = _u_generator(w_MGB, dw_MGB, T_MGB, u_gen_max * 1.44)

        elif (T_MGB > 40.0) and (T_MGB < 98.0) and (Q_BT <= 0.9 * _Q_BT_IC):
            u = _u_generator(w_MGB, dw_MGB, T_MGB, u_gen_max)

        elif (T_MGB >= 120.0) and (Q_BT > 0.2 * _Q_BT_IC):
            u = _u_motor(w_MGB, dw_MGB, T_MGB, u_motor_max)

        elif (T_MGB < 0.0) and (Q_BT <= 1.2 * _Q_BT_IC):
            u = _u_regen(w_MGB, dw_MGB, T_MGB, 1.0)

        else:
            u = 0.0

    elif gear == 2:
        u_motor_max, u_gen_max, t_th = 0.35, -0.52, 34.0

        if (T_MGB > 0.0) and (T_MGB <= t_th):
            if Q_BT >= 0.02 * _Q_BT_IC:
                u = 1.0
            else:
                u = _u_generator(w_MGB, dw_MGB, T_MGB, -3.0)

        elif (T_MGB > t_th) and (T_MGB < 98.0) and (Q_BT <= 1.0 * _Q_BT_IC):
            u = _u_generator(w_MGB, dw_MGB, T_MGB, u_gen_max)

        elif (T_MGB >= 120.0) and (Q_BT > 0.2 * _Q_BT_IC):
            u = _u_motor(w_MGB, dw_MGB, T_MGB, u_motor_max)

        elif (T_MGB < 0.0) and (Q_BT <= 1.2 * _Q_BT_IC):
            u = _u_regen(w_MGB, dw_MGB, T_MGB, 1.0)

        else:
            u = 0.0

    elif gear == 3:
        u_motor_max, u_gen_max, t_th = 0.3, -0.52, 36.0

        if (T_MGB > 0.0) and (T_MGB <= t_th):
            if Q_BT >= 0.02 * _Q_BT_IC:
                u = 1.0
            else:
                u = _u_generator(w_MGB, dw_MGB, T_MGB, -3.0)

        elif (T_MGB > t_th) and (T_MGB < 98.0) and (Q_BT <= 1.0 * _Q_BT_IC):
            u = _u_generator(w_MGB, dw_MGB, T_MGB, u_gen_max)

        elif (T_MGB >= 120.0) and (Q_BT > 0.2 * _Q_BT_IC):
            u = _u_motor(w_MGB, dw_MGB, T_MGB, u_motor_max)

        elif (T_MGB < 0.0) and (Q_BT <= 1.2 * _Q_BT_IC):
            u = _u_regen(w_MGB, dw_MGB, T_MGB, 1.0)

        else:
            u = 0.0

    elif gear == 4:
        u_motor_max, u_gen_max, t_th = 0.25, -0.52, 32.0

        if (T_MGB > 0.0) and (T_MGB <= t_th):
            if Q_BT >= 0.02 * _Q_BT_IC:
                u = 1.0
            else:
                u = _u_generator(w_MGB, dw_MGB, T_MGB, -3.0)

        elif (T_MGB > t_th) and (T_MGB < 98.0) and (Q_BT <= 1.0 * _Q_BT_IC):
            u = _u_generator(w_MGB, dw_MGB, T_MGB, u_gen_max)

        elif (T_MGB >= 120.0) and (Q_BT > 0.2 * _Q_BT_IC):
            u = _u_motor(w_MGB, dw_MGB, T_MGB, u_motor_max)

        elif (T_MGB < 0.0) and (Q_BT <= 1.2 * _Q_BT_IC):
            u = _u_regen(w_MGB, dw_MGB, T_MGB, 1.0)

        else:
            u = 0.0

    elif gear == 5:
        u_motor_max, u_gen_max, t_th = 0.2, -0.52, 36.0

        if (T_MGB > 0.0) and (T_MGB <= t_th):
            if Q_BT >= 0.02 * _Q_BT_IC:
                u = 1.0
            else:
                u = _u_generator(w_MGB, dw_MGB, T_MGB, -3.0)

        elif (T_MGB > t_th) and (T_MGB < 98.0) and (Q_BT <= 1.0 * _Q_BT_IC):
            u = _u_generator(w_MGB, dw_MGB, T_MGB, u_gen_max)

        elif (T_MGB >= 120.0) and (Q_BT > 0.2 * _Q_BT_IC):
            u = _u_motor(w_MGB, dw_MGB, T_MGB, u_motor_max)

        elif (T_MGB < 0.0) and (Q_BT <= 1.2 * _Q_BT_IC):
            u = _u_regen(w_MGB, dw_MGB, T_MGB, 1.0)

        else:
            u = 0.0

    else:  # gear >= 6
        u_motor_max, u_gen_max, t_th = 0.2, -0.52, 36.0

        if (T_MGB > 0.0) and (T_MGB <= t_th):
            if Q_BT >= 0.02 * _Q_BT_IC:
                u = 1.0
            else:
                u = _u_generator(w_MGB, dw_MGB, T_MGB, -3.0)

        elif (T_MGB > t_th) and (T_MGB < 98.0) and (Q_BT <= 1.0 * _Q_BT_IC):
            u = _u_generator(w_MGB, dw_MGB, T_MGB, u_gen_max)

        elif (T_MGB >= 120.0) and (Q_BT > 0.15 * _Q_BT_IC):
            u = _u_motor(w_MGB, dw_MGB, T_MGB, u_motor_max)

        elif (T_MGB < 0.0) and (Q_BT <= 1.2 * _Q_BT_IC):
            u = _u_regen(w_MGB, dw_MGB, T_MGB, 1.0)

        else:
            u = 0.0

    # ── Post-processing corrections (stop_time == 1220) ──────────────────
    if v == 0.0:
        u = 0.0
    elif v < 1.2:
        u = 1.0

    if 190.0 < w_MGB < 230.0:
        u = u * 0.9811

    if dw_MGB == 0.0:
        u = u * 0.8557
    elif 0.0 < dw_MGB < 18.0:
        u = u - 0.06

    if (d_T_MGB > 40.0) and (u < 0.6):
        u = u * 0.9

    return float(u)


def _controller_u_ftp75(
    w_MGB: float, dw_MGB: float, T_MGB: float,
    gear: int, Q_BT: float, v: float, d_T_MGB: float,
) -> float:
    """Gear-dependent rule set for stop_time == 1877 (FTP-75)."""

    if gear == 1:
        u_motor_max, u_gen_max, t_th = 0.5, -0.19, 37.0
        electric_cond = Q_BT > 0.0
    elif gear == 2:
        u_motor_max, u_gen_max, t_th = 0.1, -0.52, 37.0
        electric_cond = Q_BT >= 0.0
    elif gear == 3:
        u_motor_max, u_gen_max, t_th = 0.1, -0.52, 37.0
        electric_cond = Q_BT >= 0.0
    elif gear == 4:
        u_motor_max, u_gen_max, t_th = 0.1, -0.52, 35.0
        electric_cond = Q_BT >= 0.0
    elif gear == 5:
        u_motor_max, u_gen_max, t_th = 0.02, -0.56, 37.0
        electric_cond = Q_BT >= 0.0
    else:  # gear >= 6
        u_motor_max, u_gen_max, t_th = 0.05, -0.56, 35.0
        electric_cond = Q_BT > 0.0

    if 0.0 < T_MGB <= t_th:
        if electric_cond:
            u = 1.0                                            # Electric Driving
        else:
            u = _u_generator(w_MGB, dw_MGB, T_MGB, -3.0)       # LPS-Generator mode

    elif (T_MGB > t_th) and (T_MGB < 98.0) and (Q_BT <= 0.96 * _Q_BT_IC):
        u = _u_generator(w_MGB, dw_MGB, T_MGB, u_gen_max)      # LPS-Generator mode

    elif (T_MGB >= 120.0) and (Q_BT > 0.0):
        u = _u_motor(w_MGB, dw_MGB, T_MGB, u_motor_max)        # LPS-Motor mode

    elif T_MGB < 0.0:
        if gear >= 6:
            # NOTE: original code uses `max(...,1)` here (not `min`),
            # which for a negative numerator/T_MGB<0 ratio is unusual
            # but is ported verbatim.
            t_em_max_neg = -_interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w_MGB)
            val = (t_em_max_neg + abs(_THETA_EM * dw_MGB) + _EPSILON) / T_MGB
            u = max(val, 1.0)
        else:
            u = _u_regen(w_MGB, dw_MGB, T_MGB, 1.0)            # Regeneration

    else:
        u = 0.0                                                # Engine only

    # ── Post-processing corrections (stop_time == 1877) ──────────────────
    if dw_MGB == 0.0:                       # Cruising Mode
        u = u - 0.27
    elif 0.0 < dw_MGB < 18.0:
        u = u - 0.06

    if v == 0.0:
        u = 0.0
    elif v < 1.2:
        u = 1.0

    if (190.0 < w_MGB < 230.0) and (u < 0.6):
        u = u / 0.9811

    if (d_T_MGB > 80.0) and (u < 0.6):
        u = u * 0.95

    return float(u)


# ---------------------------------------------------------------------------
# Stateful wrapper (mirrors VehicleDynamics / Tank / Battery pattern)
# ---------------------------------------------------------------------------

class AdvancedController:
    """
    Stateful wrapper around the advanced rule-based Controller subsystem.

    State
    -----
    T_MGB_prev : float
        Previous timestep's T_MGB, used to compute d_T_MGB = T_MGB -
        T_MGB_prev (the `[A]` / `1/z` delay signal in image 2).

    Usage
    -----
        ctrl = AdvancedController(cycle_name="NEDC")
        ctrl.reset()
        out = ctrl.step(w_MGB=..., dw_MGB=..., T_MGB=..., gear=...,
                         Q_BT=..., v=...)
        # out["u"]        -> torque-split factor
        # out["state_CE"] -> 0 (engine off) or 1 (engine on)
        # out["d_T_MGB"]  -> diagnostic, rate of change of T_MGB
    """

    def __init__(self, cycle_name: str = "NEDC") -> None:
        name = cycle_name.upper().replace("-", "").replace("_", "")
        if name == "NEDC":
            self._u_fn = _controller_u_nedc
        elif name == "FTP75":
            self._u_fn = _controller_u_ftp75
        else:
            raise ValueError(f"Unknown cycle '{cycle_name}'. Use 'NEDC' or 'FTP75'.")

        self.cycle_name = name
        self.T_MGB_prev: float = 0.0

    def reset(self) -> None:
        """Reset state. Call at the start of every episode."""
        self.T_MGB_prev = 0.0

    def step(
        self,
        w_MGB: float,
        dw_MGB: float,
        T_MGB: float,
        gear: int,
        Q_BT: float,
        v: float,
    ) -> dict:
        """
        Advance one timestep.

        Parameters
        ----------
        w_MGB, dw_MGB, T_MGB : float  Flywheel speed/accel/torque — from Gearbox
        gear  : int    Gear number (1-5, or >=6) — from Driving Cycle
        Q_BT  : float  Battery charge [As] — fed back from Battery block
        v     : float  Vehicle speed [m/s] — from Driving Cycle

        Returns
        -------
        dict with keys:
            u        [-]   Torque-split factor
            state_CE [-]   0 (engine off) or 1 (engine on)
            d_T_MGB  [Nm]  Rate of change of T_MGB (diagnostic)
        """
        d_T_MGB: float = T_MGB - self.T_MGB_prev

        u: float = self._u_fn(w_MGB, dw_MGB, T_MGB, gear, Q_BT, v, d_T_MGB)
        state_CE: int = controller_state(u)

        self.T_MGB_prev = T_MGB

        return {
            "u"       : u,
            "state_CE": state_CE,
            "d_T_MGB" : d_T_MGB,
        }


# ---------------------------------------------------------------------------
# Outer Control Unit wiring (advanced) — handles state_CE == 0 (engine off)
# ---------------------------------------------------------------------------

def control_unit_advanced(
    w_MGB: float,
    dw_MGB: float,
    T_MGB: float,
    u: float,
    state_CE: int,
) -> dict:
    """
    Outer Control Unit wiring for the advanced controller.

    Same Sum/Switch/Multiply structure as `control_unit_baseline`
    (T_EM = u*T_MGB, T_CE = T_MGB*(1-u) when state_CE==1), but now
    state_CE can be 0 (pure electric driving, u==1):

        state_CE == 1 (engine on):
            T_EM = u * T_MGB
            T_CE = T_MGB * (1 - u)

        state_CE == 0 (engine off, u==1):
            T_EM = T_MGB              (motor carries full load)
            T_CE = 0                  (engine fully unloaded; the
                                       combustion_engine block's own
                                       idle/cutoff logic will then
                                       apply, same as the baseline's
                                       T_CE==0 cases)

    Note: state_CE==0 only occurs when u==1, in which case
    T_MGB*(1-u)=0 anyway -- so both branches give the SAME T_CE/T_EM
    values. state_CE is kept as a separate diagnostic output (it does
    NOT gate a -inf branch here, unlike the baseline's Switch, because
    this controller never produces the state_CE==0-with-engine-forced-
    off-via-(-inf) case from the original Supervision-Engine-Stop logic
    -- it's a simplified/renamed state flag per state_Bharati_Ramawat_
    Sharma).

    Returns
    -------
    dict with keys matching control_unit_baseline()'s output:
        w_gear, dw_gear, T_CE, T_EM, u, state_CE
    """
    t_em: float = u * T_MGB
    t_ce: float = T_MGB * (1.0 - u)

    return {
        "w_gear" : w_MGB,
        "dw_gear": dw_MGB,
        "T_CE"   : t_ce,
        "T_EM"   : t_em,
        "u"      : u,
        "state_CE": state_CE,
    }