"""
rule_based.py
==============
Baseline rule-based Energy Management Strategy (EMS) — the "Control Unit
(seminar)" Simulink subsystem, baseline configuration.

Baseline configuration (current task)
--------------------------------------
  - Controller subsystem outputs `u` via the Load-Point-Shifting (LPS) /
    regeneration logic shown in the Interpreted MATLAB Fcn block
    (image 4 of the Controller subsystem).
  - `state_CE` is a CONSTANT = 1 (the Supervision-Engine-Stop subsystem is
    present in the diagram but not yet wired into state_CE in this baseline;
    it becomes active only in the later rule-based revision).
  - The outer Control Unit wiring (image 2) — Sum, Switch (gated by
    state_CE), -inf constant, and Multiply blocks — converts (u, state_CE,
    T_MGB) into (T_CE, T_EM).

Reference result (NEDC, 1220 s, this baseline):
    V_liter (Tank)        = 4.513   L/100km
    V_CE_equiv (EFC)      = 4.52
    Q_BT (final)          = 1.784e+04  As

Simulink block mapping (Control Unit top level, image 2)
---------------------------------------------------------
  Inputs : w_MGB, dw_MGB, T_MGB, i, v, Q_BT
  Outputs: w_MGB (pass-through -> w_gear), dw_MGB (pass-through -> dw_gear),
           T_CE, T_EM

  Controller subsystem (image 4, baseline):
      u        = controller_u(w_MGB, dw_MGB, T_MGB)   <- LPS/regen Fcn
      state_CE = 1                                     <- constant

  Outer wiring (image 2):
      T_CE  = (T_MGB - T_EM_request) routed through Switch on state_CE
              with -inf as the "off" branch
      T_EM  = u * T_MGB                                <- Multiply block

  See `control_unit_baseline()` below for the exact signal chain.
"""

from __future__ import annotations

import numpy as np

from src.env.powertrain import _w_EM_max_row, _T_EM_max_arr, _interp1d_linear


# ---------------------------------------------------------------------------
# Controller constants (from the Interpreted MATLAB Fcn, image 4)
# ---------------------------------------------------------------------------

_THETA_EM    = 0.1    # motor inertia [kg*m^2]  (same as powertrain._THETA_EM)
_EPSILON     = 0.01   # epsilon margin (Slide 3-8 / 3-10)
_U_LPS_MAX   = 0.3    # max torque-split factor for load-point shifting

# Torque threshold for switching into load-point-shifting mode.
# NEDC value (Slide 3-8). Use 39.5 for FTP-75.
_T_MGB_TH_NEDC  = 60.0
_T_MGB_TH_FTP75 = 39.5


# ---------------------------------------------------------------------------
# Controller subsystem — computes u (baseline, image 4)
# ---------------------------------------------------------------------------

def controller_u(
    w_MGB:  float,
    dw_MGB: float,
    T_MGB:  float,
    t_mgb_threshold: float = _T_MGB_TH_NEDC,
) -> float:
    """
    Baseline Controller Fcn block — computes the torque-split factor `u`.

    Direct port of:

        function u = controller(input)
        w_MGB = input(1);
        dw_MGB = input(2);
        T_MGB = input(3);
        global w_EM_max;
        global T_EM_max;
        theta_EM = 0.1;
        T_MGB_th = 60;        % NEDC
        epsilon = 0.01;
        u_LPS_max = 0.3;
        if T_MGB < 0
            u = min((interp1(w_EM_max,-T_EM_max,w_MGB)
                     + abs(theta_EM*dw_MGB) + epsilon)/T_MGB, 1);
        elseif T_MGB >= T_MGB_th
            u = min((interp1(w_EM_max, T_EM_max,w_MGB)
                     - abs(theta_EM*dw_MGB) - epsilon)/T_MGB, u_LPS_max);
        else
            u = 0;
        end

    `w_EM_max` / `T_EM_max` are taken from `env.powertrain`
    (`_w_EM_max_row`, `_T_EM_max_arr`) — same motor-max-torque curve used by
    the Electric Motor block's overload check.

    Parameters
    ----------
    w_MGB, dw_MGB, T_MGB : float
        Flywheel speed [rad/s], acceleration [rad/s^2], torque [Nm].
    t_mgb_threshold : float
        T_MGB_th. Default 60.0 (NEDC). Pass 39.5 for FTP-75.

    Returns
    -------
    u : float
        Torque-split factor.
            u < 0          -> regeneration  (T_EM negative, recovers energy)
            0 < u <= 0.3    -> load-point shifting (motor assists engine)
            u == 0          -> engine-only
    """
    inertia_term = abs(_THETA_EM * dw_MGB)

    if T_MGB < 0.0:
        # Regeneration branch (Slide 3-10)
        t_em_max_neg = -_interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w_MGB)
        u = min((t_em_max_neg + inertia_term + _EPSILON) / T_MGB, 1.0)

    elif T_MGB >= t_mgb_threshold:
        # Load-point-shifting branch (Slide 3-8)
        t_em_max_pos = _interp1d_linear(_w_EM_max_row, _T_EM_max_arr, w_MGB)
        u = min((t_em_max_pos - inertia_term - _EPSILON) / T_MGB, _U_LPS_MAX)

    else:
        # Engine-only
        u = 0.0

    return float(u)


# ---------------------------------------------------------------------------
# Outer Control Unit wiring (image 2, baseline: state_CE = 1 constant)
# ---------------------------------------------------------------------------

def control_unit_baseline(
    w_MGB:  float,
    dw_MGB: float,
    T_MGB:  float,
    i:      float,
    v:      float,
    Q_BT:   float,
    t_mgb_threshold: float = _T_MGB_TH_NEDC,
) -> dict:
    """
    Baseline "Control Unit (seminar)" subsystem.

    Reproduces the full signal chain of images 1-4:

      Controller (image 4):
          u        = controller_u(w_MGB, dw_MGB, T_MGB)
          state_CE = 1                                    (constant, baseline)

      Outer wiring (image 2):
          T_EM = u * T_MGB                                (Multiply block)
          T_CE = T_MGB - T_EM   if state_CE == 1 (engine running)
               = -inf            if state_CE == 0 (engine off, Switch
                                                     selects -inf branch)
                                  -- not reachable in this baseline since
                                     state_CE is always 1.

      Pass-through outputs (image 2, top two wires):
          w_gear  = w_MGB
          dw_gear = dw_MGB

    Parameters
    ----------
    w_MGB, dw_MGB, T_MGB : float
        Flywheel kinematics/torque from the Manual Gear Box.
    i, v, Q_BT : float
        Gear number, vehicle speed, battery charge — present as Control
        Unit inputs (images 1/2) but not used by the baseline strategy.
        Kept in the signature for interface parity with the future
        rule-based controller (which uses Q_BT / v / i).
    t_mgb_threshold : float
        T_MGB_th passed to controller_u(). Default NEDC (60.0).

    Returns
    -------
    dict with keys matching Simulink Control Unit output ports:
        w_gear   [rad/s]   -> Combustion Engine & Electric Motor w_gear
        dw_gear  [rad/s^2] -> Combustion Engine & Electric Motor dw_gear
        T_CE     [Nm]      -> Combustion Engine T_gear
        T_EM     [Nm]      -> Electric Motor T_gear
        u        [-]       diagnostic, torque-split factor
        state_CE [-]       diagnostic, engine on/off flag (always 1, baseline)
    """
    # Controller subsystem (image 4)
    u: float = controller_u(w_MGB, dw_MGB, T_MGB, t_mgb_threshold)
    state_CE: int = 1  # baseline: constant, supervision_engine_stop not wired

    # Outer wiring (image 2)
    t_em: float = u * T_MGB

    if state_CE == 1:
        t_ce: float = T_MGB - t_em
    else:
        # Switch selects the -inf branch when state_CE == 0
        # (engine commanded off). Not reached in baseline.
        t_ce = float("-inf")

    return {
        "w_gear" : w_MGB,
        "dw_gear": dw_MGB,
        "T_CE"   : t_ce,
        "T_EM"   : t_em,
        "u"      : u,
        "state_CE": state_CE,
    }
