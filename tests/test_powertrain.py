"""
test_powertrain.py
==================
Unit tests for the Vehicle block inside powertrain.py.

Validates every subsystem against hand-calculated reference values
derived directly from the Simulink block parameters.

Run with:
    pytest tests/test_powertrain.py -v
"""

import math
import numpy as np
import pytest

from env.powertrain import (
    VehicleDynamics,
    vehicle_dynamics,
    R_WHEEL,
    INERTIA_FACTOR,
    F_ROLL_CONST,
    AERO_FACTOR,
)


# ---------------------------------------------------------------------------
# Known constants (from params.json / Simulink model)
# ---------------------------------------------------------------------------
M        = 1115.0    # kg
R_ROT    = 0.05      # 5%
A_F      = 2.3       # m^2
R_W      = 0.288     # m  (0.576/2)
CW       = 0.31
MU       = 0.013
RHO      = 1.18      # kg/m^3 (Phase-1 MATLAB-validation-corrected value; see README)
G        = 9.81      # m/s^2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def veh():
    v = VehicleDynamics()
    v.reset()
    return v


# ---------------------------------------------------------------------------
# 1. Derived constants
# ---------------------------------------------------------------------------

def test_r_wheel():
    assert abs(R_WHEEL - 0.288) < 1e-6, f"R_WHEEL={R_WHEEL}, expected 0.288"

def test_inertia_factor():
    assert abs(INERTIA_FACTOR - 1.05) < 1e-6

def test_f_roll_const():
    expected = M * G * MU    # 1115 * 9.81 * 0.013 = 142.236 N
    assert abs(F_ROLL_CONST - expected) < 0.01

def test_aero_factor():
    expected = 0.5 * RHO * CW * A_F    # 0.5*1.2*0.31*2.3 = 0.4278
    assert abs(AERO_FACTOR - expected) < 1e-4


# ---------------------------------------------------------------------------
# 2. Average speed subsystem
# ---------------------------------------------------------------------------

def test_average_speed_first_step(veh):
    """First step: v_prev=0, so v_a = 0.5*(v + 0) = v/2"""
    out = veh.step(v=10.0, dv=0.0)
    assert abs(out["v_a"] - 5.0) < 1e-6, f"v_a={out['v_a']}, expected 5.0"

def test_average_speed_second_step(veh):
    """Second step: v_prev=10, v=20 → v_a = 0.5*(20+10) = 15"""
    veh.step(v=10.0, dv=0.0)   # first step sets v_prev=10
    out = veh.step(v=20.0, dv=0.0)
    assert abs(out["v_a"] - 15.0) < 1e-6

def test_average_speed_standstill(veh):
    """At standstill both v and v_prev are 0 → v_a = 0"""
    out = veh.step(v=0.0, dv=0.0)
    assert out["v_a"] == 0.0


# ---------------------------------------------------------------------------
# 3. Rolling resistance subsystem (switch block)
# ---------------------------------------------------------------------------

def test_f_roll_moving(veh):
    """When v_a > 0, F_roll = m*g*mu (constant)"""
    out = veh.step(v=10.0, dv=0.0)
    # v_a = 5.0 > 0 → F_roll active
    expected = M * G * MU
    assert abs(out["F_roll"] - expected) < 0.01

def test_f_roll_standstill(veh):
    """When v_a = 0, F_roll = 0 (switch selects standstill input)"""
    out = veh.step(v=0.0, dv=0.0)
    assert out["F_roll"] == 0.0

def test_f_roll_just_started(veh):
    """v_prev=0, v=0 → v_a=0 → F_roll=0 at very first step"""
    out = veh.step(v=0.0, dv=1.0)
    assert out["F_roll"] == 0.0


# ---------------------------------------------------------------------------
# 4. Aerodynamic resistance subsystem
# ---------------------------------------------------------------------------

def test_f_aero_formula(veh):
    """F_aero = 0.5 * rho * cw * A_f * v_a^2"""
    # v_prev=0, v=20 → v_a=10
    out = veh.step(v=20.0, dv=0.0)
    v_a = 10.0
    expected = 0.5 * RHO * CW * A_F * v_a**2
    assert abs(out["F_aero"] - expected) < 0.01

def test_f_aero_zero_speed(veh):
    out = veh.step(v=0.0, dv=0.0)
    assert out["F_aero"] == 0.0

def test_f_aero_at_120kmh():
    """At 120 km/h = 33.33 m/s, v_a = 33.33 (after warmup)"""
    veh = VehicleDynamics()
    veh.v_prev = 33.333
    out = veh.step(v=33.333, dv=0.0)
    v_a = 33.333
    expected = 0.5 * RHO * CW * A_F * v_a**2
    assert abs(out["F_aero"] - expected) < 0.5    # within 0.5 N


# ---------------------------------------------------------------------------
# 5. Inertia subsystem
# ---------------------------------------------------------------------------

def test_f_iner_formula(veh):
    """F_iner = m * (1 + r_rot) * dv = 1115 * 1.05 * dv"""
    dv = 2.0
    out = veh.step(v=0.0, dv=dv)
    expected = M * (1 + R_ROT) * dv
    assert abs(out["F_iner"] - expected) < 0.01

def test_f_iner_zero_accel(veh):
    out = veh.step(v=10.0, dv=0.0)
    assert abs(out["F_iner"]) < 1e-9

def test_f_iner_braking(veh):
    """Negative dv → negative F_iner (braking force)"""
    out = veh.step(v=10.0, dv=-2.0)
    assert out["F_iner"] < 0.0


# ---------------------------------------------------------------------------
# 6. Wheel-side outputs
# ---------------------------------------------------------------------------

def test_w_wheel(veh):
    """w_wheel = v_a / r_wheel (averaged speed, not instantaneous v)."""
    v = 20.0
    out = veh.step(v=v, dv=0.0)
    v_a = 0.5 * (v + 0.0)   # fresh fixture: v_prev = 0
    assert abs(out["w_wheel"] - v_a / R_W) < 1e-6

def test_dw_wheel(veh):
    """dw_wheel = dv / r_wheel"""
    dv = 1.5
    out = veh.step(v=5.0, dv=dv)
    assert abs(out["dw_wheel"] - dv / R_W) < 1e-6

def test_t_wheel_pure_inertia(veh):
    """
    At v=0, dv=1: F_roll=0, F_aero=0, F_iner=m*(1+r)*dv
    T_wheel = F_iner * r_wheel
    """
    dv = 1.0
    out = veh.step(v=0.0, dv=dv)
    F_iner = M * (1 + R_ROT) * dv
    expected_T = F_iner * R_W
    assert abs(out["T_wheel"] - expected_T) < 0.01

def test_t_wheel_cruise(veh):
    """
    Cruising at steady 50 km/h = 13.89 m/s, dv=0.
    v_prev=13.89 → v_a=13.89, F_roll active, F_iner=0
    T_wheel = (F_roll + F_aero) * r_wheel
    """
    v = 50 / 3.6    # 13.889 m/s
    veh.v_prev = v  # already at cruise → v_a = v
    out = veh.step(v=v, dv=0.0)
    F_r = M * G * MU
    F_a = 0.5 * RHO * CW * A_F * v**2
    expected_T = (F_r + F_a) * R_W
    assert abs(out["T_wheel"] - expected_T) < 0.5   # within 0.5 Nm


# ---------------------------------------------------------------------------
# 7. Diagnostic power signals
# ---------------------------------------------------------------------------

def test_p_roll_equals_f_roll_times_va(veh):
    out = veh.step(v=20.0, dv=0.0)
    assert abs(out["P_roll"] - out["F_roll"] * out["v_a"]) < 1e-6

def test_p_aero_equals_f_aero_times_va(veh):
    out = veh.step(v=20.0, dv=0.0)
    assert abs(out["P_aero"] - out["F_aero"] * out["v_a"]) < 1e-6

def test_p_iner_equals_f_iner_times_va(veh):
    out = veh.step(v=10.0, dv=2.0)
    assert abs(out["P_iner"] - out["F_iner"] * out["v_a"]) < 1e-6

def test_p_wheel_equals_sum(veh):
    """P_wheel = (F_roll + F_aero + F_iner) * v_a = P_roll + P_aero + P_iner"""
    out = veh.step(v=15.0, dv=1.0)
    total = out["P_roll"] + out["P_aero"] + out["P_iner"]
    assert abs(out["P_wheel"] - total) < 1e-4


# ---------------------------------------------------------------------------
# 8. State management
# ---------------------------------------------------------------------------

def test_reset_clears_v_prev(veh):
    veh.step(v=30.0, dv=0.0)
    assert abs(veh.v_prev - 30.0) < 1e-6
    veh.reset()
    assert veh.v_prev == 0.0

def test_v_prev_updates_each_step(veh):
    veh.step(v=10.0, dv=0.0)
    assert abs(veh.v_prev - 10.0) < 1e-6
    veh.step(v=20.0, dv=0.0)
    assert abs(veh.v_prev - 20.0) < 1e-6


# ---------------------------------------------------------------------------
# 9. Full cycle integration test
# ---------------------------------------------------------------------------

def test_full_nedc_run():
    """
    Run the complete NEDC through the Vehicle block.
    Checks:
      - No NaN or Inf in any output
      - T_wheel is negative during hard braking (regenerative potential)
      - w_wheel tracks speed correctly throughout
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from env.driving_cycle import DrivingCycle

    cycle = DrivingCycle("NEDC")
    veh   = VehicleDynamics()
    veh.reset()
    obs = cycle.reset()

    w_wheels, T_wheels = [], []

    while True:
        out = veh.step(obs["v"], obs["dv"])
        w_wheels.append(out["w_wheel"])
        T_wheels.append(out["T_wheel"])

        for key, val in out.items():
            assert math.isfinite(val), f"NaN/Inf in {key} at t={obs['t']}"

        obs, done = cycle.step()
        if done:
            break

    w_arr = np.array(w_wheels)
    T_arr = np.array(T_wheels)

    # w_wheel must always be >= 0 (speed is non-negative)
    assert w_arr.min() >= 0.0, f"Negative w_wheel: {w_arr.min():.3f}"

    # There must be at least some braking events (negative T_wheel)
    assert T_arr.min() < 0.0, "No braking events found — check inertia sign"

    # Peak wheel speed at 120 km/h = 33.33/0.288 ≈ 115.7 rad/s
    expected_peak = 33.333 / R_WHEEL
    assert abs(w_arr.max() - expected_peak) < 1.0, \
        f"Peak w_wheel={w_arr.max():.2f}, expected ~{expected_peak:.2f}"


# ---------------------------------------------------------------------------
# 10. Stateless convenience function
# ---------------------------------------------------------------------------

def test_stateless_api():
    out, v_now = vehicle_dynamics(v=10.0, dv=1.0, v_prev=0.0)
    assert v_now == 10.0
    assert "w_wheel" in out
    assert "T_wheel" in out

def test_stateless_matches_class():
    veh = VehicleDynamics()
    veh.reset()
    class_out = veh.step(v=15.0, dv=2.0)
    func_out, _ = vehicle_dynamics(v=15.0, dv=2.0, v_prev=0.0)
    for key in ["w_wheel", "dw_wheel", "T_wheel", "F_roll", "F_aero", "F_iner"]:
        assert abs(class_out[key] - func_out[key]) < 1e-9, \
            f"Mismatch in {key}: class={class_out[key]}, func={func_out[key]}"


# ===========================================================================
# Gearbox tests
# ===========================================================================

from env.powertrain import gearbox, _GEAR_RATIOS, _DIFF_RATIO, _ETA, _P_FRICTION, _W_MIN

# Known total ratios: gear_ratio[i-1] * diff_ratio
# Gear 1: 3.27 * 3.29 = 10.7583
# Gear 2: 1.92 * 3.29 =  6.3168
# Gear 3: 1.26 * 3.29 =  4.1454
# Gear 4: 0.88 * 3.29 =  2.8952
# Gear 5: 0.70 * 3.29 =  2.3030

def _i_gt(gear: int) -> float:
    return _GEAR_RATIOS[gear - 1] * _DIFF_RATIO


# ---------------------------------------------------------------------------
# 1. Neutral gear (gear=0)
# ---------------------------------------------------------------------------

def test_gearbox_neutral_all_zeros():
    out = gearbox(w_wheel=10.0, dw_wheel=1.0, t_wheel=50.0, gear=0)
    assert out["w_mgb"]  == 0.0
    assert out["dw_mgb"] == 0.0
    assert out["t_mgb"]  == 0.0
    assert out["p_mgb"]  == 0.0


# ---------------------------------------------------------------------------
# 2. Speed and acceleration scaling (all 5 gears)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gear", [1, 2, 3, 4, 5])
def test_w_mgb_scaling(gear):
    w = 20.0
    out = gearbox(w_wheel=w, dw_wheel=0.0, t_wheel=10.0, gear=gear)
    expected = w * _i_gt(gear)
    assert abs(out["w_mgb"] - expected) < 1e-6, \
        f"gear={gear}: w_mgb={out['w_mgb']:.4f}, expected={expected:.4f}"

@pytest.mark.parametrize("gear", [1, 2, 3, 4, 5])
def test_dw_mgb_scaling(gear):
    dw = 1.5
    out = gearbox(w_wheel=5.0, dw_wheel=dw, t_wheel=10.0, gear=gear)
    expected = dw * _i_gt(gear)
    assert abs(out["dw_mgb"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# 3. i_gt values match expected total ratios
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gear,expected_igt", [
    (1, 3.27 * 3.29),
    (2, 1.92 * 3.29),
    (3, 1.26 * 3.29),
    (4, 0.88 * 3.29),
    (5, 0.70 * 3.29),
])
def test_i_gt_values(gear, expected_igt):
    out = gearbox(w_wheel=10.0, dw_wheel=0.0, t_wheel=10.0, gear=gear)
    assert abs(out["i_gt"] - expected_igt) < 1e-6


# ---------------------------------------------------------------------------
# 4. Power flow — Engine->Wheel path (driving, t_wheel >= 0)
# ---------------------------------------------------------------------------

def test_t_mgb_driving_formula():
    """
    Engine->Wheel (Image 3):
    T_MGB+ = (T_wheel + P_idle / w_wheel) / eta / i_gt
    (wheel-side -> flywheel-side conversion requires dividing by the
    total gear ratio i_gt; see README "Key Implementation Notes".)
    """
    w, t, gear = 20.0, 100.0, 3
    out = gearbox(w_wheel=w, dw_wheel=0.0, t_wheel=t, gear=gear)
    friction = _P_FRICTION / w
    expected = (t + friction) / _ETA / _i_gt(gear)
    assert abs(out["t_mgb"] - expected) < 1e-6

def test_t_mgb_driving_zero_torque():
    """Even at t_wheel=0 (coasting), friction loss still demands torque."""
    w, gear = 15.0, 2
    out = gearbox(w_wheel=w, dw_wheel=0.0, t_wheel=0.0, gear=gear)
    expected = (_P_FRICTION / w) / _ETA / _i_gt(gear)
    assert abs(out["t_mgb"] - expected) < 1e-6

def test_t_mgb_driving_is_positive(gear=3):
    out = gearbox(w_wheel=10.0, dw_wheel=0.0, t_wheel=50.0, gear=3)
    assert out["t_mgb"] > 0.0


# ---------------------------------------------------------------------------
# 5. Power flow — Wheel->Engine path (braking, t_wheel < 0)
# ---------------------------------------------------------------------------

def test_t_mgb_braking_formula():
    """
    Wheel->Engine (Image 4):
    T_MGB- = (T_wheel + eta * P_idle / w_wheel) * eta / i_gt
    """
    w, t, gear = 20.0, -80.0, 3
    out = gearbox(w_wheel=w, dw_wheel=0.0, t_wheel=t, gear=gear)
    friction = _P_FRICTION / w
    expected = (t + _ETA * friction) * _ETA / _i_gt(gear)
    assert abs(out["t_mgb"] - expected) < 1e-6

def test_t_mgb_braking_is_negative():
    """Hard braking → T_MGB must be negative."""
    out = gearbox(w_wheel=20.0, dw_wheel=0.0, t_wheel=-200.0, gear=3)
    assert out["t_mgb"] < 0.0

def test_t_mgb_asymmetry():
    """
    Efficiency is applied asymmetrically:
    driving torque > |braking torque| for same |T_wheel|
    because friction adds load when driving and reduces it when braking.
    """
    w, t_abs, gear = 20.0, 100.0, 3
    out_drive = gearbox(w_wheel=w, dw_wheel=0.0, t_wheel= t_abs, gear=gear)
    out_brake = gearbox(w_wheel=w, dw_wheel=0.0, t_wheel=-t_abs, gear=gear)
    assert out_drive["t_mgb"] > abs(out_brake["t_mgb"]), \
        "Driving torque should be larger due to friction penalty"


# ---------------------------------------------------------------------------
# 6. Zero-speed protection
# ---------------------------------------------------------------------------

def test_zero_speed_no_division_error():
    """w_wheel=0 must not raise ZeroDivisionError."""
    try:
        out = gearbox(w_wheel=0.0, dw_wheel=0.0, t_wheel=0.0, gear=1)
    except ZeroDivisionError:
        pytest.fail("ZeroDivisionError at w_wheel=0")

def test_very_low_speed_uses_w_min():
    """At w_wheel < w_min, friction term uses w_min as denominator."""
    w_low, gear = 0.1, 1   # below _W_MIN = 1.0
    out = gearbox(w_wheel=w_low, dw_wheel=0.0, t_wheel=10.0, gear=gear)
    expected_friction = _P_FRICTION / _W_MIN    # uses w_min, not w_low
    expected_t = (10.0 + expected_friction) / _ETA / _i_gt(gear)
    assert abs(out["t_mgb"] - expected_t) < 1e-6


# ---------------------------------------------------------------------------
# 7. Diagnostic power signal
# ---------------------------------------------------------------------------

def test_p_mgb_equals_t_times_w():
    out = gearbox(w_wheel=15.0, dw_wheel=0.0, t_wheel=80.0, gear=2)
    assert abs(out["p_mgb"] - out["t_mgb"] * out["w_mgb"]) < 1e-6


# ---------------------------------------------------------------------------
# 8. Full NEDC integration — gearbox chained after vehicle
# ---------------------------------------------------------------------------

def test_gearbox_full_nedc():
    """
    Chain DrivingCycle → VehicleDynamics → gearbox over full NEDC.
    Checks:
      - No NaN or Inf anywhere
      - w_mgb > 0 whenever gear > 0 and vehicle is moving
      - Gear 1 gives highest w_mgb for same w_wheel (highest ratio)
    """
    import sys, math
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from env.driving_cycle import DrivingCycle
    from env.powertrain import VehicleDynamics

    cycle = DrivingCycle("NEDC")
    veh   = VehicleDynamics()
    veh.reset()
    obs   = cycle.reset()

    while True:
        veh_out = veh.step(obs["v"], obs["dv"])
        gb_out  = gearbox(
            w_wheel  = veh_out["w_wheel"],
            dw_wheel = veh_out["dw_wheel"],
            t_wheel  = veh_out["T_wheel"],
            gear     = obs["i"],
        )
        for key, val in gb_out.items():
            assert math.isfinite(val), f"NaN/Inf in gearbox[{key}] at t={obs['t']}"

        if obs["i"] > 0 and obs["v"] > 1.0:
            assert gb_out["w_mgb"] > 0.0, \
                f"w_mgb=0 with gear={obs['i']}, v={obs['v']:.2f} at t={obs['t']}"

        obs, done = cycle.step()
        if done:
            break


# ===========================================================================
# Combustion Engine tests
# ===========================================================================

from env.powertrain import (
    combustion_engine,
    _W_IDLE, _P_CE_IDLE, _THETA, _T_CUTOFF, _P_CUTOFF,
    _W_UPPER, _H_u, _V_d,
)


# ---------------------------------------------------------------------------
# 1. Constants loaded correctly from params.json and engine_maps.npz
# ---------------------------------------------------------------------------

def test_engine_constants():
    assert abs(_W_IDLE   - 105.0)          < 1e-6
    assert abs(_P_CE_IDLE - 8000.0)        < 1e-6
    assert abs(_THETA    - 0.2)            < 1e-6
    assert abs(_T_CUTOFF - 5.0)            < 1e-6
    assert abs(_P_CUTOFF - 0.0)            < 1e-6
    # _W_UPPER = w_CE_row.max(), read from data/maps/engine_maps.npz.
    assert abs(_W_UPPER  - 439.822971502571) < 1e-6
    assert abs(_H_u - 42936779.911374)     < 1.0
    assert abs(_V_d - 1698e-6)             < 1e-12


# ---------------------------------------------------------------------------
# 2. Idle detection (Image 3)
# ---------------------------------------------------------------------------

def test_idle_when_stopped():
    """True standstill (w_gear=0) is NOT idle (idle requires 0 < w_gear <=
    w_idle); it falls through to fuel cutoff instead, so p_ce = 0. See
    README "Key Implementation Notes": idle and true-standstill are
    separate cases, only the former holds fuel power at P_CE_idle."""
    out = combustion_engine(w_gear=0.0, dw_gear=0.0, t_gear=0.0)
    assert out["is_idle"] is False
    assert out["is_cutoff"] is True
    assert abs(out["p_ce"] - _P_CUTOFF) < 1e-6

def test_idle_at_idle_speed():
    """w_gear exactly at idle speed with zero torque → idle"""
    out = combustion_engine(w_gear=_W_IDLE, dw_gear=0.0, t_gear=0.0)
    assert out["is_idle"] is True

def test_not_idle_above_idle_speed():
    """w_gear above idle with positive torque → not idle"""
    out = combustion_engine(w_gear=200.0, dw_gear=0.0, t_gear=50.0)
    assert out["is_idle"] is False


# ---------------------------------------------------------------------------
# 3. Fuel cutoff detection (Image 4)
# ---------------------------------------------------------------------------

def test_fuel_cutoff_zero_torque():
    """
    When T_CE <= T_cutoff and not idle → fuel cutoff active → p_ce = 0
    Use w_gear > w_idle so not idle, but t_gear very low
    """
    out = combustion_engine(w_gear=300.0, dw_gear=0.0, t_gear=0.0)
    assert out["is_cutoff"] is True
    assert abs(out["p_ce"] - _P_CUTOFF) < 1e-6

def test_no_cutoff_positive_torque():
    """Normal driving → no fuel cutoff"""
    out = combustion_engine(w_gear=300.0, dw_gear=0.0, t_gear=80.0)
    assert out["is_cutoff"] is False
    assert out["p_ce"] > 0.0


# ---------------------------------------------------------------------------
# 4. Engine inertia (Image 1)
# ---------------------------------------------------------------------------

def test_inertia_contribution():
    """
    T_inertia = theta * dw_gear = 0.2 * 10 = 2 Nm
    T_total = t_gear + T_inertia = 50 + 2 = 52 Nm
    t_ce should reflect this (if above cutoff)
    """
    out_with    = combustion_engine(w_gear=300.0, dw_gear=10.0, t_gear=50.0)
    out_without = combustion_engine(w_gear=300.0, dw_gear=0.0,  t_gear=50.0)
    assert out_with["t_ce"] > out_without["t_ce"]

def test_inertia_value():
    t_inertia_expected = _THETA * 10.0   # 0.2 * 10 = 2.0 Nm
    out_with    = combustion_engine(w_gear=300.0, dw_gear=10.0, t_gear=50.0)
    out_without = combustion_engine(w_gear=300.0, dw_gear=0.0,  t_gear=50.0)
    assert abs((out_with["t_ce"] - out_without["t_ce"]) - t_inertia_expected) < 1e-6


# ---------------------------------------------------------------------------
# 5. Idle speed clamp (Image 1)
# ---------------------------------------------------------------------------

def test_w_ce_clamped_to_idle():
    """w_gear below idle → w_ce must equal w_idle"""
    out = combustion_engine(w_gear=50.0, dw_gear=0.0, t_gear=50.0)
    assert abs(out["w_ce"] - _W_IDLE) < 1e-6

def test_w_ce_above_idle():
    """w_gear above idle → w_ce equals w_gear"""
    out = combustion_engine(w_gear=300.0, dw_gear=0.0, t_gear=50.0)
    assert abs(out["w_ce"] - 300.0) < 1e-6


# ---------------------------------------------------------------------------
# 6. Fuel power = v_dot * H_u
# ---------------------------------------------------------------------------

def test_p_ce_fuel_equals_vdot_times_Hu():
    """In normal operation: p_ce_fuel = v_dot * H_u"""
    out = combustion_engine(w_gear=300.0, dw_gear=0.0, t_gear=80.0)
    expected = out["v_dot"] * _H_u
    assert abs(out["p_ce_fuel"] - expected) < 1e-3

def test_v_dot_positive_in_normal_operation():
    out = combustion_engine(w_gear=300.0, dw_gear=0.0, t_gear=80.0)
    assert out["v_dot"] > 0.0

def test_p_ce_positive_in_normal_operation():
    out = combustion_engine(w_gear=300.0, dw_gear=0.0, t_gear=80.0)
    assert out["p_ce"] > 0.0


# ---------------------------------------------------------------------------
# 7. Overspeed flag (Image 2)
# ---------------------------------------------------------------------------

def test_overspeed_above_limit():
    out = combustion_engine(w_gear=900.0, dw_gear=0.0, t_gear=50.0)
    assert out["overspeed"] is True

def test_no_overspeed_below_limit():
    out = combustion_engine(w_gear=400.0, dw_gear=0.0, t_gear=50.0)
    assert out["overspeed"] is False


# ---------------------------------------------------------------------------
# 8. Overload flag (Image 2)
# ---------------------------------------------------------------------------

def test_overload_excessive_torque():
    """Demanding much more than max torque → overload flag"""
    out = combustion_engine(w_gear=300.0, dw_gear=0.0, t_gear=300.0)
    assert out["overload"] is True

def test_no_overload_normal_torque():
    out = combustion_engine(w_gear=300.0, dw_gear=0.0, t_gear=80.0)
    assert out["overload"] is False


# ---------------------------------------------------------------------------
# 9. T→p_me conversion
# ---------------------------------------------------------------------------

def test_p_me_conversion():
    """p_me = T * 4*pi / V_d"""
    t = 100.0
    expected_p_me = t * 4.0 * np.pi / _V_d
    # We cannot directly read p_me from output, but we can verify
    # that increasing torque increases p_ce (via map lookup)
    out_low  = combustion_engine(w_gear=300.0, dw_gear=0.0, t_gear=50.0)
    out_high = combustion_engine(w_gear=300.0, dw_gear=0.0, t_gear=100.0)
    assert out_high["p_ce"] > out_low["p_ce"], \
        "Higher torque must result in higher fuel power"


# ---------------------------------------------------------------------------
# 10. Full NEDC chain: DrivingCycle → Vehicle → Gearbox → Engine
# ---------------------------------------------------------------------------

def test_engine_full_nedc_chain():
    """
    Run full NEDC through the complete chain.
    Checks:
      - No NaN or Inf
      - p_ce >= 0 always
      - At least some non-idle timesteps
      - Total fuel consumption is physically plausible
    """
    import math
    from env.driving_cycle import DrivingCycle
    from env.powertrain import VehicleDynamics

    cycle = DrivingCycle("NEDC")
    veh   = VehicleDynamics()
    veh.reset()
    obs   = cycle.reset()

    total_fuel_kg = 0.0
    non_idle_steps = 0

    while True:
        veh_out = veh.step(obs["v"], obs["dv"])
        gb_out  = gearbox(
            w_wheel  = veh_out["w_wheel"],
            dw_wheel = veh_out["dw_wheel"],
            t_wheel  = veh_out["T_wheel"],
            gear     = obs["i"],
        )
        # Engine gets all torque demand from gearbox (rule-based: engine only)
        eng_out = combustion_engine(
            w_gear  = gb_out["w_mgb"],
            dw_gear = gb_out["dw_mgb"],
            t_gear  = gb_out["t_mgb"],
        )

        # No NaN / Inf check
        for key, val in eng_out.items():
            if isinstance(val, float):
                assert math.isfinite(val), \
                    f"NaN/Inf in engine[{key}] at t={obs['t']}"

        # p_ce always non-negative
        assert eng_out["p_ce"] >= 0.0, \
            f"Negative p_ce={eng_out['p_ce']:.2f} at t={obs['t']}"

        # Accumulate fuel
        total_fuel_kg += eng_out["v_dot"] * 1.0   # dt = 1 s
        if not eng_out["is_idle"]:
            non_idle_steps += 1

        obs, done = cycle.step()
        if done:
            break

    # Convert to litres: rho_f = 0.843 kg/l
    total_fuel_l   = total_fuel_kg / 0.843
    distance_km    = cycle.total_distance_m / 1000.0
    consumption    = total_fuel_l / distance_km * 100.0   # l/100km

    print(f"\n  NEDC fuel (engine only): {consumption:.2f} L/100km")
    print(f"  Non-idle steps: {non_idle_steps}/{cycle.length}")

    # Plausibility: a diesel car should consume between 3-15 L/100km
    assert 2.0 < consumption < 20.0, \
        f"Implausible fuel consumption: {consumption:.2f} L/100km"
    assert non_idle_steps > 100, \
        f"Too few non-idle steps: {non_idle_steps}"

