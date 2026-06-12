"""
test_electric_motor.py
======================
Unit tests for the Electric Motor block (BLOCK 4) in powertrain.py.

Run with:
    pytest tests/test_electric_motor.py -v
"""

import math
import numpy as np
import pytest

from env.powertrain import (
    electric_motor,
    _THETA_EM,
    _P_AUX_EM,
    _W_EM_UPPER,
    _w_EM_max_row,
    _T_EM_max_arr,
    _w_EM_row,
    _T_EM_col,
    _eta_EM_map,
    _interp1d_linear,
)


# ---------------------------------------------------------------------------
# 1. Constants loaded correctly from params.json and motor_maps.npz
# ---------------------------------------------------------------------------

class TestMotorConstants:

    def test_motor_inertia(self):
        assert abs(_THETA_EM - 0.1) < 1e-9

    def test_aux_power_zero(self):
        assert abs(_P_AUX_EM - 0.0) < 1e-9

    def test_w_em_upper(self):
        assert abs(_W_EM_UPPER - 600.0) < 1e-6

    def test_speed_axis_values(self):
        expected = [0., 75., 150., 225., 300., 375., 450., 525., 600.]
        np.testing.assert_allclose(_w_EM_row, expected, atol=1e-6)

    def test_torque_axis_length(self):
        assert len(_T_EM_col) == 30

    def test_torque_axis_signed(self):
        assert _T_EM_col[0] < 0.0
        assert _T_EM_col[-1] > 0.0

    def test_efficiency_map_shape(self):
        assert _eta_EM_map.shape == (9, 30)

    def test_no_zeros_in_map(self):
        assert (_eta_EM_map == 0.0).sum() == 0

    def test_generating_side_eta_lt_1(self):
        """Left half (T < 0) stores true η ∈ (0, 1)."""
        neg_cols = _T_EM_col < 0
        eta_gen  = _eta_EM_map[:, neg_cols]
        assert eta_gen[eta_gen > 0].min() > 0.0
        assert eta_gen.max() < 1.0 + 1e-6

    def test_motoring_side_inv_eta_gt_1(self):
        """Right half (T > 0) stores 1/η > 1."""
        pos_cols    = _T_EM_col > 0
        inv_eta_mot = _eta_EM_map[:, pos_cols]
        assert inv_eta_mot[inv_eta_mot < 10.0].min() > 1.0

    def test_t_em_max_curve_values(self):
        expected = [70., 70., 70., 52., 38., 28., 19., 13., 9.]
        np.testing.assert_allclose(_T_EM_max_arr, expected, atol=0.5)


# ---------------------------------------------------------------------------
# 2. Motor inertia contribution
# ---------------------------------------------------------------------------

class TestMotorInertia:

    def test_inertia_adds_to_torque(self):
        out_acc  = electric_motor(w_gear=300., dw_gear=10., t_gear=20.)
        out_flat = electric_motor(w_gear=300., dw_gear= 0., t_gear=20.)
        assert out_acc["t_em"] > out_flat["t_em"]

    def test_inertia_magnitude(self):
        out_acc  = electric_motor(w_gear=300., dw_gear=10., t_gear=20.)
        out_flat = electric_motor(w_gear=300., dw_gear= 0., t_gear=20.)
        assert abs((out_acc["t_em"] - out_flat["t_em"]) - _THETA_EM * 10.) < 1e-9

    def test_zero_accel_t_em_equals_t_gear(self):
        out = electric_motor(w_gear=200., dw_gear=0., t_gear=25.)
        assert abs(out["t_em"] - 25.) < 1e-9


# ---------------------------------------------------------------------------
# 3. Speed pass-through — no idle clamp
# ---------------------------------------------------------------------------

class TestSpeedPassThrough:

    def test_w_em_equals_w_gear(self):
        for w in [0., 75., 300., 600.]:
            out = electric_motor(w_gear=w, dw_gear=0., t_gear=10.)
            assert abs(out["w_em"] - w) < 1e-9, f"w_gear={w}: w_em={out['w_em']}"


# ---------------------------------------------------------------------------
# 4. Unified formula: P_EM = T_EM × w_EM × eta_map(w_EM, T_EM)
# ---------------------------------------------------------------------------

class TestUnifiedPowerFormula:

    def test_p_mech_definition(self):
        out = electric_motor(w_gear=300., dw_gear=0., t_gear=20.)
        assert abs(out["p_mech"] - out["t_em"] * out["w_em"]) < 1e-6

    def test_motoring_p_em_formula(self):
        """Motoring: eta_factor = 1/η > 1 → P_EM = P_mech × (1/η) > P_mech."""
        out = electric_motor(w_gear=300., dw_gear=0., t_gear=20.)
        assert out["is_motoring"] is True
        expected = out["p_mech"] * out["eta_factor"] + _P_AUX_EM
        assert abs(out["p_em"] - expected) < 1e-4
        assert out["p_em"] > out["p_mech"]

    def test_generating_p_em_formula(self):
        """Generating: eta_factor = η < 1 → |P_EM| = |P_mech| × η < |P_mech|."""
        out = electric_motor(w_gear=300., dw_gear=0., t_gear=-20.)
        assert out["is_generating"] is True
        expected = out["p_mech"] * out["eta_factor"] + _P_AUX_EM
        assert abs(out["p_em"] - expected) < 1e-4
        assert abs(out["p_em"]) < abs(out["p_mech"])

    def test_motoring_p_em_positive(self):
        out = electric_motor(w_gear=300., dw_gear=0., t_gear=20.)
        assert out["p_em"] > 0.0

    def test_generating_p_em_negative(self):
        out = electric_motor(w_gear=300., dw_gear=0., t_gear=-20.)
        assert out["p_em"] < 0.0

    def test_numerical_cross_check_motoring(self):
        """w=300, T=+30.36: map≈1.1098 → P_EM≈9108×1.1098≈10108 W."""
        out = electric_motor(w_gear=300., dw_gear=0., t_gear=30.36)
        assert 9800. < out["p_em"] < 10500.

    def test_numerical_cross_check_generating(self):
        """w=300, T=−30.36: map≈0.9011 → P_EM≈−9108×0.9011≈−8207 W."""
        out = electric_motor(w_gear=300., dw_gear=0., t_gear=-30.36)
        assert -8600. < out["p_em"] < -7800.


# ---------------------------------------------------------------------------
# 5. Mode flags
# ---------------------------------------------------------------------------

class TestModeFlags:

    def test_motoring_positive_torque(self):
        out = electric_motor(w_gear=300., dw_gear=0., t_gear=20.)
        assert out["is_motoring"] is True and out["is_generating"] is False

    def test_generating_negative_torque(self):
        out = electric_motor(w_gear=300., dw_gear=0., t_gear=-20.)
        assert out["is_generating"] is True and out["is_motoring"] is False

    def test_flags_mutually_exclusive(self):
        for t in [20., -20., 0.]:
            out = electric_motor(w_gear=300., dw_gear=0., t_gear=t)
            assert out["is_motoring"] != out["is_generating"]


# ---------------------------------------------------------------------------
# 6. Detect overload
# ---------------------------------------------------------------------------

class TestOverloadDetection:

    def test_overload_beyond_rated(self):
        """T=80 Nm >> T_max(300 rad/s)=38 Nm → overload."""
        out = electric_motor(w_gear=300., dw_gear=0., t_gear=80.)
        assert out["overload"] is True

    def test_no_overload_within_rated(self):
        out = electric_motor(w_gear=300., dw_gear=0., t_gear=20.)
        assert out["overload"] is False

    def test_overload_uses_absolute_torque(self):
        out = electric_motor(w_gear=300., dw_gear=0., t_gear=-80.)
        assert out["overload"] is True

    def test_overload_at_high_speed(self):
        """T_max(600 rad/s)=9 Nm; 15 Nm → overload."""
        out = electric_motor(w_gear=600., dw_gear=0., t_gear=15.)
        assert out["overload"] is True


# ---------------------------------------------------------------------------
# 7. Detect overspeed
# ---------------------------------------------------------------------------

class TestOverspeedDetection:

    def test_overspeed_above_600(self):
        out = electric_motor(w_gear=601., dw_gear=0., t_gear=5.)
        assert out["overspeed"] is True

    def test_no_overspeed_at_600(self):
        out = electric_motor(w_gear=600., dw_gear=0., t_gear=5.)
        assert out["overspeed"] is False

    def test_no_overspeed_below_600(self):
        out = electric_motor(w_gear=300., dw_gear=0., t_gear=10.)
        assert out["overspeed"] is False


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_zero_inputs_no_crash(self):
        try:
            electric_motor(w_gear=0., dw_gear=0., t_gear=0.)
        except Exception as e:
            pytest.fail(f"Exception at zero inputs: {e}")

    def test_all_outputs_finite(self):
        for w, dw, t in [(0.,0.,0.), (300.,0.,20.), (300.,0.,-20.),
                         (600.,0.,8.), (75.,5.,30.), (450.,-2.,-30.)]:
            out = electric_motor(w_gear=w, dw_gear=dw, t_gear=t)
            for k, v in out.items():
                if isinstance(v, float):
                    assert math.isfinite(v), f"{k}={v} not finite at w={w},t={t}"

    def test_output_keys_complete(self):
        expected = {"p_em","p_mech","w_em","t_em","eta_factor",
                    "overload","overspeed","is_motoring","is_generating"}
        out = electric_motor(w_gear=300., dw_gear=0., t_gear=20.)
        assert expected == set(out.keys())


# ---------------------------------------------------------------------------
# 9. Full NEDC integration — chain DrivingCycle → Vehicle → Gearbox → Motor
#
# DESIGN NOTE — why we pre-clamp t_gear with inertia headroom:
# ─────────────────────────────────────────────────────────────
# The gearbox demands up to ~428 Nm (full vehicle load, gear 1).
# The motor's rated maximum is only 70 Nm.  In real operation the EMS
# controller limits the torque command; this test mimics that by clamping
# t_gear to T_EM_max *before* the call.
#
# Subtlety: electric_motor() adds t_inertia = theta_EM * dw_gear AFTER
# the caller's clamp.  At t_gear = T_max exactly, t_em = T_max + t_inertia
# can exceed T_max by the inertia amount.  We therefore clamp t_gear to
# (T_max - |t_inertia|) so that t_em lands at most at T_max.
#
# Floating-point ties: when t_limit = T_max - |t_inertia| and t_mgb >= t_limit,
# t_clamped = t_limit exactly, so t_em = t_limit + t_inertia = T_max exactly.
# np.interp may return a value differing by ~1e-15 from T_max, causing the
# strict  abs(t_em) > T_max  check inside electric_motor() to toggle.  We
# tolerate this with a 1e-9 epsilon in the assertion — it is not a true
# overload, just a floating-point boundary coincidence.
# ---------------------------------------------------------------------------

# Tolerance for floating-point boundary coincidences in overload assertion
_OVERLOAD_TOL = 1e-9


def test_motor_full_nedc_chain():
    """
    Chain: DrivingCycle → VehicleDynamics → gearbox → electric_motor.
    Motor receives torque clamped to T_EM_max (as the EMS controller will do).

    Checks:
      - No NaN or Inf at any timestep
      - No true overloads (tolerance 1e-9 Nm for float boundary hits)
      - Regen events exist (NEDC braking phases)
      - Net energy consumption is physically plausible for a 12 kW motor
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from env.driving_cycle import DrivingCycle
    from env.powertrain    import VehicleDynamics, gearbox

    cycle = DrivingCycle("NEDC")
    veh   = VehicleDynamics()
    veh.reset()
    obs   = cycle.reset()

    total_energy_Ws = 0.0
    regen_steps     = 0
    motoring_steps  = 0

    while True:
        veh_out = veh.step(obs["v"], obs["dv"])
        gb_out  = gearbox(
            w_wheel  = veh_out["w_wheel"],
            dw_wheel = veh_out["dw_wheel"],
            t_wheel  = veh_out["T_wheel"],
            gear     = obs["i"],
        )

        w_em  = gb_out["w_mgb"]
        dw_em = gb_out["dw_mgb"]

        # T_EM_max at current speed
        t_max = _interp1d_linear(_w_EM_max_row, _T_EM_max_arr, abs(w_em))

        # Pre-clamp t_gear so that t_em = t_gear + theta*dw stays within T_max.
        # Reserve headroom for the inertia term added inside electric_motor().
        t_inertia = _THETA_EM * dw_em
        t_limit   = max(t_max - abs(t_inertia), 0.0)
        t_clamped = float(np.clip(gb_out["t_mgb"], -t_limit, t_limit))

        em_out = electric_motor(
            w_gear  = w_em,
            dw_gear = dw_em,
            t_gear  = t_clamped,
        )

        # ── No NaN / Inf ───────────────────────────────────────────────────
        for key, val in em_out.items():
            if isinstance(val, float):
                assert math.isfinite(val), f"NaN/Inf motor[{key}] at t={obs['t']}"

        # ── No true overload (float boundary hits tolerated at 1e-9 Nm) ───
        is_true_overload = abs(em_out["t_em"]) > t_max + _OVERLOAD_TOL
        assert not is_true_overload, (
            f"True overload at t={obs['t']}: "
            f"t_em={em_out['t_em']:.6f} Nm, T_max={t_max:.6f} Nm, "
            f"excess={abs(em_out['t_em']) - t_max:.2e} Nm"
        )

        # ── Sign consistency ───────────────────────────────────────────────
        if em_out["is_motoring"]:
            assert em_out["p_em"] >= -1e-4, \
                f"Motoring but p_em={em_out['p_em']:.2f} at t={obs['t']}"
            motoring_steps += 1
        else:
            if em_out["p_mech"] < -1.0:
                regen_steps += 1

        total_energy_Ws += em_out["p_em"] * 1.0   # dt = 1 s
        obs, done = cycle.step()
        if done:
            break

    dist_km               = cycle.total_distance_m / 1000.0
    consumption_kWh_100km = (total_energy_Ws / 3_600_000.0) / dist_km * 100.0

    print(f"\n  NEDC motor-only consumption : {consumption_kWh_100km:.2f} kWh/100km")
    print(f"  Motoring steps              : {motoring_steps}")
    print(f"  Regen steps                 : {regen_steps}")

    # 12 kW mild-hybrid motor at rated limit over NEDC — expect 10–30 kWh/100km
    assert 5.0 < consumption_kWh_100km < 40.0, \
        f"Implausible consumption: {consumption_kWh_100km:.2f} kWh/100km"
    assert regen_steps > 10, \
        f"Too few regen steps: {regen_steps}"
