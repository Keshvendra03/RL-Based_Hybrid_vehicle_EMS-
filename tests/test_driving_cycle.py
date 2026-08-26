"""
test_driving_cycle.py
=====================
Unit tests for DrivingCycle — validates the Python block against
known reference values from the NEDC and FTP-75 standards.

Run with:
    pytest tests/test_driving_cycle.py -v
"""

import numpy as np
import pytest

# conftest.py at the project root adds src/ to sys.path automatically.
# PyCharm users: make sure pytest.ini is in the project root too.
from env.driving_cycle import DrivingCycle


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def nedc():
    return DrivingCycle("NEDC")


@pytest.fixture
def ftp75():
    return DrivingCycle("FTP75")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cycle length (number of timesteps)
# ─────────────────────────────────────────────────────────────────────────────

def test_nedc_length(nedc):
    # NEDC CSV carries one extra trailing row (time_s=length+1, v=0, dv=0,
    # gear=0) to match MATLAB's sample count; see README "Key Implementation
    # Notes". 1220 "real" rows + 1 trailing row = 1221.
    assert nedc.length == 1221, f"Expected 1221 timesteps, got {nedc.length}"


def test_ftp75_length(ftp75):
    assert ftp75.length == 1877, f"Expected 1877 timesteps, got {ftp75.length}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Total distance — matches standard cycle specs
# ─────────────────────────────────────────────────────────────────────────────

def test_nedc_total_distance(nedc):
    dist_km = nedc.total_distance_m / 1000
    assert abs(dist_km - 10.932) < 0.01, \
        f"NEDC distance {dist_km:.3f} km deviates from expected 10.932 km"


def test_ftp75_total_distance(ftp75):
    dist_km = ftp75.total_distance_m / 1000
    assert abs(dist_km - 17.767) < 0.01, \
        f"FTP75 distance {dist_km:.3f} km deviates from expected 17.767 km"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Maximum speed — matches standard cycle specs
# ─────────────────────────────────────────────────────────────────────────────

def test_nedc_max_speed(nedc):
    max_kmh = nedc.max_speed_ms * 3.6
    assert abs(max_kmh - 120.0) < 0.5, \
        f"NEDC max speed {max_kmh:.1f} km/h, expected 120.0 km/h"


def test_ftp75_max_speed(ftp75):
    max_kmh = ftp75.max_speed_ms * 3.6
    assert abs(max_kmh - 91.2) < 0.5, \
        f"FTP75 max speed {max_kmh:.1f} km/h, expected ~91.2 km/h"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gear values — only 0-5 allowed
# ─────────────────────────────────────────────────────────────────────────────

def test_nedc_gear_range(nedc):
    obs = nedc.reset()
    gears = [obs["i"]]
    while True:
        obs, done = nedc.step()
        gears.append(obs["i"])
        if done:
            break
    gears = np.array(gears)
    assert gears.min() >= 0,  f"Gear below 0 found: {gears.min()}"
    assert gears.max() <= 5,  f"Gear above 5 found: {gears.max()}"


def test_ftp75_gear_range(ftp75):
    obs = ftp75.reset()
    gears = [obs["i"]]
    while True:
        obs, done = ftp75.step()
        gears.append(obs["i"])
        if done:
            break
    gears = np.array(gears)
    assert gears.min() >= 0
    assert gears.max() <= 5


# ─────────────────────────────────────────────────────────────────────────────
# 5. Speed non-negative throughout
# ─────────────────────────────────────────────────────────────────────────────

def test_speed_non_negative(nedc):
    obs = nedc.reset()
    while True:
        assert obs["v"] >= 0.0, f"Negative speed at t={obs['t']}: {obs['v']}"
        obs, done = nedc.step()
        if done:
            break


# ─────────────────────────────────────────────────────────────────────────────
# 6. x_tot is monotonically non-decreasing
# ─────────────────────────────────────────────────────────────────────────────

def test_distance_monotonic(nedc):
    obs = nedc.reset()
    prev_x = obs["x_tot"]
    while True:
        obs, done = nedc.step()
        assert obs["x_tot"] >= prev_x - 1e-6, \
            f"Distance decreased at t={obs['t']}: {obs['x_tot']:.2f} < {prev_x:.2f}"
        prev_x = obs["x_tot"]
        if done:
            break


# ─────────────────────────────────────────────────────────────────────────────
# 7. Reset brings cycle back to t=0
# ─────────────────────────────────────────────────────────────────────────────

def test_reset_restores_state(nedc):
    nedc.reset()
    for _ in range(50):
        nedc.step()
    assert nedc.t == 50

    obs = nedc.reset()
    assert nedc.t == 0
    assert obs["t"] == 1        # time_s starts at 1 in CSV (Simulink convention)
    assert obs["x_tot"] == 0.0
    assert obs["v"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. step() returns done=True at the last timestep
# ─────────────────────────────────────────────────────────────────────────────

def test_done_flag(nedc):
    """reset() places the cycle at t=0 (not counted as a step); step()
    then advances t by 1 each call until t == length-1, so exactly
    length-1 step() calls are needed to first see done=True."""
    nedc.reset()
    done = False
    steps = 0
    while not done:
        _, done = nedc.step()
        steps += 1
    assert steps == nedc.length - 1, \
        f"Expected {nedc.length - 1} steps to complete, got {steps}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. obs dict has all required keys (matches Simulink port names)
# ─────────────────────────────────────────────────────────────────────────────

def test_obs_keys(nedc):
    obs = nedc.reset()
    required_keys = {"v", "dv", "i", "x_tot", "t"}
    assert required_keys == set(obs.keys()), \
        f"Obs keys mismatch. Got: {set(obs.keys())}"


# ─────────────────────────────────────────────────────────────────────────────
# 10. dv is consistent with finite difference of v
# ─────────────────────────────────────────────────────────────────────────────

def test_dv_consistency(nedc):
    """dv[t] should equal v[t+1] - v[t] for all but the last step."""
    nedc.reset()
    obs_prev = nedc.current
    tolerance = 1e-4

    for _ in range(nedc.length - 2):
        nedc.step()
        obs_curr = nedc.current
        expected_dv = obs_curr["v"] - obs_prev["v"]
        assert abs(obs_prev["dv"] - expected_dv) < tolerance, (
            f"dv inconsistency at t={obs_prev['t']}: "
            f"stored={obs_prev['dv']:.5f}, computed={expected_dv:.5f}"
        )
        obs_prev = obs_curr


# ─────────────────────────────────────────────────────────────────────────────
# 11. Invalid cycle name raises ValueError
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_cycle_name():
    with pytest.raises(ValueError, match="Unknown cycle"):
        DrivingCycle("WLTP")


# ─────────────────────────────────────────────────────────────────────────────
# 12. progress property
# ─────────────────────────────────────────────────────────────────────────────

def test_progress(nedc):
    nedc.reset()
    assert nedc.progress == 0.0

    for _ in range(nedc.length - 1):
        nedc.step()
    assert abs(nedc.progress - 1.0) < 1e-6
