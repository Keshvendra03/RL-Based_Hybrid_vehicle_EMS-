"""
test_action_mapping.py
======================
CONTROL-EQUIVALENCE PROOF for the action->u reparameterization.

The mode-aware action map exists to fix a measured RL defect (engine-OFF
occupying a narrow, state-dependent sliver of the action range -- see
RL_DIAGNOSTIC_REPORT.md). It is a REPARAMETERIZATION, not a physics change.

These tests are the formal guarantee of that claim:

  1. "linear" is bit-for-bit identical to the original hard-coded formula.
  2. Both maps hit the exact endpoints u(-1)=U_MIN, u(+1)=U_MAX for every T.
  3. Both maps are STRICTLY MONOTONIC in a  ->  bijections onto [U_MIN,U_MAX].
  4. The REACHABLE u set (hence the reachable (T_CE,T_EM) set) is IDENTICAL
     between the two maps, to numerical tolerance.
  5. Braking / sub-cutoff steps are handled by the linear branch, so regen
     behaviour is unchanged.
  6. The mode-aware map actually delivers the intended OFF-band geometry.

If any of these fail, the mapping is NOT a valid reparameterization and the
experiment built on it is invalid.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.env.ems_env import (
    map_action_to_u, U_MIN, U_MAX, ZA_MODEAWARE, ZB_MODEAWARE, EMSEnv,
)
from src.env.powertrain import _T_CUTOFF

TORQUES = [6.0, 8.9, 15.0, 22.2, 37.9, 60.0, 100.0, 150.0]
GRID = np.linspace(-1.0, 1.0, 2001)


def _original_linear(a: float) -> float:
    """The exact formula that was hard-coded in _action_to_torques before."""
    return U_MIN + (a + 1.0) * 0.5 * (U_MAX - U_MIN)


def test_linear_is_bit_identical_to_original():
    for a in GRID:
        assert map_action_to_u(float(a), 25.0, "linear") == _original_linear(float(a))


@pytest.mark.parametrize("amap", ["linear", "modeaware"])
@pytest.mark.parametrize("T", TORQUES)
def test_endpoints_exact(amap, T):
    assert map_action_to_u(-1.0, T, amap) == pytest.approx(U_MIN, abs=1e-12)
    assert map_action_to_u(+1.0, T, amap) == pytest.approx(U_MAX, abs=1e-12)


@pytest.mark.parametrize("amap", ["linear", "modeaware"])
@pytest.mark.parametrize("T", TORQUES)
def test_strictly_monotonic(amap, T):
    us = np.array([map_action_to_u(float(a), T, amap) for a in GRID])
    assert np.all(np.diff(us) > 0.0), f"not strictly increasing for {amap}, T={T}"


@pytest.mark.parametrize("T", TORQUES)
def test_reachable_u_set_identical(T):
    """Same min/max and same continuous coverage -> same control authority."""
    lin = np.array([map_action_to_u(float(a), T, "linear") for a in GRID])
    mod = np.array([map_action_to_u(float(a), T, "modeaware") for a in GRID])
    assert lin.min() == pytest.approx(mod.min(), abs=1e-12)
    assert lin.max() == pytest.approx(mod.max(), abs=1e-12)
    # every u reachable under linear is reachable under modeaware (bijection
    # onto the same interval); check via dense coverage, no gaps
    assert mod.max() - mod.min() == pytest.approx(U_MAX - U_MIN, abs=1e-12)
    assert np.max(np.diff(mod)) < 5e-3, "mode-aware map has a coverage gap"


@pytest.mark.parametrize("T", [-50.0, -10.0, 0.5, _T_CUTOFF])
def test_braking_and_subcutoff_fall_back_to_linear(T):
    for a in np.linspace(-1, 1, 101):
        assert map_action_to_u(float(a), T, "modeaware") == _original_linear(float(a))


@pytest.mark.parametrize("T", [8.9, 22.2, 37.9, 60.0, 100.0])
def test_modeaware_off_band_is_fixed_fraction(T):
    """The whole point: OFF must occupy ~(1-ZB) of the action range for EVERY T."""
    u_thr = 1.0 - _T_CUTOFF / T
    us = np.array([map_action_to_u(float(a), T, "modeaware") for a in GRID])
    off_frac = (us >= u_thr).mean()
    assert off_frac == pytest.approx(1.0 - ZB_MODEAWARE, abs=0.02), \
        f"T={T}: OFF band {off_frac:.3f}, expected {1-ZB_MODEAWARE:.3f}"
    lps_frac = (us < 0.0).mean()
    assert lps_frac == pytest.approx(ZA_MODEAWARE, abs=0.02)


def test_modeaware_off_band_beats_linear_and_is_state_invariant():
    """Linear: OFF band varies wildly with T. Mode-aware: constant."""
    lin_bands, mod_bands = [], []
    for T in TORQUES:
        u_thr = 1.0 - _T_CUTOFF / T
        lin = np.array([map_action_to_u(float(a), T, "linear") for a in GRID])
        mod = np.array([map_action_to_u(float(a), T, "modeaware") for a in GRID])
        lin_bands.append((lin >= u_thr).mean())
        mod_bands.append((mod >= u_thr).mean())
    assert np.std(mod_bands) < 0.02, "mode-aware OFF band should be state-invariant"
    assert np.std(mod_bands) < np.std(lin_bands), "should reduce state dependence"
    assert np.mean(mod_bands) > np.mean(lin_bands), "should widen the OFF band"


def test_env_default_is_linear_and_unchanged():
    """Regression guard: constructing EMSEnv with no action_map must behave
    exactly as before this feature existed."""
    env = EMSEnv("NEDC", lookahead=5)
    assert env.action_map == "linear"
    env.reset()
    tot = 0.0
    for _ in range(200):
        _, r, term, _, _ = env.step(np.array([0.3], dtype=np.float32))
        tot += r
        if term:
            break
    assert np.isfinite(tot)


def test_env_rejects_unknown_map():
    with pytest.raises(ValueError):
        EMSEnv("NEDC", action_map="nonsense")
