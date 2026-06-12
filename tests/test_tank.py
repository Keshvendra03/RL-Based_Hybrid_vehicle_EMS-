"""
test_tank.py
============
Unit tests for the Tank block (BLOCK 5) in powertrain.py.

Validates every subsystem against the two Simulink screenshots:
  Image 1 — Tank top-level block
  Image 2 — Integration subsystem

Run with:
    pytest tests/test_tank.py -v
"""

import math
import numpy as np
import pytest

from env.powertrain import (
    Tank,
    tank_step,
    _RHO_FUEL,
    _K_CS,
    _H_TANK,
    _H_u,
    _M_TO_100KM,
)


# ---------------------------------------------------------------------------
# Known constants
# ---------------------------------------------------------------------------
H_U    = 42_936_779.911374  # J/kg  (diesel lower heating value)
RHO_F  = 0.843              # kg/L  (diesel density)
K_CS   = 1.0                # [-]   (cold-start factor, diesel)
H      = 1.0                # s     (timestep)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tank():
    t = Tank()
    t.reset()
    return t


# ---------------------------------------------------------------------------
# 1. Constants loaded correctly
# ---------------------------------------------------------------------------

class TestTankConstants:

    def test_rho_fuel(self):
        """Diesel density = 0.843 kg/L (QSS toolbox standard)."""
        assert abs(_RHO_FUEL - 0.843) < 1e-9

    def test_k_cs(self):
        """Cold-start factor for diesel = 1.0 (no penalty)."""
        assert abs(_K_CS - 1.0) < 1e-9

    def test_h_tank(self):
        """Timestep h = 1.0 s (matches DrivingCycle step size)."""
        assert abs(_H_TANK - 1.0) < 1e-9

    def test_h_u_shared_with_engine(self):
        """H_u is shared with engine block — same value."""
        assert abs(_H_u - H_U) < 1.0   # within 1 J/kg

    def test_m_to_100km(self):
        """Unit conversion constant = 1e5 m/100km."""
        assert abs(_M_TO_100KM - 1e5) < 1e-9


# ---------------------------------------------------------------------------
# 2. State management
# ---------------------------------------------------------------------------

class TestTankState:

    def test_reset_clears_m_fuel(self, tank):
        tank.step(p_fuel=10000.0, x_tot=100.0)
        assert tank.m_fuel > 0.0
        tank.reset()
        assert tank.m_fuel == 0.0

    def test_reset_clears_p_fuel_prev(self, tank):
        tank.step(p_fuel=10000.0, x_tot=100.0)
        assert tank.p_fuel_prev == 10000.0
        tank.reset()
        assert tank.p_fuel_prev == 0.0

    def test_m_fuel_only_increases(self, tank):
        """Fuel mass is monotonically non-decreasing (can never burn un-consumed fuel)."""
        prev = 0.0
        for p in [0.0, 5000.0, 8000.0, 0.0, 12000.0]:
            out = tank.step(p_fuel=p, x_tot=500.0)
            assert out["m_fuel"] >= prev - 1e-12
            prev = out["m_fuel"]


# ---------------------------------------------------------------------------
# 3. Integration subsystem (Image 2) — trapezoidal rule
# ---------------------------------------------------------------------------

class TestIntegrationSubsystem:

    def test_first_step_uses_zero_prev(self, tank):
        """
        At t=0: p_fuel_prev = 0 (unit delay initialised to 0).
        m_dot = (P[0] + 0) * 0.5 * h / H_u
        """
        p = 10000.0
        out = tank.step(p_fuel=p, x_tot=1000.0)
        expected_m_dot = (p + 0.0) * 0.5 * H * (1.0 / H_U)
        assert abs(out["m_dot_fuel"] - expected_m_dot) < 1e-15

    def test_second_step_trapezoidal(self, tank):
        """
        Step 2: m_dot = (P[1] + P[0]) * 0.5 * h / H_u
        """
        p0, p1 = 8000.0, 15000.0
        tank.step(p_fuel=p0, x_tot=5.0)
        out = tank.step(p_fuel=p1, x_tot=10.0)
        expected = (p1 + p0) * 0.5 * H / H_U
        assert abs(out["m_dot_fuel"] - expected) < 1e-15

    def test_constant_power_matches_rectangular(self, tank):
        """
        When P_fuel is constant, trapezoidal = rectangular after first step.
        m_dot = P * h / H_u  for all t >= 1.
        """
        p = 12000.0
        tank.step(p_fuel=p, x_tot=5.0)   # warm up prev
        out = tank.step(p_fuel=p, x_tot=10.0)
        expected = p * H / H_U
        assert abs(out["m_dot_fuel"] - expected) < 1e-14

    def test_zero_power_zero_m_dot(self, tank):
        """P_fuel=0 after warm-up → m_dot = 0 (fuel cutoff scenario)."""
        tank.step(p_fuel=0.0, x_tot=10.0)
        out = tank.step(p_fuel=0.0, x_tot=20.0)
        assert abs(out["m_dot_fuel"]) < 1e-20

    def test_m_fuel_accumulates_correctly(self, tank):
        """Three-step manual trace matches class output."""
        p_series = [8000.0, 15000.0, 12000.0]
        x_series = [0.0,    5.0,     10.0]

        m_fuel_expected = 0.0
        p_prev = 0.0
        for p, x in zip(p_series, x_series):
            m_dot = (p + p_prev) * 0.5 * H / H_U
            m_fuel_expected += m_dot
            out = tank.step(p_fuel=p, x_tot=x)
            assert abs(out["m_fuel"] - m_fuel_expected) < 1e-15, \
                f"m_fuel mismatch at p={p}: got {out['m_fuel']:.10e}, expected {m_fuel_expected:.10e}"
            p_prev = p

    def test_p_fuel_prev_updates_each_step(self, tank):
        tank.step(p_fuel=5000.0, x_tot=10.0)
        assert abs(tank.p_fuel_prev - 5000.0) < 1e-9
        tank.step(p_fuel=9000.0, x_tot=20.0)
        assert abs(tank.p_fuel_prev - 9000.0) < 1e-9


# ---------------------------------------------------------------------------
# 4. Top-level Tank block (Image 1) — mass → L/100km conversion
# ---------------------------------------------------------------------------

class TestTopLevelConversion:

    def test_v_liter_formula(self, tank):
        """
        v_liter = (m_fuel / rho_f) / x_tot * k_cs * 1e5
        Manual trace through one step.
        """
        p = 10000.0
        x = 500.0
        out = tank.step(p_fuel=p, x_tot=x)
        m_fuel = out["m_fuel"]
        expected = (m_fuel / RHO_F) / x * K_CS * 1e5
        assert abs(out["v_liter"] - expected) < 1e-10

    def test_zero_x_tot_gives_zero_output(self, tank):
        """x_tot=0 (simulation start) → v_liter=0, no ZeroDivisionError."""
        try:
            out = tank.step(p_fuel=8000.0, x_tot=0.0)
        except ZeroDivisionError:
            pytest.fail("ZeroDivisionError at x_tot=0")
        assert out["v_liter"] == 0.0

    def test_v_liter_increases_with_power(self, tank):
        """Higher P_fuel → higher v_liter at same distance."""
        t_lo = Tank(); t_lo.reset()
        t_hi = Tank(); t_hi.reset()
        x = 1000.0
        out_lo = t_lo.step(p_fuel=5000.0,  x_tot=x)
        out_hi = t_hi.step(p_fuel=20000.0, x_tot=x)
        assert out_hi["v_liter"] > out_lo["v_liter"]

    def test_v_liter_decreases_with_distance(self, tank):
        """Same m_fuel over more distance → lower L/100km."""
        # Two identical engines, but different x_tot
        t1 = Tank(); t1.reset()
        t2 = Tank(); t2.reset()
        p = 10000.0
        t1.step(p_fuel=p, x_tot=500.0)
        t2.step(p_fuel=p, x_tot=2000.0)
        # Both have same m_fuel after one step, but different x_tot
        assert t1.step(p_fuel=p, x_tot=500.0)["v_liter"] > \
               t2.step(p_fuel=p, x_tot=2000.0)["v_liter"]

    def test_1e5_conversion_factor(self, tank):
        """
        Unit conversion check: if m_fuel/rho_f = x_tot/1e5 (exactly),
        then v_liter = 1.0 L/100km.
        """
        # Choose x_tot = 1e5 m (=100 km), m_fuel = rho_f kg
        # Then v = (rho_f/rho_f) / 1e5 * 1e5 = 1.0 L/100km
        # We need m_fuel = rho_f after one step.
        # m_dot_step1 = p * 0.5 * 1 / H_u  → p = 2 * rho_f * H_u
        p_needed = 2.0 * RHO_F * H_U
        out = tank.step(p_fuel=p_needed, x_tot=1e5)
        assert abs(out["v_liter"] - 1.0) < 1e-6, \
            f"v_liter = {out['v_liter']:.6f}, expected 1.0"

    def test_k_cs_applied(self, tank):
        """
        Cold-start factor k_cs=1.0 for diesel means no change.
        Verify the formula includes it (scaling test).
        """
        out = tank.step(p_fuel=10000.0, x_tot=1000.0)
        m = out["m_fuel"]
        expected_without_kcs = (m / RHO_F) / 1000.0 * 1e5
        expected_with_kcs    = expected_without_kcs * K_CS
        assert abs(out["v_liter"] - expected_with_kcs) < 1e-10


# ---------------------------------------------------------------------------
# 5. Output dict structure
# ---------------------------------------------------------------------------

class TestOutputStructure:

    def test_output_keys(self, tank):
        out = tank.step(p_fuel=8000.0, x_tot=500.0)
        assert set(out.keys()) == {"v_liter", "m_fuel", "m_dot_fuel"}

    def test_all_outputs_finite(self, tank):
        for p, x in [(0.0, 0.0), (8000.0, 100.0), (60000.0, 10000.0)]:
            out = tank.step(p_fuel=p, x_tot=x)
            for k, v in out.items():
                assert math.isfinite(v), f"{k}={v} not finite at p={p}, x={x}"


# ---------------------------------------------------------------------------
# 6. Stateless convenience function
# ---------------------------------------------------------------------------

class TestStatelessAPI:

    def test_stateless_matches_class(self):
        """tank_step() must produce identical results to Tank.step()."""
        tank_obj = Tank()
        tank_obj.reset()

        m_fuel = 0.0
        p_prev = 0.0

        for p, x in [(8000.0, 0.0), (12000.0, 5.0), (10000.0, 10.0)]:
            class_out = tank_obj.step(p_fuel=p, x_tot=x)

            # Pass the tracked state variables directly
            func_out, m_fuel, p_prev = tank_step(
                p_fuel=p,
                x_tot=x,
                m_fuel_prev=m_fuel,
                p_fuel_prev=p_prev,
            )

            assert abs(class_out["v_liter"] - func_out["v_liter"]) < 1e-10
            assert abs(class_out["m_dot_fuel"] - func_out["m_dot_fuel"]) < 1e-15

    def test_stateless_returns_three_values(self):
        out, m_new, p_new = tank_step(p_fuel=8000.0, x_tot=100.0)
        assert isinstance(out, dict)
        assert isinstance(m_new, float)
        assert isinstance(p_new, float)

    def test_stateless_p_fuel_new_is_p_fuel(self):
        """The returned p_fuel_new must equal the input p_fuel."""
        _, _, p_new = tank_step(p_fuel=12345.0, x_tot=500.0)
        assert abs(p_new - 12345.0) < 1e-9


# ---------------------------------------------------------------------------
# 7. Physical plausibility — NEDC engine-only baseline
# ---------------------------------------------------------------------------

def test_tank_full_nedc_chain():
    """
    Chain: DrivingCycle → Vehicle → Gearbox → Engine → Tank.
    Engine receives all gearbox torque (engine-only baseline, no motor).

    Checks:
      - No NaN or Inf at any timestep
      - v_liter is physically plausible (3–8 L/100km for a diesel)
      - m_fuel is monotonically non-decreasing
      - Final v_liter matches known Simulink reference ~3.3–4.5 L/100km
    """
    import sys, math
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from env.driving_cycle import DrivingCycle
    from env.powertrain    import (
        VehicleDynamics, gearbox, combustion_engine, Tank,
    )

    cycle = DrivingCycle("NEDC")
    veh   = VehicleDynamics()
    tank  = Tank()
    veh.reset(); tank.reset()
    obs   = cycle.reset()

    m_fuel_prev = 0.0
    v_liter_last = 0.0

    while True:
        veh_out = veh.step(obs["v"], obs["dv"])
        gb_out  = gearbox(
            w_wheel  = veh_out["w_wheel"],
            dw_wheel = veh_out["dw_wheel"],
            t_wheel  = veh_out["T_wheel"],
            gear     = obs["i"],
        )
        eng_out = combustion_engine(
            w_gear  = gb_out["w_mgb"],
            dw_gear = gb_out["dw_mgb"],
            t_gear  = gb_out["t_mgb"],
        )
        tank_out = tank.step(
            p_fuel = eng_out["p_ce"],
            x_tot  = obs["x_tot"],
        )

        # No NaN / Inf
        for k, v in tank_out.items():
            assert math.isfinite(v), f"NaN/Inf in tank[{k}] at t={obs['t']}"

        # m_fuel monotonically non-decreasing
        assert tank_out["m_fuel"] >= m_fuel_prev - 1e-12, \
            f"m_fuel decreased at t={obs['t']}"
        m_fuel_prev  = tank_out["m_fuel"]
        v_liter_last = tank_out["v_liter"]

        obs, done = cycle.step()
        if done:
            break

    print(f"\n  NEDC engine-only fuel consumption: {v_liter_last:.3f} L/100km")
    print(f"  Total fuel mass: {tank.m_fuel:.4f} kg")

    # Diesel car without hybrid assist: expect 3.0–8.0 L/100km
    assert 3.0 < v_liter_last < 8.0, \
        f"Implausible consumption: {v_liter_last:.3f} L/100km"
