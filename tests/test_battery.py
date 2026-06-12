"""
test_battery.py
================
Unit tests for the Battery block (BLOCK 6) and Equivalent Fuel
Consumption block (BLOCK 7) in powertrain.py.

All constants and formulas were extracted directly from the Simulink
model "qss_hybrid_electric_vehicle_example" (Charging/Discharging Fcn
expressions, Idle gain/constant, Supervision Sum signs, Saturation
limits, and the "-C-" constant value).

Run with:
    pytest tests/test_battery.py -v
"""

import math
import numpy as np
import pytest

from env.powertrain import (
    Battery,
    battery_step,
    equivalent_fuel_consumption,
    _Q_BT_0,
    _Q_BT_IC,
    _I_0,
    _I_BT_MAX,
    _H_BATT,
    _C_BT_L1, _C_BT_L2, _C_BT_L3, _C_BT_L4,
    _C_BT_E1, _C_BT_E2, _C_BT_E3, _C_BT_E4,
    _V_BT_C_CONST,
    _u_oc,
    _ETA_BT, _ETA_EM, _ETA_CE,
    _EFC_GAIN,
    _SAT_LOWER, _SAT_UPPER,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batt():
    b = Battery()
    b.reset()
    return b


# ---------------------------------------------------------------------------
# 1. Constants — extracted directly from Simulink
# ---------------------------------------------------------------------------

class TestBatteryConstants:

    def test_q_bt_0(self):
        """Q_BT_0 = capacity_Ah * 3600 = 10 * 3600 = 36000 As."""
        assert abs(_Q_BT_0 - 36000.0) < 1e-6

    def test_q_bt_ic_50_percent(self):
        """Q_BT_IC = 18000 As = 50% of Q_BT_0 (matches Simulink workspace value)."""
        assert abs(_Q_BT_IC - 18000.0) < 1e-6
        assert abs(_Q_BT_IC / _Q_BT_0 - 0.5) < 1e-9

    def test_i_0(self):
        """I_0 = 10 A (from Gain '1/I_0' = 0.1)."""
        assert abs(_I_0 - 10.0) < 1e-9

    def test_i_bt_max(self):
        """I_BT_max = 300 A (Constant7 in Supervision battery)."""
        assert abs(_I_BT_MAX - 300.0) < 1e-9

    def test_h_batt(self):
        """Battery charge integrator timestep = 1.0 s."""
        assert abs(_H_BATT - 1.0) < 1e-9

    def test_charging_coefficients(self):
        assert abs(_C_BT_L1 - 39.0)  < 1e-9
        assert abs(_C_BT_L2 - 0.013) < 1e-9
        assert abs(_C_BT_L3 - 15.6)  < 1e-9
        assert abs(_C_BT_L4 - 0.0)   < 1e-9

    def test_discharging_coefficients(self):
        assert abs(_C_BT_E1 - 39.0)  < 1e-9
        assert abs(_C_BT_E2 - 0.013) < 1e-9
        assert abs(_C_BT_E3 - 15.6)  < 1e-9
        assert abs(_C_BT_E4 - 0.0)   < 1e-9

    def test_v_bt_c_const(self):
        """
        '-C-' constant = Q_BT_IC_rel/100 * c_BT_L3 + c_BT_L1
                       = 0.5 * 15.6 + 39 = 46.8
        Confirmed directly from the Simulink Constant1 block mask.
        """
        assert abs(_V_BT_C_CONST - 46.8) < 1e-6

    def test_efc_efficiencies(self):
        """From params.json equivalent_consumption section."""
        assert abs(_ETA_BT - 0.9)  < 1e-9
        assert abs(_ETA_EM - 0.8)  < 1e-9
        assert abs(_ETA_CE - 0.25) < 1e-9

    def test_efc_saturation_bounds(self):
        """Confirmed from Simulink Saturation block: Upper=inf, Lower=0."""
        assert _SAT_LOWER == 0.0
        assert _SAT_UPPER == np.inf

    def test_efc_gain(self):
        """
        EFC_GAIN = 1 / (eta_BT*eta_EM*eta_CE*(H_u/3.6e6)*rho_f)
        Numerically ~ 0.552552
        """
        assert abs(_EFC_GAIN - 0.552552) < 1e-5


# ---------------------------------------------------------------------------
# 2. State management
# ---------------------------------------------------------------------------

class TestBatteryState:

    def test_initial_soc_is_50_percent(self, batt):
        assert abs(batt.q_bt / _Q_BT_0 - 0.5) < 1e-9

    def test_reset_restores_initial_charge(self, batt):
        batt.step(p_bt=1000.0, x_tot=100.0)
        assert batt.q_bt != _Q_BT_IC
        batt.reset()
        assert batt.q_bt == _Q_BT_IC


# ---------------------------------------------------------------------------
# 3. Idle mode (P_BT == 0) — U_BT = U_oc(Q_BT)
# ---------------------------------------------------------------------------

class TestIdleMode:

    def test_idle_voltage_at_ic(self, batt):
        """At 50% SoC, idle U_BT = U_oc(Q_BT_IC) = 46.8 V exactly."""
        out = batt.step(p_bt=0.0, x_tot=10.0)
        assert abs(out["u_bt"] - 46.8) < 1e-6

    def test_idle_current_is_zero(self, batt):
        out = batt.step(p_bt=0.0, x_tot=10.0)
        assert abs(out["i_bt"]) < 1e-12

    def test_idle_q_bt_unchanged(self, batt):
        """I_BT=0 -> Q_BT[k] = Q_BT[k-1] - h*0 = Q_BT[k-1] (unchanged)."""
        out = batt.step(p_bt=0.0, x_tot=10.0)
        assert abs(out["q_bt"] - _Q_BT_IC) < 1e-9

    def test_idle_matches_u_oc_function(self, batt):
        """Idle U_BT must equal _u_oc(Q_BT) for any Q_BT."""
        # Drive Q_BT away from IC first via discharge
        batt.step(p_bt=2000.0, x_tot=10.0)
        q_now = batt.q_bt
        out = batt.step(p_bt=0.0, x_tot=20.0)
        assert abs(out["u_bt"] - _u_oc(q_now)) < 1e-9


# ---------------------------------------------------------------------------
# 4. Charging mode (P_BT < 0)
# ---------------------------------------------------------------------------

class TestChargingMode:

    def test_charging_voltage_above_ocv(self, batt):
        """
        Charging: discriminant > a^2 (since -u2 > 0 for negative P_BT),
        so U_BT_L > a = OCV. Voltage RISES above open-circuit during charge.
        """
        out = batt.step(p_bt=-1000.0, x_tot=10.0)
        assert out["u_bt"] > 46.8

    def test_charging_reference_value(self, batt):
        """
        At SoC=0.5, P_BT=-1000W:
          u1=0.5, u2=-100
          a = 15.6*0.5+39 = 46.8
          b = 0*0.5+0.013 = 0.013
          disc = 46.8^2 + 4*0.013*100 = 2192.04+5.2 = 2197.24
          U_BT_L = (46.8 + sqrt(2197.24))/2 = 46.827761...
        """
        out = batt.step(p_bt=-1000.0, x_tot=10.0)
        assert abs(out["u_bt"] - 46.827761) < 1e-4

    def test_charging_current_negative(self, batt):
        """P_BT < 0 -> I_BT = P_BT/U_BT < 0 (current flows INTO battery)."""
        out = batt.step(p_bt=-1000.0, x_tot=10.0)
        assert out["i_bt"] < 0.0

    def test_charging_increases_q_bt(self, batt):
        """Q_BT[k] = Q_BT[k-1] - h*I_BT, and I_BT<0 during charging -> Q_BT increases."""
        out = batt.step(p_bt=-1000.0, x_tot=10.0)
        assert out["q_bt"] > _Q_BT_IC


# ---------------------------------------------------------------------------
# 5. Discharging mode (P_BT > 0)
# ---------------------------------------------------------------------------

class TestDischargingMode:

    def test_discharging_voltage_below_ocv(self, batt):
        """
        Discharging: discriminant < a^2 (since -u2 < 0 for positive P_BT),
        so U_BT_E < a = OCV. Voltage SAGS below open-circuit during discharge.
        """
        out = batt.step(p_bt=1000.0, x_tot=10.0)
        assert out["u_bt"] < 46.8

    def test_discharging_reference_value(self, batt):
        """
        At SoC=0.5, P_BT=+1000W:
          u1=0.5, u2=+100
          a = 15.6*0.5+39 = 46.8
          b = 0.013
          disc = 46.8^2 + 4*0.013*(-100) = 2192.04-5.2 = 2186.84
          U_BT_E = (46.8 + sqrt(2186.84))/2 = 46.772206...
        """
        out = batt.step(p_bt=1000.0, x_tot=10.0)
        assert abs(out["u_bt"] - 46.772206) < 1e-4

    def test_discharging_current_positive(self, batt):
        out = batt.step(p_bt=1000.0, x_tot=10.0)
        assert out["i_bt"] > 0.0

    def test_discharging_decreases_q_bt(self, batt):
        out = batt.step(p_bt=1000.0, x_tot=10.0)
        assert out["q_bt"] < _Q_BT_IC

    def test_charging_voltage_gt_discharging_voltage(self, batt):
        """At same SoC and |P_BT|, charging voltage > discharging voltage."""
        b1 = Battery(); b1.reset()
        b2 = Battery(); b2.reset()
        out_chg = b1.step(p_bt=-1000.0, x_tot=10.0)
        out_dis = b2.step(p_bt= 1000.0, x_tot=10.0)
        assert out_chg["u_bt"] > 46.8 > out_dis["u_bt"]


# ---------------------------------------------------------------------------
# 6. Charge integrator (1/z)
# ---------------------------------------------------------------------------

class TestChargeIntegrator:

    def test_q_bt_formula(self, batt):
        """Q_BT[k] = Q_BT[k-1] - h*I_BT[k-1] for any P_BT."""
        out = batt.step(p_bt=2000.0, x_tot=10.0)
        expected_q = _Q_BT_IC - _H_BATT * out["i_bt"]
        assert abs(out["q_bt"] - expected_q) < 1e-9

    def test_soc_matches_q_bt_over_q_bt_0(self, batt):
        out = batt.step(p_bt=1500.0, x_tot=10.0)
        assert abs(out["soc"] - out["q_bt"] / _Q_BT_0) < 1e-12

    def test_q_bt_monotonic_under_constant_discharge(self, batt):
        """Continuous positive P_BT must monotonically decrease Q_BT."""
        prev_q = _Q_BT_IC
        for _ in range(5):
            out = batt.step(p_bt=2000.0, x_tot=10.0)
            assert out["q_bt"] < prev_q
            prev_q = out["q_bt"]


# ---------------------------------------------------------------------------
# 7. V_BT energy-consumption formula
# ---------------------------------------------------------------------------

class TestVBT:

    def test_v_bt_zero_at_x_tot_zero(self, batt):
        out = batt.step(p_bt=1000.0, x_tot=0.0)
        assert out["v_bt"] == 0.0

    def test_v_bt_zero_for_idle_at_ic(self, batt):
        """At Q_BT=Q_BT_IC with no current flow, E(Q)=E_initial -> V_BT=0."""
        out = batt.step(p_bt=0.0, x_tot=10.0)
        assert abs(out["v_bt"]) < 1e-9

    def test_v_bt_positive_for_discharge(self, batt):
        """Discharging reduces Q_BT and U_oc(Q_BT), so E(Q) decreases -> V_BT>0."""
        out = batt.step(p_bt=1000.0, x_tot=10.0)
        assert out["v_bt"] > 0.0

    def test_v_bt_negative_for_charge(self, batt):
        """Charging increases Q_BT and U_oc(Q_BT), so E(Q) increases -> V_BT<0."""
        out = batt.step(p_bt=-1000.0, x_tot=10.0)
        assert out["v_bt"] < 0.0

    def test_v_bt_reference_value(self, batt):
        """
        P_BT=1000W, x_tot=100m:
          U_BT_E = 46.772206 V, I_BT = 21.3804 A
          Q_BT[1] = 18000 - 21.3804 = 17978.6198
          U_oc(Q_BT[1]) = (15.6/36000)*17978.6198 + 39 = 46.78406...
          E(Q_BT[1]) = 0.5*46.78406*17978.6198
          E_initial  = 0.5*46.8*18000 = 421200
          V_BT = (E_initial - E(Q_BT[1])) / 100 / 36 = 0.162106...
        """
        out = batt.step(p_bt=1000.0, x_tot=100.0)
        assert abs(out["v_bt"] - 0.162106) < 1e-4

    def test_v_bt_formula_matches_energy_difference(self, batt):
        """Direct re-derivation of v_bt from q_bt should match the returned value."""
        out = batt.step(p_bt=1500.0, x_tot=200.0)
        e_now = 0.5 * _u_oc(out["q_bt"]) * out["q_bt"]
        e_init = 0.5 * _V_BT_C_CONST * _Q_BT_IC
        expected_v_bt = (e_init - e_now) / 200.0 / 36.0
        assert abs(out["v_bt"] - expected_v_bt) < 1e-9


# ---------------------------------------------------------------------------
# 8. Supervision — undervoltage
# ---------------------------------------------------------------------------

class TestUndervoltageSupervision:

    def test_no_undervoltage_in_normal_operation(self, batt):
        """
        threshold = (c_BT_E3/Q_BT_0)*Q_BT + c_BT_E1 ~ 46.8 V at 50% SoC.
        2*U_BT ~ 87-99V >> threshold -> no undervoltage in normal use.
        """
        for p_bt in [-3000., -500., 0., 500., 3000.]:
            b = Battery(); b.reset()
            out = b.step(p_bt=p_bt, x_tot=10.0)
            assert out["stop_uv"] is False, f"Unexpected stop_uv at p_bt={p_bt}"

    def test_undervoltage_formula(self, batt):
        """stop_uv = (2*U_BT - threshold) < 0."""
        out = batt.step(p_bt=1000.0, x_tot=10.0)
        threshold = (_C_BT_E3 / _Q_BT_0) * _Q_BT_IC + _C_BT_E1
        expected_stop_uv = (2.0 * out["u_bt"] - threshold) < 0.0
        assert out["stop_uv"] == expected_stop_uv


# ---------------------------------------------------------------------------
# 9. Supervision — overcurrent
# ---------------------------------------------------------------------------

class TestOvercurrentSupervision:

    def test_no_overcurrent_below_limit(self, batt):
        """At P_BT=13000W (50% SoC), I_BT ~ 280A < 300A -> no overcurrent."""
        out = batt.step(p_bt=13000.0, x_tot=10.0)
        assert abs(out["i_bt"]) < _I_BT_MAX
        assert out["stop_oc"] is False

    def test_overcurrent_above_limit(self, batt):
        """At P_BT=15000W (50% SoC), I_BT ~ 323A > 300A -> overcurrent."""
        out = batt.step(p_bt=15000.0, x_tot=10.0)
        assert abs(out["i_bt"]) > _I_BT_MAX
        assert out["stop_oc"] is True

    def test_overcurrent_formula(self, batt):
        """stop_oc = (I_BT_max - |I_BT|) < 0  <=>  |I_BT| > I_BT_max."""
        out = batt.step(p_bt=15000.0, x_tot=10.0)
        expected = (_I_BT_MAX - abs(out["i_bt"])) < 0.0
        assert out["stop_oc"] == expected

    def test_overcurrent_symmetric_for_charging(self, batt):
        """High-magnitude negative P_BT (fast charge) should also trip stop_oc."""
        out = batt.step(p_bt=-15000.0, x_tot=10.0)
        assert out["stop_oc"] is True


# ---------------------------------------------------------------------------
# 10. Stateless API
# ---------------------------------------------------------------------------

class TestStatelessAPI:

    def test_battery_step_matches_class(self):
        out_class, q_new_class = (lambda b: (b.step(p_bt=1000., x_tot=100.), b.q_bt))(
            Battery()
        )
        # Re-run with stateless function from same initial condition
        out_func, q_new_func = battery_step(p_bt=1000., x_tot=100., q_bt_prev=_Q_BT_IC)
        assert abs(out_class["v_bt"] - out_func["v_bt"]) < 1e-12
        assert abs(q_new_class - q_new_func) < 1e-12

    def test_battery_step_returns_tuple(self):
        out, q_new = battery_step(p_bt=500.0, x_tot=50.0)
        assert isinstance(out, dict)
        assert isinstance(q_new, float)

    def test_battery_step_chains_correctly(self):
        """Two sequential battery_step calls match two Battery.step calls."""
        b = Battery(); b.reset()
        out1_class = b.step(p_bt=1000.0, x_tot=10.0)
        out2_class = b.step(p_bt=-500.0, x_tot=20.0)

        out1_func, q1 = battery_step(p_bt=1000.0, x_tot=10.0, q_bt_prev=_Q_BT_IC)
        out2_func, q2 = battery_step(p_bt=-500.0, x_tot=20.0, q_bt_prev=q1)

        assert abs(out1_class["v_bt"] - out1_func["v_bt"]) < 1e-12
        assert abs(out2_class["v_bt"] - out2_func["v_bt"]) < 1e-12
        assert abs(out2_class["q_bt"] - q2) < 1e-12


# ---------------------------------------------------------------------------
# 11. Output structure / finiteness
# ---------------------------------------------------------------------------

class TestOutputStructure:

    def test_output_keys(self, batt):
        out = batt.step(p_bt=1000.0, x_tot=10.0)
        expected = {"v_bt", "q_bt", "u_bt", "i_bt", "soc", "stop_uv", "stop_oc"}
        assert set(out.keys()) == expected

    def test_all_outputs_finite(self, batt):
        for p_bt, x in [(-5000.,10.), (0.,10.), (5000.,10.), (0.,0.), (20000.,5.)]:
            b = Battery(); b.reset()
            out = b.step(p_bt=p_bt, x_tot=x)
            for k, v in out.items():
                if isinstance(v, float):
                    assert math.isfinite(v), f"{k}={v} not finite at p_bt={p_bt}"

    def test_flag_types_are_bool(self, batt):
        out = batt.step(p_bt=1000.0, x_tot=10.0)
        assert isinstance(out["stop_uv"], bool)
        assert isinstance(out["stop_oc"], bool)


# ---------------------------------------------------------------------------
# 12. Equivalent Fuel Consumption (BLOCK 7)
# ---------------------------------------------------------------------------

class TestEquivalentFuelConsumption:

    def test_output_keys(self):
        out = equivalent_fuel_consumption(v_ce=3.5, v_bt=1.5)
        assert set(out.keys()) == {"v_ce_equiv", "v_bt_converted"}

    def test_discharging_v_bt_adds_to_v_ce(self):
        """V_BT > 0 (discharging): converted value passes through (Upper=inf)
        and adds to V_CE."""
        out = equivalent_fuel_consumption(v_ce=3.5, v_bt=1.5)
        expected_converted = 1.5 * _EFC_GAIN
        assert abs(out["v_bt_converted"] - expected_converted) < 1e-6
        assert abs(out["v_ce_equiv"] - (3.5 + expected_converted)) < 1e-6

    def test_charging_v_bt_saturates_to_zero(self):
        """
        V_BT < 0 (net charging/regen): converted value would be negative,
        but Lower=0 saturation clamps it to 0 -> no credit, v_ce_equiv == v_ce.
        """
        out = equivalent_fuel_consumption(v_ce=3.5, v_bt=-1.5)
        assert out["v_bt_converted"] == 0.0
        assert abs(out["v_ce_equiv"] - 3.5) < 1e-12

    def test_zero_v_bt(self):
        out = equivalent_fuel_consumption(v_ce=3.5, v_bt=0.0)
        assert out["v_bt_converted"] == 0.0
        assert abs(out["v_ce_equiv"] - 3.5) < 1e-12

    def test_reference_value(self):
        """V_CE=3.5, V_BT=1.5 -> v_ce_equiv ~ 4.3288 (EFC_GAIN ~ 0.552552)."""
        out = equivalent_fuel_consumption(v_ce=3.5, v_bt=1.5)
        assert abs(out["v_ce_equiv"] - 4.328828) < 1e-4

    def test_no_upper_saturation_for_large_v_bt(self):
        """Upper=inf means even very large V_BT passes through unchanged."""
        out = equivalent_fuel_consumption(v_ce=0.0, v_bt=1000.0)
        assert math.isfinite(out["v_ce_equiv"])
        assert abs(out["v_bt_converted"] - 1000.0 * _EFC_GAIN) < 1e-6

    def test_all_outputs_finite(self):
        for v_ce, v_bt in [(0.,0.), (3.5,1.5), (3.5,-1.5), (0.,-100.), (10.,50.)]:
            out = equivalent_fuel_consumption(v_ce=v_ce, v_bt=v_bt)
            for k, v in out.items():
                assert math.isfinite(v), f"{k}={v} not finite at v_ce={v_ce},v_bt={v_bt}"


# ---------------------------------------------------------------------------
# 13. Full NEDC integration — chain through Battery and EFC
# ---------------------------------------------------------------------------

def test_battery_full_nedc_chain():
    """
    Chain: DrivingCycle -> Vehicle -> Gearbox -> Engine + Motor -> Tank + Battery
           -> EquivalentFuelConsumption

    Checks:
      - No NaN/Inf at any timestep
      - SoC stays within [0,1] (warns if not, since real models can deplete)
      - No undervoltage/overcurrent trips for a torque-limited motor command
      - V_CE_equiv is finite and >= V_CE (since EFC saturates V_BT at 0)
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from env.driving_cycle import DrivingCycle
    from env.powertrain import (
        VehicleDynamics, gearbox, combustion_engine, electric_motor,
        Tank, Battery, equivalent_fuel_consumption,
        _w_EM_max_row, _T_EM_max_arr, _interp1d_linear, _THETA_EM,
    )

    cycle = DrivingCycle("NEDC")
    veh   = VehicleDynamics()
    tank  = Tank()
    batt  = Battery()
    veh.reset(); tank.reset(); batt.reset()
    obs   = cycle.reset()

    v_ce_equiv_last = 0.0
    v_ce_last       = 0.0

    while True:
        veh_out = veh.step(obs["v"], obs["dv"])
        gb_out  = gearbox(
            w_wheel  = veh_out["w_wheel"],
            dw_wheel = veh_out["dw_wheel"],
            t_wheel  = veh_out["T_wheel"],
            gear     = obs["i"],
        )

        # Send ALL torque to the engine (engine-only baseline);
        # motor gets zero command -> P_EM is just inertia + aux (near 0).
        eng_out = combustion_engine(
            w_gear=gb_out["w_mgb"], dw_gear=gb_out["dw_mgb"], t_gear=gb_out["t_mgb"],
        )
        em_out = electric_motor(
            w_gear=gb_out["w_mgb"], dw_gear=gb_out["dw_mgb"], t_gear=0.0,
        )

        tank_out = tank.step(p_fuel=eng_out["p_ce"], x_tot=obs["x_tot"])
        batt_out = batt.step(p_bt=em_out["p_em"], x_tot=obs["x_tot"])

        efc_out = equivalent_fuel_consumption(
            v_ce=tank_out["v_liter"], v_bt=batt_out["v_bt"],
        )

        # No NaN/Inf anywhere
        for d, name in [(tank_out,"tank"), (batt_out,"battery"), (efc_out,"efc")]:
            for k, v in d.items():
                if isinstance(v, float):
                    assert math.isfinite(v), f"NaN/Inf in {name}[{k}] at t={obs['t']}"

        # No supervision trips with zero motor torque command
        assert not batt_out["stop_uv"], f"Unexpected undervoltage at t={obs['t']}"
        assert not batt_out["stop_oc"], f"Unexpected overcurrent at t={obs['t']}"

        v_ce_equiv_last = efc_out["v_ce_equiv"]
        v_ce_last       = tank_out["v_liter"]

        obs, done = cycle.step()
        if done:
            break

    print(f"\n  Final V_CE       : {v_ce_last:.4f} L/100km")
    print(f"  Final V_CE_equiv : {v_ce_equiv_last:.4f} L/100km")
    print(f"  Final SoC        : {batt.q_bt / _Q_BT_0:.4f}")

    # With zero motor torque, V_BT should be ~0 (only aux power, if any),
    # so v_ce_equiv should be close to v_ce.
    assert v_ce_equiv_last >= v_ce_last - 1e-6
    assert 0.0 <= batt.q_bt / _Q_BT_0 <= 1.0
