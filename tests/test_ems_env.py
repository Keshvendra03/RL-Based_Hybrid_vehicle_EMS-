"""
test_ems_env.py
===============
Rigorous validation of the EMSEnv Gymnasium environment.

  1. Fast bilinear interpolation is numerically identical to the scipy
     RegularGridInterpolator path on the (clamped) in-bounds domain.
  2. Env wiring is exact: driving the env with the advanced rule-based
     controller's torque commands reproduces evaluate_advanced.py's final
     v_liter / v_ce_equiv / SoC to machine precision.
  3. The per-step reward decomposition is exact: summed fuel terms
     reconstruct the Tank's m_fuel, summed electrical terms reconstruct
     E_init - E_final.
  4. Gymnasium API check + action-mask safety invariants under random play.

Run:  python -m tests.test_ems_env
"""
from __future__ import annotations

import numpy as np

from src.env import powertrain as pt
from src.env.ems_env import (
    EMSEnv, _fast_interp2d_linear, K_FUEL_L_PER_KG, K_ELEC_L_PER_J,
    REWARD_SCALE, SOC_TARGET,
)
from src.env.powertrain import (
    _battery_energy, _Q_BT_IC, _RHO_FUEL, _K_CS,
    _w_EM_max_row, _T_EM_max_arr,
)
from src.baselines.advanced_rule_based import AdvancedController, control_unit_advanced
from src.evaluate_advanced import run_advanced


def test_fast_interp_exact():
    from scipy.interpolate import RegularGridInterpolator
    rng = np.random.default_rng(0)
    rows = np.sort(rng.uniform(0, 100, 12))
    cols = np.sort(rng.uniform(-50, 50, 17))
    table = rng.normal(size=(12, 17))
    interp = RegularGridInterpolator((rows, cols), table, method="linear",
                                     bounds_error=False, fill_value=None)
    max_err = 0.0
    for _ in range(5000):
        r = rng.uniform(rows[0], rows[-1])
        c = rng.uniform(cols[0], cols[-1])
        max_err = max(max_err, abs(_fast_interp2d_linear(rows, cols, table, r, c)
                                   - float(interp([r, c])[0])))
    assert max_err < 1e-10, max_err
    print(f"[1] fast interp == scipy interp   (max |err| = {max_err:.2e})  OK")


class _AdvancedMirrorEnv(EMSEnv):
    """EMSEnv with the action layer replaced by the advanced rule-based
    controller — used ONLY to prove the env's powertrain wiring is identical
    to evaluate_advanced.py."""

    def __init__(self, cycle_name="NEDC"):
        super().__init__(cycle_name=cycle_name, fast_interp=False)
        self.ctrl = AdvancedController(cycle_name=cycle_name)

    def reset(self, **kw):
        self.ctrl.reset()
        return super().reset(**kw)

    def _action_to_torques(self, action):
        d = self._demand
        c = self.ctrl.step(d["w_MGB"], d["dw_MGB"], d["T_MGB"],
                           d["gear"], self._Q_BT, d["v"])
        cu = control_unit_advanced(d["w_MGB"], d["dw_MGB"], d["T_MGB"],
                                   c["u"], c["state_CE"])
        return cu["T_CE"], cu["T_EM"], c["u"], "mirror"


def test_env_wiring_matches_evaluate(cycle="NEDC"):
    ref = run_advanced(cycle, verbose=False)

    env = _AdvancedMirrorEnv(cycle)
    obs, _ = env.reset()
    fuel_sum, elec_sum = 0.0, 0.0
    while True:
        obs, r, term, trunc, info = env.step(np.zeros(1, dtype=np.float32))
        fuel_sum += info["fuel_liters_step"]
        elec_sum += info["elec_liters_step"]
        if term:
            break
    fin = info["episode_final"]

    assert abs(fin["v_liter"] - ref["v_liter"]) < 1e-9, (fin["v_liter"], ref["v_liter"])
    assert abs(fin["v_ce_equiv"] - ref["v_ce_equiv"]) < 1e-9
    assert abs(fin["soc_final"] - ref["soc"]) < 1e-12
    print(f"[2] env wiring == evaluate_advanced ({cycle}): "
          f"v_liter {fin['v_liter']:.6f} == {ref['v_liter']:.6f}  OK")

    # reward decomposition exactness
    m_fuel_from_steps = fuel_sum / K_FUEL_L_PER_KG
    assert abs(m_fuel_from_steps - env.tank.m_fuel) < 1e-12
    dE_from_steps = elec_sum / K_ELEC_L_PER_J
    dE_true = _battery_energy(_Q_BT_IC) - _battery_energy(env._Q_BT)
    assert abs(dE_from_steps - dE_true) < 1e-6
    # reconstruct the display metric from per-step terms
    v_liter_rebuilt = m_fuel_from_steps / _RHO_FUEL / fin["x_tot_m"] * _K_CS * 1e5
    assert abs(v_liter_rebuilt - fin["v_liter"]) < 1e-9
    print(f"[3] per-step reward terms exactly decompose the metric  OK")


def test_random_play_safety(cycle="NEDC", episodes=2):
    env = EMSEnv(cycle, fast_interp=True)
    rng = np.random.default_rng(42)
    for ep in range(episodes):
        obs, _ = env.reset()
        n_overload = n_uv = n_oc = 0
        while True:
            a = rng.uniform(-1, 1, size=(1,)).astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            assert np.all(np.isfinite(obs)) and np.isfinite(r)
            soc = info["soc"]
            assert 0.01 < soc < 0.99, f"SoC out of safe band: {soc}"
            if term:
                break
        fin = info["episode_final"]
        print(f"[4] random ep{ep}: v_liter={fin['v_liter']:.3f}  "
              f"v_ce_equiv={fin['v_ce_equiv']:.3f}  SoC={fin['soc_final']*100:.1f}%  OK")


def test_gym_api():
    from gymnasium.utils.env_checker import check_env
    env = EMSEnv("NEDC")
    check_env(env, skip_render_check=True)
    print("[5] gymnasium check_env passed  OK")


def test_k_fb_costate_feedback(cycle="NEDC"):
    """k_fb=0 must reproduce the flat-eq_factor reward exactly (regression
    guard for the pre-existing exact-telescoping property). k_fb != 0 must
    shift the reward by EXACTLY the algebraic amount the formula in
    ems_env.py's docstring predicts -- not just 'in the right direction' --
    and must NOT change the underlying physics (fuel/elec liters), only the
    reward's battery-price term.
    """
    eq_factor = 1.3125
    k_fb = 8.0
    forced_soc = 0.40  # below target -> battery should look MORE expensive at k_fb>0

    def make(kfb):
        env = EMSEnv(cycle, eq_factor=eq_factor, k_fb=kfb, fast_interp=True)
        obs, _ = env.reset()
        q = forced_soc * pt._Q_BT_0
        env._Q_BT = q
        env._E_prev = _battery_energy(q)
        return env

    env0 = make(0.0)     # baseline: static price
    env1 = make(k_fb)    # ECMS-style feedback price
    action = np.array([0.5], dtype=np.float32)

    _, r0, _, _, info0 = env0.step(action)
    _, r1, _, _, info1 = env1.step(action)

    # k_fb must not touch the plant: identical commanded torques -> identical
    # fuel/elec decomposition regardless of the reward's pricing term.
    assert abs(info0["fuel_liters_step"] - info1["fuel_liters_step"]) < 1e-15
    assert abs(info0["elec_liters_step"] - info1["elec_liters_step"]) < 1e-15
    assert info0["mode"] == info1["mode"]

    elec_liters = info0["elec_liters_step"]
    expected_delta = -REWARD_SCALE * (k_fb * (SOC_TARGET - forced_soc)) * elec_liters
    actual_delta = r1 - r0
    assert abs(actual_delta - expected_delta) < 1e-8, (actual_delta, expected_delta)
    print(f"[6] k_fb costate feedback: reward shift matches formula exactly "
          f"(delta={actual_delta:.6f} == expected {expected_delta:.6f})  OK")


if __name__ == "__main__":
    test_fast_interp_exact()
    test_env_wiring_matches_evaluate("NEDC")
    test_random_play_safety("NEDC")
    test_gym_api()
    test_k_fb_costate_feedback("NEDC")
    print("\nALL ENV VALIDATION TESTS PASSED")
