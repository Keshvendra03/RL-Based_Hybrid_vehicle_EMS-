"""
diagnose_subsystems.py
=======================
Standalone diagnostic to check each subsystem (Control Unit, Electric Motor,
Battery) in isolation using fixed, hand-picked inputs that mirror the
NEDC trace you posted (t=53-55, where T_MGB ~ 410-430, u ~ 0.15-0.16).

Run from project root:
    python -m src.diagnose_subsystems

This does three things:

  1. Checks controller_u() against the MATLAB formula by hand for one
     specific (w_MGB, dw_MGB, T_MGB) point -> confirms Controller is correct.

  2. Checks control_unit_baseline() T_CE/T_EM split sums back to T_MGB
     correctly (T_CE + T_EM should equal T_MGB, since Sum is T_MGB - u*T_MGB
     and T_EM = u*T_MGB -> T_CE + T_EM = T_MGB always, by construction).

  3. Runs ONLY electric_motor + Battery over a synthetic symmetric
     "discharge then regen" cycle (T_EM = +50 for 10s, then T_EM = -50 for
     10s, same w_MGB both times) and checks whether q_bt returns close to
     its starting value. If it does NOT return close to start, the bug is
     in electric_motor's efficiency map / sign convention, NOT in the
     Control Unit -- because regen recovers LESS than discharge consumed
     (efficiency losses), so q_bt should end up slightly BELOW start, never
     drastically above.

     If q_bt ends up FAR ABOVE the starting value after this symmetric
     test, electric_motor or Battery has a sign/scale bug.
"""

from __future__ import annotations

from src.env.powertrain import (
    electric_motor, Battery, _Q_BT_0, _Q_BT_IC,
)
from src.baselines.rule_based import controller_u, control_unit_baseline


def check_1_controller_u():
    print("=" * 70)
    print("CHECK 1: controller_u() formula sanity (point from t=53)")
    print("=" * 70)

    # Approx values around t=53 in your trace: T_MGB=428.43, gear=1
    # w_MGB / dw_MGB are not in your printout -- use placeholders and show
    # the formula breakdown so you can compare against MATLAB Workspace
    # values for the SAME timestep (run MATLAB sim, inspect w_MGB/dw_MGB
    # at t=53 and substitute below).
    w_MGB  = 100.0   # <-- REPLACE with actual w_MGB from MATLAB at t=53
    dw_MGB = 0.0     # <-- REPLACE with actual dw_MGB from MATLAB at t=53
    T_MGB  = 428.43

    u = controller_u(w_MGB, dw_MGB, T_MGB)
    print(f"  Inputs : w_MGB={w_MGB}, dw_MGB={dw_MGB}, T_MGB={T_MGB}")
    print(f"  Branch : T_MGB >= 60 -> Load Point Shifting")
    print(f"  u      = {u:.4f}   (your trace shows u=0.154 at this T_MGB)")
    print()
    print("  ACTION: Replace w_MGB/dw_MGB above with the real MATLAB values")
    print("          at t=53 and re-run. If u differs from 0.154 by more")
    print("          than ~1e-3, the bug is in controller_u() or the")
    print("          w_EM_max/T_EM_max map values.")
    print()


def check_2_torque_split_identity():
    print("=" * 70)
    print("CHECK 2: T_CE + T_EM == T_MGB identity (Sum block |+- structure)")
    print("=" * 70)

    test_points = [
        (100.0, 0.0, 428.43),
        (100.0, 0.0, -180.60),
        (50.0, 0.0, 68.16),
    ]
    all_ok = True
    for w_MGB, dw_MGB, T_MGB in test_points:
        out = control_unit_baseline(w_MGB, dw_MGB, T_MGB, i=1, v=10.0, Q_BT=_Q_BT_IC)
        residual = (out["T_CE"] + out["T_EM"]) - T_MGB
        ok = abs(residual) < 1e-9
        all_ok &= ok
        print(f"  T_MGB={T_MGB:8.2f}  u={out['u']:7.4f}  "
              f"T_CE={out['T_CE']:8.2f}  T_EM={out['T_EM']:8.2f}  "
              f"T_CE+T_EM-T_MGB={residual:+.2e}  {'OK' if ok else 'MISMATCH'}")
    print()
    if all_ok:
        print("  -> Torque split identity holds. Control Unit wiring formula")
        print("     (T_CE = T_MGB - u*T_MGB, T_EM = u*T_MGB) is internally")
        print("     consistent. If Simulink's Sum/Multiply do something")
        print("     DIFFERENT from this identity, that's the bug source --")
        print("     check the Sum block's actual two INPUT WIRES in image 2")
        print("     (does input2 come from u*T_MGB, or from something else")
        print("     like u*T_EM_max or a different signal entirely?).")
    print()


def check_3_symmetric_discharge_regen():
    print("=" * 70)
    print("CHECK 3: Symmetric discharge/regen -> Battery should NOT net-charge")
    print("=" * 70)

    w_MGB = 100.0  # rad/s, arbitrary fixed speed for both phases

    batt = Battery()
    batt.reset()
    q_start = batt.q_bt
    print(f"  q_bt start = {q_start:.2f}  (SoC={q_start/_Q_BT_0*100:.1f}%)")

    # Phase 1: motoring/discharge, T_EM=+50 Nm for 10 steps
    for step in range(10):
        em_out = electric_motor(w_gear=w_MGB, dw_gear=0.0, t_gear=50.0)
        batt_out = batt.step(p_bt=em_out["p_em"], x_tot=1000.0 + step)
    q_after_discharge = batt.q_bt
    print(f"  After 10s discharge (T_EM=+50): q_bt = {q_after_discharge:.2f}  "
          f"(delta={q_after_discharge - q_start:+.2f})")
    print(f"    p_em during discharge = {em_out['p_em']:.2f} W "
          f"(eta_factor={em_out['eta_factor']:.4f})")

    # Phase 2: regen/charge, T_EM=-50 Nm for 10 steps (same |torque|, same speed)
    for step in range(10):
        em_out = electric_motor(w_gear=w_MGB, dw_gear=0.0, t_gear=-50.0)
        batt_out = batt.step(p_bt=em_out["p_em"], x_tot=1010.0 + step)
    q_after_regen = batt.q_bt
    print(f"  After 10s regen    (T_EM=-50): q_bt = {q_after_regen:.2f}  "
          f"(delta from discharge end={q_after_regen - q_after_discharge:+.2f})")
    print(f"    p_em during regen     = {em_out['p_em']:.2f} W "
          f"(eta_factor={em_out['eta_factor']:.4f})")

    print()
    net = q_after_regen - q_start
    print(f"  NET change over full symmetric cycle: {net:+.2f} As")
    print(f"  Expected: net <= 0 (efficiency losses mean regen recovers LESS")
    print(f"            than discharge consumed -> q_bt should end up AT or")
    print(f"            slightly BELOW q_start, never significantly above).")
    print()
    if net > 1.0:
        print("  *** PROBLEM FOUND ***")
        print("  Battery NET-CHARGED from a symmetric discharge/regen cycle.")
        print("  This violates energy conservation and points to a bug in")
        print("  electric_motor()'s eta_factor sign convention OR Battery's")
        print("  charging/discharging formula sign.")
        print()
        print("  Specifically check: for T_EM>0 (motoring), eta_factor should")
        print("  be 1/eta (>1, INCREASES p_em magnitude vs p_mech). For")
        print("  T_EM<0 (regen), eta_factor should be eta (<1, DECREASES")
        print("  |p_em| vs |p_mech|). If eta_factor printed above for regen")
        print("  is > 1 or for discharge is < 1, the eta map lookup or its")
        print("  sign-indexing (T_EM_col) is inverted.")
    else:
        print("  -> Battery/Motor pass the conservation check. The bug is")
        print("     elsewhere (likely Control Unit's u/T_EM values driven")
        print("     by REAL w_MGB/dw_MGB from the cycle, or a cumulative")
        print("     rounding/feedback issue with Q_BT -> SoC -> controller_u")
        print("     if your FUTURE rule-based version uses Q_BT -- but in")
        print("     THIS baseline, Q_BT is not fed into controller_u(), so")
        print("     check next: print SUM of all T_EM*dt over the full NEDC")
        print("     run and compare its sign/magnitude to SUM of T_MGB*dt")
        print("     for T_MGB<0 segments (regen) vs T_MGB>=60 segments (LPS).")


if __name__ == "__main__":
    check_1_controller_u()
    check_2_torque_split_identity()
    check_3_symmetric_discharge_regen()