"""
validate_vehicle.py
====================
Standalone validation of VehicleDynamics against MATLAB-logged ground truth.

Feeds the EXACT (v, dv) pairs logged from Simulink directly into
VehicleDynamics.step(), and compares w_wheel/T_wheel against the
MATLAB-logged values -- isolated from DrivingCycle/Gearbox/etc.

NOTE: VehicleDynamics carries state (v_prev for average-speed calc), so
this script steps through the sequence in order starting from v=0
(matching the cycle's initial idle period) to get the internal state
correct by the time we reach the comparison points.

Run from project root:
    python -m src.validate_vehicle
"""

from __future__ import annotations

from src.env.powertrain import VehicleDynamics


# Sequence leading up to and including the comparison points.
# (t, v, dv) -- MATLAB 0-indexed t (t=52 is the last idle sample,
# t=53,54 are the first two moving samples)
SEQUENCE = [
    (52, 0.000000, 0.000000),
    (53, 1.041667, 1.041667),
    (54, 2.083333, 1.041667),
]

# Ground truth to check against (t, MATLAB_w_wheel, MATLAB_T_wheel)
GROUND_TRUTH = {
    53: (1.808449, 392.210298),
    54: (5.425347, 392.473217),
}


def main() -> None:
    veh = VehicleDynamics()
    veh.reset()

    print(f"{'t':>4} {'v':>10} {'dv':>10} "
          f"{'Py w_wheel':>12} {'ML w_wheel':>12} "
          f"{'Py T_wheel':>12} {'ML T_wheel':>12} {'err%':>7}")
    print("-" * 90)

    for t, v, dv in SEQUENCE:
        out = veh.step(v=v, dv=dv)

        if t in GROUND_TRUTH:
            ml_w, ml_t = GROUND_TRUTH[t]
            err_pct = 100.0 * (out["T_wheel"] - ml_t) / ml_t if ml_t != 0 else float("nan")
            print(f"{t:4d} {v:10.6f} {dv:10.6f} "
                  f"{out['w_wheel']:12.6f} {ml_w:12.6f} "
                  f"{out['T_wheel']:12.6f} {ml_t:12.6f} {err_pct:6.2f}%")
        else:
            print(f"{t:4d} {v:10.6f} {dv:10.6f} "
                  f"{out['w_wheel']:12.6f} {'--':>12} "
                  f"{out['T_wheel']:12.6f} {'--':>12} {'--':>7}")


if __name__ == "__main__":
    main()