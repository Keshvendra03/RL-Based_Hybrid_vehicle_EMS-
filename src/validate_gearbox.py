"""
validate_gearbox.py
====================
Standalone validation of gearbox() against MATLAB-logged ground truth.

Feeds the EXACT (w_wheel, T_wheel, gear) triples logged from Simulink
directly into Python's gearbox(), and compares w_mgb/t_mgb against the
MATLAB-logged w_MGB/T_MGB -- completely isolated from DrivingCycle,
VehicleDynamics, Control Unit, etc.

Run from project root:
    python -m src.validate_gearbox
"""

from __future__ import annotations

from src.env.powertrain import gearbox


# (t, w_wheel, T_wheel, gear, MATLAB_w_MGB, MATLAB_T_MGB)
GROUND_TRUTH = [
    (53,  1.808400,  392.210300, 1, 19.4558,  42.4453),
    (54,  5.425300,  392.473200, 1, 58.3675,  38.9737),
    (55,  9.042200,  392.999100, 1, 97.2792,  38.3243),
    (65, 13.261960, -191.430168, 1, 13.2620, -16.736957),
    (66, 10.850694, -192.014432, 1, 10.8507, -16.634426),
    (102, 29.578173, 299.507030, 2, None,     48.928142),
    (103, 30.864198,  50.525013, 2, None,      8.685135),
]


def main() -> None:
    print(f"{'t':>4} {'w_wheel':>10} {'T_wheel':>10} {'gear':>4} "
          f"{'Py w_mgb':>10} {'ML w_MGB':>10} {'Py t_mgb':>10} {'ML T_MGB':>10} {'err%':>7}")
    print("-" * 90)

    for t, w_wheel, t_wheel, gear, ml_w_mgb, ml_t_mgb in GROUND_TRUTH:
        out = gearbox(w_wheel=w_wheel, dw_wheel=0.0, t_wheel=t_wheel, gear=gear)

        err_pct = 100.0 * (out["t_mgb"] - ml_t_mgb) / ml_t_mgb if ml_t_mgb != 0 else float("nan")

        ml_w_str = f"{ml_w_mgb:10.4f}" if ml_w_mgb is not None else f"{'--':>10}"

        print(f"{t:4d} {w_wheel:10.4f} {t_wheel:10.4f} {gear:4d} "
              f"{out['w_mgb']:10.4f} {ml_w_str} "
              f"{out['t_mgb']:10.4f} {ml_t_mgb:10.4f} {err_pct:6.2f}%")


if __name__ == "__main__":
    main()