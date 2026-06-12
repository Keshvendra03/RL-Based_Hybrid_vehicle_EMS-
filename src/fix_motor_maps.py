"""
fix_motor_maps.py
==================
Patches data/maps/motor_maps.npz: replaces the incorrect w_EM_max_row /
T_EM_max arrays with the correct values read from the MATLAB workspace
(used by Workspace 'w_EM_max' / 'T_EM_max', referenced by the Controller's
Interpreted MATLAB Fcn block via interp1(w_EM_max, T_EM_max, w_MGB)).

All OTHER arrays in motor_maps.npz (w_EM_row, T_EM_col, eta_EM_map,
w_EM_upper, ...) are preserved unchanged.

Run from project root:
    python -m src.fix_motor_maps

This OVERWRITES data/maps/motor_maps.npz in place. A backup is saved
alongside it as motor_maps.npz.bak before overwriting.
"""

from __future__ import annotations

import shutil
import numpy as np

from src.env.powertrain import _MOTOR_MAP_PATH


# Correct values from MATLAB workspace:
#   disp(w_EM_max) -> 0 100 200 300 400 500 600 700 800
#   disp(T_EM_max) -> 60.0000 60.0000 60.0000 44.5714 32.5714 24.0000 16.2857 11.1429 7.7143
_W_EM_MAX_ROW_CORRECT = np.array(
    [0, 100, 200, 300, 400, 500, 600, 700, 800], dtype=np.float64
)
_T_EM_MAX_CORRECT = np.array(
    [60.0000, 60.0000, 60.0000, 44.5714, 32.5714, 24.0000, 16.2857, 11.1429, 7.7143],
    dtype=np.float64,
)


def main() -> None:
    path = _MOTOR_MAP_PATH
    print(f"Patching: {path}")

    if not path.exists():
        raise FileNotFoundError(f"motor_maps.npz not found at {path}")

    # Load all existing arrays
    with np.load(str(path)) as data:
        arrays = {key: data[key] for key in data.files}

    print("\nBefore:")
    print("  w_EM_max_row:", arrays.get("w_EM_max_row"))
    print("  T_EM_max    :", arrays.get("T_EM_max"))

    # Backup original file
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    print(f"\nBackup saved to: {backup_path}")

    # Replace only the two target arrays
    arrays["w_EM_max_row"] = _W_EM_MAX_ROW_CORRECT
    arrays["T_EM_max"]     = _T_EM_MAX_CORRECT

    # Re-save, preserving all other keys
    np.savez(str(path), **arrays)

    print("\nAfter:")
    print("  w_EM_max_row:", arrays["w_EM_max_row"])
    print("  T_EM_max    :", arrays["T_EM_max"])
    print(f"\nAll other keys preserved: {[k for k in arrays if k not in ('w_EM_max_row', 'T_EM_max')]}")
    print("\nDone. Re-run 'python -m src.evaluate --cycle NEDC' to check results.")


if __name__ == "__main__":
    main()