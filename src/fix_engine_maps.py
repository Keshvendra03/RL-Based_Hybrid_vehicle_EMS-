"""
fix_engine_maps.py
==================
Rebuilds data/maps/engine_maps.npz so the consumption map is indexed by
(w_CE_row, T_CE_col) -- SPEED x TORQUE -- matching the Simulink Lookup2D
block "Engine consumption map V = f(w, T)":
    Row index input values    : w_CE_row   (1x31)
    Column index input values : T_CE_col   (1x28)
    Table data                : V_CE_map   (31x28)

The OLD npz had a (15,) w_CE_row and a p_me_col pressure axis with a
p_me = T_CE*4*pi/V_d conversion -- that was WRONG. The real block looks
up TORQUE directly, no pressure conversion.

This script reads the three CSVs you exported from MATLAB and writes a
corrected engine_maps.npz, preserving the other keys (w_CE_max_fine,
T_CE_max, H_u) that combustion_engine() also uses.

Place the three CSVs (w_CE_row.csv, T_CE_col.csv, V_CE_map.csv) in the
project root (or edit CSV_DIR below), then run:
    python -m src.fix_engine_maps
"""

from __future__ import annotations

import shutil
from pathlib import Path
import numpy as np

from src.env.powertrain import _MAP_PATH  # data/maps/engine_maps.npz

CSV_DIR = Path(".")   # where w_CE_row.csv etc. live; "." = project root


def main():
    w_CE_row = np.loadtxt(CSV_DIR / "w_CE_row.csv", delimiter=",").ravel()
    T_CE_col = np.loadtxt(CSV_DIR / "T_CE_col.csv", delimiter=",").ravel()
    V_CE_map = np.loadtxt(CSV_DIR / "V_CE_map.csv", delimiter=",")

    assert V_CE_map.shape == (len(w_CE_row), len(T_CE_col)), \
        f"shape mismatch: V_CE_map{V_CE_map.shape} vs ({len(w_CE_row)},{len(T_CE_col)})"

    path = _MAP_PATH
    print(f"Patching: {path}")

    # Load existing arrays so we preserve keys we are NOT replacing
    with np.load(str(path)) as data:
        arrays = {k: data[k] for k in data.files}

    print("Existing keys:", list(arrays.keys()))

    # Backup
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    print(f"Backup -> {backup}")

    # Replace the consumption-map arrays with the correct torque-indexed ones.
    # New canonical keys: w_CE_row, T_CE_col, V_CE_map
    arrays["w_CE_row"] = w_CE_row.astype(np.float64)
    arrays["T_CE_col"] = T_CE_col.astype(np.float64)
    arrays["V_CE_map"] = V_CE_map.astype(np.float64)

    # Remove the now-obsolete pressure axis if present
    arrays.pop("p_me_col", None)

    np.savez(str(path), **arrays)
    print("Wrote corrected engine_maps.npz")
    print("  w_CE_row:", w_CE_row.shape, "range", w_CE_row.min(), w_CE_row.max())
    print("  T_CE_col:", T_CE_col.shape, "range", T_CE_col.min(), T_CE_col.max())
    print("  V_CE_map:", V_CE_map.shape)
    print("\nNow apply the combustion_engine() patch (see engine_map_lookup_patch.txt)")


if __name__ == "__main__":
    main()