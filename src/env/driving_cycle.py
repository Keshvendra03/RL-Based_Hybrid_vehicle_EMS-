"""
driving_cycle.py
================
Python equivalent of the Simulink "Driving Cycle" source block.

Simulink block outputs (reproduced exactly):
  Port 1 -> v      : target speed            [m/s]
  Port 2 -> dv     : target acceleration     [m/s^2]
  Port 3 -> i      : gear number             [-]  (0 = neutral/idle)
  Port 4 -> x_tot  : total distance traveled [m]

Supported cycles
----------------
  "NEDC"   -- 1220 s,  10.932 km,  120 km/h peak
  "FTP75"  -- 1877 s,  17.767 km,   91 km/h peak

Usage
-----
    cycle = DrivingCycle("NEDC")
    obs   = cycle.reset()
    while True:
        obs, done = cycle.step()
        if done:
            break
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def _find_project_root() -> Path:
    """
    Walk up from this file until we find the project root.
    The project root is identified by containing BOTH:
      - a 'data' folder
      - a 'src'  folder
    This works regardless of how PyCharm resolves __file__.
    """
    candidate = Path(__file__).resolve()
    for _ in range(10):          # max 10 levels up
        candidate = candidate.parent
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError(
        f"Cannot find project root from {Path(__file__).resolve()}.\n"
        f"Make sure the project has both a 'data/' and 'src/' folder."
    )


_PROJECT_ROOT = _find_project_root()
_DATA_DIR     = _PROJECT_ROOT / "data" / "drive_cycles"

_CYCLE_FILES = {
    "NEDC" : _DATA_DIR / "nedc.csv",
    "FTP75": _DATA_DIR / "ftp75.csv",
}


class DrivingCycle:
    """
    Stateful iterator over a pre-defined driving cycle.

    Parameters
    ----------
    cycle_name : str
        "NEDC" or "FTP75"
    dt : float
        Simulation timestep in seconds. Default 1.0 s (matches Simulink).
    data_dir : str | Path | None
        Override the default data directory (useful in tests).
    """

    def __init__(
        self,
        cycle_name: str = "NEDC",
        dt: float = 1.0,
        data_dir: str | Path | None = None,
    ) -> None:

        name = cycle_name.upper().replace("-", "").replace("_", "")
        if name not in _CYCLE_FILES:
            raise ValueError(
                f"Unknown cycle '{cycle_name}'. "
                f"Available: {list(_CYCLE_FILES.keys())}"
            )

        self.cycle_name = name
        self.dt = float(dt)

        # Resolve CSV path
        if data_dir is not None:
            csv_path = Path(data_dir) / f"{name.lower()}.csv"
        else:
            csv_path = _CYCLE_FILES[name]

        if not csv_path.exists():
            raise FileNotFoundError(
                f"\n\nCSV file not found at:\n"
                f"  {csv_path}\n\n"
                f"Project root detected as:\n"
                f"  {_PROJECT_ROOT}\n\n"
                f"Please place nedc.csv and ftp75.csv in:\n"
                f"  {_DATA_DIR}\n"
            )

        # Load with numpy only -- no pandas dependency
        data = np.genfromtxt(
            csv_path,
            delimiter=",",
            skip_header=1,
            dtype=np.float64,
        )

        # Columns: time_s, v_ms, dv_ms2, gear, x_tot_m
        self._time = data[:, 0].astype(np.float64)
        self._v = data[:, 1].astype(np.float64)
        self._dv = data[:, 2].astype(np.float64)
        self._gear = data[:, 3].astype(np.int32)
        self._x_tot = data[:, 4].astype(np.float64)

        self._length = len(self._time)
        self._t: int = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset to start of cycle. Returns observation at t=0."""
        self._t = 0
        return self._make_obs()

    def step(self) -> Tuple[dict, bool]:
        """
        Advance one timestep (dt = 1 s) and return the NEW observation.

        Returns
        -------
        obs  : dict  -- signals at the NEW timestep (after advancing)
        done : bool  -- True when the new timestep is the last sample
        """
        if self._t < self._length - 1:
            self._t += 1
        obs = self._make_obs()
        done = self._t >= self._length - 1
        return obs, done

    @property
    def current(self) -> dict:
        """Return current signals without advancing the timestep."""
        return self._make_obs()

    @property
    def length(self) -> int:
        """Total number of timesteps in this cycle."""
        return self._length

    @property
    def progress(self) -> float:
        """Fraction of cycle completed [0.0, 1.0]."""
        return self._t / (self._length - 1)

    @property
    def v(self) -> float:
        """Current speed [m/s]."""
        return float(self._v[self._t])

    @property
    def dv(self) -> float:
        """Current acceleration [m/s^2]."""
        return float(self._dv[self._t])

    @property
    def gear(self) -> int:
        """Current gear (0 = neutral/idle)."""
        return int(self._gear[self._t])

    @property
    def x_tot(self) -> float:
        """Cumulative distance traveled [m]."""
        return float(self._x_tot[self._t])

    @property
    def t(self) -> int:
        """Current timestep index."""
        return self._t

    @property
    def total_distance_m(self) -> float:
        """Total cycle distance [m]."""
        return float(self._x_tot[-1])

    @property
    def max_speed_ms(self) -> float:
        """Peak speed in the cycle [m/s]."""
        return float(self._v.max())

    def summary(self) -> str:
        """One-line summary string for logging."""
        return (
            f"{self.cycle_name}  |  "
            f"{self._length} s  |  "
            f"{self.total_distance_m / 1000:.3f} km  |  "
            f"{self.max_speed_ms * 3.6:.1f} km/h peak  |  "
            f"gears {np.unique(self._gear).tolist()}"
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _make_obs(self) -> dict:
        """
        Package the 4 Simulink output-port signals into a dict.
        Keys match the Simulink wire labels exactly.
        """
        i = self._t
        return {
            "v"    : float(self._v[i]),
            "dv"   : float(self._dv[i]),
            "i"    : int(self._gear[i]),
            "x_tot": float(self._x_tot[i]),
            "t"    : int(self._time[i]),
        }