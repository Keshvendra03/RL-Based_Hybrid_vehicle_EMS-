"""
conftest.py  (place in project root)
=====================================
Adds src/ to sys.path before any test runs.
This fixes the 'from env.driving_cycle import DrivingCycle' import
in all test files, on Windows, macOS, and Linux.
"""
import sys
from pathlib import Path

# Project root = the folder this file lives in
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
