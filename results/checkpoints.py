"""
checkpoints.py
==============
Read-side utilities for a train_sac.py run directory (models/<cycle>/ or
models/multi_<cycles>/): loads run_config.json + eval_history.csv +
best_score.txt into one object, and discovers all run directories under a
models root. Used by results/figures.py for post-training analysis.

    from results.checkpoints import load_run, discover_runs
    run = load_run("models/NEDC")
    run.eval_history        # pandas DataFrame, one row per (timestep, cycle) eval
    run.best_score          # float or None
    run.config              # dict: the exact CLI args + git commit that produced it
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

_EVAL_CSV_FIELDS = ["timesteps", "cycle", "v_liter", "v_ce_equiv", "soc_final",
                    "cycle_score", "mean_score", "is_best",
                    "rule_based_benchmark", "ecms_target"]


@dataclass
class RunResult:
    out_dir: Path
    config: dict = field(default_factory=dict)
    eval_history: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=_EVAL_CSV_FIELDS))
    best_score: Optional[float] = None

    @property
    def cycles(self) -> list[str]:
        if self.eval_history.empty:
            return []
        return sorted(self.eval_history["cycle"].unique().tolist())

    @property
    def name(self) -> str:
        return self.out_dir.name

    def history_for(self, cycle: str) -> pd.DataFrame:
        return self.eval_history[self.eval_history["cycle"] == cycle].sort_values("timesteps")

    def best_row_for(self, cycle: str) -> Optional[pd.Series]:
        h = self.history_for(cycle)
        if h.empty:
            return None
        return h.loc[h["v_ce_equiv"].idxmin()]

    def checkpoint_path(self, which: str = "best") -> Path:
        """which: 'best' or 'last'."""
        name = "sac_ems_best" if which == "best" else "sac_ems_last"
        return self.out_dir / name

    def has_checkpoint(self, which: str = "best") -> bool:
        return self.checkpoint_path(which).with_suffix(".zip").exists()


def load_run(out_dir: str | Path) -> RunResult:
    out_dir = Path(out_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f"No run directory at {out_dir}")

    cfg_path = out_dir / "run_config.json"
    config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    hist_path = out_dir / "eval_history.csv"
    if hist_path.exists() and hist_path.stat().st_size > 0:
        eval_history = pd.read_csv(hist_path)
    else:
        eval_history = pd.DataFrame(columns=_EVAL_CSV_FIELDS)

    best_path = out_dir / "best_score.txt"
    best_score = float(best_path.read_text()) if best_path.exists() else None

    return RunResult(out_dir=out_dir, config=config, eval_history=eval_history, best_score=best_score)


def discover_runs(models_root: str | Path = "models") -> list[Path]:
    """Find every train_sac.py output directory under `models_root`
    (identified by the presence of run_config.json -- the pre-refactor
    checkpoints directly in models/ won't have one and are correctly
    excluded, since they predate this bookkeeping and shouldn't be trusted
    as current-pipeline results anyway)."""
    root = Path(models_root)
    if not root.exists():
        return []
    return sorted({p.parent for p in root.rglob("run_config.json")})
