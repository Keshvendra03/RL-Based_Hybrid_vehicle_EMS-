# Changelog

All notable changes to this project will be documented in this file.
Format: `[Version] - Date - Description`

---

## [3.1.0] - 2026-08-17 — Phase 3: unguided-pipeline audit & consolidation

See `VALIDATION.md` ("Phase 3 — RL pipeline audit & fixes") for the full
write-up with how each item was confirmed. Summary:

### Fixed
- 18 failing tests in `tests/` traced to stale hardcoded constants (not
  physics bugs) and fixed — 211/211 now pass.
- `models/best_score.txt` silently describing a different checkpoint than
  the adjacent `.zip` — fixed with per-cycle output dirs + atomic writes +
  a `run_config.json` provenance sidecar.
- `evaluate_rl.py` / `mode_breakdown_rl.py` crashing on lookahead-trained
  checkpoints — now auto-infer the lookahead window from the checkpoint.
- `src/agents/per.py` (PER) implemented but never wired into training —
  now selectable via `train_sac.py --per`.
- Training-eval history previously lived only in memory and was lost on
  stop/crash — now persisted to `<out_dir>/eval_history.csv`.

### Removed
- Guided training as the default: `--prefill-mode benchmark` (seeded the
  replay buffer with the rule-based controller's own rollouts) deleted
  entirely from `train_sac.py`. New default: blank-slate, no prefill.
- `train_sac_fix2.py`, `train_sac_nstep.py`, `train_sac_lookahead.py` —
  near-duplicate forks, superseded by flags on one canonical `train_sac.py`.
- `src/env/ems_env_lookahead.py` — merged into `ems_env.py` as a
  `lookahead=` constructor arg (default 0 = old behavior, verified via diff
  + `test_ems_env.py`).

### Added
- `results/checkpoints.py`, `results/figures.py` — post-training analysis:
  training-curve + mode-breakdown plots and a plain-text trend/verdict
  diagnosis (`python -m results.figures --run models/<cycle>`).
- TensorBoard logging (`<out_dir>/tb`), on by default.

### Marked legacy (kept, not deleted, not used by the default pipeline)
- `pretrain_bc.py`, `finetune_bcreg.py`, `finetune_sac.py` — guided
  (behaviour-cloning) alternative; docstring-flagged as GUIDED / LEGACY.

---

## [3.0.0] - 2026-08-17 — Phase 3: SAC Agent Refactor

### Added
- Prioritized Experience Replay (`src/agents/per.py`)
- Behaviour Cloning pretraining (`src/agents/pretrain_bc.py`)
- BC regularized finetuning (`src/agents/finetune_bcreg.py`)
- SAC finetuning script (`src/agents/finetune_sac.py`)
- N-step SAC training (`src/agents/train_sac_nstep.py`)
- Lookahead SAC training (`src/agents/train_sac_lookahead.py`)
- Fixed SAC training v2 (`src/agents/train_sac_fix2.py`)
- Lookahead environment (`src/env/ems_env_lookahead.py`)
- ECMS baseline controller (`src/baselines/ecms.py`)
- Mode breakdown analysis (`src/agents/mode_breakdown_rl.py`)
- Trained model checkpoints (best + last)
- VALIDATION.md with results

### Changed
- Refactored monolithic `sac.py` into modular training scripts
- Updated `ems_env.py` and `driving_cycle.py`

### Removed
- Old debug/diagnostic scripts (`check_*.py`, `fix_*.py`, `validate_*.py`)
- Old `Networks.py` (merged into training scripts)
- Old `replay_buffer.py` (replaced by PER)
- Duplicate/outdated data files

---

## [2.0.0] - 2026-08 — Phase 2: Custom Gymnasium Environment

### Added
- Pure-Python Gymnasium environment (`src/env/ems_env.py`)
- Custom driving cycle loader (`src/env/driving_cycle.py`)
- Powertrain validation against MATLAB baseline

---

## [1.0.0] - 2026-08 — Phase 1: Rule-Based Baseline

### Added
- Initial project structure
- Rule-based EMS baseline (award-winning MATLAB/Simulink port)
- Advanced rule-based controller
- Pure-Python powertrain environment

---
