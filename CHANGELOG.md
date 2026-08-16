# Changelog

All notable changes to this project will be documented in this file.
Format: `[Version] - Date - Description`

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
