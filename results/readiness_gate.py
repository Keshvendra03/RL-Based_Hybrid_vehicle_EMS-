"""
readiness_gate.py
==================
Pre-flight checklist to run BEFORE committing to a full-length (1.5M-step)
train_sac.py run. Point it at a completed smoke-test run directory
(e.g. a 150k-step run under models_trial_eqfix/<cycle>) and it prints a
PASS/FAIL gate report, one line per check, with the actual numbers behind
each verdict -- not just a green/red light.

    python -m results.readiness_gate --run models_trial_eqfix/NEDC

Exits 0 if every check passes, 1 otherwise, so it can gate a script
("run the smoke test, then only launch the full run if this passes").

WHAT IT CHECKS (and why each one matters)
------------------------------------------
1. Unit tests (`pytest tests/`) pass in full.
   Why: a full run takes ~40-60x longer than a smoke test. If the physics/
   env layer has a regression, you want to find out in ~30 seconds, not
   after an hour of wasted compute.

2. Working tree is git-clean.
   Why: `run_config.json` records `git_commit` as the run's provenance.
   That record is worthless if the tree was dirty when the run started --
   you'd have no way to reproduce or audit the result later.

3. The smoke-test run actually completed (reached its full requested
   timesteps) and has a best checkpoint to analyze.
   Why: a run killed early (crash, session teardown) gives partial,
   noisier data that can look better or worse than reality by chance --
   see VERIFIED_FACTS.md 2026-08-26 for a real example of this happening.

4. Mode breakdown vs. the ECMS reference (src/agents/mode_breakdown_rl.py):
   the "engine OFF / pure electric" gap and "ASSIST" gap are both within
   tolerance.
   Why: this is the actual, direct measurement of the ASSIST BLOB failure
   mode this project has been diagnosing -- not a proxy. A config that
   still shows a large gap here will not suddenly fix itself over a much
   longer run; it needs a different change first.

5. SoC quartile trend is not diverging away from the 50% target.
   Why: a policy can look fine on fuel alone while quietly drifting off
   charge-sustaining -- the quartile trend catches that even when the
   headline fuel number looks OK.

This script does NOT replace judgement. It's a fast, repeatable filter so
an obviously-not-ready config doesn't burn a 1.5M-step run. A PASS here is
a green light to scale up; a FAIL tells you exactly which of the five
things to fix first.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from results.checkpoints import load_run


def _run_pytest() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        capture_output=True, text=True,
    )
    ok = proc.returncode == 0
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "(no output)"
    return ok, tail


def _git_clean() -> tuple[bool, str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True,
    )
    dirty_lines = [l for l in proc.stdout.splitlines() if l.strip()]
    ok = len(dirty_lines) == 0
    detail = "clean" if ok else f"{len(dirty_lines)} uncommitted change(s)"
    return ok, detail


def _mode_gate(run_dir: Path, cycle: str, off_tol: float, assist_tol: float):
    from src.agents.mode_breakdown_rl import REF, classify_rollout
    from stable_baselines3 import SAC

    run = load_run(run_dir)
    if not run.has_checkpoint("best"):
        return False, "no best checkpoint found", None

    model = SAC.load(str(run.checkpoint_path("best")))
    r = classify_rollout(model, cycle)
    mv = r["moving"]
    pct = lambda x: 100.0 * x / mv if mv else 0.0

    ec = REF.get(cycle, {}).get("ecms", {})
    off_gap = pct(r["engine_off"]) - ec.get("off", 0.0)
    assist_gap = pct(r["assist"]) - ec.get("assist", 0.0)

    ok = abs(off_gap) <= off_tol and abs(assist_gap) <= assist_tol
    detail = (
        f"OFF={pct(r['engine_off']):.1f}% (ECMS {ec.get('off', 0.0):.1f}%, gap {off_gap:+.1f}pp)  "
        f"ASSIST={pct(r['assist']):.1f}% (ECMS {ec.get('assist', 0.0):.1f}%, gap {assist_gap:+.1f}pp)  "
        f"LPS={pct(r['lps']):.1f}%  ONLY={pct(r['engine_only']):.1f}%  REGEN={pct(r['regen']):.1f}%"
    )
    return ok, detail, (off_gap, assist_gap)


def _soc_trend_gate(run_dir: Path, cycle: str, max_drift: float):
    run = load_run(run_dir)
    h = run.history_for(cycle)
    if len(h) < 8:
        return False, f"only {len(h)} evals -- need >= 8 to assess a quartile trend"

    n = len(h)
    q = n // 4
    quartiles = [h.iloc[i * q:(i + 1) * q] if i < 3 else h.iloc[3 * q:] for i in range(4)]
    soc_q = [seg["soc_final"].mean() for seg in quartiles]
    monotonic = all(soc_q[i] < soc_q[i + 1] for i in range(3)) or \
                all(soc_q[i] > soc_q[i + 1] for i in range(3))
    drift = abs(soc_q[-1] - soc_q[0])

    ok = not (monotonic and drift > max_drift)
    detail = "SoC Q1..Q4: " + " -> ".join(f"{v*100:.1f}%" for v in soc_q) + \
             (f"  (monotonic drift {drift*100:.1f}pp)" if monotonic else "  (not monotonic)")
    return ok, detail


def main():
    p = argparse.ArgumentParser(description="Pre-flight readiness gate before a full-length train_sac.py run")
    p.add_argument("--run", required=True, help="smoke-test run directory, e.g. models_trial_eqfix/NEDC")
    p.add_argument("--cycle", default=None, help="defaults to the run directory's own cycle")
    p.add_argument("--off-tol", type=float, default=10.0, help="max allowed |OFF% - ECMS OFF%| gap (pp)")
    p.add_argument("--assist-tol", type=float, default=10.0, help="max allowed |ASSIST% - ECMS ASSIST%| gap (pp)")
    p.add_argument("--soc-drift-tol", type=float, default=0.05, help="max allowed monotonic SoC drift (fraction, 0.05 = 5pp)")
    p.add_argument("--skip-pytest", action="store_true", help="skip the unit-test check (not recommended)")
    args = p.parse_args()

    run_dir = Path(args.run)
    run = load_run(run_dir)
    cycle = args.cycle or (run.cycles[0] if run.cycles else None)
    if cycle is None:
        print(f"FAIL  no eval_history.csv rows in {run_dir} -- nothing to gate on")
        sys.exit(1)

    requested = run.config.get("timesteps")
    latest = int(run.history_for(cycle)["timesteps"].max()) if not run.history_for(cycle).empty else 0

    results = []

    if args.skip_pytest:
        results.append(("unit tests", True, "SKIPPED (--skip-pytest)"))
    else:
        ok, detail = _run_pytest()
        results.append(("unit tests (pytest tests/)", ok, detail))

    ok, detail = _git_clean()
    results.append(("git tree clean (provenance)", ok, detail))

    ran_to_completion = requested is not None and latest >= 0.98 * requested
    results.append((
        "smoke run completed",
        ran_to_completion,
        f"{latest:,} / {requested:,} steps" if requested else f"{latest:,} steps (no requested total recorded)",
    ))

    ok, detail, _gaps = _mode_gate(run_dir, cycle, args.off_tol, args.assist_tol)
    results.append(("mode breakdown vs. ECMS", ok, detail))

    ok, detail = _soc_trend_gate(run_dir, cycle, args.soc_drift_tol)
    results.append(("SoC quartile trend", ok, detail))

    print(f"\n=== Readiness gate: {run_dir} ({cycle}) ===\n")
    all_ok = True
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{status}] {name}\n         {detail}")

    print()
    if all_ok:
        print("VERDICT: READY -- safe to scale this config up to a full-length run.")
    else:
        print("VERDICT: NOT READY -- fix the FAILed check(s) above before committing to a full run.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
