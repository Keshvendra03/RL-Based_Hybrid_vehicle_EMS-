"""
figures.py
==========
Post-training analysis: reads a train_sac.py run directory (via
results/checkpoints.py) and produces (1) a training-curve plot with the
rule-based/ECMS reference lines, (2) a mode-breakdown bar chart for the best
checkpoint, and (3) a plain-text diagnosis (improving / plateaued /
worsening, beats-benchmark verdict, and an explicit check for the "assist
blob" failure mode diagnosed during the Phase-3 pipeline audit).

    python -m results.figures --run models/NEDC
    python -m results.figures --run models/FTP75 --no-plots

Works on a run that's still in progress or was stopped early -- it only
reads eval_history.csv, which train_sac.py appends to after every
evaluation, not just a completed run's final summary.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results.checkpoints import RunResult, load_run


def plot_training_curve(run: RunResult, cycle: str, save_path: Path | None = None):
    h = run.history_for(cycle)
    if h.empty:
        raise ValueError(f"No eval history for cycle {cycle!r} in {run.out_dir}")

    rb = h["rule_based_benchmark"].dropna().iloc[0] if h["rule_based_benchmark"].notna().any() else None
    ecms = h["ecms_target"].dropna().iloc[0] if h["ecms_target"].notna().any() else None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h["timesteps"], h["v_ce_equiv"], marker="o", ms=3, lw=1, label="SAC (deterministic eval)")
    best = run.best_row_for(cycle)
    if best is not None:
        ax.scatter([best["timesteps"]], [best["v_ce_equiv"]], color="red", zorder=5,
                   label=f"best ({best['v_ce_equiv']:.3f} @ {int(best['timesteps']):,} steps)")
    if rb is not None:
        ax.axhline(rb, color="orange", ls="--", label=f"rule-based benchmark ({rb:.3f})")
    if ecms is not None:
        ax.axhline(ecms, color="green", ls=":", label=f"ECMS target ({ecms:.3f})")

    ax.set_xlabel("training timesteps")
    ax.set_ylabel("V_CE_equiv [L/100km]  (lower = better)")
    title = f"{cycle} training curve" if run.name == cycle else f"{run.name} — {cycle} training curve"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
    return fig


def plot_mode_breakdown(cycle: str, model_path: Path, save_path: Path | None = None):
    from stable_baselines3 import SAC
    from src.agents.mode_breakdown_rl import classify_rollout, REF, _infer_lookahead

    model = SAC.load(str(model_path))
    r = classify_rollout(model, cycle)
    m = r["moving"]
    pct = lambda x: 100.0 * x / m if m else 0.0
    rl = dict(off=pct(r["engine_off"]), assist=pct(r["assist"]), lps=pct(r["lps"]),
              only=pct(r["engine_only"]), regen=pct(r["regen"]))

    ref = REF.get(cycle, {})
    rb, ec = ref.get("rule_based"), ref.get("ecms")
    modes = [("off", "engine OFF"), ("assist", "ASSIST"), ("lps", "LPS"),
             ("only", "engine ONLY"), ("regen", "REGEN")]
    x = range(len(modes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    if rb:
        ax.bar([i - width for i in x], [rb[k] for k, _ in modes], width, label="rule-based")
    ax.bar(list(x), [rl[k] for k, _ in modes], width, label="RL agent")
    if ec:
        ax.bar([i + width for i in x], [ec[k] for k, _ in modes], width, label="ECMS")
    ax.set_xticks(list(x))
    ax.set_xticklabels([lbl for _, lbl in modes])
    ax.set_ylabel("% of moving steps")
    ax.set_title(f"{cycle} — mode-by-mode usage (V_CE_equiv={r['v_ce']:.3f})")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
    return fig, rl


def diagnose(run: RunResult, mode_breakdown: bool = True) -> str:
    lines = [f"=== {run.name}  ({run.out_dir}) ==="]
    if not run.config:
        lines.append("(no run_config.json found -- pre-refactor / unlabeled run)")
    else:
        lines.append(f"n_step={run.config.get('n_step')}  per={run.config.get('per')}  "
                      f"lookahead={run.config.get('lookahead')}  "
                      f"prefill_mode={run.config.get('prefill_mode')}  "
                      f"timesteps_requested={run.config.get('timesteps')}")

    if not run.cycles:
        lines.append("No eval_history.csv rows yet -- run hasn't reached its first eval interval.")
        return "\n".join(lines)

    for cycle in run.cycles:
        h = run.history_for(cycle)
        best = run.best_row_for(cycle)
        rb = best["rule_based_benchmark"] if best is not None else None
        ecms = best["ecms_target"] if best is not None else None
        soc_ok = best is not None and abs(best["soc_final"] - 0.5) <= 0.02
        beats_rb = best is not None and rb and best["v_ce_equiv"] < rb and soc_ok

        lines.append(f"\n--- {cycle} ---")
        lines.append(f"evals so far: {len(h)}   latest @ {int(h['timesteps'].max()):,} steps")
        if best is not None:
            lines.append(f"best V_CE_equiv: {best['v_ce_equiv']:.4f}  "
                          f"@ step {int(best['timesteps']):,}  SoC={best['soc_final']*100:.1f}%")
            if rb:
                lines.append(f"rule-based benchmark: {rb:.4f}  |  ECMS stretch target: {ecms:.4f}")
            lines.append(f"charge-sustaining (|SoC-50%|<=2%): {'yes' if soc_ok else 'NO'}")
            lines.append(f"VERDICT: {'BEATS rule-based benchmark' if beats_rb else 'does NOT yet beat benchmark'}")

        if len(h) >= 6:
            first = h.head(3)["v_ce_equiv"].mean()
            last = h.tail(3)["v_ce_equiv"].mean()
            delta = last - first
            if delta < -0.02:
                trend = f"IMPROVING (first-3 avg {first:.3f} -> last-3 avg {last:.3f})"
            elif delta > 0.02:
                trend = f"WORSENING (first-3 avg {first:.3f} -> last-3 avg {last:.3f}) -- check for instability/collapse"
            else:
                trend = f"PLATEAUED (first-3 avg {first:.3f} -> last-3 avg {last:.3f})"
            lines.append(f"trend (endpoints): {trend}")
        else:
            lines.append("trend: not enough evals yet to assess (need >= 6)")

        # Quartile view: catches a "flat/noisy fuel curve but the endpoints
        # happen to differ" false-IMPROVING read, and separately tracks SoC
        # drift -- a policy can look fine on fuel alone while quietly walking
        # away from charge-sustaining, which the endpoint-only trend above
        # cannot see at all.
        if len(h) >= 8:
            n = len(h)
            q = n // 4
            quartiles = [h.iloc[i*q:(i+1)*q] if i < 3 else h.iloc[3*q:] for i in range(4)]
            fuel_q = [seg["v_ce_equiv"].mean() for seg in quartiles]
            soc_q = [seg["soc_final"].mean() for seg in quartiles]
            lines.append("quartile fuel (Q1..Q4): " + " -> ".join(f"{v:.3f}" for v in fuel_q))
            lines.append("quartile SoC  (Q1..Q4): " + " -> ".join(f"{v*100:.1f}%" for v in soc_q))

            fuel_flat = abs(fuel_q[-1] - fuel_q[0]) < 0.05 * fuel_q[0]
            monotonic_drift = all(soc_q[i] < soc_q[i + 1] for i in range(3)) or \
                              all(soc_q[i] > soc_q[i + 1] for i in range(3))
            soc_drift_mag = abs(soc_q[-1] - soc_q[0])
            if monotonic_drift and soc_drift_mag > 0.05:
                direction = "up (charging)" if soc_q[-1] > soc_q[0] else "down (depleting)"
                lines.append(f"  -> SoC DRIFT: monotonically {direction} across all 4 quartiles "
                             f"({soc_q[0]*100:.1f}% -> {soc_q[-1]*100:.1f}%), "
                             f"{'while fuel stayed flat -- ' if fuel_flat else ''}"
                             "not converging to charge-sustaining. If this persists in the full "
                             "run, the SoC penalty (lambda_soc/soc_deadband) is likely too weak "
                             "relative to the fuel term at this eq_factor -- a reward-shaping "
                             "issue, not a 'needs more steps' issue.")
            if fuel_flat and rb and best is not None and best["v_ce_equiv"] > 1.15 * rb:
                lines.append(f"  -> FUEL PLATEAU well above benchmark (Q1 {fuel_q[0]:.3f} ~= Q4 "
                             f"{fuel_q[-1]:.3f}, both >15% above {rb:.3f}): the endpoint trend "
                             "above can read 'IMPROVING' from noise alone -- trust this quartile "
                             "view over the 2-point endpoint trend.")

        if mode_breakdown and run.has_checkpoint("best"):
            try:
                from src.agents.mode_breakdown_rl import REF
                from stable_baselines3 import SAC
                model = SAC.load(str(run.checkpoint_path("best")))
                from src.agents.mode_breakdown_rl import classify_rollout
                r = classify_rollout(model, cycle)
                mv = r["moving"]
                pct = lambda x: 100.0 * x / mv if mv else 0.0
                ec = REF.get(cycle, {}).get("ecms", {})
                off_gap = pct(r["engine_off"]) - ec.get("off", 0.0)
                assist_gap = pct(r["assist"]) - ec.get("assist", 0.0)
                lines.append(f"mode breakdown (best ckpt): OFF={pct(r['engine_off']):.1f}%  "
                              f"ASSIST={pct(r['assist']):.1f}%  LPS={pct(r['lps']):.1f}%  "
                              f"ONLY={pct(r['engine_only']):.1f}%  REGEN={pct(r['regen']):.1f}%")
                if assist_gap > 10.0 and off_gap < -10.0:
                    lines.append("  -> 'ASSIST BLOB' pattern present: agent is blending engine+motor "
                                 "where ECMS commits to pure electric. This is the failure mode "
                                 "n-step returns were added to fix -- if it persists after "
                                 "substantial training, check TensorBoard entropy/actor-loss curves.")
            except Exception as e:  # pragma: no cover - diagnostic best-effort
                lines.append(f"mode breakdown: skipped ({e})")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run output directory, e.g. models/NEDC")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-mode-breakdown", action="store_true")
    args = p.parse_args()

    run = load_run(args.run)
    print(diagnose(run, mode_breakdown=not args.no_mode_breakdown))

    if not args.no_plots and run.cycles:
        analysis_dir = run.out_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        for cycle in run.cycles:
            curve_path = analysis_dir / f"training_curve_{cycle}.png"
            plot_training_curve(run, cycle, save_path=curve_path)
            print(f"\n[saved] {curve_path}")
            if run.has_checkpoint("best"):
                mb_path = analysis_dir / f"mode_breakdown_{cycle}.png"
                plot_mode_breakdown(cycle, run.checkpoint_path("best"), save_path=mb_path)
                print(f"[saved] {mb_path}")


if __name__ == "__main__":
    main()
