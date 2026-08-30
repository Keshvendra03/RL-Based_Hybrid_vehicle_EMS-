"""
phase8_figures.py -- figures for the Phase 8 forensic decision report.
    python -m results.phase8_figures            # forensic figs (8A/8B/8G)
    python -m results.phase8_figures --with-8c  # + mixture-actor before/after + training curves
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/phase8")
RB = {"NEDC": 3.5056, "FTP75": 3.2323}
ECMS = {"NEDC": 3.1887, "FTP75": 2.8097}
BANDS = ["0-15", "15-30", "30-35", "35-50", "50-75", ">75"]


def fig_engine_op():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for r, cyc in enumerate(("NEDC", "FTP75")):
        d = json.load(open(OUT / f"data/engine_op_counterfactual_{cyc}.json"))["regions"]
        rd = json.load(open(OUT / f"data/reward_counterfactual_{cyc}.json"))["regions"]
        bands = [b for b in BANDS if b in d]
        x = np.arange(len(bands)); w = 0.26
        ax = axes[r, 0]
        ax.bar(x - w, [d[b]["q_at_actor_load"] for b in bands], w, label="Q @ actor load", color="#d62728")
        ax.bar(x, [d[b]["q_at_ecms_load"] for b in bands], w, label="Q @ ECMS load", color="#2ca02c")
        ax.bar(x + w, [d[b]["q_at_maxload"] if d[b]["q_at_maxload"] is not None else np.nan for b in bands],
               w, label="Q @ max load", color="#1f77b4")
        ax.set_xticks(x); ax.set_xticklabels(bands); ax.set_title(f"{cyc}: critic value vs engine load (matched ECMS states)")
        ax.set_ylabel("min-Q"); ax.legend(fontsize=8)
        ax2 = axes[r, 1]
        ax2.bar(x - w, [rd[b]["tce_actor"] for b in bands], w, label="actor T_CE", color="#d62728")
        ax2.bar(x, [rd[b]["tce_ecms"] for b in bands], w, label="ECMS T_CE", color="#2ca02c")
        ax2.bar(x + w, [rd[b]["tce_argmaxr"] for b in bands], w, label="argmax-r T_CE", color="#ff7f0e")
        ax2.set_xticks(x); ax2.set_xticklabels(bands); ax2.set_title(f"{cyc}: engine T_CE -- actor vs ECMS vs reward-optimal")
        ax2.set_ylabel("engine torque [Nm]"); ax2.legend(fontsize=8)
    fig.suptitle("PHASE 8G -- engine operating-point: reward wants HARD load, critic endorses actor's SOFT load", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "figures/engine_operating_point.png", dpi=110); plt.close(fig)
    print("[saved] figures/engine_operating_point.png")


def fig_ceiling_bars():
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for i, cyc in enumerate(("NEDC", "FTP75")):
        d = json.load(open(OUT / f"data/qoracle_ceiling_{cyc}.json"))
        vals = [d["A_current_sac"]["v_ce"]["mean"], d["B_q_oracle"]["v_ce"]["mean"], RB[cyc], ECMS[cyc]]
        errs = [d["A_current_sac"]["v_ce"]["std"], d["B_q_oracle"]["v_ce"]["std"], 0, 0]
        labels = ["A: current\nSAC actor", "B: SAC-Q\noracle", "advanced\nrule-based", "ECMS"]
        cols = ["#d62728", "#1f77b4", "#7f7f7f", "#2ca02c"]
        ax[i].bar(labels, vals, yerr=errs, capsize=4, color=cols)
        ax[i].axhline(RB[cyc], color="#7f7f7f", ls="--", lw=1)
        for j, v in enumerate(vals):
            ax[i].text(j, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        cs_b = d["B_q_oracle"]["charge_sustaining"]
        ax[i].set_title(f"{cyc}: V_CE_equiv  (Q-oracle CS={cs_b})")
        ax[i].set_ylabel("L/100km")
    fig.suptitle("PHASE 8B -- Q-oracle ceiling: exploiting the trained critic does NOT beat the benchmark", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "figures/ceiling_bars.png", dpi=110); plt.close(fig)
    print("[saved] figures/ceiling_bars.png")


def _load_eval_hist(path):
    if not Path(path).exists():
        return None
    rows = list(csv.DictReader(open(path)))
    return [(int(r["timesteps"]), float(r["v_ce_equiv"]), float(r["soc_final"])) for r in rows]


def fig_8c(cycles=("NEDC", "FTP75")):
    # training curves + before/after mode probability
    fig, axes = plt.subplots(1, len(cycles), figsize=(8 * len(cycles), 5))
    if len(cycles) == 1:
        axes = [axes]
    for ax, cyc in zip(axes, cycles):
        plotted = False
        for s in range(3):
            for tag, style in ((f"models_p8c_N{s}" if cyc == "NEDC" else f"models_p8c_F{s}", "-"),):
                h = _load_eval_hist(f"{tag}/{cyc}/eval_history.csv")
                if h:
                    ax.plot([x[0] for x in h], [x[1] for x in h], style, alpha=0.7, label=f"8C mix seed{s}")
                    plotted = True
        # control reference
        for s, tag in enumerate((["models_p5s0_k2.5", "models_p5_k2.5", "models_p5_k2.5_s2"] if cyc == "NEDC"
                                 else ["models_p5f_k2.5_s0", "models_p5f_k2.5_s1", "models_p5f_k2.5_s2"])):
            h = _load_eval_hist(f"{tag}/{cyc}/eval_history.csv")
            if h:
                ax.plot([x[0] for x in h], [x[1] for x in h], "--", color="0.6", alpha=0.5,
                        label="CONTROL" if s == 0 else None)
        ax.axhline(RB[cyc], color="k", ls=":", label="rule-based")
        ax.axhline(ECMS[cyc], color="g", ls=":", label="ECMS")
        ax.set_ylim(3.0, 5.5); ax.set_xlabel("training step"); ax.set_ylabel("eval V_CE_equiv")
        ax.set_title(f"{cyc}: 8C mixture actor vs CONTROL training curves")
        if plotted:
            ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figures/phase8c_training_curves.png", dpi=110); plt.close(fig)
    print("[saved] figures/phase8c_training_curves.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-8c", action="store_true")
    a = ap.parse_args()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig_engine_op()
    fig_ceiling_bars()
    if a.with_8c:
        fig_8c()


if __name__ == "__main__":
    main()
