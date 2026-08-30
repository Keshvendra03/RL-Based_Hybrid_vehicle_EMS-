"""
phase7_figures.py -- summary figures for the Phase 7 forensic report.
    python -m results.phase7_figures
Reads results/phase7/data/*.json produced by phase7_forensics.py.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/phase7")
LAM_REF = 1.3125


def fig_price(cycle, ax):
    d = json.load(open(OUT / f"data/effective_price_{cycle}.json"))
    bands = [b for b in d["by_soc_band"] if b.get("n", 0) >= 5]
    x = np.arange(len(bands))
    med = [b["price_median"] for b in bands]
    ax.bar(x, med, color="#1f77b4", label="SAC eff. price (median)")
    ax.axhline(LAM_REF, color="k", ls="--", lw=1, label=f"lambda_ref = {LAM_REF}")
    ecms_med = d["ecms_effective_lambda_own_rollout"]["median"]
    ax.axhline(ecms_med, color="m", ls=":", lw=1.5, label=f"ECMS eff. lambda median = {ecms_med:.2f}")
    ax.set_xticks(x); ax.set_xticklabels([b["band"] for b in bands])
    ax.set_title(f"{cycle}: effective battery price vs SoC band (ECMS units)")
    ax.set_xlabel("SoC band [%]"); ax.set_ylabel("lambda_eff [ECMS units]")
    ax.legend(fontsize=7)


def fig_actor_gap(cycle, ax):
    d = json.load(open(OUT / f"data/matched_states_{cycle}_summary.json"))["per_region"]
    regs = [r for r in ["15-30", "30-35", "35-50", "50-75"] if r in d]
    x = np.arange(len(regs)); w = 0.27
    sac = [d[r]["mode_sac_OFF_pct"] for r in regs]
    q = [d[r]["mode_argmaxQ_OFF_pct"] for r in regs]
    ec = [d[r]["mode_ecms_OFF_pct"] for r in regs]
    ax.bar(x - w, sac, w, label="SAC actor OFF%", color="#d62728")
    ax.bar(x, q, w, label="SAC argmax-Q OFF%", color="#1f77b4")
    ax.bar(x + w, ec, w, label="ECMS OFF%", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels([f"{r} Nm" for r in regs])
    ax.set_title(f"{cycle}: engine-OFF usage -- actor vs its own critic vs ECMS")
    ax.set_ylabel("OFF share [%]"); ax.legend(fontsize=7)


def fig_error(cycle, ax):
    d = json.load(open(OUT / f"data/matched_states_{cycle}_summary.json"))["per_region"]
    regs = [r for r in ["15-30", "30-35", "35-50", "50-75"] if r in d]
    x = np.arange(len(regs)); w = 0.38
    er = [d[r]["ERROR_reward"]["median"] for r in regs]
    ec = [d[r]["ERROR_critic"]["median"] for r in regs]
    ax.bar(x - w / 2, er, w, label="median ERROR_reward = r(a_ECMS)-r(a_SAC)", color="#ff7f0e")
    ax.bar(x + w / 2, ec, w, label="median ERROR_critic = Q(a_ECMS)-Q(a_SAC)", color="#1f77b4")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{r} Nm" for r in regs])
    ax.set_title(f"{cycle}: does the ECMS action lose on reward or on critic value?")
    ax.set_ylabel("L/100km-equiv reward units"); ax.legend(fontsize=7)


def fig_gap(cycle, ax):
    d = json.load(open(OUT / f"data/ecms_gap_{cycle}.json"))
    regs = list(d["regions"].keys())
    x = np.arange(len(regs)); w = 0.38
    df = [d["regions"][r]["dfuel"] for r in regs]
    de = [d["regions"][r]["delec"] for r in regs]
    ax.bar(x - w / 2, df, w, label="dFuel (SAC-ECMS)", color="#d62728")
    ax.bar(x + w / 2, de, w, label="dElec (SAC-ECMS)", color="#1f77b4")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(regs, rotation=30)
    ax.set_title(f"{cycle}: SAC-ECMS gap by torque region  (total {d['gap_split']['gap_total']:+.3f})")
    ax.set_ylabel("L/100km"); ax.legend(fontsize=7)


def main():
    for cycle in ("NEDC", "FTP75"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig_price(cycle, axes[0, 0])
        fig_actor_gap(cycle, axes[0, 1])
        fig_error(cycle, axes[1, 0])
        fig_gap(cycle, axes[1, 1])
        fig.suptitle(f"PHASE 7 forensic summary -- {cycle}  (CONTROL: gated k_fb=2.5)", fontsize=13)
        fig.tight_layout()
        p = OUT / f"figures/phase7_summary_{cycle}.png"
        fig.savefig(p, dpi=110); plt.close(fig)
        print("[saved]", p)


if __name__ == "__main__":
    main()
