"""phase9_figures.py -- summary figures for the Phase 9 report."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/phase9")
REG = ["OFF", "LOW", "ECMS_NBHD", "HIGH_EFF", "MAX"]
BANDS = ["15-30", "30-35", "35-50", "50-75"]


def fig_region_q():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, cyc in zip(axes, ("NEDC", "FTP75")):
        d = json.load(open(OUT / f"data/critic_error_map_{cyc}.json"))["band_regions"]
        bands = [b for b in BANDS if b in d]
        x = np.arange(len(bands)); w = 0.16
        for k, r in enumerate(REG):
            vals = [d[b][r]["minQ"] if d[b].get(r, {}).get("n", 0) else np.nan for b in bands]
            ax.bar(x + (k - 2) * w, vals, w, label=r)
        ax.set_xticks(x); ax.set_xticklabels([f"{b} Nm" for b in bands])
        ax.set_title(f"{cyc}: region-averaged min(Q1,Q2) @ ECMS-traj states")
        ax.set_ylabel("min-Q"); ax.legend(fontsize=7)
    fig.suptitle("PHASE 9 §3/§4 -- critic ranks HIGH_EFF >= ECMS_NBHD >= LOW >= OFF (no gross error on-distribution)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "figures/region_minq.png", dpi=110); plt.close(fig)
    print("[saved] figures/region_minq.png")


def fig_decomp():
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    cyc_list = ["NEDC", "FTP75"]
    A = [json.load(open(OUT / f"data/engine_physics_{c}.json"))["A_operating_point_bsfc"] for c in cyc_list]
    BD = [json.load(open(OUT / f"data/engine_physics_{c}.json"))["B_onoff_decisions"] +
          json.load(open(OUT / f"data/engine_physics_{c}.json"))["D_residual"] for c in cyc_list]
    C = [json.load(open(OUT / f"data/engine_physics_{c}.json"))["C_battery_soc_equiv"] for c in cyc_list]
    x = np.arange(len(cyc_list))
    ax.bar(x, A, 0.5, label="A operating-point BSFC", color="#d62728")
    ax.bar(x, BD, 0.5, bottom=A, label="B+D mode-selection & timing", color="#1f77b4")
    ax.bar(x, C, 0.5, bottom=np.array(A) + np.array(BD), label="C battery / SoC", color="#2ca02c")
    for i, c in enumerate(cyc_list):
        tot = A[i] + BD[i] + C[i]
        ax.text(i, tot + 0.01, f"gap {tot:.3f}\nA={A[i]/tot*100:.0f}%  B+D={BD[i]/tot*100:.0f}%",
                ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(cyc_list)
    ax.set_ylabel("L/100km"); ax.set_title("PHASE 9 §10 -- physical SAC-ECMS gap decomposition")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "figures/gap_decomposition.png", dpi=110); plt.close(fig)
    print("[saved] figures/gap_decomposition.png")


if __name__ == "__main__":
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig_region_q()
    fig_decomp()
