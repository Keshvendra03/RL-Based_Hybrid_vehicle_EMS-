"""
mode_breakdown_rl.py
====================
Mode-by-mode usage breakdown for your TRAINED SAC agent, classified with the
EXACT same logic used to profile the rule-based and ECMS controllers, so the
three are directly comparable.

Place at:   src/agents/mode_breakdown_rl.py
Run:        python -m src.agents.mode_breakdown_rl --cycle NEDC
            python -m src.agents.mode_breakdown_rl --cycle NEDC --model models/sac_ems_best

WHY A SEPARATE SCRIPT
---------------------
The RL agent's behaviour is in the trained weights (sac_ems_best.zip). It can
only be revealed by rolling the policy out deterministically through the SAME
env you trained on. This script does exactly that and tags every moving step
into one of five mutually exclusive modes:

  * engine OFF / pure electric : commanded engine torque <= fuel cutoff (5 Nm)
  * engine + motor ASSIST      : engine on AND motor adds positive torque
  * load-point-shift (charge)  : engine on AND motor charges (negative torque)
  * engine ONLY                : engine on, motor idle
  * REGEN (braking)            : T_MGB < 0, recuperation

The classification thresholds are identical to the ones that produced:

    Rule-based (NEDC):  OFF 59.0% | ASSIST 0.0% | LPS 23.8% | ONLY 0.2% | REGEN 17.0%
    ECMS       (NEDC):  OFF 53.1% | ASSIST 0.2% | LPS 29.7% | ONLY 0.0% | REGEN 17.0%

so your agent's row drops straight into that table.

REFERENCE (reproduced on this plant, charge-sustaining):
    NEDC  rule-based 3.506   ECMS 3.189
    FTP75 rule-based 3.232   ECMS 2.810
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from src.env.ems_env import EMSEnv
from src.env.powertrain import _T_CUTOFF


def classify_rollout(model: SAC, cycle: str) -> dict:
    """Deterministic greedy rollout; tag every moving step into one mode.

    Uses the env's own info dict (T_CE_cmd, T_EM_cmd, mode) so the split the
    agent ACTUALLY commanded (post-feasibility-masking) is what gets counted —
    not the raw action. This is the true executed behaviour.
    """
    env = EMSEnv(cycle)
    obs, _ = env.reset()

    moving = 0
    engine_off = 0
    assist = 0
    lps = 0
    engine_only = 0
    regen = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)

        d = env._demand              # current-step demand (already advanced after step)
        # NOTE: info reflects the step we just took; use the commanded torques in info
        T_ce = info["T_CE_cmd"]
        T_em = info["T_EM_cmd"]
        mode = info["mode"]

        # only count steps where the vehicle is actually moving and demanding/braking
        if mode == "stop":
            if not term:
                continue
        else:
            moving += 1
            if mode == "regen":
                regen += 1
            else:
                # traction step (T_MGB > 0)
                if T_ce <= _T_CUTOFF:
                    engine_off += 1
                elif T_em > 1e-6:
                    assist += 1
                elif T_em < -1e-6:
                    lps += 1
                else:
                    engine_only += 1

        if term:
            final = info["episode_final"]
            break

    return dict(
        v_liter=final["v_liter"],
        v_ce=final["v_ce_equiv"],
        soc=final["soc_final"],
        moving=moving,
        engine_off=engine_off,
        assist=assist,
        lps=lps,
        engine_only=engine_only,
        regen=regen,
    )


# Reference rows measured on this exact plant (for the comparison table).
REF = {
    "NEDC": {
        "rule_based": dict(v=3.506, off=59.0, assist=0.0, lps=23.8, only=0.2, regen=17.0),
        "ecms":       dict(v=3.189, off=53.1, assist=0.2, lps=29.7, only=0.0, regen=17.0),
    },
    "FTP75": {
        "rule_based": dict(v=3.232, off=46.3, assist=0.4, lps=22.4, only=5.2, regen=25.7),
        "ecms":       dict(v=2.810, off=40.4, assist=6.0, lps=27.9, only=0.0, regen=25.7),
    },
}


def main():
    p = argparse.ArgumentParser(description="Mode-by-mode breakdown of trained SAC agent")
    p.add_argument("--cycle", default="NEDC", choices=["NEDC", "FTP75"])
    p.add_argument("--model", default="models/sac_ems_best",
                   help="path to the saved SAC model (without .zip)")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(
            f"No model at {model_path}.zip — point --model at your trained agent."
        )

    model = SAC.load(str(model_path))
    r = classify_rollout(model, args.cycle)
    m = r["moving"]

    def pct(x):
        return 100.0 * x / m if m else 0.0

    rl = dict(off=pct(r["engine_off"]), assist=pct(r["assist"]), lps=pct(r["lps"]),
              only=pct(r["engine_only"]), regen=pct(r["regen"]))

    ref = REF.get(args.cycle, {})
    rb = ref.get("rule_based")
    ec = ref.get("ecms")

    print("\n" + "=" * 70)
    print(f"  MODE-BY-MODE BREAKDOWN — {args.cycle}   (% of moving steps)")
    print("=" * 70)
    print(f"  Your RL agent: V_liter={r['v_liter']:.3f}  "
          f"V_CE_equiv={r['v_ce']:.3f}  SoC_end={r['soc']*100:.1f}%  "
          f"(moving steps={m})")
    print("-" * 70)
    hdr = f"  {'mode':<26}{'RULE-BASED':>12}{'YOUR RL':>12}{'ECMS':>12}"
    print(hdr)
    print("-" * 70)
    rows = [
        ("engine OFF / electric", "off"),
        ("engine + motor ASSIST", "assist"),
        ("load-point-shift",      "lps"),
        ("engine ONLY",           "only"),
        ("REGEN (braking)",       "regen"),
    ]
    for label, key in rows:
        rbv = f"{rb[key]:.1f}%" if rb else "  -"
        ecv = f"{ec[key]:.1f}%" if ec else "  -"
        print(f"  {label:<26}{rbv:>12}{rl[key]:>11.1f}%{ecv:>12}")
    print("-" * 70)
    rbf = f"{rb['v']:.3f}" if rb else "  -"
    ecf = f"{ec['v']:.3f}" if ec else "  -"
    print(f"  {'V_CE_equiv [L/100km]':<26}{rbf:>12}{r['v_ce']:>11.3f}{ecf:>12}")
    print("=" * 70)
    print("\n  READING THIS TABLE")
    print("  - If your RL 'load-point-shift' % is BELOW ECMS's, you are leaving")
    print("    the main margin uncaptured (LPS depth is where ECMS wins).")
    print("  - If your RL 'engine OFF' % is at/above the benchmark, engine-off is")
    print("    NOT your problem (it almost never is on this plant).")
    print("  - REGEN % should match both references closely; it is hard-coded.")


if __name__ == "__main__":
    main()
