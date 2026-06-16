"""
evaluate_rl.py
==============
Load a trained SAC policy, roll it out deterministically over a driving
cycle, and report fuel economy against the advanced rule-based benchmark.

    python -m src.agents.evaluate_rl --cycle NEDC --model models/sac_ems_best

The headline number is V_liter (actual litres of Diesel per 100 km, exactly
the metric your evaluate.py reports). V_CE_equiv additionally charges the
agent for any NET battery depletion over the cycle, so the two numbers
coincide only when the policy is charge-sustaining (SoC_end == SoC_start).
A policy is only a legitimate winner when BOTH it beats the benchmark on
V_liter AND it is charge-sustaining.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from stable_baselines3 import SAC

from src.env.ems_env import EMSEnv, SOC_TARGET

# Benchmark fuel economy (advanced rule-based controller, this sandbox's maps)
BENCHMARK = {"NEDC": 3.5056, "FTP75": None}


def evaluate(model_path: str, cycle: str, verbose: bool = True) -> dict:
    model = SAC.load(model_path)
    env = EMSEnv(cycle)
    obs, _ = env.reset()

    modes = Counter()
    us, engine_on, t_ce_abs = [], 0, []
    soc_trace = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        modes[info["mode"]] += 1
        us.append(info["u"])
        if abs(info["T_CE_cmd"]) > 1e-6 and info["p_ce"] > 0.0:
            engine_on += 1
        t_ce_abs.append(abs(info["T_CE_cmd"]))
        soc_trace.append(info["soc"])
        if term:
            final = info["episode_final"]
            break

    n = sum(modes.values())
    final["engine_on_frac"] = engine_on / n
    final["u_mean"] = float(np.mean(us))
    final["u_std"] = float(np.std(us))
    final["modes"] = dict(modes)

    if verbose:
        bm = BENCHMARK.get(cycle)
        print(f"\n=== SAC policy on {cycle} ===")
        print(f"  V_liter      : {final['v_liter']:.4f}  L/100km   (actual fuel)")
        print(f"  V_CE_equiv   : {final['v_ce_equiv']:.4f}  L/100km   "
              f"(fuel + net battery energy)")
        print(f"  SoC final    : {final['soc_final']*100:.2f} %   "
              f"(target {SOC_TARGET*100:.0f} %)")
        print(f"  engine-on    : {final['engine_on_frac']*100:.1f} % of steps")
        print(f"  u  mean/std  : {final['u_mean']:+.3f} / {final['u_std']:.3f}")
        print(f"  modes        : {final['modes']}")
        if bm is not None:
            charge_sustaining = abs(final["soc_final"] - SOC_TARGET) <= 0.02
            beats = final["v_liter"] < bm and charge_sustaining
            print(f"\n  benchmark    : {bm:.4f}  L/100km  (advanced rule-based)")
            print(f"  delta        : {final['v_liter'] - bm:+.4f}  L/100km "
                  f"({(final['v_liter']/bm - 1)*100:+.1f} %)")
            print(f"  charge-sust. : {'yes' if charge_sustaining else 'NO'}")
            print(f"  VERDICT      : {'BEATS benchmark' if beats else 'does NOT beat benchmark'}")
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycle", default="NEDC", choices=["NEDC", "FTP75"])
    p.add_argument("--model", default="models/sac_ems_best")
    args = p.parse_args()
    evaluate(args.model, args.cycle)


if __name__ == "__main__":
    main()
