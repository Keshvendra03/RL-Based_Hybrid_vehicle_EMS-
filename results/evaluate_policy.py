"""
evaluate_policy.py
==================
THE single authoritative evaluation function. Every controller -- SAC, TD3,
ECMS, advanced rule-based -- is scored through `evaluate()` under identical
conditions, so no controller gets a bespoke evaluation shortcut.

    python -m results.evaluate_policy --checkpoint models_x/NEDC/sac_ems_best --cycle NEDC
    python -m results.evaluate_policy --controller ecms --cycle NEDC
    python -m results.evaluate_policy --controller rule_based --cycle FTP75

Returns / prints, for every controller:
    v_liter, v_ce_equiv, soc_init, soc_final, d_soc,
    off_pct, assist_pct, lps_pct, only_pct, regen_pct,
    constraint_violations, battery_throughput_kJ, engine_on_time_s,
    total_reward, action stats (RL only)

DESIGN NOTE -- why this file exists:
the previous pipeline scored RL via train_sac.rollout_deterministic, the
benchmarks via their own scripts, and modes via mode_breakdown_rl. Three code
paths meant a metric could differ by *which script computed it*. This
consolidates them. It does NOT re-implement physics: it drives the same
validated EMSEnv / powertrain blocks the benchmarks use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.env.ems_env import EMSEnv, SOC_TARGET
from src.env.powertrain import _T_CUTOFF


def _classify(mode: str, t_ce: float) -> str:
    """Identical thresholds to src/agents/mode_breakdown_rl.classify_rollout."""
    if mode == "stop":
        return "stop"
    if mode == "regen":
        return "regen"
    if t_ce <= _T_CUTOFF:
        return "off"
    if mode == "assist":
        return "assist"
    if mode == "lps_gen":
        return "lps"
    return "only"


def _summarize(recs: list[dict], fin: dict, rewards: list[float],
               actions: list[float] | None) -> dict:
    moving = [r for r in recs if r["cls"] != "stop"]
    n = max(len(moving), 1)
    cnt = lambda k: sum(1 for r in moving if r["cls"] == k)
    # battery energy throughput: integral of |P_batt| dt  (dt = 1 s)
    throughput_kJ = sum(abs(r["p_em"]) for r in recs) / 1000.0
    engine_on_s = sum(1 for r in recs if r["t_ce"] > _T_CUTOFF)
    out = dict(
        v_liter=fin["v_liter"],
        v_ce_equiv=fin["v_ce_equiv"],
        soc_init=0.5,
        soc_final=fin["soc_final"],
        d_soc_pp=(fin["soc_final"] - 0.5) * 100.0,
        soc_min=min(r["soc"] for r in recs),
        soc_max=max(r["soc"] for r in recs),
        charge_sustaining=abs(fin["soc_final"] - SOC_TARGET) <= 0.02,
        off_pct=100.0 * cnt("off") / n,
        assist_pct=100.0 * cnt("assist") / n,
        lps_pct=100.0 * cnt("lps") / n,
        only_pct=100.0 * cnt("only") / n,
        regen_pct=100.0 * cnt("regen") / n,
        moving_steps=len(moving),
        constraint_violations=sum(1 for r in recs if r["viol"]),
        battery_throughput_kJ=throughput_kJ,
        engine_on_time_s=engine_on_s,
        total_reward=float(np.sum(rewards)) if rewards else float("nan"),
    )
    if actions:
        a = np.asarray(actions)
        out.update(
            action_mean=float(a.mean()), action_std=float(a.std()),
            action_p25=float(np.percentile(a, 25)),
            action_p50=float(np.percentile(a, 50)),
            action_p75=float(np.percentile(a, 75)),
            action_sat_pct=float(100.0 * (np.abs(a) > 0.99).mean()),
            action_delta_mean=float(np.abs(np.diff(a)).mean()) if len(a) > 1 else 0.0,
        )
    return out


def _drive(env: EMSEnv, policy_fn, collect_actions: bool):
    obs, _ = env.reset()
    recs, rewards, actions = [], [], []
    while True:
        a = policy_fn(obs, env)
        if collect_actions:
            actions.append(float(np.asarray(a).reshape(-1)[0]))
        obs, r, term, _, info = env.step(a)
        rewards.append(r)
        viol = bool(info.get("p_em") is None)  # placeholder, replaced below
        recs.append(dict(
            cls=_classify(info["mode"], info["T_CE_cmd"]),
            t_ce=info["T_CE_cmd"], t_em=info["T_EM_cmd"],
            soc=info["soc"], p_em=info["p_em"], u=info["u"],
            viol=not (0.05 <= info["soc"] <= 0.95),
        ))
        if term:
            return recs, rewards, actions, info["episode_final"]


def evaluate(checkpoint: str | None = None, cycle: str = "NEDC",
             controller: str = "rl", eq_factor: float = 1.0, k_fb: float = 0.0,
             action_map: str = "linear", lookahead: int | None = None,
             seed: int = 0) -> dict:
    """Evaluate ANY controller under identical conditions. Returns a metric dict."""
    if controller == "rl":
        from stable_baselines3 import SAC
        model = SAC.load(checkpoint)
        if lookahead is None:
            od = int(model.observation_space.shape[0])
            lookahead = 0 if od <= 16 else od - 15
        env = EMSEnv(cycle, eq_factor=eq_factor, k_fb=k_fb,
                     action_map=action_map, lookahead=lookahead)
        fn = lambda obs, e: model.predict(obs, deterministic=True)[0]
        recs, rewards, actions, fin = _drive(env, fn, True)
        res = _summarize(recs, fin, rewards, actions)
        res["controller"] = f"SAC:{checkpoint}"

    elif controller == "rule_based":
        from src.baselines.advanced_rule_based import (
            AdvancedController, control_unit_advanced)
        env = EMSEnv(cycle, eq_factor=eq_factor, k_fb=k_fb,
                     action_map=action_map, lookahead=lookahead or 0)
        ctrl = AdvancedController(cycle_name=cycle)
        ctrl.reset()

        def patched(self, action):
            d = self._demand
            c = ctrl.step(d["w_MGB"], d["dw_MGB"], d["T_MGB"], d["gear"],
                          self._Q_BT, d["v"])
            cu = control_unit_advanced(d["w_MGB"], d["dw_MGB"], d["T_MGB"],
                                       c["u"], c["state_CE"])
            return cu["T_CE"], cu["T_EM"], c["u"], (
                "regen" if d["T_MGB"] < 0 else
                ("stop" if d["T_MGB"] == 0 or d["w_MGB"] <= 0 else
                 ("lps_gen" if cu["T_EM"] < 0 else
                  ("assist" if cu["T_EM"] > 0 else "engine"))))
        import types
        env._action_to_torques = types.MethodType(patched, env)
        recs, rewards, actions, fin = _drive(
            env, lambda o, e: np.zeros(1, np.float32), False)
        res = _summarize(recs, fin, rewards, None)
        res["controller"] = "advanced_rule_based"

    elif controller == "ecms":
        from src.baselines.ecms import _hamiltonian_best_u, ECMS_TARGET
        lam0 = {"NEDC": 1.3125, "FTP75": 2.4062}[cycle]
        env = EMSEnv(cycle, eq_factor=eq_factor, k_fb=k_fb,
                     action_map=action_map, lookahead=lookahead or 0)

        def patched(self, action):
            d = self._demand
            w, dw, T = d["w_MGB"], d["dw_MGB"], d["T_MGB"]
            soc = self._Q_BT / 36000.0
            if T == 0.0 or w <= 0.0:
                return 0.0, 0.0, 0.0, "stop"
            lam = lam0 + 8.0 * (SOC_TARGET - soc)
            u = _hamiltonian_best_u(w, dw, T, soc, lam, 81)
            t_em = u * T
            t_ce = T - t_em
            mode = ("regen" if T < 0 else
                    ("lps_gen" if t_em < 0 else
                     ("assist" if t_em > 0 else "engine")))
            return t_ce, t_em, u, mode
        import types
        env._action_to_torques = types.MethodType(patched, env)
        recs, rewards, actions, fin = _drive(
            env, lambda o, e: np.zeros(1, np.float32), False)
        res = _summarize(recs, fin, rewards, None)
        res["controller"] = f"ECMS(lam0={lam0},k_fb=8)"
    else:
        raise ValueError(f"unknown controller {controller!r}")

    res["cycle"] = cycle
    res["action_map"] = action_map
    res["seed"] = seed
    return res


def fmt(r: dict) -> str:
    L = [f"controller           : {r['controller']}",
         f"cycle / action_map   : {r['cycle']} / {r['action_map']}",
         f"V_liter              : {r['v_liter']:.4f} L/100km",
         f"V_CE_equiv           : {r['v_ce_equiv']:.4f} L/100km   <-- PRIMARY",
         f"SoC init/final/dSoC  : {r['soc_init']*100:.2f}% / {r['soc_final']*100:.2f}% / {r['d_soc_pp']:+.2f}pp",
         f"SoC min/max          : {r['soc_min']*100:.2f}% / {r['soc_max']*100:.2f}%",
         f"charge-sustaining    : {'YES' if r['charge_sustaining'] else 'NO'}",
         f"modes (moving={r['moving_steps']}) : OFF={r['off_pct']:.1f}%  ASSIST={r['assist_pct']:.1f}%  "
         f"LPS={r['lps_pct']:.1f}%  ONLY={r['only_pct']:.1f}%  REGEN={r['regen_pct']:.1f}%",
         f"constraint violations: {r['constraint_violations']}",
         f"battery throughput   : {r['battery_throughput_kJ']:.1f} kJ",
         f"engine on-time       : {r['engine_on_time_s']} s",
         f"total reward         : {r['total_reward']:.3f}"]
    if "action_mean" in r:
        L.append(f"action mean/std      : {r['action_mean']:+.4f} / {r['action_std']:.4f}  "
                 f"p25/p50/p75={r['action_p25']:+.3f}/{r['action_p50']:+.3f}/{r['action_p75']:+.3f}")
        L.append(f"action sat / |da|    : {r['action_sat_pct']:.2f}% / {r['action_delta_mean']:.4f}")
    return "\n".join("  " + x for x in L)


def main():
    p = argparse.ArgumentParser(description="Authoritative single evaluation function")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--cycle", default="NEDC", choices=["NEDC", "FTP75"])
    p.add_argument("--controller", default="rl", choices=["rl", "ecms", "rule_based"])
    p.add_argument("--eq-factor", type=float, default=1.0)
    p.add_argument("--k-fb", type=float, default=0.0)
    p.add_argument("--action-map", default="linear")
    p.add_argument("--lookahead", type=int, default=None)
    p.add_argument("--json", default=None, help="also write metrics to this JSON path")
    a = p.parse_args()
    r = evaluate(a.checkpoint, a.cycle, a.controller, a.eq_factor, a.k_fb,
                 a.action_map, a.lookahead)
    print(f"\n=== EVALUATION ===")
    print(fmt(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2))
        print(f"\n  [saved] {a.json}")


if __name__ == "__main__":
    main()
