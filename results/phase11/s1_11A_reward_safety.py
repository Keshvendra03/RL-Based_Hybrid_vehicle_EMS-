"""
11A -- REWARD / SAFETY FORENSIC AUDIT  (NO TRAINING, NO CODE CHANGE).

Verify the eq_eff sign-inversion threshold from the ACTUAL source, then scan
every available CONTROL trajectory + replay buffer for triggering steps.

eq_eff = eq_factor + k_fb*(SOC_TARGET - soc_before)          (ems_env.py step())
       < 0   <=>   soc_before > SOC_TARGET + eq_factor/k_fb
reward = -reward_scale * (fuel_liters + eq_eff * elec_liters) (ems_env.py step())

Outputs: results/phase11/data/s1_11A.json + console.
"""
import copy, json, warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")

from stable_baselines3 import SAC
from src.env.ems_env import EMSEnv, SOC_TARGET
from src.env.powertrain import _Q_BT_0

CTRL = dict(action_map="modeaware_gated", k_fb=2.5,
            eq_factor={"NEDC": 0.2717, "FTP75": 0.4981}, lookahead=5)
CKPTS = {
    "NEDC": ["models_p5s0_k2.5/NEDC", "models_p5_k2.5/NEDC", "models_p5_k2.5_s2/NEDC"],
    "FTP75": ["models_p5f_k2.5_s0/FTP75", "models_p5f_k2.5_s1/FTP75", "models_p5f_k2.5_s2/FTP75"],
}


def threshold(cycle):
    return SOC_TARGET + CTRL["eq_factor"][cycle] / CTRL["k_fb"]


def scan_rollout(cycle, run_dir):
    """Deterministic CONTROL rollout; record SoC and eq_eff per step."""
    m = SAC.load(f"{run_dir}/sac_ems_best")
    env = EMSEnv(cycle, eq_factor=CTRL["eq_factor"][cycle], k_fb=CTRL["k_fb"],
                 action_map=CTRL["action_map"], lookahead=CTRL["lookahead"])
    obs, _ = env.reset()
    thr = threshold(cycle)
    socs, eqeff = [], []
    while True:
        soc_before = env._Q_BT / _Q_BT_0
        eqe = CTRL["eq_factor"][cycle] + CTRL["k_fb"] * (SOC_TARGET - soc_before)
        socs.append(soc_before); eqeff.append(eqe)
        a, _ = m.predict(obs, deterministic=True)
        obs, r, term, _, info = env.step(a)
        if term:
            break
    socs = np.array(socs); eqeff = np.array(eqeff)
    return dict(
        n_steps=int(len(socs)), soc_min=float(socs.min()), soc_max=float(socs.max()),
        soc_median=float(np.median(socs)),
        n_soc_gt_threshold=int((socs > thr).sum()),
        n_eqeff_lt_0=int((eqeff < 0).sum()),
        min_eqeff=float(eqeff.min()),
    )


def scan_replay(cycle, run_dir):
    m = SAC.load(f"{run_dir}/sac_ems_best")
    try:
        m.load_replay_buffer(f"{run_dir}/replay_buffer.pkl")
    except Exception as e:
        return {"error": f"replay load failed: {e}"}
    rb = m.replay_buffer
    n = rb.buffer_size if rb.full else rb.pos
    obs = rb.observations[:n, 0, :]           # (n, obs_dim)
    soc_before = (obs[:, 4] + 1.0) / 2.0      # obs[4] = 2*soc - 1
    thr = threshold(cycle)
    eqe = CTRL["eq_factor"][cycle] + CTRL["k_fb"] * (SOC_TARGET - soc_before)
    return dict(
        n_transitions=int(n), soc_min=float(soc_before.min()),
        soc_max=float(soc_before.max()), soc_median=float(np.median(soc_before)),
        n_soc_gt_threshold=int((soc_before > thr).sum()),
        n_eqeff_lt_0=int((eqe < 0).sum()),
        min_eqeff=float(eqe.min()),
    )


if __name__ == "__main__":
    out = {"source_expression": {
        "eq_eff": "eq_factor + k_fb*(SOC_TARGET - soc_before)  [ems_env.py EMSEnv.step, soc_before = pre-decision SoC]",
        "reward": "-reward_scale * (fuel_liters + eq_factor_eff * elec_liters)  [ems_env.py EMSEnv.step]",
        "fuel_liters": "dm_fuel * K_FUEL_L_PER_KG ; dm_fuel = tank trapezoidal m_dot_fuel [kg]",
        "elec_liters": "dE_batt * K_ELEC_L_PER_J ; dE_batt = E(Q_prev) - E(Q_now), signed (+ = discharge)",
        "sign_inversion_condition": "eq_eff < 0  <=>  soc_before > SOC_TARGET + eq_factor/k_fb",
    }, "note_WLTP": "No WLTP cycle exists in this repo. Available cycles: NEDC, FTP75.",
       "prior_report_threshold_STALE": {
           "reported": "66.41% (NEDC) / 80.06-80.08% (FTP75)  [RL_DIAGNOSTIC P2 / Phase 2 sec18]",
           "why_stale": "those were computed with the OLD config eq_factor=1.3125, k_fb=8.0 "
                        "(0.5 + 1.3125/8 = 0.6641). Phase 2 sec18 note 5 claimed dividing both by "
                        "4.8309 keeps the ratio -- but CONTROL k_fb is 2.5, NOT 8.0/4.8309=1.656, "
                        "so the ratio eq_factor/k_fb changed and the threshold MOVED.",
       },
       "actual_CONTROL_threshold": {}, "rollouts": {}, "replay_buffers": {}}

    for cyc in ("NEDC", "FTP75"):
        thr = threshold(cyc)
        out["actual_CONTROL_threshold"][cyc] = round(thr, 5)
        out["rollouts"][cyc] = {}
        out["replay_buffers"][cyc] = {}
        for rd in CKPTS[cyc]:
            out["rollouts"][cyc][rd] = scan_rollout(cyc, rd)
            out["replay_buffers"][cyc][rd] = scan_replay(cyc, rd)

    Path("results/phase11/data").mkdir(parents=True, exist_ok=True)
    Path("results/phase11/data/s1_11A.json").write_text(json.dumps(out, indent=2))

    for cyc in ("NEDC", "FTP75"):
        thr = out["actual_CONTROL_threshold"][cyc]
        print(f"\n===== {cyc}   eq_eff<0 threshold = SoC > {thr*100:.2f}%  "
              f"(eq_factor {CTRL['eq_factor'][cyc]}, k_fb {CTRL['k_fb']})   "
              f"[prior reports said {'66.41' if cyc=='NEDC' else '80.08'}% -- STALE]")
        print("  -- CONTROL deterministic rollouts")
        for rd, r in out["rollouts"][cyc].items():
            print(f"     {rd:<28} steps={r['n_steps']:>4}  SoC[min,med,max]="
                  f"[{r['soc_min']*100:.1f}, {r['soc_median']*100:.1f}, {r['soc_max']*100:.1f}]%  "
                  f"n(SoC>thr)={r['n_soc_gt_threshold']}  n(eq_eff<0)={r['n_eqeff_lt_0']}  "
                  f"min eq_eff={r['min_eqeff']:.4f}")
        print("  -- CONTROL replay buffers")
        for rd, r in out["replay_buffers"][cyc].items():
            if "error" in r:
                print(f"     {rd:<28} {r['error']}"); continue
            print(f"     {rd:<28} n={r['n_transitions']:>6}  SoC[min,med,max]="
                  f"[{r['soc_min']*100:.1f}, {r['soc_median']*100:.1f}, {r['soc_max']*100:.1f}]%  "
                  f"n(SoC>thr)={r['n_soc_gt_threshold']}  n(eq_eff<0)={r['n_eqeff_lt_0']}  "
                  f"min eq_eff={r['min_eqeff']:.4f}")
    print("\n[saved] results/phase11/data/s1_11A.json")
