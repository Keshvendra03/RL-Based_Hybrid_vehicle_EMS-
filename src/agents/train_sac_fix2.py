"""
train_sac.py
============
Train a SAC agent on the EMSEnv and benchmark it against the advanced
rule-based controller.

    python -m src.agents.train_sac --cycle NEDC --timesteps 300000

Key design decisions
--------------------
1. Replay-buffer pre-fill (--prefill-eps, default 20):
   Before gradient updates start, the replay buffer is filled with
   transitions from a DIVERSE SET of fixed heuristic policies
   (engine-only, mild-assist, strong-assist, mild-LPS, regen).
   This is NOT behaviour cloning — the agent never copies these policies.
   It just starts with richer off-policy data so gradients are informative
   from the very first update. Pre-fill is cheap (~5 s for 20 episodes).

2. gamma = 0.9999 (was 0.999):
   0.999^1220 = 0.30 — terminal SoC penalty discounted to 30% at episode
   start. 0.9999^1220 = 0.885 — terminal signal stays visible all episode.

3. eq_factor = 1.0 (set in EMSEnv defaults; ALIGNED):
   With a single equivalence factor the per-step reward telescopes to exactly
   (minus) the true v_ce_equiv objective, so training optimises the reported
   metric directly. (An earlier note suggested 0.42 to under-price battery
   energy; that DE-aligns the reward from the metric and was NOT adopted.)

4. SOC_DEADBAND = 0.30 (set in EMSEnv defaults):
   Free SoC swing of +/-0.30 around target; lets SoC reach the ~0.20 band the
   bank-and-spend strategy uses without per-step penalty inside the band.

Realistic target (tested & proven): the charge-sustaining ECMS optimum from
src/baselines/ecms.py — NEDC 3.189, FTP75 2.810 L/100km. These replace the
former unsubstantiated "DP ceiling" numbers as the achievable reference.

Model selection is on the TRUE objective: v_ce_equiv + charge-sustaining
penalty. The best checkpoint is kept separately from the last checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from src.env.ems_env import EMSEnv, SOC_TARGET, TERM_TOL


# --------------------------------------------------------------------------- #
# Heuristic policies for replay-buffer pre-fill
# --------------------------------------------------------------------------- #

# Fixed u values → env action a = 2*(u - U_MIN)/(U_MAX - U_MIN) - 1
_U_MIN, _U_MAX = -0.6, 1.0

def _u_to_action(u: float) -> np.ndarray:
    a = 2.0 * (u - _U_MIN) / (_U_MAX - _U_MIN) - 1.0
    return np.array([np.clip(a, -1.0, 1.0)], dtype=np.float32)


PREFILL_POLICIES = [
    ("engine_only",    _u_to_action(0.00)),   # u=0.0 engine only
    ("mild_assist",    _u_to_action(0.25)),   # u=0.25 mild electric assist
    ("strong_assist",  _u_to_action(0.60)),   # u=0.6 heavy assist
    ("full_electric",  _u_to_action(1.00)),   # u=1.0 pure electric (when possible)
    ("mild_lps",       _u_to_action(-0.20)),  # u=-0.2 mild load-point shift / charge
    ("strong_lps",     _u_to_action(-0.45)),  # u=-0.45 aggressive charging
]


def prefill_buffer(model: SAC, env: EMSEnv, n_episodes: int, verbose: bool = True) -> None:
    """Fill the SAC replay buffer with diverse heuristic rollouts.

    Cycles through PREFILL_POLICIES repeatedly until n_episodes are done.
    Each episode uses one fixed action — covering a wide range of SoC
    trajectories so the critic sees varied (s, a, r, s') from step 1.
    """
    if verbose:
        print(f"[prefill] filling buffer with {n_episodes} heuristic episodes "
              f"({len(PREFILL_POLICIES)} policies, cycling)...")

    n_policies = len(PREFILL_POLICIES)
    total_steps = 0
    for ep in range(n_episodes):
        policy_name, action = PREFILL_POLICIES[ep % n_policies]
        obs, _ = env.reset()
        ep_steps = 0
        while True:
            # Add (obs, action, reward, next_obs, done) directly to the buffer
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            model.replay_buffer.add(
                obs          = obs.reshape(1, -1),
                next_obs     = next_obs.reshape(1, -1),
                action       = action.reshape(1, -1),
                reward       = np.array([reward]),
                done         = np.array([done]),
                infos        = [info],
            )
            obs = next_obs
            ep_steps += 1
            total_steps += 1
            if done:
                break

    if verbose:
        buf_size = model.replay_buffer.size()
        print(f"[prefill] done — {total_steps} transitions in buffer "
              f"({buf_size} / {model.replay_buffer.buffer_size} capacity used)")


def prefill_buffer_benchmark(model: SAC, env: EMSEnv, cycle: str,
                             n_episodes: int, noise: float = 0.1,
                             verbose: bool = True) -> None:
    """Seed the replay buffer with the ADVANCED RULE-BASED controller's own
    rollouts (+ small Gaussian action noise), instead of constant-u policies.

    Rationale
    ---------
    The constant-u prefill only brackets the action space; it never contains a
    good *schedule*, so with the aligned (eq=1.0) reward SAC can still collapse
    into the engine-only local optimum. Seeding with the benchmark places
    ~3.5 L/100km trajectories in the buffer, so the critic has a strong
    near-optimal basin from the first gradient step. This is NOT behaviour
    cloning — SAC still explores freely and optimises the true reward; it just
    starts from informative off-policy data.
    """
    from src.baselines.advanced_rule_based import AdvancedController

    if verbose:
        print(f"[prefill] seeding buffer with {n_episodes} benchmark rollouts "
              f"(advanced rule-based, noise sigma={noise})...")

    ctrl = AdvancedController(cycle)
    total_steps = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ctrl.reset()
        while True:
            d = env._demand
            u = ctrl.step(d["w_MGB"], d["dw_MGB"], d["T_MGB"],
                          d["gear"], env._Q_BT, d["v"])["u"]
            a = 2.0 * (u - _U_MIN) / (_U_MAX - _U_MIN) - 1.0
            a = float(np.clip(a + np.random.randn() * noise, -1.0, 1.0))
            action = np.array([a], dtype=np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            model.replay_buffer.add(
                obs      = obs.reshape(1, -1),
                next_obs = next_obs.reshape(1, -1),
                action   = action.reshape(1, -1),
                reward   = np.array([reward]),
                done     = np.array([done]),
                infos    = [info],
            )
            obs = next_obs
            total_steps += 1
            if done:
                break

    if verbose:
        buf_size = model.replay_buffer.size()
        print(f"[prefill] done — {total_steps} transitions in buffer "
              f"({buf_size} / {model.replay_buffer.buffer_size} capacity used)")


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #

def rollout_deterministic(model: SAC, cycle: str, eq_factor: float,
                           soc_deadband: float) -> dict:
    """Roll the current policy out greedily; return final metrics."""
    env = EMSEnv(cycle, eq_factor=eq_factor, soc_deadband=soc_deadband)
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        if term:
            return info["episode_final"]


def score(final: dict) -> float:
    """True objective + charge-sustaining penalty (lower = better)."""
    miss = max(abs(final["soc_final"] - SOC_TARGET) - TERM_TOL, 0.0)
    return final["v_ce_equiv"] + 10.0 * miss


class EvalAndCheckpoint(BaseCallback):
    def __init__(self, cycle: str, every_steps: int, out_dir: Path,
                 eq_factor: float, soc_deadband: float, verbose: int = 1):
        super().__init__(verbose)
        self.cycle       = cycle
        self.every       = every_steps
        self.out_dir     = out_dir
        self.eq_factor   = eq_factor
        self.soc_deadband = soc_deadband
        self.best        = np.inf
        self.history     = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0:
            final = rollout_deterministic(
                self.model, self.cycle, self.eq_factor, self.soc_deadband)
            s = score(final)
            self.history.append((self.num_timesteps, final["v_liter"],
                                  final["v_ce_equiv"], final["soc_final"], s))
            tag = ""
            if s < self.best:
                self.best = s
                self.model.save(self.out_dir / "sac_ems_best")
                tag = "  <-- new best (saved)"
            if self.verbose:
                print(f"[eval @ {self.num_timesteps:>7}] "
                      f"V_liter={final['v_liter']:.3f}  "
                      f"V_CE_equiv={final['v_ce_equiv']:.3f}  "
                      f"SoC={final['soc_final']*100:5.1f}%  "
                      f"score={s:.3f}{tag}")
        return True


# --------------------------------------------------------------------------- #
# FIX 2 — Entropy annealing for fine power-split control                      #
# --------------------------------------------------------------------------- #
#
# WHY (measured, not assumed): instrumenting the ECMS optimum on this exact
# plant shows its advantage over the rule-based benchmark is NOT mode choice
# (engine-off frequency is already ~53-59% for both) and NOT regen (identical,
# hard-coded). It is the *continuous precision* of the load-point-shift split:
# ECMS sits the engine on its best-BSFC point via a fine, near-deterministic u.
# SAC's maximum-entropy objective keeps the policy deliberately stochastic, so
# the action is smeared across the engine fuel map's non-convex surface; by
# Jensen the expected fuel of a smeared u is strictly worse than the fuel of the
# deterministic optimum at exactly the threshold-heavy operating points where
# the last ~0.3 L/100km lives. We evaluate deterministically already, so the
# trained policy MEAN is what we ship — this callback aligns that mean with the
# deterministic optimum.
#
# DESIGN (lowest-risk): do NOT touch early exploration. Keep auto-alpha for the
# first `anneal_start` fraction of training (the phase your logs show escaping
# the engine-only local optimum). Then read the auto-LEARNED alpha and linearly
# drive it down to a small floor over the remaining steps. This only ever makes
# the late policy *tighter*, never noisier, and it touches the training script
# alone — the validated env / plant / reward are untouched.
#
# This is NOT guided reward and NOT cycle-specific: entropy temperature is a
# generic SAC exploration control, identical in spirit on any cycle. It injects
# no ECMS / rule-based knowledge into the objective.
class EntropyAnneal(BaseCallback):
    def __init__(self, total_timesteps: int, anneal_start: float = 0.6,
                 alpha_floor: float = 0.02, log_every: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.total       = int(total_timesteps)
        self.anneal_start = float(anneal_start)
        self.alpha_floor = float(alpha_floor)
        self.log_every   = int(log_every)
        self._alpha_at_start = None      # auto-learned alpha captured at anneal_start
        self._active = False

    def _current_alpha(self) -> float:
        import torch as th
        # SB3 stores log_ent_coef when ent_coef='auto'; exp() to get alpha.
        return float(th.exp(self.model.log_ent_coef.detach()).item())

    def _set_alpha(self, alpha: float) -> None:
        import torch as th
        with th.no_grad():
            # Freeze auto-tuning by writing log_ent_coef directly and detaching
            # its optimizer step (we stop updating it once annealing is active).
            self.model.log_ent_coef.fill_(float(np.log(max(alpha, 1e-8))))

    def _on_step(self) -> bool:
        # Only meaningful when ent_coef was 'auto' (log_ent_coef exists & is a leaf).
        if not hasattr(self.model, "log_ent_coef") or self.model.log_ent_coef is None:
            return True
        progress = self.num_timesteps / self.total
        if progress < self.anneal_start:
            return True
        if not self._active:
            self._alpha_at_start = self._current_alpha()
            self._active = True
            # stop SB3 from continuing to optimize alpha during the anneal phase
            self.model.ent_coef_optimizer = None
            if self.verbose:
                print(f"[entropy] annealing begins @ {self.num_timesteps} "
                      f"(captured auto alpha = {self._alpha_at_start:.4f} "
                      f"-> floor {self.alpha_floor:.4f})")
        frac = (progress - self.anneal_start) / (1.0 - self.anneal_start)
        frac = min(max(frac, 0.0), 1.0)
        alpha = self._alpha_at_start * (1.0 - frac) + self.alpha_floor * frac
        self._set_alpha(alpha)
        if self.log_every and self.num_timesteps % self.log_every == 0 and self.verbose:
            print(f"[entropy @ {self.num_timesteps:>7}] alpha={alpha:.4f} "
                  f"(frac={frac:.2f})")
        return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycle",        default="NEDC", choices=["NEDC", "FTP75"])
    p.add_argument("--timesteps",    type=int,   default=300_000)
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--out",          default="models")
    p.add_argument("--eq-factor",    type=float, default=1.0,
                   help="equivalence factor for battery energy in the reward. "
                        "1.0 == aligned: reward telescopes to (minus) the true "
                        "v_ce_equiv objective. (The old 0.15 asymmetric value "
                        "rewarded battery throughput and misranked policies.)")
    p.add_argument("--lambda-soc",   type=float, default=2.0,
                   help="per-step SoC-band penalty weight")
    p.add_argument("--soc-deadband", type=float, default=0.10,
                   help="free SoC swing ± around target before the per-step "
                        "penalty kicks in. 0.30 (old) gave no control (SoC locked "
                        "at 67%%); 0.05 over-controlled (fuel stuck at 3.9 above "
                        "benchmark). 0.10 bounds SoC while freeing the 45-55%% band "
                        "for bank-and-spend regen capture.")
    p.add_argument("--prefill-mode", default="benchmark",
                   choices=["benchmark", "constant"],
                   help="'benchmark' seeds the buffer with advanced rule-based "
                        "rollouts (recommended); 'constant' uses the 6 fixed-u "
                        "heuristic policies.")
    p.add_argument("--prefill-eps",  type=int,   default=20,
                   help="heuristic episodes to pre-fill replay buffer before training "
                        "(0 = disabled; 20 ≈ 5 s, covers all 6 fixed policies)")
    p.add_argument("--eval-every-eps", type=int, default=2,
                   help="print a deterministic eval every N episodes")
    p.add_argument("--resume",       action="store_true",
                   help="continue from <out>/sac_ems_last (+ replay buffer)")
    # --- FIX 2: entropy annealing ---
    p.add_argument("--anneal-entropy", action="store_true",
                   help="FIX 2: after --anneal-start fraction of training, "
                        "linearly drive the auto-learned SAC entropy coef down to "
                        "--alpha-floor, tightening the policy for fine power-split "
                        "control (load-point-shift precision). Touches training "
                        "only; env/plant/reward unchanged.")
    p.add_argument("--anneal-start", type=float, default=0.6,
                   help="fraction of total timesteps after which to begin "
                        "entropy annealing (default 0.6 — keeps full auto-alpha "
                        "exploration for the first 60%% as in the baseline run)")
    p.add_argument("--alpha-floor", type=float, default=0.02,
                   help="entropy coefficient floor reached at end of training "
                        "(default 0.02; smaller = tighter terminal policy)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = EMSEnv(args.cycle,
                 eq_factor=args.eq_factor,
                 lambda_soc=args.lambda_soc,
                 soc_deadband=args.soc_deadband)
    ep_len = env.cycle.length - 1

    if args.resume and (out_dir / "sac_ems_last.zip").exists():
        model = SAC.load(out_dir / "sac_ems_last", env=env)
        buf = out_dir / "replay_buffer.pkl"
        if buf.exists():
            model.load_replay_buffer(buf)
        print(f"[resume] loaded sac_ems_last  "
              f"(buffer: {model.replay_buffer.size()} transitions)")
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate    = 3e-4,
            buffer_size      = 300_000,
            learning_starts  = max(2 * ep_len, args.prefill_eps * ep_len),
            batch_size       = 512,
            tau              = 0.005,
            gamma            = 0.9999,   # long episodes: 0.9999^N >> 0.999^N
            train_freq       = 64,
            gradient_steps   = 64,
            ent_coef         = "auto",
            policy_kwargs    = dict(net_arch=[256, 256]),
            seed             = args.seed,
            verbose          = 0,
        )

        # Pre-fill replay buffer with diverse heuristic rollouts BEFORE training.
        # This is NOT guided learning — SAC explores freely from step 1.
        # Pre-fill just ensures the critic has seen varied (s, a, r, s')
        # from the start instead of spending 30+ episodes on random exploration.
        if args.prefill_eps > 0:
            if args.prefill_mode == "benchmark":
                prefill_buffer_benchmark(model, env, args.cycle,
                                         n_episodes=args.prefill_eps, verbose=True)
            else:
                prefill_buffer(model, env, n_episodes=args.prefill_eps, verbose=True)

    cb = EvalAndCheckpoint(
        args.cycle,
        every_steps  = args.eval_every_eps * ep_len,
        out_dir      = out_dir,
        eq_factor    = args.eq_factor,
        soc_deadband = args.soc_deadband,
        verbose      = 1,
    )
    best_file = out_dir / "best_score.txt"
    if args.resume and best_file.exists():
        cb.best = float(best_file.read_text())

    print(f"\n[train] {args.timesteps:,} steps  |  cycle={args.cycle}  "
          f"eq_factor={args.eq_factor}  deadband={args.soc_deadband}")
    benchmarks  = {"NEDC": 3.506,  "FTP75": 3.232}
    # Realistic, TESTED & PROVEN near-optimal target from the ECMS ruler
    # (src/baselines/ecms.py), charge-sustaining at SoC_end == 50%. This REPLACES
    # the former "DP ceiling" constants, which had no solver behind them in this
    # repo. Reproduce with: python -m src.baselines.ecms --cycle NEDC / FTP75
    ecms_targets = {"NEDC": 3.1887,  "FTP75": 2.8097}   # proven, charge-sustaining
    ecms_target = ecms_targets.get(args.cycle, "?")
    benchmark  = benchmarks.get(args.cycle, "?")
    print(f"[train] Benchmark: {benchmark} L/100km ({args.cycle} advanced rule-based)")
    print(f"[train] ECMS target (tested & proven, charge-sustaining): {ecms_target} L/100km\n")

    # FIX 2: assemble callbacks. EvalAndCheckpoint always runs; EntropyAnneal is
    # opt-in via --anneal-entropy and only affects the LATE policy (tightening).
    callbacks = [cb]
    if args.anneal_entropy:
        from stable_baselines3.common.callbacks import CallbackList
        ent_cb = EntropyAnneal(
            total_timesteps = args.timesteps,
            anneal_start    = args.anneal_start,
            alpha_floor     = args.alpha_floor,
            log_every       = ep_len * args.eval_every_eps,
            verbose         = 1,
        )
        callbacks.append(ent_cb)
        print(f"[train] FIX 2 active: entropy anneal from {args.anneal_start:.0%} "
              f"of training down to alpha_floor={args.alpha_floor}\n")
        run_callback = CallbackList(callbacks)
    else:
        run_callback = cb

    model.learn(
        total_timesteps    = args.timesteps,
        callback           = run_callback,
        progress_bar       = False,
        reset_num_timesteps = not args.resume,
    )
    model.save(out_dir / "sac_ems_last")
    model.save_replay_buffer(out_dir / "replay_buffer.pkl")

    # Final evaluation
    final = rollout_deterministic(model, args.cycle, args.eq_factor, args.soc_deadband)
    s = score(final)
    if s < cb.best:
        cb.best = s
        model.save(out_dir / "sac_ems_best")
    best_file.write_text(str(cb.best))

    print(f"\n[final eval] V_liter={final['v_liter']:.3f}  "
          f"V_CE_equiv={final['v_ce_equiv']:.3f}  "
          f"SoC={final['soc_final']*100:.1f}%  score={s:.3f}")
    print(f"\nBest score:  {cb.best:.4f}  (benchmark={benchmark}, ECMS target={ecms_target})")
    print(f"Saved:       {out_dir / 'sac_ems_best.zip'}")


if __name__ == "__main__":
    main()
