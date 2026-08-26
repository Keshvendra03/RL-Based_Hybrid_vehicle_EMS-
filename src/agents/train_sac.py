"""
train_sac.py
============
Single canonical entry point for training a SAC agent on EMSEnv, unguided,
and benchmarking it against the advanced rule-based controller / ECMS.

    python -m src.agents.train_sac --cycle NEDC --timesteps 1500000

This file replaces train_sac_fix2.py / train_sac_nstep.py / train_sac_lookahead.py
(all deleted — they were near-duplicate forks of this script; keeping several
divergent copies made it impossible to know which config produced a given
checkpoint). Everything they offered is available here via flags.

UNGUIDED BY DESIGN
-------------------
There is NO behaviour-cloning and NO benchmark-seeded replay buffer in this
script (the old `--prefill-mode benchmark`, which seeded the buffer with the
advanced rule-based controller's own rollouts, has been removed entirely —
that is guided learning). The only optional pre-fill (`--prefill-mode
constant`, default OFF) plays a handful of FIXED, arbitrary torque-split
values (engine-only, half-assist, full-electric, etc.) into the buffer before
training starts; it never contains a policy or a schedule, only isolated
(s, a, r, s') tuples spanning the action range, so the critic isn't staring
at an empty buffer for the first `learning_starts` steps. Default is 0
episodes: a genuinely blank-slate run.
`src/agents/pretrain_bc.py` / `finetune_bcreg.py` / `finetune_sac.py` remain
in the repo as a GUIDED alternative for comparison, but are not part of this
pipeline and are not invoked here.

WHY THESE HYPERPARAMETERS
--------------------------
1. gamma = 0.9999 (not 0.999): episodes are ~1220-1877 steps; 0.999^1220 =
   0.30 (the terminal charge-sustaining signal is 70% decayed by the time it
   reaches early-episode transitions), vs 0.9999^1220 = 0.885. The whole
   reward's terminal SoC term would otherwise be nearly invisible to the
   critic at the start of the episode.
2. lookahead (default 5, causal): appends the next 5 PRESCRIBED speeds from
   the drive cycle (route/ADAS-preview information, not controller
   knowledge) to the observation, replacing the absolute cycle-progress
   scalar (a cycle fingerprint that hurts generalization). This is standard
   partial-observability handling, not guidance: the info comes from the
   environment's own future demand, never from a controller's action.
3. n-step returns (default n=5): diagnosed via mode_breakdown_rl.py that
   trained agents park ~24% of moving time in "engine + small motor assist"
   where the near-optimal controllers spend ~0% — those two actions are
   adjacent in action space and differ by a tiny per-step fuel amount, so a
   1-step TD critic under gamma=0.9999 can't tell them apart. n-step returns
   inject several real rewards into the bootstrap target, sharpening exactly
   that ranking. This changes ONLY how existing reward is propagated into the
   critic target -- it does not touch env/reward/action semantics.
4. --per switches to Prioritized Experience Replay (Schaul et al., 2016)
   instead of n-step (the two replay-buffer overrides are mutually exclusive
   in this script; combine only if you're prepared to verify a merged
   buffer/algorithm class yourself). PER prioritizes transitions by TD error,
   which speeds convergence on CPU-bound training. It was implemented
   (src/agents/per.py) but never wired into any training script before this.
5. eq_factor = 1.0 (env default): the reward telescopes EXACTLY to (minus)
   the true v_ce_equiv metric being optimized against the benchmark, so
   training objective == evaluation objective.

Realistic targets (tested & proven, src/baselines/ecms.py):
    NEDC  rule-based 3.506 | ECMS 3.189      FTP75  rule-based 3.232 | ECMS 2.810
Primary goal: beat the rule-based benchmark. ECMS is a stretch ceiling (its
lambda is tuned with whole-cycle information a causal controller doesn't
have, so it isn't a strictly fair target for an online agent).

BOOKKEEPING FIX
----------------
Checkpoints are written under <out>/<cycle>/ (previously a single shared
models/ directory was used for different cycles and configs, which is how
models/best_score.txt ended up describing a checkpoint that had since been
overwritten by a different, worse run -- the file and the .zip had silently
drifted apart). best_score.txt is now written next to its checkpoint, in the
same directory, immediately after the checkpoint save, and a run_config.json
sidecar records the exact CLI args + git commit that produced it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback

from src.agents.nstep_sac import NStepSAC, NStepReplayBuffer
from src.agents.per import SACPER, PrioritizedReplayBuffer
from src.env.ems_env import EMSEnv, SOC_TARGET, TERM_TOL
from src.agents.instrumentation import SACDiagnostics, CheckpointRule

# --------------------------------------------------------------------------- #
# Tested & proven reference numbers (src/baselines/ecms.py, evaluate_advanced.py)
# --------------------------------------------------------------------------- #
RULE_BASED_BENCHMARK = {"NEDC": 3.506, "FTP75": 3.232}
ECMS_TARGET = {"NEDC": 3.1887, "FTP75": 2.8097}  # proven, charge-sustaining


# --------------------------------------------------------------------------- #
# Optional (non-guided) replay-buffer pre-fill: fixed, arbitrary torque splits
# --------------------------------------------------------------------------- #

_U_MIN, _U_MAX = -0.85, 1.0  # must match EMSEnv.U_MIN / U_MAX


def _u_to_action(u: float) -> np.ndarray:
    a = 2.0 * (u - _U_MIN) / (_U_MAX - _U_MIN) - 1.0
    return np.array([np.clip(a, -1.0, 1.0)], dtype=np.float32)


PREFILL_POLICIES = [
    ("engine_only",   _u_to_action(0.00)),
    ("mild_assist",   _u_to_action(0.25)),
    ("strong_assist", _u_to_action(0.60)),
    ("full_electric", _u_to_action(1.00)),
    ("mild_lps",      _u_to_action(-0.20)),
    ("strong_lps",    _u_to_action(-0.60)),
]


def prefill_buffer(model: SAC, env: EMSEnv, n_episodes: int, verbose: bool = True) -> None:
    """Seed the replay buffer with FIXED, arbitrary torque-split episodes.

    Not guidance: none of these six actions is a policy or a schedule --
    each episode holds ONE constant action for its whole length, so the
    buffer only gains isolated (s, a, r, s') coverage of the action range,
    never a good trajectory to imitate.
    """
    if verbose:
        print(f"[prefill] {n_episodes} fixed-action episodes "
              f"({len(PREFILL_POLICIES)} actions, cycling)...")
    n_policies = len(PREFILL_POLICIES)
    total_steps = 0
    for ep in range(n_episodes):
        _, action = PREFILL_POLICIES[ep % n_policies]
        obs, _ = env.reset()
        while True:
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            model.replay_buffer.add(
                obs=obs.reshape(1, -1), next_obs=next_obs.reshape(1, -1),
                action=action.reshape(1, -1), reward=np.array([reward]),
                done=np.array([done]), infos=[info],
            )
            obs = next_obs
            total_steps += 1
            if done:
                break
    if verbose:
        print(f"[prefill] done — {total_steps} transitions in buffer "
              f"({model.replay_buffer.size()} / {model.replay_buffer.buffer_size} capacity)")


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #

def rollout_deterministic(model: SAC, cycle: str, eq_factor: float = 1.0,
                           soc_deadband: float = 0.10, lookahead: int = 0,
                           k_fb: float = 0.0, action_map: str = "linear",
                           obs_clean: bool = False) -> dict:
    """Roll the current policy out greedily; return final metrics."""
    env = EMSEnv(cycle, eq_factor=eq_factor, soc_deadband=soc_deadband,
                 lookahead=lookahead, k_fb=k_fb, action_map=action_map,
                 obs_clean=obs_clean)
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


def _write_best_score(out_dir: Path, value: float) -> None:
    """Atomic write so best_score.txt can never describe a half-written or
    stale checkpoint (previously a plain write left a window where a crash
    or a differently-configured run could leave the file and the .zip
    describing two different policies)."""
    tmp = out_dir / "best_score.txt.tmp"
    tmp.write_text(str(value))
    tmp.replace(out_dir / "best_score.txt")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


_EVAL_CSV_FIELDS = ["timesteps", "cycle", "v_liter", "v_ce_equiv", "soc_final",
                    "cycle_score", "mean_score", "is_best",
                    "rule_based_benchmark", "ecms_target"]


class EvalAndCheckpoint(BaseCallback):
    """Periodic deterministic eval + checkpointing.

    Every eval is appended to <out_dir>/eval_history.csv as it happens (not
    just kept in self.history in memory), specifically so that a run stopped
    early (e.g. via TaskStop) still leaves an analyzable record on disk --
    see results/figures.py / results/checkpoints.py for the reader side.
    """

    def __init__(self, cycles, every_steps: int, out_dir: Path,
                 eq_factor: float, soc_deadband: float, lookahead: int = 0,
                 k_fb: float = 0.0, action_map: str = "linear",
                 obs_clean: bool = False, verbose: int = 1):
        super().__init__(verbose)
        # cycles: single cycle name or list (multi-cycle). Model selection uses
        # the MEAN score across all eval cycles -> best CROSS-CYCLE policy.
        self.cycles = cycles if isinstance(cycles, list) else [cycles]
        self.every = every_steps
        self.out_dir = out_dir
        self.eq_factor = eq_factor
        self.soc_deadband = soc_deadband
        self.lookahead = lookahead
        self.k_fb = k_fb
        self.action_map = action_map
        self.obs_clean = obs_clean
        self.rule = CheckpointRule(soc_tol=TERM_TOL)
        self.best = np.inf
        self.history = []
        self.csv_path = out_dir / "eval_history.csv"
        if not self.csv_path.exists():
            import csv
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=_EVAL_CSV_FIELDS).writeheader()

    def _append_csv(self, rows: list[dict]) -> None:
        import csv
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_EVAL_CSV_FIELDS)
            for row in rows:
                w.writerow(row)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0:
            finals = [rollout_deterministic(self.model, c, self.eq_factor,
                                             self.soc_deadband, self.lookahead,
                                             self.k_fb, self.action_map,
                                             self.obs_clean)
                      for c in self.cycles]
            scores = [score(f) for f in finals]
            s = float(np.mean(scores))
            self.history.append((self.num_timesteps, s))
            tag = ""
            is_best = s < self.best
            if is_best:
                self.best = s
                self.model.save(self.out_dir / "sac_ems_best")
                _write_best_score(self.out_dir, self.best)
                tag = "  <-- new best (saved)"

            for c, f in zip(self.cycles, finals):
                self.rule.offer(step=self.num_timesteps, cycle=c, seed=None,
                                v_ce_equiv=f["v_ce_equiv"],
                                soc_final=f["soc_final"], violations=0)
            self.rule.save(self.out_dir)

            self._append_csv([
                {
                    "timesteps": self.num_timesteps,
                    "cycle": c,
                    "v_liter": f["v_liter"],
                    "v_ce_equiv": f["v_ce_equiv"],
                    "soc_final": f["soc_final"],
                    "cycle_score": sc,
                    "mean_score": s,
                    "is_best": is_best,
                    "rule_based_benchmark": RULE_BASED_BENCHMARK.get(c, ""),
                    "ecms_target": ECMS_TARGET.get(c, ""),
                }
                for c, f, sc in zip(self.cycles, finals, scores)
            ])

            if self.verbose:
                parts = "  ".join(
                    f"{c}:{f['v_ce_equiv']:.3f}(SoC{f['soc_final']*100:.0f}%)"
                    for c, f in zip(self.cycles, finals))
                print(f"[eval @ {self.num_timesteps:>8}] {parts}  "
                      f"mean_score={s:.3f}{tag}")
        return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycle", default="NEDC", choices=["NEDC", "FTP75"])
    p.add_argument("--cycles", default=None,
                   help="comma-separated cycles to interleave per episode, e.g. "
                        "NEDC,FTP75 (overrides --cycle). Round-robins the training "
                        "cycle each reset for cross-cycle generalization.")
    p.add_argument("--timesteps", type=int, default=1_500_000,
                   help="default raised from 300k: at gamma=0.9999 over "
                        "~1220-1877-step episodes, 300k steps (~245 NEDC "
                        "episodes) is thin for long-horizon SoC credit "
                        "assignment from a blank policy.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models")
    p.add_argument("--eq-factor", type=float, default=1.0)
    p.add_argument("--target-entropy", default="auto",
                   help="SAC target entropy. 'auto' = -dim(A) = -1.0 (SB3 "
                        "heuristic). More negative permits a more "
                        "deterministic policy. Phase-2 section 16.")
    p.add_argument("--obs-clean", action="store_true",
                   help="drop the 2 dead observation channels (v_next is "
                        "byte-identical to fut_v1; gear_oh6 is always 0). "
                        "20 -> 18 dims. Phase-2 section 21 ablation.")
    p.add_argument("--algo", default="sac", choices=["sac", "td3"],
                   help="td3 = deterministic policy, no entropy term. "
                        "Phase-2 section 26 secondary comparison.")
    p.add_argument("--gamma", type=float, default=0.9999,
                   help="discount factor. 0.9999 was justified in-code by the "
                        "need to propagate the TERMINAL SoC signal; that term "
                        "was measured at 0.77%% of episode reward, voiding the "
                        "justification. High gamma inflates Q magnitude and "
                        "variance: the critic's RMS TD residual (1.611) exceeds "
                        "the total Q variation across the action range "
                        "(0.18-1.08), so the actor cannot rank actions. "
                        "See RL_DIAGNOSTIC_REPORT.md.")
    p.add_argument("--action-map", default="linear", choices=["linear", "modeaware"],
                   help="action->u reparameterization. 'linear' (default) is the "
                        "original mapping. 'modeaware' allocates fixed fractions "
                        "of the action range to LPS/ASSIST/OFF by anchoring on the "
                        "true OFF boundary u_thr=1-T_CUTOFF/T_MGB, making the OFF "
                        "band a state-invariant 40% instead of a moving 9-12% "
                        "sliver. Identical reachable control set -- proved in "
                        "tests/test_action_mapping.py.")
    p.add_argument("--k-fb", type=float, default=0.0,
                   help="ECMS-style closed-loop costate feedback on eq_factor: "
                        "eq_factor_eff = eq_factor + k_fb*(0.5 - soc). 0.0 "
                        "(default) reproduces the old flat-price behavior "
                        "exactly. ecms.py's proven charge-sustaining value is "
                        "8.0 -- see VERIFIED_FACTS.md for why a flat price "
                        "(k_fb=0) provably cannot hit the SoC target on this "
                        "plant even for the optimal controller.")
    p.add_argument("--lambda-soc", type=float, default=2.0)
    p.add_argument("--soc-deadband", type=float, default=0.10)
    p.add_argument("--lookahead", type=int, default=5,
                   help="causal upcoming-speed window appended to the "
                        "observation (0 disables, = original 16-dim obs)")
    p.add_argument("--n-step", type=int, default=5,
                   help="n-step return horizon for the critic target. "
                        "1 = stock single-step SAC. Mutually exclusive with --per.")
    p.add_argument("--per", action="store_true",
                   help="use Prioritized Experience Replay instead of n-step "
                        "returns. Mutually exclusive with --n-step > 1.")
    p.add_argument("--prefill-mode", default="none", choices=["none", "constant"],
                   help="'none' (default): blank-slate, no pre-fill at all. "
                        "'constant': seed the buffer with fixed-action episodes "
                        "(NOT guidance -- no policy/schedule, see module "
                        "docstring). The old 'benchmark' mode (seeding with the "
                        "rule-based controller's rollouts) has been removed.")
    p.add_argument("--prefill-eps", type=int, default=0,
                   help="episodes for --prefill-mode constant (ignored if 'none')")
    p.add_argument("--eval-every-eps", type=int, default=2)
    p.add_argument("--gradient-steps", type=int, default=64,
                   help="SAC gradient updates per train_freq=64 env steps "
                        "(default 64 = 1:1, the original setting). Lower "
                        "values reduce update aggressiveness -- see "
                        "VERIFIED_FACTS.md 2026-08-26 TensorBoard finding: "
                        "critic_loss diverges on FTP75 under gamma=0.9999 "
                        "at the default 64, worth testing whether this is "
                        "why.")
    p.add_argument("--no-tensorboard", action="store_true",
                   help="disable TensorBoard logging (on by default under <out>/tb)")
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>/<cycle>/sac_ems_last (+ replay buffer)")
    args = p.parse_args()

    if args.n_step > 1 and args.per:
        raise SystemExit("--n-step > 1 and --per are mutually exclusive in this "
                          "script (each overrides the replay buffer + train() "
                          "loop independently; combine only behind a verified "
                          "merged implementation).")

    cycle_list = [c.strip() for c in args.cycles.split(",")] if args.cycles else [args.cycle]
    run_name = args.cycle if len(cycle_list) == 1 else "multi_" + "_".join(cycle_list)
    out_dir = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_env(cname):
        return EMSEnv(cname, eq_factor=args.eq_factor, lambda_soc=args.lambda_soc,
                      soc_deadband=args.soc_deadband, lookahead=args.lookahead,
                      k_fb=args.k_fb, action_map=args.action_map,
                      obs_clean=args.obs_clean)

    if len(cycle_list) > 1:
        from gymnasium import Wrapper

        class MultiCycle(Wrapper):
            def __init__(self, cnames):
                self._envs = [make_env(c) for c in cnames]
                self._i = 0
                super().__init__(self._envs[0])

            def reset(self, **kw):
                self.env = self._envs[self._i % len(self._envs)]
                self._i += 1
                return self.env.reset(**kw)

        env = MultiCycle(cycle_list)
        ep_len = max(e.cycle.length for e in env._envs) - 1
        print(f"[train] MULTI-CYCLE interleave: {cycle_list}")
    else:
        env = make_env(args.cycle)
        ep_len = env.cycle.length - 1

    print(f"[train] lookahead={args.lookahead}  obs_dim={env.observation_space.shape[0]}  "
          f"n_step={args.n_step}  per={args.per}")

    tb_log = None if args.no_tensorboard else str(out_dir / "tb")

    if args.algo == "td3":
        from stable_baselines3 import TD3
        model_cls, buf_cls = TD3, ReplayBuffer
    elif args.n_step > 1:
        model_cls, buf_cls = NStepSAC, NStepReplayBuffer
    elif args.per:
        model_cls, buf_cls = SACPER, PrioritizedReplayBuffer
    else:
        model_cls, buf_cls = SAC, ReplayBuffer

    if args.resume and (out_dir / "sac_ems_last.zip").exists():
        model = model_cls.load(out_dir / "sac_ems_last", env=env, tensorboard_log=tb_log)
        buf = out_dir / "replay_buffer.pkl"
        if buf.exists():
            model.load_replay_buffer(buf)
        print(f"[resume] loaded sac_ems_last  (buffer: {model.replay_buffer.size()} transitions)")
    else:
        extra_buf_kwargs = {}
        if buf_cls is NStepReplayBuffer:
            extra_buf_kwargs = dict(n_step=args.n_step, gamma=args.gamma)
        model = model_cls(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=300_000,
            learning_starts=max(2 * ep_len, args.prefill_eps * ep_len),
            batch_size=512,
            tau=0.005,
            gamma=args.gamma,
            train_freq=64,
            gradient_steps=args.gradient_steps,
            **({} if args.algo == "td3" else dict(
                ent_coef="auto",
                target_entropy=("auto" if args.target_entropy == "auto"
                                else float(args.target_entropy)))),
            policy_kwargs=dict(net_arch=[256, 256]),
            seed=args.seed,
            verbose=0,
            tensorboard_log=tb_log,
            replay_buffer_class=buf_cls if buf_cls is not ReplayBuffer else None,
            replay_buffer_kwargs=extra_buf_kwargs or None,
        )

        if args.prefill_mode == "constant" and args.prefill_eps > 0:
            prefill_buffer(model, env if len(cycle_list) == 1 else env._envs[0],
                           n_episodes=args.prefill_eps, verbose=True)

    cb = EvalAndCheckpoint(
        cycle_list,
        every_steps=args.eval_every_eps * ep_len,
        out_dir=out_dir,
        eq_factor=args.eq_factor,
        soc_deadband=args.soc_deadband,
        lookahead=args.lookahead,
        k_fb=args.k_fb,
        action_map=args.action_map,
        obs_clean=args.obs_clean,
        verbose=1,
    )
    best_file = out_dir / "best_score.txt"
    if args.resume and best_file.exists():
        cb.best = float(best_file.read_text())

    print(f"\n[train] {args.timesteps:,} steps  |  run={run_name}  "
          f"eq_factor={args.eq_factor}  k_fb={args.k_fb}  action_map={args.action_map}  gamma={args.gamma}  deadband={args.soc_deadband}")
    for c in cycle_list:
        print(f"[train] {c}: rule-based benchmark {RULE_BASED_BENCHMARK.get(c,'?')}  "
              f"ECMS target {ECMS_TARGET.get(c,'?')}")

    (out_dir / "run_config.json").write_text(json.dumps(
        {**vars(args), "git_commit": _git_commit(), "obs_dim": int(env.observation_space.shape[0])},
        indent=2))

    from stable_baselines3.common.callbacks import CallbackList
    diag = SACDiagnostics(out_dir, batch_size=512, log_every=5000)
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([cb, diag]),
        progress_bar=False,
        reset_num_timesteps=not args.resume,
        # log_interval is in EPISODES, not steps: SB3's TensorBoard writer only
        # flushes at dump_logs() calls (every `log_interval` episodes) and on
        # clean shutdown. A killed/crashed run loses everything buffered since
        # the last flush -- confirmed by a controlled test where a clean 30k-step
        # exit produced 6 regular dumps but a run killed externally at 420k
        # steps had only the FIRST dump (from step 4880) on disk. log_interval=1
        # minimizes how much is lost if a run is stopped abruptly again; it does
        # not affect training, only how often already-computed metrics are
        # written out. eval_history.csv is unaffected either way (synchronous
        # per-eval file writes, not a buffered background writer).
        log_interval=1,
    )
    model.save(out_dir / "sac_ems_last")
    model.save_replay_buffer(out_dir / "replay_buffer.pkl")

    finals = [rollout_deterministic(model, c, args.eq_factor, args.soc_deadband,
                                     args.lookahead, args.k_fb, args.action_map,
                                     args.obs_clean)
              for c in cycle_list]
    s = float(np.mean([score(f) for f in finals]))
    if s < cb.best:
        cb.best = s
        model.save(out_dir / "sac_ems_best")
    _write_best_score(out_dir, cb.best)

    for c, fin in zip(cycle_list, finals):
        print(f"\n[final eval {c}] V_liter={fin['v_liter']:.3f}  "
              f"V_CE_equiv={fin['v_ce_equiv']:.3f}  SoC={fin['soc_final']*100:.1f}%")
    print(f"\nBest mean score: {cb.best:.4f}")
    print(f"Saved: {out_dir / 'sac_ems_best.zip'}")


if __name__ == "__main__":
    main()
