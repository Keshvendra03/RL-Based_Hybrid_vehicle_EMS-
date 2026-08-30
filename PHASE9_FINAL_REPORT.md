# PHASE 9 — CRITIC VALUE-FIDELITY FORENSICS

## Distinguish OOD-OFF Overvaluation from High-Load Undervaluation

> **EXECUTIVE VERDICT: Did the Phase-9 critic intervention materially close the
> SAC→RB→ECMS gap, and what measured evidence proves the mechanism?**
>
> **NO. Experiment A (CQL conservative critic) FAILS at every coefficient tried
> (`α ∈ {0.01, 0.05, 1.0}`): the trained policy runs SoC away to 78–86 % (V_CE
> 4.7–5.6, 0/3 charge-sustaining) or is CS only with 100+ constraint violations;
> every Q-oracle on a CQL critic is catastrophically non-CS (ΔSoC +34 to
> +46 pp). Gap "closed" vs the Phase-8 CONTROL = −213 %.** The critic-
> regularisation route is rejected. The best validated controller is unchanged
> (Phase-8 CONTROL: NEDC 3.7666 / +7.4 % vs RB; FTP75 3.2889 / +1.8 %).
>
> **What Phase 9 DID establish (measured):**
>
> * **The critic is not grossly miscalibrated on-distribution.** Region-averaged
>   `min(Q1,Q2)` at matched ECMS-trajectory states ranks
>   `HIGH_EFF ≳ ECMS_NBHD ≳ LOW ≳ OFF` in every band on both cycles — the same
>   order as the reward and the next-SoC consequence. Neither pre-registered
>   error type (Type-1 OFF-overvaluation, Type-2 high-load-undervaluation) is
>   cleanly triggered.
> * **The learnable defect is a mild, compounding LOW-load bias in the per-state
>   arg-max** across 15–35 Nm (arg-max-Q ∈ {OFF, LOW, ECMS_NBHD} ~98 %; HIGH_EFF
>   ~2 %), in a region the replay buffer barely covers (HIGH_EFF support 8–12 %,
>   twin-Q disagreement 0.056–0.066 vs 0.004–0.01 elsewhere). It is **not** a
>   far-OOD spike (the Q-oracle's own states are no more OOD than the ECMS ones).
> * **CQL moved the pathology, it did not repair it:** it cut the OFF arg-max
>   (38 % → 2 % at 15–30 Nm) but shifted the mass to **LOW, not** the efficient
>   region (30–35 Nm LOW arg-max 52 % → 78 %) and inflated Q roughly uniformly.
> * **Physical SAC→ECMS decomposition (§10, new, BSFC-grounded):** operating-point
>   inefficiency = **NEDC +0.192 (39 %) / FTP75 +0.082 (18 %)**; mode-selection &
>   timing = **NEDC +0.306 (61 %) / FTP75 +0.373 (81 %)**; battery/SoC ≈ 0. ECMS
>   keeps the engine in the low-BSFC high-load island (255 vs 290 g/kWh, η 0.35
>   vs 0.32) and runs it less often but harder (NEDC 260 vs 376 engine-on steps,
>   79 vs 55 Nm).
>
> **Next authorised experiment (§16):** (B) targeted training-time coverage of
> the efficient high-engine-load region — changes no reward, directly attacks
> the thin-support defect. Then (H) a reward term penalising part-load operation
> if B is insufficient. Algorithm swap stays gated.
>
> ---
>
> The forensics **revised the Phase-8 framing**:
>
> 1. **On the in-distribution (ECMS-trajectory, good-SoC) states, the critic is
>    NOT grossly wrong.** Region-averaged `min(Q1,Q2)` ranks
>    `HIGH_EFF ≳ ECMS_NBHD ≳ LOW ≳ OFF` in every torque band on both cycles —
>    the same order as the immediate reward and the next-SoC consequence.
>    Neither pre-registered error type (**Type 1** OFF-overvaluation, **Type 2**
>    high-load-undervaluation) is cleanly triggered by the region-averaged data.
> 2. **The critic's per-state arg-max is biased LOW-load in the 15–35 Nm band**
>    (arg-max-Q ∈ {OFF, LOW, ECMS_NBHD} ~98 % of states; HIGH_EFF/MAX ~2 %),
>    while the reward's optimum sits at higher engine load. This is a *mild*
>    Type-2-flavoured effect, not a gross one.
> 3. **The Phase-8 Q-oracle SoC-collapse is a COMPOUNDING many-small-errors
>    effect, not one egregiously misvalued action.** OFF is the arg-max-Q region
>    in ~25 % of states with only ~27 % replay support; a Q-greedy policy taking
>    OFF/LOW that often accumulates a SoC deficit even though no single state's
>    Q is badly wrong.
> 4. **Physical SAC→ECMS decomposition (§10, new, physically grounded):**
>    operating-point BSFC inefficiency = **NEDC +0.192 (39 %) / FTP75 +0.082
>    (18 %)**; mode-selection & timing = **NEDC +0.306 (61 %) / FTP75 +0.373
>    (81 %)**; battery/SoC ≈ 0 both. ECMS keeps the engine in the low-BSFC
>    high-load island (255 vs 290 g/kWh, η 0.35 vs 0.32) and runs it **less
>    often but harder** (NEDC 260 vs 376 engine-on steps, 79 vs 55 Nm).
>
> The CQL experiment tests whether penalising OOD-action value tightens the
> arg-max toward the data-supported efficient region and makes Q-greedy
> charge-sustaining. **Result: §"EXPERIMENT A".**

Git commit (Phase-9 start): `f1f45c559e126e67a7fc01634895a22b6e08e8de`
Raw: `results/phase9/logs/*.txt` · JSON: `results/phase9/data/*.json` ·
Figures: `results/phase9/figures/*.png`

---

## §1 — VERIFIED PHASE-8 STARTING POINT (reproduced)

| Controller | NEDC | FTP75 | CS |
|---|---|---|---|
| SAC CONTROL | 3.7666 | 3.2889 | 3/3 both |
| Advanced rule-based | 3.5056 | 3.2323 | — |
| ECMS | 3.1887 | 2.8097 | — |
| SAC Q-oracle | 3.9404 | 3.3545 | NEDC 1/3, FTP75 0/3 |
| SAC mixture actor (8C) | 3.8730 | 3.2462 | NEDC 1/3, FTP75 3/3 |

Remaining ECMS gap: NEDC **+0.578**, FTP75 **+0.479** L/100km (using the CONTROL).
All Phase-8 closed hypotheses (γ, k_fb, global OFF-coverage, reward-efficiency,
actor-class) are carried forward as closed. Algorithm switch remains gated.

---

## §3 / §4 — MAP THE CRITIC ERROR

`results/phase9_critic_diag.py`. Matched states = fresh clean `EMSEnv`
deep-copied at every traction step along the charge-sustaining **ECMS SoC
trajectory** (Phase-7/8 methodology, unchanged). Dense 161-point action grid.
Twin-Q **averaged over the 3 CONTROL critics**. Actions binned into 5
engine-load regions, state-conditioned on the ECMS engine load:
`OFF` (T_CE ≤ 5 Nm) · `LOW` (< 0.75·T_CE,ECMS) · `ECMS_NBHD` (0.75–1.3·) ·
`HIGH_EFF` (1.3·–0.9·T_CE,max) · `MAX` (> 0.9·T_CE,max).
Replay support = fraction of the 400 nearest buffer transitions (in scaled
obs-space) whose action lands in that region. Data:
`results/phase9/data/critic_error_map_{NEDC,FTP75}.json`.

### NEDC — region-averaged `min(Q1,Q2)`, reward, next-SoC, replay support

| band | region | supp % | minQ | \|Q1−Q2\| | reward | next-SoC (env) | BSFC | η |
|---|---|---|---|---|---|---|---|---|
| 15–30 | OFF | 36 | −0.257 | 0.005 | −0.192 | 29.5 % | — | — |
| | LOW | 54 | −0.252 | 0.005 | −0.181 | 29.8 % | 388 | 0.24 |
| | ECMS_NBHD | 27 | −0.192 | 0.004 | −0.149 | 33.4 % | 319 | 0.46 |
| | **HIGH_EFF** | 44 | **−0.165** | 0.006 | **−0.130** | **40.8 %** | 288 | 0.29 |
| 30–35 | OFF | 30 | −0.319 | 0.012 | −0.256 | 26.3 % | — | — |
| | LOW | 57 | −0.307 | 0.008 | −0.241 | 26.7 % | 323 | 0.28 |
| | **ECMS_NBHD** | 24 | **−0.296** | 0.004 | **−0.231** | 26.7 % | 277 | 0.41 |
| 35–50 | OFF | 36 | −0.258 | 0.008 | −0.197 | 31.9 % | — | — |
| | ECMS_NBHD | 25 | −0.245 | 0.006 | −0.191 | 30.2 % | 270 | 0.59 |
| | **HIGH_EFF** | 8 | **−0.137** | 0.005 | **−0.119** | **41.5 %** | 239 | 0.35 |
| 50–75 | OFF | 11 | −0.850 | **0.056** | −0.865 | 26.0 % | — | — |
| | ECMS_NBHD | 49 | −0.368 | 0.007 | −0.310 | 26.0 % | 237 | 0.35 |
| | **HIGH_EFF** | 12 | **−0.154** | 0.005 | **−0.065** | **38.6 %** | 232 | 0.36 |

FTP75 shows the same ordering (`results/phase9/data/critic_error_map_FTP75.json`)
with smaller inter-region spreads (0.003–0.06).

### Error-type classification (pre-registered thresholds)

| band | NEDC | FTP75 |
|---|---|---|
| 15–30 | neither cleanly triggered | neither |
| 30–35 | neither | neither |
| 35–50 | neither | neither |
| 50–75 | neither | neither |

* **Type 1 (OFF overvaluation): NOT triggered.** In every band `minQ(OFF) <
  minQ(ECMS_NBHD)` — the critic values OFF *below* the ECMS neighbourhood, not
  above it. (Phase 7's positive `ΔQ(OFF−ASSIST)` compared OFF to a *mild-assist*
  probe, not to the ECMS point — the critic dislikes the mid part-load region,
  which is consistent with both statements.)
* **Type 2 (high-load undervaluation): NOT cleanly triggered** — where HIGH_EFF
  has grid actions its `minQ` is actually the *highest* in the band. **BUT** the
  per-state arg-max analysis (§OOD below) shows HIGH_EFF wins the arg-max in only
  ~2 % of 15–35 Nm states: the region is thin at low demand and its Q ≈
  ECMS_NBHD's, so it rarely takes the max.
* **Q1−Q2 disagreement** is small in the supported regions (0.004–0.01) and large
  only for OFF/LOW at ≥50 Nm (0.056–0.066), i.e. exactly where OFF is infeasible
  and unsupported — the twin critics disagree about actions they have no data for.

**⇒ The critic error is best described as `Type 3-mild / compounding`: a small
systematic low-load bias in the per-state arg-max across the 15–35 Nm band,
plus twin-Q uncertainty on unsupported OFF/LOW actions at high torque — not a
single gross misvaluation.**

---

## OOD TEST — is the Q-oracle failure a distribution-shift effect?

`results/phase9_ood_test.py`. Same region analysis at TWO state distributions:
(a) ECMS-trajectory states; (b) the Q-oracle's OWN visited traction states
(seed0 critic). Data: `results/phase9/data/ood_test_NEDC.json`.

| measure | ECMS-traj states | Q-oracle's own states |
|---|---|---|
| SoC range | 12.7–50.2 % | 10.6–50.0 % |
| OFF is the arg-max-Q region in | **25 %** of states | **25 %** of states |
| mean OFF replay support | 27 % | 28 % |
| arg-max-Q region, 15–30 Nm | OFF 38 / LOW 38 / ECMS 22 / HIGH 2 % | OFF 38 / LOW 35 / ECMS 28 % |
| arg-max-Q region, 30–35 Nm | LOW 52 / ECMS 48 % | LOW 50 / ECMS 45 / OFF 5 % |

**Distribution-shift is NOT the dominant mechanism** — the Q-oracle's own states
are not markedly more OOD (same OFF-arg-max fraction, same replay support) than
the ECMS states. The SoC collapse instead comes from the critic's *persistent
low-load arg-max bias* being applied at ~50 % of steps (LOW) + ~25 % (OFF):
compounded over a cycle this drains SoC. **Refined H9-A is therefore: the critic
systematically ranks LOW/OFF at or above the efficient region in the 15–35 Nm
band — not that it spikes on far-OOD actions.**

---

## §10 — PHYSICAL SAC-vs-ECMS DECOMPOSITION

`results/phase9_engine_physics.py`. Matched-demand rollout, per-step engine
BSFC / efficiency / speed / fuel-rate from the validated `combustion_engine`
block. Figure: `results/phase9/figures/bsfc_map_{NEDC,FTP75}.png` (engine
operating points on the BSFC map).

| quantity | NEDC SAC / ECMS | FTP75 SAC / ECMS |
|---|---|---|
| V_CE_equiv (gap) | 3.686 / 3.189 (**+0.498**) | 3.270 / 2.810 (**+0.460**) |
| engine-on steps | **376 / 260** | 504 / 501 |
| mean engine T_CE when ON | **55 / 79 Nm** | 70 / 68 Nm |
| mean BSFC when ON | **290 / 255 g/kWh** | 253 / 241 g/kWh |
| mean engine efficiency | **0.324 / 0.352** | 0.328 / 0.354 |
| mean engine speed | 1804 / 1777 rpm | 1712 / 1731 rpm |

### Gap decomposition (A/B/C/D, §10)

| component | NEDC | FTP75 |
|---|---|---|
| **A — inefficient operating points** (BSFC, both engines on) | **+0.192 (39 %)** | **+0.082 (18 %)** |
| **B — different ON/OFF decisions** (raw) | +0.603 | +0.348 |
| **D — residual** (electrical credit ECMS pays for its extra OFF; transients) | −0.297 | +0.026 |
| **B + D — net mode-selection & timing** | **+0.306 (61 %)** | **+0.373 (81 %)** |
| **C — battery-energy timing / SoC-equivalence** | **+0.000** | **+0.005** |

### Physical mechanism — "ECMS: more torque, less fuel"

* **NEDC:** ECMS runs the engine **116 fewer steps** and **24 Nm harder** when
  on, placing every operating point in the low-BSFC island (75–140 Nm,
  1500–2500 rpm). SAC smears the same total work across **more** engine-on time
  at **part load** (25–60 Nm) — see the BSFC-map figure: SAC's red points sit in
  the 300–380 g/kWh part-load region and a ~1000-rpm low-efficiency cluster,
  ECMS's white points cluster in the 220–255 g/kWh island. Part-load BSFC
  penalty ⇒ +0.192 L/100km even though SAC's peak torque is lower.
* **FTP75:** engine-on *counts* match (504 vs 501), but they occur at
  **different times / with the wrong split** (249 "exactly-one-on" steps). The
  penalty is 81 % mode-timing, only 18 % operating-point.

### §A/§B/§C/§D — quantified answers

| | NEDC | FTP75 |
|---|---|---|
| **A** excess SAC fuel from inefficient operating points | **+0.192 L/100km (39 %)** | **+0.082 (18 %)** |
| **B** from different engine ON/OFF decisions (net of D) | **+0.306 (61 %)** | **+0.373 (81 %)** |
| **C** from battery-energy timing / SoC management | **≈ 0** | **≈ 0** |
| **D** remaining after A+B+C removed | absorbed into B (−0.297 raw residual) | +0.026 |

---

## §11 — DOES THE 60/25/15 DECOMPOSITION SURVIVE?

**Partly — and the operating-point share is LARGER than the Phase-8 estimate on
NEDC.** Phase 8 estimated ~60 % mode-selection / ~25 % operating-point / ~15 %
other. The physically-grounded §10 numbers:

| | Phase-8 estimate | Phase-9 measured (NEDC) | Phase-9 measured (FTP75) |
|---|---|---|---|
| mode-selection & timing | ~60 % | **61 %** | **81 %** |
| engine operating point | ~25 % | **39 %** | **18 %** |
| battery / SoC / other | ~15 % | **≈ 0 %** | **≈ 1 %** |

The "~15 % other" was really operating-point on NEDC. The split will be
re-measured again for the best Phase-9 controller (§15).

---

## EXPERIMENT A — CQL CONSERVATIVE CRITIC

> *"§5: implement a CQL-style conservative critic as the first critic-side
> intervention. The only scientific intervention is the critic objective.
> Document the exact modified critic loss mathematically. Use a pre-specified
> conservative coefficient."*

**Modified critic loss** (`results/phase9_cql.py::CQLSAC.train`, per critic *i*):

```
L_i        = L_TD_i  +  cql_alpha · ( logsumexp_{a∈A_s} Q_i(s,a)  −  Q_i(s, a_data) )

L_TD_i     = 0.5 · MSE( Q_i(s,a_data),  r + (1−done)·γ·[ min_j Q̄_j(s',a') − α·log π(a'|s') ] )   (unchanged)

A_s        = { n a_rand ~ U[−1,1]  with weight −log(0.5^{dimA}) }
           ∪ { n a ~ π(·|s)        with weight −log π(a|s).detach() }
           ∪ { n a ~ π(·|s')       with weight −log π(a|s').detach() }      (CQL(H), importance-corrected)
```

Pre-specified: `cql_alpha = 1.0`, `n = 10`. **Everything else frozen** at CONTROL
values: reward, state, unimodal actor, γ 0.20, n_step 1, k_fb 2.5, env, action
map, ECMS, rule-based, evaluator, 150k budget, batch 512, buffer 300k, actor lr
3e-4, seeds {0,1,2}, net_arch [256,256], `tau` 0.005, entropy auto.

### A.1 — `cql_alpha = 1.0` (pre-specified): **FAIL**

`models_p9a_N{0,1,2}`, 3 seeds, 150k steps. Data:
`results/phase9/data/cql_forensics_NEDC.json`.

| metric | CONTROL | **CQL α=1.0 (3 seeds)** |
|---|---|---|
| normal-actor V_CE mean ± σ | 3.7666 ± 0.079 | **4.9996 ± 0.230** |
| per-seed V_CE / ΔSoC | 3.69/3.84/3.77, ≈0 pp | 4.76/5.02/5.22, **−3.1/−2.4/+10.3 pp** |
| charge-sustaining | 3/3 | **0/3** (SoC runaway to 47–86 %) |
| constraint violations | 0 | 0 / 0 / **121** (seed0) |
| Q-oracle on the CQL critic | 3.9404, 1/3 CS | **4.4020 ± 0.392, 1/3 CS** |

**Gap "closed" vs CONTROL: −213 % (the gap widened massively).** The Q-oracle
did **not** become charge-sustaining and did **not** improve over the Phase-8
Q-oracle.

**Why (scale analysis, pre-registered as the failure explanation):** the CQL(H)
term `logsumexp_{30 a}(Q) − Q(s,a_data) ≈ ln 30 ≈ 3.4` per sample, while the TD
loss `0.5·MSE(td_err ≈ 0.1) ≈ 5e-3` — the conservative gradient outweighs the TD
gradient by **~700×** at this reward scale (reward ≈ −0.2/step, Q ≈ −0.3). The
critic collapses to "avoid every non-logged action", the actor charges to escape
the penalty, and SoC runs away. This is a **coefficient-scale failure, not a
refutation of conservatism** — §5 permits a weak/baseline/strong sensitivity
comparison when "scientifically necessary", and a ~700× scale mismatch qualifies.

### §6 — what `α=1.0` pessimism actually did to the Q landscape

| band | CONTROL arg-max-Q region | CQL α=1.0 arg-max-Q region |
|---|---|---|
| 15–30 | OFF 38 / LOW 38 / ECMS 22 / HIGH 2 % | **OFF 2** / LOW 65 / ECMS 30 / HIGH 2 % |
| 30–35 | LOW 52 / ECMS 48 % | **LOW 78** / ECMS 22 % |
| 35–50 | OFF 15 / LOW 18 / ECMS 65 / HIGH 2 % | OFF 2 / LOW 22 / ECMS 68 / HIGH 8 % |
| 50–75 | LOW 18 / ECMS 80 / HIGH 2 % | ECMS 80 / **HIGH 18** % |

Pessimism **suppressed the OFF arg-max** (15–30: 38 % → 2 %) — as intended — but
**shifted the mass to LOW, not to the efficient ECMS_NBHD/HIGH_EFF** (30–35 LOW
52 % → 78 %). It also inflated `minQ` roughly uniformly for the supported
regions (ECMS_NBHD −0.191 → −0.014). Per §6 this is **not** success: the OFF
overvaluation was replaced by a *worse* low-load bias, and Q moved uniformly
rather than re-ordering toward the fuel-optimal region.

### A.2 — weak/baseline/strong sensitivity (`α ∈ {0.01, 0.05}`, scale-corrected): **ALSO FAIL**

Scale-corrected baseline ≈ 1/700 ≈ 0.0015; bracket `{0.01, 0.05}` at seed 0.

| α | end-of-training checkpoint | `sac_ems_best` checkpoint | Q-oracle on the critic |
|---|---|---|---|
| 0.01 | V_CE 4.77, SoC **78 %** (runaway) | V_CE 5.01, ΔSoC −0.2 pp, **123 violations** | V_CE 5.23, **ΔSoC +45.7 pp** |
| 0.05 | V_CE 4.76, SoC **79 %** (runaway) | V_CE 4.74, ΔSoC +1.7 pp, **147 violations** | V_CE 5.79, **ΔSoC +34.1 pp** |
| 1.0 | V_CE 4.9–5.6, SoC 78–86 % (3 seeds) | — | V_CE 4.40, 1/3 CS |

**CQL fails at every coefficient tried** (`α ∈ {0.01, 0.05, 1.0}`): the policy
either runs SoC away to 78–86 % or, at the "best" checkpoint, is
charge-sustaining only by racking up **100+ SoC constraint violations** with
V_CE ≈ 5.0. Every Q-oracle on a CQL critic is catastrophically non-CS
(ΔSoC +34 to +46 pp). The conservative term destabilises the delicate
`k_fb` + SoC-penalty charge-sustaining balance regardless of strength — because
it penalises the current policy's own (discharge/OFF) actions, and the SoC
penalty then makes charging the locally safe escape.

> **§14 verdict for Experiment A: FAIL.** "Still causes Q-oracle SoC collapse"
> ✓, "still grossly misvalues" ✓, "still fails to value the ECMS/high-load
> region" ✓ (§6 below). The CQL / critic-regularisation route is **rejected**.

### §7 — Q-oracle on the repaired (CQL α=1.0) critic

| Metric | Phase-8 CONTROL | Phase-8 Q-oracle | **Phase-9 CQL Q-oracle** |
|---|---|---|---|
| NEDC V_CE (3 seeds) | 3.7666 | 3.9404 | **4.4020 ± 0.392** |
| NEDC CS | 3/3 | 1/3 | **1/3** |
| ΔSoC | −0.07 pp | −2.48 pp | +1.0 / −9.2 / −2.8 pp |
| OFF % | 38.6 | 40.7 | 40.0 / 22.4 / 38.0 |
| mean engine T_CE when ON | 53.3 | 61.6 | 55.8–60.4 |
| violations | 0 | 0 | 0 |

**Did NOT become charge-sustaining and did NOT improve over the Phase-8
Q-oracle** — it is 0.46 L/100km *worse*.

### §6 — did pessimism fix the *right* thing? (CONTROL vs CQL α=1.0)

| band | CONTROL arg-max-Q region | CQL α=1.0 arg-max-Q region |
|---|---|---|
| 15–30 | OFF 38 / LOW 38 / ECMS 22 / HIGH 2 % | **OFF 2** / LOW 65 / ECMS 30 / HIGH 2 % |
| 30–35 | LOW 52 / ECMS 48 % | **LOW 78** / ECMS 22 % |
| 35–50 | OFF 15 / LOW 18 / ECMS 65 / HIGH 2 % | OFF 2 / LOW 22 / ECMS 68 / HIGH 8 % |
| 50–75 | LOW 18 / ECMS 80 / HIGH 2 % | ECMS 80 / **HIGH 18** % |

* **OFF region:** Q-greedy OFF frequency **decreased** (15–30: 38 % → 2 %). ✓ for
  the stated §6 sub-goal — but see the caveat.
* **ECMS / high-load region:** `minQ(ECMS_NBHD)` and `minQ(HIGH_EFF)` were
  **inflated roughly uniformly** (−0.191 → −0.014; −0.137 → −0.028), i.e. Q went
  *up* everywhere rather than *re-ordering* toward the fuel-optimal region.
  HIGH_EFF still wins the per-state arg-max in only 2–18 % of states.
* **Caveat (§6, decisive):** "A lower Q everywhere is NOT success" — here it is
  Q *inflation* everywhere, and the OFF overvaluation was replaced by a **worse
  LOW-load bias** (30–35 Nm LOW arg-max 52 % → 78 %). CQL moved the pathology,
  it did not repair the value ordering.

### §8 — the normal SAC actor on the repaired critic

The CQL run **is** a normal unimodal SAC actor trained jointly with the
conservative critic. 3-seed NEDC result: **V_CE 4.9996 ± 0.230, 0/3
charge-sustaining** (SoC 47–86 %), one seed with 121 violations. **The normal
actor cannot exploit the CQL critic — it is far worse than CONTROL.**

---

## §9 — EXPERIMENT B — TARGETED HIGH-LOAD COVERAGE

**Precondition (partially met):** §6 shows the CQL α=1.0 critic *did* suppress the
OFF arg-max (15–30 Nm 38 % → 2 %) while the ECMS/high-load value ordering stayed
wrong (HIGH_EFF arg-max 2–18 %). But because Experiment A's *trained policy*
collapsed (SoC runaway at every α), a coverage intervention that also perturbs
the SoC balance is **lower-confidence**. Design (unchanged): a training-time
exploration schedule that, when the engine is ON and a higher feasible load
exists, commands a uniform draw from the feasible **high-load interval** with
small probability — a coverage intervention, **no ECMS/benchmark action used**.
**NOT STARTED — recommended as the next authorised experiment**, alongside the
reward-shaping option below, because §10 shows the largest physically-grounded
residual (part-load BSFC) is a *data/representation* problem in the efficient
region, not a regularisation problem.

---

## §15 — ECMS GAP-CLOSURE SCORECARD

`gap closed = (Phase8_SAC − New_SAC) / (Phase8_SAC − ECMS)`

| | NEDC (Phase-8 CONTROL → Phase-9 CQL α=1.0, 3 seeds) |
|---|---|
| Phase-8 CONTROL V_CE | 3.7666 (3/3 CS) |
| Phase-9 CQL SAC V_CE | **4.9996 ± 0.230 (0/3 CS)** |
| SAC − ECMS (abs / %) | +1.811 / **+56.8 %** (was +0.578 / +18.1 %) |
| **gap closed vs Phase-8 CONTROL** | **−213 %** (the gap more than tripled) |
| SAC − RB | +1.494 / +42.6 % (was +0.261 / +7.4 %) |
| improvement over Phase-8 CONTROL | **−32.7 % (regression)** |

FTP75 CQL not run — Experiment A is a comprehensive NEDC failure, so an FTP75
replication has no diagnostic value.

---

## §16 — FINAL VERDICT (12 questions)

1. **Did the conservative critic fix OOD engine-OFF overvaluation?** **No net
   fix.** §3/§4 found no gross OFF-*overvaluation* to begin with (`minQ(OFF) <
   minQ(ECMS_NBHD)` everywhere). §6: CQL did cut the OFF *arg-max* frequency
   (38 % → 2 % at 15–30 Nm) but replaced it with a **worse LOW-load bias**
   (30–35 Nm 52 % → 78 %) and inflated Q roughly uniformly — not a re-ordering.
2. **Did it fix high-load engine undervaluation?** **No.** HIGH_EFF still wins
   the per-state arg-max in only 2–18 % of states after CQL; its `minQ` was
   already the highest where supported. The problem is that HIGH_EFF is *thin*
   at 15–35 Nm demand (8–12 % replay support, high twin-Q disagreement), which
   CQL cannot create.
3. **Were those one problem or two?** **One, mild, compounding** — a systematic
   low-load bias in the per-state arg-max across 15–35 Nm. Not two independent
   gross errors; not OOD-action overvaluation (the Q-oracle's own states are no
   more OOD than the ECMS states).
4. **Did Q-oracle become charge-sustaining?** **No** — CQL Q-oracle ΔSoC +1.0 /
   −9.2 / −2.8 pp (α=1.0), +34–46 pp (weak α). Worse than the Phase-8 Q-oracle.
5. **Did Q-oracle improve over Phase 8?** **No** — 4.4020 vs 3.9404 (**+0.46
   worse**).
6. **Did normal SAC exploit the repaired critic?** **No** — 4.9996 ± 0.230,
   0/3 CS, one seed with 121 violations. A regression of −1.23 L/100km.
7. **Did SAC beat advanced rule-based?** **No.** The best validated controller
   remains the Phase-8 CONTROL (NEDC +7.4 %, FTP75 +1.8 % over RB). Phase 9
   produced no controller that beats it.
8. **How much of the original ECMS gap was closed?** **Negative — the CQL
   intervention widened the gap by 213 %.** No gap was closed.
9. **What physical mechanism explains the remaining (CONTROL→ECMS) gap?**
   **§10 (measured):** NEDC ≈ **39 % part-load engine-BSFC inefficiency** +
   ≈ **61 % engine ON/OFF timing**; FTP75 ≈ 18 % / 81 %; battery/SoC ≈ 0. ECMS
   runs the engine **116 fewer steps** and **24 Nm harder** (NEDC), keeping every
   operating point in the low-BSFC island (255 vs 290 g/kWh; η 0.35 vs 0.32).
10. **Is missing replay support still the bottleneck?** **Plausibly yes for the
    efficient region** — HIGH_EFF at 15–35 Nm and OFF/LOW at ≥50 Nm have 8–12 %
    support and twin-Q disagreement 0.056–0.066 (vs 0.004–0.01 elsewhere).
    §9 (targeted high-load coverage) is the direct test — **not yet run**.
11. **Is state representation now implicated?** **Not yet ruled in.** The §12
    fallback (feature sufficiency) is only in scope after §9 also fails. The
    reward already carries the fuel consequence, so the missing ingredient is
    more likely *coverage of the efficient region*, not a missing feature.
12. **Is SAC itself finally implicated?** **No.** The §13 gate is still not met:
    (1) mode-aware actor tested (Phase 8) ✓, (2) Q-oracle far from ECMS ✓,
    (3) reward/action/state counterfactual proving the critic *cannot* represent
    the value — **not shown** (the critic ranks HIGH_EFF highest where it has
    data; it lacks data, it is not incapable), (4) training-stability addressed
    partially. Algorithm swap remains **gated out**.

### One-line verdict

> **Experiment A (CQL conservative critic) FAILS at every coefficient.** Neither
> Phase-8's actor-capacity route nor Phase-9's critic-regularisation route closes
> the gap. The **physically-grounded residual is the engine operating point** —
> SAC runs the engine at part-load (worse BSFC) where ECMS keeps it in the
> efficient high-load island — and the forensics locate the learnable defect as
> a **mild compounding low-load arg-max bias in a region the replay buffer
> barely covers**. The two next authorised experiments both target *that*:
> **(B) targeted high-load / efficient-region training coverage** (§9), and
> **(H) a reward term that explicitly penalises low engine efficiency / part-load
> operation** (§16 note — the current reward's arg-max is already at higher
> engine load than ECMS, Phase-8 §16, but the *learned policy* never gets there,
> so a shaping term that makes the efficient region the reward-gradient
> attractor is justified). Run B first (it changes no reward). Algorithm swap
> stays gated.

---

## OUTPUTS

```
results/phase9/
  logs/phase9_critic_diag_{NEDC,FTP75}.txt   logs/phase9_ood_test_NEDC.txt
  logs/phase9_engine_physics_{NEDC,FTP75}.txt
  data/critic_error_map_{NEDC,FTP75}.json    data/ood_test_NEDC.json
  data/engine_physics_{NEDC,FTP75}.json
  figures/bsfc_map_{NEDC,FTP75}.png
results/phase9_critic_diag.py  phase9_ood_test.py  phase9_engine_physics.py
results/phase9_cql.py   (CQLSAC + trainer)
models_p9a_N{0,1,2}     (CQL α=1.0, NEDC, 3 seeds)
```

No validated physics / reward / benchmark / evaluator touched. Phases ≤8 intact.
