"""
V1-A / V1-B  --  analytical costate-currency derivation (NO TRAINING, NO CODE CHANGE).

Re-derives, from the frozen source, the SAC reward's per-step objective in the
SAME variables ECMS uses (P_fuel, P_batt), and compares the implied effective
battery costate lambda_SAC(SoC) with lambda_ECMS(SoC).

Distinguishes THREE quantities (task section 3):
  A. dE_ledger = E(Q_{t-1}) - E(Q_t)          [what the reward integrates]
  B. P_EM = p_bt = electrical power at motor  [what ECMS's H uses]
  C. lambda_ECMS(SoC) = lam0 + kfb*(0.5-SoC)

Outputs: results/phase11/data/v1_analytical.json  + console table.
"""
import json, math
import numpy as np
from pathlib import Path

# ---- constants pulled straight from the frozen source -----------------------
from src.env.powertrain import (
    _C_BT_L1, _C_BT_L3, _Q_BT_0, _Q_BT_IC, _I_0,
    _C_BT_E1, _C_BT_E2, _C_BT_E3, _C_BT_E4,
    _H_u, _RHO_FUEL, _K_CS, _EFC_GAIN,
    _battery_energy, _u_oc,
)
from src.env.ems_env import (
    K_FUEL_L_PER_KG, K_ELEC_L_PER_J, SOC_TARGET,
    REWARD_SCALE, LAMBDA_SOC, LAMBDA_SOC_LIN, SOC_DEADBAND,
)

c = _C_BT_L3 / _Q_BT_0            # 15.6/36000
d = _C_BT_L1                     # 39.0

# canonical per-joule conversions (RL_DIAGNOSTIC_REPORT sec 2)
k_f = K_FUEL_L_PER_KG / _H_u      # L per J of fuel LHV      ~3.1772e-8
k_e = K_ELEC_L_PER_J             # L per J of ledger energy  ~1.5349e-7
NOMINAL_RATIO = k_e / k_f         # ~4.8309   (Stage-0 used this)

# ECMS costate law (frozen: evaluate_policy.py / ecms.py)
ECMS_LAM0 = {"NEDC": 1.3125, "FTP75": 2.4062}
ECMS_KFB = 8.0

# CONTROL reward config (frozen: run_config.json)
CTRL_EQF = {"NEDC": 0.2717, "FTP75": 0.4981}
CTRL_KFB = 2.5

# CONTROL operating-SoC distribution (Phase 7 measured, moving-traction steps)
CTRL_SOC_MEDIAN = {"NEDC": 0.375, "FTP75": 0.474}


def dEdq(q):
    """d/dq [ 0.5*(c*q+d)*q ]  = c*q + 0.5*d   (exact, E is quadratic in q)."""
    return c * q + 0.5 * d


def u_bt_discharge(q, p_bt):
    """Terminal voltage under load, discharge branch (Battery.step, p_bt>0)."""
    soc = q / _Q_BT_0
    a = _C_BT_E3 * soc + _C_BT_E1
    b = _C_BT_E4 * soc + _C_BT_E2
    disc = a * a + 4.0 * b * (-(p_bt / _I_0))
    return (a + math.sqrt(max(disc, 0.0))) / 2.0


def C_factor(q, p_bt):
    """
    C(SoC, P_EM) = dE_ledger / (P_EM * dt)
      dE_ledger ~= dEdq(q) * (q_prev - q_new),   q_prev - q_new = p_bt / u_bt
      P_EM * dt = p_bt          (dt = 1 s)
    => C = dEdq(q) / u_bt(q, p_bt)
    """
    return dEdq(q) / u_bt_discharge(q, p_bt)


def analyse(cycle):
    lam0 = ECMS_LAM0[cycle]
    eqf_base = CTRL_EQF[cycle]

    socs = [0.30, 0.375, 0.40, 0.50, 0.60, 0.70, CTRL_SOC_MEDIAN[cycle]]
    # representative discharge power for C(): use a mid-range assist ~3 kW and
    # also the zero-power limit (C -> dEdq/U_oc) for reference
    P_REP = 3000.0

    rows = []
    for soc in socs:
        q = soc * _Q_BT_0
        Uoc = _u_oc(q)
        C_load = C_factor(q, P_REP)
        C_oc = dEdq(q) / Uoc                      # zero-power limit
        eqf_eff = eqf_base + CTRL_KFB * (SOC_TARGET - soc)
        lam_ecms = lam0 + ECMS_KFB * (SOC_TARGET - soc)

        # ---- lambda_SAC on a common P_EM basis --------------------------------
        # Reward per step (u-dependent part), MARGINAL w.r.t. the current action:
        #   fuel term  : dm_fuel = 0.5*(P_CE(u)+P_CE_prev)/H_u   ->  d/dP_CE = 0.5/H_u
        #                (trapezoidal: only HALF of P_CE(u) is charged this step;
        #                 the other half lands in step t+1 via Tank.p_fuel_prev,
        #                 which is NOT in the observation -> critic cannot see it)
        #   batt term  : eqf_eff * dE_ledger * k_e ,  dE_ledger = C * P_EM * dt
        # ratio (batt marginal)/(fuel marginal)
        #   = [eqf_eff * C * k_e] / [0.5 * k_f]
        #   = eqf_eff * C * (k_e/k_f) / 0.5
        #   = eqf_eff * C * NOMINAL_RATIO * 2
        lam_sac_marg = eqf_eff * C_load * NOMINAL_RATIO * 2.0
        # Stage-0's (incomplete) basis: no trapezoidal factor, C applied
        lam_sac_stage0 = eqf_eff * C_load * NOMINAL_RATIO
        # return-level effective fuel weight at gamma=0.20 is 0.5*(1+gamma)=0.60
        # (each P_CE_t appears in r_t w/ 0.5 and r_{t+1} w/ 0.5*gamma)
        lam_sac_return = eqf_eff * C_load * NOMINAL_RATIO / (0.5 * (1 + 0.20))

        rows.append(dict(
            soc=round(soc, 4),
            U_oc_V=round(Uoc, 3),
            dEdq=round(dEdq(q), 4),
            C_zeroP=round(C_oc, 5),
            C_at_3kW=round(C_load, 5),
            eqf_eff=round(eqf_eff, 5),
            lam_ECMS=round(lam_ecms, 4),
            lam_SAC_marginal=round(lam_sac_marg, 4),
            lam_SAC_stage0basis=round(lam_sac_stage0, 4),
            lam_SAC_returnbasis=round(lam_sac_return, 4),
            ratio_marg_over_ECMS=round(lam_sac_marg / lam_ecms, 4),
            ratio_stage0_over_ECMS=round(lam_sac_stage0 / lam_ecms, 4),
        ))
    return dict(
        cycle=cycle, lam0=lam0, eqf_base=eqf_base, ctrl_kfb=CTRL_KFB,
        nominal_k_e_over_k_f=NOMINAL_RATIO,
        k_f_per_J_fuel=k_f, k_e_per_J_ledger=k_e,
        capacitor_note=(
            "E(Q)=0.5*U_oc(Q)*Q  => dE/dQ = c*Q + 0.5*d = U_oc(Q) - 19.5 ; "
            "C = dE/dQ / u_bt ~ 0.55-0.61 over 5-95% SoC (0.583 at 50% SoC, zero power)."
        ),
        trapezoidal_note=(
            "Reward fuel term is trapezoidal dm_fuel=0.5*(P_CE+P_CE_prev)/H_u. "
            "MARGINAL fuel weight of the current action is 0.5/H_u (half lands next step "
            "via Tank.p_fuel_prev, which is NOT observable). Battery term dE_ledger is a "
            "full per-step energy delta. => battery is weighted 2x relative to marginal fuel, "
            "which CANCELS most of the 0.583 capacitor factor and REVERSES the sign of the "
            "Stage-0 mismatch: lam_SAC_marginal = eqf_eff * C * 4.8309 * 2 ~ eqf_eff * 5.6."
        ),
        rows=rows,
    )


if __name__ == "__main__":
    out = {"NEDC": analyse("NEDC"), "FTP75": analyse("FTP75")}
    Path("results/phase11/data").mkdir(parents=True, exist_ok=True)
    Path("results/phase11/data/v1_analytical.json").write_text(json.dumps(out, indent=2))

    for cyc in ("NEDC", "FTP75"):
        o = out[cyc]
        print(f"\n================  {cyc}  (lam0={o['lam0']}, eqf_base={o['eqf_base']}, kfb_ctrl={CTRL_KFB})")
        print("  k_e/k_f (nominal) =", round(o["nominal_k_e_over_k_f"], 4),
              " | trapezoidal marginal doubles battery weight -> effective ~", round(o["nominal_k_e_over_k_f"]*2, 3))
        hdr = ("SoC", "U_oc", "C@3kW", "eqf_eff", "lam_ECMS", "lamSAC_marg", "marg/ECMS", "lamSAC_stage0", "s0/ECMS")
        print("  " + "".join(f"{h:>13}" for h in hdr))
        for r in o["rows"]:
            print("  " + "".join(f"{v:>13}" for v in (
                r["soc"], r["U_oc_V"], r["C_at_3kW"], r["eqf_eff"], r["lam_ECMS"],
                r["lam_SAC_marginal"], r["ratio_marg_over_ECMS"],
                r["lam_SAC_stage0basis"], r["ratio_stage0_over_ECMS"])))
    print("\n[saved] results/phase11/data/v1_analytical.json")
