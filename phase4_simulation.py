"""
Phase 4: Simulation & Forecasting  — TRADE CREATION SHOCK

Theory
------
Trade CREATION (Viner 1950) means that regional integration generates
genuinely NEW trade flows that did not exist before, because lower
tariffs / NTBs make previously uneconomical bilateral trade viable.

In an MRIO context this is modelled by:

  1. Scale up intra-bloc intermediate flow entries Z_{ij}  (i≠j, both in
     bloc) by factor (1 + RHO).  This is the "new trade" — additional
     intermediate inputs that bloc economies now purchase from each other.
     Non-bloc flows are LEFT UNCHANGED (no diversion).

  2. Recompute output totals to clear the market:
        X_new = (I - A_new)^{-1} · y
     where y = final demand (held fixed — the shock is on the production/
     input side, not on final consumption) and A_new is recomputed from
     Z_new and X_new via fixed-point iteration.

     Because Z_new = Z + ΔZ  and  A_new = Z_new / X_new, and X is itself
     a function of A (through Leontief), the correct procedure is:

       a. Compute the shocked Z: Z_star = Z + ΔZ  (ΔZ = RHO * Z_intra)
       b. First-pass A_star = Z_star / X  (using old X as denominator)
       c. Solve for new equilibrium output:
            X_star = L_star · (Y.sum(axis=(1,2)))
          where L_star = (I - A_star)^{-1}
       d. Rebuild A_star = Z_star / X_star  (consistent with new X)
       e. Run full KWW decomposition on (Z_star, Y, X_star)

  This correctly captures:
    - Increased intra-bloc trade volumes (trade creation)
    - No mechanical reduction in non-bloc flows (no trade diversion)
    - General-equilibrium output adjustments — bloc economies produce
      more because they face more demand from bloc partners
    - GVC indices rise because more VA crosses borders within the bloc

Parameters
----------
RHO : float
    Proportional increase in intra-bloc intermediate flows.
    RHO = 0.10  →  intra-bloc Z entries rise 10% above baseline.
    This is consistent with gravity-model estimates for a deep FTA
    reducing bilateral trade costs by ~5-8%.
"""

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (NORM_DIR, OUT_DIR, YEARS, ALL_ECONOMIES_ORDERED,
                    N_ECONOMIES, N_SECTORS, N_FD_COLS,
                    BLOC_CODES, BLOC_NAMES)
from phase2_leontief import (technical_coefficients, leontief_inverse,
                              kww_decompose, build_results_df)

DTYPE = np.float32

# Proportional increase in intra-bloc intermediate trade flows
RHO = 0.10


# ── Helper: identify bloc row/col masks ──────────────────────────────────────

def _bloc_masks():
    N, S = N_ECONOMIES, N_SECTORS
    NS   = N * S
    bloc_eco_indices = [
        ALL_ECONOMIES_ORDERED.index(c)
        for c in BLOC_CODES if c in ALL_ECONOMIES_ORDERED
    ]
    bloc_rows = np.zeros(NS, dtype=bool)
    for i in bloc_eco_indices:
        bloc_rows[i*S:(i+1)*S] = True
    return bloc_eco_indices, bloc_rows


# ── Core shock: trade creation ────────────────────────────────────────────────

def apply_trade_creation_shock(Z: np.ndarray, X: np.ndarray,
                                Y: np.ndarray, rho: float = RHO):
    """
    Model trade creation: scale up intra-bloc intermediate flows by (1+rho),
    leave all other flows unchanged, then solve for the new general-equilibrium
    output vector and return the consistent (Z_star, X_star) pair.

    Parameters
    ----------
    Z   : (NS, NS) intermediate flow matrix
    X   : (NS,)   gross output vector
    Y   : (NS, N, F) final demand matrix
    rho : proportional increase applied to intra-bloc Z entries

    Returns
    -------
    Z_star : shocked intermediate flow matrix
    X_star : consistent new gross output vector
    """
    N, S = N_ECONOMIES, N_SECTORS
    NS   = N * S

    bloc_eco_indices, bloc_rows = _bloc_masks()

    # ── Step 1: Build ΔZ — add new intra-bloc trade ──────────────────────────
    # Mask selects cells where BOTH source row and destination column belong
    # to a bloc economy, BUT source and destination are DIFFERENT economies
    # (we do not boost within-country domestic flows).

    Z_star = Z.copy().astype(np.float64)  # use float64 for precision

    for i_eco in bloc_eco_indices:
        for j_eco in bloc_eco_indices:
            if i_eco == j_eco:
                continue          # skip domestic flows
            r_s = i_eco * S
            r_e = (i_eco + 1) * S
            c_s = j_eco * S
            c_e = (j_eco + 1) * S
            Z_star[r_s:r_e, c_s:c_e] *= (1.0 + rho)

    # ── Step 2: Solve for new equilibrium output ──────────────────────────────
    # Final demand y = row sums of Y (summed over all destination economies
    # and final demand categories).
    y = Y.reshape(NS, N * N_FD_COLS).sum(axis=1).astype(np.float64)

    # First-pass A using OLD output as denominator (starting point)
    X_f64  = X.astype(np.float64)
    Xsafe  = np.where(X_f64 > 0, X_f64, 1.0)
    A_star = (Z_star / Xsafe[np.newaxis, :])

    # Solve (I - A_star) X_star = y
    from scipy.linalg import solve
    I      = np.eye(NS)
    L_star = solve(I - A_star, I, assume_a="gen", check_finite=False)
    X_star = L_star @ y

    # ── Step 3: Rebuild A consistently with X_star ────────────────────────────
    Xstar_safe = np.where(X_star > 0, X_star, 1.0)
    A_star2    = (Z_star / Xstar_safe[np.newaxis, :])

    # Verify stability: spectral radius of A_star2 should be < 1
    # (skip expensive eig; just check column sums as a proxy)
    col_sums = A_star2.sum(axis=0)
    max_col  = col_sums.max()
    print(f"  Trade-creation shock diagnostics:")
    print(f"    max A column sum (A_star)  = {max_col:.6f}  (must be <1)")
    print(f"    X_star / X ratio  mean     = {(X_star / Xsafe).mean():.6f}")
    print(f"    X_star / X ratio  max      = {(X_star / Xsafe).max():.6f}")

    return Z_star.astype(DTYPE), X_star.astype(DTYPE)


# ── Counterfactual 2021 ───────────────────────────────────────────────────────

def counterfactual_2021(norm_dir: str, rho: float = RHO):
    path = os.path.join(norm_dir, "mrio_2021.npz")
    data = np.load(path, allow_pickle=True)
    Z21  = data["Z"].astype(DTYPE)
    Y21  = data["Y"].astype(DTYPE)
    X21  = data["X"].astype(DTYPE)

    print(f"  Applying trade-creation shock (rho={rho:.0%}) to 2021 MRIO …")
    Z21_star, X21_star = apply_trade_creation_shock(Z21, X21, Y21, rho=rho)

    print("  Computing counterfactual KWW for 2021 …", end=" ", flush=True)
    res_cf = kww_decompose(Z21_star, Y21, X21_star)
    print("done")
    return res_cf


# ── Trend projection ──────────────────────────────────────────────────────────

def fit_linear_trend(years: list, values: np.ndarray):
    x    = np.array(years, dtype=float)
    y    = np.array(values, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return 0.0, float(np.nanmean(y))
    xm, ym = x[mask].mean(), y[mask].mean()
    slope  = np.sum((x[mask]-xm)*(y[mask]-ym)) / np.sum((x[mask]-xm)**2)
    return slope, float(ym - slope * xm)


def project_bau(actual_df: pd.DataFrame, forecast_years: list) -> pd.DataFrame:
    """
    BAU: extrapolate pre-2021 historical trend into 2022-2024.
    Uses all available actual data up to and including 2021.
    """
    metrics   = ["gvc_back", "gvc_fwd", "gvc_total", "gross_exp"]
    pre_event = actual_df[actual_df["year"] <= 2021]
    rows = []
    for eco in ALL_ECONOMIES_ORDERED:
        sub = pre_event[pre_event["economy"] == eco].sort_values("year")
        for fy in forecast_years:
            row = {"year": fy, "economy": eco,
                   "name": BLOC_NAMES.get(eco, eco),
                   "in_bloc": eco in BLOC_CODES, "scenario": "BAU"}
            for m in metrics:
                slope, intercept = fit_linear_trend(
                    sub["year"].tolist(), sub[m].values)
                row[m] = slope * fy + intercept
            rows.append(row)
    return pd.DataFrame(rows)


def project_integrated(bau_df: pd.DataFrame,
                        actual_2021: dict,
                        cf_2021: dict,
                        forecast_years: list) -> pd.DataFrame:
    """
    Integrated path = BAU + delta from 2021 trade-creation shock.

    Delta = counterfactual_index - actual_index.

    Because trade creation raises output in the bloc, the GVC indices
    (FVA/E and DVX/E) are expected to be positive for bloc members:
      - FVA rises because they now import more intermediate VA from bloc partners
      - DVX rises because they now supply more intermediate VA to bloc partners
      - E (gross exports) also rises, so the index ratios reflect genuine
        deeper GVC participation, not just mechanical scaling.

    The delta is assumed to be time-invariant (it is anchored to the 2021
    structural shock) and added uniformly to each forecast year's BAU level.
    A more sophisticated model would apply a growth path to the delta, but
    the uniform-delta approach is the standard reduced-form counterfactual.
    """
    metrics = ["gvc_back", "gvc_fwd", "gvc_total"]

    # actual_2021 is indexed by sorted economy order
    actual_2021_sorted = actual_2021   # already sorted by economy in run()
    delta = {m: cf_2021[m] - actual_2021_sorted[m] for m in metrics}

    int_rows = []
    for _, row in bau_df.iterrows():
        eco     = row["economy"]
        idx     = ALL_ECONOMIES_ORDERED.index(eco)
        new_row = row.to_dict()
        new_row["scenario"] = "Integrated"
        for m in metrics:
            new_row[m] = row[m] + float(delta[m][idx])
        int_rows.append(new_row)
    return pd.DataFrame(int_rows)


# ── main ──────────────────────────────────────────────────────────────────────

def run(norm_dir: str = None, out_dir: str = None,
        baseline_df: pd.DataFrame = None, rho: float = RHO):
    nd = norm_dir or NORM_DIR
    od = out_dir  or OUT_DIR
    os.makedirs(od, exist_ok=True)

    print(f"\n[Step 5] Trade-creation shock (rho={rho:.0%})")
    cf_2021_res = counterfactual_2021(nd, rho=rho)

    if baseline_df is None:
        baseline_df = pd.read_csv(os.path.join(od, "gvc_baseline.csv"))

    # Sort by economy so index aligns with ALL_ECONOMIES_ORDERED
    actual_2021_rows = (baseline_df[baseline_df["year"] == 2021]
                        .set_index("economy")
                        .reindex(ALL_ECONOMIES_ORDERED))
    actual_2021 = {m: actual_2021_rows[m].values
                   for m in ["gvc_back", "gvc_fwd", "gvc_total", "gross_exp"]}

    # Print summary of deltas for bloc economies
    print("\n  Counterfactual vs Actual 2021 (bloc economies only):")
    print(f"  {'Economy':<12} {'ΔGVC_total':>12} {'ΔGVC_back':>12} {'ΔGVC_fwd':>12}")
    for eco in BLOC_CODES:
        if eco not in ALL_ECONOMIES_ORDERED:
            continue
        idx = ALL_ECONOMIES_ORDERED.index(eco)
        d_tot  = float(cf_2021_res["gvc_total"][idx] - actual_2021["gvc_total"][idx])
        d_back = float(cf_2021_res["gvc_back"][idx]  - actual_2021["gvc_back"][idx])
        d_fwd  = float(cf_2021_res["gvc_fwd"][idx]   - actual_2021["gvc_fwd"][idx])
        print(f"  {eco:<12} {d_tot:>+12.4f} {d_back:>+12.4f} {d_fwd:>+12.4f}")

    print("\n[Step 6] Forecasting 2022–2024")
    forecast_years = [2022, 2023, 2024]
    bau_df  = project_bau(baseline_df, forecast_years)
    int_df  = project_integrated(bau_df, actual_2021, cf_2021_res, forecast_years)

    cf_rows = []
    for i, eco in enumerate(ALL_ECONOMIES_ORDERED):
        cf_rows.append({
            "year": 2021, "economy": eco,
            "name": BLOC_NAMES.get(eco, eco),
            "in_bloc": eco in BLOC_CODES,
            "scenario": "Integrated_2021",
            "gvc_back":  float(cf_2021_res["gvc_back"][i]),
            "gvc_fwd":   float(cf_2021_res["gvc_fwd"][i]),
            "gvc_total": float(cf_2021_res["gvc_total"][i]),
            "gross_exp": float(cf_2021_res["gross_exp"][i]),
        })
    cf_2021_df = pd.DataFrame(cf_rows)

    bau_df.to_csv(os.path.join(od, "forecast_bau.csv"),         index=False)
    int_df.to_csv(os.path.join(od, "forecast_integrated.csv"),  index=False)
    cf_2021_df.to_csv(os.path.join(od, "counterfactual_2021.csv"), index=False)

    print(f"\nPhase 4 complete -> {od}")
    return bau_df, int_df, cf_2021_df


if __name__ == "__main__":
    run()
