"""
Phase 2 & 3: Leontief Engine + Full KWW (2014) Decomposition

Implements the complete Koopman, Wang & Wei (2014) value-added decomposition.
Reference: Koopman R, Wang Z, Wei S-J (2014) "Tracing Value-Added and Double
Counting in Gross Exports." American Economic Review 104(2):459-494.

DVA vs DVX — the key distinction
──────────────────────────────────
DVA (Domestic Value Added absorbed abroad):
  Country i's VA that travels directly to final demand in country j.
  Uses the OWN-COUNTRY Leontief block L_ii only.
  Formula: DVA_i = Σ_{j≠i}  v_i · L_ii · E^Y_{ij}
  where E^Y_{ij} = final-demand exports from i to j.

DVX (Domestic Value Added re-exported):
  Country i's VA that is sold as intermediate input to country j,
  which then processes and re-exports it to any third country k.
  Uses OFF-DIAGONAL Leontief blocks L_ij (i's output flowing via j).
  Formula: DVX_i = Σ_{j≠i} Σ_{k≠j}  v_i · L_ij · A_jk · L_kk · E^total_k
  This is what makes DVX genuinely different from DVA — it measures
  upstream supplier participation, not direct final-demand reach.

FVA (Foreign Value Added in i's exports):
  VA created in country j that is embodied in i's total gross exports.
  Formula: FVA_i = Σ_{j≠i}  v_j · L_ji · E^total_i
  Uses L_ji: how j's sectors contribute to i's production.

GVC indices (normalised by gross exports E_i):
  Backward participation = FVA_i / E_i   (i buys foreign VA to export)
  Forward participation  = DVX_i / E_i   (i supplies VA for others' exports)
  Total participation    = (FVA_i + DVX_i) / E_i
"""

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from config import (NORM_DIR, OUT_DIR, YEARS, ALL_ECONOMIES_ORDERED,
                    N_ECONOMIES, N_SECTORS, N_FD_COLS, BLOC_CODES, BLOC_NAMES)

DTYPE       = np.float32
MAX_WORKERS = min(4, len(YEARS))

N  = N_ECONOMIES
S  = N_SECTORS
F  = N_FD_COLS
NS = N * S


# ── Leontief building blocks ──────────────────────────────────────────────────

def technical_coefficients(Z: np.ndarray, X: np.ndarray) -> np.ndarray:
    """A_ij = Z_ij / X_j  (input requirement per unit output of j)."""
    Xsafe = np.where(X > 0, X, np.float32(1.0))
    return (Z / Xsafe[np.newaxis, :]).astype(DTYPE)


def leontief_inverse(A: np.ndarray) -> np.ndarray:
    """L = (I - A)^{-1} via LAPACK. float32 in, float32 out."""
    try:
        from scipy.linalg import solve
        I = np.eye(A.shape[0], dtype=A.dtype)
        return solve(I - A, I, assume_a="gen", check_finite=False).astype(DTYPE)
    except ImportError:
        I = np.eye(A.shape[0], dtype=A.dtype)
        return np.linalg.solve(I - A, I).astype(DTYPE)


def value_added_coeff(A: np.ndarray) -> np.ndarray:
    """v_j = max(0, 1 - Σ_i A_ij)  —  VA share of output in sector j."""
    return np.clip(np.float32(1.0) - A.sum(axis=0), np.float32(0), None)


# ── Gross exports (vectorised) ────────────────────────────────────────────────

def gross_exports_vec(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Total gross exports of each sector-economy.
    E_i = Σ_{j≠i} Z_ij  +  Σ_{j≠i} Y_ij
    Returns shape (NS,).
    """
    # --- intermediate exports ---
    Zr = Z.reshape(N, S, N, S).copy()
    idx = np.arange(N)
    Zr[idx, :, idx, :] = 0                 # zero own-country diagonal blocks
    E_int = Zr.reshape(NS, NS).sum(axis=1)

    # --- final demand exports ---
    Yr = Y.copy()                           # (NS, N, F)
    for i in range(N):
        Yr[i*S:(i+1)*S, i, :] = 0          # zero own-country final demand
    E_fd = Yr.sum(axis=(1, 2))

    return (E_int + E_fd).astype(DTYPE)


def final_demand_exports(Y: np.ndarray) -> np.ndarray:
    """
    E^Y_{ij}: final-demand exports FROM economy i TO economy j.
    Returns shape (N, S, N) — [src_eco, src_sec, dst_eco], own diagonal zeroed.
    """
    # Y shape: (NS, N, F)  →  (N, S, N, F)
    Yr = Y.reshape(N, S, N, F).copy()
    for i in range(N):
        Yr[i, :, i, :] = 0                 # zero own-economy final demand
    return Yr.sum(axis=3)                   # (N, S, N)  summed over FD categories


# ── Full KWW decomposition ────────────────────────────────────────────────────

def kww_decompose(Z: np.ndarray, Y: np.ndarray, X: np.ndarray) -> dict:
    """
    Full KWW (2014) decomposition.

    All block-level operations use NumPy reshape + einsum.
    Index convention for 4-D reshapes:
      first two indices  = source  (economy i, sector s)
      last  two indices  = destination (economy j, sector t)
    """
    A  = technical_coefficients(Z, X)      # (NS, NS)
    L  = leontief_inverse(A)               # (NS, NS)
    v  = value_added_coeff(A)              # (NS,)
    E  = gross_exports_vec(Z, Y)           # (NS,)  total gross exports
    EY = final_demand_exports(Y)           # (N, S, N)  final-demand exports

    # Economy-level total gross exports: shape (N,)
    E_eco = E.reshape(N, S).sum(axis=1)   # Σ_s E_{is}

    # ── Reshape L and A into (N, S, N, S) economy blocks ─────────────────────
    Lb = L.reshape(N, S, N, S)            # L_{is,jt}  →  Lb[i,s,j,t]
    Ab = A.reshape(N, S, N, S)            # A_{is,jt}  →  Ab[i,s,j,t]

    # Diagonal own-country Leontief blocks: L_ii  shape (N, S, S)
    L_diag = np.stack([Lb[i, :, i, :] for i in range(N)])   # (N, S, S)

    # ── VA coefficient reshaped ───────────────────────────────────────────────
    vb = v.reshape(N, S)                  # (N, S)

    # ─────────────────────────────────────────────────────────────────────────
    # DVA: domestic VA absorbed directly as final demand abroad
    #
    #   DVA_i = Σ_{j≠i}  v_i · L_ii · E^Y_{ij}
    #
    # For each source economy i:
    #   step 1: vL_ii[i, t] = Σ_s  v[i,s] * L_ii[i, s, t]   shape (N, S)
    #   step 2: DVA_i = Σ_{j≠i} Σ_t  vL_ii[i, t] * EY[i, t, j]
    #
    # einsum 'is,ist->it' gives vL_ii: (N,S)
    # then 'it,itj->i' gives DVA but INCLUDING own j=i diagonal
    # subtract j=i term separately.
    # ─────────────────────────────────────────────────────────────────────────
    vL_diag = np.einsum("is,ist->it", vb, L_diag, optimize=True)  # (N, S)

    # Total over all j (incl i=j), then subtract own-j term
    dva_total = np.einsum("it,itj->i", vL_diag, EY, optimize=True)   # (N,)
    # Own-j diagonal contribution: j=i, i.e., EY[i, t, i]
    dva_own   = np.einsum("it,it->i",  vL_diag,
                          np.stack([EY[i, :, i] for i in range(N)]),
                          optimize=True)                               # (N,)
    DVA = dva_total - dva_own                                          # (N,)

    # ─────────────────────────────────────────────────────────────────────────
    # DVX: domestic VA of i re-exported by j onward to k
    #
    #   DVX_i = Σ_{j≠i} Σ_{k≠j}  v_i · L_ij · A_jk · L_kk · E^total_k
    #
    # Steps:
    #   1. vL_ij[i,t] = Σ_s v[i,s] * Lb[i,s,j,t]      for each (i,j)  → (N,N,S)
    #   2. vLA_ijk[i,u]= Σ_t vL_ij[i,j,t] * Ab[j,t,k,u] for each (k)  → (N,N,N,S)
    #   3. vLAL_i = Σ_u  vLA_ijk[i,u] * L_kk[k,u,·] summed w/ E_k
    #
    # The formula collapses to:
    #   DVX_i = Σ_{j≠i} Σ_{k≠j} [v_i · L_ij] · [A_jk · L_kk · E_k]
    #
    # Define:
    #   B_jk[t] = Σ_u Σ_m  A[j,t,k,u] * L_kk[k,u,m] * E[k,m]
    #           = [Ab[j,:,k,:] @ L_diag[k] @ Eb[k,:]]_t     shape scalar per (j,k,t)
    #
    # Then:
    #   DVX_i = Σ_{j≠i} Σ_{k≠j}  vL_ij[i,j,:] · B_jk[:]
    # ─────────────────────────────────────────────────────────────────────────
    Eb = E.reshape(N, S)                  # (N, S)

    # B_jk[t]: for each (j,k,t),  Σ_u Σ_m  A[j,t,k,u] * L_kk[k,u,m] * E[k,m]
    # = Ab[j,:,k,:] @ (L_diag[k] @ Eb[k])
    # First compute L_kk @ E_k for each k → shape (N, S)
    LkEk = np.einsum("kst,kt->ks", L_diag, Eb, optimize=True)  # (N, S)

    # B[j,k,t] = Σ_u Ab[j,t,k,u] * LkEk[k,u]
    # einsum: 'jtku, ku -> jkt'
    B = np.einsum("jtku,ku->jkt", Ab, LkEk, optimize=True)     # (N, N, S)
    # B[j,k,t]: the demand-pull from k that flows back through j's sector t

    # vL_ij[i,j,t] = Σ_s v[i,s] * Lb[i,s,j,t]
    # einsum: 'is, isjt -> ijt'
    vLb = np.einsum("is,isjt->ijt", vb, Lb, optimize=True)     # (N, N, S)

    # DVX_i = Σ_{j≠i} Σ_{k≠j}  Σ_t  vLb[i,j,t] * B[j,k,t]
    # Use inclusion-exclusion on the two diagonal constraints (j≠i, k≠j).
    # Let idx = arange(N) for diagonal extraction.
    _idx = np.arange(N)

    # Full sum over all (i,j,k): einsum 'ijt,jkt->i'
    dvx_all = np.einsum("ijt,jkt->i", vLb, B, optimize=True)   # (N,)

    # Subtract j==i terms: vLb[i,i,t] is the i-diagonal slice → shape (N,S)
    vLb_diag_j = vLb[_idx, _idx, :]                             # (N, S)  vLb[i,i,:]
    dvx_sub_ji = np.einsum("it,ikt->i", vLb_diag_j,
                            B, optimize=True)                    # (N,)

    # Subtract k==j terms: B[j,j,t] is the j-diagonal slice → shape (N,S)
    B_diag_k = B[_idx, _idx, :]                                  # (N, S)  B[j,j,:]
    dvx_sub_kj = np.einsum("ijt,jt->i",
                            vLb, B_diag_k, optimize=True)        # (N,)

    # Add back the doubly-subtracted j==i AND k==j term
    dvx_add_back = np.einsum("it,it->i",
                              vLb_diag_j, B_diag_k,
                              optimize=True)                      # (N,)

    DVX = dvx_all - dvx_sub_ji - dvx_sub_kj + dvx_add_back     # (N,)

    # ─────────────────────────────────────────────────────────────────────────
    # FVA: foreign VA embodied in i's gross exports
    #
    #   FVA_i = Σ_{j≠i}  v_j · L_ji · E^total_i
    #
    # L_ji[j,s,t] = Lb[j,s,i,t]  (j's output that feeds into i's sector t)
    # v_j · L_ji  →  vL_ji[j,i,t] = Σ_s v[j,s] * Lb[j,s,i,t]
    #
    # FVA_i = Σ_{j≠i} Σ_t  vL_ji[j,i,t] * E[i,t]
    #
    # vLb[j,i,t] already computed above (same array, different index reading).
    # FVA_i = Σ_j Σ_t vLb[j,i,t] * Eb[i,t]  minus own j=i diagonal.
    # ─────────────────────────────────────────────────────────────────────────

    # fva_all[i] = Σ_j Σ_t  vLb[j,i,t] * Eb[i,t]   (all j incl. j=i)
    # einsum: 'jit, it -> i'
    fva_all = np.einsum("jit,it->i", vLb, Eb, optimize=True)   # (N,)

    # Own diagonal (j==i): Σ_t vLb[i,i,t] * Eb[i,t]
    fva_own = np.einsum("it,it->i",
                         vLb[_idx, _idx, :], Eb,
                         optimize=True)                          # (N,)

    FVA = fva_all - fva_own                                     # (N,)

    # ── GVC indices ───────────────────────────────────────────────────────────
    E_safe    = np.where(E_eco > 0, E_eco, np.nan)
    gvc_back  = FVA / E_safe
    gvc_fwd   = DVX / E_safe
    gvc_total = (FVA + DVX) / E_safe

    return {
        "dva": DVA, "fva": FVA, "dvx": DVX,
        "gvc_back":  gvc_back,
        "gvc_fwd":   gvc_fwd,
        "gvc_total": gvc_total,
        "gross_exp": E_eco,
    }


# ── Per-year worker ───────────────────────────────────────────────────────────

def _process_one_year(args):
    year, nd = args
    try:
        path = os.path.join(nd, f"mrio_{year}.npz")
        data = np.load(path, allow_pickle=True)
        Z = data["Z"].astype(DTYPE)
        Y = data["Y"].astype(DTYPE)
        X = data["X"].astype(DTYPE)
        print(f"  [{year}] KWW …", end=" ", flush=True)
        res = kww_decompose(Z, Y, X)
        print("done")
        return year, True, res
    except Exception as e:
        import traceback
        return year, False, traceback.format_exc()


def build_results_df(results: dict) -> pd.DataFrame:
    rows = []
    for year, res in results.items():
        for i, eco in enumerate(ALL_ECONOMIES_ORDERED):
            rows.append({
                "year":      year,
                "economy":   eco,
                "name":      BLOC_NAMES.get(eco, eco),
                "in_bloc":   eco in BLOC_CODES,
                "dva":       float(res["dva"][i]),
                "fva":       float(res["fva"][i]),
                "dvx":       float(res["dvx"][i]),
                "gvc_back":  float(res["gvc_back"][i]),
                "gvc_fwd":   float(res["gvc_fwd"][i]),
                "gvc_total": float(res["gvc_total"][i]),
                "gross_exp": float(res["gross_exp"][i]),
            })
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def run(norm_dir: str = None, out_dir: str = None, workers: int = MAX_WORKERS):
    nd = norm_dir or NORM_DIR
    od = out_dir  or OUT_DIR
    os.makedirs(od, exist_ok=True)

    print(f"Phase 2/3: {len(YEARS)} years | workers={workers} | dtype={DTYPE}")
    print("KWW method: full Koopman-Wang-Wei (2014) three-country decomposition\n")

    tasks = [(year, nd) for year in YEARS]

    if workers == 1:
        raw = [_process_one_year(t) for t in tasks]
    else:
        raw = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_process_one_year, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                raw.append(fut.result())

    results, leontief_cache = {}, {}
    for item in sorted(raw, key=lambda r: r[0]):
        year, ok = item[0], item[1]
        if ok:
            results[year] = item[2]
        else:
            print(f"  [{year}] ✗  ERROR:\n{item[2]}")

    # Rebuild A/L cache for phase4
    for year in YEARS:
        path = os.path.join(nd, f"mrio_{year}.npz")
        data = np.load(path, allow_pickle=True)
        Z = data["Z"].astype(DTYPE)
        Y = data["Y"].astype(DTYPE)
        X = data["X"].astype(DTYPE)
        A = technical_coefficients(Z, X)
        L = leontief_inverse(A)
        leontief_cache[year] = {"A": A, "L": L, "Z": Z, "Y": Y, "X": X}

    df = build_results_df(results)
    csv_path = os.path.join(od, "gvc_baseline.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅  Phase 2/3 complete → {csv_path}")

    cache_path = os.path.join(od, "leontief_cache.npz")
    save_dict  = {}
    for year, d in leontief_cache.items():
        for key, val in d.items():
            save_dict[f"{year}_{key}"] = val
    np.savez_compressed(cache_path, **save_dict)
    print(f"   Leontief cache → {cache_path}")

    return df, leontief_cache


if __name__ == "__main__":
    run()
