"""
Phase 1: Preprocessing & Harmonization  (OPTIMISED)

Speed improvements over original:
  1. Uses `calamine` engine (Rust-based) for Excel reading — 5-10x faster than openpyxl.
     Falls back to openpyxl automatically if calamine is not installed.
  2. Replaces pure-Python nested loops in reindex() with NumPy advanced indexing.
  3. Uses float32 instead of float64 — halves memory, speeds up downstream matmul.
  4. Parallel year processing via ProcessPoolExecutor (one process per year).
"""

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from config import (DATA_DIR, NORM_DIR, RAW_FILES, YEARS,
                    ALL_ECONOMIES_ORDERED, N_ECONOMIES, N_SECTORS, N_FD_COLS)

DTYPE = np.float32
MAX_WORKERS = min(4, len(YEARS))


# ── Excel reading ─────────────────────────────────────────────────────────────

def _best_engine() -> str:
    try:
        import python_calamine  # noqa: F401
        return "calamine"
    except ImportError:
        return "openpyxl"


def _sheet_name(year: int, path: str) -> str:
    candidates = [str(year), f"ADB MRIO {year}"]
    xl = pd.ExcelFile(path, engine=_best_engine())
    for name in xl.sheet_names:
        if name in candidates:
            return name
    return xl.sheet_names[0]


def load_raw_excel(year: int, src_dir: str) -> pd.DataFrame:
    path   = os.path.join(src_dir, RAW_FILES[year])
    engine = _best_engine()
    sheet  = _sheet_name(year, path)
    print(f"  [{year}] Reading ({engine}) …", end=" ", flush=True)
    df = pd.read_excel(path, sheet_name=sheet, header=None,
                       engine=engine, dtype=object)
    print(f"shape={df.shape}")
    return df


# ── Structure parsing ─────────────────────────────────────────────────────────

def extract_economy_order(df: pd.DataFrame) -> list:
    col2 = df.iloc[7:, 2].fillna("").astype(str).values
    col3 = df.iloc[7:, 3].fillna("").astype(str).values
    order, seen = [], set()
    for eco, sec in zip(col2, col3):
        eco = eco.strip()
        if eco not in ("", "nan", "ToT") and sec.startswith("c") and eco not in seen:
            order.append(eco)
            seen.add(eco)
    return order


def build_index_map(actual_order: list) -> np.ndarray:
    pos = {eco: i for i, eco in enumerate(actual_order)}
    return np.array([pos.get(e, -1) for e in ALL_ECONOMIES_ORDERED], dtype=np.int32)


# ── Matrix extraction (vectorised — no Python loops) ─────────────────────────

def extract_matrices(df: pd.DataFrame, eco_order: list):
    N = len(eco_order)
    S = N_SECTORS
    F = N_FD_COLS

    col3     = df.iloc[7:, 3].fillna("").astype(str).values
    row_mask = np.array([v.startswith("c") for v in col3])
    row_idx  = np.where(row_mask)[0] + 7

    z_col_end = 4 + N * S
    y_col_end = z_col_end + N * F

    data_block = df.iloc[row_idx, :]
    Z_raw = data_block.iloc[:, 4:z_col_end].values.astype(DTYPE)
    Y_raw = data_block.iloc[:, z_col_end:y_col_end].values.astype(DTYPE)
    # X (gross output) is in the TOTAL column immediately after the Y block.
    # Using iloc[:, -1] is fragile — some files (e.g. 2020) have trailing NaN
    # columns, so we anchor on y_col_end which is always correct.
    X_raw = data_block.iloc[:, y_col_end].values.astype(DTYPE)

    eco_idx = build_index_map(eco_order)

    def _flat_perm(idx_arr, stride):
        perm = np.full(N_ECONOMIES * stride, -1, dtype=np.int32)
        for ci, ai in enumerate(idx_arr):
            if ai >= 0:
                perm[ci*stride:(ci+1)*stride] = np.arange(
                    ai*stride, (ai+1)*stride, dtype=np.int32)
        return perm

    row_perm   = _flat_perm(eco_idx, S)
    col_perm_Z = _flat_perm(eco_idx, S)
    col_perm_Y = _flat_perm(eco_idx, F)

    NS_out = N_ECONOMIES * S
    r_dst  = np.where(row_perm >= 0)[0]
    r_src  = row_perm[r_dst]
    c_dst  = np.where(col_perm_Z >= 0)[0]
    c_src  = col_perm_Z[c_dst]

    Z = np.zeros((NS_out, NS_out), dtype=DTYPE)
    Z[np.ix_(r_dst, c_dst)] = Z_raw[np.ix_(r_src, c_src)]

    c_dst_Y = np.where(col_perm_Y >= 0)[0]
    c_src_Y = col_perm_Y[c_dst_Y]
    Y_flat  = np.zeros((NS_out, N_ECONOMIES * F), dtype=DTYPE)
    Y_flat[np.ix_(r_dst, c_dst_Y)] = Y_raw[np.ix_(r_src, c_src_Y)]
    Y = Y_flat.reshape(NS_out, N_ECONOMIES, F)

    X = np.zeros(NS_out, dtype=DTYPE)
    X[r_dst] = X_raw[r_src]

    return Z, Y, X


# ── Per-year worker ───────────────────────────────────────────────────────────

def _process_one_year(args):
    year, src_dir, dst_dir = args
    try:
        df        = load_raw_excel(year, src_dir)
        eco_order = extract_economy_order(df)
        Z, Y, X   = extract_matrices(df, eco_order)
        out_path  = os.path.join(dst_dir, f"mrio_{year}.npz")
        np.savez_compressed(out_path, Z=Z, Y=Y, X=X,
                            eco_order=np.array(eco_order))
        return year, True, Z.shape, float(Z.sum()), float(X.sum())
    except Exception as e:
        import traceback
        return year, False, None, None, traceback.format_exc()


# ── main ──────────────────────────────────────────────────────────────────────

def run(src_dir: str = None, out_dir: str = None, workers: int = MAX_WORKERS):
    src = src_dir or DATA_DIR
    dst = out_dir or NORM_DIR
    os.makedirs(dst, exist_ok=True)

    print(f"Phase 1: {len(YEARS)} years | workers={workers} | engine={_best_engine()} | dtype={DTYPE}\n")
    tasks = [(year, src, dst) for year in YEARS]

    if workers == 1:
        results = [_process_one_year(t) for t in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_process_one_year, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                results.append(fut.result())

    summary = {}
    for year, ok, shape, z_sum, extra in sorted(results, key=lambda r: r[0]):
        if ok:
            print(f"  [{year}] ✓  Z={shape}  Z_sum={z_sum:.0f}  X_sum={extra:.0f}")
            summary[year] = {"shape": shape, "Z_sum": z_sum}
        else:
            print(f"  [{year}] ✗  ERROR:\n{extra}")

    print(f"\n✅  Phase 1 complete → {dst}")
    return summary


if __name__ == "__main__":
    run()
