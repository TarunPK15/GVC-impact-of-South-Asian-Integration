"""
run_all.py  –  Master pipeline runner  (OPTIMISED)

Usage:
    python run_all.py [--data-dir PATH] [--norm-dir PATH] [--out-dir PATH]
                      [--workers N] [--skip-phase1] [--install-calamine]

Defaults:
    --data-dir   data/raw
    --norm-dir   data/normalized
    --out-dir    outputs
    --workers    4   (parallel Excel readers / KWW workers)
"""

import argparse, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",  default=str(ROOT / "data" / "raw"))
    p.add_argument("--norm-dir",  default=str(ROOT / "data" / "normalized"))
    p.add_argument("--out-dir",   default=str(ROOT / "outputs"))
    p.add_argument("--workers",   type=int, default=4,
                   help="Parallel workers for Phase 1 and 2/3 (default 4). "
                        "Set to 1 to run sequentially.")
    p.add_argument("--skip-phase1", action="store_true",
                   help="Skip Phase 1 if .npz files already exist")
    p.add_argument("--install-calamine", action="store_true",
                   help="Auto-install python-calamine before running (fast Excel engine)")
    return p.parse_args()


def try_install_calamine():
    import subprocess
    print("Installing python-calamine (fast Excel engine) …")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "python-calamine", "-q"])
    print("  ✓ python-calamine installed\n")


def check_raw_files(data_dir: str):
    from config import RAW_FILES
    missing = [f for f in RAW_FILES.values()
               if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        print("\n⚠️  Missing raw files in", data_dir)
        for f in missing:
            print("   •", f)
        sys.exit(1)
    print(f"✅  All {len(RAW_FILES)} raw files found in {data_dir}")


def banner(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def main():
    args = parse_args()

    if args.install_calamine:
        try_install_calamine()

    import config as cfg
    cfg.DATA_DIR = args.data_dir
    cfg.NORM_DIR = args.norm_dir
    cfg.OUT_DIR  = args.out_dir

    for d in [args.data_dir, args.norm_dir, args.out_dir]:
        os.makedirs(d, exist_ok=True)

    check_raw_files(args.data_dir)

    total_start = time.time()

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if args.skip_phase1:
        banner("PHASE 1: SKIPPED (--skip-phase1)")
    else:
        banner("PHASE 1: Preprocessing & Harmonization")
        t0 = time.time()
        import phase1_preprocess as p1
        p1.run(src_dir=args.data_dir, out_dir=args.norm_dir, workers=args.workers)
        print(f"   ⏱  {time.time()-t0:.1f}s")

    # ── Phase 2/3 ─────────────────────────────────────────────────────────────
    banner("PHASE 2/3: Leontief Engine & KWW Decomposition")
    t0 = time.time()
    import phase2_leontief as p2
    baseline_df, _ = p2.run(norm_dir=args.norm_dir, out_dir=args.out_dir,
                             workers=args.workers)
    print(f"   ⏱  {time.time()-t0:.1f}s")

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    banner("PHASE 4: Simulation & Forecasting")
    t0 = time.time()
    import phase4_simulation as p4
    p4.run(norm_dir=args.norm_dir, out_dir=args.out_dir, baseline_df=baseline_df)
    print(f"   ⏱  {time.time()-t0:.1f}s")

    # ── Phase 5 ───────────────────────────────────────────────────────────────
    banner("PHASE 5: Visualization & Final Analysis")
    t0 = time.time()
    import phase5_visualize as p5
    p5.run(out_dir=args.out_dir)
    print(f"   ⏱  {time.time()-t0:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    banner(f"✅  ALL PHASES COMPLETE  ({elapsed/60:.1f} min total)")
    print(f"\n Output directory: {args.out_dir}")
    for f in sorted(os.listdir(args.out_dir)):
        size = os.path.getsize(os.path.join(args.out_dir, f))
        print(f"   {f:45s} {size/1024:8.1f} KB")


if __name__ == "__main__":
    main()
