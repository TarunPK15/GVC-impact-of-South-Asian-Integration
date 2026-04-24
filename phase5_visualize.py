"""
Phase 5: Visualization & Final Analysis  (CORRECTED FRAMING)

Core framing change
───────────────────
Previous version plotted raw GVC index *levels*, which meant the 2020-2021
COVID crash dominated every chart and made integration appear harmful.

All plots now show the *gain* from integration — the difference between the
Integrated and BAU paths — so the question answered is always:
  "What does integration add relative to where we'd be without it?"

The primary reference point is 2024 (the policy endpoint), not 2021 (the
shock year used only as the simulation entry point).

Plot inventory
──────────────
  1. GVC Gain Time Series     – Integrated minus BAU, all years 2022-2024,
                                 bloc average, for Total / Backward / Forward
  2. 2024 Gain by Country     – Horizontal bar chart of gain at 2024 per economy
                                 for all three GVC metrics side-by-side
  3. Gain Heatmap             – Country × Metric matrix of 2024 gains (ppt)
  4. Forward vs Backward      – Scatter: each economy's Fwd gain vs Bwd gain at
                                 2024, sized by gross exports — shows which
                                 economies gain more as suppliers vs buyers
  5. Dashboard (6-panel)      – Summary view combining gain series + 2024
                                 country bars + cumulative gain bar + scatter
  6. Summary Excel workbook
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import (OUT_DIR, YEARS, BLOC_CODES, BLOC_NAMES, ALL_ECONOMIES_ORDERED)

# ── Style ─────────────────────────────────────────────────────────────────────
BAU_COLOR   = "#6baed6"   # muted blue  — BAU reference
INT_COLOR   = "#2ca02c"   # green       — integrated path / gain
FWD_COLOR   = "#d62728"   # red         — forward linkage
BWD_COLOR   = "#ff7f0e"   # orange      — backward linkage
TOT_COLOR   = "#9467bd"   # purple      — total GVC
NEG_COLOR   = "#d62728"   # red         — negative gain
FIG_DPI     = 150

plt.rcParams.update({
    "figure.dpi":        FIG_DPI,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
})

METRICS = {
    "gvc_total": ("Total GVC",    TOT_COLOR),
    "gvc_back":  ("Backward GVC (FVA)", BWD_COLOR),
    "gvc_fwd":   ("Forward GVC (DVX)",  FWD_COLOR),
}
FORECAST_YEARS = [2022, 2023, 2024]


# ── Data loading & gain computation ──────────────────────────────────────────

def load_data(out_dir: str):
    baseline = pd.read_csv(os.path.join(out_dir, "gvc_baseline.csv"))
    bau      = pd.read_csv(os.path.join(out_dir, "forecast_bau.csv"))
    integ    = pd.read_csv(os.path.join(out_dir, "forecast_integrated.csv"))
    cf2021   = pd.read_csv(os.path.join(out_dir, "counterfactual_2021.csv"))
    return baseline, bau, integ, cf2021


def compute_gain(bau: pd.DataFrame, integ: pd.DataFrame) -> pd.DataFrame:
    """
    gain = integrated - BAU, per economy per year per metric.
    This is the policy-relevant quantity: what integration ADDS over baseline.
    Units: percentage points (index already expressed as fraction, so *100 for ppt).
    """
    metrics = list(METRICS.keys()) + ["gross_exp"]
    merged  = bau.merge(integ, on=["year", "economy", "in_bloc"],
                        suffixes=("_bau", "_int"))
    for m in metrics:
        merged[f"gain_{m}"] = (merged[f"{m}_int"] - merged[f"{m}_bau"]) * 100
    return merged


def bloc_gain(gain_df: pd.DataFrame, metric: str) -> pd.Series:
    """Bloc-average gain for a given metric, indexed by year."""
    sub = gain_df[gain_df["economy"].isin(BLOC_CODES)]
    return sub.groupby("year")[f"gain_{metric}"].mean()


def country_gain_2024(gain_df: pd.DataFrame) -> pd.DataFrame:
    """Per-country gains at 2024, bloc economies only."""
    sub = gain_df[(gain_df["year"] == 2024) & (gain_df["economy"].isin(BLOC_CODES))].copy()
    sub["name"] = sub["economy"].map(BLOC_NAMES)
    return sub.sort_values("gain_gvc_total", ascending=True)


# ── Plot 1: GVC Gain Time Series ──────────────────────────────────────────────

def plot_gain_series(gain_df: pd.DataFrame, out_dir: str) -> str:
    """
    Line chart: Integrated − BAU gain over 2022-2024, bloc average.
    Three lines: Total, Backward, Forward GVC gain.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for m, (label, color) in METRICS.items():
        s = bloc_gain(gain_df, m)
        ax.plot(s.index, s.values, "o-", color=color, lw=2.5, ms=8,
                label=label, zorder=3)
        # annotate 2024 endpoint
        ax.annotate(f"{s.iloc[-1]:+.2f} ppt",
                    xy=(s.index[-1], s.iloc[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=9, color=color, va="center")

    ax.axhline(0, color="black", lw=1.0, zorder=2)
    ax.fill_between(FORECAST_YEARS,
                    bloc_gain(gain_df, "gvc_total").values,
                    0, alpha=0.08, color=TOT_COLOR)

    ax.set_title("Integration Gain over BAU: GVC Participation\n"
                 "South Asia–ASEAN Bloc Average (Integrated − BAU, ppt)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Gain in GVC Index (percentage points)")
    ax.set_xticks(FORECAST_YEARS)
    ax.legend(fontsize=10, loc="upper left")

    fig.tight_layout()
    path = os.path.join(out_dir, "plot1_gain_series.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ── Plot 2: 2024 Gain by Country ──────────────────────────────────────────────

def plot_country_gain_2024(gain_df: pd.DataFrame, out_dir: str) -> str:
    """
    Horizontal grouped bar chart: per-country gain at 2024 for
    Total / Backward / Forward GVC.
    """
    cdf   = country_gain_2024(gain_df)
    names = cdf["name"].tolist()
    n     = len(names)
    y     = np.arange(n)
    w     = 0.25

    fig, ax = plt.subplots(figsize=(11, max(6, n * 0.55)))

    bar_specs = [
        ("gain_gvc_total", "Total GVC",          TOT_COLOR, -w),
        ("gain_gvc_back",  "Backward (FVA)",      BWD_COLOR,  0),
        ("gain_gvc_fwd",   "Forward (DVX)",        FWD_COLOR, +w),
    ]
    for col, label, color, offset in bar_specs:
        vals   = cdf[col].values
        colors = [color if v >= 0 else NEG_COLOR for v in vals]
        ax.barh(y + offset, vals, w * 0.9, color=colors, alpha=0.85, label=label)

    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Gain over BAU at 2024 (percentage points)")
    ax.set_title("Per-Country Integration Gain at 2024\n"
                 "Integrated Path vs Business-As-Usual (ppt)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    path = os.path.join(out_dir, "plot2_country_gain_2024.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ── Plot 3: Gain Heatmap (Country × Metric) ───────────────────────────────────

def plot_gain_heatmap(gain_df: pd.DataFrame, out_dir: str) -> str:
    """
    Heatmap: rows = economies, cols = [Total, Backward, Forward],
    values = gain at 2024 in ppt. Colour scale centred at zero.
    Red = negative gain, Green = positive gain.
    """
    cdf     = country_gain_2024(gain_df).set_index("name")
    cols    = ["gain_gvc_total", "gain_gvc_back", "gain_gvc_fwd"]
    labels  = ["Total GVC", "Backward\n(FVA)", "Forward\n(DVX)"]
    heat    = cdf[cols].values
    names   = cdf.index.tolist()

    vmax = max(abs(heat.min()), abs(heat.max()), 0.01)

    fig, ax = plt.subplots(figsize=(7, max(6, len(names) * 0.55)))
    im = ax.imshow(heat, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Gain over BAU at 2024 (ppt)", shrink=0.8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title("Integration Gain Heatmap at 2024\n"
                 "(Integrated − BAU, percentage points)",
                 fontsize=11, fontweight="bold")

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i,j]:+.3f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(heat[i,j]) < 0.6 * vmax else "white")

    fig.tight_layout()
    path = os.path.join(out_dir, "plot3_gain_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ── Plot 4: Forward vs Backward Scatter ───────────────────────────────────────

def plot_fwd_vs_bwd_scatter(gain_df: pd.DataFrame, bau: pd.DataFrame,
                             out_dir: str) -> str:
    """
    Scatter plot at 2024:
      x-axis = Backward GVC gain (FVA, measures buyer integration)
      y-axis = Forward GVC gain  (DVX, measures supplier integration)
      size   = BAU gross exports (economy size)
      colour = which quadrant

    Quadrant interpretation:
      Top-right:    gains as BOTH supplier and buyer — deepest integration
      Bottom-right: gains mainly as buyer (imports more foreign VA)
      Top-left:     gains mainly as supplier (others process your VA)
      Bottom-left:  little gain from either channel
    """
    cdf  = country_gain_2024(gain_df).copy()
    # Get BAU 2024 gross exports for bubble sizing
    bau24 = (bau[bau["year"] == 2024]
             [bau["economy"].isin(BLOC_CODES)]
             .set_index("economy")["gross_exp"])
    cdf["gross_exp_bau"] = cdf["economy"].map(bau24)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Quadrant shading
    xlim = (cdf["gain_gvc_back"].min() - 0.005,
            cdf["gain_gvc_back"].max() + 0.005)
    ylim = (cdf["gain_gvc_fwd"].min()  - 0.005,
            cdf["gain_gvc_fwd"].max()  + 0.005)
    ax.axhline(0, color="gray", lw=0.8, zorder=1)
    ax.axvline(0, color="gray", lw=0.8, zorder=1)
    ax.fill_between([0, xlim[1]], [0, 0], [ylim[1], ylim[1]],
                    alpha=0.05, color=INT_COLOR)  # top-right: both gains
    ax.fill_between([xlim[0], 0], [0, 0], [ylim[1], ylim[1]],
                    alpha=0.05, color=FWD_COLOR)  # top-left: supplier only
    ax.fill_between([0, xlim[1]], [ylim[0], ylim[0]], [0, 0],
                    alpha=0.05, color=BWD_COLOR)  # bottom-right: buyer only

    # Bubbles
    sizes = (cdf["gross_exp_bau"].fillna(cdf["gross_exp_bau"].median())
             / cdf["gross_exp_bau"].max() * 1200 + 80)
    sc = ax.scatter(cdf["gain_gvc_back"], cdf["gain_gvc_fwd"],
                    s=sizes, alpha=0.75, zorder=3,
                    c=cdf["gain_gvc_total"], cmap="RdYlGn",
                    vmin=-abs(cdf["gain_gvc_total"]).max(),
                    vmax= abs(cdf["gain_gvc_total"]).max(),
                    edgecolors="white", linewidths=0.8)
    plt.colorbar(sc, ax=ax, label="Total GVC Gain (ppt)", shrink=0.7)

    # Labels
    for _, row in cdf.iterrows():
        ax.annotate(row["name"],
                    xy=(row["gain_gvc_back"], row["gain_gvc_fwd"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)

    # Quadrant labels
    ax.text(xlim[1] * 0.95, ylim[1] * 0.95,
            "Both gains\n(deep integration)", ha="right", va="top",
            fontsize=8, color=INT_COLOR, style="italic")
    ax.text(xlim[0] * 0.95, ylim[1] * 0.95,
            "Supplier gains\n(forward only)", ha="left", va="top",
            fontsize=8, color=FWD_COLOR, style="italic")
    ax.text(xlim[1] * 0.95, ylim[0] * 0.95,
            "Buyer gains\n(backward only)", ha="right", va="bottom",
            fontsize=8, color=BWD_COLOR, style="italic")

    ax.set_xlabel("Backward GVC Gain (FVA/E, ppt) — Buyer integration")
    ax.set_ylabel("Forward GVC Gain (DVX/E, ppt) — Supplier integration")
    ax.set_title("Forward vs Backward Integration Gains at 2024\n"
                 "Bubble size = BAU Gross Exports",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    fig.tight_layout()
    path = os.path.join(out_dir, "plot4_fwd_vs_bwd_scatter.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ── Plot 5: Dashboard (6-panel) ───────────────────────────────────────────────

def plot_dashboard(gain_df: pd.DataFrame, baseline: pd.DataFrame,
                   bau: pd.DataFrame, integ: pd.DataFrame,
                   out_dir: str) -> str:
    """
    6-panel summary dashboard. Every panel is gain-based or shows the
    integrated vs BAU comparison at 2024 — not raw levels.
    """
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "South Asia–ASEAN GVC Integration: Policy Impact Dashboard\n"
        "All values show Integrated Path gain over Business-As-Usual",
        fontsize=14, fontweight="bold", y=1.01
    )
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    axs = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # ── Panel 1: Gain time series (bloc avg, 3 metrics) ──────────────────────
    ax = axs[0]
    for m, (label, color) in METRICS.items():
        s = bloc_gain(gain_df, m)
        ax.plot(s.index, s.values, "o-", color=color, lw=2, ms=6, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Bloc-Average GVC Gain over BAU\n(2022–2024, ppt)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_xticks(FORECAST_YEARS)
    ax.set_ylabel("Gain (ppt)")
    ax.legend(fontsize=7)

    # ── Panel 2: 2024 gain per country (Total GVC only) ──────────────────────
    ax = axs[1]
    cdf = country_gain_2024(gain_df)
    vals   = cdf["gain_gvc_total"].values
    names  = cdf["name"].tolist()
    colors = [INT_COLOR if v >= 0 else NEG_COLOR for v in vals]
    y      = np.arange(len(names))
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Total GVC Gain (ppt)")
    ax.set_title("Total GVC Gain per Country\nat 2024 (ppt)", fontsize=10, fontweight="bold")

    # ── Panel 3: Forward vs Backward gain per country ─────────────────────────
    ax = axs[2]
    fwd_vals = cdf["gain_gvc_fwd"].values
    bwd_vals = cdf["gain_gvc_back"].values
    w = 0.38
    ax.barh(y - w/2, bwd_vals, w, color=BWD_COLOR, alpha=0.85,
            label="Backward (FVA)")
    ax.barh(y + w/2, fwd_vals, w, color=FWD_COLOR, alpha=0.85,
            label="Forward (DVX)")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Gain (ppt)")
    ax.set_title("Forward vs Backward Gain\nper Country at 2024",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

    # ── Panel 4: Cumulative gain bar (sum across bloc, per year) ─────────────
    ax = axs[3]
    for m, (label, color) in METRICS.items():
        s = bloc_gain(gain_df, m)
        ax.bar(np.array(FORECAST_YEARS) + list(METRICS.keys()).index(m) * 0.25 - 0.25,
               s.values, 0.23, color=color, alpha=0.85, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(FORECAST_YEARS)
    ax.set_xlabel("Year")
    ax.set_ylabel("Gain (ppt)")
    ax.set_title("GVC Gain by Year & Component\n(Bloc Average, ppt)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

    # ── Panel 5: Fwd vs Bwd scatter at 2024 ──────────────────────────────────
    ax = axs[4]
    bau24 = (bau[bau["year"] == 2024]
             [bau["economy"].isin(BLOC_CODES)]
             .set_index("economy")["gross_exp"])
    cdf2 = cdf.copy()
    cdf2["gross_exp_bau"] = cdf2["economy"].map(bau24)
    sizes = (cdf2["gross_exp_bau"].fillna(cdf2["gross_exp_bau"].median())
             / cdf2["gross_exp_bau"].max() * 600 + 40)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    sc = ax.scatter(cdf2["gain_gvc_back"], cdf2["gain_gvc_fwd"],
                    s=sizes, alpha=0.75, zorder=3,
                    c=cdf2["gain_gvc_total"], cmap="RdYlGn",
                    edgecolors="white", linewidths=0.6)
    for _, row in cdf2.iterrows():
        ax.annotate(row["name"][:3],
                    xy=(row["gain_gvc_back"], row["gain_gvc_fwd"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Backward Gain (ppt)", fontsize=9)
    ax.set_ylabel("Forward Gain (ppt)", fontsize=9)
    ax.set_title("Forward vs Backward Gain\nat 2024 (bubble=exports)",
                 fontsize=10, fontweight="bold")

    # ── Panel 6: Gain heatmap (mini) ─────────────────────────────────────────
    ax = axs[5]
    cols  = ["gain_gvc_total", "gain_gvc_back", "gain_gvc_fwd"]
    hlabs = ["Total", "Bwd", "Fwd"]
    heat  = cdf.set_index("name")[cols].values
    hnames = cdf["name"].tolist()
    vmax  = max(abs(heat.min()), abs(heat.max()), 0.001)
    im    = ax.imshow(heat, aspect="auto", cmap="RdYlGn",
                      vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(3)); ax.set_xticklabels(hlabs, fontsize=9)
    ax.set_yticks(range(len(hnames))); ax.set_yticklabels(hnames, fontsize=7)
    ax.set_title("Gain Heatmap at 2024\n(ppt, green=positive)",
                 fontsize=10, fontweight="bold")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i,j]:+.3f}", ha="center", va="center",
                    fontsize=6.5, color="black")

    fig.tight_layout()
    path = os.path.join(out_dir, "plot5_dashboard.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ── Plot 6: Event Study (proper DiD coefficient plot) ─────────────────────────

def plot_event_study(baseline: pd.DataFrame, bau: pd.DataFrame,
                     integ: pd.DataFrame, out_dir: str) -> str:
    """
    Proper event study (Difference-in-Differences coefficient plot).

    DATA SOURCES (key design decision)
    ───────────────────────────────────
    • BAU path (all 8 years): *actual* MRIO-derived baseline data.
      Since the integration policy has NOT been implemented, the observed
      2022–2024 data represents the world without integration → correct BAU.
      Using a linear-trend extrapolation instead would introduce a downward
      bias from the COVID dip and create spurious negative post-coefficients.

    • Integrated path (2021 onwards): actual_t + per-country delta, where
        delta_i = CF_gvc_i(2021) − actual_gvc_i(2021)
      The counterfactual delta captures the structural GVC gain that would
      result from the trade-creation shock.  It is added to each post-period
      actual value so the integrated path inherits the observed trend PLUS
      the integration premium.

    NORMALISATION
    ─────────────
    All coefficients are expressed as deviations from the pre-period mean
    (average of 2017–2020 for each country), so that
      • pre-period coefficients test for pre-trends (parallel-trends check)
      • the base year 2021 shows the immediate shock at t = 0
      • post-period coefficients show persistent effects above the trend

    ERROR BARS
    ──────────
    95 % CI = 1.96 × (cross-country std / √N), N = 16 bloc economies.
    Statistical significance tested with one-sample t-test, H₀: β = 0.
    """
    from scipy import stats as sp_stats

    # ── 0. Per-country delta from the 2021 counterfactual ────────────────────
    # delta_i = integrated_gvc_i(first post year) − bau_gvc_i(first post year)
    # (the delta is constant across forecast years by construction)
    first_post = sorted(bau["year"].unique())[0]
    bau_fp  = (bau[bau["year"]   == first_post].set_index("economy")["gvc_total"] * 100)
    integ_fp = (integ[integ["year"] == first_post].set_index("economy")["gvc_total"] * 100)

    valid_bloc = [c for c in BLOC_CODES
                  if c in bau_fp.index and c in integ_fp.index]
    N = len(valid_bloc)
    delta = {c: float(integ_fp[c] - bau_fp[c]) for c in valid_bloc}   # ppt

    # ── 1. Actual 2021 per country — normalisation base (2021 BAU = 0) ────────
    pre_years  = sorted(yr for yr in baseline["year"].unique() if yr < 2021)
    post_years = sorted(yr for yr in baseline["year"].unique() if yr > 2021)
    all_years  = pre_years + [2021] + post_years

    actual_2021 = (baseline[baseline["year"] == 2021]
                   .set_index("economy")["gvc_total"] * 100)  # %
    base_val = {c: float(actual_2021.get(c, np.nan)) for c in valid_bloc}

    # ── 2. Helper: (β, SE, p-value) from cross-country deltas ────────────────
    def _coeff(deltas: np.ndarray):
        deltas = deltas[np.isfinite(deltas)]
        n    = len(deltas)
        if n == 0:
            return np.nan, np.nan, np.nan
        beta = float(deltas.mean())
        se   = float(deltas.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        t    = beta / se if se > 1e-12 else 0.0
        pval = float(2 * (1 - sp_stats.t.cdf(abs(t), df=n - 1)))
        return beta, se, pval

    # ── 3. Compute coefficients for all years ─────────────────────────────────
    bau_beta, bau_ci95, bau_pval = [], [], []
    int_beta, int_ci95, int_pval = [], [], []

    for yr in all_years:
        sl = baseline[baseline["year"] == yr].set_index("economy")["gvc_total"] * 100
        # BAU deviation: actual_t − actual_2021  (so 2021 BAU = 0)
        d_bau = np.array([sl.get(c, np.nan) - base_val[c] for c in valid_bloc], dtype=float)
        b, se, pv = _coeff(d_bau)
        bau_beta.append(b); bau_ci95.append(1.96 * se if np.isfinite(se) else 0)
        bau_pval.append(pv)

        if yr <= 2021:
            # Pre-period + 2021 actual: integrated path = actual (no shock yet)
            # At 2021, we also show the shock point: actual_2021 + delta
            if yr == 2021:
                d_int = np.array(
                    [sl.get(c, np.nan) + delta.get(c, 0) - base_val[c] for c in valid_bloc],
                    dtype=float)
            else:
                d_int = d_bau.copy()   # pre-period: both paths identical
        else:
            # Post-period: integrated = actual + delta
            d_int = np.array(
                [sl.get(c, np.nan) + delta.get(c, 0) - base_val[c] for c in valid_bloc],
                dtype=float)

        b2, se2, pv2 = _coeff(d_int)
        int_beta.append(b2); int_ci95.append(1.96 * se2 if np.isfinite(se2) else 0)
        int_pval.append(pv2)

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7))
    OFFSET = 0.12

    ax_years = np.array(all_years, dtype=float)
    pre_mask  = ax_years < 2021
    base_mask = ax_years == 2021
    post_mask = ax_years > 2021

    bau_b  = np.array(bau_beta,  dtype=float)
    bau_ci = np.array(bau_ci95,  dtype=float)
    int_b  = np.array(int_beta,  dtype=float)
    int_ci = np.array(int_ci95,  dtype=float)

    # Pre-period: single series (both paths equal), circles
    ax.errorbar(ax_years[pre_mask], bau_b[pre_mask], yerr=bau_ci[pre_mask],
                fmt="o", color=BAU_COLOR, ms=9, lw=2, capsize=6, capthick=2.5,
                label=f"Pre-period (actual, n={N})", zorder=4)
    for x, b, ci, pv in zip(ax_years[pre_mask], bau_b[pre_mask],
                              bau_ci[pre_mask], np.array(bau_pval)[pre_mask]):
        if np.isfinite(pv) and pv < 0.05:
            ax.text(x, b + ci + 0.02, "**" if pv < 0.01 else "*",
                    ha="center", va="bottom", fontsize=11,
                    color=BAU_COLOR, fontweight="bold")

    # 2021 base year: show both (BAU=actual, INT=actual+delta), offset slightly
    ax.errorbar([2021 - OFFSET], [bau_b[base_mask][0]], yerr=[bau_ci[base_mask][0]],
                fmt="s", color=BAU_COLOR, ms=9, lw=2, capsize=6, capthick=2.5,
                alpha=0.75, zorder=4)
    ax.errorbar([2021 + OFFSET], [int_b[base_mask][0]], yerr=[int_ci[base_mask][0]],
                fmt="D", color=INT_COLOR, ms=9, lw=2, capsize=6, capthick=2.5,
                zorder=5)

    # Post-period: BAU (squares, left-shifted) and Integrated (diamonds, right-shifted)
    ax.errorbar(ax_years[post_mask] - OFFSET, bau_b[post_mask], yerr=bau_ci[post_mask],
                fmt="s", color=BAU_COLOR, ms=9, lw=2, capsize=6, capthick=2.5,
                alpha=0.75, label="Without integration (actual data as BAU)", zorder=4)
    ax.errorbar(ax_years[post_mask] + OFFSET, int_b[post_mask], yerr=int_ci[post_mask],
                fmt="D", color=INT_COLOR, ms=9, lw=2, capsize=6, capthick=2.5,
                label="With integration (actual + counterfactual delta)", zorder=5)

    # Significance stars for post-period integrated
    for x, b, ci, pv in zip(ax_years[post_mask] + OFFSET, int_b[post_mask],
                              int_ci[post_mask], np.array(int_pval)[post_mask]):
        if np.isfinite(pv) and pv < 0.05:
            ax.text(x, b + ci + 0.02, "**" if pv < 0.01 else "*",
                    ha="center", va="bottom", fontsize=11,
                    color=INT_COLOR, fontweight="bold")

    # Treatment effect gap annotation in post period
    for i, yr in enumerate(all_years):
        if yr <= 2021:
            continue
        j = list(all_years).index(yr)
        gap = int_b[j] - bau_b[j]
        mid = (int_b[j] + bau_b[j]) / 2
        ax.annotate(f"Δ={gap:+.2f}ppt",
                    xy=(yr, mid),
                    xytext=(20, 0), textcoords="offset points",
                    fontsize=8.5, color=INT_COLOR, va="center",
                    arrowprops=dict(arrowstyle="-", color=INT_COLOR,
                                   lw=1.0, linestyle="dotted"))

    # Reference lines and shading
    ax.axhline(0, color="black", lw=1.0, zorder=2)
    ax.axvline(2021, color="#444444", linestyle="--", lw=1.8, zorder=1)
    ax.axvspan(2021.5, max(all_years) + 0.5, alpha=0.04, color=INT_COLOR)

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    ax.text(2018.5, ymax - span * 0.06, "Pre-treatment",
            ha="center", fontsize=9, color="gray", style="italic")
    ax.text(max(all_years) - 0.5, ymax - span * 0.06, "Post-treatment",
            ha="center", fontsize=9, color=INT_COLOR, style="italic")
    ax.text(2021.05, ymin + span * 0.03, "Treatment\nyear (2021)",
            color="#444444", fontsize=9, va="bottom")

    ax.text(0.01, -0.10,
            f"* p<0.05   ** p<0.01   Error bars = 95% CI   N = {N} bloc economies   "
            "Coefficients = deviation from 2017–2020 country mean   "
            "BAU = actual observed data   Integrated = actual + trade-creation delta",
            transform=ax.transAxes, fontsize=7.5, color="gray")

    ax.set_title(
        "Event Study: Effect of South Asia–ASEAN Integration on Total GVC Participation\n"
        "DiD Coefficients ± 95% CI  |  BAU = actual observed data  |  "
        "Base = 2017–2020 country mean",
        fontsize=11, fontweight="bold")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Δ Total GVC Participation relative to pre-period mean (ppt)", fontsize=11)
    ax.set_xticks(all_years)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)

    fig.tight_layout()
    path = os.path.join(out_dir, "plot6_event_study.png")
    fig.savefig(path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


    """
    Proper event study (Difference-in-Differences coefficient plot).

    For each year t, we estimate:
        β_t = mean_i [ path_i(t) − actual_i(2021) ]
    where the average is taken across all N bloc countries (i).
    Error bars = 1.96 × SE  (SE = cross-country std / √N),
    which gives the 95% confidence interval.

    PRE-PERIOD  (2017–2020): path = actual baseline data.
      β_t tests whether GVC was trending pre-treatment (parallel trends check).
      All should be statistically indistinguishable from 0.

    BASE YEAR   (2021): β = 0 by normalisation (the anchoring point).

    POST-PERIOD (2022–2024): TWO series shown side-by-side per year:
      ○  BAU  (blue)  — without integration
      ◆  Integrated (green) — with trade-creation shock
      The vertical gap between them is the treatment effect at each horizon.

    Statistical significance (against H₀: β = 0) marked with * (p<0.05),
    ** (p<0.01) above/below the integrated-path bars.
    """
    from scipy import stats as sp_stats

    # ── 0. Collect country-level base-year values ────────────────────────────
    base = (baseline[baseline["year"] == 2021]
            .set_index("economy")["gvc_total"] * 100)   # in %

    valid_bloc = [c for c in BLOC_CODES if c in base.index]
    N = len(valid_bloc)

    pre_years  = sorted(yr for yr in baseline["year"].unique() if yr < 2021)
    post_years = sorted(bau["year"].unique())

    # ── 1. Helper: compute (β, SE, p-value) for a set of country deltas ─────
    def _coeff(deltas: np.ndarray):
        """deltas shape (N,): country-level deviations from base year."""
        n    = len(deltas)
        beta = float(deltas.mean())
        se   = float(deltas.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        t    = beta / se if se > 1e-12 else 0.0
        pval = float(2 * (1 - sp_stats.t.cdf(abs(t), df=n - 1)))
        return beta, se, pval

    # ── 2. Pre-period coefficients ───────────────────────────────────────────
    pre_beta, pre_ci95, pre_pval = [], [], []
    for yr in pre_years:
        yr_slice = (baseline[baseline["year"] == yr]
                    .set_index("economy")["gvc_total"] * 100)
        deltas = np.array([yr_slice.get(c, np.nan) - base[c] for c in valid_bloc],
                          dtype=float)
        deltas = deltas[np.isfinite(deltas)]
        b, se, pv = _coeff(deltas)
        pre_beta.append(b); pre_ci95.append(1.96 * se); pre_pval.append(pv)

    # Base year: β = 0, CI = 0 (by normalisation — no uncertainty)
    pre_years_full  = pre_years + [2021]
    pre_beta_full   = pre_beta  + [0.0]
    pre_ci95_full   = pre_ci95  + [0.0]
    pre_pval_full   = pre_pval  + [1.0]

    # ── 3. Post-period BAU coefficients ─────────────────────────────────────
    bau_beta, bau_ci95, bau_pval = [], [], []
    for yr in post_years:
        yr_slice = (bau[bau["year"] == yr]
                    .set_index("economy")["gvc_total"] * 100)
        deltas = np.array([yr_slice.get(c, np.nan) - base[c] for c in valid_bloc],
                          dtype=float)
        deltas = deltas[np.isfinite(deltas)]
        b, se, pv = _coeff(deltas)
        bau_beta.append(b); bau_ci95.append(1.96 * se); bau_pval.append(pv)

    # ── 4. Post-period Integrated coefficients ───────────────────────────────
    int_beta, int_ci95, int_pval = [], [], []
    for yr in post_years:
        yr_slice = (integ[integ["year"] == yr]
                    .set_index("economy")["gvc_total"] * 100)
        deltas = np.array([yr_slice.get(c, np.nan) - base[c] for c in valid_bloc],
                          dtype=float)
        deltas = deltas[np.isfinite(deltas)]
        b, se, pv = _coeff(deltas)
        int_beta.append(b); int_ci95.append(1.96 * se); int_pval.append(pv)

    # ── 5. Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7))

    OFFSET = 0.12   # horizontal jitter so BAU and Integrated bars don't overlap

    # --- Pre-period + base year (single series, grey-blue) ---
    px = np.array(pre_years_full, dtype=float)
    ax.errorbar(px, pre_beta_full, yerr=pre_ci95_full,
                fmt="o", color=BAU_COLOR, ms=9, lw=2, capsize=6, capthick=2.5,
                label=f"Pre-period / base year (actual, n={N})", zorder=4)

    # Add significance stars for pre-period (should have none at 95%)
    for x, b, ci, pv in zip(px, pre_beta_full, pre_ci95_full, pre_pval_full):
        if pv < 0.05 and x != 2021:
            star = "**" if pv < 0.01 else "*"
            ax.text(x, b + ci + 0.02, star, ha="center", va="bottom",
                    fontsize=11, color=BAU_COLOR, fontweight="bold")

    # --- Post-period BAU (squares, lighter blue, left-shifted) ---
    bx = np.array(post_years, dtype=float) - OFFSET
    ax.errorbar(bx, bau_beta, yerr=bau_ci95,
                fmt="s", color=BAU_COLOR, ms=9, lw=2, capsize=6, capthick=2.5,
                alpha=0.75, label="Without integration (BAU)", zorder=4)

    for x, b, ci, pv in zip(bx, bau_beta, bau_ci95, bau_pval):
        if pv < 0.05:
            star = "**" if pv < 0.01 else "*"
            ax.text(x, b + ci + 0.02, star, ha="center", va="bottom",
                    fontsize=11, color=BAU_COLOR, fontweight="bold")

    # --- Post-period Integrated (diamonds, green, right-shifted) ---
    ix = np.array(post_years, dtype=float) + OFFSET
    ax.errorbar(ix, int_beta, yerr=int_ci95,
                fmt="D", color=INT_COLOR, ms=9, lw=2, capsize=6, capthick=2.5,
                label="With integration (trade creation)", zorder=5)

    for x, b, ci, pv in zip(ix, int_beta, int_ci95, int_pval):
        if pv < 0.05:
            star = "**" if pv < 0.01 else "*"
            ax.text(x, b + ci + 0.02, star, ha="center", va="bottom",
                    fontsize=11, color=INT_COLOR, fontweight="bold")

    # --- Treatment effect gap annotation ---
    for yr, bb, ib, bci, ici in zip(post_years, bau_beta, int_beta, bau_ci95, int_ci95):
        gap = ib - bb
        mid = (ib + bb) / 2
        ax.annotate(f"Δ={gap:+.2f}ppt",
                    xy=(yr, mid),
                    xytext=(18, 0), textcoords="offset points",
                    fontsize=8.5, color=INT_COLOR, va="center",
                    arrowprops=dict(arrowstyle="-", color=INT_COLOR,
                                   lw=1.2, linestyle="dotted"))

    # --- Reference lines ---
    ax.axhline(0, color="black", lw=1.0, zorder=2, linestyle="-")
    ax.axvline(2021, color="#444444", linestyle="--", lw=1.8, zorder=1)

    # Shade post-treatment region
    ax.axvspan(2021.5, max(post_years) + 0.5, alpha=0.04, color=INT_COLOR)

    # Label regions
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    ax.text(2018.5, ymax - span * 0.06, "Pre-treatment period",
            ha="center", fontsize=9, color="gray", style="italic")
    ax.text(2023.0, ymax - span * 0.06, "Post-treatment period",
            ha="center", fontsize=9, color=INT_COLOR, style="italic")
    ax.text(2021.04, ymin + span * 0.03, "Base year\n(2021)",
            color="#444444", fontsize=9, va="bottom")

    # Legend note for stars
    ax.text(0.01, -0.09,
            "* p<0.05   ** p<0.01   Error bars = 95% CI   "
            f"N = {N} bloc economies   Coefficients normalised to 2021 = 0",
            transform=ax.transAxes, fontsize=8, color="gray")

    ax.set_title(
        "Event Study: Effect of South Asia–ASEAN Integration on Total GVC Participation\n"
        "Difference-in-Differences Coefficients ± 95% CI (cross-country variation), "
        "Base Year: 2021",
        fontsize=11, fontweight="bold")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Δ Total GVC Participation relative to 2021 (ppt)", fontsize=11)
    ax.set_xticks(pre_years_full + post_years)

    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
    fig.tight_layout()

    path = os.path.join(out_dir, "plot6_event_study.png")
    fig.savefig(path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ── Summary Excel ─────────────────────────────────────────────────────────────

def build_summary_excel(gain_df: pd.DataFrame, baseline: pd.DataFrame,
                         bau: pd.DataFrame, integ: pd.DataFrame,
                         cf2021: pd.DataFrame, out_dir: str) -> str:
    path = os.path.join(out_dir, "gvc_summary.xlsx")
    metrics = list(METRICS.keys()) + ["gross_exp"]

    with pd.ExcelWriter(path, engine="openpyxl") as writer:

        # Sheet 1: Raw gain table (all years, all economies)
        gain_cols = ["year", "economy"] + [f"gain_{m}" for m in metrics]
        gain_out  = gain_df[gain_df["economy"].isin(BLOC_CODES)][gain_cols].copy()
        gain_out.columns = (["year", "economy"] +
                            [f"gain_{m}_ppt" for m in metrics])
        gain_out.to_excel(writer, sheet_name="Integration Gains (ppt)", index=False)

        # Sheet 2: 2024 snapshot per country
        cdf = country_gain_2024(gain_df)
        snap_cols = ["name", "economy",
                     "gain_gvc_total", "gain_gvc_back", "gain_gvc_fwd",
                     "gain_gross_exp"]
        cdf[snap_cols].to_excel(writer, sheet_name="2024 Country Snapshot", index=False)

        # Sheet 3: Bloc-average gain time series
        bloc_rows = []
        for yr in FORECAST_YEARS:
            row = {"year": yr}
            sub = gain_df[(gain_df["year"] == yr) &
                          (gain_df["economy"].isin(BLOC_CODES))]
            for m in metrics:
                row[f"gain_{m}_ppt"] = sub[f"gain_{m}"].mean()
            bloc_rows.append(row)
        pd.DataFrame(bloc_rows).to_excel(
            writer, sheet_name="Bloc Avg Gain Series", index=False)

        # Sheet 4: Raw BAU levels (reference)
        bau.to_excel(writer, sheet_name="BAU Levels", index=False)

        # Sheet 5: Raw Integrated levels (reference)
        integ.to_excel(writer, sheet_name="Integrated Levels", index=False)

        # Sheet 6: Actual baseline for context
        baseline.to_excel(writer, sheet_name="Actual Baseline", index=False)

        # Sheet 7: CF 2021
        cf2021.to_excel(writer, sheet_name="Counterfactual 2021", index=False)

    print(f"  Saved → {path}")
    return path


# ── main ──────────────────────────────────────────────────────────────────────

def run(out_dir: str = None):
    od = out_dir or OUT_DIR
    os.makedirs(od, exist_ok=True)
    print("\n[Phase 5] Generating gain-based visualizations …")

    baseline, bau, integ, cf2021 = load_data(od)
    gain_df = compute_gain(bau, integ)

    p1 = plot_gain_series(gain_df, od)
    p2 = plot_country_gain_2024(gain_df, od)
    p3 = plot_gain_heatmap(gain_df, od)
    p4 = plot_fwd_vs_bwd_scatter(gain_df, bau, od)
    p5 = plot_dashboard(gain_df, baseline, bau, integ, od)
    p6 = plot_event_study(baseline, bau, integ, od)
    ex = build_summary_excel(gain_df, baseline, bau, integ, cf2021, od)

    print(f"\n✅  Phase 5 complete → {od}")
    return [p1, p2, p3, p4, p5, p6, ex]


if __name__ == "__main__":
    run()
