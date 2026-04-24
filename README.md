# South Asia–ASEAN GVC Integration Study
## ADB MRIO 72-Economy Pipeline (2017–2024)

---

## Overview

This project implements a full 5-phase Global Value Chain (GVC) analysis using
the Asian Development Bank (ADB) Multi-Regional Input-Output (MRIO) tables
across 8 years (2017–2024). It simulates a trade integration shock between
**16 South Asian and ASEAN economies** and measures its cumulative effect on
GVC participation indices.

### Countries in the Integration Bloc
| Country     | ADB Code | Country     | ADB Code |
|-------------|----------|-------------|----------|
| Bangladesh  | BAN      | Brunei      | BRU      |
| Bhutan      | BHU      | Cambodia    | CAM      |
| India       | IND      | Indonesia   | INO      |
| Maldives    | MLD      | Laos        | LAO      |
| Nepal       | NEP      | Malaysia    | MAL      |
| Pakistan    | PAK      | Philippines | PHI      |
| Sri Lanka   | SRI      | Singapore   | SIN      |
|             |          | Vietnam     | VIE      |
|             |          | Thailand    | THA      |

> **Note:** Afghanistan and Myanmar are **not present** in the ADB MRIO dataset
> and are therefore excluded from the analysis.

---

## System Requirements

- Python 3.9+
- RAM: **16 GB minimum** (each MRIO file is ~2500 × 2925 cells; the
  Leontief inverse of the full 2555×2555 matrix requires significant memory)
- Disk: ~2 GB for all intermediate .npz files

---

## Installation

```bash
# 1. Clone or unzip the project folder
cd gvc_analysis

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Setup: Place Raw Data Files

Copy your 8 ADB MRIO Excel files into `data/raw/`:

```
data/raw/
├── ADB-MRIO72-2017-August_2025.xlsx
├── ADB-MRIO-2018_September_2024.xlsx
├── ADB-MRIO-2019_December_2024.xlsx
├── ADB-MRIO-2020_September_2024.xlsx
├── ADB-MRIO-2021_August_2024.xlsx
├── ADB-MRIO72-2022_July_2025.xlsx
├── ADB-MRIO72-2023_July_2025.xlsx
└── ADB-MRIO72-2024_August_2025__1_.xlsx
```

---

## Running the Full Pipeline

```bash
python run_all.py
```

This runs all 5 phases sequentially. Expected runtime: **30–90 minutes**
depending on your machine (dominated by Phase 1 Excel parsing and Phase 2/3
Leontief inversion of 2555×2555 matrices × 8 years).

### Optional flags

```bash
# Skip Phase 1 if you've already generated the .npz files:
python run_all.py --skip-phase1

# Custom paths:
python run_all.py --data-dir /path/to/xlsx --norm-dir /path/to/npz --out-dir /path/to/results
```

### Running phases individually

```bash
python phase1_preprocess.py    # ~20-40 min
python phase2_leontief.py      # ~20-40 min
python phase4_simulation.py    # ~5-10 min
python phase5_visualize.py     # ~1 min
```

---

## Output Files

All outputs land in `outputs/`:

| File | Description |
|------|-------------|
| `gvc_baseline.csv` | Actual KWW GVC indices for all 73 economies × 8 years |
| `forecast_bau.csv` | BAU linear projection for 2022–2024 |
| `forecast_integrated.csv` | Integrated path (BAU + 2021 shock delta) for 2022–2024 |
| `counterfactual_2021.csv` | Counterfactual 2021 KWW indices under integration shock |
| `leontief_cache.npz` | Cached A and L matrices for all years |
| `gvc_summary.xlsx` | Multi-sheet Excel summary workbook |
| `plot1_event_study.png` | **Main event study plot** — actual vs integrated divergence |
| `plot2_fwd_bwd.png` | Forward vs backward GVC decomposition |
| `plot3_country_heatmap.png` | Per-country integration gain heatmap |
| `plot4_cumulative_gap.png` | Cumulative BAU vs integrated gap (2022–2024) |
| `plot5_dashboard.png` | 6-panel summary dashboard |

---

## Methodology

### Phase 1 — Preprocessing
Raw ADB MRIO Excel tables are parsed. Each file has a 2555×2555 intermediate
transactions block (73 economies × 35 sectors) plus a 2555×365 final demand
block. All are harmonised to a canonical economy order and saved as compressed
NumPy arrays (`.npz`).

### Phase 2 — Technical Coefficients & Leontief Inverse
For each year:
- **A = Z · X̂⁻¹** (technical coefficient matrix)
- **L = (I − A)⁻¹** (Leontief multiplier matrix, solved via `numpy.linalg.solve`)

### Phase 3 — KWW Decomposition
The Koopman-Wang-Wei (2014) value-added decomposition is applied to produce:
- **DVA** — Domestic Value Added embodied in exports
- **FVA** — Foreign Value Added embodied in exports (backward participation)
- **DVX** — Domestic VA re-exported by trade partners (forward participation)
- **GVC Total** = (FVA + DVX) / Gross Exports

### Phase 4 — Simulation
**Step 5 (Shock):** Intra-bloc A-matrix coefficients are boosted by 15%
(configurable via `INTEGRATION_SHOCK` in `config.py`), simulating reduced
trade costs from a hypothetical 2021 integration agreement.

**Step 6 (Forecast):** A Business-As-Usual (BAU) linear trend is fitted on
2017–2021 actuals and projected to 2024. The integrated path adds the
2021 shock delta to the BAU projection as a floor estimate.

### Phase 5 — Visualization
Five plots and a summary Excel workbook are produced, including the key
event study chart showing the divergence between actual, BAU, and integrated
GVC trajectories with a vertical shock line at 2021.

---

## Customisation

Edit `config.py` to:
- **Change the shock size:** `INTEGRATION_SHOCK = 0.15` (15% intra-bloc boost)
- **Add/remove bloc members:** modify `TARGET_COUNTRIES`
- **Change years:** modify `YEARS` list (update `RAW_FILES` accordingly)

---

## Citation

Data source: Asian Development Bank (ADB) Multi-Regional Input-Output Tables.
https://mrio.adbx.online/

Methodology: Koopman, R., Wang, Z., & Wei, S.-J. (2014). Tracing Value-Added
and Double Counting in Gross Exports. *American Economic Review*, 104(2), 459–494.
