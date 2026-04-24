"""
Configuration: file paths, economy codes, and parameters
"""
import os

# ── File paths ────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
NORM_DIR = os.path.join(os.path.dirname(__file__), "data", "normalized")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")

YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

RAW_FILES = {
    2017: "ADB-MRIO72-2017-August_2025.xlsx",
    2018: "ADB-MRIO-2018_September_2024.xlsx",
    2019: "ADB-MRIO-2019_December_2024.xlsx",
    2020: "ADB-MRIO-2020_September_2024.xlsx",
    2021: "ADB-MRIO-2021_August_2024.xlsx",
    2022: "ADB-MRIO72-2022_July_2025.xlsx",
    2023: "ADB-MRIO72-2023_July_2025.xlsx",
    2024: "ADB-MRIO72-2024_August_2025__1_.xlsx",
}

# ── Economy mapping ────────────────────────────────────────────────────────────
# Full list of 72 + RoW economies (excluding ToT aggregator)
ALL_ECONOMIES_ORDERED = [
    'AUS','AUT','BEL','BGR','BRA','CAN','SWI','PRC','CYP','CZE',
    'GER','DEN','SPA','EST','FIN','FRA','UKG','GRC','HRV','HUN',
    'INO','IND','IRE','ITA','JPN','KOR','LTU','LUX','LVA','MEX',
    'MLT','NET','NOR','POL','POR','ROM','RUS','SVK','SVN','SWE',
    'TUR','TAP','USA','BAN','MAL','PHI','THA','VIE','KAZ','MON',
    'SRI','PAK','FIJ','LAO','BRU','BHU','KGZ','CAM','MLD','NEP',
    'SIN','HKG','ARG','COL','ECU','ARM','GEO','EGY','KUW','SAU',
    'UAE','NZL','RoW',
]
N_ECONOMIES = len(ALL_ECONOMIES_ORDERED)   # 73 (incl. RoW)
N_SECTORS   = 35

SECTOR_NAMES = [
    "Agriculture, hunting, forestry, and fishing",
    "Mining and quarrying",
    "Food, beverages, and tobacco",
    "Textiles and textile products",
    "Leather, leather products, and footwear",
    "Wood and products of wood and cork",
    "Pulp, paper, paper products, printing, and publishing",
    "Coke, refined petroleum, and nuclear fuel",
    "Chemicals and chemical products",
    "Rubber and plastics",
    "Other nonmetallic minerals",
    "Basic metals and fabricated metal",
    "Machinery, nec",
    "Electrical and optical equipment",
    "Transport equipment",
    "Manufacturing, nec; recycling",
    "Electricity, gas, and water supply",
    "Construction",
    "Sale, maintenance, and repair of motor vehicles",
    "Wholesale trade and commission trade",
    "Retail trade",
    "Hotels and restaurants",
    "Inland transport",
    "Water transport",
    "Air transport",
    "Other supporting transport activities",
    "Post and telecommunications",
    "Financial intermediation",
    "Real estate activities",
    "Renting of M&Eq and other business activities",
    "Public administration and defense",
    "Education",
    "Health and social work",
    "Other community and personal services",
    "Private households with employed persons",
]

# ── Target integration bloc ────────────────────────────────────────────────────
# Afghanistan and Myanmar are NOT in the ADB MRIO dataset — excluded.
TARGET_COUNTRIES = {
    "Bangladesh":  "BAN",
    "Bhutan":      "BHU",
    "India":       "IND",
    "Maldives":    "MLD",
    "Nepal":       "NEP",
    "Pakistan":    "PAK",
    "Sri Lanka":   "SRI",
    "Brunei":      "BRU",
    "Cambodia":    "CAM",
    "Indonesia":   "INO",
    "Laos":        "LAO",
    "Malaysia":    "MAL",
    "Myanmar":     None,   # not in dataset
    "Philippines": "PHI",
    "Singapore":   "SIN",
    "Vietnam":     "VIE",
    "Thailand":    "THA",
    "Afghanistan": None,   # not in dataset
}

# Only codes that actually exist in the data
BLOC_CODES = [v for v in TARGET_COUNTRIES.values() if v is not None]
BLOC_NAMES = {v: k for k, v in TARGET_COUNTRIES.items() if v is not None}

# ── Simulation parameters ─────────────────────────────────────────────────────
# Fraction by which intra-bloc A-matrix coefficients are boosted (trade integration shock)
INTEGRATION_SHOCK = 0.15   # 15% increase in intra-bloc input coefficients

# Final-demand component columns per economy (F1..F5)
N_FD_COLS = 5
