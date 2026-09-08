"""
merge_wind_into_optuna_csvs.py
===============================
Ubacuje wind_speed_ms rezultate iz optuna_vjetar_svi_horizonti.csv
u optuna_xgboost_svi_horizonti.csv i optuna_lightgbm_svi_horizonti.csv,
da bi ih best_model_for() / generate_selection_json.py mogli pronaći.

Pokretanje: python merge_wind_into_optuna_csvs.py
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

VJETAR_CSV = BASE / "optuna_vjetar_svi_horizonti.csv"
XGB_CSV = BASE / "optuna_xgboost_svi_horizonti.csv"
LGB_CSV = BASE / "optuna_lightgbm_svi_horizonti.csv"

vjetar_df = pd.read_csv(VJETAR_CSV)

vjetar_df = vjetar_df[vjetar_df["target"] == "wind_speed_ms"]

if vjetar_df.empty:
    raise ValueError("Nema redova s target='wind_speed_ms' u optuna_vjetar_svi_horizonti.csv!")

for model_type, target_csv in [("XGBoost", XGB_CSV), ("LightGBM", LGB_CSV)]:
    sub = vjetar_df[vjetar_df["model"] == model_type].copy()

    if sub.empty:
        print(f"Nema {model_type} redova za wind_speed_ms — preskačem.")
        continue

    existing = pd.read_csv(target_csv)

    before = len(existing)
    existing = existing[existing["target"] != "wind_speed_ms"]
    removed = before - len(existing)
    if removed:
        print(f"{target_csv.name}: uklonjeno {removed} starih wind_speed_ms redova.")

    all_cols = set(existing.columns) | set(sub.columns)
    for col in all_cols:
        if col not in existing.columns:
            existing[col] = pd.NA
        if col not in sub.columns:
            sub[col] = pd.NA

    sub = sub[existing.columns]

    merged = pd.concat([existing, sub], ignore_index=True)
    merged.to_csv(target_csv, index=False)

    print(f"{target_csv.name}: dodano {len(sub)} wind_speed_ms redova (ukupno sada {len(merged)}).")

print("\nGotovo. Sada pokreni generate_selection_json.py.")