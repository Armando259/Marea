# model_utils.py
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

XGB_CSV = BASE / "optuna_xgboost_svi_horizonti.csv"
LGB_CSV = BASE / "optuna_lightgbm_svi_horizonti.csv"

xgb_df = pd.read_csv(XGB_CSV)
lgb_df = pd.read_csv(LGB_CSV)

META_COLS = ["model", "target", "horizon_hours", "best_MAE", "n_trials"]

def get_config(df, target, horizon):
    row = df[
        (df["target"] == target) &
        (df["horizon_hours"] == horizon)
    ].iloc[0]
    params = row.drop(META_COLS).dropna().to_dict()
    return params

def best_model_for(target, horizon):
    x_row = xgb_df[
        (xgb_df["target"] == target) &
        (xgb_df["horizon_hours"] == horizon)
    ]
    l_row = lgb_df[
        (lgb_df["target"] == target) &
        (lgb_df["horizon_hours"] == horizon)
    ]

    if x_row.empty and l_row.empty:
        raise ValueError(f"Nema rezultata za {target}, {horizon}h")

    if x_row.empty:
        return "LightGBM", get_config(lgb_df, target, horizon)
    if l_row.empty:
        return "XGBoost", get_config(xgb_df, target, horizon)

    mae_x = x_row["best_MAE"].iloc[0]
    mae_l = l_row["best_MAE"].iloc[0]

    if mae_x <= mae_l:
        return "XGBoost", get_config(xgb_df, target, horizon)
    else:
        return "LightGBM", get_config(lgb_df, target, horizon)