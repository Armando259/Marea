# train_all.py
"""
train_all.py
============
Trenira model za svaki (target, horizont) par i sprema:
  models1/{model_name}_{target}_{horizon}h.joblib
  models1/{model_name}_{target}_{horizon}h_medians.json
  models1/{model_name}_{target}_{horizon}h_features.json
  models1/rezultati_svi_horizonti.csv

Pokretanje:  python train_all.py

Napomena: Ovaj script koristi SAMO RIJEKU (bez Pule).
"""

import gc
import json
import os
import sys
from model_utils import best_model_for
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Uvoz iz features.py
from features import add_features, add_future_target, prepare_X  # noqa: E402


# ---------------------------------------------------------------------------
# Konfiguracija
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
VALIDATION_FRACTION = 0.10
BUFFER_STUPNJEVI = 0.3

CSV_PATH = os.path.join(os.path.dirname(__file__), 'ERA5_SPOJENO_2015_2023.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models2')

MAX_TRAIN_ROWS = None

LAT_RIJEKA, LON_RIJEKA = 45.3271, 14.4422

TARGETS = [
    't2m_c', 'wind_speed_ms', 'swh', 'mwp', 'mwd',
    'sst_c', 'msl_hpa', 'tcc', 'cape', 'blh'
]

HORIZONS = [1, 3, 6, 9, 12]

os.makedirs(MODELS_DIR, exist_ok=True)


def make_model(target, horizon):
    model_name, params = best_model_for(target, horizon)

    if model_name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mae",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )

    if model_name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            objective="mae",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )

    raise ValueError(f"Nepoznat model: {model_name}")


def load_and_split():
    df = pd.read_csv(CSV_PATH)
    df['valid_time'] = pd.to_datetime(df['valid_time'])
    df = df.sort_values(['latitude', 'longitude', 'valid_time']).reset_index(drop=True)

    locations = df[['latitude', 'longitude']].drop_duplicates().copy()
    locations['dist_rijeka'] = np.sqrt(
        (locations['latitude'] - LAT_RIJEKA) ** 2 +
        (locations['longitude'] - LON_RIJEKA) ** 2
    )

    rijeka_grid = locations.loc[locations['dist_rijeka'].idxmin(), ['latitude', 'longitude']]

    df['dist_rijeka'] = np.sqrt(
        (df['latitude'] - LAT_RIJEKA) ** 2 +
        (df['longitude'] - LON_RIJEKA) ** 2
    )
    je_rijeka = (df['latitude'] == rijeka_grid['latitude']) & (df['longitude'] == rijeka_grid['longitude'])
    blizu = df['dist_rijeka'] < BUFFER_STUPNJEVI

    df_rijeka = df.loc[je_rijeka].copy()
    df_train = df.loc[~je_rijeka & ~blizu].copy()

    if MAX_TRAIN_ROWS is not None and len(df_train) > MAX_TRAIN_ROWS:
        df_train = df_train.sort_values(['valid_time', 'latitude', 'longitude']).iloc[:MAX_TRAIN_ROWS].copy()

    print(f'Rijeka: {len(df_rijeka):,} | Trening: {len(df_train):,}')

    df_train = add_features(df_train)
    df_rijeka = add_features(df_rijeka)

    return df_train, df_rijeka


def train_one(target, horizon, train_data, test_dict):
    model_name, _ = best_model_for(target, horizon)  # samo za logging
    target_col = f"{target}_future_{horizon}h"

    train_data = add_future_target(train_data, target, horizon)

    if target_col not in train_data.columns:
        print(f"{target}/{horizon}h: target ne postoji")
        return None

    valid_idx = train_data[target_col].notna()
    n_valid = int(valid_idx.sum())
    if n_valid < 1000:
        print(f"{target}/{horizon}h: preskačem, samo {n_valid} valjanih redaka")
        return None

    target_data = train_data.loc[valid_idx]
    y = target_data[target_col].astype("float32")
    X, medians = prepare_X(target_data)
    feature_names = X.columns.tolist()

    cutoff = int(len(X) * (1 - VALIDATION_FRACTION))
    X_fit, X_val = X.iloc[:cutoff], X.iloc[cutoff:]
    y_fit, y_val = y.iloc[:cutoff], y.iloc[cutoff:]

    print(f"{model_name}/{target}/{horizon}h: treniranje na {len(X_fit):,} redaka, {len(feature_names)} featurea")

    model = make_model(target, horizon)
    model.fit(X_fit, y_fit)
    pred_val = model.predict(X_val)

    mae = float(mean_absolute_error(y_val, pred_val))
    rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
    r2 = float(r2_score(y_val, pred_val))

    # Spremi model i metapodatke
    stem = os.path.join(MODELS_DIR, f"{model_name}_{target}_{horizon}h")
    joblib.dump(model, f"{stem}.joblib")
    with open(f"{stem}_features.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f)
    with open(f"{stem}_medians.json", "w", encoding="utf-8") as f:
        json.dump(medians.to_dict(), f)

    return {
        "model": model_name,
        "target": target,
        "horizon_hours": horizon,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "n_train": len(X_fit),
        "n_val": len(X_val),
        "n_features": len(feature_names),
    }


def main():
    df_train, df_rijeka = load_and_split()
    test_dict = {"rijeka": df_rijeka}

    all_results = []
    for horizon in HORIZONS:
        for target in TARGETS:
            r = train_one(target, horizon, df_train, test_dict)
            if r is not None:
                all_results.append(r)

    if not all_results:
        print("Nema rezultata za spremanje (svi target/horizon parovi preskočeni).")
        return

    results_df = pd.DataFrame(all_results)

    out_csv = os.path.join(
        os.path.dirname(__file__),
        "rezultati_svi_horizonti.csv"
    )
    results_df.to_csv(out_csv, index=False)
    print(f"\nGotovo. Rezultati spremljeni u {out_csv}")
    print(results_df)


if __name__ == '__main__':
    main()