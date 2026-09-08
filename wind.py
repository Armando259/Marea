"""
optuna_vjetar.py
=================
Optuna optimizacija hiperparametara SAMO za target 'wind_speed_ms'
(brzina vjetra), zasebno za svaki horizont (1, 3, 6, 9, 12 sati).

Testira LightGBM i XGBoost za svaki horizont i sprema rezultate u
CSV, spreman za korištenje u train_all.py preko model_utils.py.

ISPRAVAK DATA LEAKAGE BUGA UKLJUČEN:
prepare_X() izbacuje SVE stupce koji sadrže '_future' u imenu
(npr. 'wind_speed_ms_future_1h'), ne samo one koji na to završavaju.
Stara verzija (.endswith('_future')) je propuštala target da uđe kao
feature, što je davalo lažno savršene rezultate na treningu, ali
potpuno pogrešne predikcije na živim podacima.

Pokretanje:  python optuna_vjetar.py

Izlaz:  optuna_vjetar_svi_horizonti.csv
"""

import gc
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Konfiguracija
# ---------------------------------------------------------------------------
TARGET = 'wind_speed_ms'
LAG_VARS = ['t2m_c', 'msl_hpa', 'wind_speed_ms', 'tcc', 'cape']

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.10
BUFFER_STUPNJEVI = 0.3
CSV_PATH = 'ERA5_SPOJENO_2015_2023.csv'
OUTPUT_CSV = 'optuna_vjetar_svi_horizonti.csv'

LAT_PULA, LON_PULA = 44.8666, 13.8496
LAT_RIJEKA, LON_RIJEKA = 45.3271, 14.4422

HORIZONS = [1, 3, 6, 9, 12]

OPTUNA_TRIALS_BY_HORIZON = {
    1: 5,
    3: 8,
    6: 12,
    9: 15,
    12: 20,
}

OPTUNA_MAX_ROWS = None
MAX_TRAIN_ROWS = None


# ---------------------------------------------------------------------------
# 1. Učitavanje i priprema podataka
# ---------------------------------------------------------------------------
print('Učitavam podatke...')
df = pd.read_csv(CSV_PATH)
df['valid_time'] = pd.to_datetime(df['valid_time'])
df = df.sort_values(['latitude', 'longitude', 'valid_time']).reset_index(drop=True)

print(f'Ukupno redaka: {len(df):,}')
print(f'Period: {df["valid_time"].min()} do {df["valid_time"].max()}')


# ---------------------------------------------------------------------------
# 2. Izdvajanje Pula/Rijeka i trening skupa
# ---------------------------------------------------------------------------
locations = df[['latitude', 'longitude']].drop_duplicates().copy()
locations['dist_pula'] = np.sqrt((locations['latitude'] - LAT_PULA) ** 2 + (locations['longitude'] - LON_PULA) ** 2)
locations['dist_rijeka'] = np.sqrt((locations['latitude'] - LAT_RIJEKA) ** 2 + (locations['longitude'] - LON_RIJEKA) ** 2)

pula_grid = locations.loc[locations['dist_pula'].idxmin(), ['latitude', 'longitude']]
rijeka_grid = locations.loc[locations['dist_rijeka'].idxmin(), ['latitude', 'longitude']]

df['dist_pula'] = np.sqrt((df['latitude'] - LAT_PULA) ** 2 + (df['longitude'] - LON_PULA) ** 2)
df['dist_rijeka'] = np.sqrt((df['latitude'] - LAT_RIJEKA) ** 2 + (df['longitude'] - LON_RIJEKA) ** 2)
je_pula = (df['latitude'] == pula_grid['latitude']) & (df['longitude'] == pula_grid['longitude'])
je_rijeka = (df['latitude'] == rijeka_grid['latitude']) & (df['longitude'] == rijeka_grid['longitude'])
blizu = (df['dist_pula'] < BUFFER_STUPNJEVI) | (df['dist_rijeka'] < BUFFER_STUPNJEVI)

df_train = df.loc[~je_pula & ~je_rijeka & ~blizu].copy()

if MAX_TRAIN_ROWS is not None and len(df_train) > MAX_TRAIN_ROWS:
    df_train = df_train.sort_values(['valid_time', 'latitude', 'longitude']).iloc[:MAX_TRAIN_ROWS].copy()

print(f'Trening: {len(df_train):,}')
del locations, df
gc.collect()


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def add_features(data):
    data = data.sort_values(['latitude', 'longitude', 'valid_time']).copy()
    groups = data.groupby(['latitude', 'longitude'])

    for var in LAG_VARS:
        if var not in data.columns:
            continue
        data[f'{var}_lag3h'] = groups[var].shift(3)
        data[f'{var}_lag6h'] = groups[var].shift(6)
        data[f'{var}_trend3h'] = data[var] - data[f'{var}_lag3h']
        data[f'{var}_rolling6h_mean'] = groups[var].transform(lambda s: s.rolling(6, min_periods=1).mean())

    if 'wind_dir_deg' in data.columns:
        data['wind_dir_sin'] = np.sin(np.radians(data['wind_dir_deg']))
        data['wind_dir_cos'] = np.cos(np.radians(data['wind_dir_deg']))

    hour = data['valid_time'].dt.hour
    day = data['valid_time'].dt.dayofyear
    data['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    data['day_sin'] = np.sin(2 * np.pi * day / 365.25)
    data['day_cos'] = np.cos(2 * np.pi * day / 365.25)

    if {'t2m_c', 'd2m_c'}.issubset(data.columns):
        data['temp_dewpoint_diff'] = data['t2m_c'] - data['d2m_c']
    if 'msl_hpa' in data.columns:
        data['pressure_gradient'] = groups['msl_hpa'].diff()

    return data


def add_future_target(data, target, horizon):
    """Doda JEDAN target stupac za zadani horizont, npr. 'wind_speed_ms_future_3h'."""
    data = data.sort_values(['latitude', 'longitude', 'valid_time']).copy()
    groups = data.groupby(['latitude', 'longitude'])
    if target in data.columns:
        data[f'{target}_future_{horizon}h'] = groups[target].shift(-horizon)
    return data


df_train = add_features(df_train)
print('Broj stupaca nakon feature engineeringa:', df_train.shape[1])


# ---------------------------------------------------------------------------
# 4. Priprema featurea (ISPRAVLJENA verzija — bez data leakage bug-a)
# ---------------------------------------------------------------------------
DROP_COLUMNS = ['valid_time', 'latitude', 'longitude', 'dist_pula', 'dist_rijeka', 'hour', 'day']


def prepare_X(data, feature_names=None, medians=None):
    drop_cols = [c for c in DROP_COLUMNS if c in data.columns]

    # ISPRAVAK: '_future' in c (ne .endswith('_future')) — hvata sve
    # oblike poput 'wind_speed_ms_future_1h', 'wind_speed_ms_future_12h'.
    # Stara verzija je propuštala target da uđe kao feature (data leakage).
    drop_cols += [c for c in data.columns if '_future' in c]

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    selected_cols = [c for c in numeric_cols if c not in drop_cols]

    if feature_names is not None:
        selected_cols = [c for c in feature_names if c in data.columns]

    X = data.loc[:, selected_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    if medians is None:
        medians = X.median()

    X = X.fillna(medians)

    if feature_names is not None:
        X = X.reindex(columns=feature_names, fill_value=0)

    X = X.astype(np.float32)
    medians = medians.astype(np.float32)

    return X, medians


# ---------------------------------------------------------------------------
# 5. Priprema train/val splita za dani horizont
# ---------------------------------------------------------------------------
def get_train_val_split(horizon):
    target_col = f'{TARGET}_future_{horizon}h'
    data_h = add_future_target(df_train, TARGET, horizon)

    valid_idx = data_h[target_col].notna()
    data_valid = data_h.loc[valid_idx]

    if OPTUNA_MAX_ROWS is not None and len(data_valid) > OPTUNA_MAX_ROWS:
        data_valid = (
            data_valid
            .sort_values(['valid_time', 'latitude', 'longitude'])
            .iloc[:OPTUNA_MAX_ROWS]
        )

    y = data_valid[target_col].astype(np.float32)
    X, _ = prepare_X(data_valid)

    # Sanity check: target ne smije biti među featurima
    assert target_col not in X.columns, f'DATA LEAKAGE: {target_col} je u featurima!'

    cutoff = int(len(X) * (1 - VALIDATION_FRACTION))
    X_fit, X_val = X.iloc[:cutoff], X.iloc[cutoff:]
    y_fit, y_val = y.iloc[:cutoff], y.iloc[cutoff:]

    return X_fit, X_val, y_fit, y_val, len(data_valid)


# ---------------------------------------------------------------------------
# 6. Optuna objective funkcije
# ---------------------------------------------------------------------------
def objective_lightgbm(trial, horizon):
    X_fit, X_val, y_fit, y_val, n_valid = get_train_val_split(horizon)

    params = {
        'objective': 'regression',
        'metric': 'l1',
        'verbosity': -1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 4, 14),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        eval_names=['valid_0'],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    pred = model.predict(X_val)
    return mean_absolute_error(y_val, pred)


def objective_xgboost(trial, horizon):
    X_fit, X_val, y_fit, y_val, n_valid = get_train_val_split(horizon)

    params = {
        'objective': 'reg:absoluteerror',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0,
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-2, 20.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'early_stopping_rounds': 30,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)
    return mean_absolute_error(y_val, pred)


OBJECTIVES = {
    'LightGBM': objective_lightgbm,
    'XGBoost': objective_xgboost,
}


# ---------------------------------------------------------------------------
# 7. Glavna petlja — Optuna za svaki horizont x oba modela
# ---------------------------------------------------------------------------
svi_rezultati = []

print(f'\n{"=" * 60}')
print(f'TARGET: {TARGET}')
print(f'{"=" * 60}')

for horizon in HORIZONS:
    target_col = f'{TARGET}_future_{horizon}h'
    data_h = add_future_target(df_train, TARGET, horizon)

    if target_col not in data_h.columns:
        print(f'  [{horizon}h] target ne postoji, preskačem.')
        continue

    n_valid = data_h[target_col].notna().sum()
    if n_valid < 1000:
        print(f'  [{horizon}h] premalo podataka ({n_valid}), preskačem.')
        continue

    n_trials_for_horizon = OPTUNA_TRIALS_BY_HORIZON.get(horizon, 10)
    print(f'\n  --- Horizont: {horizon}h ({n_valid:,} valjanih redaka, {n_trials_for_horizon} trials) ---')

    for model_name, objective_fn in OBJECTIVES.items():
        study = optuna.create_study(
            direction='minimize',
            study_name=f'{model_name}_{TARGET}_{horizon}h',
            sampler=TPESampler(seed=RANDOM_STATE),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=20),
        )

        study.optimize(
            lambda trial, h=horizon: objective_fn(trial, h),
            n_trials=n_trials_for_horizon,
            show_progress_bar=False,
        )

        print(f'    {model_name}: najbolji MAE = {study.best_value:.4f}')

        svi_rezultati.append({
            'model': model_name,
            'target': TARGET,
            'horizon_hours': horizon,
            'best_MAE': study.best_value,
            'n_trials': len(study.trials),
            **study.best_params,
        })

        gc.collect()

print('\n\nOptuna pretraga završena za sve horizonte.')


# ---------------------------------------------------------------------------
# 8. Spremanje rezultata
# ---------------------------------------------------------------------------
optuna_df = pd.DataFrame(svi_rezultati)
optuna_df = optuna_df.sort_values(['horizon_hours', 'model']).reset_index(drop=True)
optuna_df.to_csv(OUTPUT_CSV, index=False)

print(f'\nSpremljeno u: {OUTPUT_CSV}')
print(optuna_df.to_string())

print('\n--- Najbolji model po horizontu ---')
najbolji = optuna_df.loc[optuna_df.groupby('horizon_hours')['best_MAE'].idxmin()]
print(najbolji[['horizon_hours', 'model', 'best_MAE']].to_string(index=False))