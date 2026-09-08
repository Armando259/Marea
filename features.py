"""
features.py
============
Jedan izvor istine za feature engineering. Koristi ga i trening skripta
(train_all.py) i live predikcija (predict.py) da model uvijek vidi
featurese izračunate na identičan način.
"""

import numpy as np
import pandas as pd

LAG_VARS = ['t2m_c', 'msl_hpa', 'wind_speed_ms', 'tcc', 'cape']

DROP_COLUMNS = ['valid_time', 'latitude', 'longitude', 'dist_pula', 'dist_rijeka', 'hour', 'day']


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: lagovi, rolling prosjeci, ciklički vremenski featurei."""
    data = data.sort_values(['latitude', 'longitude', 'valid_time']).copy()
    groups = data.groupby(['latitude', 'longitude'])

    for var in LAG_VARS:
        if var not in data.columns:
            continue
        data[f'{var}_lag3h'] = groups[var].shift(3)
        data[f'{var}_lag6h'] = groups[var].shift(6)
        data[f'{var}_trend3h'] = data[var] - data[f'{var}_lag3h']
        data[f'{var}_rolling6h_mean'] = groups[var].transform(
            lambda s: s.rolling(6, min_periods=1).mean()
        )

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


def prepare_X(data: pd.DataFrame, feature_names=None, medians=None):
    """Priprema X matricu: izbaci meta-stupce i SVE '_future' (target) stupce."""
    drop_cols = [c for c in DROP_COLUMNS if c in data.columns]
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


def add_future_target(data: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Doda samo JEDAN target za zadani horizont, npr. 't2m_c_future_3h'."""
    data = data.sort_values(['latitude', 'longitude', 'valid_time']).copy()
    groups = data.groupby(['latitude', 'longitude'])
    if target in data.columns:
        data[f'{target}_future_{horizon}h'] = groups[target].shift(-horizon)
    return data