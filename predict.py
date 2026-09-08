"""
predict.py
==========
Učita spremljene modele iz models2/ (trenirane s train_all.py, gdje je
za svaki (target, horizont) odabran najbolji tip modela — LightGBM ili
XGBoost — preko model_utils.best_model_for()). Uzme live podatke,
predvidi za svaki trenirani horizont i interpolira u graf 1..12h.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from features import add_features, prepare_X
from fetch_live import fetch_all

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models2")

TARGETS = [
    "t2m_c",
    "wind_speed_ms",
    "swh",
    "mwp",
    "mwd",
    "sst_c",
    "msl_hpa",
    "tcc",
    "cape",
    "blh",
]

HORIZONS = [1, 3, 6, 9, 12]
GRAPH_HOURS = list(range(1, 13))

ORDER_BASE = [
    "t2m_c",
    "wind_speed_ms",
    "wind_dir_deg",
    "mwd",
    "swh",
    "mwp",
    "sst_c",
    "msl_hpa",
    "tcc",
    "cape",
    "blh",
]

_model_cache = {}
_model_selection = None


def _get_model_type(target, horizon):
    global _model_selection
    if _model_selection is None:
        with open(os.path.join(MODELS_DIR, "model_selection.json")) as f:
            _model_selection = json.load(f)
    return _model_selection[f"{target}_{horizon}h"]


def _load(target, horizon):
    key = (target, horizon)
    if key in _model_cache:
        return _model_cache[key]

    model_type = _get_model_type(target, horizon)
    stem = os.path.join(MODELS_DIR, f"{model_type}_{target}_{horizon}h")
    model_path = f"{stem}.joblib"
    if not os.path.exists(model_path):
        _model_cache[key] = None
        return None

    model = joblib.load(model_path)
    with open(f"{stem}_features.json") as f:
        feature_names = json.load(f)
    with open(f"{stem}_medians.json") as f:
        medians = pd.Series(json.load(f))

    _model_cache[key] = (model, feature_names, medians)
    return _model_cache[key]


def predict_location(name: str, raw_df: pd.DataFrame, debug: bool = False) -> dict:
    # 1) Feature engineering
    df = add_features(raw_df)
    latest = df.iloc[[-1]]
    latest_time = raw_df["valid_time"].iloc[-1]

    if debug:
        print(f"\n=== DEBUG {name} — sirovi zadnji redak ===")
        print(
            raw_df.tail(3)[
                ["valid_time", "t2m_c", "wind_speed_ms",
                 "msl_hpa", "tcc", "cape", "blh"]
            ].to_string()
        )

    forecast = {target: {} for target in TARGETS}

    for target in TARGETS:
        for horizon in HORIZONS:
            loaded = _load(target, horizon)
            if loaded is None:
                continue
            model, feature_names, medians = loaded
            X, _ = prepare_X(latest, feature_names, medians)
            pred = float(model.predict(X)[0])
            forecast[target][horizon] = pred

    series: dict[str, list[dict]] = {}
    for target in TARGETS:
        known = forecast[target]
        if not known:
            continue
        xs = sorted(known.keys())  
        ys = [known[h] for h in xs]
        interp_vals = np.interp(GRAPH_HOURS, xs, ys)
        series[target] = [
            {
                "hour": h,
                "time": str(latest_time + pd.Timedelta(hours=h)),
                "value": round(float(v), 3),
            }
            for h, v in zip(GRAPH_HOURS, interp_vals)
        ]

    if "wind_dir_deg" in raw_df.columns:
        dir_latest = float(raw_df["wind_dir_deg"].iloc[-1])
        series["wind_dir_deg"] = [
            {
                "hour": h,
                "time": str(latest_time + pd.Timedelta(hours=h)),
                "value": dir_latest,
            }
            for h in GRAPH_HOURS
        ]

    order = [key for key in ORDER_BASE if key in series]

    return {
        "location": name,
        "base_time": str(latest_time),
        "series": series,
        "order": order,
    }


def predict_all() -> dict:
    live = fetch_all()
    return {name: predict_location(name, df) for name, df in live.items()}


if __name__ == "__main__":
    import pprint

    live = fetch_all()
    for name, df in live.items():
        result = predict_location(name, df, debug=True)
        pprint.pprint(result)