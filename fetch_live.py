"""
fetch_live.py
=============
Povlači zadnjih ~48h opažanja s Open-Meteo (besplatno, bez API key-a) za
Pulu i Rijeku i pretvara ih u DataFrame s istim nazivima stupaca kao ERA5
podaci u notebooku, tako da add_features() radi bez izmjena.
"""

import requests
import pandas as pd

LOCATIONS = {
    'pula': (44.8666, 13.8496),
    'rijeka': (45.3271, 14.4422),
}

PAST_HOURS = 48  

WEATHER_URL = 'https://api.open-meteo.com/v1/forecast'
MARINE_URL = 'https://marine-api.open-meteo.com/v1/marine'

WEATHER_HOURLY = [
    'temperature_2m', 'dew_point_2m', 'pressure_msl', 'cloud_cover',
    'cape', 'wind_speed_10m', 'wind_direction_10m', 'boundary_layer_height',
]

MARINE_HOURLY = [
    'wave_height', 'wave_period', 'wave_direction', 'sea_surface_temperature',
]


def _fetch_json(url, params):
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_location(name: str, lat: float, lon: float) -> pd.DataFrame:
    weather = _fetch_json(WEATHER_URL, {
        'latitude': lat, 'longitude': lon,
        'hourly': ','.join(WEATHER_HOURLY),
        'past_hours': PAST_HOURS,
        'forecast_hours': 1,
        'timezone': 'UTC',
        'wind_speed_unit': 'ms',
    })

    marine = _fetch_json(MARINE_URL, {
        'latitude': lat, 'longitude': lon,
        'hourly': ','.join(MARINE_HOURLY),
        'past_hours': PAST_HOURS,
        'forecast_hours': 1,
        'timezone': 'UTC',
    })

    df_w = pd.DataFrame(weather['hourly'])
    df_m = pd.DataFrame(marine['hourly'])
    df = df_w.merge(df_m, on='time', how='left')

    df = df.rename(columns={
        'time': 'valid_time',
        'temperature_2m': 't2m_c',
        'dew_point_2m': 'd2m_c',
        'pressure_msl': 'msl_hpa',
        'cloud_cover': 'tcc',
        'cape': 'cape',
        'wind_speed_10m': 'wind_speed_ms',
        'wind_direction_10m': 'wind_dir_deg',
        'boundary_layer_height': 'blh',
        'wave_height': 'swh',
        'wave_period': 'mwp',
        'wave_direction': 'mwd',
        'sea_surface_temperature': 'sst_c',
    })

    df['valid_time'] = pd.to_datetime(df['valid_time'])
    df['latitude'] = lat
    df['longitude'] = lon

    if 'tcc' in df.columns:
        df['tcc'] = df['tcc'] / 100.0

    return df


def fetch_all() -> dict:
    """Vraća {'pula': df, 'rijeka': df} spremne za add_features()."""
    out = {}
    for name, (lat, lon) in LOCATIONS.items():
        out[name] = fetch_location(name, lat, lon)
    return out


if __name__ == '__main__':
    data = fetch_all()
    for name, df in data.items():
        print(f'--- {name} ---')
        print(df.tail(3))
