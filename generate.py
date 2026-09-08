"""
generate_selection_json.py
===========================
Brzo (bez treninga) generira models2/model_selection.json na temelju
model_utils.best_model_for() — koristi se ako je trening već proveden
ali selection.json fajl nedostaje.

Pokretanje:  python generate_selection_json.py
"""

import json
import os

from model_utils import best_model_for

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models2')
TARGETS = ['t2m_c', 'wind_speed_ms', 'swh', 'mwp', 'mwd', 'sst_c', 'msl_hpa', 'tcc', 'cape', 'blh']
HORIZONS = [1, 3, 6, 9, 12]

selection = {}
for target in TARGETS:
    for horizon in HORIZONS:
        model_type, _ = best_model_for(target, horizon)
        selection[f'{target}_{horizon}h'] = model_type
        print(f'{target}_{horizon}h -> {model_type}')

out_path = os.path.join(MODELS_DIR, 'model_selection.json')
with open(out_path, 'w') as f:
    json.dump(selection, f, indent=2)

print(f'\nSpremljeno: {out_path}')