# Meteo prognoza — lokalni sustav

## Struktura
```
features.py        - dijeljena feature engineering logika (train + live)
train/train_all.py - trenira model za svaki (target, horizont) i sprema u models/
fetch_live.py       - povlači zadnjih 48h opažanja s Open-Meteo (weather + marine API)
predict.py          - učita modele, predvidi za trenirane horizonte, interpolira 1-12h
app.py              - FastAPI server, osvježava predikciju svaki sat
frontend/index.html - dashboard s grafovima (Chart.js)
```

## Postavljanje

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Trening (jednom, ili kad se model mijenja)

Stavi `ERA5_SPOJENO_2015_2023.csv` u root ovog projekta, zatim:

```bash
python train/train_all.py
```

Ovo trenira po 5 modela (horizonti 1,3,6,9,12h) za svih 10 targeta =
50 modela, sprema ih u `models/`. Traje ovisno o veličini CSV-a i broju
jezgri — za ExtraTrees/RandomForest s `n_jobs=-1` očekuj od par minuta do
sat vremena. Ako imaš previše RAM pritiska, postavi `MAX_TRAIN_ROWS` u
`train_all.py`.

Za CatBoost promijeni `MODEL_NAME = 'CatBoost'` na vrhu skripte i pokreni
ponovno (spremit će se odvojeni fajlovi, ne pregazi ExtraTrees modele).

## 2. Pokretanje servera

```bash
uvicorn app:app --reload --port 8000
```

Pri pokretanju odmah izračuna prvu prognozu (poziva Open-Meteo), a zatim
je osvježava svaki sat u pozadini (APScheduler). Endpoints:

- `GET /forecast` — obje lokacije
- `GET /forecast/pula`, `GET /forecast/rijeka`
- `POST /refresh` — ručno prisili osvježavanje

## 3. Frontend

Samo otvori `frontend/index.html` u browseru (dupli klik ili
`open frontend/index.html`). Fetcha `http://localhost:8000` pa server
mora raditi.

## Napomene / poznata ograničenja

- **Jedinice/nazivi**: Open-Meteo i ERA5 nisu 100% identični u definicijama
  (npr. cloud_cover, boundary_layer_height). `fetch_live.py` radi
  najbolje moguće mapiranje i konverziju (npr. tcc iz % u 0-1), ali ako
  primijetiš da su predikcije "čudne", prvo provjeri tu funkciju — model
  je osjetljiv na to da live podaci izgledaju kao ERA5.
- **MWD (smjer valova)** je kutna varijabla — interpolacija linearnom
  metodom (`np.interp` u predict.py) može dati čudne rezultate blizu
  0°/360° granice. Za sad je ostavljeno jednostavno; javi ako treba
  kutna interpolacija.
- Graf 1-12h je **interpolacija** između 5 stvarno treniranih horizonata
  (1,3,6,9,12h), ne 12 zasebno treniranih modela — dovoljno za vizualni
  dojam, ali sate između npr. 6h i 9h ne treba čitati kao potpuno
  neovisnu predikciju.
