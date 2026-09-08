"""
app.py
======
Lokalni FastAPI server. Predikcije se računaju u pozadini svakih sat
vremena i drže u memoriji; endpoint samo vraća zadnji rezultat (brzo,
ne računa live na svaki request).

Pokretanje:
    pip install fastapi uvicorn apscheduler
    uvicorn app:app --reload --port 8000
     
    python -m http.server 5500
Frontend: otvori frontend/index.html u browseru ( localhost:5500).
"""

from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predict import predict_all

STATE = {'data': None, 'error': None}


def refresh():
    try:
        STATE['data'] = predict_all()
        STATE['error'] = None
        print('Predikcije osvježene.')
    except Exception as e:  # noqa: BLE001
        STATE['error'] = str(e)
        print(f'Greška pri osvježavanju: {e}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    refresh()  # izračunaj odmah pri pokretanju
    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh, 'interval', hours=1)
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title='Meteo prognoza', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/forecast')
def forecast_all():
    if STATE['data'] is None:
        raise HTTPException(status_code=503, detail=STATE['error'] or 'Podaci se još pripremaju, pokušaj za par sekundi.')
    return STATE['data']


@app.get('/forecast/{location}')
def forecast_location(location: str):
    if STATE['data'] is None:
        raise HTTPException(status_code=503, detail=STATE['error'] or 'Podaci se još pripremaju, pokušaj za par sekundi.')
    if location not in STATE['data']:
        raise HTTPException(status_code=404, detail=f'Nepoznata lokacija: {location}')
    return STATE['data'][location]


@app.post('/refresh')
def manual_refresh():
    refresh()
    if STATE['error']:
        raise HTTPException(status_code=500, detail=STATE['error'])
    return {'status': 'ok'}
