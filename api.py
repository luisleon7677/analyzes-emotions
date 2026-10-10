"""One API process with an asynchronous queue and results in RAM."""
import hmac
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from audio_download import _filename_from_s3_url
from job_store import JobConflict, JobStore, QueueFull
from worker import InternalWorker

HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', '8000'))
MAX_QUEUE_SIZE = int(os.environ.get('MAX_QUEUE_SIZE', '20'))
JOB_TTL_SECONDS = float(os.environ.get('JOB_TTL_SECONDS', '86400'))
API_TOKEN = os.environ.get('API_TOKEN', '').strip()
DOWNLOAD_MAX_WAIT_SECONDS = float(os.environ.get('DOWNLOAD_MAX_WAIT_SECONDS', '3300'))
ALLOWED_SUFFIXES = {'.wav', '.flac', '.ogg', '.oga', '.mp3', '.mpeg', '.mpga', '.m4a', '.aac'}
_store = None
_worker = None

def get_store():
    global _store
    if _store is None:
        _store = JobStore(MAX_QUEUE_SIZE, JOB_TTL_SECONDS)
    return _store

@asynccontextmanager
async def lifespan(_app):
    global _store, _worker
    _store = JobStore(MAX_QUEUE_SIZE, JOB_TTL_SECONDS)
    worker = InternalWorker(_store)
    _worker = worker
    worker.start()
    try:
        yield
    finally:
        worker.stop()
        _worker = None
        _store = None

app = FastAPI(title='Analisis emocional de audio', lifespan=lifespan)

class AnalyzeRequest(BaseModel):
    s3_url: str = Field(..., min_length=1)
    fragmento_segundos: float = Field(3.0, ge=1.0, le=10.0)

def _require_api_key(x_api_key: str | None = Header(default=None)):
    if API_TOKEN and (not x_api_key or not hmac.compare_digest(x_api_key, API_TOKEN)):
        raise HTTPException(401, 'API key invalida.')

def download_deadline(url):
    if urlparse(url).scheme == 's3':
        return None
    deadline = time.time() + DOWNLOAD_MAX_WAIT_SECONDS
    query = parse_qs(urlparse(url).query)
    try:
        if 'X-Amz-Date' in query and 'X-Amz-Expires' in query:
            signed = datetime.strptime(query['X-Amz-Date'][0], '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
            deadline = min(deadline, signed.timestamp() + int(query['X-Amz-Expires'][0]) - 60)
        elif 'Expires' in query:
            deadline = min(deadline, float(query['Expires'][0]) - 60)
    except (ValueError, OverflowError) as exc:
        raise HTTPException(400, 'Fecha de expiracion de URL invalida.') from exc
    return deadline

@app.get('/health')
def health():
    counts = get_store().counts()
    return {'status': 'error' if _worker and _worker.error else ('ok' if _worker and _worker.model_ready else 'cargando'), 'cola': counts.get('en_cola', 0) + counts.get('iniciando', 0),
            'cola_maxima': MAX_QUEUE_SIZE, 'en_proceso': counts.get('procesando', 0)}

@app.get('/jobs/{job_id}', dependencies=[Depends(_require_api_key)])
def job_status(job_id: str):
    job = get_store().get(job_id)
    if job is None:
        raise HTTPException(404, 'Trabajo no encontrado o resultado caducado.')
    estado = 'procesando' if job['status'] == 'iniciando' else job['status']
    content = {'job_id': job['id'], 'estado': estado}
    if job['status'] == 'error':
        return JSONResponse(status_code=422, content={**content, 'error': job['error']})
    if job['status'] == 'listo':
        return JSONResponse(content={**content, **job['result']})
    return JSONResponse(status_code=202, content=content)

@app.post('/analyze', dependencies=[Depends(_require_api_key)])
def analyze(payload: AnalyzeRequest, idempotency_key: str | None = Header(default=None)):
    if _worker and _worker.error:
        raise HTTPException(503, 'Modelo no disponible. Revisa los logs y reinicia el servicio.')
    url = payload.s3_url.strip()
    parsed = urlparse(url)
    if not parsed.netloc or parsed.scheme not in {'s3', 'http', 'https'}:
        raise HTTPException(400, 's3_url debe incluir servidor o bucket.')
    filename = _filename_from_s3_url(url)
    if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(400, 'Formato de audio no admitido.')
    if idempotency_key is not None and (not idempotency_key.strip() or len(idempotency_key) > 255):
        raise HTTPException(400, 'Idempotency-Key debe tener entre 1 y 255 caracteres.')
    try:
        job = get_store().enqueue(url, filename, payload.fragmento_segundos,
                                  idempotency_key, download_deadline(url))
    except JobConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except QueueFull as exc:
        raise HTTPException(429, f'Cola llena ({MAX_QUEUE_SIZE}). Reintenta mas tarde.') from exc
    return JSONResponse(status_code=202, content={'job_id': job['id'], 'estado': 'en_cola'},
                        headers={'Location': f"/jobs/{job['id']}"})

if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT, workers=1)
