"""Servicio HTTP de análisis emocional.

Un solo worker atiende la cola para no saturar la CPU ni duplicar el modelo.
Ejecutar un único proceso:

    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hmac
import os
import queue
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import soundfile as sf
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from analyzer import DEFAULT_CHUNK_SECONDS, EmotionAnalyzer, resumen_llamada

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "8"))
JOB_WAIT_TIMEOUT = float(os.environ.get("JOB_WAIT_TIMEOUT", "900"))
JOB_TTL_SECONDS = float(os.environ.get("JOB_TTL_SECONDS", "3600"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))
API_TOKEN = os.environ.get("API_TOKEN", "").strip()
ALLOWED_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}

_analyzer: EmotionAnalyzer | None = None
_jobs: dict[str, "Job"] = {}
_jobs_lock = threading.Lock()
_work: queue.Queue[str | None] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
_worker: threading.Thread | None = None
_busy = 0
_busy_lock = threading.Lock()


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@dataclass
class Job:
    id: str
    path: str
    filename: str
    chunk_seconds: float
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    status: str = "en_cola"
    result: dict | None = None
    error: str | None = None
    event: threading.Event = field(default_factory=threading.Event)


class AnalyzeRequest(BaseModel):
    s3_url: str = Field(
        ...,
        description="URL del audio en S3. Acepta https presigned URL o s3://bucket/key.",
    )
    fragmento_segundos: float = Field(DEFAULT_CHUNK_SECONDS, ge=1.0, le=10.0)


def _queue_depth() -> int:
    return _work.qsize()


def _purge_jobs() -> None:
    now = time.time()
    with _jobs_lock:
        stale = [
            job_id
            for job_id, job in _jobs.items()
            if job.finished_at is not None and now - job.finished_at > JOB_TTL_SECONDS
        ]
        for job_id in stale:
            _jobs.pop(job_id, None)


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not API_TOKEN:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, API_TOKEN):
        raise HTTPException(status_code=401, detail="API key invalida.")


def _worker_loop() -> None:
    global _busy
    assert _analyzer is not None
    while True:
        job_id = _work.get()
        if job_id is None:
            _work.task_done()
            break
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            _work.task_done()
            continue
        job.status = "procesando"
        with _busy_lock:
            _busy += 1
        started_at = time.time()
        _log(f"[cola] procesando {job.id} archivo={job.filename}")
        try:
            info = sf.info(job.path)
            _log(
                "[analisis] "
                f"{job.id} audio={info.duration:.2f}s "
                f"fragmento={job.chunk_seconds}s"
            )

            def report_progress(progress: float, label: str) -> None:
                _log(f"[analisis] {job.id} {label} ({progress * 100:.1f}%)")

            results = _analyzer.analyze(
                job.path,
                chunk_seconds=job.chunk_seconds,
                progress_callback=report_progress,
            )
            if not results:
                raise ValueError("El audio no produjo fragmentos válidos.")
            resumen = resumen_llamada(results)
            job.result = {
                "archivo": job.filename,
                "duracion_segundos": round(float(info.duration), 2),
                "fragmentos": len(results),
                "fragmento_segundos": job.chunk_seconds,
                "valencia": resumen["valencia"],
                "tono": resumen["tono"],
                "emociones": resumen["emociones"],
                "detalle": [
                    {
                        "tiempo_segundos": round(item.time_sec, 2),
                        "emocion": item.dominant_es.lower(),
                        "confianza": round(item.confidence, 1),
                        "valencia": round(item.valence_pct, 1),
                    }
                    for item in results
                ],
            }
            job.status = "listo"
            _log(
                "[cola] "
                f"listo {job.id} fragmentos={len(results)} "
                f"valencia={resumen['valencia']} "
                f"tiempo={time.time() - started_at:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001
            job.status = "error"
            job.error = str(exc)
            _log(f"[cola] error {job.id} tiempo={time.time() - started_at:.1f}s: {exc}")
        finally:
            job.finished_at = time.time()
            Path(job.path).unlink(missing_ok=True)
            job.event.set()
            with _busy_lock:
                _busy -= 1
            _work.task_done()
            _purge_jobs()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _analyzer, _worker
    _log("[api] cargando modelo")
    _analyzer = EmotionAnalyzer()
    _worker = threading.Thread(target=_worker_loop, name="analisis", daemon=True)
    _worker.start()
    _log(f"[api] cola lista (maximo {MAX_QUEUE_SIZE})")
    yield
    _work.put(None)


app = FastAPI(title="Análisis emocional de audio", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    with _busy_lock:
        busy = _busy
    return {
        "status": "ok" if _analyzer is not None else "cargando",
        "cola": _queue_depth(),
        "cola_maxima": MAX_QUEUE_SIZE,
        "en_proceso": busy,
    }


@app.get("/jobs/{job_id}", dependencies=[Depends(_require_api_key)])
def job_status(job_id: str) -> JSONResponse:
    _purge_jobs()
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
    if job.status == "error":
        return JSONResponse(
            status_code=422,
            content={"job_id": job.id, "estado": job.status, "error": job.error},
        )
    if job.status != "listo":
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job.id,
                "estado": job.status,
                "cola": _queue_depth(),
            },
        )
    return JSONResponse(content={"job_id": job.id, "estado": job.status, **job.result})


@app.post("/analyze", dependencies=[Depends(_require_api_key)])
def analyze(payload: AnalyzeRequest) -> JSONResponse:
    request_started_at = time.time()
    s3_url = payload.s3_url.strip()
    fragmento_segundos = payload.fragmento_segundos
    filename = _filename_from_s3_url(s3_url)
    _log(
        "[api] "
        f"POST /analyze recibido s3_url={_safe_url_for_log(s3_url)} "
        f"archivo={filename} "
        f"fragmento={fragmento_segundos}s"
    )
    if _analyzer is None:
        _log("[api] POST /analyze rechazado: modelo cargando")
        raise HTTPException(status_code=503, detail="El modelo todavía se está cargando.")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        _log(f"[api] POST /analyze rechazado: formato no admitido archivo={filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Formato no admitido. Usa: {', '.join(sorted(ALLOWED_SUFFIXES))}.",
        )

    path = _download_s3_audio(s3_url, suffix)
    upload_size_mb = path.stat().st_size / (1024 * 1024)
    _log(
        "[api] "
        f"archivo descargado archivo={filename} tamano={upload_size_mb:.2f}MB "
        f"tiempo={time.time() - request_started_at:.1f}s"
    )
    job = Job(
        id=uuid.uuid4().hex,
        path=str(path),
        filename=filename,
        chunk_seconds=fragmento_segundos,
    )
    with _jobs_lock:
        _jobs[job.id] = job
    try:
        _work.put_nowait(job.id)
    except queue.Full:
        with _jobs_lock:
            _jobs.pop(job.id, None)
        path.unlink(missing_ok=True)
        _log(f"[cola] rechazada {job.id} archivo={filename}: cola llena")
        raise HTTPException(
            status_code=429,
            detail=f"Cola llena ({MAX_QUEUE_SIZE}). Reintenta más tarde.",
        ) from None

    _log(f"[cola] encolado {job.id} archivo={filename} profundidad={_queue_depth()}")
    _log(f"[api] esperando resultado job={job.id} timeout={JOB_WAIT_TIMEOUT}s")
    finished = job.event.wait(JOB_WAIT_TIMEOUT)
    if not finished:
        _log(
            "[api] "
            f"POST /analyze responde 202 job={job.id} estado={job.status} "
            f"tiempo={time.time() - request_started_at:.1f}s"
        )
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job.id,
                "estado": job.status,
                "mensaje": "El análisis sigue en cola. Consulta GET /jobs/{job_id}.",
            },
        )
    if job.status == "error":
        _log(
            "[api] "
            f"POST /analyze responde 422 job={job.id} "
            f"tiempo={time.time() - request_started_at:.1f}s"
        )
        return JSONResponse(
            status_code=422,
            content={"job_id": job.id, "estado": job.status, "error": job.error},
        )
    _log(
        "[api] "
        f"POST /analyze responde 200 job={job.id} "
        f"tiempo={time.time() - request_started_at:.1f}s"
    )
    return JSONResponse(content={"job_id": job.id, "estado": job.status, **(job.result or {})})


def _filename_from_s3_url(s3_url: str) -> str:
    parsed = urlparse(s3_url)
    if parsed.scheme == "s3":
        filename = Path(unquote(parsed.path)).name
    elif parsed.scheme in {"http", "https"}:
        filename = Path(unquote(parsed.path)).name
    else:
        raise HTTPException(
            status_code=400,
            detail="s3_url debe ser una URL https de S3 o s3://bucket/key.",
        )
    if not filename:
        raise HTTPException(status_code=400, detail="La URL de S3 no contiene archivo.")
    return filename


def _safe_url_for_log(s3_url: str) -> str:
    parsed = urlparse(s3_url)
    if parsed.scheme in {"http", "https"} and parsed.query:
        return parsed._replace(query="...").geturl()
    return s3_url


def _download_s3_audio(s3_url: str, suffix: str) -> Path:
    limit = MAX_UPLOAD_MB * 1024 * 1024
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    path = Path(handle.name)
    try:
        parsed = urlparse(s3_url)
        if parsed.scheme == "s3":
            bucket = parsed.netloc
            key = unquote(parsed.path.lstrip("/"))
            if not bucket or not key:
                raise HTTPException(
                    status_code=400,
                    detail="La URL s3:// debe incluir bucket y key.",
                )
            _download_with_boto3(bucket, key, handle, path, limit)
        elif parsed.scheme in {"http", "https"}:
            _download_with_http(s3_url, handle, path, limit)
        else:
            raise HTTPException(
                status_code=400,
                detail="s3_url debe ser una URL https de S3 o s3://bucket/key.",
            )
    except Exception:
        handle.close()
        path.unlink(missing_ok=True)
        raise
    handle.close()
    if path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="El archivo de audio esta vacio.")
    return path


def _write_download_chunk(handle, path: Path, chunk: bytes, current_size: int, limit: int) -> int:
    current_size += len(chunk)
    if current_size > limit:
        raise HTTPException(
            status_code=413,
            detail=f"El audio supera el maximo de {MAX_UPLOAD_MB} MB.",
        )
    handle.write(chunk)
    _log(
        "[api] "
        f"descargando archivo temporal={path.name} "
        f"tamano={current_size / (1024 * 1024):.2f}MB"
    )
    return current_size


def _download_with_http(s3_url: str, handle, path: Path, limit: int) -> None:
    size = 0
    request = Request(s3_url, headers={"User-Agent": "emotion-analyzer/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            status = getattr(response, "status", 200)
            if status >= 400:
                raise HTTPException(
                    status_code=400,
                    detail=f"No se pudo descargar el audio desde S3. HTTP {status}.",
                )
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size = _write_download_chunk(handle, path, chunk, size, limit)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo descargar el audio desde la URL: {exc}",
        ) from exc


def _download_with_boto3(bucket: str, key: str, handle, path: Path, limit: int) -> None:
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Para usar URLs s3:// instala boto3 y configura permisos IAM.",
        ) from exc

    size = 0
    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"]
        while True:
            chunk = body.read(1024 * 1024)
            if not chunk:
                break
            size = _write_download_chunk(handle, path, chunk, size, limit)
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo descargar s3://{bucket}/{key}: {exc}",
        ) from exc


async def _save_upload(upload, suffix: str) -> Path:
    limit = MAX_UPLOAD_MB * 1024 * 1024
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    path = Path(handle.name)
    size = 0
    try:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise HTTPException(
                    status_code=413,
                    detail=f"El audio supera el máximo de {MAX_UPLOAD_MB} MB.",
                )
            handle.write(chunk)
            _log(
                "[api] "
                f"recibiendo archivo temporal={path.name} "
                f"tamano={size / (1024 * 1024):.2f}MB"
            )
    except Exception:
        handle.close()
        path.unlink(missing_ok=True)
        raise
    handle.close()
    if size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="El archivo de audio está vacío.")
    return path


def main() -> None:
    uvicorn.run("api:app", host=HOST, port=PORT, workers=1)


if __name__ == "__main__":
    main()
