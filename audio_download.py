"""Audio download helpers used only by the durable worker."""
import os
import tempfile
import time
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
from fastapi import HTTPException
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "150"))
DOWNLOAD_TIMEOUT_SECONDS = float(os.environ.get("DOWNLOAD_TIMEOUT_SECONDS", "600"))
AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", "data/audio")).resolve()
def _log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

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
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=AUDIO_DIR)
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
    deadline = time.monotonic() + DOWNLOAD_TIMEOUT_SECONDS
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
                if time.monotonic() >= deadline:
                    raise HTTPException(status_code=400, detail="Tiempo maximo de descarga superado.")
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
        from botocore.config import Config
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Para usar URLs s3:// instala boto3 y configura permisos IAM.",
        ) from exc

    size = 0
    deadline = time.monotonic() + DOWNLOAD_TIMEOUT_SECONDS
    try:
        body = boto3.client("s3", config=Config(connect_timeout=60, read_timeout=60,
                           retries={"total_max_attempts": 1})).get_object(Bucket=bucket, Key=key)["Body"]
        try:
            while True:
                if time.monotonic() >= deadline:
                    raise HTTPException(status_code=400, detail="Tiempo maximo de descarga superado.")
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                size = _write_download_chunk(handle, path, chunk, size, limit)
        finally:
            body.close()
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo descargar s3://{bucket}/{key}: {exc}",
        ) from exc


