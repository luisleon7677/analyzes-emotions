"""Internal threads started by the API. No separate service or command."""
import threading
from pathlib import Path
import time

from audio_download import AUDIO_DIR, _download_s3_audio, _log


def download_one(store, download=None):
    download = download or _download_s3_audio
    job = store.claim(download=True)
    if not job:
        return False
    try:
        if job['deadline'] is not None and time.time() >= job['deadline']:
            raise ValueError('URL vencida o limite de espera de descarga superado. Reenvia una URL vigente con una nueva Idempotency-Key.')
        path = download(job['url'], Path(job['filename']).suffix.lower())
        try:
            if not store.staged(job['id'], path):
                Path(path).unlink(missing_ok=True)
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise
    except Exception as exc:
        store.finish(job['id'], error=str(getattr(exc, 'detail', exc)))
    return True


def process_one(store, process):
    job = store.claim()
    if not job:
        return False
    try:
        store.finish(job['id'], result=process(job))
    except Exception as exc:
        store.finish(job['id'], error=str(exc))
    finally:
        Path(job['path']).unlink(missing_ok=True)
    return True


def make_processor():
    import soundfile as sf
    from analyzer import EmotionAnalyzer, resumen_llamada
    analyzer = EmotionAnalyzer()

    def process(job):
        info = sf.info(job['path'])
        results = analyzer.analyze(job['path'], chunk_seconds=job['chunk'],
            progress_callback=lambda progress, label: _log(f"{job['id']} {label} ({progress * 100:.1f}%)"))
        if not results:
            raise ValueError('El audio no produjo fragmentos validos.')
        resumen = resumen_llamada(results)
        return {'archivo': job['filename'], 'duracion_segundos': round(float(info.duration), 2),
                'fragmentos': len(results), 'fragmento_segundos': job['chunk'],
                'valencia': resumen['valencia'], 'tono': resumen['tono'], 'emociones': resumen['emociones'],
                'detalle': [{'tiempo_segundos': round(item.time_sec, 2), 'emocion': item.dominant_es.lower(),
                             'confianza': round(item.confidence, 1), 'valencia': round(item.valence_pct, 1)}
                            for item in results]}
    return process


class InternalWorker:
    def __init__(self, store):
        self.store = store
        self.stop_event = threading.Event()
        self.model_ready = False
        self.error = None
        self.threads = []

    def start(self):
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        self.threads = [threading.Thread(target=self._download_loop, name='descargas', daemon=True),
                        threading.Thread(target=self._process_loop, name='analisis', daemon=True)]
        for thread in self.threads:
            thread.start()

    def _download_loop(self):
        while not self.stop_event.is_set():
            try:
                if not download_one(self.store):
                    self.stop_event.wait(0.2)
            except Exception as exc:
                _log(f'Descarga: {exc}')
                self.stop_event.wait(1)

    def _process_loop(self):
        try:
            process = make_processor()
            self.model_ready = True
        except Exception as exc:
            self.error = str(exc)
            self.stop_event.set()
            for path in self.store.fail_pending('No se pudo cargar el modelo: ' + self.error):
                Path(path).unlink(missing_ok=True)
            _log('Error cargando modelo: ' + self.error)
            return
        while not self.stop_event.is_set():
            if not process_one(self.store, process):
                self.stop_event.wait(0.2)

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=1)
