from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi.testclient import TestClient
import api
from job_store import JobStore
from worker import download_one, process_one


class JobsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = JobStore()
        self.patch = patch.object(api, '_store', self.store)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client = TestClient(api.app)
        self.addCleanup(self.client.close)
        self.payload = {'s3_url': 'https://example.test/audio.wav', 'fragmento_segundos': 3}

    def post(self, key=None):
        return self.client.post('/analyze', json=self.payload,
                                headers={'Idempotency-Key': key} if key else {})

    def stage(self, job_id):
        def fake_download(url, suffix):
            path = self.root / f'{job_id}.wav'
            path.write_bytes(b'fake')
            return path
        self.assertTrue(download_one(self.store, fake_download))

    def test_fast_post_and_thirty_minute_simulation(self):
        # A simulated inference clock advances 1800 seconds while the worker
        # stays blocked on a gate. No actual 30-minute wait or external service.
        start = time.monotonic()
        response = self.post()
        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(response.status_code, 202)
        job_id = response.json()['job_id']
        self.assertEqual(response.json()['estado'], 'en_cola')
        self.assertEqual(self.client.get(f'/jobs/{job_id}').json()['estado'], 'en_cola')
        self.stage(job_id)
        self.assertEqual(self.store.get(job_id)['status'], 'iniciando')
        started, release = threading.Event(), threading.Event()
        clock = [0]
        result = {'emociones': {'alegre': 50, 'neutral': 50},
                  'detalle': [{'tiempo_segundos': 3, 'emocion': 'alegre', 'confianza': 90, 'valencia': 75}]}
        def process(job):
            started.set()
            if not release.wait(5):
                raise RuntimeError('test gate timed out')
            clock[0] += 1800
            return result
        worker = threading.Thread(target=process_one, args=(self.store, process))
        worker.start()
        try:
            self.assertTrue(started.wait(2))
            with ThreadPoolExecutor(max_workers=8) as pool:
                responses = list(pool.map(lambda _: self.client.get(f'/jobs/{job_id}'), range(20)))
            self.assertTrue(all(r.status_code == 202 and r.json()['estado'] == 'procesando' for r in responses))
            self.assertTrue(worker.is_alive())
            # Close the original client; work continues independently.
            self.client.close()
        finally:
            release.set()
            worker.join(5)
        self.assertEqual(clock[0], 1800)
        other = TestClient(api.app)
        self.addCleanup(other.close)
        ready = other.get(f'/jobs/{job_id}')
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json(), {'job_id': job_id, 'estado': 'listo', **result})
        self.assertFalse((self.root / f'{job_id}.wav').exists())

    def test_idempotency_concurrent_threads(self):
        def submit(_):
            store = self.store
            return store.enqueue(self.payload['s3_url'], 'audio.wav', 3.0, 'same-key')['id']
        with ThreadPoolExecutor(max_workers=10) as pool:
            ids = list(pool.map(submit, range(30)))
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(self.post('same-key').json()['job_id'], ids[0])
        self.payload['s3_url'] = 'https://example.test/other.wav'
        self.assertEqual(self.post('same-key').status_code, 409)
        self.payload['s3_url'] = 'https://example.test/audio.wav'
        self.payload['fragmento_segundos'] = 4
        self.assertEqual(self.post('same-key').status_code, 409)

    def test_http_idempotency_concurrent_requests(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(lambda _: self.post('http-concurrent'), range(20)))
        self.assertTrue(all(response.status_code == 202 for response in responses))
        self.assertEqual(len({response.json()['job_id'] for response in responses}), 1)

    def test_post_acknowledges_queue_and_get_reports_actual_state(self):
        response = self.post('contract')
        job_id = response.json()['job_id']
        self.assertEqual(response.json(), {'job_id': job_id, 'estado': 'en_cola'})
        self.stage(job_id)
        pending = self.client.get(f'/jobs/{job_id}')
        self.assertEqual(pending.status_code, 202)
        self.assertEqual(pending.json(), {'job_id': job_id, 'estado': 'procesando'})
        self.store.finish(job_id, result={'emociones': {'neutral': 100}, 'detalle': []})
        self.assertEqual(self.post('contract').json(), {'job_id': job_id, 'estado': 'en_cola'})
        self.assertEqual(self.client.get(f'/jobs/{job_id}').json()['estado'], 'listo')

    def test_existing_result_serializer_preserves_scales(self):
        from types import SimpleNamespace
        from worker import make_processor
        item = SimpleNamespace(time_sec=3, dominant_es='Alegre', confidence=90.0, valence_pct=75.0)
        fake_analyzer = SimpleNamespace(EmotionAnalyzer=lambda: SimpleNamespace(analyze=lambda *a, **k: [item]),
            resumen_llamada=lambda results: {'valencia': 75, 'tono': 'positivo', 'emociones': {'alegre': 100}})
        fake_sf = SimpleNamespace(info=lambda path: SimpleNamespace(duration=1800.0))
        with patch.dict(sys.modules, {'analyzer': fake_analyzer, 'soundfile': fake_sf}):
            result = make_processor()({'id': 'test', 'path': 'fake.wav', 'filename': 'fake.wav', 'chunk': 3})
        self.assertEqual(result['duracion_segundos'], 1800.0)
        self.assertEqual(result['valencia'], 75)
        self.assertEqual(result['detalle'], [{'tiempo_segundos': 3, 'emocion': 'alegre', 'confianza': 90.0, 'valencia': 75.0}])

    def test_restart_loses_jobs_and_keys(self):
        job_id = self.post('restart').json()['job_id']
        with patch.object(api, '_store', JobStore()):
            self.assertEqual(self.client.get(f'/jobs/{job_id}').status_code, 404)
            self.assertNotEqual(self.post('restart').json()['job_id'], job_id)

    def test_single_service_starts_threads_and_returns_before_download(self):
        started, release, processing, finish = [threading.Event() for _ in range(4)]
        def download(url, suffix):
            started.set()
            if not release.wait(5):
                raise RuntimeError('test gate timed out')
            path = self.root / 'download.wav'
            path.write_bytes(b'fake')
            return path
        def process(job):
            processing.set()
            if not finish.wait(5):
                raise RuntimeError('test gate timed out')
            return {'emociones': {'neutral': 100}, 'detalle': []}
        with patch('worker._download_s3_audio', download), patch('worker.make_processor', return_value=process):
            try:
                with TestClient(api.app) as client:
                    start = time.monotonic()
                    response = client.post('/analyze', json=self.payload)
                    self.assertLess(time.monotonic()-start, 2)
                    job_id = response.json()['job_id']
                    self.assertEqual(response.json()['estado'], 'en_cola')
                    self.assertTrue(started.wait(2))
                    self.assertEqual(client.get(f'/jobs/{job_id}').json()['estado'], 'procesando')
                    release.set()
                    self.assertTrue(processing.wait(2))
                    self.assertEqual(client.get(f'/jobs/{job_id}').json()['estado'], 'procesando')
                    finish.set()
                    deadline = time.monotonic() + 2
                    while client.get(f'/jobs/{job_id}').status_code != 200 and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertEqual(client.get(f'/jobs/{job_id}').status_code, 200)
                    self.assertEqual(len(api._worker.threads), 2)
            finally:
                release.set()
                finish.set()

    def test_download_continues_while_analysis_is_blocked(self):
        first = self.post().json()['job_id']
        self.stage(first)
        second = self.post().json()['job_id']
        started, release = threading.Event(), threading.Event()
        def process(job):
            started.set()
            release.wait(5)
            return {'emociones': {}, 'detalle': []}
        worker = threading.Thread(target=process_one, args=(self.store, process))
        worker.start()
        try:
            self.assertTrue(started.wait(2))
            self.stage(second)
            self.assertIsNotNone(self.store.get(second)['path'])
            self.assertTrue(worker.is_alive())
        finally:
            release.set()
            worker.join(5)

    def test_failure_expiry_and_capacity(self):
        job_id = self.post().json()['job_id']
        def fail_download(*args):
            raise RuntimeError('download failed')
        download_one(self.store, fail_download)
        error = self.client.get(f'/jobs/{job_id}')
        self.assertEqual(error.status_code, 422)
        self.assertEqual(error.json()['error'], 'download failed')
        second = self.post().json()['job_id']
        self.stage(second)
        def fail_process(job):
            raise RuntimeError('inference failed')
        process_one(self.store, fail_process)
        self.assertEqual(self.client.get(f'/jobs/{second}').json()['error'], 'inference failed')
        expired = self.store.enqueue('https://example.test/a.wav', 'a.wav', 3, deadline=time.time()-1)
        with patch('worker._download_s3_audio') as network:
            download_one(self.store, network)
            network.assert_not_called()
        self.assertEqual(self.store.get(expired['id'])['status'], 'error')
        self.store.capacity = 1
        self.assertEqual(self.post('capacity').status_code, 202)
        self.assertEqual(self.post('capacity').status_code, 202)
        self.assertEqual(self.post().status_code, 429)

    def test_auth_validation_and_retention(self):
        with patch.object(api, 'API_TOKEN', 'secret'):
            self.assertEqual(self.post().status_code, 401)
            self.assertEqual(self.client.get('/jobs/missing').status_code, 401)
            response = self.client.post('/analyze', json=self.payload, headers={'x-api-key': 'secret'})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(self.client.get('/jobs/'+response.json()['job_id'], headers={'x-api-key':'secret'}).status_code, 202)
        self.assertEqual(self.client.post('/analyze', json={**self.payload, 'fragmento_segundos': 20}).status_code, 422)
        expired = api.download_deadline('https://example.test/a.wav?X-Amz-Date=20200101T000000Z&X-Amz-Expires=3600')
        self.assertLess(expired, time.time())
        job = self.store.enqueue('s3://bucket/test.wav', 'test.wav', 3, 'ttl')
        self.store.finish(job['id'], result={'emociones': {}})
        with self.store.lock:
            self.store.jobs[job['id']]['finished'] = time.time()-90000
        self.assertIsNone(self.store.get(job['id']))
        self.assertNotEqual(self.store.enqueue('s3://bucket/test.wav', 'test.wav', 3, 'ttl')['id'], job['id'])


if __name__ == '__main__':
    unittest.main()
