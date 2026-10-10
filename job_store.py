"""Thread-safe jobs and results in RAM, scoped to one API process."""
import threading
import time
import uuid

class JobConflict(Exception):
    pass

class QueueFull(Exception):
    pass

class JobStore:
    def __init__(self, capacity=20, ttl=86400):
        self.capacity, self.ttl = capacity, ttl
        self.jobs = {}
        self.keys = {}
        self.lock = threading.Lock()

    def _purge(self):
        now = time.time()
        stale = [key for key, job in self.jobs.items()
                 if job['finished'] is not None and now - job['finished'] > self.ttl]
        for key in stale:
            job = self.jobs.pop(key)
            if job['idem']:
                self.keys.pop(job['idem'], None)

    def enqueue(self, url, filename, chunk, idem=None, deadline=None):
        with self.lock:
            self._purge()
            if idem in self.keys:
                job = self.jobs[self.keys[idem]]
                if (job['url'], job['chunk']) != (url, chunk):
                    raise JobConflict('Idempotency-Key ya utilizada con otra URL o fragmento.')
                return dict(job)
            pending = sum(job['status'] in ('en_cola', 'iniciando') for job in self.jobs.values())
            if pending >= self.capacity:
                raise QueueFull()
            job = dict(id=uuid.uuid4().hex, url=url, filename=filename, chunk=chunk,
                       idem=idem, status='en_cola', deadline=deadline, path=None,
                       result=None, error=None, finished=None)
            self.jobs[job['id']] = job
            if idem:
                self.keys[idem] = job['id']
            return dict(job)

    def get(self, job_id):
        with self.lock:
            self._purge()
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def claim(self, download=False):
        with self.lock:
            for job in self.jobs.values():
                ready = job['status'] == 'en_cola' if download else job['status'] == 'iniciando' and job['path'] is not None
                if ready:
                    job['status'] = 'iniciando' if download else 'procesando'
                    return dict(job)
            return None

    def staged(self, job_id, path):
        with self.lock:
            if self.jobs[job_id]['finished'] is not None:
                return False
            self.jobs[job_id]['path'] = str(path)
            return True

    def finish(self, job_id, result=None, error=None):
        with self.lock:
            self.jobs[job_id].update(status='error' if error is not None else 'listo',
                                     result=result, error=error, finished=time.time(), path=None)

    def counts(self):
        with self.lock:
            self._purge()
            counts = {}
            for job in self.jobs.values():
                if job['finished'] is None:
                    counts[job['status']] = counts.get(job['status'], 0) + 1
            return counts

    def fail_pending(self, error):
        with self.lock:
            paths = []
            for job in self.jobs.values():
                if job['finished'] is None:
                    if job['path']:
                        paths.append(job['path'])
                    job.update(status='error', error=error, finished=time.time(), path=None)
            return paths
