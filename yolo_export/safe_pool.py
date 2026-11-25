"""safe_pool: a process pool that survives a worker dying outright or hanging.

multiprocessing.Pool loses the task of a worker that dies hard (an access violation in a C
extension, e.g. numpy-2-era pickles loaded under numpy 1.x) and then waits for its result
forever. Here each worker is a spawn process with its own pipe and runs one job at a time, so the
parent always knows which job a worker holds:

- a worker that dies mid-job fails that job only (with its exit code); a fresh worker replaces it
- a job that runs longer than `timeout` seconds has its worker killed and fails the same way
- an exception that escapes `fn` fails that job only

Failed jobs are passed to `on_fail(job, message)`, whose return value is yielded in place of the
job's result, so callers see one result per job and carry on. Python 3.8-clean, stdlib only.
"""
import multiprocessing as mp
import time
from multiprocessing.connection import wait

# Windows NTSTATUS codes a crashed worker exits with (exit code > 0xFFFF)
_NTSTATUS = {0xC0000005: "access violation", 0xC00000FD: "stack overflow", 0xC0000409: "stack buffer overrun",
             0xC0000017: "out of memory", 0xC000001D: "illegal instruction"}


def describe_exit(code):
    """A readable reason for a worker's exit code."""
    if code is None:
        return "still running"
    if code < 0:
        return "killed by signal %d" % -code
    if code > 0xFFFF:
        code &= 0xFFFFFFFF
        return "exit code 0x%08X (%s)" % (code, _NTSTATUS.get(code, "crash"))
    return "exit code %d" % code


def _worker_main(conn, initializer):
    if initializer is not None:
        initializer()
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError, KeyboardInterrupt):   # parent gone, or Ctrl-C (the parent stops us)
            return
        if msg is None:
            return
        fn, job = msg
        try:
            out = (True, fn(job))
        except BaseException as exc:  # noqa: BLE001 -- reported to the parent as this job's failure
            out = (False, "%s: %s" % (type(exc).__name__, exc))
        try:
            conn.send(out)
        except Exception as exc:  # noqa: BLE001 -- e.g. a result that cannot be pickled
            conn.send((False, "result could not be sent back: %s: %s" % (type(exc).__name__, exc)))


class _Slot(object):
    __slots__ = ("proc", "conn", "job", "started", "n_done")

    def __init__(self, proc, conn):
        self.proc, self.conn, self.job, self.started, self.n_done = proc, conn, None, 0.0, 0


class SafePool(object):
    """`workers` spawn processes; see the module docstring. Use as a context manager or call
    close() (after all results were read) or terminate()."""

    def __init__(self, workers, initializer=None, maxtasksperchild=None, timeout=None, ctx=None):
        self.workers = max(1, int(workers))
        self.initializer = initializer
        self.maxtasksperchild = maxtasksperchild
        self.timeout = timeout
        self.ctx = ctx or mp.get_context("spawn")
        self.slots = []

    # ---- worker lifecycle
    def _start(self):
        parent, child = self.ctx.Pipe(duplex=True)
        proc = self.ctx.Process(target=_worker_main, args=(child, self.initializer), daemon=True)
        proc.start()
        child.close()
        return _Slot(proc, parent)

    @staticmethod
    def _kill(slot):
        try:
            if slot.proc.is_alive():
                slot.proc.terminate()
            slot.proc.join(5)
            if slot.proc.is_alive():
                slot.proc.kill()
                slot.proc.join(5)
        except Exception:  # noqa: BLE001
            pass
        try:
            slot.conn.close()
        except Exception:  # noqa: BLE001
            pass

    def _retire(self, slot):
        """Ask an idle worker to exit; kill it if it does not."""
        try:
            slot.conn.send(None)
        except Exception:  # noqa: BLE001
            pass
        slot.proc.join(10)
        self._kill(slot)

    def close(self):
        for s in self.slots:
            self._retire(s)
        self.slots = []

    def terminate(self):
        for s in self.slots:
            self._kill(s)
        self.slots = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.terminate()
        return False

    # ---- running jobs
    def imap_unordered(self, fn, jobs, on_fail):
        """Yield one result per job, in completion order: fn(job), or on_fail(job, message) when the
        job raised, its worker died, or it ran past the timeout."""
        jobs = list(jobs)
        nxt, pending = 0, len(jobs)
        if not jobs:
            return
        while len(self.slots) < min(self.workers, len(jobs)):
            self.slots.append(self._start())
        while pending:
            for s in self.slots:                                   # hand out work
                if s.job is None and nxt < len(jobs):
                    s.job, s.started = jobs[nxt], time.time()
                    nxt += 1
                    try:
                        s.conn.send((fn, s.job))
                    except Exception:  # noqa: BLE001 -- dead already: picked up below
                        pass
            busy = [s for s in self.slots if s.job is not None]
            ready = wait([s.conn for s in busy] + [s.proc.sentinel for s in busy], timeout=1.0)
            now = time.time()
            for i, s in enumerate(list(self.slots)):
                if s.job is None:
                    continue
                result, failed = None, None
                has_data = s.conn in ready
                if not has_data and s.proc.sentinel in ready:
                    try:                                           # it may have answered just before exiting
                        has_data = s.conn.poll()
                    except (EOFError, OSError):
                        has_data = False
                if has_data:
                    try:
                        ok, value = s.conn.recv()
                        if ok:
                            result = (value,)
                        else:
                            failed = value
                    except (EOFError, OSError):
                        s.proc.join(5)
                        failed = "worker process died (%s)" % describe_exit(s.proc.exitcode)
                elif s.proc.sentinel in ready or not s.proc.is_alive():
                    s.proc.join(5)
                    failed = "worker process died (%s)" % describe_exit(s.proc.exitcode)
                elif self.timeout and now - s.started > self.timeout:
                    failed = "timed out after %.0f s; worker stopped" % (now - s.started)
                else:
                    continue
                job, s.job = s.job, None
                pending -= 1
                s.n_done += 1
                if failed is not None and (not s.proc.is_alive() or failed.startswith("timed out")):
                    self._kill(s)                                  # replace the dead / stuck worker
                    self.slots[i] = self._start() if nxt < len(jobs) else None
                elif self.maxtasksperchild and s.n_done >= self.maxtasksperchild and nxt < len(jobs):
                    self._retire(s)                                # bound memory growth
                    self.slots[i] = self._start()
                yield result[0] if failed is None else on_fail(job, failed)
            self.slots = [s for s in self.slots if s is not None]
