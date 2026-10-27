"""Background cleaning jobs, so a long AI run never hits an HTTP timeout.

The extension starts a job, then polls it for progress. Jobs live in memory
only (nothing is written to disk) and the newest few are kept.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional

MAX_JOBS = 10


class CleaningJob:
    def __init__(self) -> None:
        self.id = uuid.uuid4().hex
        self.status = "running"  # running | done | error | cancelled
        self.progress: Dict[str, Any] = {}
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.task: Optional["asyncio.Task[None]"] = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "jobId": self.id,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
        }


_jobs: Dict[str, CleaningJob] = {}


def start_job(
    run: Callable[[Callable[[Dict[str, Any]], None]], Awaitable[Dict[str, Any]]],
    describe_error: Callable[[BaseException], str],
) -> CleaningJob:
    """Run ``run(report_progress)`` in the background and track it."""
    job = CleaningJob()
    while len(_jobs) >= MAX_JOBS:
        _jobs.pop(next(iter(_jobs)))
    _jobs[job.id] = job

    async def runner() -> None:
        try:
            job.result = await run(lambda info: job.progress.update(info))
            job.status = "done"
        except asyncio.CancelledError:
            job.status = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 - reported to the poller
            job.error = describe_error(exc)
            job.status = "error"

    job.task = asyncio.get_running_loop().create_task(runner())
    return job


def get_job(job_id: str) -> Optional[CleaningJob]:
    return _jobs.get(job_id)


def cancel_job(job_id: str) -> bool:
    job = _jobs.get(job_id)
    if job is None:
        return False
    if job.task and not job.task.done():
        job.task.cancel()
    job.status = "cancelled" if job.status == "running" else job.status
    return True
