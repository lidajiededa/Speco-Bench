from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiohttp import web

from .config import BenchmarkConfig
from .matrix import (
    DatasetSpec,
    _error_row,
    _slug,
    parse_concurrencies,
    parse_num_prompts,
    report_to_csv_row,
    resolve_datasets,
    write_csv,
)
from .models import ProgressUpdate
from .output import save_report
from .service import BenchmarkService


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
MAX_TASK_NAME_LENGTH = 80


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _optional_text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value.strip() or None


def _task_directory_name(task_name: str, job_id: str) -> str:
    safe_name = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in task_name
    )
    while "--" in safe_name:
        safe_name = safe_name.replace("--", "-")
    safe_name = safe_name.strip("._-")[:60] or "task"
    return f"{safe_name}-{job_id}"


def _integer(
    payload: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int | None:
    value = payload.get(key, default)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _integer_default(payload: dict[str, Any], key: str, *, default: int) -> int:
    value = _integer(payload, key, default=default)
    return default if value is None else value


def _number(payload: dict[str, Any], key: str, *, default: float) -> float:
    value = payload.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc


def _boolean(payload: dict[str, Any], key: str, *, default: bool = False) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _string_values(value: Any, key: str) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(f"{key} must be a string or list")
    if not all(isinstance(item, str) for item in values):
        raise ValueError(f"{key} must contain strings")
    return values


@dataclass(frozen=True, slots=True)
class WebRunSpec:
    dataset: DatasetSpec
    config: BenchmarkConfig
    requested_num_prompts: int | None


@dataclass(frozen=True, slots=True)
class WebRandomWorkload:
    input_len: int
    output_len: int
    concurrencies: list[int]
    num_prompts: list[int | None]


def _concurrency_plan(
    payload: dict[str, Any],
    *,
    context: str | None = None,
) -> tuple[list[int], list[int | None]]:
    try:
        concurrencies = parse_concurrencies(
            _string_values(payload.get("concurrencies", ["1"]), "concurrencies")
        )
        raw_num_prompts = payload.get("num_prompts")
        num_prompts = parse_num_prompts(
            (
                _string_values(raw_num_prompts, "num_prompts")
                if raw_num_prompts not in (None, "", [])
                else None
            ),
            expected_count=len(concurrencies),
        )
    except ValueError as exc:
        if context is None:
            raise
        raise ValueError(f"{context}: {exc}") from exc
    return concurrencies, num_prompts


def _random_workloads(payload: dict[str, Any]) -> list[WebRandomWorkload]:
    raw_workloads = payload.get("random_workloads")
    if raw_workloads is None:
        concurrencies, num_prompts = _concurrency_plan(payload)
        return [
            WebRandomWorkload(
                input_len=_integer_default(payload, "random_input_len", default=1024),
                output_len=_integer_default(payload, "random_output_len", default=128),
                concurrencies=concurrencies,
                num_prompts=num_prompts,
            )
        ]
    if not isinstance(raw_workloads, list) or not raw_workloads:
        raise ValueError("random_workloads must be a non-empty list")

    workloads: list[WebRandomWorkload] = []
    for index, raw_workload in enumerate(raw_workloads, start=1):
        context = f"random_workloads[{index - 1}]"
        if not isinstance(raw_workload, dict):
            raise ValueError(f"{context} must be an object")
        concurrencies, num_prompts = _concurrency_plan(
            raw_workload,
            context=context,
        )
        workloads.append(
            WebRandomWorkload(
                input_len=_integer_default(raw_workload, "input_len", default=1024),
                output_len=_integer_default(raw_workload, "output_len", default=128),
                concurrencies=concurrencies,
                num_prompts=num_prompts,
            )
        )
    return workloads


@dataclass(slots=True)
class WebJob:
    job_id: str
    name: str
    status: str
    created_at: str
    configuration: dict[str, Any]
    run_specs: list[WebRunSpec] = field(repr=False)
    started_at: str | None = None
    finished_at: str | None = None
    progress: dict[str, Any] | None = None
    runs: list[dict[str, Any]] = field(default_factory=list)
    result_dir: str | None = None
    csv_path: str | None = None
    error: str | None = None
    task: asyncio.Task[None] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.job_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "configuration": self.configuration,
            "progress": self.progress,
            "runs": self.runs,
            "result_dir": self.result_dir,
            "csv_path": self.csv_path,
            "error": self.error,
        }


class WebJobManager:
    def __init__(
        self,
        *,
        dataset_root: Path,
        output_root: Path,
        service: BenchmarkService | None = None,
        max_history: int = 1000,
    ):
        self.dataset_root = dataset_root.resolve()
        self.output_root = output_root.resolve()
        self.service = service or BenchmarkService()
        self.max_history = max_history
        self.jobs: dict[str, WebJob] = {}
        self._lock = asyncio.Lock()

    def available_datasets(self) -> list[dict[str, str]]:
        if not self.dataset_root.is_dir():
            return []
        datasets = []
        for path in sorted(self.dataset_root.glob("*/question.jsonl")):
            datasets.append({"name": path.parent.name, "path": str(path.resolve())})
        return datasets

    def _build_run_specs(
        self,
        payload: dict[str, Any],
        job_dir: Path,
    ) -> tuple[list[WebRunSpec], dict[str, Any]]:
        base_url = _required_text(payload, "base_url")
        model = _required_text(payload, "model")
        dataset_name = str(payload.get("dataset_name", "custom"))
        if dataset_name not in {"custom", "random"}:
            raise ValueError("dataset_name must be custom or random")

        endpoint_type = str(payload.get("endpoint_type", "chat"))
        if endpoint_type not in {"chat", "completions"}:
            raise ValueError("endpoint_type must be chat or completions")

        if dataset_name == "random":
            workloads = _random_workloads(payload)
            datasets = [
                DatasetSpec(
                    name=f"random-in{workload.input_len}-out{workload.output_len}",
                    path=Path("<generated>"),
                )
                for workload in workloads
            ]
            concurrencies = [
                concurrency
                for workload in workloads
                for concurrency in workload.concurrencies
            ]
            num_prompts_values = [
                num_prompts
                for workload in workloads
                for num_prompts in workload.num_prompts
            ]
        else:
            workloads = []
            concurrencies, num_prompts_values = _concurrency_plan(payload)
            raw_datasets = payload.get("datasets")
            if raw_datasets in (None, "", []):
                raise ValueError("at least one dataset is required")
            datasets = resolve_datasets(
                _string_values(raw_datasets, "datasets"),
                self.dataset_root,
            )

        extra_body = payload.get("extra_body", {})
        if not isinstance(extra_body, dict):
            raise ValueError("extra_body must be a JSON object")

        common: dict[str, Any] = {
            "base_url": base_url,
            "model": model,
            "endpoint_type": endpoint_type,
            "temperature": _number(payload, "temperature", default=0.0),
            "top_p": _number(payload, "top_p", default=1.0),
            "seed": _integer_default(payload, "seed", default=0),
            "warmup_requests": _integer_default(
                payload,
                "warmup_requests",
                default=1,
            ),
            "request_timeout_seconds": _number(
                payload,
                "request_timeout_seconds",
                default=3600.0,
            ),
            "api_key": _optional_text(payload, "api_key"),
            "metrics_url": _optional_text(payload, "metrics_url"),
            "ignore_eos": _boolean(payload, "ignore_eos"),
            "extra_body": extra_body,
        }

        specs: list[WebRunSpec] = []

        def add_run_spec(
            *,
            dataset: DatasetSpec,
            concurrency: int,
            num_prompts: int | None,
            result_dir: Path,
            random_input_len: int = 1024,
            random_output_len: int = 128,
        ) -> None:
            config = BenchmarkConfig(
                **common,
                dataset_name=dataset_name,
                dataset_path=(dataset.path if dataset_name == "custom" else None),
                output_dir=result_dir,
                concurrency=concurrency,
                num_prompts=num_prompts,
                max_tokens=(
                    _integer(payload, "max_tokens")
                    if dataset_name == "custom"
                    else None
                ),
                random_input_len=random_input_len,
                random_output_len=random_output_len,
                random_range_ratio=_number(
                    payload,
                    "random_range_ratio",
                    default=0.0,
                ),
                random_image_width=(
                    _integer(payload, "random_image_width")
                    if dataset_name == "random"
                    else None
                ),
                random_image_height=(
                    _integer(payload, "random_image_height")
                    if dataset_name == "random"
                    else None
                ),
                random_images_per_prompt=_integer_default(
                    payload,
                    "random_images_per_prompt",
                    default=1,
                ),
                tokenizer=_optional_text(payload, "tokenizer"),
                trust_remote_code=_boolean(payload, "trust_remote_code"),
            )
            config.validate()
            specs.append(
                WebRunSpec(
                    dataset=dataset,
                    config=config,
                    requested_num_prompts=num_prompts,
                )
            )

        if dataset_name == "random":
            for workload_index, (dataset, workload) in enumerate(
                zip(datasets, workloads),
                start=1,
            ):
                for concurrency, num_prompts in zip(
                    workload.concurrencies,
                    workload.num_prompts,
                ):
                    prompt_label = (
                        str(num_prompts) if num_prompts is not None else "all"
                    )
                    result_dir = (
                        job_dir
                        / (
                            f"workload-{workload_index:02d}"
                            f"-input-{workload.input_len}"
                            f"-output-{workload.output_len}"
                        )
                        / f"concurrency-{concurrency}-prompts-{prompt_label}"
                    )
                    add_run_spec(
                        dataset=dataset,
                        concurrency=concurrency,
                        num_prompts=num_prompts,
                        result_dir=result_dir,
                        random_input_len=workload.input_len,
                        random_output_len=workload.output_len,
                    )
        else:
            for dataset in datasets:
                for concurrency, num_prompts in zip(
                    concurrencies,
                    num_prompts_values,
                ):
                    prompt_label = (
                        str(num_prompts) if num_prompts is not None else "all"
                    )
                    result_dir = (
                        job_dir
                        / _slug(dataset.name)
                        / f"concurrency-{concurrency}-prompts-{prompt_label}"
                    )
                    add_run_spec(
                        dataset=dataset,
                        concurrency=concurrency,
                        num_prompts=num_prompts,
                        result_dir=result_dir,
                    )

        random_workloads = [
            {
                "input_len": workload.input_len,
                "output_len": workload.output_len,
                "concurrencies": workload.concurrencies,
                "num_prompts": workload.num_prompts,
            }
            for workload in workloads
        ]

        configuration = {
            "base_url": base_url,
            "model": model,
            "dataset_name": dataset_name,
            "datasets": [dataset.name for dataset in datasets],
            "concurrencies": concurrencies,
            "num_prompts": num_prompts_values,
            "random_workloads": random_workloads,
            "endpoint_type": endpoint_type,
            "tokenizer": _optional_text(payload, "tokenizer"),
            "random_image_width": _integer(payload, "random_image_width"),
            "random_image_height": _integer(payload, "random_image_height"),
            "random_images_per_prompt": _integer_default(
                payload,
                "random_images_per_prompt",
                default=1,
            ),
            "total_runs": len(specs),
        }
        return specs, configuration

    def _trim_history(self) -> None:
        while len(self.jobs) > self.max_history:
            oldest_terminal_id = next(
                (
                    job_id
                    for job_id, job in self.jobs.items()
                    if job.status in TERMINAL_STATUSES
                ),
                None,
            )
            if oldest_terminal_id is None:
                return
            del self.jobs[oldest_terminal_id]

    async def create_job(self, payload: dict[str, Any]) -> WebJob:
        async with self._lock:
            job_id = (
                datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                + "-"
                + uuid.uuid4().hex[:6]
            )
            task_name = _optional_text(payload, "task_name")
            if task_name is not None:
                if len(task_name) > MAX_TASK_NAME_LENGTH:
                    raise ValueError(
                        f"task_name must not exceed {MAX_TASK_NAME_LENGTH} characters"
                    )
                if any(ord(character) < 32 for character in task_name):
                    raise ValueError("task_name must not contain control characters")
            name = task_name or job_id
            directory_name = (
                _task_directory_name(task_name, job_id)
                if task_name is not None
                else job_id
            )
            job_dir = self.output_root / directory_name
            run_specs, configuration = self._build_run_specs(payload, job_dir)
            configuration["task_name"] = name
            job = WebJob(
                job_id=job_id,
                name=name,
                status="queued",
                created_at=_now(),
                configuration=configuration,
                run_specs=run_specs,
                result_dir=str(job_dir),
                csv_path=str(job_dir / "matrix.csv"),
            )
            self.jobs[job_id] = job
            self._trim_history()
            job.task = asyncio.create_task(self._execute(job))
            return job

    async def _execute(self, job: WebJob) -> None:
        job.status = "running"
        job.started_at = _now()
        csv_rows: list[dict[str, Any]] = []
        total_runs = len(job.run_specs)
        try:
            for index, spec in enumerate(job.run_specs, start=1):
                run_state: dict[str, Any] = {
                    "index": index,
                    "dataset": spec.dataset.name,
                    "concurrency": spec.config.concurrency,
                    "num_prompts": spec.requested_num_prompts,
                    "status": "running",
                    "result_dir": str(spec.config.output_dir),
                }
                if spec.config.dataset_name == "random":
                    run_state.update(
                        {
                            "random_input_len": spec.config.random_input_len,
                            "random_output_len": spec.config.random_output_len,
                        }
                    )
                job.runs.append(run_state)

                def progress_callback(
                    update: ProgressUpdate,
                    *,
                    run_index: int = index,
                    run_spec: WebRunSpec = spec,
                ) -> None:
                    overall = (
                        (run_index - 1) + update.progress_percent / 100
                    ) / total_runs * 100
                    job.progress = {
                        **update.to_dict(),
                        "run_index": run_index,
                        "run_count": total_runs,
                        "dataset": run_spec.dataset.name,
                        "concurrency": run_spec.config.concurrency,
                        "overall_percent": overall,
                    }
                    if run_spec.config.dataset_name == "random":
                        job.progress.update(
                            {
                                "random_input_len": (
                                    run_spec.config.random_input_len
                                ),
                                "random_output_len": (
                                    run_spec.config.random_output_len
                                ),
                            }
                        )

                try:
                    report = await self.service.run(
                        spec.config,
                        progress_callback=progress_callback,
                        progress_interval_seconds=0.5,
                    )
                    summary_path, requests_path = save_report(
                        report,
                        spec.config.output_dir,
                    )
                    csv_rows.append(
                        report_to_csv_row(
                            spec.dataset,
                            report,
                            requested_num_prompts=spec.requested_num_prompts,
                            summary_path=summary_path,
                            requests_path=requests_path,
                        )
                    )
                    run_state.update(
                        {
                            "status": "completed",
                            "report": report.to_dict(),
                            "summary_path": str(summary_path),
                            "requests_path": str(requests_path),
                        }
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    csv_rows.append(
                        _error_row(
                            spec.dataset,
                            spec.config.concurrency,
                            spec.requested_num_prompts,
                            exc,
                        )
                    )
                    run_state.update(
                        {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                finally:
                    write_csv(csv_rows, Path(job.csv_path or "matrix.csv"))

            job.status = "completed"
            job.progress = {
                "phase": "completed",
                "completed": total_runs,
                "total": total_runs,
                "successful": sum(
                    run["status"] == "completed" for run in job.runs
                ),
                "failed": sum(run["status"] == "failed" for run in job.runs),
                "elapsed_seconds": 0,
                "request_throughput": 0,
                "eta_seconds": 0,
                "progress_percent": 100,
                "run_index": total_runs,
                "run_count": total_runs,
                "overall_percent": 100,
            }
        except asyncio.CancelledError:
            job.status = "cancelled"
            for run in job.runs:
                if run["status"] == "running":
                    run["status"] = "cancelled"
            raise
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished_at = _now()
            self._trim_history()

    def get_job(self, job_id: str) -> WebJob:
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise KeyError(f"unknown benchmark job: {job_id}") from exc

    async def cancel_job(self, job_id: str) -> WebJob:
        job = self.get_job(job_id)
        if job.status in TERMINAL_STATUSES:
            return job
        if job.task is not None:
            job.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await job.task
        return job

    async def close(self) -> None:
        tasks = [
            job.task
            for job in self.jobs.values()
            if job.task is not None and not job.task.done()
        ]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task


MANAGER_KEY = web.AppKey("manager", WebJobManager)
ASSET_ROOT = Path(__file__).with_name("web_assets")


def _manager(request: web.Request) -> WebJobManager:
    return request.app[MANAGER_KEY]


async def _index(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(ASSET_ROOT / "index.html")


async def _api_configuration(request: web.Request) -> web.Response:
    manager = _manager(request)
    return web.json_response(
        {
            "datasets": manager.available_datasets(),
            "dataset_root": str(manager.dataset_root),
            "output_root": str(manager.output_root),
        }
    )


def _job_matches_query(job: WebJob, query: str) -> bool:
    created = datetime.fromisoformat(job.created_at).astimezone()
    searchable_values = (
        job.name,
        job.job_id,
        job.created_at,
        created.strftime("%Y-%m-%d %H:%M:%S"),
        created.strftime("%Y/%m/%d %H:%M:%S"),
        f"{created.year}/{created.month}/{created.day} {created:%H:%M:%S}",
        f"{created.year}年{created.month}月{created.day}日 {created:%H:%M:%S}",
    )
    return query in " ".join(searchable_values).casefold()


async def _api_jobs(request: web.Request) -> web.Response:
    try:
        page = int(request.query.get("page", "1"))
        page_size = int(request.query.get("page_size", "10"))
    except ValueError:
        return web.json_response(
            {"error": "page and page_size must be integers"},
            status=400,
        )
    if page < 1 or page_size < 1:
        return web.json_response(
            {"error": "page and page_size must be positive"},
            status=400,
        )
    if page_size > 100:
        return web.json_response(
            {"error": "page_size must not exceed 100"},
            status=400,
        )

    manager = _manager(request)
    query = request.query.get("q", "").strip().casefold()
    jobs = list(reversed(manager.jobs.values()))
    if query:
        jobs = [job for job in jobs if _job_matches_query(job, query)]

    total = len(jobs)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    page_jobs = jobs[start : start + page_size]
    has_running_jobs = any(
        job.status in {"queued", "running"} for job in manager.jobs.values()
    )
    return web.json_response(
        {
            "jobs": [job.to_dict() for job in page_jobs],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
                "query": request.query.get("q", "").strip(),
            },
            "has_running_jobs": has_running_jobs,
        }
    )


async def _api_create_job(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        job = await _manager(request).create_job(payload)
    except json.JSONDecodeError:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(job.to_dict(), status=202)


async def _api_job(request: web.Request) -> web.Response:
    try:
        job = _manager(request).get_job(request.match_info["job_id"])
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    return web.json_response(job.to_dict())


async def _api_cancel_job(request: web.Request) -> web.Response:
    try:
        job = await _manager(request).cancel_job(request.match_info["job_id"])
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    return web.json_response(job.to_dict())


def _write_results_archive(job: WebJob) -> Path:
    archive_path = Path(job.result_dir or "") / "results.zip"
    temporary_path = archive_path.with_name(
        f".{archive_path.name}.{uuid.uuid4().hex}.tmp"
    )
    files: list[tuple[Path, str]] = []
    csv_path = Path(job.csv_path or "")
    if csv_path.is_file():
        files.append((csv_path, "matrix.csv"))
    for run in job.runs:
        directory = (
            f"{run['index']:02d}-{_slug(run['dataset'])}"
            f"-concurrency-{run['concurrency']}"
        )
        for key, filename in (
            ("summary_path", "summary.json"),
            ("requests_path", "requests.jsonl"),
        ):
            path = Path(run.get(key, ""))
            if path.is_file():
                files.append((path, f"{directory}/{filename}"))
    if not files:
        raise FileNotFoundError("result files are not ready")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for path, archive_name in files:
                archive.write(path, archive_name)
        temporary_path.replace(archive_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return archive_path


async def _api_job_file(request: web.Request) -> web.StreamResponse:
    try:
        job = _manager(request).get_job(request.match_info["job_id"])
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)

    filename = request.match_info["filename"]
    if filename == "matrix.csv":
        path = Path(job.csv_path or "")
    elif filename == "results.zip":
        if job.status not in TERMINAL_STATUSES:
            return web.json_response(
                {"error": "result archive is not ready"},
                status=409,
            )
        try:
            path = await asyncio.to_thread(_write_results_archive, job)
        except FileNotFoundError as exc:
            return web.json_response({"error": str(exc)}, status=404)
    else:
        try:
            run_index_text, name = filename.split("-", 1)
            run_index = int(run_index_text)
            run = job.runs[run_index - 1]
            key = {
                "summary.json": "summary_path",
                "requests.jsonl": "requests_path",
            }[name]
            path = Path(run[key])
        except (ValueError, IndexError, KeyError):
            return web.json_response({"error": "unknown result file"}, status=404)
    if not path.is_file():
        return web.json_response({"error": "result file is not ready"}, status=404)
    return web.FileResponse(
        path,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def _on_cleanup(app: web.Application) -> None:
    await app[MANAGER_KEY].close()


def create_web_app(
    *,
    dataset_root: Path = Path("dataset"),
    output_root: Path = Path("results/web"),
    manager: WebJobManager | None = None,
) -> web.Application:
    app = web.Application(client_max_size=1024 * 1024)
    app[MANAGER_KEY] = manager or WebJobManager(
        dataset_root=dataset_root,
        output_root=output_root,
    )
    app.router.add_get("/", _index)
    app.router.add_static("/assets/", ASSET_ROOT, show_index=False)
    app.router.add_get("/api/configuration", _api_configuration)
    app.router.add_get("/api/jobs", _api_jobs)
    app.router.add_post("/api/jobs", _api_create_job)
    app.router.add_get("/api/jobs/{job_id}", _api_job)
    app.router.add_post("/api/jobs/{job_id}/cancel", _api_cancel_job)
    app.router.add_get(
        "/api/jobs/{job_id}/files/{filename}",
        _api_job_file,
    )
    app.on_cleanup.append(_on_cleanup)
    return app


def run_web_server(
    *,
    host: str,
    port: int,
    dataset_root: Path,
    output_root: Path,
) -> None:
    app = create_web_app(
        dataset_root=dataset_root,
        output_root=output_root,
    )
    print(f"Speco-Bench web console: http://{host}:{port}")
    web.run_app(app, host=host, port=port, print=None)
