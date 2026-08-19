import asyncio
import io
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from speco_bench.models import (
    BenchmarkReport,
    ProgressUpdate,
    RequestResult,
    SpecDecodeStats,
)
from speco_bench.stats import summarize_requests
from speco_bench.web import WebJobManager, create_web_app


class FakeBenchmarkService:
    def __init__(self):
        self.configs = []

    async def run(
        self,
        config,
        *,
        progress_callback=None,
        progress_interval_seconds=1.0,
    ):
        self.configs.append(config)
        if progress_callback is not None:
            progress_callback(
                ProgressUpdate(
                    phase="benchmark",
                    completed=1,
                    total=1,
                    successful=1,
                    failed=0,
                    elapsed_seconds=0.1,
                    request_throughput=10.0,
                    eta_seconds=0.0,
                )
            )
        await asyncio.sleep(0)
        result = RequestResult(
            request_id=0,
            success=True,
            input_tokens=8,
            output_tokens=4,
            ttft_ms=10,
            tpot_ms=3,
            e2e_ms=19,
            generated_text="done",
        )
        summary = summarize_requests(
            [result],
            duration_seconds=0.1,
            dataset=str(config.dataset_path or "random"),
            concurrency=config.concurrency,
        )
        return BenchmarkReport(
            summary=summary,
            requests=[result],
            spec_decode=SpecDecodeStats(available=False),
        )


class BlockingBenchmarkService:
    async def run(
        self,
        config,
        *,
        progress_callback=None,
        progress_interval_seconds=1.0,
    ):
        await asyncio.Event().wait()


class WebJobManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "dataset"
        for name in ("gsm8k", "math500"):
            directory = self.dataset_root / name
            directory.mkdir(parents=True)
            (directory / "question.jsonl").write_text(
                '{"prompt":"hello","max_tokens":4}\n',
                encoding="utf-8",
            )
        self.service = FakeBenchmarkService()
        self.manager = WebJobManager(
            dataset_root=self.dataset_root,
            output_root=self.root / "results",
            service=self.service,
        )

    async def asyncTearDown(self):
        await self.manager.close()
        self.temporary.cleanup()

    async def test_random_mode_separates_served_model_and_tokenizer_path(self):
        job = await self.manager.create_job(
            {
                "base_url": "http://localhost:8000",
                "model": "served-name",
                "dataset_name": "random",
                "tokenizer": "/models/local-model",
                "concurrencies": ["2"],
                "num_prompts": ["5"],
                "random_input_len": 128,
                "random_output_len": 32,
                "random_image_width": 640,
                "random_image_height": 480,
                "random_images_per_prompt": 2,
            }
        )
        await job.task

        config = self.service.configs[0]
        self.assertEqual(config.model, "served-name")
        self.assertEqual(config.tokenizer, "/models/local-model")
        self.assertIsNone(config.dataset_path)
        self.assertEqual(config.num_prompts, 5)
        self.assertEqual(config.random_image_width, 640)
        self.assertEqual(config.random_image_height, 480)
        self.assertEqual(config.random_images_per_prompt, 2)

    async def test_runs_dataset_concurrency_matrix_and_writes_csv(self):
        job = await self.manager.create_job(
            {
                "base_url": "http://localhost:8000",
                "model": "model",
                "dataset_name": "custom",
                "datasets": ["gsm8k", "math500"],
                "concurrencies": ["1", "4"],
                "num_prompts": ["2", "8"],
                "max_tokens": 16,
            }
        )
        await job.task

        self.assertEqual(job.status, "completed")
        self.assertEqual(len(job.runs), 4)
        self.assertEqual(len(self.service.configs), 4)
        self.assertTrue(Path(job.csv_path).is_file())
        self.assertTrue(
            all(run["status"] == "completed" for run in job.runs)
        )
        self.assertEqual(job.progress["overall_percent"], 100)

    async def test_custom_task_name_is_exposed_and_used_in_result_path(self):
        job = await self.manager.create_job(
            {
                "task_name": "Qwen VL 基线 / 8并发",
                "base_url": "http://localhost:8000",
                "model": "model",
                "dataset_name": "custom",
                "datasets": ["gsm8k"],
                "concurrencies": ["1"],
                "num_prompts": ["1"],
            }
        )
        await job.task

        self.assertEqual(job.name, "Qwen VL 基线 / 8并发")
        self.assertEqual(job.to_dict()["name"], job.name)
        self.assertEqual(job.configuration["task_name"], job.name)
        self.assertTrue(
            Path(job.result_dir).name.startswith("Qwen-VL-基线-8并发-")
        )

    async def test_blank_task_name_falls_back_to_job_id(self):
        job = await self.manager.create_job(
            {
                "task_name": "   ",
                "base_url": "http://localhost:8000",
                "model": "model",
                "dataset_name": "custom",
                "datasets": ["gsm8k"],
                "concurrencies": ["1"],
                "num_prompts": ["1"],
            }
        )
        await job.task

        self.assertEqual(job.name, job.job_id)
        self.assertEqual(Path(job.result_dir).name, job.job_id)

    async def test_cancels_active_job(self):
        manager = WebJobManager(
            dataset_root=self.dataset_root,
            output_root=self.root / "cancel-results",
            service=BlockingBenchmarkService(),
        )
        job = await manager.create_job(
            {
                "base_url": "http://localhost:8000",
                "model": "model",
                "dataset_name": "custom",
                "datasets": ["gsm8k"],
                "concurrencies": ["1"],
                "num_prompts": ["1"],
            }
        )
        await asyncio.sleep(0)

        await manager.cancel_job(job.job_id)

        self.assertEqual(job.status, "cancelled")
        self.assertEqual(job.runs[0]["status"], "cancelled")
        await manager.close()

    async def test_runs_multiple_web_jobs_concurrently(self):
        manager = WebJobManager(
            dataset_root=self.dataset_root,
            output_root=self.root / "parallel-results",
            service=BlockingBenchmarkService(),
        )
        payload = {
            "base_url": "http://localhost:8000",
            "model": "model",
            "dataset_name": "custom",
            "datasets": ["gsm8k"],
            "concurrencies": ["1"],
            "num_prompts": ["1"],
        }

        first = await manager.create_job({**payload, "task_name": "first"})
        second = await manager.create_job({**payload, "task_name": "second"})
        await asyncio.sleep(0)

        self.assertEqual(first.status, "running")
        self.assertEqual(second.status, "running")
        self.assertEqual(len(first.runs), 1)
        self.assertEqual(len(second.runs), 1)
        self.assertNotEqual(first.result_dir, second.result_dir)
        await manager.cancel_job(first.job_id)
        await manager.cancel_job(second.job_id)
        await manager.close()


class WebApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        dataset = root / "dataset" / "gsm8k"
        dataset.mkdir(parents=True)
        (dataset / "question.jsonl").write_text(
            '{"prompt":"hello","max_tokens":4}\n',
            encoding="utf-8",
        )
        self.manager = WebJobManager(
            dataset_root=root / "dataset",
            output_root=root / "results",
            service=FakeBenchmarkService(),
        )
        self.client = TestClient(
            TestServer(create_web_app(manager=self.manager))
        )
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.temporary.cleanup()

    async def test_serves_console_and_configuration(self):
        response = await self.client.get("/")
        self.assertEqual(response.status, 200)
        page = await response.text()
        self.assertIn("Speco-Bench", page)
        self.assertIn('name="task_name"', page)
        self.assertIn('id="historySearch"', page)

        response = await self.client.get("/api/configuration")
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(payload["datasets"][0]["name"], "gsm8k")

    async def test_rejects_unpaired_concurrency_and_prompt_counts(self):
        response = await self.client.post(
            "/api/jobs",
            json={
                "base_url": "http://localhost:8000",
                "model": "model",
                "dataset_name": "custom",
                "datasets": ["gsm8k"],
                "concurrencies": ["1", "4"],
                "num_prompts": ["10"],
            },
        )

        self.assertEqual(response.status, 400)
        payload = await response.json()
        self.assertIn("exactly one value per concurrency", payload["error"])

    async def test_rejects_task_name_over_length_limit(self):
        response = await self.client.post(
            "/api/jobs",
            json={
                "task_name": "x" * 81,
                "base_url": "http://localhost:8000",
                "model": "model",
                "dataset_name": "custom",
                "datasets": ["gsm8k"],
                "concurrencies": ["1"],
                "num_prompts": ["1"],
            },
        )

        self.assertEqual(response.status, 400)
        payload = await response.json()
        self.assertIn("must not exceed 80 characters", payload["error"])

    async def test_paginates_and_searches_run_history(self):
        jobs = []
        for index in range(12):
            response = await self.client.post(
                "/api/jobs",
                json={
                    "task_name": f"Baseline {index:02d}",
                    "base_url": "http://localhost:8000",
                    "model": "model",
                    "dataset_name": "custom",
                    "datasets": ["gsm8k"],
                    "concurrencies": ["1"],
                    "num_prompts": ["1"],
                },
            )
            payload = await response.json()
            job = self.manager.get_job(payload["id"])
            await job.task
            jobs.append(job)

        response = await self.client.get(
            "/api/jobs", params={"page": "2", "page_size": "5"}
        )
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(payload["pagination"]["total"], 12)
        self.assertEqual(payload["pagination"]["total_pages"], 3)
        self.assertEqual(payload["pagination"]["page"], 2)
        self.assertEqual(
            [job["name"] for job in payload["jobs"]],
            [
                "Baseline 06",
                "Baseline 05",
                "Baseline 04",
                "Baseline 03",
                "Baseline 02",
            ],
        )
        self.assertFalse(payload["has_running_jobs"])

        response = await self.client.get(
            "/api/jobs", params={"q": "BASELINE 1"}
        )
        payload = await response.json()
        self.assertEqual(payload["pagination"]["total"], 2)
        self.assertEqual(
            [job["name"] for job in payload["jobs"]],
            ["Baseline 11", "Baseline 10"],
        )

        jobs[0].created_at = "2024-03-05T01:02:03+00:00"
        created = datetime.fromisoformat(jobs[0].created_at).astimezone()
        displayed_time = (
            f"{created.year}/{created.month}/{created.day} "
            f"{created.hour:02d}:{created.minute:02d}"
        )
        response = await self.client.get(
            "/api/jobs", params={"q": displayed_time}
        )
        payload = await response.json()
        self.assertEqual(payload["pagination"]["total"], 1)
        self.assertEqual(payload["jobs"][0]["name"], "Baseline 00")

        response = await self.client.get(
            "/api/jobs", params={"page": "99", "page_size": "5"}
        )
        payload = await response.json()
        self.assertEqual(payload["pagination"]["page"], 3)
        self.assertEqual(len(payload["jobs"]), 2)

        response = await self.client.get("/api/jobs", params={"page": "0"})
        self.assertEqual(response.status, 400)

    async def test_downloads_request_results_and_complete_archive(self):
        response = await self.client.post(
            "/api/jobs",
            json={
                "task_name": "export",
                "base_url": "http://localhost:8000",
                "model": "model",
                "dataset_name": "custom",
                "datasets": ["gsm8k"],
                "concurrencies": ["1"],
                "num_prompts": ["1"],
            },
        )
        job_payload = await response.json()
        job = self.manager.get_job(job_payload["id"])
        await job.task

        response = await self.client.get(
            f"/api/jobs/{job.job_id}/files/1-requests.jsonl"
        )
        self.assertEqual(response.status, 200)
        self.assertIn('"generated_text": "done"', await response.text())

        response = await self.client.get(
            f"/api/jobs/{job.job_id}/files/results.zip"
        )
        self.assertEqual(response.status, 200)
        with zipfile.ZipFile(io.BytesIO(await response.read())) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {
                    "matrix.csv",
                    "01-gsm8k-concurrency-1/summary.json",
                    "01-gsm8k-concurrency-1/requests.jsonl",
                },
            )


if __name__ == "__main__":
    unittest.main()
