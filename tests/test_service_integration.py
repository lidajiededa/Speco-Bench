import json
import tempfile
import unittest
from pathlib import Path

from aiohttp import web

from speco_bench.config import BenchmarkConfig
from speco_bench.service import BenchmarkService


class ServiceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.drafts = 0
        self.draft_tokens = 0
        self.accepted_tokens = 0

        async def completions(request):
            payload = await request.json()
            self.assertTrue(payload["stream"])
            self.assertTrue(payload["stream_options"]["include_usage"])
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream"},
            )
            await response.prepare(request)
            events = [
                {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                },
            ]
            for event in events:
                await response.write(f"data: {json.dumps(event)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            self.drafts += 1
            self.draft_tokens += 2
            self.accepted_tokens += 1
            return response

        async def metrics(_request):
            return web.Response(
                text=(
                    f"vllm:spec_decode_num_drafts_total {self.drafts}\n"
                    f"vllm:spec_decode_num_draft_tokens_total {self.draft_tokens}\n"
                    f"vllm:spec_decode_num_accepted_tokens_total {self.accepted_tokens}\n"
                    "vllm:spec_decode_num_accepted_tokens_per_pos"
                    f'{{position="0"}} {self.accepted_tokens}\n'
                )
            )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", completions)
        app.router.add_get("/metrics", metrics)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        sockets = self.site._server.sockets
        self.base_url = f"http://127.0.0.1:{sockets[0].getsockname()[1]}"

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_end_to_end_service(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "data.jsonl"
            dataset.write_text(
                '{"messages":[{"role":"user","content":"hi"}],"max_tokens":2}\n',
                encoding="utf-8",
            )
            config = BenchmarkConfig(
                base_url=self.base_url,
                model="mock-model",
                dataset_path=dataset,
                output_dir=Path(directory) / "results",
                concurrency=2,
                num_prompts=3,
                warmup_requests=0,
            )
            report = await BenchmarkService().run(config)
            self.assertEqual(report.summary["successful_requests"], 3)
            self.assertEqual(report.summary["total_input_tokens"], 12)
            self.assertEqual(report.summary["total_output_tokens"], 6)
            self.assertTrue(report.spec_decode.available)
            self.assertEqual(report.spec_decode.num_drafts, 3)
            self.assertEqual(report.spec_decode.mean_acceptance_length, 2.0)
            self.assertEqual(report.spec_decode.position_acceptance_rates, [1.0])
            self.assertFalse(report.warnings)


if __name__ == "__main__":
    unittest.main()

