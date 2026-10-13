#!/usr/bin/env python3
"""CPU-only contract test for HarmoRAG's AsyncLLMEngine bridge."""

from __future__ import annotations

import asyncio
import itertools
import queue
import threading
import time
import unittest
from types import SimpleNamespace

from harmorag.generator.generator import VLLMGenerator


class _FakeAsyncEngine:
    def __init__(self):
        self.aborted = []
        self.shutdown_called = False

    async def generate(
        self,
        prompt,
        sampling_params,
        request_id,
        *,
        lora_request=None,
        priority=0,
    ):
        delay = 0.25 if str(prompt) == "slow" else 0.01
        await asyncio.sleep(delay)
        if request_id in self.aborted:
            return
        choice = SimpleNamespace(
            text=f"done:{prompt}",
            token_ids=[7],
            logprobs=None,
            finish_reason="stop",
            stop_reason=None,
        )
        yield SimpleNamespace(
            request_id=request_id,
            outputs=[choice],
            finished=True,
            prompt_token_ids=[1, 2],
            num_cached_tokens=0,
            metrics=None,
        )

    async def abort(self, request_ids):
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        self.aborted.extend(str(value) for value in request_ids)

    def shutdown(self):
        self.shutdown_called = True


def _loop_worker(loop, ready):
    asyncio.set_event_loop(loop)
    ready.set()
    loop.run_forever()
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    if pending:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()


def make_generator():
    generator = VLLMGenerator.__new__(VLLMGenerator)
    generator._config = {"seed": 20260904}
    generator.generation_params = {"temperature": 0, "max_tokens": 4}
    generator.use_lora = False
    generator._closed = False
    generator.model = _FakeAsyncEngine()
    generator._incremental_request_meta = {}
    generator._incremental_futures = {}
    generator._incremental_completed = queue.SimpleQueue()
    generator._incremental_lock = threading.RLock()
    generator._request_serial = itertools.count()
    generator._engine_loop = asyncio.new_event_loop()
    ready = threading.Event()
    generator._engine_thread = threading.Thread(
        target=_loop_worker,
        args=(generator._engine_loop, ready),
        daemon=True,
    )
    generator._engine_thread.start()
    ready.wait(timeout=5)
    return generator


class AsyncGeneratorAdapterContractTest(unittest.TestCase):
    def test_completion_event_wakes_without_busy_polling(self):
        generator = make_generator()
        try:
            generator.add_requests(["first"])
            started = time.perf_counter()
            self.assertTrue(generator.wait_for_request_completion(1.0))
            elapsed = time.perf_counter() - started
            self.assertGreater(elapsed, 0.005)
            self.assertLess(elapsed, 0.25)
            completed = generator.step_requests()
            self.assertEqual([row["text"] for row in completed], ["done:first"])
        finally:
            generator.close()

    def test_admission_is_nonblocking_and_completions_are_polled(self):
        generator = make_generator()
        try:
            started = time.perf_counter()
            request_ids = generator.add_requests(
                ["first", "second"],
                request_metadata=[{"index": 0}, {"index": 1}],
            )
            self.assertLess(time.perf_counter() - started, 0.1)
            self.assertEqual(len(request_ids), 2)
            self.assertTrue(all(request_id.isdigit() for request_id in request_ids))
            self.assertTrue(generator.has_unfinished_requests())

            # Engine requests may already be retired while their completions
            # are waiting for the synchronous pipeline to poll them.
            time.sleep(0.05)
            self.assertTrue(generator.has_unfinished_requests())
            completed = []
            deadline = time.perf_counter() + 3
            while len(completed) < 2 and time.perf_counter() < deadline:
                completed.extend(generator.step_requests())
                time.sleep(0.005)
            self.assertEqual({row["text"] for row in completed}, {"done:first", "done:second"})
            self.assertEqual(
                {row["metadata"]["index"] for row in completed}, {0, 1}
            )
            self.assertFalse(generator.has_unfinished_requests())
        finally:
            generator.close()
        self.assertTrue(generator.model.shutdown_called)

    def test_abort_retires_engine_and_adapter_state(self):
        generator = make_generator()
        try:
            request_id = generator.add_requests(["slow"])[0]
            generator.abort_requests([request_id])
            self.assertIn(request_id, generator.model.aborted)
            self.assertFalse(generator.has_unfinished_requests())
            time.sleep(0.02)
            self.assertEqual(generator.step_requests(), [])
        finally:
            generator.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
