import asyncio
import atexit
import itertools
import math
import os
import queue
import threading
import time
from copy import deepcopy
from typing import Iterable, List, Optional, Sequence

import numpy as np
from transformers import AutoTokenizer

from harmorag.generator.utils import resolve_max_tokens


class BaseGenerator:
    """`BaseGenerator` is a base object of Generator model."""

    def __init__(self, config):
        self._config = config
        self.update_config()

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config_data):
        self._config = config_data
        self.update_config()

    def update_config(self):
        self.update_base_setting()
        self.update_additional_setting()
    def update_base_setting(self):
        self.model_name = self._config["generator_model"]
        self.model_path = self._config["generator_model_path"]

        self.max_input_len = self._config["generator_max_input_len"]
        self.batch_size = self._config["generator_batch_size"]
        self.device = self._config["device"]
        self.gpu_num = self._config['gpu_num']
        self.generation_params = self._config["generation_params"]

    def update_additional_setting(self):
        pass

    def generate(self, input_list: list) -> List[str]:
        """Get responses from the generater.

        Args:
            input_list: it contains input texts, each item represents a sample.

        Returns:
            list: contains generator's response of each input sample.
        """
        pass


class VLLMGenerator(BaseGenerator):
    """Decoder-only generator backed by vLLM ``AsyncLLMEngine``.

    The adapter preserves a synchronous compatibility API while exposing the
    non-blocking admission, completion polling, and abort operations required
    by HarmoRAG. A dedicated event loop owns one ``AsyncLLMEngine``; all live
    HarmoRAG generation chains therefore share vLLM's continuous-batching
    scheduler without private engine stepping.
    """

    def __init__(self, config):
        super().__init__(config)

        kv_free_block_policy = str(
            self._config["vllm_kv_free_block_policy"]
            if "vllm_kv_free_block_policy" in self._config
            else os.environ.get("VLLM_KV_FREE_BLOCK_POLICY", "")
        ).strip().lower()
        if kv_free_block_policy:
            if kv_free_block_policy not in {"heat", "lru"}:
                raise ValueError(
                    "vllm_kv_free_block_policy must be 'heat' or 'lru'"
                )
            # Read by the server's optional free-block queue at engine
            # construction time. Upstream vLLM safely ignores this variable.
            os.environ["VLLM_KV_FREE_BLOCK_POLICY"] = kv_free_block_policy

        engine_kwargs = {
            "model": self.model_path,
            "trust_remote_code": True,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_logprobs": 32016,
            "max_model_len": self.max_model_len,
            "enable_prefix_caching": self.enable_prefix_caching,
            # The server's vLLM build reads per-request metrics in its offline
            # runner; disabling log stats leaves those metrics as None.
            "disable_log_stats": False,
        }
        engine_kwargs.update(self.vllm_engine_kwargs)
        if self.use_lora:
            engine_kwargs.update(enable_lora=True, max_lora_rank=64)

        self.serving_engine = "vllm_AsyncLLMEngine"
        self._engine_kwargs = dict(engine_kwargs)
        self._engine_loop = asyncio.new_event_loop()
        self._engine_ready = threading.Event()
        self._engine_start_error = None
        self._closed = False
        self.model = None
        self._engine_thread = threading.Thread(
            target=self._run_engine_loop,
            name="harmorag-vllm-async-engine",
            daemon=True,
        )
        self._engine_thread.start()
        self._engine_ready.wait()
        if self._engine_start_error is not None:
            raise RuntimeError("failed to construct vLLM AsyncLLMEngine") from self._engine_start_error

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._incremental_request_meta = {}
        self._incremental_futures = {}
        self._incremental_completed = queue.SimpleQueue()
        # Wake the synchronous HarmoRAG coordinator when the AsyncLLMEngine
        # event-loop thread publishes a semantic completion. Without this
        # edge-triggered signal the coordinator has to busy-poll the queue,
        # which can starve retrieval and engine bookkeeping at low load.
        self._incremental_completion_event = threading.Event()
        self._incremental_external_wakeup_event = None
        self._incremental_lock = threading.RLock()
        self._request_serial = itertools.count()
        atexit.register(self.close)

    def _run_engine_loop(self):
        """Construct and own AsyncLLMEngine on one persistent event loop."""

        asyncio.set_event_loop(self._engine_loop)
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine

            self.model = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(**self._engine_kwargs)
            )
        except BaseException as error:
            self._engine_start_error = error
            self._engine_ready.set()
            self._engine_loop.close()
            return
        self._engine_ready.set()
        try:
            self._engine_loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._engine_loop)
            for task in pending:
                task.cancel()
            if pending:
                self._engine_loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._engine_loop.run_until_complete(
                self._engine_loop.shutdown_asyncgens()
            )
            self._engine_loop.close()

    def _submit(self, coroutine):
        if self._closed or not self._engine_thread.is_alive():
            raise RuntimeError("vLLM AsyncLLMEngine is not running")
        return asyncio.run_coroutine_threadsafe(coroutine, self._engine_loop)

    def _new_request_id(self, kind):
        # Upstream AsyncLLMEngine accepts arbitrary strings.  The deployed
        # vLLM free-block policy additionally orders cache entries by
        # ``int(request.request_id)``; use an engine-local numeric ID so the
        # public API and that audited server extension are both satisfied.
        del kind
        return str(next(self._request_serial))

    def _lora_request(self):
        if not self.use_lora:
            return None
        from vllm.lora.request import LoRARequest

        return LoRARequest("lora_module", 1, self.lora_path)

    async def _collect_one(self, prompt, sampling_params, request_id, lora_request):
        final_output = None
        try:
            async for output in self.model.generate(
                prompt,
                sampling_params,
                request_id,
                lora_request=lora_request,
            ):
                final_output = output
        except asyncio.CancelledError:
            # Cancelling the Python consumer alone does not guarantee that the
            # engine-side sequence is retired.  Explicitly abort it so a failed
            # sibling in blocking ``generate`` cannot leave hidden GPU work.
            await self.model.abort(request_id)
            raise
        if final_output is None:
            raise RuntimeError(f"AsyncLLMEngine produced no output for {request_id}")
        return final_output

    async def _collect_many(self, prompts, sampling_params, lora_request):
        request_ids = [self._new_request_id("blocking") for _ in prompts]
        tasks = [
            asyncio.create_task(
                self._collect_one(
                    prompt,
                    deepcopy(sampling_params),
                    request_id,
                    lora_request,
                )
            )
            for prompt, request_id in zip(prompts, request_ids)
        ]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # ``abort`` is idempotent for already-finished request IDs in the
            # pinned vLLM runtime and closes the small race between a consumer
            # exception and its cancellation handler.
            await self.model.abort(request_ids)
            raise

    async def _consume_incremental(
        self,
        prompt,
        sampling_params,
        request_id,
        priority,
        lora_request,
    ):
        """Consume one streaming request and publish only semantic completions."""

        try:
            async for output in self.model.generate(
                prompt,
                sampling_params,
                request_id,
                lora_request=lora_request,
                priority=priority,
            ):
                with self._incremental_lock:
                    meta = self._incremental_request_meta.get(request_id)
                    if meta is None:
                        return
                    choice = output.outputs[0] if output.outputs else None
                    finished = bool(getattr(output, "finished", False))
                    gate_abort_margin = meta.get("gate_abort_margin")
                    gate_token_ids = meta.get("gate_token_ids")
                    metadata = meta.get("metadata", {})

                    # Preserve the first-token event even when vLLM is using
                    # FINAL_ONLY output mode.  vLLM's request metrics use a
                    # monotonic clock, so convert the first-token delta onto
                    # the Unix-ns service interval anchored at admission.
                    if meta.get("first_token_ns") is None:
                        metrics = getattr(output, "metrics", None)
                        first_token_time = getattr(
                            metrics, "first_token_time", None
                        )
                        arrival_time = getattr(metrics, "arrival_time", None)
                        if (
                            first_token_time is not None
                            and arrival_time is not None
                            and float(first_token_time) >= float(arrival_time)
                        ):
                            meta["first_token_ns"] = int(
                                int(meta["service_started_ns"])
                                + round(
                                    (float(first_token_time) - float(arrival_time))
                                    * 1e9
                                )
                            )
                        elif (
                            choice is not None
                            and list(getattr(choice, "token_ids", ()) or ())
                        ):
                            meta["first_token_ns"] = time.time_ns()

                    low_confidence_answer = False
                    if not finished and not meta.get("gate_checked"):
                        gate = self._extract_gate_stats(choice, gate_token_ids)
                        token_ids = (
                            list(getattr(choice, "token_ids", ()) or ())
                            if choice is not None
                            else []
                        )
                        if token_ids:
                            meta["gate_checked"] = True
                        answer_token_ids = {
                            int(value)
                            for value in (gate_token_ids or {}).get("answer", ())
                        }
                        low_confidence_answer = bool(
                            gate_abort_margin is not None
                            and str(metadata.get("kind")) == "SEARCHING"
                            and gate is not None
                            and gate.get("complete")
                            and gate.get("logprob_margin") is not None
                            and token_ids
                            and int(token_ids[0]) in answer_token_ids
                            and float(gate["logprob_margin"])
                            < float(gate_abort_margin)
                        )

                if low_confidence_answer:
                    await self.model.abort(request_id)
                    with self._incremental_lock:
                        meta = self._incremental_request_meta.get(request_id, meta)
                    cache_stats = self._cache_stats(output, gate_token_ids)
                    cache_stats["finish_reason"] = "gate_confidence_abort"
                    cache_stats["gate_aborted_below_margin"] = True
                    self._publish_incremental_completion(
                        {
                            "request_id": request_id,
                            "text": (
                                getattr(choice, "text", "A")
                                if choice is not None
                                else "A"
                            ),
                            "cache_stats": cache_stats,
                            "metadata": meta.get("metadata", {}),
                            "service_interval_ns": (
                                int(meta["service_started_ns"]),
                                time.time_ns(),
                            ),
                            "first_token_ns": meta.get("first_token_ns"),
                            "raw_output": output,
                        }
                    )
                    with self._incremental_lock:
                        self._incremental_request_meta.pop(request_id, None)
                    return

                if not finished:
                    continue
                with self._incremental_lock:
                    meta = self._incremental_request_meta.get(request_id, meta)
                self._publish_incremental_completion(
                    {
                        "request_id": request_id,
                        "text": (
                            getattr(choice, "text", "")
                            if choice is not None
                            else ""
                        ),
                        "cache_stats": self._cache_stats(
                            output, meta.get("gate_token_ids")
                        ),
                        "metadata": meta.get("metadata", {}),
                        "service_interval_ns": (
                            int(meta["service_started_ns"]),
                            time.time_ns(),
                        ),
                        "first_token_ns": meta.get("first_token_ns"),
                        "raw_output": output,
                    }
                )
                with self._incremental_lock:
                    self._incremental_request_meta.pop(request_id, None)
                return
            raise RuntimeError(
                f"AsyncLLMEngine stream ended without a terminal output for "
                f"{request_id}"
            )
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            with self._incremental_lock:
                meta = self._incremental_request_meta.get(request_id)
            if meta is not None:
                self._publish_incremental_completion(
                    {
                        "request_id": request_id,
                        "error": error,
                        "service_interval_ns": (
                            int(meta["service_started_ns"]),
                            time.time_ns(),
                        ),
                    }
                )
                with self._incremental_lock:
                    self._incremental_request_meta.pop(request_id, None)

    def _publish_incremental_completion(self, item) -> None:
        """Publish one completion and wake the synchronous coordinator."""

        self._incremental_completed.put(item)
        self._get_incremental_completion_event().set()
        external_event = getattr(
            self, "_incremental_external_wakeup_event", None
        )
        if external_event is not None:
            external_event.set()

    def _get_incremental_completion_event(self) -> threading.Event:
        """Return the completion event, including for legacy test adapters."""

        event = getattr(self, "_incremental_completion_event", None)
        if event is None:
            event = threading.Event()
            self._incremental_completion_event = event
        return event

    def _forget_future(self, request_id):
        def callback(_future):
            with self._incremental_lock:
                self._incremental_futures.pop(request_id, None)

        return callback

    async def _abort_many(self, request_ids):
        if request_ids:
            await self.model.abort(request_ids)

    def close(self):
        """Stop requests and release vLLM worker processes deterministically."""

        if self._closed:
            return
        self._closed = True
        if not getattr(self, "_engine_thread", None) or not self._engine_thread.is_alive():
            return
        with self._incremental_lock:
            request_ids = list(self._incremental_request_meta)
            futures = list(self._incremental_futures.values())
            self._incremental_request_meta.clear()
            self._incremental_futures.clear()
        if request_ids:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._abort_many(request_ids), self._engine_loop
                ).result(timeout=30)
            except BaseException:
                pass
        for future in futures:
            future.cancel()

        def shutdown():
            try:
                if self.model is not None:
                    self.model.shutdown()
            finally:
                self._engine_loop.stop()

        try:
            self._engine_loop.call_soon_threadsafe(shutdown)
            self._engine_thread.join(timeout=60)
        except BaseException:
            pass

    def update_additional_setting(self):
        if "gpu_memory_utilization" not in self._config:
            self.gpu_memory_utilization = 0.85
        else:
            self.gpu_memory_utilization = self._config["gpu_memory_utilization"]
        generator_gpu_num = self._config["generator_gpu_num"] if "generator_gpu_num" in self._config else None
        generator_gpu_num = self.gpu_num if generator_gpu_num is None else int(generator_gpu_num)
        if generator_gpu_num < 1:
            raise ValueError("generator_gpu_num must be positive for the vLLM framework")
        if generator_gpu_num != 1 and generator_gpu_num % 2 != 0:
            self.tensor_parallel_size = generator_gpu_num - 1
        else:
            self.tensor_parallel_size = generator_gpu_num

        self.lora_path = None if "generator_lora_path" not in self._config else self._config["generator_lora_path"]
        self.use_lora = False
        if self.lora_path is not None:
            self.use_lora = True
        self.max_model_len = self._config['generator_max_input_len']
        self.enable_prefix_caching = bool(
            self._config["enable_prefix_caching"]
            if "enable_prefix_caching" in self._config
            else False
        )
        engine_kwargs = (
            self._config["vllm_engine_kwargs"]
            if "vllm_engine_kwargs" in self._config
            else None
        )
        self.vllm_engine_kwargs = dict(engine_kwargs or {})

    def _prepare_sampling_params(
        self, params, *, return_scores=False, gate_logprobs=0
    ):
        """Build identical SamplingParams for blocking and stepped decode."""

        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            do_sample_flag = generation_params.pop("do_sample")
            if not do_sample_flag:
                generation_params["temperature"] = 0
        generation_params["seed"] = self._config["seed"]
        generation_params = resolve_max_tokens(
            params, generation_params, prioritize_new_tokens=False
        )

        stop = generation_params.get("stop")
        if stop is None:
            stop = []
        elif isinstance(stop, str):
            stop = [stop]
        else:
            stop = list(stop)
        if "<|eot_id|>" not in stop:
            stop.append("<|eot_id|>")
        generation_params["stop"] = stop
        generation_params.setdefault("include_stop_str_in_output", False)

        requested_logprobs = int(gate_logprobs or 0)
        if return_scores:
            requested_logprobs = max(requested_logprobs, 100)
        if requested_logprobs:
            generation_params["logprobs"] = max(
                int(generation_params.get("logprobs") or 0),
                requested_logprobs,
            )

        structured = generation_params.get("structured_outputs")
        if isinstance(structured, dict):
            generation_params["structured_outputs"] = StructuredOutputsParams(
                **structured
            )
        return SamplingParams(**generation_params)

    @staticmethod
    def _logsumexp(values):
        finite = [float(value) for value in values if value is not None]
        if not finite:
            return None
        peak = max(finite)
        return peak + math.log(sum(math.exp(value - peak) for value in finite))

    def _extract_gate_stats(self, choice, gate_token_ids):
        """Return compact A/N first-token logprobs without serializing top-k."""

        if not gate_token_ids or choice is None:
            return None
        token_ids = list(getattr(choice, "token_ids", ()) or ())
        choice_logprobs = getattr(choice, "logprobs", None)
        first_scores = choice_logprobs[0] if choice_logprobs else None
        if not first_scores:
            return {
                "first_token_id": token_ids[0] if token_ids else None,
                "a_logprob": None,
                "n_logprob": None,
                "logprob_margin": None,
                "answer_probability": None,
                "complete": False,
            }

        normalized = {}
        for raw_id, entry in first_scores.items():
            try:
                normalized[int(raw_id)] = entry
            except (TypeError, ValueError):
                continue

        def group_stats(raw_ids):
            entries = [normalized.get(int(token_id)) for token_id in raw_ids]
            entries = [entry for entry in entries if entry is not None]
            value = self._logsumexp(
                [getattr(entry, "logprob", None) for entry in entries]
            )
            ranks = [
                int(getattr(entry, "rank"))
                for entry in entries
                if getattr(entry, "rank", None) is not None
            ]
            return value, min(ranks) if ranks else None

        a_logprob, a_rank = group_stats(gate_token_ids.get("answer", ()))
        n_logprob, n_rank = group_stats(gate_token_ids.get("no", ()))
        complete = a_logprob is not None and n_logprob is not None
        margin = a_logprob - n_logprob if complete else None
        if margin is None:
            probability = None
        elif margin >= 0:
            probability = 1.0 / (1.0 + math.exp(-min(margin, 80.0)))
        else:
            exp_margin = math.exp(max(margin, -80.0))
            probability = exp_margin / (1.0 + exp_margin)
        first_token_id = token_ids[0] if token_ids else None
        return {
            "first_token_id": int(first_token_id)
            if first_token_id is not None
            else None,
            "a_logprob": a_logprob,
            "n_logprob": n_logprob,
            "a_rank": a_rank,
            "n_rank": n_rank,
            "logprob_margin": margin,
            "answer_probability": probability,
            "complete": complete,
        }

    def _cache_stats(self, output, gate_token_ids=None):
        first_choice = output.outputs[0] if output.outputs else None
        cached_tokens = getattr(output, "num_cached_tokens", None)
        if cached_tokens is None and getattr(output, "metrics", None) is not None:
            cached_tokens = getattr(output.metrics, "num_cached_tokens", None)
        prompt_token_ids = getattr(output, "prompt_token_ids", None)
        row = {
            "request_id": getattr(output, "request_id", None),
            "prompt_tokens": len(prompt_token_ids)
            if prompt_token_ids is not None
            else None,
            "num_cached_tokens": cached_tokens,
            "generated_tokens": (
                len(getattr(first_choice, "token_ids", ()) or ())
                if first_choice is not None
                else 0
            ),
            "finish_reason": getattr(first_choice, "finish_reason", None),
            "stop_reason": getattr(first_choice, "stop_reason", None),
        }
        gate = self._extract_gate_stats(first_choice, gate_token_ids)
        if gate is not None:
            row["answerability_gate"] = gate
        return row

    def generate(
        self,
        input_list: List,
        return_raw_output=False,
        return_scores=False,
        return_cache_stats=False,
        return_gate_logprobs=False,
        gate_token_ids=None,
        gate_logprobs=0,
        **params,
    ):
        if isinstance(input_list, str):
            input_list = [input_list]
        sampling_params = self._prepare_sampling_params(
            params,
            return_scores=return_scores,
            gate_logprobs=(gate_logprobs if return_gate_logprobs else 0),
        )
        outputs = self._submit(
            self._collect_many(
                list(input_list), sampling_params, self._lora_request()
            )
        ).result()

        if return_raw_output:
            base_output = outputs
        else:
            base_output = [
                [choice.text for choice in output.outputs]
                if len(output.outputs) > 1
                else output.outputs[0].text
                for output in outputs
            ]
        if return_scores:
            scores = []
            for output in outputs:
                output_scores = []
                for single_output in output.outputs:
                    if single_output.logprobs:
                        token_probs = [
                            np.exp(list(score_dict.values())[0].logprob)
                            for score_dict in single_output.logprobs
                        ]
                        output_scores.append(token_probs)
                    else:
                        output_scores.append([])
                scores.append(
                    output_scores[0] if len(output_scores) == 1 else output_scores
                )
            return base_output, scores
        if return_cache_stats or return_gate_logprobs:
            return base_output, [
                self._cache_stats(output, gate_token_ids) for output in outputs
            ]
        return base_output

    def add_requests(
        self,
        input_list: List,
        *,
        return_cache_stats=False,
        return_gate_logprobs=False,
        gate_token_ids=None,
        gate_logprobs=0,
        gate_abort_margin=None,
        request_metadata: Optional[Sequence[dict]] = None,
        priorities: Optional[Sequence[int]] = None,
        **params,
    ) -> List[str]:
        """Admit requests to AsyncLLMEngine without blocking for decoding."""

        from vllm.sampling_params import RequestOutputKind

        if isinstance(input_list, (str, dict)):
            input_list = [input_list]
        else:
            input_list = list(input_list)
        if request_metadata is None:
            request_metadata = [{} for _ in input_list]
        if len(request_metadata) != len(input_list):
            raise ValueError("request_metadata and input_list must have equal length")
        if priorities is None:
            priorities = [0 for _ in input_list]
        else:
            priorities = list(priorities)
        if len(priorities) != len(input_list):
            raise ValueError("priorities and input_list must have equal length")
        normalized_priorities = []
        for priority in priorities:
            if isinstance(priority, (bool, np.bool_)) or not isinstance(
                priority, (int, np.integer)
            ):
                raise TypeError("every request priority must be an integer")
            normalized_priorities.append(int(priority))

        sampling_params = self._prepare_sampling_params(
            params,
            gate_logprobs=(gate_logprobs if return_gate_logprobs else 0),
        )
        sampling_params.output_kind = (
            RequestOutputKind.CUMULATIVE
            if gate_abort_margin is not None
            else RequestOutputKind.FINAL_ONLY
        )
        lora_request = None
        if self.use_lora:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("lora_module", 1, self.lora_path)

        request_ids = []
        try:
            for prompt, metadata, priority in zip(
                input_list, request_metadata, normalized_priorities
            ):
                request_id = self._new_request_id("stream")
                request_ids.append(request_id)
                with self._incremental_lock:
                    self._incremental_request_meta[request_id] = {
                        "metadata": dict(metadata or {}),
                        # Measurement-only timestamp.  This is carried to the
                        # completion queue so the benchmark can measure the
                        # request's actual AsyncLLMEngine service lifetime
                        # instead of timing the non-blocking completion poll.
                        "service_started_ns": time.time_ns(),
                        "first_token_ns": None,
                        "return_cache_stats": bool(return_cache_stats),
                        "gate_token_ids": gate_token_ids
                        if return_gate_logprobs
                        else None,
                        "gate_abort_margin": (
                            float(gate_abort_margin)
                            if gate_abort_margin is not None
                            else None
                        ),
                        "gate_checked": False,
                    }
                future = self._submit(
                    self._consume_incremental(
                        prompt,
                        deepcopy(sampling_params),
                        request_id,
                        priority,
                        lora_request,
                    )
                )
                with self._incremental_lock:
                    self._incremental_futures[request_id] = future
                future.add_done_callback(self._forget_future(request_id))
        except BaseException:
            if request_ids:
                self.abort_requests(request_ids)
            raise
        return request_ids

    def step_requests(self) -> List[dict]:
        """Drain completions; AsyncLLMEngine advances on its own event loop."""

        # Clear before draining. A producer racing with the drain sets the
        # event again, while a completion already in the queue is consumed in
        # this pass. Re-arm defensively if the queue is still non-empty.
        completion_event = self._get_incremental_completion_event()
        completion_event.clear()
        completed = []
        while True:
            try:
                item = self._incremental_completed.get_nowait()
            except queue.Empty:
                break
            error = item.get("error")
            if error is not None:
                raise RuntimeError(
                    f"AsyncLLMEngine request {item['request_id']} failed"
                ) from error
            completed.append(item)
        if not self._incremental_completed.empty():
            completion_event.set()
        return completed

    def wait_for_request_completion(self, timeout_s: float) -> bool:
        """Wait until a semantic completion is published or timeout expires."""

        return self._get_incremental_completion_event().wait(
            max(0.0, float(timeout_s))
        )

    def set_request_wakeup_event(
        self, event: Optional[threading.Event]
    ) -> None:
        """Register the coordinator event shared with retrieval producers."""

        self._incremental_external_wakeup_event = event

    def has_unfinished_requests(self) -> bool:
        with self._incremental_lock:
            active = bool(self._incremental_request_meta)
        # A terminal output remains unfinished from the pipeline's point of
        # view until ``step_requests`` consumes it.  Omitting this queue check
        # creates a race where retrieval finishes between the async producer
        # retiring its engine request and the synchronous scheduler observing
        # the completion.
        return active or not self._incremental_completed.empty()

    def abort_requests(self, request_ids: Iterable[str]) -> None:
        request_ids = [str(request_id) for request_id in request_ids]
        if not request_ids:
            return
        self._submit(self._abort_many(request_ids)).result()
        with self._incremental_lock:
            futures = [
                self._incremental_futures.pop(request_id, None)
                for request_id in request_ids
            ]
            for request_id in request_ids:
                self._incremental_request_meta.pop(request_id, None)
        for future in futures:
            if future is not None:
                future.cancel()
