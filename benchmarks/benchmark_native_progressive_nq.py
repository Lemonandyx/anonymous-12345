#!/usr/bin/env python3
"""Shared helpers for the FlashRAG/HarmoRAG NQ latency benchmarks.

This module intentionally contains only workload construction, configuration,
dataset loading and evaluation helpers.  Timed retrieval/generation code lives
in the benchmark drivers so setup and metric computation remain outside the
measured interval.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np


NATIVE_SYSTEM_PROMPT = (
    "Answer the question based on the given document."
    "Only give me the answer and do not output any other words."
    "\nThe following are given documents.\n\n{reference}"
)
NATIVE_USER_PROMPT = "Question: {question}"


@dataclass(frozen=True)
class ArrivalWindow:
    batch_id: int
    start: int
    stop: int
    release_s: float

    @property
    def size(self) -> int:
        return self.stop - self.start


def build_windows(
    num_requests: int, arrival_rate: float, seed: int
) -> list[ArrivalWindow]:
    """Build the legacy one-Poisson-batch-per-second workload."""

    if num_requests < 1 or arrival_rate <= 0:
        raise ValueError("num_requests and arrival_rate must be positive")
    rng = np.random.default_rng(seed)
    result: list[ArrivalWindow] = []
    start = 0
    second = 0
    while start < num_requests:
        stop = min(num_requests, start + max(1, int(rng.poisson(arrival_rate))))
        result.append(ArrivalWindow(len(result), start, stop, float(second)))
        start = stop
        second += 1
    return result


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def latency_summary(
    completion_s: Sequence[float], windows: Sequence[ArrivalWindow]
) -> dict[str, float | None]:
    arrivals = [0.0] * len(completion_s)
    for window in windows:
        arrivals[window.start : window.stop] = [float(window.release_s)] * window.size
    values = [float(done) - arrival for done, arrival in zip(completion_s, arrivals)]
    if not values:
        return {
            "mean_s": None,
            "p50_s": None,
            "p90_s": None,
            "p95_s": None,
            "p99_s": None,
            "max_s": None,
        }
    return {
        "mean_s": statistics.fmean(values),
        "p50_s": percentile(values, 50),
        "p90_s": percentile(values, 90),
        "p95_s": percentile(values, 95),
        "p99_s": percentile(values, 99),
        "max_s": max(values),
    }


def _arg(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def common_config(args: Any):
    """Create the exact Config shared by stock FlashRAG and HarmoRAG."""

    from harmorag.config import Config

    output = Path(args.output).resolve()
    search_max_tokens = _arg(args, "search_max_tokens")
    if search_max_tokens is None:
        search_max_tokens = args.max_tokens
    first_snapshot = _arg(args, "m2_first_snapshot_nprobe")
    if first_snapshot is None:
        first_snapshot = args.bootstrap_nprobe
    native_permits = _arg(args, "native_global_scan_permits")
    if native_permits is None:
        native_permits = args.native_worker_threads
    topk = int(args.topk)
    gate_ordinals = list(_arg(args, "m2_search_gate_ordinals", []))
    config_dict = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "save_dir": str(output.parent),
        "gpu_id": str(args.gpu_id),
        "dataset_name": str(args.dataset_name),
        "split": [str(args.warmup_split), str(args.measured_split)],
        "test_sample_num": max(int(args.num_requests), int(args.warmup)),
        "random_sample": False,
        "seed": 2024,
        "save_intermediate_data": False,
        "save_note": f"{args.dataset_name}_arrival{args.arrival_rate:g}_{args.mode}",
        "is_reasoning": False,
        "retrieval_method": "e5",
        "retrieval_model_path": str(Path(args.retriever_model_path).resolve()),
        "index_path": str(Path(args.index_path).resolve()),
        "faiss_gpu": False,
        "corpus_path": str(Path(args.corpus_path).resolve()),
        "retrieval_topk": topk,
        "retrieval_batch_size": 256,
        "retrieval_use_fp16": True,
        "retrieval_query_max_length": 128,
        "retrieval_pooling_method": "mean",
        "save_retrieval_cache": False,
        "use_retrieval_cache": False,
        "silent_retrieval": True,
        "framework": "vllm",
        "generator_model": "llama3-8B-instruct",
        "generator_model_path": str(Path(args.generator_model_path).resolve()),
        "generator_max_input_len": int(args.max_input_len),
        "generator_batch_size": int(_arg(args, "native_dynamic_batch_limit", 256)),
        "generation_params": {
            "max_tokens": int(args.max_tokens),
            "temperature": 0,
        },
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "generator_gpu_num": 1,
        "metrics": ["em", "f1", "acc"],
        "metric_setting": {
            "retrieval_recall_topk": topk,
            "tokenizer_name": "gpt-4",
        },
        "save_metric_score": True,
        "progressive_retrieval": args.mode != "native",
        "progressive_retrieval_backend": str(
            _arg(args, "progressive_retrieval_backend", "native_continuous_ivf")
        ),
        "progressive_enable_native_continuous_scanner": bool(
            _arg(args, "enable_native_continuous_scanner", True)
        ),
        "progressive_native_scanner_module": str(
            _arg(args, "native_scanner_module", "harmorag_native_scanner")
        ),
        "progressive_native_scanner_source": str(
            _arg(args, "native_scanner_source", "")
        ),
        "progressive_bootstrap_nprobe": int(args.bootstrap_nprobe),
        "progressive_m2_first_snapshot_nprobe": int(first_snapshot),
        "progressive_fast_max_nprobe": int(args.nprobe),
        "progressive_max_nprobe": int(args.nprobe),
        "progressive_heap_topk": int(args.heap_size),
        "progressive_wrd_topk": int(args.monitor_size),
        "progressive_wrd_alpha": float(args.wrd_alpha),
        "progressive_wrd_epsilon": float(args.wrd_epsilon),
        "progressive_wrd_window": int(args.wrd_window),
        "progressive_wrd_beta": float(args.wrd_beta),
        "progressive_wrd_mu": float(args.wrd_mu),
        "progressive_wrd_patience": int(args.wrd_patience),
        "progressive_index_mmap": True,
        "progressive_retrieval_device": str(args.retrieval_device),
        "progressive_omp_threads": int(args.omp_threads),
        "progressive_native_worker_threads": int(args.native_worker_threads),
        "progressive_native_global_scan_permits": int(native_permits),
        "progressive_native_lookahead_lists": int(args.native_lookahead_lists),
        "progressive_pipeline_batch_size": 32,
        "progressive_prefetch_batches": 1,
        "progressive_event_queue_size": int(args.event_queue_size),
        "progressive_coalesce_ranking_events": bool(args.coalesce_ranking_events),
        "progressive_continuation_priority_scheduler": bool(
            args.continuation_priority_scheduler
        ),
        "progressive_current_batch_priority_scheduler": bool(
            args.current_batch_priority_scheduler
        ),
        "progressive_rolling_m3_scanners": bool(args.rolling_m3_scanners),
        "progressive_max_generation_chains": int(args.max_generation_chains),
        "progressive_no_answer_sentinel": "No",
        "progressive_search_document_limit": topk,
        "progressive_search_gate_ordinals": gate_ordinals,
        "progressive_bulk_final_on_retrieval_terminal": bool(
            args.bulk_final_on_retrieval_terminal
        ),
        "progressive_full_context_only": bool(args.full_context_only),
        "progressive_max_evidence_documents": 0,
        "progressive_generation_max_tokens": int(search_max_tokens),
        # FINAL is specified as one concise answer span. Stop before a model
        # starts a second line so the 128-token cap remains a safety ceiling,
        # rather than silently accepting a length-truncated explanation.
        "progressive_final_generation_stop": ["\n", "\r"],
        "progressive_store_prompt_trajectory": bool(args.trace_generation_details),
        "progressive_store_retrieval_trace": bool(args.trace_generation_details),
        "progressive_use_token_prompts": True,
        "progressive_capture_prefix_cache_stats": True,
        "progressive_require_grounded_answer": True,
        "progressive_max_answer_tokens": int(args.max_answer_tokens),
        "progressive_final_max_answer_tokens": int(args.max_tokens),
        "progressive_compact_answerability_prompt": bool(
            args.compact_answerability_prompt
        ),
        "progressive_single_snapshot_gate": bool(args.single_snapshot_gate),
        "progressive_conversational_prefix": bool(args.conversational_prefix),
        "progressive_minimal_control_tail": bool(args.minimal_control_tail),
        "progressive_answerability_logprob_margin": float(
            args.answerability_logprob_margin
        ),
        "progressive_answerability_early_abort_below_margin": bool(
            args.answerability_early_abort_below_margin
        ),
        "progressive_answerability_gate_logprobs": int(
            args.answerability_gate_logprobs
        ),
        "progressive_answerability_require_logprobs": bool(
            args.answerability_require_logprobs
        ),
        "progressive_answerability_retention_topk": topk,
        "progressive_answerability_structured_output": bool(
            args.answerability_structured_output
        ),
        "progressive_accumulated_answerability": bool(
            args.accumulated_answerability
        ),
        "progressive_answerability_semantic_rescue": bool(
            args.answerability_semantic_rescue
        ),
        "progressive_answerability_semantic_rescue_margin": float(
            args.answerability_semantic_rescue_margin
        ),
        "progressive_answerability_semantic_rescue_min_question_coverage": float(
            args.answerability_semantic_rescue_min_question_coverage
        ),
        "progressive_answerability_lexical_no_skip": bool(
            args.answerability_lexical_no_skip
        ),
        "progressive_event_driven_generation": bool(args.event_driven_generation),
        "progressive_generation_coalesce_ms": float(args.generation_coalesce_ms),
        "progressive_generation_coalesce_max_batch": int(
            args.generation_coalesce_max_batch
        ),
        "progressive_generation_coalesce_max_steps": int(
            args.generation_coalesce_max_steps
        ),
        "progressive_minimal_structured_gate_logprobs": bool(
            args.minimal_structured_gate_logprobs
        ),
        "progressive_prefix_block_alignment": bool(args.prefix_block_alignment),
        "progressive_prefix_cache_block_size": 16,
        "progressive_enable_answer_verifier": False,
        "progressive_final_topk": topk,
        "disable_save": False,
    }
    return Config(config_dict=config_dict)


def _split_path(config: Any, split: str) -> Path:
    root = Path(config["dataset_path"])
    for suffix in ("jsonl", "json", "parquet"):
        candidate = root / f"{split}.{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no dataset file for split {split!r} under {root}")


def load_data(
    config: Any,
    num_requests: int,
    warmup: int,
    *,
    measured_split: str = "test",
    warmup_split: str = "dev",
):
    from harmorag.dataset import Dataset

    measured = Dataset(
        config,
        str(_split_path(config, measured_split)),
        sample_num=int(num_requests),
        random_sample=False,
    )
    warm = Dataset(
        config,
        str(_split_path(config, warmup_split)),
        sample_num=int(warmup),
        random_sample=False,
    )
    if len(measured) != int(num_requests):
        raise RuntimeError(
            f"requested {num_requests} measured rows, loaded {len(measured)}"
        )
    if len(warm) != int(warmup):
        raise RuntimeError(f"requested {warmup} warmup rows, loaded {len(warm)}")
    return measured, warm


def evaluate(config: Any, dataset: Any) -> dict:
    from harmorag.evaluator import Evaluator

    return Evaluator(config).evaluate(dataset)


def output_column(dataset: Any, names: Sequence[str], default: Any = None) -> list[Any]:
    rows = []
    for item in dataset:
        output = getattr(item, "output", {}) or {}
        value = default
        for name in names:
            if name in output:
                value = output[name]
                break
            data = getattr(item, "data", {}) or {}
            if name in data:
                value = data[name]
                break
        rows.append(value)
    return rows


def per_sample_saved_metric(dataset: Any, metric: str) -> list[float | None]:
    result = []
    for item in dataset:
        scores = (getattr(item, "output", {}) or {}).get("metric_score", {})
        value = scores.get(metric)
        result.append(float(value) if value is not None and math.isfinite(float(value)) else None)
    return result


def per_sample_em(dataset: Any) -> list[float | None]:
    return per_sample_saved_metric(dataset, "em")
