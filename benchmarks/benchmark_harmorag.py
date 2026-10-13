#!/usr/bin/env python3
"""Controlled independent ablation for the three HarmoRAG mechanisms.

Groups are cumulative and intentionally not freely combinable:

* native: stock nprobe=128, top-5 retrieval and one native generation.
* m1: WRD-stable-or-128 retrieval, then one native generation.
* m2: full nprobe=128 retrieval plus progressive evidence/generation; WRD
  early termination is disabled so the measured delta belongs to M2 alone.
* m3: stock nprobe=128 retrieval and one native generation with one-window
  cross-batch retrieval/generation overlap; M1 and M2 are disabled.
* m1_m2: M1 plus progressive generation while the current batch's native
  retrieval continues uninterrupted in the background.
* m1_m2_m3: M1+M2 plus one released next batch.  With event-driven generation,
  the current and lookahead native sessions both keep scanning and are polled
  round-robin; the lookahead is never parked at nprobe=4.  The synchronous
  control path retains the legacy bootstrap-and-park schedule.
* full: alias for the production M1+M2+M3 path used by the independent
  component-ablation figure.

The script records answer-containment Recall@1..5 for both the ranked retrieval
endpoint and the ordered unique documents actually exposed to generation.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import statistics
import string
import sys
import threading
import time
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from fig13_vllm_arrivals import arrivals_for_rate, trace_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_native_progressive_nq import (  # noqa: E402
    ArrivalWindow,
    NATIVE_SYSTEM_PROMPT,
    NATIVE_USER_PROMPT,
    build_windows as build_legacy_windows,
    common_config,
    evaluate,
    latency_summary as legacy_latency_summary,
    load_data,
    output_column,
    per_sample_em,
    per_sample_saved_metric,
    percentile,
)


@dataclass(frozen=True)
class ContinuousArrivalWindow:
    """Non-anticipating microbatch cut from one continuous Poisson trace."""

    batch_id: int
    start: int
    stop: int
    release_s: float
    arrival_times: Tuple[float, ...]

    @property
    def size(self) -> int:
        return self.stop - self.start


def build_continuous_poisson_windows(
    num_requests: int,
    arrival_rate: float,
    seed: int,
    batching_window_ms: float,
) -> List[ContinuousArrivalWindow]:
    """Match vLLM's seeded Poisson trace and release each request on arrival."""

    if num_requests < 1 or arrival_rate <= 0:
        raise ValueError("num_requests and arrival_rate must be positive")
    if batching_window_ms != 0:
        raise ValueError("the aligned Figure 13 protocol forbids admission windows")
    arrivals = arrivals_for_rate(num_requests, arrival_rate, seed)
    windows: List[ContinuousArrivalWindow] = []
    for start, arrival in enumerate(arrivals):
        windows.append(
            ContinuousArrivalWindow(
                len(windows),
                start,
                start + 1,
                float(arrival),
                (float(arrival),),
            )
        )
    return windows


def build_closed_loop_windows(num_requests: int) -> List[ContinuousArrivalWindow]:
    """Create one logical request per window for sequential closed-loop service.

    Release timestamps are placeholders. Admission below is gated on completion
    of the preceding query, and the report uses the measured retrieval-session
    start as each request's arrival so queue time is exactly zero.
    """

    if num_requests < 1:
        raise ValueError("num_requests must be positive")
    return [
        ContinuousArrivalWindow(index, index, index + 1, 0.0, (0.0,))
        for index in range(num_requests)
    ]


def window_arrivals(window) -> List[float]:
    values = getattr(window, "arrival_times", None)
    if values is None:
        return [float(window.release_s)] * int(window.size)
    return [float(value) for value in values]


def latency_summary(completion_s: Sequence[float], windows: Sequence[object]):
    arrivals = [0.0] * len(completion_s)
    for window in windows:
        arrivals[window.start : window.stop] = window_arrivals(window)
    latencies = [float(done - arrived) for done, arrived in zip(completion_s, arrivals)]
    return {
        "mean_s": statistics.fmean(latencies),
        "p50_s": percentile(latencies, 50),
        "p90_s": percentile(latencies, 90),
        "p95_s": percentile(latencies, 95),
        "p99_s": percentile(latencies, 99),
        "max_s": max(latencies),
    }


GROUPS = {
    "native": {
        "m1": False,
        "m2": False,
        "m3_same_batch_background": False,
        "m3_next_batch_bootstrap": False,
    },
    "m1": {
        "m1": True,
        "m2": False,
        "m3_same_batch_background": False,
        "m3_next_batch_bootstrap": False,
    },
    "m2": {
        "m1": False,
        "m2": True,
        "m3_same_batch_background": False,
        "m3_next_batch_bootstrap": False,
    },
    "m3": {
        "m1": False,
        "m2": False,
        "m3_same_batch_background": False,
        "m3_next_batch_bootstrap": True,
    },
    "m1_m2": {
        "m1": True,
        "m2": True,
        "m3_same_batch_background": True,
        "m3_next_batch_bootstrap": False,
    },
    "m1_m2_m3": {
        "m1": True,
        "m2": True,
        "m3_same_batch_background": True,
        "m3_next_batch_bootstrap": True,
    },
    "full": {
        "m1": True,
        "m2": True,
        "m3_same_batch_background": True,
        "m3_next_batch_bootstrap": True,
    },
}

SHARED_MECHANISM_PROFILES = {
    "vanilla": {
        "m1": False,
        "m2": False,
        "m3_same_batch_background": False,
        "m3_next_batch_bootstrap": False,
    },
    "raet": {
        "m1": True,
        "m2": False,
        "m3_same_batch_background": False,
        "m3_next_batch_bootstrap": False,
    },
    "ica": {
        "m1": False,
        "m2": True,
        "m3_same_batch_background": False,
        "m3_next_batch_bootstrap": False,
    },
    "ars": {
        "m1": False,
        "m2": False,
        "m3_same_batch_background": True,
        "m3_next_batch_bootstrap": True,
    },
    "full": {
        "m1": True,
        "m2": True,
        "m3_same_batch_background": True,
        "m3_next_batch_bootstrap": True,
    },
}


STOCK_RETRIEVAL_BACKEND = "stock_faiss_ivf_nprobe128"
PROGRESSIVE_RETRIEVAL_BACKEND = "native_continuous_ivf"
KNOWN_LEGACY_PER_LIST_RETRIEVER_SHA256 = {
    "40caf269f94807c50cfea3487d06f74fb4c18b44fcf102e6be2f5ed0a4a8935c",
}


def item_field(item, field: str):
    value = getattr(item, field, None)
    if value is not None:
        return value
    data = getattr(item, "data", None)
    return data.get(field) if isinstance(data, dict) else None


def references_for(item) -> List[str]:
    for field in ("golden_answers", "golden_answer", "answers"):
        value = item_field(item, field)
        if value is not None:
            return [str(value)] if isinstance(value, str) else [str(x) for x in value]
    raise RuntimeError("NQ item is missing golden answers")


def normalize_containment(value: object) -> str:
    text = str(value or "").lower()
    text = "".join(" " if char in string.punctuation else char for char in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def document_contents(document: dict) -> str:
    return str(document.get("contents") or document.get("text") or "")


def containment_row(item, documents: Sequence[dict]) -> List[bool]:
    answers = tuple(
        dict.fromkeys(
            answer
            for answer in (normalize_containment(value) for value in references_for(item))
            if answer
        )
    )
    row = []
    for document in documents:
        normalized = f" {normalize_containment(document_contents(document))} "
        row.append(any(f" {answer} " in normalized for answer in answers))
    return row


def recall_curve(hit_rows: Sequence[Sequence[bool]], max_k: int = 5) -> dict:
    counts = {
        str(k): sum(any(row[:k]) for row in hit_rows)
        for k in range(1, max_k + 1)
    }
    denominator = len(hit_rows)
    return {
        "hit_count": counts,
        "recall": {
            key: value / denominator if denominator else None
            for key, value in counts.items()
        },
        "denominator": denominator,
    }


def ordered_unique(values: Iterable[int]) -> List[int]:
    seen = set()
    result = []
    for value in values:
        value = int(value)
        if value >= 0 and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def audit_search_document_selection(
    search_ids: Sequence[Sequence[int]],
    search_budgets: Sequence[Sequence[int]],
    selection_candidate_ids: Sequence[Sequence[Sequence[int]]],
) -> List[List[dict]]:
    """Prove each M2 document was the highest-ranked unsent item at its snapshot."""

    if not (
        len(search_ids) == len(search_budgets) == len(selection_candidate_ids)
    ):
        raise RuntimeError("M2 selection audit inputs have inconsistent row counts")
    result: List[List[dict]] = []
    for query_index, (ids, budgets, candidate_rows) in enumerate(
        zip(search_ids, search_budgets, selection_candidate_ids)
    ):
        ids = [int(value) for value in (ids or [])]
        budgets = [int(value) for value in (budgets or [])]
        candidate_rows = list(candidate_rows or [])
        if not (len(ids) == len(budgets) == len(candidate_rows)):
            raise RuntimeError(
                f"query {query_index} has different M2 document/budget/heap counts"
            )

        sent = set()
        rows = []
        for document_id, budget, raw_candidates in zip(
            ids, budgets, candidate_rows
        ):
            candidates = tuple(
                int(value) for value in (raw_candidates or []) if int(value) >= 0
            )
            eligible = [value for value in candidates if value not in sent]
            if not eligible:
                raise RuntimeError(
                    f"query {query_index} budget {budget} has no unsent heap candidate"
                )
            if document_id != eligible[0]:
                raise RuntimeError(
                    f"query {query_index} selected {document_id} at budget {budget}, "
                    f"but highest-ranked unsent document was {eligible[0]}"
                )
            selected_rank = candidates.index(document_id)
            rows.append(
                {
                    "budget": budget,
                    "document_id": document_id,
                    "selected_rank": selected_rank,
                    "snapshot_document_ids": list(candidates),
                }
            )
            sent.add(document_id)
        result.append(rows)
    return result


def materialize_rows(retriever, id_rows: Sequence[Sequence[int]]) -> List[List[dict]]:
    lengths = [len(row) for row in id_rows]
    flat = [int(value) for row in id_rows for value in row]
    documents = retriever.materialize_documents(flat) if flat else []
    rows = []
    cursor = 0
    for length in lengths:
        rows.append(documents[cursor : cursor + length])
        cursor += length
    if cursor != len(documents):
        raise RuntimeError("materialized document slicing is inconsistent")
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_file_sha256(module_name: str) -> str | None:
    """Hash an installed module without assuming a machine-specific path."""

    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin)
    return file_sha256(path) if path.is_file() else None


def json_safe(value):
    """Convert tracing payloads to bounded, JSON-serializable primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return repr(value)


def prompt_trace_payload(prompt, tokenizer, *, include_text: bool) -> dict:
    """Describe the exact generator prompt without forcing huge JSON output."""

    token_ids = None
    decoded = None
    if isinstance(prompt, str):
        raw = prompt.encode("utf-8")
        decoded = prompt
        if tokenizer is not None:
            try:
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            except Exception:
                token_ids = None
    elif isinstance(prompt, dict) and "prompt_token_ids" in prompt:
        token_ids = [int(value) for value in prompt["prompt_token_ids"]]
        raw = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        if include_text and tokenizer is not None:
            try:
                decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
            except Exception:
                decoded = None
    else:
        serializable = json_safe(prompt)
        raw = json.dumps(
            serializable, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        decoded = str(serializable)

    payload = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "kind": (
            "text"
            if isinstance(prompt, str)
            else "token_ids"
            if token_ids is not None
            else type(prompt).__name__
        ),
        "token_count": len(token_ids) if token_ids is not None else None,
        "character_count": len(decoded) if decoded is not None else None,
    }
    if include_text:
        payload["prompt"] = decoded
    return payload


def merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    merged: List[List[float]] = []
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([float(start), float(stop)])
        else:
            merged[-1][1] = max(merged[-1][1], float(stop))
    return [(start, stop) for start, stop in merged]


def interval_total(intervals: Sequence[Tuple[float, float]]) -> float:
    return sum(stop - start for start, stop in merge_intervals(intervals))


def interval_overlap(
    first: Sequence[Tuple[float, float]], second: Sequence[Tuple[float, float]]
) -> float:
    left = merge_intervals(first)
    right = merge_intervals(second)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def subtract_intervals(
    base: Sequence[Tuple[float, float]],
    covered: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Return ``base`` minus ``covered`` as disjoint half-open intervals."""
    remaining: List[Tuple[float, float]] = []
    cuts = merge_intervals(covered)
    for start, stop in merge_intervals(base):
        cursor = start
        for cut_start, cut_stop in cuts:
            if cut_stop <= cursor:
                continue
            if cut_start >= stop:
                break
            if cut_start > cursor:
                remaining.append((cursor, min(cut_start, stop)))
            cursor = max(cursor, cut_stop)
            if cursor >= stop:
                break
        if cursor < stop:
            remaining.append((cursor, stop))
    return remaining


def useful_overlap_breakdown(
    retrieval_by_query: dict[int, Sequence[Tuple[float, float]]],
    generation_by_query: dict[int, Sequence[Tuple[float, float]]],
    cohort_by_query: dict[int, int],
) -> dict:
    """Partition active retrieval into mutually exclusive overlap sources."""
    generation_by_cohort: dict[int, List[Tuple[float, float]]] = {}
    all_generation: List[Tuple[float, float]] = []
    for query_index, rows in generation_by_query.items():
        cohort = cohort_by_query[int(query_index)]
        generation_by_cohort.setdefault(cohort, []).extend(rows)
        all_generation.extend(rows)
    all_generation = merge_intervals(all_generation)
    generation_by_cohort = {
        cohort: merge_intervals(rows)
        for cohort, rows in generation_by_cohort.items()
    }

    total_s = same_s = intra_s = cross_s = 0.0
    per_query = []
    for query_index in sorted(retrieval_by_query):
        retrieval = merge_intervals(retrieval_by_query[query_index])
        same_generation = merge_intervals(
            generation_by_query.get(query_index, ())
        )
        cohort_generation = generation_by_cohort.get(
            cohort_by_query[query_index], []
        )
        query_total = interval_total(retrieval)
        query_same = interval_overlap(retrieval, same_generation)
        after_same = subtract_intervals(retrieval, same_generation)
        query_intra = interval_overlap(after_same, cohort_generation)
        after_intra = subtract_intervals(after_same, cohort_generation)
        query_cross = interval_overlap(after_intra, all_generation)
        query_unhidden = query_total - query_same - query_intra - query_cross
        if query_unhidden < -1e-7:
            raise RuntimeError(
                "exclusive useful-overlap accounting exceeded retrieval time"
            )
        query_unhidden = max(0.0, query_unhidden)
        total_s += query_total
        same_s += query_same
        intra_s += query_intra
        cross_s += query_cross
        per_query.append(
            {
                "query_index": query_index,
                "cohort_id": cohort_by_query[query_index],
                "retrieval_s": query_total,
                "same_request_s": query_same,
                "same_cohort_other_request_s": query_intra,
                "cross_cohort_s": query_cross,
                "unhidden_s": query_unhidden,
            }
        )
    unhidden_s = total_s - same_s - intra_s - cross_s
    denominator = total_s or 1.0
    return {
        "definition": (
            "query-weighted active retrieval service, exclusively classified "
            "as same-request, same-cohort other-request, cross-cohort, or unhidden"
        ),
        "priority": [
            "same_request",
            "same_cohort_other_request",
            "cross_cohort",
            "unhidden",
        ],
        "retrieval_busy_s": total_s,
        "same_request_overlap_s": same_s,
        "same_cohort_other_request_overlap_s": intra_s,
        "cross_cohort_overlap_s": cross_s,
        "unhidden_retrieval_s": unhidden_s,
        "same_request_ratio": same_s / denominator,
        "same_cohort_other_request_ratio": intra_s / denominator,
        "cross_cohort_ratio": cross_s / denominator,
        "useful_overlap_ratio": (same_s + intra_s + cross_s) / denominator,
        "unhidden_ratio": unhidden_s / denominator,
        "accounting_exact": abs(
            same_s + intra_s + cross_s + unhidden_s - total_s
        ) <= 1e-7,
        "per_query": per_query,
    }


def m3_lookahead_eligibility_s(
    position: int,
    release_s: float,
    window_done_s: Sequence[float],
) -> float | None:
    """Return when ``position`` can legally become the single lookahead.

    The scheduler admits only the first unfinished window and its immediate
    successor.  Because that successor can answer before an older window,
    adjacent completion times need not be ordered.  Window ``j`` is eligible
    only after it has arrived and every window older than ``j - 1`` has
    completed.  It is a real M3 overlap opportunity only if this timestamp is
    still earlier than window ``j - 1``'s completion.
    """

    position = int(position)
    if not 0 < position < len(window_done_s):
        return None
    older_prefix_done = max(
        (float(value) for value in window_done_s[: position - 1]),
        default=0.0,
    )
    eligible_s = max(float(release_s), older_prefix_done)
    if eligible_s >= float(window_done_s[position - 1]):
        return None
    return eligible_s


def progressive_settings(args, *, incremental: bool, prefetch_batches: int) -> dict:
    generator_identity = str(
        args.generator_model_name or args.generator_model_path
    ).casefold()
    is_qwen3_generator = "qwen3" in generator_identity
    settings = {
        # The controlled progressive groups must use the continuous native
        # scanner.  Keep these knobs in Config rather than importing a concrete
        # extension here so the retriever adapter owns backend construction.
        "progressive_native_continuous_scan": args.enable_native_continuous_scanner,
        "progressive_retrieval_backend": args.progressive_retrieval_backend,
        "progressive_native_module": args.native_scanner_module,
        "progressive_native_source": str(args.native_scanner_source),
        "progressive_native_source_sha256": args.native_scanner_source_sha256,
        "progressive_native_header": str(args.native_scanner_header),
        "progressive_native_header_sha256": args.native_scanner_header_sha256,
        "progressive_native_worker_threads": args.native_worker_threads,
        "progressive_native_lookahead": args.native_lookahead_lists,
        "progressive_native_continuation_quantum": (
            args.native_continuation_quantum
        ),
        "progressive_native_request_aligned_priority": bool(
            args.native_request_aligned_priority
        ),
        "progressive_native_request_aligned_priority_burst": (
            args.native_request_aligned_priority_burst
        ),
        "progressive_native_request_aligned_max_fifo_wait_ms": (
            args.native_request_aligned_max_fifo_wait_ms
        ),
        "enable_prefix_caching": True,
        "progressive_retrieval": True,
        "progressive_bootstrap_nprobe": args.bootstrap_nprobe,
        # M2 generation latch.  The native scanner observes WRD from @1 inside
        # the fast prefix; this boundary controls publication, not the first
        # stability observation.
        "progressive_m2_first_snapshot_nprobe": (
            args.m2_first_snapshot_nprobe
        ),
        "progressive_fast_max_nprobe": args.nprobe,
        "progressive_max_nprobe": args.nprobe,
        "progressive_heap_topk": args.heap_size,
        "progressive_wrd_topk": args.monitor_size,
        "progressive_final_topk": args.topk,
        "progressive_wrd_alpha": args.wrd_alpha,
        "progressive_wrd_epsilon": args.wrd_epsilon,
        "progressive_wrd_window": args.wrd_window,
        "progressive_wrd_beta": args.wrd_beta,
        "progressive_wrd_mu": args.wrd_mu,
        "progressive_wrd_patience": args.wrd_patience,
        "progressive_enable_wrd": True,
        "progressive_index_mmap": True,
        "progressive_retrieval_device": args.retrieval_device,
        "progressive_omp_threads": args.omp_threads,
        "progressive_pipeline_batch_size": 1,
        "progressive_prefetch_batches": prefetch_batches,
        # Two live native sessions can publish one heap version per query/list
        # while vLLM is inside an engine step.  A 64-item FIFO therefore
        # back-pressures retrieval for tens of seconds at the target load,
        # violating M3's continuous-background-search contract.  Keep the
        # queue bounded, but large enough to absorb the deterministic worst
        # burst from the two-session scheduler.
        "progressive_event_queue_size": int(args.event_queue_size),
        "progressive_coalesce_ranking_events": bool(
            args.coalesce_ranking_events
        ),
        "progressive_max_generation_chains": int(args.max_generation_chains),
        "progressive_rolling_m3_scanners": bool(args.rolling_m3_scanners),
        # Keep retrieval control out of the task answer vocabulary.  HotpotQA
        # contains legitimate yes/no questions, so the natural-language
        # answer ``No`` cannot also be the internal insufficient-evidence
        # sentinel.  Compact SEARCHING still emits N/A; its N decision is
        # mapped to this unambiguous internal value by the pipeline.
        "progressive_no_answer_sentinel": "[INSUFFICIENT_EVIDENCE]",
        # Keep M2's append-only evidence budget equal to the stock top-k in
        # explicit native-parameter sweeps. The production default stays 5.
        "progressive_search_document_limit": int(args.topk),
        "progressive_max_evidence_documents": args.heap_size,
        "progressive_capture_prefix_cache_stats": True,
        "progressive_store_prompt_trajectory": bool(
            args.trace_generation_details
        ),
        "progressive_store_retrieval_trace": bool(
            args.trace_generation_details
        ),
        "progressive_require_grounded_answer": True,
        "progressive_max_answer_tokens": args.max_answer_tokens,
        "progressive_final_max_answer_tokens": max(10, args.max_answer_tokens),
        "progressive_answer_consistency_observations": 2,
        "progressive_answer_retention_topk": int(args.topk),
        "progressive_answer_fast_supporting_documents": 2,
        "progressive_answer_stable_supporting_documents": 1,
        "progressive_compact_answerability_prompt": bool(
            args.compact_answerability_prompt
        ),
        "progressive_single_snapshot_gate": bool(args.single_snapshot_gate),
        "progressive_conversational_prefix": bool(
            args.conversational_prefix
        ),
        "progressive_minimal_control_tail": bool(
            args.minimal_control_tail
        ),
        "progressive_answerability_logprob_margin": float(
            args.answerability_logprob_margin
        ),
        "progressive_answerability_early_abort_below_margin": bool(
            args.answerability_early_abort_below_margin
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
        "progressive_answerability_gate_logprobs": int(
            args.answerability_gate_logprobs
        ),
        "progressive_answerability_require_logprobs": bool(
            args.answerability_require_logprobs
        ),
        "progressive_answerability_retention_topk": int(args.topk),
        "progressive_answerability_structured_output": bool(
            args.answerability_structured_output
        ),
        "progressive_gate_component_stage": str(args.gate_component_stage),
        "progressive_accumulated_answerability": bool(
            args.accumulated_answerability
        ),
        "progressive_answerability_lexical_no_skip": bool(
            args.answerability_lexical_no_skip
        ),
        "progressive_search_gate_ordinals": tuple(
            int(value) for value in args.m2_search_gate_ordinals
        ),
        "progressive_bulk_final_on_retrieval_terminal": bool(
            args.bulk_final_on_retrieval_terminal
        ),
        "progressive_full_context_only": bool(args.full_context_only),
        "progressive_event_driven_generation": bool(
            args.event_driven_generation
        ),
        "progressive_continuation_priority_scheduler": bool(
            args.continuation_priority_scheduler
        ),
        "progressive_current_batch_priority_scheduler": bool(
            args.current_batch_priority_scheduler
        ),
        "progressive_full_retrieval_continuous_batching": bool(
            args.full_retrieval_continuous_batching
        ),
        "progressive_max_active_retrieval_sessions": int(
            args.max_active_retrieval_sessions
        ),
        "progressive_generation_coalesce_ms": float(
            args.generation_coalesce_ms
        ),
        "progressive_generation_coalesce_max_batch": int(
            args.generation_coalesce_max_batch
        ),
        "progressive_generation_coalesce_max_steps": int(
            args.generation_coalesce_max_steps
        ),
        "progressive_generation_poll_backoff_ms": float(
            args.generation_poll_backoff_ms
        ),
        "progressive_minimal_structured_gate_logprobs": bool(
            args.minimal_structured_gate_logprobs
        ),
        "progressive_prefix_block_alignment": bool(args.prefix_block_alignment),
        "progressive_prefix_cache_block_size": 16,
        # SEARCHING can emit only No or an answer of at most
        # max_answer_tokens words.  Leave two subword tokens per answer word
        # plus two delimiters, while avoiding a 16-token invalid continuation
        # from becoming the straggler for an entire synchronous vLLM batch.
        "progressive_generation_max_tokens": (
            int(args.search_max_tokens)
            if args.search_max_tokens is not None
            else min(args.max_tokens, max(4, 2 * args.max_answer_tokens + 2))
        ),
        "progressive_final_generation_max_tokens": args.max_tokens,
        "progressive_final_generation_stop": ["\n", "\r", "<|eot_id|>"],
        "progressive_final_length_repair_stop": ["\n", "\r", "."],
        "progressive_final_generation_structured_output_regex": (
            r"[^<\n\r]{1,120}" if is_qwen3_generator else None
        ),
        "progressive_final_generation_bad_words": (
            ["<think>", "</think>"] if is_qwen3_generator else []
        ),
        "progressive_enable_incremental_generation": incremental,
    }
    if args.native_global_scan_permits is not None:
        settings["progressive_native_global_scan_permits"] = int(
            args.native_global_scan_permits
        )
    return settings


def validate_rolling_m3_args(args) -> None:
    """Reject rolling-scanner combinations that cannot preserve M3 semantics."""

    if not args.rolling_m3_scanners:
        return
    if args.group not in {"m1_m2_m3", "full"}:
        raise ValueError("rolling M3 scanners require the full M1+M2+M3 group")
    if not args.event_driven_generation:
        raise ValueError("rolling M3 scanners require event-driven generation")
    if args.max_generation_chains < 1:
        raise ValueError("rolling M3 scanners require a positive generation chain cap")


def validate_args(args) -> None:
    if args.m2_first_snapshot_nprobe is None:
        args.m2_first_snapshot_nprobe = int(args.bootstrap_nprobe)
    validate_rolling_m3_args(args)
    if args.continuation_priority_scheduler and (
        not args.event_driven_generation
        or args.group not in {"m2", "m1_m2", "m1_m2_m3", "full"}
    ):
        raise ValueError(
            "continuation priority scheduling requires an event-driven M2 group"
        )
    if args.calculate_kv_scales and args.kv_cache_dtype != "fp8":
        raise ValueError("dynamic KV scales require --kv-cache-dtype fp8")
    if args.conversational_prefix and args.minimal_control_tail:
        raise ValueError(
            "conversational prefix and minimal control tail are mutually exclusive"
        )
    if args.nprobe != 128 and not args.allow_native_parameter_experiment:
        raise ValueError(
            "the paper default uses nprobe=128; pass "
            "--allow-native-parameter-experiment for an explicit nprobe sweep"
        )
    if args.nprobe < 1:
        raise ValueError("nprobe must be positive")
    if not 1 <= args.topk <= args.heap_size:
        raise ValueError("topk must be within [1, heap-size]")
    if args.topk not in {1, 5, 10} and not args.allow_native_parameter_experiment:
        raise ValueError(
            "paper comparisons use top-k in {1, 5, 10}; pass "
            "--allow-native-parameter-experiment for another value"
        )
    if not 1 <= args.bootstrap_nprobe <= args.nprobe:
        raise ValueError(
            "bootstrap nprobe must be within [1, native nprobe]"
        )
    if not (
        args.bootstrap_nprobe
        <= args.m2_first_snapshot_nprobe
        <= args.nprobe
    ):
        raise ValueError(
            "M2 first snapshot nprobe must be within "
            "[physical bootstrap nprobe, native nprobe]"
        )
    if (
        args.bootstrap_nprobe != 28
        and not args.allow_bootstrap_nprobe_experiment
    ):
        raise ValueError(
            "non-default bootstrap nprobe requires "
            "--allow-bootstrap-nprobe-experiment"
        )
    if (
        (args.heap_size != 100 or args.monitor_size != 100)
        and not args.allow_wrd_parameter_experiment
    ):
        raise ValueError("controlled comparison requires heap=monitor=100")
    observed = (
        args.wrd_alpha,
        args.wrd_epsilon,
        args.wrd_window,
        args.wrd_beta,
        args.wrd_mu,
        args.wrd_patience,
    )
    if (
        observed != (1.0, 0.001, 6, 0.4, 0.75, 16)
        and not args.allow_wrd_parameter_experiment
    ):
        raise ValueError("controlled comparison requires the tuned WRD formula parameters")
    if args.progressive_retrieval_backend != "native_continuous_ivf":
        raise ValueError(
            "controlled progressive groups require the native_continuous_ivf backend"
        )
    if not args.enable_native_continuous_scanner:
        raise ValueError("controlled progressive groups require the native scanner flag")
    if args.native_worker_threads < 1:
        raise ValueError("native worker thread count must be positive")
    if args.max_active_retrieval_sessions < 1:
        raise ValueError("max active retrieval sessions must be positive")
    if args.max_online_embedding_batch < 1:
        raise ValueError("max online embedding batch must be positive")
    if args.native_dynamic_batch_limit < 1:
        raise ValueError("native dynamic batch limit must be positive")
    if (
        args.native_global_scan_permits is not None
        and args.native_global_scan_permits < 1
    ):
        raise ValueError("native global scan permit count must be positive")
    if args.search_max_tokens is not None and args.search_max_tokens < 2:
        raise ValueError("SEARCHING decode cap must be at least two tokens")
    gate_spec = str(args.m2_search_gate_ordinals).strip().casefold()
    if gate_spec == "all":
        gate_ordinals = tuple(range(1, int(args.topk)))
    else:
        try:
            gate_ordinals = tuple(
                sorted(
                    {
                        int(value.strip())
                        for value in gate_spec.split(",")
                        if value.strip()
                    }
                )
            )
        except ValueError as error:
            raise ValueError(
                "M2 search gate ordinals must be 'all' or comma-separated integers"
            ) from error
    if args.topk > 1 and 1 not in gate_ordinals:
        raise ValueError("M2 search gate ordinals must retain document 1")
    if any(ordinal < 1 or ordinal >= args.topk for ordinal in gate_ordinals):
        raise ValueError(
            "M2 search gate ordinals must be between 1 and topk-1"
        )
    args.m2_search_gate_ordinals = gate_ordinals
    requested_full_retrieval = bool(args.full_retrieval_continuous_batching)
    profile_is_full = (
        args.shared_mechanism_profile in (None, "full")
    )
    args.full_retrieval_continuous_batching = bool(
        requested_full_retrieval
        and args.group == "full"
        and profile_is_full
    )
    if args.full_retrieval_continuous_batching:
        # Production full-continuous retrieval has no current/successor role.
        # Accept legacy launchers, but normalize their old scheduling flags so
        # the manifest records the effective scanner-wide policy.
        args.current_batch_priority_scheduler = False
        args.rolling_m3_scanners = False
    if args.shared_mechanism_profile is not None:
        profile = SHARED_MECHANISM_PROFILES[args.shared_mechanism_profile]
        if args.group != "full":
            raise ValueError(
                "shared mechanism profiles must use --group full so every "
                "configuration enters the same progressive execution path"
            )
        expected_full_context = not bool(profile["m2"])
        if bool(args.full_context_only) != expected_full_context:
            raise ValueError(
                f"profile {args.shared_mechanism_profile} requires "
                f"full_context_only={expected_full_context}"
            )
        expected_rolling = bool(
            profile["m3_next_batch_bootstrap"]
            and not args.full_retrieval_continuous_batching
        )
        if bool(args.rolling_m3_scanners) != expected_rolling:
            raise ValueError(
                f"profile {args.shared_mechanism_profile} requires "
                f"rolling_m3_scanners={expected_rolling}"
            )
        if not args.event_driven_generation:
            raise ValueError(
                "shared mechanism profiles require the common event-driven "
                "admission/batching execution layer"
            )
        if (
            not args.full_retrieval_continuous_batching
            and not args.current_batch_priority_scheduler
        ):
            raise ValueError(
                "shared mechanism profiles require the common current-batch "
                "priority scheduler"
            )
    if args.full_context_only and args.group != "full":
        raise ValueError("Full Context requires the full M1+M2+M3 pipeline")
    if (
        args.bulk_final_on_retrieval_terminal
        and gate_ordinals != tuple(range(1, int(args.topk)))
    ):
        raise ValueError(
            "terminal-driven bulk FINAL requires all pre-final SEARCHING "
            "ordinals while retrieval remains live"
        )
    if args.answerability_early_abort_below_margin and not (
        args.event_driven_generation
        and args.single_snapshot_gate
        and args.compact_answerability_prompt
        and args.answerability_structured_output
        and args.answerability_require_logprobs
    ):
        raise ValueError(
            "low-confidence decode abort requires event-driven structured "
            "single-snapshot generation with gate logprobs"
        )
    if args.generation_coalesce_ms < 0:
        raise ValueError("generation coalescing window must be non-negative")
    if args.generation_coalesce_max_batch < 1:
        raise ValueError("generation coalescing batch threshold must be positive")
    if args.generation_coalesce_max_steps < 1:
        raise ValueError("generation coalescing step bound must be positive")
    if args.generation_poll_backoff_ms < 0:
        raise ValueError("generation poll backoff must be non-negative")
    if args.event_queue_size < 1:
        raise ValueError("event queue size must be positive")
    if args.max_generation_chains < 1:
        raise ValueError("generation chain cap must be positive")
    if args.native_lookahead_lists < 1:
        raise ValueError("native scanner lookahead must be positive")
    if args.native_continuation_quantum < 0:
        raise ValueError("native continuation quantum must be non-negative")
    if (
        args.native_request_aligned_priority
        and args.native_continuation_quantum < 1
    ):
        raise ValueError(
            "request-aligned scan priority requires a positive continuation quantum"
        )
    if args.native_request_aligned_priority_burst < 0:
        raise ValueError(
            "native request-aligned priority burst must be non-negative"
        )
    if (
        args.native_request_aligned_priority_burst > 0
        and not args.native_request_aligned_priority
    ):
        raise ValueError(
            "native priority burst requires request-aligned priority"
        )
    if args.native_request_aligned_max_fifo_wait_ms < 0:
        raise ValueError(
            "native request-aligned max FIFO wait must be non-negative"
        )
    if (
        args.native_request_aligned_max_fifo_wait_ms > 0
        and not args.native_request_aligned_priority
    ):
        raise ValueError(
            "native max FIFO wait requires request-aligned priority"
        )
    if args.trace_full_prompts_max_requests < 0:
        raise ValueError("trace-full-prompts-max-requests must be non-negative")
    if not args.native_scanner_module.strip():
        raise ValueError("native scanner module must be non-empty")
    source = Path(args.native_scanner_source)
    if not source.is_file():
        raise ValueError(f"native scanner source does not exist: {source}")
    args.native_scanner_source = source.resolve()
    args.native_scanner_source_sha256 = file_sha256(args.native_scanner_source)
    header = args.native_scanner_source.with_name(
        "harmorag_scan_permit_limiter.h"
    )
    if not header.is_file():
        raise ValueError(f"native scan permit header does not exist: {header}")
    args.native_scanner_header = header.resolve()
    args.native_scanner_header_sha256 = file_sha256(
        args.native_scanner_header
    )
    spec = importlib.util.find_spec(args.native_scanner_module)
    if spec is None or spec.origin is None:
        raise ValueError(
            f"native scanner module is not installed: {args.native_scanner_module}"
        )
    module_path = Path(spec.origin)
    if not module_path.is_file():
        raise ValueError(f"native scanner module has no hashable binary: {module_path}")
    args.native_scanner_module_path = module_path.resolve()
    args.native_scanner_module_sha256 = file_sha256(
        args.native_scanner_module_path
    )


def validate_progressive_runtime_backend(retriever, args) -> None:
    """Refuse to benchmark a configured-native run on the legacy Python path."""

    retriever_path = PROJECT_ROOT / "harmorag" / "retriever" / "progressive_retriever.py"
    retriever_hash = file_sha256(retriever_path)
    if retriever_hash in KNOWN_LEGACY_PER_LIST_RETRIEVER_SHA256:
        raise RuntimeError(
            "known bootstrap4-then-Python-per-list retriever detected; "
            "the native-continuous retest must not use this implementation"
        )

    reported = None
    for name in (
        "retrieval_backend",
        "retrieval_backend_name",
        "backend_name",
    ):
        value = getattr(retriever, name, None)
        if callable(value):
            value = value()
        if value is not None:
            reported = str(value)
            break
    if reported is None:
        scanner = getattr(
            retriever,
            "native_scanner",
            getattr(retriever, "_native_scanner", None),
        )
        for name in ("retrieval_backend", "backend_name", "name"):
            value = getattr(scanner, name, None) if scanner is not None else None
            if callable(value):
                value = value()
            if value is not None:
                reported = str(value)
                break
    if (
        reported is None
        and bool(getattr(retriever, "use_native_continuous", False))
        and getattr(retriever, "_native_scanner", None) is not None
    ):
        reported = PROGRESSIVE_RETRIEVAL_BACKEND
    if reported != args.progressive_retrieval_backend:
        raise RuntimeError(
            "progressive retriever did not attest the configured native backend: "
            f"expected={args.progressive_retrieval_backend!r}, reported={reported!r}"
        )


def device_dense_retriever_class():
    from harmorag.retriever import DenseRetriever
    from harmorag.retriever.encoder import Encoder, STEncoder

    class DeviceDenseRetriever(DenseRetriever):
        def __init__(self, config, device):
            self._benchmark_device = device
            super().__init__(config)

        def load_model(self):
            if self.use_st:
                self.encoder = STEncoder(
                    model_name=self.retrieval_method,
                    model_path=self._config["retrieval_model_path"],
                    max_length=self.query_max_length,
                    use_fp16=self.use_fp16,
                    instruction=self.instruction,
                    silent=True,
                    device=self._benchmark_device,
                )
            else:
                self._check_pooling_method(self.retrieval_model_path, self.pooling_method)
                self.encoder = Encoder(
                    model_name=self.retrieval_method,
                    model_path=self.retrieval_model_path,
                    pooling_method=self.pooling_method,
                    max_length=self.query_max_length,
                    use_fp16=self.use_fp16,
                    instruction=self.instruction,
                    silent=True,
                    device=self._benchmark_device,
                )

    return DeviceDenseRetriever


def quality_and_samples(
    config,
    dataset,
    completion_s: Sequence[float],
    windows: Sequence[ArrivalWindow],
    ranked_documents: Sequence[Sequence[dict]],
    exposure_documents: Sequence[Sequence[dict]],
    extra_rows: Sequence[dict],
    max_recall_k: int = 5,
) -> Tuple[dict, dict, List[dict]]:
    max_recall_k = max(1, int(max_recall_k))
    quality = evaluate(config, dataset)
    sample_em = per_sample_em(dataset)
    sample_f1 = per_sample_saved_metric(dataset, "f1")
    sample_acc = per_sample_saved_metric(dataset, "acc")
    ranked_hits = [
        containment_row(item, documents)
        for item, documents in zip(dataset, ranked_documents)
    ]
    exposure_hits = [
        containment_row(item, documents)
        for item, documents in zip(dataset, exposure_documents)
    ]
    arrivals = [0.0] * len(dataset)
    for window in windows:
        arrivals[window.start : window.stop] = window_arrivals(window)
    samples = []
    for index, item in enumerate(dataset):
        output = getattr(item, "output", {}) or {}
        row = {
            "index": index,
            "question": str(item_field(item, "question") or dataset.question[index]),
            "golden_answers": references_for(item),
            "prediction": str(output.get("pred", "")),
            "em": sample_em[index],
            "f1": sample_f1[index],
            "acc": sample_acc[index],
            "arrival_s": arrivals[index],
            "completion_s": completion_s[index],
            "latency_s": completion_s[index] - arrivals[index],
            "ranked_hit_row": ranked_hits[index][:max_recall_k],
            "exposure_hit_row": exposure_hits[index][:max_recall_k],
        }
        row.update(extra_rows[index])
        samples.append(row)
    recall = {
        "definition": (
            "Recall@k is golden-answer phrase containment in at least one of the first k "
            "documents after punctuation/article normalization and token-boundary matching."
        ),
        "ranked_retrieval": recall_curve(ranked_hits, max_recall_k),
        "ordered_unique_generation_exposure": recall_curve(
            exposure_hits, max_recall_k
        ),
    }
    return quality, recall, samples


def run_native(args, config, measured, warmup, windows):
    import faiss
    import torch
    from harmorag.prompt import PromptTemplate
    from harmorag.utils import get_generator

    Retriever = device_dense_retriever_class()
    load_start = time.perf_counter()
    faiss.omp_set_num_threads(args.omp_threads)
    retriever = Retriever(config, args.retrieval_device)
    retriever.index.nprobe = args.nprobe
    generator = get_generator(config)
    prompt_template = PromptTemplate(
        config, system_prompt=NATIVE_SYSTEM_PROMPT, user_prompt=NATIVE_USER_PROMPT
    )
    load_s = time.perf_counter() - load_start

    def process(questions):
        documents = retriever.batch_search(questions)
        prompts = [
            prompt_template.get_string(question=q, retrieval_result=docs)
            for q, docs in zip(questions, documents)
        ]
        outputs = generator.generate(prompts, temperature=0, max_tokens=args.max_tokens)
        return documents, outputs

    if len(warmup):
        process(warmup.question)
        torch.cuda.synchronize()

    predictions = [None] * len(measured)
    completion_s = [None] * len(measured)
    ranked_documents: List[List[dict]] = [[] for _ in measured]
    retrieval_intervals = []
    generation_intervals = []
    queue_wait = []
    started = time.perf_counter()
    position = 0
    dynamic_batch_sizes = []
    while position < len(windows):
        window = windows[position]
        target = started + window.release_s
        if time.perf_counter() < target:
            time.sleep(target - time.perf_counter())
        service_start = time.perf_counter()
        # Model an online FlashRAG server: once the worker becomes available,
        # coalesce every request that has already arrived, bounded by the
        # configured generator batch capacity.  Fixed 100 ms admission windows
        # alone would force many tiny synchronous ``generate`` calls and
        # measure driver overhead rather than continuous-batching capacity.
        stop_position = position + 1
        merged_stop = window.stop
        elapsed = service_start - started
        while stop_position < len(windows):
            candidate = windows[stop_position]
            if candidate.release_s > elapsed:
                break
            if candidate.stop - window.start > args.native_dynamic_batch_limit:
                break
            merged_stop = candidate.stop
            stop_position += 1
        admitted_arrivals = []
        for admitted in windows[position:stop_position]:
            admitted_arrivals.extend(window_arrivals(admitted))
        queue_wait.extend(
            [max(0.0, elapsed - value) for value in admitted_arrivals]
        )
        questions = measured.question[window.start : merged_stop]
        dynamic_batch_sizes.append(merged_stop - window.start)
        t0 = time.perf_counter()
        documents = retriever.batch_search(questions)
        t1 = time.perf_counter()
        retrieval_intervals.append((t0 - started, t1 - started))
        prompts = [
            prompt_template.get_string(question=q, retrieval_result=docs)
            for q, docs in zip(questions, documents)
        ]
        t0 = time.perf_counter()
        outputs = generator.generate(prompts, temperature=0, max_tokens=args.max_tokens)
        t1 = time.perf_counter()
        generation_intervals.append((t0 - started, t1 - started))
        done = t1 - started
        predictions[window.start : merged_stop] = [str(value).strip() for value in outputs]
        completion_s[window.start : merged_stop] = [done] * (merged_stop - window.start)
        ranked_documents[window.start : merged_stop] = documents
        position = stop_position

    measured.update_output("pred", predictions)
    extra = [
        {
            "actual_scanned_nprobe": args.nprobe,
            "terminal_reason": "max_nprobe",
            "terminal_budget": args.nprobe,
            "answer_source": "native",
            "generation_rounds": 1,
            "search_documents_attempted": 0,
            "retrieval_backend": STOCK_RETRIEVAL_BACKEND,
        }
        for _ in measured
    ]
    quality, recall, samples = quality_and_samples(
        config,
        measured,
        completion_s,
        windows,
        ranked_documents,
        ranked_documents,
        extra,
        max_recall_k=args.topk,
    )
    if args.arrival_model == "closed_loop":
        for sample, arrival, service in zip(
            samples, session_starts, service_latency_s
        ):
            sample["arrival_s"] = float(arrival)
            sample["queue_wait_s"] = 0.0
            sample["latency_s"] = float(service)
    service_s = max(completion_s)
    return {
        "group": "native",
        "mechanisms": dict(GROUPS["native"]),
        "load_and_warmup_excluded_s": load_s,
        "service_s": service_s,
        "achieved_throughput_req_s": len(measured) / service_s,
        "latency": (
            {
                "mean_s": statistics.fmean(service_latency_s),
                "p50_s": percentile(service_latency_s, 50),
                "p90_s": percentile(service_latency_s, 90),
                "p95_s": percentile(service_latency_s, 95),
                "p99_s": percentile(service_latency_s, 99),
                "max_s": max(service_latency_s),
            }
            if args.arrival_model == "closed_loop"
            else latency_summary(completion_s, windows)
        ),
        "queue_wait": {
            "mean_s": statistics.fmean(queue_wait),
            "p95_s": percentile(queue_wait, 95),
        },
        "quality": quality,
        "recall_at_k": recall,
        "cost": {
            "actual_scanned_nprobe_mean": float(args.nprobe),
            "generation_rounds_mean": 1.0,
            "search_documents_attempted_mean": 0.0,
            "retrieval_busy_s": interval_total(retrieval_intervals),
            "generation_busy_s": interval_total(generation_intervals),
            "retrieval_generation_overlap_s": 0.0,
        },
        "mechanism_diagnostics": {
            "native_dynamic_admission": True,
            "native_dynamic_batch_limit": int(args.native_dynamic_batch_limit),
            "native_dynamic_batch_count": len(dynamic_batch_sizes),
            "native_dynamic_batch_size_mean": statistics.fmean(dynamic_batch_sizes),
            "native_dynamic_batch_size_max": max(dynamic_batch_sizes),
            "stable_count": 0,
            "full_nprobe_count": len(measured),
            "stable_then_resumed_to_full_count": 0,
            "next_batch_bootstrap_overlap_count": 0,
        },
        "samples": samples,
    }


def run_m3(args, config, measured, warmup, windows):
    """Run M3 alone with stock retrieval and stock one-shot generation.

    The only changed execution rule is a one-window rolling prefetch: while
    the generator serves window j, one retrieval worker may scan window j+1.
    Retrieval remains nprobe=128/topk=5, and no progressive answer gate is
    used, so quality should match the native path up to deterministic decoder
    reproducibility.
    """

    from concurrent.futures import ThreadPoolExecutor

    import faiss
    import torch
    from harmorag.prompt import PromptTemplate
    from harmorag.utils import get_generator

    Retriever = device_dense_retriever_class()
    load_start = time.perf_counter()
    faiss.omp_set_num_threads(args.omp_threads)
    retriever = Retriever(config, args.retrieval_device)
    retriever.index.nprobe = args.nprobe
    generator = get_generator(config)
    prompt_template = PromptTemplate(
        config, system_prompt=NATIVE_SYSTEM_PROMPT, user_prompt=NATIVE_USER_PROMPT
    )
    load_s = time.perf_counter() - load_start

    if len(warmup):
        warm_documents = retriever.batch_search(warmup.question)
        warm_prompts = [
            prompt_template.get_string(question=q, retrieval_result=docs)
            for q, docs in zip(warmup.question, warm_documents)
        ]
        generator.generate(warm_prompts, temperature=0, max_tokens=args.max_tokens)
        torch.cuda.synchronize()

    predictions = [None] * len(measured)
    completion_s = [None] * len(measured)
    ranked_documents: List[List[dict]] = [[] for _ in measured]
    retrieval_intervals = [None] * len(windows)
    generation_intervals = [None] * len(windows)
    queue_wait = []
    started = time.perf_counter()

    def retrieve_window(position: int):
        window = windows[position]
        target = started + window.release_s
        now = time.perf_counter()
        if now < target:
            time.sleep(target - now)
        t0 = time.perf_counter()
        questions = measured.question[window.start : window.stop]
        documents = retriever.batch_search(questions)
        t1 = time.perf_counter()
        return documents, (t0 - started, t1 - started)

    futures = {}
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="m3-native-retrieval") as pool:
        if windows:
            futures[0] = pool.submit(retrieve_window, 0)
        for position, window in enumerate(windows):
            if position not in futures:
                futures[position] = pool.submit(retrieve_window, position)
            documents, retrieval_interval = futures.pop(position).result()
            retrieval_intervals[position] = retrieval_interval
            queue_wait.extend(
                [max(0.0, retrieval_interval[0] - value) for value in window_arrivals(window)]
            )

            # Admit exactly one successor before current generation.  The
            # retrieval worker itself enforces the successor's release time.
            next_position = position + 1
            if next_position < len(windows) and next_position not in futures:
                futures[next_position] = pool.submit(retrieve_window, next_position)

            questions = measured.question[window.start : window.stop]
            prompts = [
                prompt_template.get_string(question=q, retrieval_result=docs)
                for q, docs in zip(questions, documents)
            ]
            t0 = time.perf_counter()
            outputs = generator.generate(
                prompts, temperature=0, max_tokens=args.max_tokens
            )
            t1 = time.perf_counter()
            generation_intervals[position] = (t0 - started, t1 - started)
            done = t1 - started
            predictions[window.start : window.stop] = [
                str(value).strip() for value in outputs
            ]
            completion_s[window.start : window.stop] = [done] * window.size
            ranked_documents[window.start : window.stop] = documents

    measured.update_output("pred", predictions)
    extra = [
        {
            "actual_scanned_nprobe": args.nprobe,
            "terminal_reason": "max_nprobe",
            "terminal_budget": args.nprobe,
            "answer_source": "m3_native_generation",
            "generation_rounds": 1,
            "search_documents_attempted": 0,
            "retrieval_backend": STOCK_RETRIEVAL_BACKEND,
        }
        for _ in measured
    ]
    quality, recall, samples = quality_and_samples(
        config,
        measured,
        completion_s,
        windows,
        ranked_documents,
        ranked_documents,
        extra,
        max_recall_k=args.topk,
    )
    valid_retrieval = [value for value in retrieval_intervals if value is not None]
    valid_generation = [value for value in generation_intervals if value is not None]
    overlap_s = interval_overlap(valid_retrieval, valid_generation)
    successor_overlap_count = sum(
        max(
            0.0,
            min(retrieval_intervals[position][1], generation_intervals[position - 1][1])
            - max(retrieval_intervals[position][0], generation_intervals[position - 1][0]),
        )
        > 0.0
        for position in range(1, len(windows))
    )
    service_s = max(completion_s)
    return {
        "group": "m3",
        "mechanisms": dict(GROUPS["m3"]),
        "load_and_warmup_excluded_s": load_s,
        "service_s": service_s,
        "achieved_throughput_req_s": len(measured) / service_s,
        "latency": latency_summary(completion_s, windows),
        "queue_wait": {
            "mean_s": statistics.fmean(queue_wait),
            "p95_s": percentile(queue_wait, 95),
        },
        "quality": quality,
        "recall_at_k": recall,
        "cost": {
            "actual_scanned_nprobe_mean": float(args.nprobe),
            "generation_rounds_mean": 1.0,
            "search_documents_attempted_mean": 0.0,
            "retrieval_busy_s": interval_total(valid_retrieval),
            "generation_busy_s": interval_total(valid_generation),
            "retrieval_generation_overlap_s": overlap_s,
        },
        "mechanism_diagnostics": {
            "stable_count": 0,
            "full_nprobe_count": len(measured),
            "stable_then_resumed_to_full_count": 0,
            "next_batch_retrieval_generation_overlap_count": successor_overlap_count,
            "next_batch_overlap_opportunity_count": max(0, len(windows) - 1),
            "max_active_retrieval_sessions": 1,
            "prefetch_depth": 1,
        },
        "samples": samples,
    }


def run_m1(args, config, measured, warmup, windows):
    import faiss
    import torch
    from harmorag.pipeline.progressive_pipeline import ProgressiveAsyncPipeline
    from harmorag.prompt import PromptTemplate
    from harmorag.retriever.progressive_retriever import ProgressiveIVFRetriever
    from harmorag.utils import get_generator

    for key, value in progressive_settings(args, incremental=False, prefetch_batches=0).items():
        config[key] = value
    load_start = time.perf_counter()
    faiss.omp_set_num_threads(args.omp_threads)
    retriever = ProgressiveIVFRetriever(config)
    validate_progressive_runtime_backend(retriever, args)
    generator = get_generator(config)
    prompt_template = PromptTemplate(
        config, system_prompt=NATIVE_SYSTEM_PROMPT, user_prompt=NATIVE_USER_PROMPT
    )
    load_s = time.perf_counter() - load_start

    def retrieve(questions, query_indices, batch_id):
        session = retriever.start_batch(
            questions,
            batch_id=batch_id,
            query_indices=query_indices,
            pause_after_bootstrap=False,
        )
        terminal = {}
        try:
            while not session.terminal:
                event = retriever.advance_batch(session)
                for snapshot in event.snapshots:
                    if snapshot.finished:
                        terminal[int(snapshot.query_index)] = snapshot
        finally:
            close_batch = getattr(retriever, "close_batch", None)
            close_counts = close_batch(session) if close_batch is not None else {}
        if set(terminal) != set(int(value) for value in query_indices):
            raise RuntimeError("M1 did not capture every query terminal snapshot")
        actual_scans = {
            int(index): int(
                (close_counts or {}).get(
                    int(index),
                    getattr(terminal[int(index)], "actual_scanned_nprobe", terminal[int(index)].scanned_budget),
                )
            )
            for index in query_indices
        }
        id_rows = [
            [int(value) for value in terminal[int(index)].document_ids[: args.topk] if int(value) >= 0]
            for index in query_indices
        ]
        return terminal, actual_scans, id_rows, materialize_rows(retriever, id_rows)

    if len(warmup):
        indices = tuple(range(len(warmup)))
        _, _, _, documents = retrieve(warmup.question, indices, -1)
        prompts = [
            prompt_template.get_string(question=q, retrieval_result=docs)
            for q, docs in zip(warmup.question, documents)
        ]
        generator.generate(prompts, temperature=0, max_tokens=args.max_tokens)
        torch.cuda.synchronize()

    predictions = [None] * len(measured)
    completion_s = [None] * len(measured)
    ranked_documents: List[List[dict]] = [[] for _ in measured]
    terminal_reasons = [None] * len(measured)
    terminal_budgets = [None] * len(measured)
    actual_scans = [None] * len(measured)
    prefetch_advised_lists = [0] * len(measured)
    prefetch_advised_bytes = [0] * len(measured)
    prefetch_advice_errors = [0] * len(measured)
    prefetch_high_water = [0] * len(measured)
    heap_updates = [0] * len(measured)
    unchanged_heap_lists = [0] * len(measured)
    ranked_ids: List[List[int]] = [[] for _ in measured]
    retrieval_intervals = []
    generation_intervals = []
    queue_wait = []
    started = time.perf_counter()
    for window in windows:
        target = started + window.release_s
        if time.perf_counter() < target:
            time.sleep(target - time.perf_counter())
        service_start = time.perf_counter()
        queue_wait.extend(
            [max(0.0, service_start - started - value) for value in window_arrivals(window)]
        )
        indices = tuple(range(window.start, window.stop))
        questions = measured.question[window.start : window.stop]
        t0 = time.perf_counter()
        terminal, window_actual_scans, ids, documents = retrieve(
            questions, indices, window.batch_id
        )
        t1 = time.perf_counter()
        retrieval_intervals.append((t0 - started, t1 - started))
        prompts = [
            prompt_template.get_string(question=q, retrieval_result=docs)
            for q, docs in zip(questions, documents)
        ]
        t0 = time.perf_counter()
        outputs = generator.generate(prompts, temperature=0, max_tokens=args.max_tokens)
        t1 = time.perf_counter()
        generation_intervals.append((t0 - started, t1 - started))
        done = t1 - started
        predictions[window.start : window.stop] = [str(value).strip() for value in outputs]
        completion_s[window.start : window.stop] = [done] * window.size
        ranked_documents[window.start : window.stop] = documents
        ranked_ids[window.start : window.stop] = ids
        for index in indices:
            snapshot = terminal[index]
            reason = str(snapshot.stop_reason)
            terminal_reasons[index] = "wrd_stable" if reason == "wrd_stable" else "max_nprobe"
            terminal_budgets[index] = int(snapshot.scanned_budget)
            actual_scans[index] = int(window_actual_scans[index])
            prefetch_advised_lists[index] = int(
                getattr(snapshot, "prefetch_advised_lists", 0)
            )
            prefetch_advised_bytes[index] = int(
                getattr(snapshot, "prefetch_advised_bytes", 0)
            )
            prefetch_advice_errors[index] = int(
                getattr(snapshot, "prefetch_advice_errors", 0)
            )
            prefetch_high_water[index] = int(
                getattr(snapshot, "prefetch_high_water", 0)
            )
            heap_updates[index] = int(getattr(snapshot, "heap_updates", 0))
            unchanged_heap_lists[index] = int(
                getattr(snapshot, "unchanged_heap_lists", 0)
            )

    measured.update_output("pred", predictions)
    extra = [
        {
            "actual_scanned_nprobe": actual_scans[index],
            "terminal_reason": terminal_reasons[index],
            "terminal_budget": terminal_budgets[index],
            "answer_source": "m1_native_generation",
            "generation_rounds": 1,
            "search_documents_attempted": 0,
            "ranked_document_ids": ranked_ids[index],
            "retrieval_backend": PROGRESSIVE_RETRIEVAL_BACKEND,
            "prefetch_advised_lists": prefetch_advised_lists[index],
            "prefetch_advised_bytes": prefetch_advised_bytes[index],
            "prefetch_advice_errors": prefetch_advice_errors[index],
            "prefetch_high_water": prefetch_high_water[index],
            "heap_updates": heap_updates[index],
            "unchanged_heap_lists": unchanged_heap_lists[index],
        }
        for index in range(len(measured))
    ]
    quality, recall, samples = quality_and_samples(
        config,
        measured,
        completion_s,
        windows,
        ranked_documents,
        ranked_documents,
        extra,
        max_recall_k=args.topk,
    )
    stable = [reason == "wrd_stable" for reason in terminal_reasons]
    full = [budget == args.nprobe for budget in terminal_budgets]
    if any(is_stable and is_full for is_stable, is_full in zip(stable, full)):
        raise RuntimeError("a WRD-stable M1 query was incorrectly resumed to nprobe=128")
    service_s = max(completion_s)
    return {
        "group": "m1",
        "mechanisms": dict(GROUPS["m1"]),
        "load_and_warmup_excluded_s": load_s,
        "service_s": service_s,
        "achieved_throughput_req_s": len(measured) / service_s,
        "latency": latency_summary(completion_s, windows),
        "queue_wait": {
            "mean_s": statistics.fmean(queue_wait),
            "p95_s": percentile(queue_wait, 95),
        },
        "quality": quality,
        "recall_at_k": recall,
        "cost": {
            "actual_scanned_nprobe_mean": statistics.fmean(actual_scans),
            "actual_scanned_nprobe_p50": percentile(actual_scans, 50),
            "actual_scanned_nprobe_p95": percentile(actual_scans, 95),
            "logical_terminal_nprobe_mean": statistics.fmean(terminal_budgets),
            "estimated_scan_reduction": 1.0 - statistics.fmean(actual_scans) / args.nprobe,
            "generation_rounds_mean": 1.0,
            "search_documents_attempted_mean": 0.0,
            "retrieval_busy_s": interval_total(retrieval_intervals),
            "generation_busy_s": interval_total(generation_intervals),
            "retrieval_generation_overlap_s": 0.0,
            "prefetch_advised_lists_mean": statistics.fmean(
                prefetch_advised_lists
            ),
            "prefetch_advised_gib": sum(prefetch_advised_bytes) / (1024 ** 3),
            "prefetch_advice_errors": sum(prefetch_advice_errors),
            "prefetch_ahead_lists_max": max(
                max(0, high - scanned)
                for high, scanned in zip(prefetch_high_water, actual_scans)
            ),
            "unchanged_heap_list_rate": (
                sum(unchanged_heap_lists) / sum(actual_scans)
                if sum(actual_scans)
                else 0.0
            ),
        },
        "mechanism_diagnostics": {
            "stable_count": sum(stable),
            "stable_rate": sum(stable) / len(measured),
            "full_nprobe_count": sum(full),
            "stable_then_resumed_to_full_count": 0,
            "next_batch_bootstrap_overlap_count": 0,
        },
        "samples": samples,
    }


def run_progressive(args, config, measured, warmup, windows, *, mode: str):
    import faiss
    import torch
    from harmorag.pipeline.progressive_pipeline import (
        ProgressiveAsyncPipeline,
        _RetrievalFailed,
        _RetrievalFinished,
        is_no_answer,
    )

    profile_name = args.shared_mechanism_profile
    effective_mechanisms = (
        SHARED_MECHANISM_PROFILES[profile_name]
        if profile_name is not None
        else GROUPS[mode]
    )
    enable_m2 = mode in {"m2", "m1_m2", "m1_m2_m3", "full"}
    enable_m3b = bool(effective_mechanisms["m3_next_batch_bootstrap"])
    enable_m1 = bool(effective_mechanisms["m1"])
    if not enable_m2:
        raise ValueError(f"unsupported progressive ablation mode: {mode}")

    for key, value in progressive_settings(
        args, incremental=True, prefetch_batches=1 if enable_m3b else 0
    ).items():
        config[key] = value
    # M2-only must consume genuine progressive heap snapshots while scanning
    # all the way to the native endpoint.  Disabling WRD in the scanner keeps
    # answer-gate cancellation intact without attributing M1's early-stop
    # savings to M2.
    config["progressive_enable_wrd"] = enable_m1

    class ArrivalPipeline(ProgressiveAsyncPipeline):
        def set_benchmark(self, release_windows):
            self._arrival_windows = list(release_windows)
            self._benchmark_started = None
            self._completion_s = []
            self._generation_intervals = []
            self._retrieval_intervals = []
            self._retrieval_prepare_intervals_ns = []
            self._retrieval_scan_intervals_ns = []
            self._generation_service_intervals_ns = []
            self._retrieval_prepare_intervals_by_query_ns = {}
            self._retrieval_scan_intervals_by_query_ns = {}
            self._generation_service_intervals_by_query_ns = {}
            self._generation_attempts_by_query_ns = {}
            self._generation_request_query_index = {}
            self._benchmark_started_ns = None
            self._bootstrap_ready_s = {}
            self._session_start_s = {}
            self._max_active_sessions_observed = 0
            self._bootstrap_query_indices_by_position = {}
            self._bootstrap_generation_started_positions = set()
            self._bootstrap_event_consumed_positions = set()
            self._bootstrap_event_consumed_s = {}
            self._m2_latch_consumed_s = {}
            self._bootstrap_generation_start_s = {}
            self._lookahead_max_budget_before_promotion = {}
            self._lookahead_search_before_current_final_count = 0
            self._lookahead_max_budget_while_previous_incomplete = {}
            self._lookahead_post_bootstrap_event_before_previous_answer_count = 0
            self._native_sessions_started_count = 0
            self._native_sessions_closed_count = 0
            self._native_close_accounted_query_count = 0
            self._native_scan_permit_diagnostics = []
            self._generation_call_trace = []
            self._generation_call_context = None
            self._event_request_trace_index = {}
            self._generation_engine_step_trace = []
            self._generation_engine_idle_step_count = 0
            self._generation_sync_call_count = 0
            self._generation_add_request_call_count = 0
            self._generation_step_call_count = 0
            self._generation_active_request_count = 0
            self._generation_active_request_high_water = 0
            self._event_queue_high_water = 0
            self._event_queue_full_block_count = 0
            self._event_queue_full_block_s = 0.0

        def _install_generation_call_tracer(self):
            if getattr(self, "_generation_call_tracer_installed", False):
                return
            original_generate = self.generator.generate
            original_add_requests = getattr(self.generator, "add_requests", None)
            original_step_requests = getattr(self.generator, "step_requests", None)
            self._generation_call_tracer_installed = True
            self._original_generator_generate = original_generate

            def traced_generate(prompts, **params):
                context = dict(self._generation_call_context or {})
                started = time.perf_counter()
                result = None
                error = None
                self._generation_sync_call_count += 1
                try:
                    result = original_generate(prompts, **params)
                    return result
                except BaseException as caught:
                    error = caught
                    raise
                finally:
                    stopped = time.perf_counter()
                    if args.trace_generation_details:
                        cache_stats = []
                        if (
                            isinstance(result, tuple)
                            and len(result) == 2
                            and isinstance(result[1], list)
                        ):
                            cache_stats = [json_safe(row) for row in result[1]]
                        include_text = bool(
                            args.num_requests
                            <= args.trace_full_prompts_max_requests
                        )
                        tokenizer = getattr(self.generator, "tokenizer", None)
                        prompt_rows = [
                            prompt_trace_payload(
                                prompt,
                                tokenizer,
                                include_text=include_text,
                            )
                            for prompt in prompts
                        ]
                        cached_tokens = sum(
                            int((row or {}).get("num_cached_tokens") or 0)
                            for row in cache_stats
                            if isinstance(row, dict)
                        )
                        prompt_tokens = sum(
                            int((row or {}).get("prompt_tokens") or 0)
                            for row in cache_stats
                            if isinstance(row, dict)
                        )
                        origin = self._benchmark_started or started
                        self._generation_call_trace.append(
                            {
                                "call_index": len(self._generation_call_trace),
                                "kind": str(context.get("kind") or "unknown"),
                                "query_indices": [
                                    int(value)
                                    for value in context.get("query_indices", ())
                                ],
                                "batch_size": len(prompts),
                                "start_s": started - origin,
                                "end_s": stopped - origin,
                                "duration_s": stopped - started,
                                "prompts": prompt_rows,
                                "cache_stats": cache_stats,
                                "cached_tokens": cached_tokens,
                                "prompt_tokens": prompt_tokens,
                                "cached_token_ratio": (
                                    cached_tokens / prompt_tokens
                                    if prompt_tokens
                                    else None
                                ),
                                "generation_params": json_safe(params),
                                "error": (
                                    f"{type(error).__name__}: {error}"
                                    if error is not None
                                    else None
                                ),
                            }
                        )

            self.generator.generate = traced_generate
            if callable(original_add_requests) and callable(original_step_requests):
                self._original_generator_add_requests = original_add_requests
                self._original_generator_step_requests = original_step_requests

                def record_service_intervals(completed):
                    for row in completed:
                        interval = row.get("service_interval_ns")
                        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                            raise RuntimeError(
                                "AsyncLLMEngine completion is missing its "
                                "admission-to-terminal service interval"
                            )
                        start_ns, stop_ns = map(int, interval)
                        if stop_ns <= start_ns:
                            raise RuntimeError(
                                "AsyncLLMEngine service interval has invalid bounds"
                            )
                        self._generation_service_intervals_ns.append(
                            (start_ns, stop_ns)
                        )
                        request_id = str(row.get("request_id"))
                        query_index = self._generation_request_query_index.pop(
                            request_id, None
                        )
                        if query_index is None:
                            raise RuntimeError(
                                "AsyncLLMEngine completion has no query-index "
                                f"mapping: request_id={request_id}"
                            )
                        self._generation_service_intervals_by_query_ns.setdefault(
                            int(query_index), []
                        ).append((start_ns, stop_ns))
                        first_token_ns = row.get("first_token_ns")
                        if first_token_ns is None:
                            raise RuntimeError(
                                "AsyncLLMEngine completion is missing first-token timing"
                            )
                        first_token_ns = int(first_token_ns)
                        if not start_ns <= first_token_ns <= stop_ns:
                            raise RuntimeError(
                                "AsyncLLMEngine first-token timestamp is outside "
                                "its service interval"
                            )
                        metadata = dict(row.get("metadata") or {})
                        self._generation_attempts_by_query_ns.setdefault(
                            int(query_index), []
                        ).append(
                            {
                                "request_id": request_id,
                                "kind": str(metadata.get("kind") or "unknown"),
                                "start_ns": start_ns,
                                "first_token_ns": first_token_ns,
                                "stop_ns": stop_ns,
                            }
                        )

                def traced_add_requests(prompts, **params):
                    prompt_list = (
                        [prompts]
                        if isinstance(prompts, (str, dict))
                        else list(prompts)
                    )
                    metadata = list(
                        params.get("request_metadata")
                        or ({} for _ in prompt_list)
                    )
                    if len(metadata) != len(prompt_list):
                        raise RuntimeError(
                            "generation request metadata length does not match "
                            "the submitted prompt batch"
                        )
                    if not args.trace_generation_details:
                        request_ids = original_add_requests(prompt_list, **params)
                        if len(request_ids) != len(metadata):
                            raise RuntimeError(
                                "generation request-id count does not match metadata"
                            )
                        for request_id, row in zip(request_ids, metadata):
                            query_index = (row or {}).get("query_index")
                            if query_index is None:
                                raise RuntimeError(
                                    "generation request metadata is missing query_index"
                                )
                            self._generation_request_query_index[str(request_id)] = int(
                                query_index
                            )
                        self._generation_add_request_call_count += 1
                        self._generation_active_request_count += len(request_ids)
                        self._generation_active_request_high_water = max(
                            self._generation_active_request_high_water,
                            self._generation_active_request_count,
                        )
                        return request_ids
                    started = time.perf_counter()
                    request_ids = original_add_requests(prompt_list, **params)
                    if len(request_ids) != len(metadata):
                        raise RuntimeError(
                            "generation request-id count does not match metadata"
                        )
                    for request_id, row in zip(request_ids, metadata):
                        query_index = (row or {}).get("query_index")
                        if query_index is None:
                            raise RuntimeError(
                                "generation request metadata is missing query_index"
                            )
                        self._generation_request_query_index[str(request_id)] = int(
                            query_index
                        )
                    self._generation_add_request_call_count += 1
                    self._generation_active_request_count += len(request_ids)
                    stopped = time.perf_counter()
                    origin = self._benchmark_started or started
                    include_text = bool(
                        args.num_requests
                        <= args.trace_full_prompts_max_requests
                    )
                    tokenizer = getattr(self.generator, "tokenizer", None)
                    kinds = {
                        str((row or {}).get("kind") or "unknown")
                        for row in metadata
                    }
                    entry = {
                        "call_index": len(self._generation_call_trace),
                        "scheduler": "event_driven_add_step",
                        "kind": next(iter(kinds)) if len(kinds) == 1 else "mixed",
                        "query_indices": [
                            int((row or {}).get("query_index"))
                            for row in metadata
                        ],
                        "batch_size": len(prompt_list),
                        "request_ids": [str(value) for value in request_ids],
                        "start_s": started - origin,
                        "submit_end_s": stopped - origin,
                        "submit_duration_s": stopped - started,
                        "end_s": None,
                        "duration_s": None,
                        "prompts": [
                            prompt_trace_payload(
                                prompt,
                                tokenizer,
                                include_text=include_text,
                            )
                            for prompt in prompt_list
                        ],
                        "cache_stats": [],
                        "cache_stats_by_request": {},
                        "completions": [],
                        "cached_tokens": 0,
                        "prompt_tokens": 0,
                        "cached_token_ratio": None,
                        "generation_params": json_safe(params),
                        "error": None,
                    }
                    self._generation_call_trace.append(entry)
                    for request_id in request_ids:
                        self._event_request_trace_index[str(request_id)] = entry
                    self._generation_active_request_high_water = max(
                        self._generation_active_request_high_water,
                        self._generation_active_request_count,
                    )
                    return request_ids

                def traced_step_requests():
                    if not args.trace_generation_details:
                        completed = original_step_requests()
                        record_service_intervals(completed)
                        self._generation_step_call_count += 1
                        self._generation_active_request_count = max(
                            0,
                            self._generation_active_request_count
                            - len(completed),
                        )
                        return completed
                    started = time.perf_counter()
                    active_before = len(self._event_request_trace_index)
                    completed = original_step_requests()
                    record_service_intervals(completed)
                    self._generation_step_call_count += 1
                    self._generation_active_request_count = max(
                        0,
                        self._generation_active_request_count - len(completed),
                    )
                    stopped = time.perf_counter()
                    origin = self._benchmark_started or started
                    completed_ids = []
                    for row in completed:
                        request_id = str(row.get("request_id"))
                        completed_ids.append(request_id)
                        entry = self._event_request_trace_index.pop(
                            request_id, None
                        )
                        if entry is None:
                            continue
                        cache_stat = json_safe(row.get("cache_stats") or {})
                        entry["cache_stats_by_request"][request_id] = cache_stat
                        entry["completions"].append(
                            {
                                "request_id": request_id,
                                "end_s": stopped - origin,
                                "metadata": json_safe(row.get("metadata") or {}),
                                "cache_stats": cache_stat,
                            }
                        )
                        if len(entry["completions"]) == entry["batch_size"]:
                            entry["cache_stats"] = [
                                entry["cache_stats_by_request"].get(
                                    str(request_id), {}
                                )
                                for request_id in entry["request_ids"]
                            ]
                            entry.pop("cache_stats_by_request", None)
                            entry["end_s"] = stopped - origin
                            entry["duration_s"] = stopped - (
                                origin + float(entry["start_s"])
                            )
                            entry["cached_tokens"] = sum(
                                int((value or {}).get("num_cached_tokens") or 0)
                                for value in entry["cache_stats"]
                            )
                            entry["prompt_tokens"] = sum(
                                int((value or {}).get("prompt_tokens") or 0)
                                for value in entry["cache_stats"]
                            )
                            entry["cached_token_ratio"] = (
                                entry["cached_tokens"] / entry["prompt_tokens"]
                                if entry["prompt_tokens"]
                                else None
                            )
                    # The event-driven scheduler may poll step_requests hundreds
                    # of thousands of times while no request completes.  Keeping
                    # one Python dictionary for every empty poll perturbs the
                    # latency being measured and can create very large JSON
                    # artifacts.  Completion steps retain the exact audit trail;
                    # empty polls are represented by a lossless aggregate count.
                    if completed_ids:
                        self._generation_engine_step_trace.append(
                            {
                                "step_index": self._generation_step_call_count - 1,
                                "start_s": started - origin,
                                "end_s": stopped - origin,
                                "duration_s": stopped - started,
                                "active_before": active_before,
                                "completed_request_ids": completed_ids,
                                "active_after": len(self._event_request_trace_index),
                            }
                        )
                    else:
                        self._generation_engine_idle_step_count += 1
                    return completed

                self.generator.add_requests = traced_add_requests
                self.generator.step_requests = traced_step_requests

        def _ready_generation_batch(
            self, states, cancelled=None, cancellation_lock=None
        ):
            query_indices, prompts = super()._ready_generation_batch(
                states, cancelled, cancellation_lock
            )
            if prompts:
                self._generation_call_context = {
                    "kind": "SEARCHING",
                    "query_indices": tuple(int(value) for value in query_indices),
                }
            return query_indices, prompts

        def _submit_event_search_requests(
            self, states, cancelled, cancellation_lock
        ):
            """Timestamp M3 bootstrap admission at the actual add-request edge.

            Recording this after ``step_requests`` is racy: a short @4 request
            can be admitted and complete in the same engine step, leaving
            ``generation_inflight`` false by the time the outer tracer regains
            control.  The production method below only builds and submits
            requests, so its call boundary is the unambiguous admission point.
            """

            admission_time = time.perf_counter() - self._benchmark_started
            submitted = super()._submit_event_search_requests(
                states, cancelled, cancellation_lock
            )
            if not submitted:
                return submitted
            for position, query_indices in tuple(
                self._bootstrap_query_indices_by_position.items()
            ):
                if (
                    position in self._bootstrap_generation_started_positions
                    or position not in self._bootstrap_event_consumed_positions
                ):
                    continue
                if any(
                    states[index].first_search_generation_budget is not None
                    and states[index].generation_inflight
                    for index in query_indices
                ):
                    self._bootstrap_generation_started_positions.add(position)
                    self._bootstrap_generation_start_s.setdefault(
                        position, admission_time
                    )
            return submitted

        def _final_generation_once(self, states, cancelled, cancellation_lock):
            query_indices = tuple(
                index
                for index, state in enumerate(states)
                if state.answer is None
                and state.final_generation_pending
                and not state.final_generation_done
            )
            if not query_indices:
                return False
            previous = self._generation_call_context
            self._generation_call_context = {
                "kind": "FINAL",
                "query_indices": query_indices,
            }
            try:
                return super()._final_generation_once(
                    states, cancelled, cancellation_lock
                )
            finally:
                self._generation_call_context = previous

        @staticmethod
        def _snapshot_budget(snapshot):
            return int(getattr(snapshot, "scanned_budget", snapshot.budget))

        def _is_valid_m2_first_latch_event(self, event):
            """Accept the physical target barrier or a genuine earlier terminal.

            WRD is observed from the first scanned list, so a query may become
            terminal before the configured M2 first-snapshot target.  The
            production pipeline intentionally latches that final heap instead
            of resuming retrieval merely to manufacture an @target snapshot.
            Keep the benchmark audit consistent with that production rule.
            """

            if int(event.budget) == int(args.bootstrap_nprobe):
                return True
            snapshots = tuple(event.snapshots)
            return bool(
                snapshots
                and all(
                    self._snapshot_budget(snapshot) < int(args.bootstrap_nprobe)
                    and self._snapshot_is_retrieval_terminal(snapshot)
                    for snapshot in snapshots
                )
            )

        def _invalid_first_latch_message(self, event, *, continuous):
            budgets = [
                self._snapshot_budget(snapshot) for snapshot in event.snapshots
            ]
            terminal = [
                self._snapshot_is_retrieval_terminal(snapshot)
                for snapshot in event.snapshots
            ]
            prefix = "continuous " if continuous else ""
            return (
                f"{prefix}progressive retrieval did not begin at configured "
                f"bootstrap nprobe={args.bootstrap_nprobe} or a genuine "
                "WRD terminal before that target: "
                f"event_budget={event.budget}, snapshot_budgets={budgets}, "
                f"snapshot_terminal={terminal}"
            )

        def _consume_event(self, event, states):
            super()._consume_event(event, states)
            position = next(
                (
                    index
                    for index, window in enumerate(self._arrival_windows)
                    if int(window.batch_id) == int(event.batch_id)
                ),
                None,
            )
            if self._is_valid_m2_first_latch_event(event):
                if position is None:
                    raise RuntimeError(
                        "M2 first-latch event references an unknown arrival batch: "
                        f"{event.batch_id}"
                    )
                self._bootstrap_event_consumed_positions.add(position)
                self._bootstrap_event_consumed_s.setdefault(
                    position,
                    time.perf_counter() - self._benchmark_started,
                )
            members = tuple(
                (getattr(self, "_runtime_m2_batch_members", None) or {}).get(
                    int(event.batch_id), ()
                )
            )
            if position is not None and not members:
                window = self._arrival_windows[position]
                members = tuple(range(window.start, window.stop))
            if (
                position is not None
                and members
                and all(
                    states[index].answer is not None
                    or states[index].bootstrap_snapshot is not None
                    for index in members
                )
            ):
                self._m2_latch_consumed_s.setdefault(
                    position,
                    time.perf_counter() - self._benchmark_started,
                )

        def _generate_once(self, states, cancelled, cancellation_lock):
            answers_before = [state.answer for state in states]
            earliest_incomplete = len(self._arrival_windows)
            for position, window in enumerate(self._arrival_windows):
                if any(
                    self._completion_s[index] is None
                    for index in range(window.start, window.stop)
                ):
                    earliest_incomplete = position
                    break

            # A lookahead batch may expose its @4 event to the main queue, but
            # neither SEARCHING nor FINAL generation is allowed before that
            # batch is promoted.  Keep its already-selected bootstrap top-1
            # intact and restore it after the current batch's generation call.
            hidden_lookahead = []
            for position, query_indices in tuple(
                self._bootstrap_query_indices_by_position.items()
            ):
                if not enable_m3b or position <= earliest_incomplete:
                    continue
                for index in query_indices:
                    state = states[index]
                    if state.answer is not None:
                        continue
                    hidden_lookahead.append(
                        (
                            index,
                            state.pending_documents,
                            state.need_document,
                            state.final_generation_pending,
                        )
                    )
                    state.pending_documents = deque()
                    state.need_document = False
                    state.final_generation_pending = False

            # Record the first real generation for a batch.  The current batch
            # begins from its locked @4 top-1 even if later heap events are
            # already queued; a lookahead records this only after promotion.
            for position, query_indices in tuple(
                self._bootstrap_query_indices_by_position.items()
            ):
                if (
                    position > earliest_incomplete
                    or position in self._bootstrap_generation_started_positions
                    or position not in self._bootstrap_event_consumed_positions
                ):
                    continue
                unanswered = [
                    index for index in query_indices
                    if states[index].answer is None
                ]
                bootstrap_pending = [
                    index for index in unanswered
                    if states[index].pending_documents
                    and states[index].bootstrap_snapshot is not None
                    and states[index].pending_documents[0][2]
                    is states[index].bootstrap_snapshot
                ]
                if unanswered and len(bootstrap_pending) == len(unanswered):
                    self._bootstrap_generation_started_positions.add(position)
                    self._bootstrap_generation_start_s[position] = (
                        time.perf_counter() - self._benchmark_started
                    )

            started = time.perf_counter()
            active_started_ns = time.time_ns()
            try:
                generated = super()._generate_once(
                    states, cancelled, cancellation_lock
                )
            finally:
                for (
                    index,
                    pending_documents,
                    need_document,
                    final_generation_pending,
                ) in hidden_lookahead:
                    state = states[index]
                    state.pending_documents = pending_documents
                    state.need_document = need_document
                    if state.answer is None and not state.final_generation_done:
                        state.final_generation_pending = final_generation_pending
                self._generation_call_context = None
            stopped = time.perf_counter()
            active_stopped_ns = time.time_ns()
            if generated:
                self._generation_intervals.append(
                    (started - self._benchmark_started, stopped - self._benchmark_started)
                )
                self._generation_service_intervals_ns.append(
                    (active_started_ns, active_stopped_ns)
                )
                now = stopped - self._benchmark_started
                for index, state in enumerate(states):
                    if (
                        answers_before[index] is None
                        and state.answer is not None
                        and self._completion_s[index] is None
                    ):
                        self._completion_s[index] = now
            return generated

        def _drive_event_generation_once(
            self, states, cancelled, cancellation_lock
        ):
            answers_before = [state.answer for state in states]
            inflight_before = [state.generation_inflight for state in states]
            started = time.perf_counter()
            generated = super()._drive_event_generation_once(
                states, cancelled, cancellation_lock
            )
            stopped = time.perf_counter()
            if generated:
                self._generation_intervals.append(
                    (
                        started - self._benchmark_started,
                        stopped - self._benchmark_started,
                    )
                )
            now = stopped - self._benchmark_started
            admission_time = started - self._benchmark_started
            for index, state in enumerate(states):
                if (
                    not inflight_before[index]
                    and state.generation_inflight
                    and state.first_search_generation_budget is not None
                ):
                    # The native retrieval worker registers newly released
                    # windows concurrently with this generation tracer.  Walk
                    # an immutable snapshot: diagnostics must never make the
                    # production scheduler fail when a batch is admitted at
                    # this exact edge.
                    for position, query_indices in tuple(
                        self._bootstrap_query_indices_by_position.items()
                    ):
                        if index in query_indices:
                            self._bootstrap_generation_started_positions.add(position)
                            # Parent event generation admits ready requests before
                            # stepping the engine.  A previous batch may complete
                            # in that same step, so using ``stopped`` here falsely
                            # reports no overlap even though admission happened
                            # first.  Record the call entry, which is a conservative
                            # upper bound on the actual add_requests timestamp.
                            self._bootstrap_generation_start_s.setdefault(
                                position, admission_time
                            )
                            break
                if (
                    answers_before[index] is None
                    and state.answer is not None
                    and self._completion_s[index] is None
                ):
                    self._completion_s[index] = now
            return generated

        def _commit_pending_single_snapshot_answers(
            self, states, cancelled, cancellation_lock
        ):
            answers_before = [state.answer for state in states]
            changed = super()._commit_pending_single_snapshot_answers(
                states, cancelled, cancellation_lock
            )
            now = time.perf_counter() - self._benchmark_started
            for index, state in enumerate(states):
                if (
                    answers_before[index] is None
                    and state.answer is not None
                    and self._completion_s[index] is None
                ):
                    self._completion_s[index] = now
            return changed

        def _retrieval_worker(
            self, questions, event_queue, cancelled, cancellation_lock, stop_event
        ):
            continuous_m3 = bool(enable_m3b and args.event_driven_generation)
            all_sessions = []
            closed_session_ids = set()
            continuous_session_started_at = {}
            bootstrap_parked_session_ids = set()

            def merge_close_accounting(counts):
                if not isinstance(counts, dict):
                    raise RuntimeError(
                        "native close_batch() must return query-index scan counts"
                    )
                high_water = getattr(self, "_runtime_actual_scanned_nprobe", None)
                if high_water is None:
                    raise RuntimeError(
                        "native close accounting has no runtime physical high-water"
                    )
                for raw_query_index, raw_actual in counts.items():
                    query_index = int(raw_query_index)
                    actual = int(raw_actual)
                    if not 0 <= query_index < len(high_water):
                        raise RuntimeError(
                            "native close accounting returned an invalid query index: "
                            f"{query_index}"
                        )
                    if not 0 <= actual <= args.nprobe:
                        raise RuntimeError(
                            "native close accounting returned an invalid physical scan: "
                            f"query={query_index}, actual={actual}"
                        )
                    high_water[query_index] = max(high_water[query_index], actual)
                self._native_close_accounted_query_count += len(counts)

            def close_one(session):
                identity = id(session)
                if identity in closed_session_ids:
                    return
                bootstrap_parked_session_ids.discard(identity)
                close_batch = getattr(self.retriever, "close_batch", None)
                if close_batch is None:
                    raise RuntimeError(
                        "native ArrivalPipeline requires retriever.close_batch()"
                    )
                merge_close_accounting(close_batch(session))
                self._retrieval_scan_intervals_ns.extend(
                    tuple(
                        (int(start_ns), int(stop_ns))
                        for start_ns, stop_ns in getattr(
                            session, "scan_active_intervals_ns", ()
                        )
                    )
                )
                for state in session.states:
                    query_index = int(state.query_index)
                    query_intervals = [
                        (int(start_ns), int(stop_ns))
                        for start_ns, stop_ns in getattr(
                            state, "scan_active_intervals_ns", ()
                        )
                    ]
                    self._retrieval_scan_intervals_by_query_ns.setdefault(
                        query_index, []
                    ).extend(query_intervals)
                permit_diagnostics = json_safe(
                    getattr(session, "scan_permit_diagnostics", {}) or {}
                )
                if permit_diagnostics:
                    self._native_scan_permit_diagnostics.append(
                        {
                            "batch_id": getattr(session, "batch_id", None),
                            **permit_diagnostics,
                        }
                    )
                started_at = continuous_session_started_at.pop(identity, None)
                if started_at is not None:
                    stopped_at = time.perf_counter()
                    self._retrieval_intervals.append(
                        (
                            started_at - self._benchmark_started,
                            stopped_at - self._benchmark_started,
                        )
                    )
                closed_session_ids.add(identity)
                self._native_sessions_closed_count += 1

            def close_all():
                for session in all_sessions:
                    close_one(session)

            try:
                active = deque()
                parked = {}
                next_window = 0
                max_sessions = (
                    int(args.max_active_retrieval_sessions)
                    if args.full_retrieval_continuous_batching
                    else (2 if enable_m3b else 1)
                )

                def emit(item):
                    while not stop_event.is_set():
                        attempt_started = time.perf_counter()
                        try:
                            event_queue.put(item, timeout=0.1)
                            scheduler_wakeup = getattr(
                                self, "_runtime_scheduler_wakeup_event", None
                            )
                            if scheduler_wakeup is not None:
                                scheduler_wakeup.set()
                            self._event_queue_high_water = max(
                                self._event_queue_high_water,
                                int(event_queue.qsize()),
                            )
                            return True
                        except queue.Full:
                            self._event_queue_full_block_count += 1
                            self._event_queue_full_block_s += (
                                time.perf_counter() - attempt_started
                            )
                            continue
                    return False

                def elapsed():
                    return time.perf_counter() - self._benchmark_started

                def released(window):
                    return elapsed() >= window.release_s

                def earliest_incomplete_window():
                    for position, window in enumerate(self._arrival_windows):
                        if any(
                            self._completion_s[index] is None
                            for index in range(window.start, window.stop)
                        ):
                            return position
                    return len(self._arrival_windows)

                def allowed_to_start(position):
                    if args.full_retrieval_continuous_batching:
                        return True
                    if continuous_m3 and args.rolling_m3_scanners:
                        # Scanner slots track live native sessions rather than
                        # logical generation windows. A completed scanner is
                        # immediately replaced by the next released window,
                        # while generation admission is bounded separately.
                        return True
                    earliest = earliest_incomplete_window()
                    lookahead = 1 if enable_m3b else 0
                    return position <= earliest + lookahead

                def timed_advance(session):
                    # Legacy synchronous control: a promoted lookahead resumes
                    # exactly once on its first post-bootstrap advance.
                    t0 = time.perf_counter()
                    event = self.retriever.advance_batch(session)
                    t1 = time.perf_counter()
                    self._retrieval_intervals.append(
                        (t0 - self._benchmark_started, t1 - self._benchmark_started)
                    )
                    self._record_completed_scan(event)
                    return event

                def try_advance_continuous(position, session):
                    event = self.retriever.try_advance_batch(session, 0)
                    if event is None:
                        return None
                    self._record_completed_scan(event)
                    if position > 0:
                        previous = self._arrival_windows[position - 1]
                        previous_incomplete = any(
                            self._completion_s[index] is None
                            for index in range(previous.start, previous.stop)
                        )
                        if previous_incomplete:
                            budget = int(event.budget)
                            previous_max = int(
                                self._lookahead_max_budget_while_previous_incomplete.get(
                                    position, 0
                                )
                            )
                            self._lookahead_max_budget_while_previous_incomplete[
                                position
                            ] = max(previous_max, budget)
                            if budget > args.bootstrap_nprobe:
                                self._lookahead_post_bootstrap_event_before_previous_answer_count += 1
                    return event

                def start_one(position, window):
                    query_indices = tuple(range(window.start, window.stop))
                    is_lookahead = bool(
                        enable_m3b
                        and active
                        and not args.full_retrieval_continuous_batching
                    )
                    park_after_bootstrap = bool(
                        is_lookahead
                        and args.current_batch_priority_scheduler
                        and not args.full_retrieval_continuous_batching
                    )
                    self._bootstrap_query_indices_by_position[position] = query_indices
                    active_started_ns = time.time_ns()
                    t0 = time.perf_counter()
                    session = self.retriever.start_batch(
                        questions[window.start : window.stop],
                        batch_id=window.batch_id,
                        query_indices=query_indices,
                        pause_after_bootstrap=park_after_bootstrap,
                    )
                    active_stopped_ns = time.time_ns()
                    self._retrieval_prepare_intervals_ns.append(
                        (active_started_ns, active_stopped_ns)
                    )
                    for query_index in query_indices:
                        self._retrieval_prepare_intervals_by_query_ns.setdefault(
                            int(query_index), []
                        ).append((active_started_ns, active_stopped_ns))
                    all_sessions.append(session)
                    self._native_sessions_started_count += 1
                    self._session_start_s[window.batch_id] = (
                        t0 - self._benchmark_started
                    )
                    if continuous_m3:
                        continuous_session_started_at[id(session)] = t0
                        if park_after_bootstrap:
                            bootstrap_parked_session_ids.add(id(session))
                        active.append((position, session))
                        return True
                    event = self.retriever.advance_batch(session)
                    t1 = time.perf_counter()
                    self._retrieval_intervals.append(
                        (t0 - self._benchmark_started, t1 - self._benchmark_started)
                    )
                    self._record_completed_scan(event)
                    if not self._is_valid_m2_first_latch_event(event):
                        raise RuntimeError(
                            self._invalid_first_latch_message(
                                event, continuous=False
                            )
                        )
                    self._bootstrap_ready_s[window.batch_id] = t1 - self._benchmark_started
                    if not emit(event):
                        return False
                    if not session.terminal:
                        if is_lookahead:
                            parked[position] = session
                            self._lookahead_max_budget_before_promotion[position] = int(
                                event.budget
                            )
                        else:
                            active.append((position, session))
                    else:
                        close_one(session)
                    return True

                def start_released_continuous_batch():
                    """Batch only requests that have already reached release time."""

                    nonlocal next_window
                    candidates = []
                    while (
                        len(active) + len(candidates) < max_sessions
                        and len(candidates) < args.max_online_embedding_batch
                        and next_window < len(self._arrival_windows)
                        and released(self._arrival_windows[next_window])
                        and allowed_to_start(next_window)
                    ):
                        window = self._arrival_windows[next_window]
                        candidates.append((next_window, window))
                        next_window += 1
                    if not candidates:
                        return True

                    flat_indices = tuple(
                        index
                        for _position, window in candidates
                        for index in range(window.start, window.stop)
                    )
                    flat_questions = [questions[index] for index in flat_indices]
                    active_started_ns = time.time_ns()
                    t0 = time.perf_counter()
                    (
                        _prepared_questions,
                        embeddings,
                        coarse_scores,
                        coarse_ids,
                    ) = self.retriever.prepare_batch_inputs(flat_questions)
                    prepared_sessions = []
                    cursor = 0
                    for position, window in candidates:
                        query_indices = tuple(range(window.start, window.stop))
                        rows = len(query_indices)
                        self._bootstrap_query_indices_by_position[position] = (
                            query_indices
                        )
                        session = self.retriever.start_prepared_batch(
                            questions[window.start : window.stop],
                            batch_id=window.batch_id,
                            query_indices=query_indices,
                            embeddings=embeddings[cursor : cursor + rows],
                            coarse_scores=coarse_scores[cursor : cursor + rows],
                            coarse_ids=coarse_ids[cursor : cursor + rows],
                            pause_after_bootstrap=False,
                        )
                        cursor += rows
                        prepared_sessions.append((position, window, session))
                    active_stopped_ns = time.time_ns()
                    self._retrieval_prepare_intervals_ns.append(
                        (active_started_ns, active_stopped_ns)
                    )
                    for query_index in flat_indices:
                        self._retrieval_prepare_intervals_by_query_ns.setdefault(
                            int(query_index), []
                        ).append((active_started_ns, active_stopped_ns))
                    for position, window, session in prepared_sessions:
                        all_sessions.append(session)
                        self._native_sessions_started_count += 1
                        self._session_start_s[window.batch_id] = (
                            t0 - self._benchmark_started
                        )
                        continuous_session_started_at[id(session)] = t0
                        active.append((position, session))
                    self._max_active_sessions_observed = max(
                        self._max_active_sessions_observed, len(active)
                    )
                    return True

                while not stop_event.is_set():
                    if continuous_m3:
                        if args.full_retrieval_continuous_batching:
                            if not start_released_continuous_batch():
                                return
                        else:
                            while (
                                len(active) < max_sessions
                                and next_window < len(self._arrival_windows)
                                and released(self._arrival_windows[next_window])
                                and allowed_to_start(next_window)
                            ):
                                if not start_one(
                                    next_window, self._arrival_windows[next_window]
                                ):
                                    return
                                next_window += 1
                                self._max_active_sessions_observed = max(
                                    self._max_active_sessions_observed, len(active)
                                )

                        if active:
                            events_in_round = 0
                            sessions_in_round = len(active)
                            for _ in range(sessions_in_round):
                                position, session = active.popleft()
                                with cancellation_lock:
                                    current_cancelled = set(cancelled)
                                self.retriever.cancel_queries(
                                    session, current_cancelled
                                )
                                if session.terminal:
                                    close_one(session)
                                    continue
                                if (
                                    id(session) in bootstrap_parked_session_ids
                                    and self._arrival_windows[position].batch_id
                                    in self._bootstrap_ready_s
                                ):
                                    active.append((position, session))
                                    continue
                                if not session.has_runnable:
                                    raise RuntimeError(
                                        "nonterminal continuous session has no "
                                        "runnable queries"
                                    )
                                event = try_advance_continuous(position, session)
                                if event is not None:
                                    events_in_round += 1
                                    window = self._arrival_windows[position]
                                    if window.batch_id not in self._bootstrap_ready_s:
                                        if not self._is_valid_m2_first_latch_event(event):
                                            raise RuntimeError(
                                                self._invalid_first_latch_message(
                                                    event, continuous=True
                                                )
                                            )
                                        self._bootstrap_ready_s[window.batch_id] = (
                                            time.perf_counter()
                                            - self._benchmark_started
                                        )
                                    if not emit(event):
                                        return
                                if session.terminal:
                                    close_one(session)
                                else:
                                    active.append((position, session))

                            if (
                                not args.full_retrieval_continuous_batching
                                and active
                            ):
                                _current_position, current_session = min(
                                    active, key=lambda row: row[0]
                                )
                                identity = id(current_session)
                                if identity in bootstrap_parked_session_ids:
                                    resume_batch = getattr(
                                        self.retriever, "resume_batch", None
                                    )
                                    if resume_batch is None:
                                        raise RuntimeError(
                                            "two-batch overlap requires resume_batch()"
                                        )
                                    resume_batch(current_session)
                                    bootstrap_parked_session_ids.remove(identity)

                            self._max_active_sessions_observed = max(
                                self._max_active_sessions_observed, len(active)
                            )
                            if events_in_round == 0:
                                stop_event.wait(0.001)
                            continue

                        if next_window >= len(self._arrival_windows):
                            break
                        wait_for_release = max(
                            0.0,
                            self._arrival_windows[next_window].release_s
                            - elapsed(),
                        )
                        stop_event.wait(
                            min(max(wait_for_release, 0.001), 0.05)
                        )
                        continue

                    earliest = earliest_incomplete_window()
                    for position in sorted(tuple(parked)):
                        if position > earliest:
                            continue
                        session = parked.pop(position)
                        max_parked_budget = int(
                            self._lookahead_max_budget_before_promotion.get(
                                position, args.bootstrap_nprobe
                            )
                        )
                        if max_parked_budget != args.bootstrap_nprobe:
                            raise RuntimeError(
                                "lookahead session advanced beyond bootstrap before promotion"
                            )
                        budgets = tuple(
                            int(state.scanned_budget)
                            for state in getattr(session, "states", ())
                        )
                        if not budgets or any(
                            budget != args.bootstrap_nprobe for budget in budgets
                        ):
                            raise RuntimeError(
                                "parked lookahead session did not remain at "
                                f"nprobe={args.bootstrap_nprobe}: "
                                f"{budgets}"
                            )
                        active.append((position, session))

                    while (
                        len(active) + len(parked) < max_sessions
                        and next_window < len(self._arrival_windows)
                        and released(self._arrival_windows[next_window])
                        and allowed_to_start(next_window)
                    ):
                        if not start_one(
                            next_window, self._arrival_windows[next_window]
                        ):
                            return
                        next_window += 1
                        self._max_active_sessions_observed = max(
                            self._max_active_sessions_observed,
                            len(active) + len(parked),
                        )

                    if active:
                        position, session = active.popleft()
                        with cancellation_lock:
                            current_cancelled = set(cancelled)
                        self.retriever.cancel_queries(session, current_cancelled)
                        if session.terminal:
                            close_one(session)
                            continue
                        if not session.has_runnable:
                            raise RuntimeError(
                                "nonterminal paused session violates stable-terminal semantics"
                            )
                        event = timed_advance(session)
                        if not emit(event):
                            return
                        if not session.terminal:
                            active.append((position, session))
                        else:
                            close_one(session)
                        self._max_active_sessions_observed = max(
                            self._max_active_sessions_observed,
                            len(active) + len(parked),
                        )
                        continue

                    if (
                        next_window >= len(self._arrival_windows)
                        and not parked
                    ):
                        break
                    if parked:
                        # A bootstrapped lookahead is intentionally frozen at
                        # nprobe=4 until the preceding window has answers.
                        stop_event.wait(0.001)
                        continue
                    wait_for_release = max(
                        0.0, self._arrival_windows[next_window].release_s - elapsed()
                    )
                    stop_event.wait(min(max(wait_for_release, 0.001), 0.05))

                if not stop_event.is_set():
                    close_all()
                    emit(_RetrievalFinished())
            except BaseException as error:
                try:
                    close_all()
                except BaseException as close_error:
                    error = close_error
                if not stop_event.is_set():
                    emit(_RetrievalFailed(error))
            finally:
                try:
                    close_all()
                except BaseException:
                    # A normal-path close failure is already routed through the
                    # retrieval failure event.  If caller shutdown set the stop
                    # flag first, the incomplete counters below still make the
                    # produced result fail its close-accounting audit.
                    pass

        def run(self, dataset, do_eval=False, pred_process_fun=None):
            self._completion_s = [None] * len(dataset)
            self._generation_intervals = []
            self._retrieval_intervals = []
            self._retrieval_prepare_intervals_ns = []
            self._retrieval_scan_intervals_ns = []
            self._generation_service_intervals_ns = []
            self._retrieval_prepare_intervals_by_query_ns = {}
            self._retrieval_scan_intervals_by_query_ns = {}
            self._generation_service_intervals_by_query_ns = {}
            self._generation_request_query_index = {}
            self._bootstrap_ready_s = {}
            self._session_start_s = {}
            self._max_active_sessions_observed = 0
            self._bootstrap_query_indices_by_position = {}
            self._bootstrap_generation_started_positions = set()
            self._bootstrap_event_consumed_positions = set()
            self._bootstrap_event_consumed_s = {}
            self._m2_latch_consumed_s = {}
            self._bootstrap_generation_start_s = {}
            self._lookahead_max_budget_before_promotion = {}
            self._lookahead_search_before_current_final_count = 0
            self._lookahead_max_budget_while_previous_incomplete = {}
            self._lookahead_post_bootstrap_event_before_previous_answer_count = 0
            self._native_sessions_started_count = 0
            self._native_sessions_closed_count = 0
            self._native_close_accounted_query_count = 0
            self._native_scan_permit_diagnostics = []
            self._generation_call_trace = []
            self._generation_call_context = None
            self._event_request_trace_index = {}
            self._generation_engine_step_trace = []
            self._generation_engine_idle_step_count = 0
            self._generation_sync_call_count = 0
            self._generation_add_request_call_count = 0
            self._generation_step_call_count = 0
            self._generation_active_request_count = 0
            self._generation_active_request_high_water = 0
            self._event_queue_high_water = 0
            self._event_queue_full_block_count = 0
            self._event_queue_full_block_s = 0.0
            self._benchmark_started_ns = time.time_ns()
            self._benchmark_started = time.perf_counter()
            output = super().run(
                dataset, do_eval=do_eval, pred_process_fun=pred_process_fun
            )
            missing = [index for index, value in enumerate(self._completion_s) if value is None]
            if missing:
                raise RuntimeError(f"missing completion timestamps for {len(missing)} queries")
            return output

    load_start = time.perf_counter()
    faiss.omp_set_num_threads(args.omp_threads)
    pipeline = ArrivalPipeline(config)
    pipeline._install_generation_call_tracer()
    validate_progressive_runtime_backend(pipeline.retriever, args)
    load_s = time.perf_counter() - load_start

    if len(warmup):
        pipeline.set_benchmark([ArrivalWindow(0, 0, len(warmup), 0.0)])
        pipeline.run(warmup, do_eval=False)
        torch.cuda.synchronize()

    pipeline.set_benchmark(windows)
    output = pipeline.run(measured, do_eval=False)
    expected_native_sessions = len(windows)
    native_close_accounting_ok = bool(
        pipeline._native_sessions_started_count == expected_native_sessions
        and pipeline._native_sessions_closed_count == expected_native_sessions
        and pipeline._native_close_accounted_query_count == len(output)
    )
    if not native_close_accounting_ok:
        raise RuntimeError(
            "native session close accounting is incomplete: "
            f"expected_sessions={expected_native_sessions}, "
            f"started={pipeline._native_sessions_started_count}, "
            f"closed={pipeline._native_sessions_closed_count}, "
            f"accounted_queries={pipeline._native_close_accounted_query_count}, "
            f"expected_queries={len(output)}"
        )
    completion_s = list(pipeline._completion_s)
    arrivals = [0.0] * len(completion_s)
    session_starts = [None] * len(completion_s)
    for window in windows:
        arrivals[window.start : window.stop] = window_arrivals(window)
        if window.batch_id not in pipeline._session_start_s:
            raise RuntimeError(
                f"missing retrieval-session start for batch {window.batch_id}"
            )
        session_start = float(pipeline._session_start_s[window.batch_id])
        session_starts[window.start : window.stop] = [session_start] * int(
            window.size
        )
    if any(value is None for value in session_starts):
        raise RuntimeError("missing per-query service-start timestamp")
    queue_wait_s = [
        max(0.0, float(start) - float(arrival))
        for start, arrival in zip(session_starts, arrivals)
    ]
    service_latency_s = [
        max(0.0, float(done) - float(start))
        for done, start in zip(completion_s, session_starts)
    ]
    if args.arrival_model == "closed_loop":
        # A new client request is issued only when the preceding request has
        # completed and this retrieval session starts.  Replace placeholder
        # releases with those observed closed-loop arrivals.
        arrivals = [float(value) for value in session_starts]
        queue_wait_s = [0.0] * len(session_starts)
    predictions = output_column(output, ("pred",), default="")
    control_sentinel = str(config["progressive_no_answer_sentinel"])
    leaked_control_indices = [
        index
        for index, prediction in enumerate(predictions)
        if str(prediction or "").strip()
        and is_no_answer(prediction, control_sentinel)
    ]
    if leaked_control_indices and args.gate_component_stage != "raw":
        raise RuntimeError(
            "progressive FINAL/early output contains the internal "
            "insufficient-evidence sentinel: "
            f"{leaked_control_indices[:8]}"
        )
    measured.update_output("pred", predictions)

    ranked_ids = output_column(output, ("retrieval_idxs",), default=[])
    search_ids = output_column(output, ("docs_attempted",), default=[])
    final_prompt_ids = output_column(output, ("final_prompt_document_ids",), default=[])
    exposure_ids = [
        ordered_unique(list(search or []) + list(final or []))
        for search, final in zip(search_ids, final_prompt_ids)
    ]
    ranked_documents = materialize_rows(pipeline.retriever, ranked_ids)
    exposure_documents = materialize_rows(pipeline.retriever, exposure_ids)
    actual_scans = [
        int(value)
        for value in output_column(output, ("actual_scanned_nprobe",), default=0)
    ]
    terminal_reasons = output_column(output, ("terminal_reason",), default=None)
    terminal_budgets = output_column(output, ("terminal_budget",), default=None)
    answer_sources = output_column(output, ("answer_source",), default=None)
    generation_traces = output_column(output, ("generation_trace",), default=[])
    answerability_skip_traces = output_column(
        output, ("answerability_skip_trace",), default=[]
    )
    terminal_bulk_armed = output_column(
        output, ("terminal_bulk_final_armed",), default=False
    )
    terminal_bulk_arm_budgets = output_column(
        output, ("terminal_bulk_final_arm_budget",), default=None
    )
    terminal_bulk_arm_reasons = output_column(
        output, ("terminal_bulk_final_arm_reason",), default=None
    )
    terminal_bulk_appended = output_column(
        output,
        ("terminal_bulk_final_appended_documents",),
        default=0,
    )
    conversation_prefix_checks = output_column(
        output, ("conversation_prefix_invariant",), default=[]
    )
    conversation_lcp_tokens = output_column(
        output, ("conversation_lcp_tokens",), default=[]
    )
    conversation_document_ids = output_column(
        output, ("conversation_document_ids",), default=[]
    )
    conversation_assistant_outputs = output_column(
        output, ("conversation_assistant_outputs",), default=[]
    )
    if args.conversational_prefix:
        invalid_conversation_prefix = [
            index
            for index, checks in enumerate(conversation_prefix_checks)
            if not checks or not all(bool(value) for value in checks)
        ]
        if invalid_conversation_prefix:
            raise RuntimeError(
                "conversational M2 violated the full-request token prefix "
                f"invariant: {invalid_conversation_prefix[:8]}"
            )
        conversation_document_mismatch = [
            index
            for index, (search, conversation) in enumerate(
                zip(search_ids, conversation_document_ids)
            )
            if [int(value) for value in search or ()]
            != [int(value) for value in conversation or ()]
        ]
        if conversation_document_mismatch:
            raise RuntimeError(
                "conversational prompt document order diverged from M2 "
                f"selection: {conversation_document_mismatch[:8]}"
            )
    search_generation_budgets = output_column(
        output, ("search_generation_budgets",), default=[]
    )
    latched_first_generation_budgets = output_column(
        output, ("first_search_generation_budget",), default=None
    )
    m2_first_snapshot_reasons = output_column(
        output, ("m2_first_snapshot_reason",), default=None
    )
    retrieval_traces = output_column(output, ("retrieval_trace",), default=[])
    selection_candidate_ids = output_column(
        output, ("search_generation_candidate_ids",), default=[]
    )
    prefetch_advised_lists = output_column(
        output, ("prefetch_advised_lists",), default=0
    )
    prefetch_advised_bytes = output_column(
        output, ("prefetch_advised_bytes",), default=0
    )
    prefetch_advice_errors = output_column(
        output, ("prefetch_advice_errors",), default=0
    )
    prefetch_high_water = output_column(
        output, ("prefetch_high_water",), default=0
    )
    heap_updates = output_column(output, ("heap_updates",), default=0)
    unchanged_heap_lists = output_column(
        output, ("unchanged_heap_lists",), default=0
    )
    search_selection_traces = audit_search_document_selection(
        search_ids,
        search_generation_budgets,
        selection_candidate_ids,
    )
    repeated_or_stale_search_budgets = []
    for index, budgets in enumerate(search_generation_budgets):
        normalized = [int(value) for value in (budgets or [])]
        frozen_budget = (
            int(terminal_budgets[index])
            if terminal_budgets[index] is not None
            else None
        )
        invalid_transition = any(
            current < previous
            or (current == previous and current != frozen_budget)
            for previous, current in zip(normalized, normalized[1:])
        )
        if invalid_transition:
            repeated_or_stale_search_budgets.append((index, normalized))
    if repeated_or_stale_search_budgets:
        raise RuntimeError(
            "SEARCHING generation reused a live stale heap snapshot (reusing "
            "the frozen M1 terminal heap is permitted): "
            f"{repeated_or_stale_search_budgets[:8]}"
        )
    verification_traces = output_column(
        output, ("answer_verification_trace",), default=[]
    )
    answerability_gate_traces = output_column(
        output, ("answerability_gate_trace",), default=[]
    )
    first_generation_budgets = []
    for gate_trace, verification_trace in zip(
        answerability_gate_traces, verification_traces
    ):
        first_budget = None
        # answer_verification_trace also contains retrieval-side invalidation,
        # terminal, and deferred-commit rows.  Under load those rows can be
        # appended before the deferred result of the first SEARCHING decode,
        # so they are not an authoritative completion-order trace.  Every
        # SEARCHING completion is recorded immediately in the dedicated gate
        # trace; use that first and retain the legacy fallback only for old
        # non-event-driven outputs that predate the gate trace.
        for row in gate_trace or []:
            if row.get("budget") is not None:
                first_budget = int(row["budget"])
                break
        if first_budget is None:
            for row in verification_trace or []:
                if row.get("decision") not in {
                    "continue",
                    "provisional",
                    "accepted",
                }:
                    continue
                for key in ("budget", "candidate_budget", "observed_budget"):
                    if row.get(key) is not None:
                        first_budget = int(row[key])
                        break
                if first_budget is not None:
                    break
        first_generation_budgets.append(first_budget)
    authoritative_first_budgets = []
    for index, (latched, budgets) in enumerate(
        zip(latched_first_generation_budgets, search_generation_budgets)
    ):
        direct = (
            int(latched)
            if latched is not None
            else (int(budgets[0]) if budgets else None)
        )
        authoritative_first_budgets.append(direct)
    if args.full_context_only:
        unexpected_searching = [
            (index, direct, list(budgets or []))
            for index, (direct, budgets) in enumerate(
                zip(authoritative_first_budgets, search_generation_budgets)
            )
            if direct is not None or budgets
        ]
        if unexpected_searching:
            raise RuntimeError(
                "Full Context issued an intermediate SEARCHING decode: "
                f"{unexpected_searching[:8]}"
            )
    else:
        invalid_first_budgets = []
        for index, (budget, reason) in enumerate(
            zip(authoritative_first_budgets, m2_first_snapshot_reasons)
        ):
            if reason == "retrieval_terminal_before_target":
                valid = (
                    budget is not None
                    and int(budget) < int(args.m2_first_snapshot_nprobe)
                    and terminal_budgets[index] is not None
                    and int(terminal_budgets[index]) == int(budget)
                    and str(terminal_reasons[index])
                    in {"wrd_stable", "max_nprobe"}
                )
            else:
                valid = (
                    budget is not None
                    and int(budget) == int(args.m2_first_snapshot_nprobe)
                )
            if not valid:
                invalid_first_budgets.append((index, budget, reason))
        if invalid_first_budgets:
            raise RuntimeError(
                "first SEARCHING generation did not use the configured M2 first "
                f"snapshot nprobe={args.m2_first_snapshot_nprobe}: "
                f"{invalid_first_budgets[:8]}"
            )
        trace_budget_disagreements = [
            (index, direct, traced)
            for index, (direct, traced) in enumerate(
                zip(authoritative_first_budgets, first_generation_budgets)
            )
            if traced is not None
            and int(traced) != int(direct)
            and not (
                args.answerability_lexical_no_skip
                and answerability_skip_traces[index]
                and int(answerability_skip_traces[index][0].get("budget", -1))
                == int(direct)
                and int(traced) >= int(direct)
            )
        ]
        if trace_budget_disagreements:
            raise RuntimeError(
                "first SEARCHING gate completion disagrees with the immutable "
                "M2 first-snapshot latch: "
                f"{trace_budget_disagreements[:8]}"
            )
    final_done = output_column(output, ("final_generation_done",), default=False)
    prefix_traces = output_column(output, ("prefix_cache_trace",), default=[])
    prefix_alignment_padding = output_column(
        output, ("prefix_alignment_padding_tokens",), default=0
    )
    final_prefix_traces = output_column(
        output, ("final_prefix_cache_trace",), default=[]
    )
    final_prompt_token_prefix_aligned = output_column(
        output, ("final_prompt_token_prefix_aligned",), default=False
    )
    final_prompt_token_counts = output_column(
        output, ("final_prompt_token_count",), default=None
    )

    stable_then_full = sum(
        str(reason) == "wrd_stable"
        and budget is not None
        and int(budget) == args.nprobe
        for reason, budget in zip(terminal_reasons, terminal_budgets)
    )
    if stable_then_full:
        raise RuntimeError("stable queries resumed to full nprobe")
    final_generation_traces = output_column(
        output, ("final_generation_trace",), default=[]
    )
    for index, source in enumerate(answer_sources):
        source = str(source or "")
        if source.startswith("final_"):
            completed = [
                row
                for row in final_generation_traces[index]
                if row.get("event") == "completed"
            ]
            if not final_done[index] or len(completed) != 1:
                raise RuntimeError(f"query {index} did not run FINAL exactly once")
            if completed[0].get("prompt_status") != "FINAL":
                raise RuntimeError(f"query {index} final prompt status is invalid")
        elif source not in {
            "early_cross_snapshot_consistent",
            "early_single_snapshot_confident",
        }:
            raise RuntimeError(f"query {index} has unknown answer source {source!r}")

    excessive_generation_rounds = [
        (index, len(trace or []))
        for index, trace in enumerate(generation_traces)
        if len(trace or []) > int(args.topk)
    ]
    if excessive_generation_rounds:
        raise RuntimeError(
            f"M2 exceeded the topk={int(args.topk)} decode/document budget: "
            f"{excessive_generation_rounds[:8]}"
        )

    generation_call_trace = list(pipeline._generation_call_trace)
    final_length_repair_count = sum(
        any(row.get("event") == "length_repair_queued" for row in trace or [])
        for trace in final_generation_traces
    )
    final_truncated_count = sum(
        any(
            row.get("event") == "completed"
            and bool(row.get("output_truncated"))
            for row in trace or []
        )
        for trace in final_generation_traces
    )
    generation_engine_step_trace = list(
        pipeline._generation_engine_step_trace
    )
    if pipeline._event_request_trace_index:
        raise RuntimeError(
            "event-driven generation trace retained unfinished request ids"
        )
    calls_by_query = [[] for _ in range(len(output))]
    for call in generation_call_trace:
        call_index = int(call["call_index"])
        for query_index in call.get("query_indices", []):
            query_index = int(query_index)
            if not 0 <= query_index < len(calls_by_query):
                raise RuntimeError(
                    f"generation trace contains invalid query index {query_index}"
                )
            calls_by_query[query_index].append(call_index)

    extra = [
        {
            "actual_scanned_nprobe": actual_scans[index],
            "terminal_reason": terminal_reasons[index],
            "terminal_budget": terminal_budgets[index],
            "answer_source": answer_sources[index],
            "generation_rounds": len(generation_traces[index] or []),
            "queue_wait_s": queue_wait_s[index],
            "service_latency_s": service_latency_s[index],
            "search_generation_budgets": list(
                search_generation_budgets[index] or []
            ),
            "search_documents_attempted": len(search_ids[index] or []),
            "first_search_generation_budget": first_generation_budgets[index],
            "ranked_document_ids": list(ranked_ids[index] or []),
            "search_document_ids": list(search_ids[index] or []),
            "final_prompt_document_ids": list(final_prompt_ids[index] or []),
            "final_prompt_token_prefix_aligned": bool(
                final_prompt_token_prefix_aligned[index]
            ),
            "final_prompt_token_count": final_prompt_token_counts[index],
            "generation_exposure_ids": exposure_ids[index],
            "retrieval_backend": PROGRESSIVE_RETRIEVAL_BACKEND,
            "search_selection_trace": search_selection_traces[index],
            "generation_call_indices": calls_by_query[index],
            "generation_trace": (
                json_safe(generation_traces[index] or [])
                if args.trace_generation_details
                else []
            ),
            "answerability_skip_trace": json_safe(
                answerability_skip_traces[index] or []
            ),
            "terminal_bulk_final_armed": bool(
                terminal_bulk_armed[index]
            ),
            "terminal_bulk_final_arm_budget": (
                int(terminal_bulk_arm_budgets[index])
                if terminal_bulk_arm_budgets[index] is not None
                else None
            ),
            "terminal_bulk_final_arm_reason": (
                str(terminal_bulk_arm_reasons[index])
                if terminal_bulk_arm_reasons[index] is not None
                else None
            ),
            "terminal_bulk_final_appended_documents": int(
                terminal_bulk_appended[index] or 0
            ),
            "conversation_prefix_invariant": list(
                conversation_prefix_checks[index] or []
            ),
            "conversation_lcp_tokens": list(
                conversation_lcp_tokens[index] or []
            ),
            "conversation_document_ids": list(
                conversation_document_ids[index] or []
            ),
            "conversation_assistant_outputs": list(
                conversation_assistant_outputs[index] or []
            ),
            "answer_verification_trace": (
                json_safe(verification_traces[index] or [])
                if args.trace_generation_details
                else []
            ),
            "retrieval_rank_trajectory": (
                json_safe(retrieval_traces[index] or [])
                if args.trace_generation_details
                else []
            ),
            "prefetch_advised_lists": prefetch_advised_lists[index],
            "prefetch_advised_bytes": prefetch_advised_bytes[index],
            "prefetch_advice_errors": prefetch_advice_errors[index],
            "prefetch_high_water": prefetch_high_water[index],
            "heap_updates": heap_updates[index],
            "unchanged_heap_lists": unchanged_heap_lists[index],
            "prefix_cache_trace": (
                json_safe(prefix_traces[index] or [])
                if args.trace_generation_details
                else []
            ),
            "answerability_gate_trace": (
                json_safe(answerability_gate_traces[index] or [])
                if args.trace_generation_details
                else []
            ),
            "prefix_alignment_padding_tokens": int(
                prefix_alignment_padding[index] or 0
            ),
            "final_prefix_cache_trace": (
                json_safe(final_prefix_traces[index] or [])
                if args.trace_generation_details
                else []
            ),
        }
        for index in range(len(output))
    ]
    quality, recall, samples = quality_and_samples(
        config,
        measured,
        completion_s,
        windows,
        ranked_documents,
        exposure_documents,
        extra,
        max_recall_k=args.topk,
    )
    # TTFT is measured to the first token of the terminal answer chain.  The
    # progressive path may issue and abort earlier SEARCHING chains; choosing
    # the last completed per-query generation attempt keeps the user-visible
    # answer boundary consistent with the sequential baselines.
    terminal_attempts = []
    ttft_s = []
    benchmark_started_ns = int(pipeline._benchmark_started_ns)
    for query_index, sample in enumerate(samples):
        attempts = list(
            pipeline._generation_attempts_by_query_ns.get(query_index, ())
        )
        if not attempts:
            raise RuntimeError(
                f"query {query_index} has no generation attempt for TTFT"
            )
        terminal = attempts[-1]
        arrival_ns = benchmark_started_ns + int(
            round(float(sample["arrival_s"]) * 1e9)
        )
        value = (int(terminal["first_token_ns"]) - arrival_ns) / 1e9
        if value < 0 or value > float(sample["latency_s"]):
            raise RuntimeError(
                f"query {query_index} has invalid TTFT {value}"
            )
        sample["ttft_s"] = value
        sample["terminal_generation_attempt"] = dict(terminal)
        sample["generation_attempt_count"] = len(attempts)
        terminal_attempts.append(terminal)
        ttft_s.append(value)

    decisions = Counter()
    reasons = Counter()
    for trace in verification_traces:
        for row in trace or []:
            decisions[str(row.get("decision"))] += 1
            reasons[str(row.get("reason"))] += 1
    cached_tokens = prompt_tokens = generated_tokens = 0
    cache_audit_rows = []
    for traces in (prefix_traces, final_prefix_traces):
        for trace in traces:
            for row in trace or []:
                cached_tokens += int(row.get("num_cached_tokens") or 0)
                prompt_tokens += int(row.get("prompt_tokens") or 0)
                generated_tokens += int(row.get("generated_tokens") or 0)
                if row.get("logical_chain_expected_cached_tokens") is not None:
                    cache_audit_rows.append(row)
    cache_transition_rows = [
        row
        for row in cache_audit_rows
        if int(row.get("logical_chain_expected_cached_tokens") or 0) > 0
    ]
    expected_chain_cached_tokens = sum(
        int(row.get("logical_chain_expected_cached_tokens") or 0)
        for row in cache_transition_rows
    )
    reused_chain_cached_tokens = sum(
        int(row.get("logical_chain_reused_tokens") or 0)
        for row in cache_transition_rows
    )
    cache_age_ms = [
        float(row["logical_chain_cache_age_ms"])
        for row in cache_transition_rows
        if row.get("logical_chain_cache_age_ms") is not None
    ]

    generation_intervals = list(pipeline._generation_intervals)
    retrieval_intervals = list(pipeline._retrieval_intervals)
    overlap_s = interval_overlap(retrieval_intervals, generation_intervals)
    # The legacy intervals above intentionally describe logical scheduler
    # occupancy.  For overlap evaluation, put both stages on one Unix
    # wall-clock domain: active retrieval preparation/scan work and the union
    # of AsyncLLMEngine request service lifetimes (admission through terminal
    # output).  Retrieval session wait/park/poll and empty generation polls are
    # excluded from the primary metric.
    def ns_intervals_as_seconds(rows):
        converted = []
        for start_ns, stop_ns in rows:
            start_ns = int(start_ns)
            stop_ns = int(stop_ns)
            if stop_ns <= start_ns:
                raise RuntimeError(
                    "active-compute interval has invalid bounds: "
                    f"{start_ns}, {stop_ns}"
                )
            converted.append((start_ns / 1e9, stop_ns / 1e9))
        return converted

    active_retrieval_intervals_ns = (
        list(pipeline._retrieval_prepare_intervals_ns)
        + list(pipeline._retrieval_scan_intervals_ns)
    )
    active_retrieval_intervals = ns_intervals_as_seconds(
        active_retrieval_intervals_ns
    )
    active_generation_intervals = ns_intervals_as_seconds(
        pipeline._generation_service_intervals_ns
    )
    active_retrieval_busy_s = interval_total(active_retrieval_intervals)
    active_generation_busy_s = interval_total(active_generation_intervals)
    active_overlap_s = interval_overlap(
        active_retrieval_intervals, active_generation_intervals
    )
    active_union_s = (
        active_retrieval_busy_s
        + active_generation_busy_s
        - active_overlap_s
    )
    active_overlap_over_retrieval = (
        active_overlap_s / active_retrieval_busy_s
        if active_retrieval_busy_s
        else 0.0
    )
    active_overlap_over_generation = (
        active_overlap_s / active_generation_busy_s
        if active_generation_busy_s
        else 0.0
    )
    active_overlap_over_min_busy = (
        active_overlap_s
        / min(active_retrieval_busy_s, active_generation_busy_s)
        if min(active_retrieval_busy_s, active_generation_busy_s) > 0
        else 0.0
    )
    active_overlap_jaccard = (
        active_overlap_s / active_union_s if active_union_s else 0.0
    )

    # Figure 13 v32 uses a cross-system active-search boundary: query
    # embedding and retrieval preparation are excluded, and only native IVF
    # scan execution is eligible for overlap with generation service.  Keep
    # the historical combined prepare+scan metrics above for provenance.
    active_search_intervals_ns = list(pipeline._retrieval_scan_intervals_ns)
    active_search_intervals = ns_intervals_as_seconds(
        active_search_intervals_ns
    )
    active_search_busy_s = interval_total(active_search_intervals)
    active_search_overlap_s = interval_overlap(
        active_search_intervals, active_generation_intervals
    )
    active_search_union_s = (
        active_search_busy_s
        + active_generation_busy_s
        - active_search_overlap_s
    )
    active_search_overlap_over_retrieval = (
        active_search_overlap_s / active_search_busy_s
        if active_search_busy_s
        else 0.0
    )
    active_search_overlap_jaccard = (
        active_search_overlap_s / active_search_union_s
        if active_search_union_s
        else 0.0
    )

    # Primary v2 metric: only retrieval work belonging to the same request and
    # occurring after that request's first generation admission is eligible to
    # be hidden.  Summation is query-wise, so unrelated requests cannot inflate
    # the numerator through ordinary cross-request pipeline overlap.
    same_request_post_retrieval_busy_s = 0.0
    same_request_total_retrieval_busy_s = 0.0
    same_request_overlap_s = 0.0
    same_request_query_rows = []
    retrieval_admission_delay_ms = []
    cohort_by_query = {}
    for window in windows:
        for query_index in range(int(window.start), int(window.stop)):
            cohort_by_query[query_index] = int(window.batch_id)
    retrieval_by_query_s = {
        int(query_index): ns_intervals_as_seconds(rows)
        for query_index, rows in pipeline._retrieval_scan_intervals_by_query_ns.items()
    }
    useful_generation_by_query_s = {
        int(query_index): ns_intervals_as_seconds(rows)
        for query_index, rows in pipeline._generation_service_intervals_by_query_ns.items()
    }
    useful_breakdown = useful_overlap_breakdown(
        retrieval_by_query_s,
        useful_generation_by_query_s,
        cohort_by_query,
    )
    benchmark_started_ns = int(pipeline._benchmark_started_ns)
    for query_index, sample in enumerate(samples):
        raw_retrieval = list(
            pipeline._retrieval_scan_intervals_by_query_ns.get(query_index, ())
        )
        raw_generation = list(
            pipeline._generation_service_intervals_by_query_ns.get(
                query_index, ()
            )
        )
        retrieval_rows = ns_intervals_as_seconds(raw_retrieval)
        generation_rows = ns_intervals_as_seconds(raw_generation)
        first_generation_start_s = (
            min(start for start, _ in generation_rows)
            if generation_rows
            else None
        )
        post_retrieval_rows = []
        if first_generation_start_s is not None:
            for start_s, stop_s in retrieval_rows:
                clipped_start_s = max(start_s, first_generation_start_s)
                if stop_s > clipped_start_s:
                    post_retrieval_rows.append((clipped_start_s, stop_s))
        query_post_busy_s = interval_total(post_retrieval_rows)
        query_total_retrieval_busy_s = interval_total(retrieval_rows)
        query_overlap_s = interval_overlap(
            post_retrieval_rows, generation_rows
        )
        same_request_post_retrieval_busy_s += query_post_busy_s
        same_request_total_retrieval_busy_s += query_total_retrieval_busy_s
        same_request_overlap_s += query_overlap_s

        query_active_rows = list(
            pipeline._retrieval_prepare_intervals_by_query_ns.get(
                query_index, ()
            )
        ) + raw_retrieval
        first_retrieval_start_ns = (
            min(int(start_ns) for start_ns, _ in query_active_rows)
            if query_active_rows
            else None
        )
        if first_retrieval_start_ns is None:
            raise RuntimeError(
                f"query {query_index} has no active retrieval interval"
            )
        arrival_ns = benchmark_started_ns + int(
            round(float(sample["arrival_s"]) * 1e9)
        )
        admission_delay_ms = max(
            0.0, (first_retrieval_start_ns - arrival_ns) / 1e6
        )
        retrieval_admission_delay_ms.append(admission_delay_ms)
        same_request_query_rows.append(
            {
                "query_index": query_index,
                "retrieval_scan_interval_count": len(raw_retrieval),
                "generation_service_interval_count": len(raw_generation),
                "post_first_evidence_retrieval_busy_s": query_post_busy_s,
                "total_retrieval_busy_s": query_total_retrieval_busy_s,
                "same_request_overlap_s": query_overlap_s,
                "retrieval_admission_delay_ms": admission_delay_ms,
            }
        )
    same_request_hidden_ratio = (
        same_request_overlap_s / same_request_post_retrieval_busy_s
        if same_request_post_retrieval_busy_s
        else 0.0
    )
    query_aligned_retrieval_hidden_ratio = (
        same_request_overlap_s / same_request_total_retrieval_busy_s
        if same_request_total_retrieval_busy_s
        else 0.0
    )
    per_query_scan_interval_count = sum(
        len(rows)
        for rows in pipeline._retrieval_scan_intervals_by_query_ns.values()
    )
    per_query_generation_interval_count = sum(
        len(rows)
        for rows in pipeline._generation_service_intervals_by_query_ns.values()
    )
    if per_query_scan_interval_count != len(
        pipeline._retrieval_scan_intervals_ns
    ):
        raise RuntimeError(
            "per-query/native-global scan interval accounting mismatch: "
            f"per_query={per_query_scan_interval_count}, "
            f"global={len(pipeline._retrieval_scan_intervals_ns)}"
        )
    if per_query_generation_interval_count != len(
        pipeline._generation_service_intervals_ns
    ):
        raise RuntimeError(
            "per-query/global generation interval accounting mismatch: "
            f"per_query={per_query_generation_interval_count}, "
            f"global={len(pipeline._generation_service_intervals_ns)}"
        )
    if pipeline._generation_request_query_index:
        raise RuntimeError(
            "generation request-to-query mappings remain after completion: "
            f"{len(pipeline._generation_request_query_index)}"
        )
    bootstrap_overlap = 0
    bootstrap_before_previous_done = 0
    session_start_before_previous_done = 0
    generation_start_before_previous_done = 0
    legal_bootstrap_overlap_count = 0
    legal_session_overlap_count = 0
    legal_generation_overlap_count = 0
    expected_lookahead_windows = 0
    m3_window_overlap_details = []
    window_done_s = [
        max(completion_s[window.start : window.stop])
        for window in windows
    ]
    for position, window in enumerate(windows):
        if position == 0:
            continue
        ready = pipeline._bootstrap_ready_s.get(window.batch_id)
        if ready is not None and any(
            start <= ready <= stop for start, stop in generation_intervals
        ):
            bootstrap_overlap += 1
        previous = windows[position - 1]
        previous_done = max(completion_s[previous.start : previous.stop])
        eligible_s = m3_lookahead_eligibility_s(
            position, float(window.release_s), window_done_s
        )
        if eligible_s is not None:
            expected_lookahead_windows += 1
        session_start = pipeline._session_start_s.get(window.batch_id)
        if session_start is not None and session_start < previous_done:
            session_start_before_previous_done += 1
        generation_start = pipeline._bootstrap_generation_start_s.get(position)
        bootstrap_consumed = pipeline._bootstrap_event_consumed_s.get(position)
        m2_latch_consumed = pipeline._m2_latch_consumed_s.get(position)
        if generation_start is not None and generation_start < previous_done:
            generation_start_before_previous_done += 1
        if ready is not None and ready < previous_done:
            bootstrap_before_previous_done += 1
        if (
            eligible_s is not None
            and session_start is not None
            and session_start < previous_done
        ):
            legal_session_overlap_count += 1
        if (
            eligible_s is not None
            and ready is not None
            and ready < previous_done
        ):
            legal_bootstrap_overlap_count += 1
        if (
            eligible_s is not None
            and generation_start is not None
            and generation_start < previous_done
        ):
            legal_generation_overlap_count += 1
        if session_start is not None and session_start < float(window.release_s):
            raise RuntimeError(
                "M3 retrieval session started before its arrival release: "
                f"position={position}, release={window.release_s}, "
                f"session={session_start}"
            )
        if generation_start is not None and ready is None:
            raise RuntimeError(
                "lookahead generation started without an observed nprobe=4 "
                f"bootstrap: position={position}"
            )
        if generation_start is not None and m2_latch_consumed is None:
            raise RuntimeError(
                "lookahead generation started without a consumed M2 first "
                f"snapshot latch: position={position}"
            )
        if (
            generation_start is not None
            and m2_latch_consumed is not None
            and generation_start + 1e-9 < m2_latch_consumed
        ):
            raise RuntimeError(
                "lookahead generation was timestamped before its M2 first-snapshot "
                "latch was consumed: "
                f"position={position}"
            )
        if (
            generation_start is not None
            and generation_start < previous_done
            and not (ready is not None and ready < previous_done)
        ):
            raise RuntimeError(
                "lookahead generation was timestamped before its nprobe=4 "
                f"bootstrap: position={position}"
            )
        m3_window_overlap_details.append(
            {
                "position": position,
                "batch_id": int(window.batch_id),
                "release_s": float(window.release_s),
                "previous_done_s": float(previous_done),
                "lookahead_eligible_s": eligible_s,
                "legal_lookahead_opportunity": eligible_s is not None,
                "available_overlap_s": max(
                    0.0, float(previous_done) - float(window.release_s)
                ),
                "session_start_s": session_start,
                "bootstrap_ready_s": ready,
                "bootstrap_consumed_s": bootstrap_consumed,
                "m2_latch_consumed_s": m2_latch_consumed,
                "generation_start_s": generation_start,
                "session_overlapped": bool(
                    session_start is not None and session_start < previous_done
                ),
                "bootstrap_overlapped": bool(
                    ready is not None and ready < previous_done
                ),
                "generation_overlapped": bool(
                    generation_start is not None
                    and generation_start < previous_done
                ),
                "legal_session_overlapped": bool(
                    eligible_s is not None
                    and session_start is not None
                    and session_start < previous_done
                ),
                "legal_bootstrap_overlapped": bool(
                    eligible_s is not None
                    and ready is not None
                    and ready < previous_done
                ),
                "legal_generation_overlapped": bool(
                    eligible_s is not None
                    and generation_start is not None
                    and generation_start < previous_done
                ),
            }
        )
    if enable_m3b and (
        args.rolling_m3_scanners
        or args.full_retrieval_continuous_batching
    ):
        # Continuous admission is governed by native-session capacity rather
        # than the legacy first-unfinished eligibility timestamp. Report the
        # directly observable adjacent-window opportunities/effectiveness.
        expected_lookahead_windows = sum(
            float(window.release_s) < window_done_s[position - 1]
            for position, window in enumerate(windows)
            if position > 0
        )
        legal_session_overlap_count = session_start_before_previous_done
        legal_bootstrap_overlap_count = bootstrap_before_previous_done
        legal_generation_overlap_count = generation_start_before_previous_done
        if (
            args.full_retrieval_continuous_batching
            and pipeline._max_active_sessions_observed
            > args.max_active_retrieval_sessions
        ):
            raise RuntimeError(
                "full retrieval continuous batching exceeded its active "
                "session bound"
            )
        if (
            args.full_retrieval_continuous_batching
            and expected_lookahead_windows
            and not legal_session_overlap_count
        ):
            raise RuntimeError(
                "full retrieval continuous batching observed no cross-window "
                "retrieval overlap despite an available opportunity"
            )
    if not enable_m3b and (
        bootstrap_before_previous_done
        or session_start_before_previous_done
        or generation_start_before_previous_done
    ):
        raise RuntimeError("M1+M2 accidentally prefetched a batch before the prior answer")
    # M3 guarantees that a released successor starts retrieval while its
    # predecessor is unfinished.  It cannot guarantee that encode + four IVF
    # lists + one vLLM admission all finish inside an arbitrarily small
    # release-to-answer gap.  Keep bootstrap/generation overlap as measured
    # effectiveness metrics rather than rejecting an otherwise valid run.
    # This distinction matters after M1 becomes faster: a successor released a
    # few milliseconds before the prior answer is a legal M3 start but may have
    # no physical opportunity to reach @4 before that answer.
    if (
        enable_m3b
        and not args.rolling_m3_scanners
        and not args.full_retrieval_continuous_batching
        and legal_session_overlap_count != expected_lookahead_windows
    ):
        raise RuntimeError(
            "full M3 retrieval-start overlap contract failed: "
            f"expected={expected_lookahead_windows}, "
            f"session={legal_session_overlap_count}, "
            f"bootstrap={legal_bootstrap_overlap_count}, "
            f"generation={legal_generation_overlap_count}; "
            "adjacent_effectiveness="
            f"{session_start_before_previous_done}/"
            f"{bootstrap_before_previous_done}/"
            f"{generation_start_before_previous_done}"
        )

    stable = [str(reason) == "wrd_stable" for reason in terminal_reasons]
    full = [
        budget is not None and int(budget) == args.nprobe
        for budget in terminal_budgets
    ]
    early = [
        str(source)
        in {"early_cross_snapshot_consistent", "early_single_snapshot_confident"}
        for source in answer_sources
    ]
    generation_coalesce_trace = list(
        getattr(pipeline, "_event_search_coalesce_trace", ()) or ()
    )
    generation_coalesce_wait_ms = [
        float(row.get("wait_ms") or 0.0)
        for row in generation_coalesce_trace
    ]
    generation_coalesce_reasons = Counter(
        str(row.get("reason")) for row in generation_coalesce_trace
    )
    native_scan_permit_sessions = list(
        getattr(pipeline, "_native_scan_permit_diagnostics", ()) or ()
    )
    native_scan_permit_global = (
        dict(native_scan_permit_sessions[-1].get("global") or {})
        if native_scan_permit_sessions
        else {}
    )
    if native_scan_permit_global:
        permit_capacity = int(native_scan_permit_global.get("capacity") or 0)
        permit_peak = int(native_scan_permit_global.get("peak_in_use") or 0)
        permit_in_use = int(native_scan_permit_global.get("in_use") or 0)
        permit_waiting = int(native_scan_permit_global.get("waiting") or 0)
        if permit_capacity < 1 or permit_peak > permit_capacity:
            raise RuntimeError(
                "native global scan permit capacity was exceeded: "
                f"capacity={permit_capacity}, peak={permit_peak}"
            )
        if permit_in_use or permit_waiting:
            raise RuntimeError(
                "native scan permits leaked after all sessions closed: "
                f"in_use={permit_in_use}, waiting={permit_waiting}"
            )
    service_s = max(completion_s)
    return {
        "group": profile_name or mode,
        "execution_group": mode,
        "shared_mechanism_profile": profile_name,
        "mechanisms": dict(effective_mechanisms),
        "load_and_warmup_excluded_s": load_s,
        "service_s": service_s,
        "achieved_throughput_req_s": len(output) / service_s,
        "latency": latency_summary(completion_s, windows),
        "ttft": {
            "mean_s": statistics.fmean(ttft_s),
            "p50_s": percentile(ttft_s, 50),
            "p90_s": percentile(ttft_s, 90),
            "p95_s": percentile(ttft_s, 95),
            "p99_s": percentile(ttft_s, 99),
            "max_s": max(ttft_s),
            "definition": (
                "scheduled request arrival to first token of the last "
                "completed per-query generation attempt"
            ),
            "record_count": len(ttft_s),
        },
        "latency_components": {
            "queue_wait": {
                "mean_s": statistics.fmean(queue_wait_s),
                "p50_s": percentile(queue_wait_s, 50),
                "p95_s": percentile(queue_wait_s, 95),
                "max_s": max(queue_wait_s),
            },
            "service": {
                "mean_s": statistics.fmean(service_latency_s),
                "p50_s": percentile(service_latency_s, 50),
                "p95_s": percentile(service_latency_s, 95),
                "max_s": max(service_latency_s),
            },
            "identity": "end_to_end_s = queue_wait_s + service_s",
        },
        "quality": quality,
        "recall_at_k": recall,
        "cost": {
            "actual_scanned_nprobe_mean": statistics.fmean(actual_scans),
            "actual_scanned_nprobe_p50": percentile(actual_scans, 50),
            "actual_scanned_nprobe_p95": percentile(actual_scans, 95),
            "estimated_scan_reduction": 1.0 - statistics.fmean(actual_scans) / args.nprobe,
            "generation_rounds_mean": statistics.fmean(
                len(value or []) for value in generation_traces
            ),
            "generation_rounds_p95": percentile(
                [len(value or []) for value in generation_traces], 95
            ),
            "generation_model_call_count": (
                pipeline._generation_sync_call_count
                + pipeline._generation_add_request_call_count
            ),
            "generation_engine_step_count": (
                pipeline._generation_step_call_count
            ),
            "generation_engine_completion_step_trace_count": len(
                pipeline._generation_engine_step_trace
            ),
            "generation_engine_idle_step_trace_omitted_count": (
                pipeline._generation_engine_idle_step_count
            ),
            "generation_active_request_high_water": (
                pipeline._generation_active_request_high_water
            ),
            "generation_chain_high_water": int(
                getattr(pipeline, "_generation_chain_high_water", 0)
            ),
            "event_queue_high_water": pipeline._event_queue_high_water,
            "event_queue_full_block_count": (
                pipeline._event_queue_full_block_count
            ),
            "event_queue_full_block_ms": (
                pipeline._event_queue_full_block_s * 1000.0
            ),
            "ranking_events_received": int(
                getattr(pipeline, "_ranking_events_received", 0)
            ),
            "ranking_events_dispatched": int(
                getattr(pipeline, "_ranking_events_dispatched", 0)
            ),
            "ranking_event_dispatch_reduction": (
                1.0
                - int(getattr(pipeline, "_ranking_events_dispatched", 0))
                / int(getattr(pipeline, "_ranking_events_received", 0))
                if int(getattr(pipeline, "_ranking_events_received", 0))
                else 0.0
            ),
            "generation_coalesce_flush_count": len(
                generation_coalesce_trace
            ),
            "generation_coalesce_wait_ms_mean": (
                statistics.fmean(generation_coalesce_wait_ms)
                if generation_coalesce_wait_ms
                else 0.0
            ),
            "generation_coalesce_wait_ms_p95": (
                percentile(generation_coalesce_wait_ms, 95)
                if generation_coalesce_wait_ms
                else 0.0
            ),
            "generation_coalesce_pending_peak_max": max(
                [
                    int(row.get("pending_peak") or 0)
                    for row in generation_coalesce_trace
                ]
                or [0]
            ),
            "native_scan_permit_capacity": int(
                native_scan_permit_global.get("capacity") or 0
            ),
            "native_scan_permit_peak_in_use": int(
                native_scan_permit_global.get("peak_in_use") or 0
            ),
            "native_scan_permit_wait_count": int(
                native_scan_permit_global.get("waits") or 0
            ),
            "native_scan_permit_wait_ms": (
                int(native_scan_permit_global.get("wait_ns") or 0) / 1e6
            ),
            "search_documents_attempted_mean": statistics.fmean(
                len(value or []) for value in search_ids
            ),
            "prefix_cached_tokens": cached_tokens,
            "prefix_prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "prefix_cached_token_ratio": (
                cached_tokens / prompt_tokens if prompt_tokens else None
            ),
            "prefix_chain_transition_count": len(cache_transition_rows),
            "prefix_chain_expected_cached_tokens": (
                expected_chain_cached_tokens
            ),
            "prefix_chain_reused_tokens": reused_chain_cached_tokens,
            "prefix_chain_cache_shortfall_tokens": (
                expected_chain_cached_tokens - reused_chain_cached_tokens
            ),
            "prefix_chain_reuse_efficiency": (
                reused_chain_cached_tokens / expected_chain_cached_tokens
                if expected_chain_cached_tokens
                else None
            ),
            "prefix_chain_cache_age_ms_mean": (
                statistics.fmean(cache_age_ms) if cache_age_ms else None
            ),
            "prefix_chain_cache_age_ms_p95": (
                percentile(cache_age_ms, 95) if cache_age_ms else None
            ),
            "retrieval_busy_s": interval_total(retrieval_intervals),
            "generation_busy_s": interval_total(generation_intervals),
            "retrieval_generation_overlap_s": overlap_s,
            "overlap_primary_metric": "same_request_retrieval_hidden_ratio",
            "overlap_primary_definition": (
                "sum over requests of same-request intersection between "
                "post-first-evidence active retrieval and generation service / "
                "sum over requests of post-first-evidence active retrieval"
            ),
            "active_retrieval_scope": (
                "query embedding + IVF coarse search + native start + "
                "native scan_codes; excludes session wait/park/poll"
            ),
            "active_generation_scope": (
                "AsyncLLMEngine request admission-to-terminal service intervals"
            ),
            "active_clock": "unix_time_ns",
            "same_request_post_first_evidence_retrieval_busy_s": (
                same_request_post_retrieval_busy_s
            ),
            "same_request_total_retrieval_busy_s": (
                same_request_total_retrieval_busy_s
            ),
            "same_request_retrieval_generation_overlap_s": (
                same_request_overlap_s
            ),
            "query_aligned_retrieval_hidden_ratio": (
                query_aligned_retrieval_hidden_ratio
            ),
            "same_request_retrieval_hidden_ratio": (
                same_request_hidden_ratio
            ),
            "same_request_query_audit": same_request_query_rows,
            "useful_overlap_breakdown": useful_breakdown,
            "retrieval_admission_delay_ms_mean": statistics.fmean(
                retrieval_admission_delay_ms
            ),
            "retrieval_admission_delay_ms_p50": percentile(
                retrieval_admission_delay_ms, 50
            ),
            "retrieval_admission_delay_ms_p95": percentile(
                retrieval_admission_delay_ms, 95
            ),
            "per_query_scan_interval_count": per_query_scan_interval_count,
            "per_query_generation_interval_count": (
                per_query_generation_interval_count
            ),
            "per_query_interval_accounting_exact": True,
            "global_overlap_secondary_metric": (
                "active_retrieval_hidden_ratio"
            ),
            "active_retrieval_prepare_interval_count": len(
                pipeline._retrieval_prepare_intervals_ns
            ),
            "active_retrieval_scan_interval_count": len(
                pipeline._retrieval_scan_intervals_ns
            ),
            "active_generation_interval_count": len(
                pipeline._generation_service_intervals_ns
            ),
            "active_retrieval_busy_s": active_retrieval_busy_s,
            # Absolute, merged half-open intervals are retained so resource
            # samples can be gated to periods when retrieval was actually
            # executing.  This deliberately excludes model loading, arrival
            # gaps, generation-only work, and retrieval queue wait.
            "resource_active_retrieval_intervals_ns": [
                [int(start_ns), int(stop_ns)]
                for start_ns, stop_ns in merge_intervals(
                    active_retrieval_intervals_ns
                )
            ],
            # Preserve raw generation-service intervals for independent audit.
            "resource_active_generation_intervals_ns": [
                [int(start_ns), int(stop_ns)]
                for start_ns, stop_ns in merge_intervals(
                    pipeline._generation_service_intervals_ns
                )
            ],
            "active_generation_busy_s": active_generation_busy_s,
            "active_retrieval_generation_overlap_s": active_overlap_s,
            "active_retrieval_generation_union_s": active_union_s,
            "active_retrieval_hidden_ratio": active_overlap_over_retrieval,
            "active_generation_coverage_ratio": active_overlap_over_generation,
            "active_shorter_stage_coverage_ratio": active_overlap_over_min_busy,
            "active_overlap_jaccard": active_overlap_jaccard,
            "active_search_scope": (
                "post-embedding native IVF scan_codes execution only; "
                "excludes query encoding, coarse-search preparation, "
                "session wait/park/poll, and document reading"
            ),
            "active_search_interval_count": len(
                pipeline._retrieval_scan_intervals_ns
            ),
            "active_search_busy_s": active_search_busy_s,
            "active_search_generation_overlap_s": active_search_overlap_s,
            "active_search_generation_union_s": active_search_union_s,
            "active_search_overlap_ratio": (
                active_search_overlap_over_retrieval
            ),
            "active_search_overlap_jaccard": active_search_overlap_jaccard,
            "resource_active_search_intervals_ns": [
                [int(start_ns), int(stop_ns)]
                for start_ns, stop_ns in merge_intervals(
                    active_search_intervals_ns
                )
            ],
            "prefetch_advised_lists_mean": statistics.fmean(
                prefetch_advised_lists
            ),
            "prefetch_advised_gib": sum(prefetch_advised_bytes) / (1024 ** 3),
            "prefetch_advice_errors": sum(prefetch_advice_errors),
            "prefetch_ahead_lists_max": max(
                max(0, high - scanned)
                for high, scanned in zip(prefetch_high_water, actual_scans)
            ),
            "unchanged_heap_list_rate": (
                sum(unchanged_heap_lists) / sum(actual_scans)
                if sum(actual_scans)
                else 0.0
            ),
        },
        "mechanism_diagnostics": {
            "stable_count": sum(stable),
            "stable_rate": sum(stable) / len(output),
            "full_nprobe_count": sum(full),
            "early_answer_count": sum(early),
            "early_answer_rate": sum(early) / len(output),
            "lexical_no_skip_count": sum(
                sum(
                    str(row.get("reason")) == "zero_entity_overlap"
                    for row in (rows or [])
                )
                for rows in answerability_skip_traces
            ),
            "lexical_no_skip_request_count": sum(
                any(
                    str(row.get("reason")) == "zero_entity_overlap"
                    for row in (rows or [])
                )
                for rows in answerability_skip_traces
            ),
            "scheduled_search_gate_skip_count": sum(
                sum(
                    str(row.get("reason"))
                    == "scheduled_search_gate_skip"
                    for row in (rows or [])
                )
                for rows in answerability_skip_traces
            ),
            "scheduled_search_gate_skip_request_count": sum(
                any(
                    str(row.get("reason"))
                    == "scheduled_search_gate_skip"
                    for row in (rows or [])
                )
                for rows in answerability_skip_traces
            ),
            "terminal_bulk_final_request_count": sum(
                bool(value) for value in terminal_bulk_armed
            ),
            "terminal_bulk_final_appended_document_count": sum(
                int(value or 0) for value in terminal_bulk_appended
            ),
            "first_search_generation_at_m2_target_count": sum(
                budget == args.m2_first_snapshot_nprobe
                for budget in first_generation_budgets
            ),
            "first_search_generation_terminal_before_target_count": sum(
                reason == "retrieval_terminal_before_target"
                for reason in m2_first_snapshot_reasons
            ),
            "final_answer_count": sum(bool(value) for value in final_done),
            "final_prompt_token_prefix_aligned_count": sum(
                bool(value) for value in final_prompt_token_prefix_aligned
            ),
            # Retain the historical manifest key for downstream-schema
            # compatibility; it now counts only leakage of the unambiguous
            # internal control sentinel, not legitimate task answers "No".
            "forbidden_no_output_count": len(leaked_control_indices),
            "stale_snapshot_search_generation_count": len(
                repeated_or_stale_search_budgets
            ),
            "answer_source_counts": dict(Counter(str(value) for value in answer_sources)),
            "verification_decision_counts": dict(decisions),
            "verification_reason_counts": dict(reasons),
            "generation_coalesce_enabled": bool(
                args.event_driven_generation
                and args.generation_coalesce_ms > 0
            ),
            "generation_coalesce_flush_reason_counts": dict(
                generation_coalesce_reasons
            ),
            "generation_coalesce_trace": generation_coalesce_trace,
            "effective_answerability_gate_logprobs": (
                2
                if args.answerability_structured_output
                and args.minimal_structured_gate_logprobs
                else args.answerability_gate_logprobs
            ),
            "conversational_prefix_enabled": bool(
                args.conversational_prefix
            ),
            "minimal_control_tail_enabled": bool(
                args.minimal_control_tail
            ),
            "conversational_prefix_check_count": sum(
                len(values or ()) for values in conversation_prefix_checks
            ),
            "conversational_prefix_violation_count": sum(
                not bool(value)
                for values in conversation_prefix_checks
                for value in (values or ())
            ),
            "stable_then_resumed_to_full_count": stable_then_full,
            "next_batch_bootstrap_overlap_count": bootstrap_overlap,
            "next_batch_bootstrap_before_previous_answer_count": (
                legal_bootstrap_overlap_count
            ),
            "next_batch_session_start_before_previous_answer_count": (
                legal_session_overlap_count
            ),
            "next_batch_generation_start_before_previous_answer_count": (
                legal_generation_overlap_count
            ),
            "next_batch_overlap_opportunity_count": expected_lookahead_windows,
            "adjacent_batch_session_overlap_count": (
                session_start_before_previous_done
            ),
            "adjacent_batch_bootstrap_overlap_count": (
                bootstrap_before_previous_done
            ),
            "adjacent_batch_generation_overlap_count": (
                generation_start_before_previous_done
            ),
            "next_batch_overlap_window_details": m3_window_overlap_details,
            "max_active_retrieval_sessions": pipeline._max_active_sessions_observed,
            "retrieval_paused_during_generation": False,
            "same_batch_background_retrieval": True,
            "next_batch_prefetch_enabled": enable_m3b,
            "m3_retrieval_schedule": (
                "scanner_wide_full_continuous_batching"
                if args.full_retrieval_continuous_batching
                else "dual_live_native_sessions_round_robin"
                if enable_m3b and args.event_driven_generation
                else "legacy_bootstrap_then_park"
                if enable_m3b
                else "disabled"
            ),
            "lookahead_max_budget_while_previous_incomplete": dict(
                pipeline._lookahead_max_budget_while_previous_incomplete
            ),
            "lookahead_post_bootstrap_event_before_previous_answer_count": (
                pipeline._lookahead_post_bootstrap_event_before_previous_answer_count
            ),
            "legacy_lookahead_max_budget_before_promotion": dict(
                pipeline._lookahead_max_budget_before_promotion
            ),
            "legacy_lookahead_search_before_current_final_count": (
                pipeline._lookahead_search_before_current_final_count
            ),
            "native_sessions_started_count": (
                pipeline._native_sessions_started_count
            ),
            "native_sessions_closed_count": (
                pipeline._native_sessions_closed_count
            ),
            "native_close_accounted_query_count": (
                pipeline._native_close_accounted_query_count
            ),
            "native_all_sessions_closed": bool(
                native_close_accounting_ok
            ),
            "native_scan_permit_final_global": native_scan_permit_global,
            "native_scan_permit_sessions": native_scan_permit_sessions,
        },
        "generation_call_trace": generation_call_trace,
        "final_length_repair_count": final_length_repair_count,
        "final_truncated_count": final_truncated_count,
        "generation_engine_step_trace": generation_engine_step_trace,
        "trace_diagnostics": {
            "enabled": bool(args.trace_generation_details),
            "full_prompt_text_included": bool(
                args.trace_generation_details
                and args.num_requests <= args.trace_full_prompts_max_requests
            ),
            "full_prompt_request_limit": args.trace_full_prompts_max_requests,
            "generation_call_count": len(generation_call_trace),
            "generation_engine_step_count": len(
                generation_engine_step_trace
            ),
            "generation_engine_step_call_count": (
                pipeline._generation_step_call_count
            ),
            "generation_engine_idle_step_trace_omitted_count": (
                pipeline._generation_engine_idle_step_count
            ),
            "per_sample_generation_and_rank_trajectory": bool(
                args.trace_generation_details
            ),
        },
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument(
        "--shared-mechanism-profile",
        choices=sorted(SHARED_MECHANISM_PROFILES),
        help=(
            "run an independent mechanism profile through the common "
            "progressive admission, batching, and scheduling path"
        ),
    )
    parser.add_argument("--dataset-name", default="nq")
    parser.add_argument("--measured-split", default="test")
    parser.add_argument("--warmup-split", default="dev")
    parser.add_argument("--num-requests", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--arrival-rate", type=float, default=24.0)
    parser.add_argument(
        "--arrival-model",
        choices=("legacy_second_batches", "continuous_poisson", "closed_loop"),
        default="continuous_poisson",
    )
    parser.add_argument(
        "--batching-window-ms",
        type=float,
        default=0.0,
        help="maximum non-anticipating admission wait for continuous Poisson requests",
    )
    parser.add_argument(
        "--native-dynamic-batch-limit",
        type=int,
        default=256,
        help="maximum pending requests coalesced by the stock FlashRAG worker",
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--nprobe", type=int, default=128)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--allow-native-parameter-experiment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "allow an explicit nprobe/topk grid; adaptive M2 uses the same "
            "topk as its maximum append-only evidence-document budget"
        ),
    )
    parser.add_argument("--bootstrap-nprobe", type=int, default=28)
    parser.add_argument(
        "--m2-first-snapshot-nprobe",
        type=int,
        default=None,
        help=(
            "pipeline-only first M2 top-1 latch budget; defaults to the "
            "physical bootstrap nprobe without changing native/M1 scanning"
        ),
    )
    parser.add_argument(
        "--allow-bootstrap-nprobe-experiment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "allow a controlled M2 bootstrap sweep while keeping native "
            "nprobe=128, topk=5, WRD, prompt, and scheduler settings fixed"
        ),
    )
    parser.add_argument(
        "--allow-wrd-parameter-experiment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "allow a controlled WRD parameter candidate while keeping the "
            "dataset, native endpoint, prompt, and scheduler fixed"
        ),
    )
    parser.add_argument("--heap-size", type=int, default=100)
    parser.add_argument("--monitor-size", type=int, default=100)
    parser.add_argument("--wrd-alpha", type=float, default=1.0)
    parser.add_argument("--wrd-epsilon", type=float, default=0.001)
    parser.add_argument("--wrd-window", type=int, default=6)
    parser.add_argument("--wrd-beta", type=float, default=0.4)
    parser.add_argument("--wrd-mu", type=float, default=0.75)
    parser.add_argument("--wrd-patience", type=int, default=16)
    parser.add_argument("--max-answer-tokens", type=int, default=5)
    parser.add_argument(
        "--search-max-tokens",
        type=int,
        default=None,
        help="optional SEARCHING decode cap; FINAL still uses --max-tokens",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--compact-answerability-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--single-snapshot-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--conversational-prefix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "render M2 as a true multi-turn append-only token prefix; requires "
            "compact answerability and the single-snapshot gate"
        ),
    )
    parser.add_argument(
        "--minimal-control-tail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "compress the repeated SEARCHING control while preserving the "
            "stock FlashRAG system and FINAL Question/Answer prompt; does not "
            "replay previous model decisions"
        ),
    )
    parser.add_argument("--answerability-logprob-margin", type=float, default=2.0)
    parser.add_argument(
        "--answerability-early-abort-below-margin",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "event-driven SEARCHING only: abort an A decode after its first "
            "token when the complete A-vs-N margin already fails the fixed gate"
        ),
    )
    parser.add_argument(
        "--answerability-semantic-rescue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "rescue a below-threshold A only with a deterministic online "
            "subject/relation support check; does not call the LLM again"
        ),
    )
    parser.add_argument(
        "--answerability-semantic-rescue-margin", type=float, default=14.0
    )
    parser.add_argument(
        "--answerability-semantic-rescue-min-question-coverage",
        type=float,
        default=0.75,
    )
    parser.add_argument("--answerability-gate-logprobs", type=int, default=100)
    parser.add_argument(
        "--answerability-require-logprobs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--answerability-structured-output",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--gate-component-stage",
        choices=(
            "production",
            "raw",
            "structure",
            "grounding",
            "confidence",
            "full",
        ),
        default="production",
        help=(
            "experiment-only cumulative Gate validation stage; production "
            "preserves the original behavior"
        ),
    )
    parser.add_argument(
        "--accumulated-answerability",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "let each one-document M2 continuation judge all accumulated "
            "evidence while retaining the append-only prefix contract"
        ),
    )
    parser.add_argument(
        "--answerability-lexical-no-skip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "append but do not decode a SEARCHING round when the new document "
            "contains none of the question's subject-bearing tokens"
        ),
    )
    parser.add_argument(
        "--m2-search-gate-ordinals",
        default="all",
        help=(
            "comma-separated pre-final document ordinals that run the unified "
            "SEARCHING gate; every document is still appended one at a time"
        ),
    )
    parser.add_argument(
        "--bulk-final-on-retrieval-terminal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after an unaccepted SEARCHING, if WRD/max-nprobe froze the heap, "
            "append every remaining document without intermediate decodes and "
            "run one accumulated FINAL"
        ),
    )
    parser.add_argument(
        "--full-context-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Full Context control: keep the full retrieval pipeline but skip "
            "all SEARCHING decodes; after WRD/max-nprobe termination, generate "
            "once from the frozen top-k context"
        ),
    )
    parser.add_argument(
        "--event-driven-generation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--continuation-priority-scheduler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "use vLLM priority scheduling for event-driven M2: FINAL=-2, "
            "continuation=-1, immutable @4 bootstrap=0"
        ),
    )
    parser.add_argument(
        "--current-batch-priority-scheduler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "M3: scan the successor fast prefix, then yield its native scan "
            "permits until current-batch retrieval terminates"
        ),
    )
    parser.add_argument(
        "--full-retrieval-continuous-batching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "admit every released retrieval window into a bounded set of "
            "nonparked native sessions backed by one scanner-wide query queue"
        ),
    )
    parser.add_argument(
        "--max-active-retrieval-sessions",
        type=int,
        default=512,
        help="resource-bounded retrieval admission window",
    )
    parser.add_argument(
        "--max-online-embedding-batch",
        type=int,
        default=16,
        help=(
            "maximum number of already released requests encoded in one E5 "
            "execution batch; this never delays a request for batching"
        ),
    )
    parser.add_argument(
        "--generation-coalesce-ms",
        type=float,
        default=0.0,
        help=(
            "bounded M2 continuation-admission window while vLLM already has "
            "unfinished requests; zero disables coalescing"
        ),
    )
    parser.add_argument(
        "--generation-coalesce-max-batch",
        type=int,
        default=8,
        help="flush continuation admission immediately at this ready batch size",
    )
    parser.add_argument(
        "--generation-coalesce-max-steps",
        type=int,
        default=1,
        help="maximum useful vLLM engine steps before a coalesced admission flush",
    )
    parser.add_argument(
        "--generation-poll-backoff-ms",
        type=float,
        default=50.0,
        help=(
            "safety timeout for the shared retrieval/generation wake event; "
            "normal progress wakes immediately without busy polling"
        ),
    )
    parser.add_argument(
        "--event-queue-size",
        type=int,
        default=65536,
        help=(
            "bounded retrieval-to-generation event FIFO; sized for the "
            "scanner-wide multi-session scheduler"
        ),
    )
    parser.add_argument(
        "--coalesce-ranking-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "collapse a drained FIFO burst to each query's newest post-@4 "
            "heap snapshot before updating the Python state machine"
        ),
    )
    parser.add_argument(
        "--rolling-m3-scanners",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "replace completed native scanner sessions with the next released "
            "window even while older generation chains are unfinished"
        ),
    )
    parser.add_argument(
        "--max-generation-chains",
        type=int,
        default=512,
        help="maximum unfinished M2 chains admitted to vLLM at once",
    )
    parser.add_argument(
        "--minimal-structured-gate-logprobs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="request only top-2 gate logprobs when regex constrains output to A/N",
    )
    parser.add_argument(
        "--prefix-block-alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-input-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "fp8"),
        default="auto",
        help="vLLM KV-cache storage type; fp8 doubles the prefix working set",
    )
    parser.add_argument(
        "--calculate-kv-scales",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="dynamically calibrate K/V scales when the KV cache uses fp8",
    )
    parser.add_argument(
        "--kv-free-block-policy",
        choices=("heat", "lru"),
        default="heat",
        help=(
            "free KV block eviction policy in the deployed vLLM build; "
            "lru avoids the local heat/time heap's MRU-like turnover"
        ),
    )
    parser.add_argument("--omp-threads", type=int, default=64)
    parser.add_argument(
        "--enable-native-continuous-scanner",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--progressive-retrieval-backend",
        choices=[PROGRESSIVE_RETRIEVAL_BACKEND],
        default=PROGRESSIVE_RETRIEVAL_BACKEND,
    )
    parser.add_argument("--native-worker-threads", type=int, default=64)
    parser.add_argument(
        "--native-global-scan-permits",
        type=int,
        default=None,
        help=(
            "scanner-global physical list-scan capacity; defaults inside the "
            "extension to native-worker-threads"
        ),
    )
    parser.add_argument("--native-lookahead-lists", type=int, default=8)
    parser.add_argument(
        "--native-continuation-quantum",
        type=int,
        default=0,
        help=(
            "maximum continuation IVF lists per persistent queue dispatch; "
            "zero preserves the unlimited production path"
        ),
    )
    parser.add_argument(
        "--native-request-aligned-priority",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "prefer quantumized continuations whose request has generation in flight"
        ),
    )
    parser.add_argument(
        "--native-request-aligned-priority-burst",
        type=int,
        default=0,
        help=(
            "maximum consecutive request-aligned priority dispatches before "
            "one FIFO fairness dispatch; zero preserves unlimited priority"
        ),
    )
    parser.add_argument(
        "--native-request-aligned-max-fifo-wait-ms",
        type=int,
        default=0,
        help=(
            "force the FIFO head once its queue residence reaches this bound; "
            "zero disables the age guard"
        ),
    )
    parser.add_argument("--native-scanner-module", default="harmorag_native_scanner")
    parser.add_argument(
        "--trace-generation-details",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "record generation-call prompts/timing/cache data and per-sample "
            "generation, verification, and retrieval-rank trajectories"
        ),
    )
    parser.add_argument(
        "--trace-full-prompts-max-requests",
        type=int,
        default=128,
        help=(
            "store decoded prompt text only when num_requests is at most this "
            "value; hashes/token counts are always retained when tracing"
        ),
    )
    parser.add_argument(
        "--native-scanner-source",
        default=str(PROJECT_ROOT / "native" / "harmorag_native_scanner.cpp"),
    )
    parser.add_argument("--gpu-id", default="0,1")
    parser.add_argument("--generator-gpu-num", type=int, default=1)
    parser.add_argument("--generator-model-name", default=None)
    parser.add_argument("--retrieval-device", default="cuda:1")
    parser.add_argument("--generator-model-path", required=True)
    parser.add_argument("--retriever-model-path", required=True)
    parser.add_argument(
        "--index-path",
        required=True,
    )
    parser.add_argument(
        "--corpus-path", required=True
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validate_args(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["VLLM_KV_FREE_BLOCK_POLICY"] = args.kv_free_block_policy
    args.mode = args.group
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.arrival_model == "continuous_poisson":
        windows = build_continuous_poisson_windows(
            args.num_requests,
            args.arrival_rate,
            args.seed,
            args.batching_window_ms,
        )
    elif args.arrival_model == "closed_loop":
        windows = build_closed_loop_windows(args.num_requests)
    else:
        windows = build_legacy_windows(args.num_requests, args.arrival_rate, args.seed)
    config = common_config(args)
    # ``common_config`` is shared with the original single-GPU artifact and
    # intentionally hard-codes the 8B defaults.  Restore the deployment
    # identity supplied by this benchmark before the AsyncLLMEngine is built;
    # otherwise a Qwen3-32B/TP4 or Llama3-70B/TP8 run silently attempts TP1.
    config["generator_gpu_num"] = int(args.generator_gpu_num)
    config["generator_model"] = str(
        args.generator_model_name or Path(args.generator_model_path).name
    )
    config["enable_prefix_caching"] = True
    config["vllm_kv_free_block_policy"] = args.kv_free_block_policy
    engine_kwargs = {
        "disable_log_stats": False,
        "max_num_seqs": int(args.max_num_seqs),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
    }
    if (
        args.continuation_priority_scheduler
        or args.current_batch_priority_scheduler
    ):
        engine_kwargs["scheduling_policy"] = "priority"
    if args.kv_cache_dtype != "auto":
        engine_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
        engine_kwargs["calculate_kv_scales"] = bool(args.calculate_kv_scales)
    config["vllm_engine_kwargs"] = engine_kwargs
    measured, warmup = load_data(
        config,
        args.num_requests,
        args.warmup,
        measured_split=args.measured_split,
        warmup_split=args.warmup_split,
    )

    if args.shared_mechanism_profile is not None:
        result = run_progressive(
            args,
            config,
            measured,
            warmup,
            windows,
            mode="full",
        )
    elif args.group == "native":
        result = run_native(args, config, measured, warmup, windows)
    elif args.group == "m1":
        result = run_m1(args, config, measured, warmup, windows)
    elif args.group == "m3":
        result = run_m3(args, config, measured, warmup, windows)
    else:
        result = run_progressive(
            args,
            config,
            measured,
            warmup,
            windows,
            mode=args.group,
        )
    effective_group = args.shared_mechanism_profile or args.group
    effective_mechanisms = (
        SHARED_MECHANISM_PROFILES[args.shared_mechanism_profile]
        if args.shared_mechanism_profile is not None
        else GROUPS[args.group]
    )
    result["group"] = effective_group
    result["mechanisms"] = dict(effective_mechanisms)
    generator_model_name = str(
        args.generator_model_name or Path(args.generator_model_path).name
    )
    is_qwen3_generator = generator_model_name.casefold().startswith("qwen3")
    thinking_count = sum(
        str(sample.get("prediction", "") or "").lower().count("<think>")
        + str(sample.get("prediction", "") or "").lower().count("</think>")
        for sample in result.get("samples", ())
    )
    if is_qwen3_generator and thinking_count:
        raise RuntimeError(
            f"Qwen thinking was disabled but {thinking_count} thinking markers were emitted"
        )
    result["thinking_count"] = thinking_count
    result["manifest"] = {
        "generation_stop": ["\n", "\r", "<|eot_id|>"],
        "group": effective_group,
        "execution_group": args.group,
        "shared_mechanism_profile": args.shared_mechanism_profile,
        "dataset_name": args.dataset_name,
        "measured_split": args.measured_split,
        "warmup_split": args.warmup_split,
        "mechanisms": dict(effective_mechanisms),
        "num_requests": args.num_requests,
        "warmup": args.warmup,
        "arrival_rate": args.arrival_rate,
        "arrival_seed": args.seed,
        "visible_gpu_ids": str(args.gpu_id),
        "tensor_parallel_size": int(args.generator_gpu_num),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "generator_max_input_len": int(args.max_input_len),
        "max_num_seqs": int(args.max_num_seqs),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "retrieval_device": str(args.retrieval_device),
        "kv_cache_dtype": args.kv_cache_dtype,
        "calculate_kv_scales": bool(args.calculate_kv_scales),
        "kv_free_block_policy": str(args.kv_free_block_policy),
        "continuation_priority_scheduler": bool(
            args.continuation_priority_scheduler
        ),
        "current_batch_priority_scheduler": bool(
            args.current_batch_priority_scheduler
        ),
        "full_retrieval_continuous_batching": bool(
            args.full_retrieval_continuous_batching
        ),
        "max_active_retrieval_sessions": int(
            args.max_active_retrieval_sessions
        ),
        "max_online_embedding_batch": int(args.max_online_embedding_batch),
        "accumulated_answerability": bool(
            args.accumulated_answerability
        ),
        "answerability_lexical_no_skip": bool(
            args.answerability_lexical_no_skip
        ),
        "gate_component_stage": str(args.gate_component_stage),
        "m2_search_gate_ordinals": list(args.m2_search_gate_ordinals),
        "bulk_final_on_retrieval_terminal": bool(
            args.bulk_final_on_retrieval_terminal
        ),
        "full_context_only": bool(args.full_context_only),
        "answerability_semantic_rescue": bool(
            args.answerability_semantic_rescue
        ),
        "answerability_semantic_rescue_margin": float(
            args.answerability_semantic_rescue_margin
        ),
        "arrival_model": str(args.arrival_model),
        "batching_window_ms": (
            float(args.batching_window_ms)
            if args.arrival_model == "continuous_poisson"
            else 0.0 if args.arrival_model == "closed_loop" else 1000.0
        ),
        "arrival_semantics": (
            "vLLM 0.11.2 serve burstiness=1 seeded Gamma(shape=1) intervals, "
            "cumulative delays normalized to N/rate; immediate per-request admission"
            if args.arrival_model == "continuous_poisson"
            else (
                "closed-loop one outstanding request; each observed retrieval-session "
                "start is the next request arrival and queue latency is zero"
                if args.arrival_model == "closed_loop"
                else "one max(1, Poisson(lambda)) microbatch released per second"
            )
        ),
        "num_windows": len(windows),
        "arrival_trace_sha256": trace_sha256(
            np.asarray([value for window in windows for value in window_arrivals(window)])
        ),
        "last_release_s": windows[-1].release_s,
        "last_request_arrival_s": window_arrivals(windows[-1])[-1],
        "achieved_offered_rate_req_s": (
            args.num_requests / window_arrivals(windows[-1])[-1]
            if args.num_requests > 1 and window_arrivals(windows[-1])[-1] > 0
            else None
        ),
        "native_nprobe": args.nprobe,
        "native_topk": args.topk,
        "retrieval_backend": (
            STOCK_RETRIEVAL_BACKEND
            if args.group in {"native", "m3"}
            else PROGRESSIVE_RETRIEVAL_BACKEND
        ),
        "native_scanner": {
            "enabled": args.enable_native_continuous_scanner,
            "scheduler": (
                "scanner_wide_full_continuous_batching"
                if args.full_retrieval_continuous_batching
                else "legacy_current_successor"
                if args.rolling_m3_scanners
                else "single_session"
            ),
            "module": args.native_scanner_module,
            "module_path": str(args.native_scanner_module_path),
            "module_sha256": args.native_scanner_module_sha256,
            "source": str(args.native_scanner_source),
            "source_sha256": args.native_scanner_source_sha256,
            "permit_header": str(args.native_scanner_header),
            "permit_header_sha256": args.native_scanner_header_sha256,
            "initial_nprobe": args.bootstrap_nprobe,
            "remaining_nprobe": args.nprobe - args.bootstrap_nprobe,
            "lookahead_lists": args.native_lookahead_lists,
            "continuation_quantum": args.native_continuation_quantum,
            "request_aligned_priority": bool(
                args.native_request_aligned_priority
            ),
            "request_aligned_priority_burst": (
                args.native_request_aligned_priority_burst
            ),
            "request_aligned_max_fifo_wait_ms": (
                args.native_request_aligned_max_fifo_wait_ms
            ),
            "omp_threads": args.native_worker_threads,
            "global_scan_permits": (
                args.native_global_scan_permits
                if args.native_global_scan_permits is not None
                else args.native_worker_threads
            ),
            "live_global_heap": True,
            "wrd_cancellable": True,
        },
        "native_session_close_accounting": {
            "required": args.group in {"m2", "m1_m2", "m1_m2_m3", "full"},
            "sessions_started_count": result.get(
                "mechanism_diagnostics", {}
            ).get("native_sessions_started_count", 0),
            "sessions_closed_count": result.get(
                "mechanism_diagnostics", {}
            ).get("native_sessions_closed_count", 0),
            "accounted_query_count": result.get(
                "mechanism_diagnostics", {}
            ).get("native_close_accounted_query_count", 0),
            "all_sessions_closed": result.get(
                "mechanism_diagnostics", {}
            ).get("native_all_sessions_closed", False),
        },
        "progressive_bootstrap_nprobe": args.bootstrap_nprobe,
        "progressive_m2_first_snapshot_nprobe": (
            args.m2_first_snapshot_nprobe
        ),
        "progressive_answerability_logprob_margin": float(
            args.answerability_logprob_margin
        ),
        "bootstrap_nprobe_experiment": bool(
            args.allow_bootstrap_nprobe_experiment
        ),
        "wrd_parameter_experiment": bool(
            args.allow_wrd_parameter_experiment
        ),
        "native_parameter_experiment": bool(
            args.allow_native_parameter_experiment
        ),
        "progressive_search_document_limit": int(args.topk),
        "progressive_event_queue": {
            "capacity": int(args.event_queue_size),
            "coalesce_ranking_events": bool(args.coalesce_ranking_events),
            "high_water": result.get("cost", {}).get(
                "event_queue_high_water", 0
            ),
            "full_block_count": result.get("cost", {}).get(
                "event_queue_full_block_count", 0
            ),
            "full_block_ms": result.get("cost", {}).get(
                "event_queue_full_block_ms", 0.0
            ),
            "ranking_events_received": result.get("cost", {}).get(
                "ranking_events_received", 0
            ),
            "ranking_events_dispatched": result.get("cost", {}).get(
                "ranking_events_dispatched", 0
            ),
        },
        "progressive_generation_chain_cap": int(args.max_generation_chains),
        "progressive_stop_policy": (
            "disabled_for_m2_only_full_nprobe128_or_answer_gate_cancel"
            if args.group == "m2"
            else "wrd_stable_or_nprobe128_terminal_no_resume"
        ),
        "progressive_heap_topk": args.heap_size,
        "progressive_wrd_topk": args.monitor_size,
        "wrd": {
            "alpha": args.wrd_alpha,
            "epsilon": args.wrd_epsilon,
            "window": args.wrd_window,
            "beta": args.wrd_beta,
            "mu": args.wrd_mu,
            "patience": args.wrd_patience,
        },
        "m2_prompt": (
            "true multi-turn conversational prefix: every next request begins "
            "with the complete preceding submitted token prompt, then appends "
            "an empty assistant closure and one new document/status turn; raw "
            "decisions stay in diagnostics and are not replayed into context; "
            "document five is a direct FINAL turn"
            if args.conversational_prefix
            else (
                "single-turn append-only evidence stream with a compressed "
                "SEARCHING control and the canonical Question/Answer cue; "
                "the system and FINAL prompt remain stock FlashRAG and prior "
                "model decisions are never replayed"
                if args.minimal_control_tail
                else "one stock-FlashRAG system/evidence prefix with a state-dependent "
                "user tail; SEARCHING adds the N-or-A control and FINAL renders "
                "exactly Question/Answer; each full prompt is canonically tokenized "
                "so vLLM reuses common blocks without inserting alignment padding"
            )
        ),
        "m2_early_acceptance": (
            "true ungated baseline: accept the first non-empty SEARCHING output "
            "without protocol, grounding, confidence, or retention validation"
            if args.gate_component_stage == "raw"
            else "structured A/N decision only; accept A without grounding, "
            "confidence, or retention validation"
            if args.gate_component_stage == "structure"
            else "structured A/N plus exact evidence grounding; no confidence "
            "or retention validation"
            if args.gate_component_stage == "grounding"
            else "structured A/N, grounding, and A-vs-N confidence margin; "
            "no live-heap retention validation"
            if args.gate_component_stage == "confidence"
            else "single SEARCHING decode; accept A only when it is not truncated, "
            "is a short exact span in the newly appended document, that support "
            "document remains in the latest heap top5, and the A-vs-N first-token "
            f"logprob margin is at least {args.answerability_logprob_margin}"
            if args.single_snapshot_gate
            else "legacy control: require the same normalized answer on two "
            "natural snapshots and two distinct exposed documents retained in "
            "the current top5"
        ),
        "m2_first_document": (
            "immutable top1 from the first-document target nprobe"
            f"={args.m2_first_snapshot_nprobe}, or from an earlier genuine WRD "
            "terminal heap; native/M1 fast physical prefix is "
            f"nprobe={args.bootstrap_nprobe}, and the scanner never waits for "
            "the generation-only batch latch"
        ),
        "m2_after_no": (
            "while retrieval is live, use the newest heap snapshot and keep "
            "one SEARCHING per appended document; after an unaccepted result "
            "and WRD/max-nprobe freezes the heap, append every remaining "
            "highest-ranked unsent document and run one FINAL; current-batch "
            "retrieval is never paused for generation"
            if args.bulk_final_on_retrieval_terminal
            else "while retrieval is live, use the newest heap snapshot; after "
            "M1 freezes the heap, append its highest-ranked unsent documents "
            "one at a time; current-batch retrieval is never paused for generation"
        ),
        "m2_document_limit": (
            "documents are appended one at a time; SEARCHING runs at ordinals "
            f"{list(args.m2_search_gate_ordinals)}; when "
            f"document {int(args.topk)} is successfully appended, cancel "
            "retrieval and decode that same append-only prefix directly as "
            f"FINAL, for at most {int(args.topk)} total decodes"
        ),
        "m2_final_decode": (
            "same unified FINAL prompt exactly once; min_tokens=1; No case/quote "
            "variants and thinking delimiters forbidden by vLLM bad_words; output "
            "is constrained to one non-newline span of at most 120 characters; "
            "a finish_reason=length completion is accepted as truncated without "
            "a retry"
            if is_qwen3_generator
            else "same unified FINAL prompt exactly once; min_tokens=1; No case/quote "
            "variants forbidden by vLLM bad_words; normal unconstrained Llama final "
            "decode; a finish_reason=length completion is accepted as truncated "
            "without a retry"
        ),
        "m2_final_output_policy": (
            {
                "structured_output_regex": r"[^<\n\r]{1,120}",
                "maximum_output_characters": 120,
                "bad_words": ["<think>", "</think>"],
                "length_finish_allowed": False,
                "length_repair_stop": ["\\n", "\\r", "."],
            }
            if is_qwen3_generator
            else None
        ),
        "m2_final_length_repair": {
            "trigger": None,
            "maximum_retries": 0,
            "same_evidence_prompt": False,
            "stop": [],
            "max_output_tokens": int(args.max_tokens),
            "accept_length_finish_as_truncated": True,
            "e2e_includes_original_and_retry": False,
            "normal_final_requests_unchanged": True,
            "observed_repair_count": int(
                result.get("final_length_repair_count", 0)
            ),
            "observed_truncated_count": int(
                result.get("final_truncated_count", 0)
            ),
        },
        "m2_search_decode": (
            "SEARCHING max_tokens="
            f"{int(args.search_max_tokens) if args.search_max_tokens is not None else min(args.max_tokens, max(4, 2 * args.max_answer_tokens + 2))}; "
            "FINAL retains the full generation limit"
        ),
        "m2_search_max_tokens": (
            int(args.search_max_tokens)
            if args.search_max_tokens is not None
            else min(args.max_tokens, max(4, 2 * args.max_answer_tokens + 2))
        ),
        "m2_generation_scheduler": {
            "event_driven": bool(args.event_driven_generation),
            "continuation_priority_scheduler": bool(
                args.continuation_priority_scheduler
            ),
            "current_batch_priority_scheduler": bool(
                args.current_batch_priority_scheduler
            ),
            "vllm_scheduling_policy": (
                "priority"
                if args.continuation_priority_scheduler
                or args.current_batch_priority_scheduler
                else "fcfs"
            ),
            "vllm_kv_free_block_policy": str(args.kv_free_block_policy),
            "conversational_prefix": bool(args.conversational_prefix),
            "minimal_control_tail": bool(args.minimal_control_tail),
            "continuation_coalesce_ms": args.generation_coalesce_ms,
            "continuation_coalesce_max_batch": (
                args.generation_coalesce_max_batch
            ),
            "continuation_coalesce_max_engine_steps": (
                args.generation_coalesce_max_steps
            ),
            "generation_poll_backoff_ms": args.generation_poll_backoff_ms,
            "idle_engine_delayed": False,
            "retrieval_worker_blocked": bool(
                result.get("cost", {}).get("event_queue_full_block_count", 0)
            ),
            "configured_gate_logprobs": args.answerability_gate_logprobs,
            "effective_gate_logprobs": (
                2
                if args.answerability_structured_output
                and args.minimal_structured_gate_logprobs
                else args.answerability_gate_logprobs
            ),
            "early_abort_below_gate_margin": bool(
                args.answerability_early_abort_below_margin
            ),
        },
        "m3_same_batch_background": effective_mechanisms[
            "m3_same_batch_background"
        ],
        "m3_prefetch_depth": (
            int(args.max_active_retrieval_sessions)
            if args.full_retrieval_continuous_batching
            else 1 if effective_mechanisms["m3_next_batch_bootstrap"] else 0
        ),
        "m3_lookahead_policy": (
            "isolated_one_window_stock_retrieval_prefetch_during_native_generation"
            if effective_group == "m3"
            else
            "scanner_wide_full_continuous_batching_without_successor_parking"
            if args.full_retrieval_continuous_batching
            else
            "successor_scans_the_complete_fast_prefix_then_yields_native_"
            "permits_until_current_batch_retrieval_is_terminal"
            if effective_mechanisms["m3_next_batch_bootstrap"]
            and args.event_driven_generation
            and args.current_batch_priority_scheduler
            else
            "two_live_native_scanner_slots_roll_forward_across_released_"
            "windows_independently_of_bounded_generation_chains"
            if effective_mechanisms["m3_next_batch_bootstrap"]
            and args.event_driven_generation
            and args.rolling_m3_scanners
            else "released_current_and_next_windows_scan_continuously_in_two_"
            "native_sessions_polled_round_robin_without_bootstrap_parking"
            if effective_mechanisms["m3_next_batch_bootstrap"]
            and args.event_driven_generation
            else f"legacy_bootstrap_to_nprobe{args.bootstrap_nprobe}_then_park_until_current"
            if effective_mechanisms["m3_next_batch_bootstrap"]
            else "disabled"
        ),
        "m3_lookahead_generation_policy": (
            "isolated_m3_uses_stock_one_shot_generation_after_each_full_nprobe128_retrieval"
            if effective_group == "m3"
            else
            "every released retrieval window enters the bounded scanner-wide "
            f"queue (max sessions={args.max_active_retrieval_sessions}); generation "
            f"starts from each window's nprobe{args.m2_first_snapshot_nprobe} top1 "
            "as soon as its M2 latch barrier is consumed; no current/successor roles"
            if args.full_retrieval_continuous_batching
            else
            f"retrieval starts at physical nprobe{args.bootstrap_nprobe}; "
            "generation starts from the lookahead batch "
            f"nprobe{args.m2_first_snapshot_nprobe} top1 as soon "
            "as its M2 latch batch barrier is consumed while both retrieval sessions "
            "continue scanning"
            if effective_mechanisms["m3_next_batch_bootstrap"]
            and args.event_driven_generation
            else "legacy control: no generation before promotion; after "
            f"promotion, generate from the locked nprobe{args.m2_first_snapshot_nprobe} "
            "top1 while scanning "
            "resumes"
            if effective_mechanisms["m3_next_batch_bootstrap"]
            else "disabled"
        ),
        "benchmark_trace": {
            "generation_details_enabled": bool(args.trace_generation_details),
            "full_prompt_text_max_requests": args.trace_full_prompts_max_requests,
            "per_sample_generation_verification_rank_trajectory": bool(
                args.trace_generation_details
            ),
        },
        "generator_model": args.generator_model_path,
        "generator_model_name": generator_model_name,
        "thinking_disabled": bool(is_qwen3_generator),
        "enable_thinking": False if is_qwen3_generator else None,
        "thinking_count": thinking_count,
        "serving_engine": "vllm_AsyncLLMEngine",
        "generation_admission_api": (
            "AsyncLLMEngine.generate async generators; one coroutine per live "
            "HarmoRAG generation chain"
        ),
        "retriever_model": args.retriever_model_path,
        "index": args.index_path,
        "corpus": args.corpus_path,
        "implementation_sha256": {
            "benchmark": file_sha256(Path(__file__).resolve()),
            "pipeline": file_sha256(
                PROJECT_ROOT / "harmorag" / "pipeline" / "progressive_pipeline.py"
            ),
            "retriever": file_sha256(
                PROJECT_ROOT / "harmorag" / "retriever" / "progressive_retriever.py"
            ),
            "generator": file_sha256(
                PROJECT_ROOT / "harmorag" / "generator" / "generator.py"
            ),
            "native_scanner": args.native_scanner_source_sha256,
            "native_scan_permit_limiter": args.native_scanner_header_sha256,
            "vllm_kv_cache_utils": module_file_sha256(
                "vllm.v1.core.kv_cache_utils"
            ),
        },
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    summary = {key: value for key, value in result.items() if key != "samples"}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
