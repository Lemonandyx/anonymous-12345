"""Progressive IVF retrieval with weighted-rank FAST pause and exact draining.

The implementation deliberately scans only the *new* IVF lists at each budget
stage.  Re-running ``index.search`` with a larger ``nprobe`` would rescan all
previous lists and can make an apparent early-stop policy more expensive than a
single full search.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import threading
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from harmorag.retriever.retriever import DenseRetriever
from harmorag.retriever.encoder import Encoder, STEncoder
from harmorag.retriever.utils import load_docs

try:
    from harmorag_native_scanner import (
        ContinuousIVFScanner as _NativeContinuousIVFScanner,
    )
except ImportError:  # Optional: the default synchronous retriever remains usable.
    _NativeContinuousIVFScanner = None


FAST = "FAST"
PAUSED_FAST = "PAUSED_FAST"
DRAINING = "DRAINING"
FINISHED = "FINISHED"


def _config_value(config, key: str, default):
    if key in config:
        value = config[key]
        if value is not None:
            return value
    return default


def power_rank_weights(k: int, alpha: float) -> np.ndarray:
    """Return ``w_r = r^-alpha / sum_i(i^-alpha)`` for one-based ranks."""

    if k < 1:
        raise ValueError("k must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive so earlier ranks weigh more")
    ranks = np.arange(1, k + 1, dtype=np.float64)
    weights = np.power(ranks, -float(alpha))
    return weights / weights.sum()


def weighted_rank_delta(
    previous_ids: Sequence[int], current_ids: Sequence[int], weights: Sequence[float]
) -> float:
    """Bidirectional weighted rank movement/new/drop cost in ``[0, 1]``.

    This is the production form of the WRS/CWRS formula in the local RAG
    experiments.  Missing FAISS labels (``-1``) are ignored.
    """

    k = len(weights)
    if k < 1:
        raise ValueError("weights must not be empty")
    previous = [int(doc_id) for doc_id in previous_ids[:k] if int(doc_id) >= 0]
    current = [int(doc_id) for doc_id in current_ids[:k] if int(doc_id) >= 0]
    if len(previous) != len(set(previous)) or len(current) != len(set(current)):
        raise ValueError("top-k document IDs must be unique")

    previous_positions = {doc_id: rank for rank, doc_id in enumerate(previous)}
    current_positions = {doc_id: rank for rank, doc_id in enumerate(current)}

    previous_cost = 0.0
    for doc_id, old_rank in previous_positions.items():
        new_rank = current_positions.get(doc_id)
        if new_rank is None:
            previous_cost += float(weights[old_rank])
        elif new_rank != old_rank:
            previous_cost += abs(float(weights[old_rank]) - float(weights[new_rank]))

    current_cost = 0.0
    for doc_id, new_rank in current_positions.items():
        old_rank = previous_positions.get(doc_id)
        if old_rank is None:
            current_cost += float(weights[new_rank])
        elif old_rank != new_rank:
            current_cost += abs(float(weights[old_rank]) - float(weights[new_rank]))

    return float(np.clip(0.5 * (previous_cost + current_cost), 0.0, 1.0))


def ewma_recent(values: Sequence[float], beta: float) -> float:
    if not values:
        return 0.0
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    array = np.asarray(values, dtype=np.float64)
    exponents = np.arange(array.shape[0] - 1, -1, -1, dtype=np.float64)
    weights = np.power(float(beta), exponents)
    weights /= weights.sum()
    return float(np.dot(array, weights))


@dataclass(frozen=True)
class StabilityObservation:
    budget: int
    delta: Optional[float]
    average_delta: Optional[float]
    volatility: Optional[float]
    risk: Optional[float]
    stable_streak: int
    should_stop: bool


class WeightedRankStabilityMonitor:
    """Per-query online WRS/CWRS state."""

    def __init__(
        self,
        topk: int,
        alpha: float = 1.0,
        threshold: float = 0.0025,
        window: int = 5,
        beta: float = 0.6,
        volatility_weight: float = 0.5,
        patience: int = 20,
        min_budget: int = 4,
    ) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if window < 2:
            raise ValueError("window must be at least 2")
        if not 0 <= beta < 1:
            raise ValueError("beta must be in [0, 1)")
        if volatility_weight < 0:
            raise ValueError("volatility_weight must be non-negative")
        if patience < 1:
            raise ValueError("patience must be positive")
        if min_budget < 1:
            raise ValueError("min_budget must be positive")

        self.weights = power_rank_weights(topk, alpha)
        self.threshold = float(threshold)
        self.window = int(window)
        self.beta = float(beta)
        self.volatility_weight = float(volatility_weight)
        self.patience = int(patience)
        self.min_budget = int(min_budget)

        self.previous_ids: Optional[Tuple[int, ...]] = None
        self.delta_history: Deque[float] = deque(maxlen=self.window)
        self.stable_streak = 0
        self.last_budget = 0

    def observe(self, ids: Sequence[int], budget: int) -> StabilityObservation:
        if budget <= self.last_budget:
            raise ValueError("retrieval budget must increase monotonically")
        current_ids = tuple(int(doc_id) for doc_id in ids[: len(self.weights)])
        self.last_budget = int(budget)

        if self.previous_ids is None:
            self.previous_ids = current_ids
            return StabilityObservation(budget, None, None, None, None, 0, False)

        delta = weighted_rank_delta(self.previous_ids, current_ids, self.weights)
        self.previous_ids = current_ids
        self.delta_history.append(delta)

        if len(self.delta_history) < self.window:
            self.stable_streak = 0
            return StabilityObservation(budget, delta, None, None, None, 0, False)

        recent = list(self.delta_history)
        average_delta = ewma_recent(recent, self.beta)
        changes = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        volatility = ewma_recent(changes, self.beta)
        risk = average_delta + self.volatility_weight * volatility

        if budget >= self.min_budget and risk <= self.threshold:
            self.stable_streak += 1
        else:
            self.stable_streak = 0
        should_stop = self.stable_streak >= self.patience
        return StabilityObservation(
            budget,
            delta,
            average_delta,
            volatility,
            risk,
            self.stable_streak,
            should_stop,
        )


def merge_topk(
    previous_scores: np.ndarray,
    previous_ids: np.ndarray,
    new_scores: np.ndarray,
    new_ids: np.ndarray,
    topk: int,
    higher_is_better: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge two independent top-k result matrices without losing candidates."""

    if previous_scores.shape != previous_ids.shape or new_scores.shape != new_ids.shape:
        raise ValueError("score/id shapes must match")
    if previous_scores.shape[0] != new_scores.shape[0]:
        raise ValueError("both result matrices must have the same batch size")

    batch_size = previous_scores.shape[0]
    fill_score = -np.inf if higher_is_better else np.inf
    merged_scores = np.full((batch_size, topk), fill_score, dtype=np.float32)
    merged_ids = np.full((batch_size, topk), -1, dtype=np.int64)

    for row in range(batch_size):
        best_by_id = {}
        for score, doc_id in zip(
            np.concatenate((previous_scores[row], new_scores[row])),
            np.concatenate((previous_ids[row], new_ids[row])),
        ):
            doc_id = int(doc_id)
            if doc_id < 0:
                continue
            score = float(score)
            old_score = best_by_id.get(doc_id)
            if old_score is None or (score > old_score if higher_is_better else score < old_score):
                best_by_id[doc_id] = score

        ordered = sorted(
            best_by_id.items(),
            key=(lambda item: (-item[1], item[0])) if higher_is_better else (lambda item: (item[1], item[0])),
        )[:topk]
        for rank, (doc_id, score) in enumerate(ordered):
            merged_ids[row, rank] = doc_id
            merged_scores[row, rank] = score

    return merged_scores, merged_ids


@dataclass(frozen=True)
class ProgressiveQuerySnapshot:
    query_index: int
    batch_id: int
    budget: int
    document_ids: Tuple[int, ...]
    scores: Tuple[float, ...]
    delta: Optional[float]
    average_delta: Optional[float]
    volatility: Optional[float]
    risk: Optional[float]
    stable_streak: int
    top1_changed: bool
    top1_is_new: bool
    top1_document: Optional[dict]
    finished: bool
    stop_reason: Optional[str]
    phase: str = FAST
    scanned_budget: int = 0
    pause_reason: Optional[str] = None
    final_top5_ready: bool = False
    # ``scanned_budget`` is the contiguous merged-prefix heap version.
    # ``actual_scanned_nprobe`` additionally counts completed speculative
    # query-list tasks and is therefore the physical cost high-water mark.
    actual_scanned_nprobe: int = 0
    native_sequence: int = 0
    scheduled_high_water: int = 0
    computed_high_water: int = 0
    # Native OnDisk rolling-prefetch and heap-work observability.  These are
    # cumulative per-query counters at this ranking version.
    prefetch_advised_lists: int = 0
    prefetch_advised_bytes: int = 0
    prefetch_advice_errors: int = 0
    prefetch_high_water: int = 0
    heap_updates: int = 0
    unchanged_heap_lists: int = 0


@dataclass(frozen=True)
class ProgressiveRetrievalEvent:
    batch_id: int
    budget: int
    snapshots: Tuple[ProgressiveQuerySnapshot, ...]
    batch_complete: bool
    has_runnable: bool = False
    terminal: bool = False


@dataclass
class _ProgressiveQueryState:
    query_index: int
    monitor: WeightedRankStabilityMonitor
    scores: np.ndarray
    ids: np.ndarray
    last_top1: Optional[int] = None
    seen_top1: set = field(default_factory=set)
    phase: str = FAST
    scanned_budget: int = 0
    pause_reason: Optional[str] = None
    final_top5_ready: bool = False
    finished: bool = False
    stop_reason: Optional[str] = None

    @property
    def runnable(self) -> bool:
        return not self.finished and self.phase in (FAST, DRAINING)


@dataclass
class ProgressiveBatchSession:
    batch_id: int
    queries: Tuple[str, ...]
    query_indices: Tuple[int, ...]
    embeddings: np.ndarray
    coarse_scores: np.ndarray
    coarse_ids: np.ndarray
    states: List[_ProgressiveQueryState]
    stage_index: int = 0

    @property
    def has_runnable(self) -> bool:
        """Whether at least one query can advance its IVF cursor."""

        return any(state.runnable for state in self.states)

    @property
    def terminal(self) -> bool:
        """Whether every query ended by stability, cancellation, or max budget."""

        return all(state.finished for state in self.states)

    @property
    def complete(self) -> bool:
        """Compatibility alias for terminal completion, as before."""

        return self.terminal


@dataclass
class _NativeQueryState:
    query_index: int
    local_index: int
    scores: np.ndarray
    ids: np.ndarray
    last_sequence: int = -1
    last_top1: Optional[int] = None
    seen_top1: set = field(default_factory=set)
    scanned_budget: int = 0
    actual_scanned_nprobe: int = 0
    scheduled_high_water: int = 0
    computed_high_water: int = 0
    finished: bool = False
    stop_reason: Optional[str] = None
    closed: bool = False
    scan_active_intervals_ns: List[Tuple[int, int]] = field(
        default_factory=list
    )

    @property
    def runnable(self) -> bool:
        return not self.finished


@dataclass
class NativeProgressiveBatchSession:
    """Python wrapper around one 2-D native handle for the whole batch."""

    batch_id: int
    queries: Tuple[str, ...]
    query_indices: Tuple[int, ...]
    embeddings: np.ndarray
    native_id: int
    states: List[_NativeQueryState]
    bootstrap_emitted: bool = False
    resumed: bool = False
    paused: bool = True
    closed: bool = False
    last_sequence: int = -1
    # A native batch publishes one sequence-ordered event stream for all
    # queries.  Non-blocking polling can therefore observe only a subset of
    # the nprobe=bootstrap snapshots on any one call.  Keep that partial
    # barrier on the session so later polls neither lose nor re-read it.  This
    # field is appended to preserve the legacy positional constructor layout.
    bootstrap_by_local: Dict[int, ProgressiveQuerySnapshot] = field(
        default_factory=dict
    )
    # Retained after native close() removes the handle, so benchmark/reporting
    # code can inspect scanner-global contention and this session's share.
    scan_permit_diagnostics: Dict[str, Any] = field(default_factory=dict)
    # Actual native scan_codes() intervals exported at close().  They use
    # Unix wall-clock nanoseconds so the benchmark can intersect them with
    # Python time.time_ns() generation intervals.  Session wait/park/poll
    # time is deliberately absent.
    scan_active_intervals_ns: List[Tuple[int, int]] = field(
        default_factory=list
    )

    @property
    def has_runnable(self) -> bool:
        # Bootstrap-parked sessions are resumable through advance_batch().
        return any(state.runnable for state in self.states)

    @property
    def terminal(self) -> bool:
        return all(state.finished for state in self.states)

    @property
    def complete(self) -> bool:
        # Keep this alias while the Pipeline supports both old and new
        # retriever sessions.
        return self.terminal


class ProgressiveIVFRetriever(DenseRetriever):
    """CPU IVF retriever that incrementally scans disjoint centroid slices."""

    def update_additional_setting(self):
        super().update_additional_setting()
        if self.use_reranker:
            raise ValueError("progressive retrieval does not currently support a reranker")
        if self.use_cache or self.save_cache:
            raise ValueError(
                "progressive retrieval does not use the upstream one-shot retrieval cache"
            )
        # The new state machine uses a per-query cursor rather than a global
        # geometric schedule: scan 0->bootstrap once, then advance one list at
        # a time.  ``budget_schedule`` remains as a compatibility/introspection
        # attribute for callers that only inspect its first/last entries.
        self.bootstrap_nprobe = int(
            _config_value(self._config, "progressive_bootstrap_nprobe", 28)
        )
        self.max_nprobe = int(
            _config_value(self._config, "progressive_max_nprobe", 128)
        )
        # Compatibility/introspection only: FAST no longer pauses at this
        # boundary.  The default therefore coincides with the terminal budget.
        self.fast_max_nprobe = int(
            _config_value(
                self._config, "progressive_fast_max_nprobe", self.max_nprobe
            )
        )
        if not 1 <= self.bootstrap_nprobe <= self.fast_max_nprobe <= self.max_nprobe:
            raise ValueError(
                "progressive nprobe limits must satisfy "
                "1 <= bootstrap_nprobe <= fast_max_nprobe <= max_nprobe"
            )
        self.budget_schedule = (self.bootstrap_nprobe,) + tuple(
            range(self.bootstrap_nprobe + 1, self.max_nprobe + 1)
        )

        # Search/merge a wider internal heap than the externally requested RAG
        # top-k.  WRD observes the leading 100 candidates from that same heap.
        self.heap_topk = int(_config_value(self._config, "progressive_heap_topk", 100))
        self.stability_topk = int(
            _config_value(self._config, "progressive_wrd_topk", 100)
        )
        if self.heap_topk < 1:
            raise ValueError("progressive_heap_topk must be positive")
        if not 1 <= self.stability_topk <= self.heap_topk:
            raise ValueError(
                "progressive_wrd_topk must be positive and no larger than progressive_heap_topk"
            )

        self.rank_alpha = float(
            _config_value(self._config, "progressive_wrd_alpha", 1.0)
        )
        self.stability_threshold = float(
            _config_value(self._config, "progressive_wrd_epsilon", 0.001)
        )
        self.stability_window = int(
            _config_value(self._config, "progressive_wrd_window", 6)
        )
        self.stability_beta = float(
            _config_value(self._config, "progressive_wrd_beta", 0.4)
        )
        self.stability_volatility_weight = float(
            _config_value(self._config, "progressive_wrd_mu", 0.75)
        )
        self.stability_patience = int(
            _config_value(self._config, "progressive_wrd_patience", 16)
        )
        self.enable_wrd = bool(
            _config_value(self._config, "progressive_enable_wrd", True)
        )
        # WRD observes the exact global heap from @1.  The physical fast-prefix
        # barrier controls publication/generation only and is not a statistical
        # warm-up point for the convergence detector.
        self.min_stability_budget = 1
        self.wrd_alpha = self.rank_alpha
        self.wrd_epsilon = self.stability_threshold
        self.wrd_window = self.stability_window
        self.wrd_beta = self.stability_beta
        self.wrd_mu = self.stability_volatility_weight
        self.wrd_patience = self.stability_patience

        # Kept only for compatibility with older configuration files.  A stable
        # WRD observation is terminal in the continuous-search contract and is
        # never resumed merely because answer generation returned ``No``.
        self.force_max_budget_for_fallback = bool(
            _config_value(self._config, "progressive_force_max_budget_for_fallback", False)
        )
        self.use_mmap = bool(_config_value(self._config, "progressive_index_mmap", True))
        self.retrieval_device = _config_value(self._config, "progressive_retrieval_device", None)
        self.omp_threads = int(_config_value(self._config, "progressive_omp_threads", 4))
        if self.omp_threads < 1:
            raise ValueError("progressive_omp_threads must be positive")
        self.use_native_continuous = bool(
            _config_value(
                self._config, "progressive_native_continuous_scan", False
            )
        )
        self.native_worker_threads = int(
            _config_value(
                self._config, "progressive_native_worker_threads", 24
            )
        )
        # This is scanner-global rather than per batch/session.  Keeping the
        # default equal to the historical worker count preserves single-batch
        # throughput while preventing M3's two live sessions from doubling the
        # number of simultaneous physical list scans.
        self.native_global_scan_permits_explicit = bool(
            "progressive_native_global_scan_permits" in self._config
            and self._config["progressive_native_global_scan_permits"] is not None
        )
        self.native_global_scan_permits = int(
            _config_value(
                self._config,
                "progressive_native_global_scan_permits",
                self.native_worker_threads,
            )
        )
        self.native_lookahead = int(
            _config_value(self._config, "progressive_native_lookahead", 8)
        )
        self.native_continuation_quantum = int(
            _config_value(
                self._config, "progressive_native_continuation_quantum", 0
            )
        )
        self.native_request_aligned_priority = bool(
            _config_value(
                self._config,
                "progressive_native_request_aligned_priority",
                False,
            )
        )
        self.native_request_aligned_priority_burst = int(
            _config_value(
                self._config,
                "progressive_native_request_aligned_priority_burst",
                0,
            )
        )
        self.native_request_aligned_max_fifo_wait_ms = int(
            _config_value(
                self._config,
                "progressive_native_request_aligned_max_fifo_wait_ms",
                0,
            )
        )
        self.native_poll_timeout_ms = int(
            _config_value(
                self._config, "progressive_native_poll_timeout_ms", 100
            )
        )
        if self.native_worker_threads < 1:
            raise ValueError("progressive_native_worker_threads must be positive")
        if self.native_global_scan_permits < 1:
            raise ValueError(
                "progressive_native_global_scan_permits must be positive"
            )
        if self.native_lookahead < 1:
            raise ValueError("progressive_native_lookahead must be positive")
        if self.native_continuation_quantum < 0:
            raise ValueError(
                "progressive_native_continuation_quantum must be non-negative"
            )
        if (
            self.native_request_aligned_priority
            and self.native_continuation_quantum < 1
        ):
            raise ValueError(
                "request-aligned priority requires a positive continuation quantum"
            )
        if self.native_request_aligned_priority_burst < 0:
            raise ValueError(
                "progressive_native_request_aligned_priority_burst must be non-negative"
            )
        if (
            self.native_request_aligned_priority_burst > 0
            and not self.native_request_aligned_priority
        ):
            raise ValueError(
                "request-aligned priority burst requires request-aligned priority"
            )
        if self.native_request_aligned_max_fifo_wait_ms < 0:
            raise ValueError(
                "progressive_native_request_aligned_max_fifo_wait_ms must be non-negative"
            )
        if (
            self.native_request_aligned_max_fifo_wait_ms > 0
            and not self.native_request_aligned_priority
        ):
            raise ValueError(
                "request-aligned FIFO wait guard requires request-aligned priority"
            )
        if self.native_poll_timeout_ms < 1:
            raise ValueError("progressive_native_poll_timeout_ms must be positive")
        self._native_scanner_factory = _config_value(
            self._config,
            "progressive_native_scanner_factory",
            _NativeContinuousIVFScanner,
        )
        self._native_scanner = None
        self._native_query_handles: Dict[int, Tuple[int, int]] = {}
        self._native_query_handles_lock = threading.Lock()

    def load_index(self):
        if self.use_faiss_gpu:
            raise ValueError(
                "ProgressiveIVFRetriever scans disjoint IVF lists on CPU; set faiss_gpu: false"
            )
        if self.index_path is None:
            raise ValueError("index_path is required")
        flags = 0
        if self.use_mmap:
            flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        self.index = faiss.read_index(self.index_path, flags) if flags else faiss.read_index(self.index_path)
        if not isinstance(self.index, faiss.IndexIVF):
            raise TypeError(
                f"Progressive retrieval requires a direct FAISS IndexIVF, got {type(self.index).__name__}"
            )
        self.ivf_index = self.index
        if not self.ivf_index.is_trained:
            raise ValueError("the IVF index is not trained")
        if self.ivf_index.quantizer.ntotal != self.ivf_index.nlist:
            raise ValueError("the IVF quantizer must contain exactly one centroid per list")
        if self.ivf_index.metric_type not in (faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2):
            raise ValueError("progressive merge currently supports only inner-product and L2 IVF indexes")
        if int(self.ivf_index.max_codes) != 0:
            raise ValueError("max_codes must be 0 so every selected IVF list is scanned completely")
        no_heap_init = int(getattr(faiss, "PARALLEL_MODE_NO_HEAP_INIT", 1024))
        if int(self.ivf_index.parallel_mode) & no_heap_init:
            raise ValueError("PARALLEL_MODE_NO_HEAP_INIT is incompatible with independent list slices")
        try:
            corpus_size = len(self.corpus)
        except (TypeError, AttributeError):
            corpus_size = None
        if corpus_size is not None and corpus_size != self.ivf_index.ntotal:
            raise ValueError(
                f"corpus rows ({corpus_size}) do not match index ntotal ({self.ivf_index.ntotal})"
            )
        if self.max_nprobe > self.ivf_index.nlist:
            raise ValueError(
                f"maximum budget {self.max_nprobe} exceeds index nlist={self.ivf_index.nlist}"
            )
        self.higher_is_better = self.ivf_index.metric_type == faiss.METRIC_INNER_PRODUCT
        self._search_lock = threading.Lock()
        faiss.omp_set_num_threads(self.omp_threads)
        # Native scanning is strictly opt-in.  Some legacy/unit-test callers
        # construct the retriever with ``__new__`` and inject only the old
        # synchronous fields, so a missing flag must preserve that path.
        if bool(getattr(self, "use_native_continuous", False)):
            factory = self._native_scanner_factory
            if factory is None:
                raise ImportError(
                    "progressive_native_continuous_scan=true requires the optional "
                    "harmorag_native_scanner module"
                )
            native_kwargs = dict(
                worker_threads=self.native_worker_threads,
                initial_nprobe=self.bootstrap_nprobe,
                max_nprobe=self.max_nprobe,
                heap_topk=self.heap_topk,
                lookahead=self.native_lookahead,
                enable_wrd=self.enable_wrd,
                wrd_alpha=self.rank_alpha,
                wrd_threshold=self.stability_threshold,
                wrd_window=self.stability_window,
                wrd_beta=self.stability_beta,
                wrd_mu=self.stability_volatility_weight,
                wrd_patience=self.stability_patience,
                wrd_topk=self.stability_topk,
                continuation_quantum=self.native_continuation_quantum,
                request_aligned_priority=self.native_request_aligned_priority,
                request_aligned_priority_burst=(
                    self.native_request_aligned_priority_burst
                ),
                request_aligned_max_fifo_wait_ms=(
                    self.native_request_aligned_max_fifo_wait_ms
                ),
            )
            # Omitting this keyword when the YAML key is absent keeps the
            # adapter source-compatible with a previously-built extension.
            # The new extension itself defaults the global capacity to
            # worker_threads. An explicit key intentionally fails fast against
            # an old binary, making a required rebuild visible at startup.
            if self.native_global_scan_permits_explicit:
                native_kwargs["global_scan_permits"] = (
                    self.native_global_scan_permits
                )
            self._native_scanner = factory(self.index_path, **native_kwargs)

    def load_model(self):
        if self.use_st:
            self.encoder = STEncoder(
                model_name=self.retrieval_method,
                model_path=self.retrieval_model_path,
                max_length=self.query_max_length,
                use_fp16=self.use_fp16,
                instruction=self.instruction,
                silent=self.silent,
                device=self.retrieval_device,
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
                silent=self.silent,
                device=self.retrieval_device,
            )

    def _new_monitor(self) -> WeightedRankStabilityMonitor:
        return WeightedRankStabilityMonitor(
            topk=self.stability_topk,
            alpha=self.rank_alpha,
            threshold=self.stability_threshold,
            window=self.stability_window,
            beta=self.stability_beta,
            volatility_weight=self.stability_volatility_weight,
            patience=self.stability_patience,
            min_budget=self.min_stability_budget,
        )

    def _search(self, *args, **kwargs):
        raise RuntimeError(
            "ProgressiveIVFRetriever does not support one-shot search(); "
            "use ProgressiveAsyncPipeline or start_batch()/advance_batch()"
        )

    def _batch_search(self, *args, **kwargs):
        raise RuntimeError(
            "ProgressiveIVFRetriever does not support one-shot batch_search(); "
            "use ProgressiveAsyncPipeline or start_batch()/advance_batch()"
        )

    def _start_native_batch(
        self,
        queries: Tuple[str, ...],
        batch_id: int,
        query_indices: Tuple[int, ...],
        embeddings: np.ndarray,
        coarse_scores: np.ndarray,
        coarse_ids: np.ndarray,
        pause_after_bootstrap: bool,
    ) -> NativeProgressiveBatchSession:
        scanner = self._native_scanner
        if scanner is None:
            raise RuntimeError("native continuous scanner was not initialized")
        fill_score = -np.inf if self.higher_is_better else np.inf
        native_id = None
        try:
            # One 2-D native handle owns the whole Python batch. This keeps one
            # shared task queue/controller and avoids multiplying native
            # worker pools by the number of queries.
            native_id = int(
                scanner.start(
                    np.ascontiguousarray(embeddings),
                    coarse_ids=np.ascontiguousarray(coarse_ids),
                    coarse_scores=np.ascontiguousarray(coarse_scores),
                    pause_after_bootstrap=bool(pause_after_bootstrap),
                )
            )
        except BaseException:
            if native_id is not None:
                try:
                    scanner.cancel(native_id)
                finally:
                    scanner.close(native_id)
            raise
        states = [
            _NativeQueryState(
                query_index=query_index,
                local_index=local_index,
                scores=np.full(self.heap_topk, fill_score, dtype=np.float32),
                ids=np.full(self.heap_topk, -1, dtype=np.int64),
            )
            for local_index, query_index in enumerate(query_indices)
        ]
        session = NativeProgressiveBatchSession(
            batch_id=int(batch_id),
            queries=queries,
            query_indices=query_indices,
            embeddings=embeddings,
            native_id=native_id,
            states=states,
            paused=bool(pause_after_bootstrap),
        )
        with self._native_query_handles_lock:
            for state in states:
                if state.query_index in self._native_query_handles:
                    raise RuntimeError(
                        "native query index already has a live scanner handle: "
                        f"{state.query_index}"
                    )
                self._native_query_handles[state.query_index] = (
                    int(native_id),
                    int(state.local_index),
                )
        return session

    def prepare_batch_inputs(self, queries: Sequence[str]):
        """Encode and coarse-rank several released requests in one call.

        This is an execution-only batching boundary: every request keeps its
        original release timestamp, batch id, and native scan session.  It
        removes the batch-size-one E5 serialization that otherwise makes the
        retrieval coordinator fall behind Poisson arrivals.
        """

        if not queries:
            raise ValueError("queries must not be empty")
        query_tuple = tuple(str(query) for query in queries)
        embeddings = self.encoder.encode(
            list(query_tuple), batch_size=self.batch_size, is_query=True
        ).astype(np.float32, copy=False)
        embeddings = np.ascontiguousarray(embeddings)
        coarse_scores, coarse_ids = self.ivf_index.quantizer.search(
            embeddings, self.max_nprobe
        )
        return (
            query_tuple,
            embeddings,
            np.ascontiguousarray(coarse_scores, dtype=np.float32),
            np.ascontiguousarray(coarse_ids, dtype=np.int64),
        )

    def start_prepared_batch(
        self,
        queries: Sequence[str],
        batch_id: int,
        query_indices: Sequence[int],
        embeddings: np.ndarray,
        coarse_scores: np.ndarray,
        coarse_ids: np.ndarray,
        pause_after_bootstrap: bool = True,
    ) -> ProgressiveBatchSession:
        """Create one logical session from already encoded request rows."""

        queries = tuple(str(query) for query in queries)
        query_indices = tuple(int(index) for index in query_indices)
        if not queries or len(query_indices) != len(queries):
            raise ValueError("prepared queries and query_indices must match")
        rows = len(queries)
        if any(array.shape[0] != rows for array in (
            embeddings, coarse_scores, coarse_ids
        )):
            raise ValueError("prepared array row counts must match queries")
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        coarse_scores = np.ascontiguousarray(coarse_scores, dtype=np.float32)
        coarse_ids = np.ascontiguousarray(coarse_ids, dtype=np.int64)
        if bool(getattr(self, "use_native_continuous", False)):
            return self._start_native_batch(
                queries,
                int(batch_id),
                query_indices,
                embeddings,
                coarse_scores,
                coarse_ids,
                bool(pause_after_bootstrap),
            )
        fill_score = -np.inf if self.higher_is_better else np.inf
        states = [
            _ProgressiveQueryState(
                query_index=query_index,
                monitor=self._new_monitor(),
                scores=np.full(self.heap_topk, fill_score, dtype=np.float32),
                ids=np.full(self.heap_topk, -1, dtype=np.int64),
            )
            for query_index in query_indices
        ]
        return ProgressiveBatchSession(
            batch_id=int(batch_id),
            queries=queries,
            query_indices=query_indices,
            embeddings=embeddings,
            coarse_scores=coarse_scores,
            coarse_ids=coarse_ids,
            states=states,
        )

    def start_batch(
        self,
        queries: Sequence[str],
        batch_id: int,
        query_indices: Optional[Sequence[int]] = None,
        pause_after_bootstrap: bool = True,
    ) -> ProgressiveBatchSession:
        if not queries:
            raise ValueError("queries must not be empty")
        queries = tuple(str(query) for query in queries)
        if query_indices is None:
            query_indices = tuple(range(len(queries)))
        else:
            query_indices = tuple(int(index) for index in query_indices)
        if len(query_indices) != len(queries):
            raise ValueError("query_indices and queries must have the same length")

        embeddings = self.encoder.encode(
            list(queries), batch_size=self.batch_size, is_query=True
        ).astype(np.float32, copy=False)
        embeddings = np.ascontiguousarray(embeddings)
        coarse_scores, coarse_ids = self.ivf_index.quantizer.search(
            embeddings, self.max_nprobe
        )

        if bool(getattr(self, "use_native_continuous", False)):
            return self._start_native_batch(
                queries,
                int(batch_id),
                query_indices,
                embeddings,
                np.ascontiguousarray(coarse_scores, dtype=np.float32),
                np.ascontiguousarray(coarse_ids, dtype=np.int64),
                bool(pause_after_bootstrap),
            )

        fill_score = -np.inf if self.higher_is_better else np.inf
        states = [
            _ProgressiveQueryState(
                query_index=query_index,
                monitor=self._new_monitor(),
                scores=np.full(self.heap_topk, fill_score, dtype=np.float32),
                ids=np.full(self.heap_topk, -1, dtype=np.int64),
            )
            for query_index in query_indices
        ]
        return ProgressiveBatchSession(
            batch_id=int(batch_id),
            queries=queries,
            query_indices=query_indices,
            embeddings=embeddings,
            coarse_scores=coarse_scores,
            coarse_ids=coarse_ids,
            states=states,
        )

    def cancel_queries(self, session, query_indices: Iterable[int]) -> None:
        """Terminally cancel answered queries in any FAST/paused/draining state."""

        cancelled = set(int(index) for index in query_indices)
        if isinstance(session, NativeProgressiveBatchSession):
            scanner = self._native_scanner
            for state in session.states:
                if state.query_index in cancelled and not state.finished:
                    # Publish the logical cancellation first so another worker
                    # iteration cannot resume this query. Native close later
                    # accounts for any bounded task already in flight.
                    state.finished = True
                    state.stop_reason = "answered"
                    # If cancellation races with collection of the batch-wide
                    # bootstrap barrier, do not later expose a stale snapshot
                    # for the answered query.
                    session.bootstrap_by_local.pop(state.local_index, None)
                    scanner.cancel(session.native_id, state.local_index)
            return
        for state in session.states:
            if state.query_index in cancelled and not state.finished:
                state.finished = True
                state.phase = FINISHED
                state.pause_reason = None
                state.final_top5_ready = False
                state.stop_reason = "answered"

    def resume_queries(
        self, session, query_indices: Iterable[int]
    ) -> Tuple[int, ...]:
        """Compatibility helper for explicitly constructed legacy paused states.

        Natural WRD-stable queries are now ``FINISHED`` and therefore cannot be
        resumed.  The cursor and accumulated heap remain preserved for callers
        that still construct a legacy ``PAUSED_FAST`` state in isolation.
        """

        if isinstance(session, NativeProgressiveBatchSession):
            # Native sessions are resumed exactly once by the second
            # advance_batch() call, never as a response to model ``No``.
            return ()
        requested = set(int(index) for index in query_indices)
        known = {state.query_index for state in session.states}
        unknown = requested - known
        if unknown:
            raise ValueError(f"query indices are not part of this session: {sorted(unknown)}")

        resumed = []
        for state in session.states:
            if state.query_index in requested and state.phase == PAUSED_FAST:
                state.phase = DRAINING
                state.pause_reason = None
                state.stop_reason = None
                resumed.append(state.query_index)
        return tuple(resumed)

    def pause_batch(self, session) -> bool:
        """Pause one native batch without discarding already completed work.

        Native ``pause`` is a blocking quiesce: tasks already in flight drain
        while logical heap progression remains frozen, and the method returns
        only once physical work is idle. ``close_batch`` later includes those
        completed speculative tasks in its physical-work accounting.
        """

        if not isinstance(session, NativeProgressiveBatchSession):
            return False
        if session.closed or session.terminal or session.paused:
            return False
        self._native_scanner.pause(session.native_id)
        session.paused = True
        return True

    def resume_batch(self, session) -> bool:
        """Resume all unfinished queries on the batch's single native handle."""

        if not isinstance(session, NativeProgressiveBatchSession):
            return False
        if session.closed or session.terminal or not session.paused:
            return False
        self._native_scanner.resume(session.native_id)
        session.resumed = True
        session.paused = False
        return True

    def set_generation_active(
        self, query_indices: Iterable[int], active: bool = True
    ) -> Tuple[int, ...]:
        """Publish per-request generation state to the native scan scheduler.

        The signal is advisory and never gates retrieval work.  It is consumed
        only by an explicitly enabled quantumized request-aligned scheduler;
        with the default unlimited continuation path this method is inert apart
        from recording the state in the native query object.
        """

        scanner = self._native_scanner
        setter = getattr(scanner, "set_generation_active", None)
        if not callable(setter):
            return ()
        requested = tuple(dict.fromkeys(int(index) for index in query_indices))
        with self._native_query_handles_lock:
            handles = [
                (index, self._native_query_handles.get(index))
                for index in requested
            ]
        updated = []
        for query_index, handle in handles:
            if handle is None:
                continue
            native_id, local_index = handle
            setter(native_id, local_index, bool(active))
            updated.append(query_index)
        return tuple(updated)

    @staticmethod
    def _native_event_int(event: Dict[str, Any], *keys: str, default: int = 0) -> int:
        for key in keys:
            value = event.get(key)
            if value is not None:
                return int(value)
        return int(default)

    def _native_physical_count(
        self, payload: Optional[Dict[str, Any]], logical_budget: int
    ) -> int:
        if not isinstance(payload, dict):
            return int(logical_budget)
        # ``computed_high_water`` is deliberately excluded: with out-of-order
        # query-list tasks it is a rank watermark, not a physical work count.
        actual = self._native_event_int(
            payload,
            "actual_scanned_lists",
            "completed_list_count",
            "actual_scanned_nprobe",
            default=logical_budget,
        )
        return max(int(logical_budget), min(self.max_nprobe, actual))

    def _poll_native_session(
        self, session: NativeProgressiveBatchSession, *, timeout_ms: int
    ) -> Optional[Dict[str, Any]]:
        event = self._native_scanner.poll(
            session.native_id,
            after_sequence=session.last_sequence,
            timeout_ms=int(timeout_ms),
            # Sequence, not budget, is the lossless cursor. In particular a
            # same-budget cancellation terminal must not be skipped.
            latest=False,
        )
        if event is None:
            return None
        event = dict(event)
        return event if bool(event.get("ready", True)) else None

    def _native_event_to_snapshot(
        self,
        session: NativeProgressiveBatchSession,
        state: _NativeQueryState,
        event: Dict[str, Any],
    ) -> Optional[ProgressiveQuerySnapshot]:
        sequence = self._native_event_int(event, "sequence", default=-1)
        if sequence <= session.last_sequence:
            raise RuntimeError(
                f"native sequence did not increase for batch {session.batch_id}: "
                f"{sequence} <= {session.last_sequence}"
            )
        local_index = self._native_event_int(event, "local_index", default=-1)
        if local_index != state.local_index:
            raise RuntimeError(
                f"native event local_index {local_index} was routed to "
                f"state {state.local_index}"
            )
        logical_budget = self._native_event_int(
            event, "merged_budget", "budget", default=state.scanned_budget
        )
        actual = self._native_physical_count(event, logical_budget)
        scheduled_high_water = self._native_event_int(
            event, "scheduled_high_water", default=logical_budget
        )
        computed_high_water = self._native_event_int(
            event, "computed_high_water", default=logical_budget
        )
        terminal = bool(event.get("terminal", False))
        reason = str(event.get("terminal_reason") or "").strip().casefold()
        if state.finished and state.stop_reason == "answered":
            # Cancellation can race with one or more already-published native
            # ranking events.  Consume their global sequence and physical-work
            # accounting, but never resurrect the query or expose stale heaps
            # to generation.
            state.last_sequence = sequence
            session.last_sequence = sequence
            state.actual_scanned_nprobe = max(
                state.actual_scanned_nprobe, actual
            )
            state.scheduled_high_water = max(
                state.scheduled_high_water, scheduled_high_water
            )
            state.computed_high_water = max(
                state.computed_high_water, computed_high_water
            )
            return None
        if terminal and reason in {"cancelled", "canceled", "answered"}:
            # Native cancellation may publish a higher-sequence terminal at
            # the same merged budget. It is an accounting event, not another
            # ranking version for the Pipeline.
            state.last_sequence = sequence
            session.last_sequence = sequence
            state.actual_scanned_nprobe = max(
                state.actual_scanned_nprobe, actual
            )
            state.scheduled_high_water = max(
                state.scheduled_high_water, scheduled_high_water
            )
            state.computed_high_water = max(
                state.computed_high_water, computed_high_water
            )
            state.finished = True
            state.stop_reason = "answered"
            return None
        if logical_budget <= state.scanned_budget:
            raise RuntimeError(
                f"native merged budget did not increase for query "
                f"{state.query_index}: {logical_budget} <= {state.scanned_budget}"
            )
        if terminal:
            valid_budget = 1 <= logical_budget <= self.max_nprobe
        else:
            valid_budget = self.bootstrap_nprobe <= logical_budget <= self.max_nprobe
        if not valid_budget:
            raise RuntimeError(
                f"native merged budget {logical_budget} is outside the configured range"
            )

        labels = np.asarray(event.get("labels", ()), dtype=np.int64).reshape(-1)
        distances = np.asarray(
            event.get("distances", ()), dtype=np.float32
        ).reshape(-1)
        if labels.size != distances.size or labels.size < self.heap_topk:
            raise RuntimeError(
                "native event must provide one labels/distances top-k row"
            )
        labels = labels[: self.heap_topk]
        distances = distances[: self.heap_topk]
        state.ids = labels.copy()
        state.scores = distances.copy()
        state.last_sequence = sequence
        session.last_sequence = sequence
        state.scanned_budget = logical_budget
        state.actual_scanned_nprobe = max(state.actual_scanned_nprobe, actual)
        state.scheduled_high_water = max(
            state.scheduled_high_water, scheduled_high_water
        )
        state.computed_high_water = max(
            state.computed_high_water, computed_high_water
        )

        if terminal:
            # Max budget wins if WRD becomes stable on the same final list.
            if logical_budget >= self.max_nprobe:
                stop_reason = "max_budget"
            elif reason in {"wrd_stable", "stable"}:
                stop_reason = "wrd_stable"
            elif reason in {"max_budget", "max_nprobe", "max"}:
                stop_reason = "max_budget"
            else:
                raise RuntimeError(
                    f"native terminal has unknown reason {reason!r}"
                )
            state.finished = True
            state.stop_reason = stop_reason
        else:
            if reason:
                raise RuntimeError(
                    "nonterminal native ranking event has a terminal reason"
                )
            stop_reason = None

        top1 = int(labels[0]) if int(labels[0]) >= 0 else None
        top1_changed = top1 is not None and top1 != state.last_top1
        top1_is_new = top1_changed and top1 not in state.seen_top1
        if top1_changed:
            state.last_top1 = top1
        if top1_is_new:
            state.seen_top1.add(top1)

        def optional_float(key: str) -> Optional[float]:
            value = event.get(key)
            return None if value is None else float(value)

        final_top5_ready = stop_reason == "max_budget"
        return ProgressiveQuerySnapshot(
            query_index=state.query_index,
            batch_id=session.batch_id,
            budget=logical_budget,
            document_ids=tuple(int(value) for value in labels),
            scores=tuple(float(value) for value in distances),
            delta=optional_float("wrd_delta"),
            average_delta=None,
            volatility=None,
            risk=optional_float("wrd_risk"),
            stable_streak=self._native_event_int(
                event, "wrd_stable_streak", default=0
            ),
            top1_changed=top1_changed,
            top1_is_new=top1_is_new,
            top1_document=None,
            finished=state.finished,
            stop_reason=stop_reason,
            phase=FINISHED if state.finished else FAST,
            scanned_budget=logical_budget,
            pause_reason=None,
            final_top5_ready=final_top5_ready,
            actual_scanned_nprobe=state.actual_scanned_nprobe,
            native_sequence=sequence,
            scheduled_high_water=state.scheduled_high_water,
            computed_high_water=state.computed_high_water,
            prefetch_advised_lists=self._native_event_int(
                event, "prefetch_advised_lists", default=0
            ),
            prefetch_advised_bytes=self._native_event_int(
                event, "prefetch_advised_bytes", default=0
            ),
            prefetch_advice_errors=self._native_event_int(
                event, "prefetch_advice_errors", default=0
            ),
            prefetch_high_water=self._native_event_int(
                event, "prefetch_high_water", default=0
            ),
            heap_updates=self._native_event_int(
                event, "heap_updates", default=0
            ),
            unchanged_heap_lists=self._native_event_int(
                event, "unchanged_heap_lists", default=0
            ),
        )

    def _attach_new_top1_documents(
        self, snapshots: Sequence[ProgressiveQuerySnapshot]
    ) -> Tuple[ProgressiveQuerySnapshot, ...]:
        new_top1_ids = list(
            dict.fromkeys(
                snapshot.document_ids[0]
                for snapshot in snapshots
                if snapshot.top1_is_new
                and snapshot.document_ids
                and snapshot.document_ids[0] >= 0
            )
        )
        if not new_top1_ids:
            return tuple(snapshots)
        documents = load_docs(self.corpus, new_top1_ids)
        document_by_id = dict(zip(new_top1_ids, documents))
        return tuple(
            replace(
                snapshot,
                top1_document=document_by_id.get(snapshot.document_ids[0]),
            )
            if snapshot.top1_is_new
            else snapshot
            for snapshot in snapshots
        )

    def _native_retrieval_event(
        self,
        session: NativeProgressiveBatchSession,
        snapshots: Sequence[ProgressiveQuerySnapshot],
    ) -> ProgressiveRetrievalEvent:
        attached = self._attach_new_top1_documents(snapshots)
        if not attached:
            raise ValueError("a native retrieval event requires a snapshot")
        return ProgressiveRetrievalEvent(
            batch_id=session.batch_id,
            budget=max(snapshot.budget for snapshot in attached),
            snapshots=attached,
            batch_complete=session.terminal,
            has_runnable=session.has_runnable,
            terminal=session.terminal,
        )

    def _try_advance_native_batch(
        self,
        session: NativeProgressiveBatchSession,
        *,
        timeout_ms: int = 0,
    ) -> Optional[ProgressiveRetrievalEvent]:
        """Poll at most one event from a native batch.

        Bootstrap is a batch-wide nprobe=4 barrier.  A single poll commonly
        contributes only one query to that barrier, so partial snapshots live
        on ``session.bootstrap_by_local`` until every non-cancelled query has
        arrived.  ``None`` means that this call has no complete ranking event
        to publish; it may still have consumed a cancellation/accounting event
        or one member of an incomplete bootstrap barrier.
        """

        timeout_ms = int(timeout_ms)
        if timeout_ms < 0:
            raise ValueError("timeout_ms must be non-negative")
        if session.closed:
            raise StopIteration("native progressive batch is closed")
        if not session.has_runnable:
            raise StopIteration("native progressive batch is terminal")
        if not session.bootstrap_emitted:
            # A cancellation can complete the barrier between polls.  Publish
            # already-buffered @4 snapshots immediately instead of waiting for
            # an event that a bootstrap-parked scanner will never produce.
            barrier_ready = all(
                state.finished
                or state.local_index in session.bootstrap_by_local
                for state in session.states
            )
            if barrier_ready:
                snapshots = tuple(
                    session.bootstrap_by_local[state.local_index]
                    for state in session.states
                    if state.local_index in session.bootstrap_by_local
                )
                session.bootstrap_by_local.clear()
                session.bootstrap_emitted = True
                if snapshots:
                    return self._native_retrieval_event(session, snapshots)
        # Once the bootstrap barrier has been exposed, an intentionally parked
        # session remains inert until resume_batch() (or blocking
        # advance_batch()) explicitly resumes it.  Bootstrap events themselves
        # must still be drainable while the native handle is parked at @4.
        if session.bootstrap_emitted and session.paused:
            return None

        event = self._poll_native_session(session, timeout_ms=timeout_ms)
        if event is None:
            return None
        local_index = self._native_event_int(event, "local_index", default=-1)
        if not 0 <= local_index < len(session.states):
            phase = "bootstrap " if not session.bootstrap_emitted else ""
            raise RuntimeError(
                f"native {phase}event has invalid local_index {local_index}"
            )

        state = session.states[local_index]
        snapshot = self._native_event_to_snapshot(session, state, event)
        snapshots: List[ProgressiveQuerySnapshot] = []

        if not session.bootstrap_emitted:
            if snapshot is not None:
                if local_index in session.bootstrap_by_local:
                    raise RuntimeError(
                        f"native query {local_index} emitted more than one "
                        "event before the batch bootstrap barrier"
                    )
                terminal_before_prefix = (
                    snapshot.finished
                    and snapshot.budget < self.bootstrap_nprobe
                )
                if (
                    snapshot.budget != self.bootstrap_nprobe
                    and not terminal_before_prefix
                ):
                    raise RuntimeError(
                        "the first native event for every query must be the "
                        "fast-prefix budget or an earlier WRD terminal"
                    )
                session.bootstrap_by_local[local_index] = snapshot
            else:
                # A cancellation event supersedes any bootstrap heap buffered
                # for the same query.
                session.bootstrap_by_local.pop(local_index, None)

            barrier_ready = all(
                state.finished
                or state.local_index in session.bootstrap_by_local
                for state in session.states
            )
            if not barrier_ready:
                return None
            snapshots.extend(
                session.bootstrap_by_local[state.local_index]
                for state in session.states
                if state.local_index in session.bootstrap_by_local
            )
            session.bootstrap_by_local.clear()
            session.bootstrap_emitted = True
        elif snapshot is not None:
            snapshots.append(snapshot)

        if not snapshots:
            return None
        return self._native_retrieval_event(session, snapshots)

    def try_advance_batch(
        self, session, timeout_ms: int = 0
    ) -> Optional[ProgressiveRetrievalEvent]:
        """Attempt one non-blocking native advance.

        The native path polls at most one sequence event and returns ``None``
        when no publishable ranking event is ready.  The legacy FAISS path has
        no asynchronous scanner to poll, so it retains its normal one-step
        synchronous behavior.
        """

        if isinstance(session, NativeProgressiveBatchSession):
            return self._try_advance_native_batch(
                session, timeout_ms=timeout_ms
            )
        if int(timeout_ms) < 0:
            raise ValueError("timeout_ms must be non-negative")
        return self.advance_batch(session)

    def _advance_native_batch(
        self, session: NativeProgressiveBatchSession
    ) -> ProgressiveRetrievalEvent:
        if session.closed or not session.has_runnable:
            raise StopIteration("native progressive batch is terminal")
        if session.bootstrap_emitted and session.paused:
            # Preserve the historical blocking API: its second call resumes a
            # batch that was parked immediately after bootstrap.
            self.resume_batch(session)

        while session.has_runnable:
            event = self.try_advance_batch(
                session, timeout_ms=self.native_poll_timeout_ms
            )
            if event is not None:
                return event
        raise StopIteration("native progressive batch produced no ranking event")

    def close_batch(self, session) -> Dict[int, int]:
        """Join native tasks and return physical scan counts by global query."""

        if not isinstance(session, NativeProgressiveBatchSession):
            return {}
        if session.closed:
            return {
                state.query_index: state.actual_scanned_nprobe
                for state in session.states
            }
        scanner = self._native_scanner
        # Stop accepting new generation-priority updates before native close()
        # releases the handle.  Match the handle as well as the query index so
        # a future isolated reuse cannot remove a newer registration.
        with self._native_query_handles_lock:
            for state in session.states:
                handle = self._native_query_handles.get(state.query_index)
                if handle == (int(session.native_id), int(state.local_index)):
                    self._native_query_handles.pop(state.query_index, None)
        for state in session.states:
            if not state.finished:
                scanner.cancel(session.native_id, state.local_index)
                state.finished = True
                state.stop_reason = "answered"
        session.bootstrap_by_local.clear()
        # A 2-D session owns exactly one native handle and must be joined and
        # erased exactly once. Per-query physical work lives under
        # summary["queries"][local_index].
        raw_summary = scanner.close(session.native_id)
        summary = dict(raw_summary) if raw_summary is not None else {}
        session.scan_permit_diagnostics = {
            "global": dict(summary.get("scan_permit_global") or {}),
            "session": dict(summary.get("scan_permit_session") or {}),
            "scheduler": {
                "continuation_quantum": int(
                    summary.get("continuation_quantum") or 0
                ),
                "request_aligned_priority": bool(
                    summary.get("request_aligned_priority", False)
                ),
                "request_aligned_priority_burst": int(
                    summary.get("request_aligned_priority_burst") or 0
                ),
                "request_aligned_max_fifo_wait_ms": int(
                    summary.get("request_aligned_max_fifo_wait_ms") or 0
                ),
                "persistent_bootstrap_dispatches": int(
                    summary.get("persistent_bootstrap_dispatches") or 0
                ),
                "persistent_continuation_dispatches": int(
                    summary.get("persistent_continuation_dispatches") or 0
                ),
                "persistent_priority_dispatches": int(
                    summary.get("persistent_priority_dispatches") or 0
                ),
                "persistent_fairness_dispatches": int(
                    summary.get("persistent_fairness_dispatches") or 0
                ),
                "persistent_continuation_requeues": int(
                    summary.get("persistent_continuation_requeues") or 0
                ),
            },
        }
        raw_scan_intervals = summary.get("scan_active_intervals_ns") or ()
        scan_intervals: List[Tuple[int, int]] = []
        for raw_interval in raw_scan_intervals:
            if len(raw_interval) != 2:
                raise RuntimeError(
                    "native scan interval must contain start and stop"
                )
            start_ns, stop_ns = map(int, raw_interval)
            if start_ns < 0 or stop_ns <= start_ns:
                raise RuntimeError(
                    "native scan interval has invalid bounds: "
                    f"{start_ns}, {stop_ns}"
                )
            scan_intervals.append((start_ns, stop_ns))
        expected_scan_intervals = summary.get("scan_active_interval_count")
        if (
            expected_scan_intervals is not None
            and int(expected_scan_intervals) != len(scan_intervals)
        ):
            raise RuntimeError(
                "native scan interval count mismatch: "
                f"summary={expected_scan_intervals}, parsed={len(scan_intervals)}"
            )
        if scan_intervals and summary.get("scan_active_clock") != "unix_time_ns":
            raise RuntimeError(
                "native scan intervals use an unsupported clock domain"
            )
        session.scan_active_intervals_ns = scan_intervals
        rows = summary.get("queries", ())
        if isinstance(rows, dict):
            rows = rows.values()
        by_local: Dict[int, Dict[str, Any]] = {}
        for raw_row in rows or ():
            row = dict(raw_row)
            local_index = self._native_event_int(
                row, "local_index", default=-1
            )
            if not 0 <= local_index < len(session.states):
                raise RuntimeError(
                    f"native close summary has invalid local_index {local_index}"
                )
            if local_index in by_local:
                raise RuntimeError(
                    f"native close summary repeats local_index {local_index}"
                )
            by_local[local_index] = row

        for state in session.states:
            row = by_local.get(state.local_index)
            if row is None and len(session.states) == 1:
                row = summary
            logical_budget = self._native_event_int(
                row or {},
                "merged_budget",
                "budget",
                default=state.scanned_budget,
            )
            actual = self._native_physical_count(
                row,
                logical_budget,
            )
            state.actual_scanned_nprobe = max(
                state.actual_scanned_nprobe, actual
            )
            raw_query_intervals = (row or {}).get(
                "scan_active_intervals_ns"
            ) or ()
            query_intervals: List[Tuple[int, int]] = []
            for raw_interval in raw_query_intervals:
                if len(raw_interval) != 2:
                    raise RuntimeError(
                        "native per-query scan interval must contain start and stop"
                    )
                start_ns, stop_ns = map(int, raw_interval)
                if start_ns < 0 or stop_ns <= start_ns:
                    raise RuntimeError(
                        "native per-query scan interval has invalid bounds: "
                        f"{start_ns}, {stop_ns}"
                    )
                query_intervals.append((start_ns, stop_ns))
            expected_query_intervals = (row or {}).get(
                "scan_active_interval_count"
            )
            if (
                expected_query_intervals is not None
                and int(expected_query_intervals) != len(query_intervals)
            ):
                raise RuntimeError(
                    "native per-query scan interval count mismatch: "
                    f"summary={expected_query_intervals}, "
                    f"parsed={len(query_intervals)}"
                )
            state.scan_active_intervals_ns = query_intervals
            state.closed = True
        session.closed = True
        return {
            state.query_index: state.actual_scanned_nprobe
            for state in session.states
        }

    def _scan_new_lists(
        self, session: ProgressiveBatchSession, active_local_indices: Sequence[int], start: int, stop: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        local = np.asarray(active_local_indices, dtype=np.int64)
        embeddings = np.ascontiguousarray(session.embeddings[local])
        list_ids = np.ascontiguousarray(session.coarse_ids[local, start:stop], dtype=np.int64)
        list_scores = np.ascontiguousarray(session.coarse_scores[local, start:stop], dtype=np.float32)
        chunk_size = stop - start
        if chunk_size < 1:
            raise ValueError("empty IVF budget increment")

        # FAISS 1.8's Python wrapper validates against the mutable nprobe field.
        # A single retrieval actor plus this lock makes that mutation safe without
        # duplicating the 81 GB index.
        with self._search_lock:
            # FAISS/OpenMP thread counts are thread-local in the server build,
            # so set it in the retrieval actor rather than only at init time.
            faiss.omp_set_num_threads(self.omp_threads)
            self.ivf_index.nprobe = chunk_size
            scores, ids = self.ivf_index.search_preassigned(
                embeddings,
                self.heap_topk,
                list_ids,
                list_scores,
            )
        return scores, ids

    def _next_scan_range(self, state: _ProgressiveQueryState) -> Tuple[int, int]:
        if not state.runnable:
            raise RuntimeError("query is not runnable")
        start = int(state.scanned_budget)
        if start == 0:
            stop = self.bootstrap_nprobe
        elif state.phase == FAST:
            stop = min(start + 1, self.max_nprobe)
        elif state.phase == DRAINING:
            stop = min(start + 1, self.max_nprobe)
        else:  # defensive: ``runnable`` already restricts the phase
            raise RuntimeError(f"unknown runnable query phase: {state.phase}")
        if stop <= start:
            raise RuntimeError(
                f"query {state.query_index} has no forward scan range in phase {state.phase}"
            )
        return start, stop

    def advance_batch(self, session) -> ProgressiveRetrievalEvent:
        """Advance every currently runnable query by one cursor step.

        Queries can have different cursors and phases.  They are grouped by the
        exact ``[start, stop)`` slice so each FAISS call remains a dense batch
        while no query rescans an IVF list.
        """

        if isinstance(session, NativeProgressiveBatchSession):
            return self._advance_native_batch(session)
        if not session.has_runnable:
            if session.terminal:
                raise StopIteration("progressive batch is terminal")
            raise StopIteration("progressive batch is paused; call resume_queries()")

        range_groups = {}
        for local_index, state in enumerate(session.states):
            if state.runnable:
                scan_range = self._next_scan_range(state)
                range_groups.setdefault(scan_range, []).append(local_index)

        snapshots_by_local = {}
        for (start, stop), local_indices in range_groups.items():
            new_scores, new_ids = self._scan_new_lists(
                session, local_indices, start, stop
            )
            old_scores = np.stack(
                [session.states[index].scores for index in local_indices]
            )
            old_ids = np.stack(
                [session.states[index].ids for index in local_indices]
            )
            merged_scores, merged_ids = merge_topk(
                old_scores,
                old_ids,
                new_scores,
                new_ids,
                self.heap_topk,
                self.higher_is_better,
            )

            for row, local_index in enumerate(local_indices):
                state = session.states[local_index]
                state.scores = merged_scores[row]
                state.ids = merged_ids[row]
                state.scanned_budget = stop
                observation = state.monitor.observe(
                    state.ids[: self.stability_topk], stop
                )

                top1 = int(state.ids[0]) if int(state.ids[0]) >= 0 else None
                top1_changed = top1 is not None and top1 != state.last_top1
                top1_is_new = top1_changed and top1 not in state.seen_top1
                if top1_changed:
                    state.last_top1 = top1
                if top1_is_new:
                    state.seen_top1.add(top1)

                if stop >= self.max_nprobe:
                    state.phase = FINISHED
                    state.finished = True
                    state.pause_reason = None
                    state.final_top5_ready = True
                    state.stop_reason = "max_budget"
                elif state.phase == FAST and observation.should_stop:
                    # Stability is a query-level terminal condition.  Freezing
                    # the heap here is essential: downstream generation cannot
                    # turn a model ``No`` into an implicit scan to nprobe=128.
                    state.phase = FINISHED
                    state.finished = True
                    state.pause_reason = None
                    state.final_top5_ready = False
                    state.stop_reason = "wrd_stable"

                snapshots_by_local[local_index] = ProgressiveQuerySnapshot(
                    query_index=state.query_index,
                    batch_id=session.batch_id,
                    budget=stop,
                    document_ids=tuple(int(value) for value in state.ids),
                    scores=tuple(float(value) for value in state.scores),
                    delta=observation.delta,
                    average_delta=observation.average_delta,
                    volatility=observation.volatility,
                    risk=observation.risk,
                    stable_streak=observation.stable_streak,
                    top1_changed=top1_changed,
                    top1_is_new=top1_is_new,
                    top1_document=None,
                    finished=state.finished,
                    stop_reason=state.stop_reason,
                    phase=state.phase,
                    scanned_budget=state.scanned_budget,
                    pause_reason=state.pause_reason,
                    final_top5_ready=state.final_top5_ready,
                )

        session.stage_index += 1
        snapshots = [
            snapshots_by_local[index] for index in sorted(snapshots_by_local)
        ]
        new_top1_ids = list(
            dict.fromkeys(
                snapshot.document_ids[0]
                for snapshot in snapshots
                if snapshot.top1_is_new
                and snapshot.document_ids
                and snapshot.document_ids[0] >= 0
            )
        )
        if new_top1_ids:
            documents = load_docs(self.corpus, new_top1_ids)
            document_by_id = {
                doc_id: document for doc_id, document in zip(new_top1_ids, documents)
            }
            snapshots = [
                replace(
                    snapshot,
                    top1_document=document_by_id.get(snapshot.document_ids[0]),
                )
                if snapshot.top1_is_new
                else snapshot
                for snapshot in snapshots
            ]

        has_runnable = session.has_runnable
        return ProgressiveRetrievalEvent(
            batch_id=session.batch_id,
            budget=max(snapshot.budget for snapshot in snapshots),
            snapshots=tuple(snapshots),
            # A batch is terminal once every query has stabilized, reached the
            # maximum budget, or was cancelled after an accepted early answer.
            batch_complete=session.terminal,
            has_runnable=has_runnable,
            terminal=session.terminal,
        )

    def materialize_documents(self, document_ids: Sequence[int]) -> List[dict]:
        valid_ids = [int(doc_id) for doc_id in document_ids if int(doc_id) >= 0]
        return load_docs(self.corpus, valid_ids) if valid_ids else []


__all__ = [
    "DRAINING",
    "FAST",
    "FINISHED",
    "NativeProgressiveBatchSession",
    "PAUSED_FAST",
    "ProgressiveBatchSession",
    "ProgressiveIVFRetriever",
    "ProgressiveQuerySnapshot",
    "ProgressiveRetrievalEvent",
    "StabilityObservation",
    "WeightedRankStabilityMonitor",
    "ewma_recent",
    "merge_topk",
    "power_rank_weights",
    "weighted_rank_delta",
]
