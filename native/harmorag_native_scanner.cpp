#include <faiss/IVFlib.h>
#include <faiss/Index.h>
#include <faiss/IndexIVF.h>
#include <faiss/index_io.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/invlists/OnDiskInvertedLists.h>
#include <faiss/utils/Heap.h>
#include <omp.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

#include "harmorag_scan_permit_limiter.h"

namespace py = pybind11;
using Clock = std::chrono::steady_clock;
using WallClock = std::chrono::system_clock;

namespace {

constexpr const char* kScheduler =
        "scanner_wide_full_continuous_batching";
#ifndef HARMORAG_PUBLISH_EVERY_BUDGET
#define HARMORAG_PUBLISH_EVERY_BUDGET 1
#endif

struct WrdState {
    std::vector<faiss::idx_t> previous_ids;
    // Reused open-addressing storage for exact weighted-rank comparison.
    // The historical implementation constructed two unordered_maps after
    // every heap-changing list, which made WRD bookkeeping a material part of
    // service time.  A query has only wrd_topk unique, non-negative labels, so
    // a power-of-two table at <= 0.5 load preserves the same calculation
    // without per-observation node allocation.
    std::vector<faiss::idx_t> rank_hash_keys;
    std::vector<int> rank_hash_positions;
    std::deque<double> deltas;
    std::vector<double> recent_scratch;
    std::vector<double> volatility_scratch;
    int stable_streak = 0;
    double last_delta = std::numeric_limits<double>::quiet_NaN();
    double last_risk = std::numeric_limits<double>::quiet_NaN();
    bool heap_changed_since_observation = false;
};

struct Event {
    uint64_t sequence = 0;
    int local_index = 0;
    int budget = 0;
    std::string phase;
    bool terminal = false;
    bool parked = false;
    std::string terminal_reason;
    double wrd_delta = std::numeric_limits<double>::quiet_NaN();
    double wrd_risk = std::numeric_limits<double>::quiet_NaN();
    int wrd_stable_streak = 0;
    long long actual_scanned_lists = 0;
    long long actual_scanned_documents = 0;
    long long prefetch_advised_lists = 0;
    long long prefetch_advised_bytes = 0;
    long long prefetch_advice_errors = 0;
    int prefetch_high_water = 0;
    long long heap_updates = 0;
    long long unchanged_heap_lists = 0;
    int scheduled_high_water = 0;
    int computed_high_water = 0;
    double elapsed_ms = 0.0;
    int64_t terminal_steady_ns = 0;
    std::vector<float> distances;
    std::vector<faiss::idx_t> labels;
};

struct QueryState {
    std::vector<float> heap_distances;
    std::vector<faiss::idx_t> heap_labels;
    std::vector<float> sorted_distances;
    std::vector<faiss::idx_t> sorted_labels;
    bool sorted_valid = false;
    std::unique_ptr<faiss::InvertedListScanner> scanner;
    WrdState wrd;
    std::atomic<bool> external_cancel{false};
    std::atomic<bool> paused{true};
    // The worker publishes this before a list scan and double-checks paused;
    // pause() can therefore establish a quiescence barrier without taking a
    // global mutex on the hot per-list path.
    std::atomic<bool> active_scan{false};
    // Published by the Python generation scheduler.  It is advisory: scan
    // work is never paused solely because generation is inactive, but a
    // quantumized persistent queue may prefer requests that currently have
    // useful generation work in flight.
    std::atomic<bool> generation_active{false};
    std::atomic<long long> continuation_quanta{0};
    std::atomic<long long> priority_dispatches{0};
    std::atomic<long long> actual_scanned_lists{0};
    std::atomic<long long> actual_scanned_documents{0};
    std::atomic<long long> prefetch_advised_lists{0};
    std::atomic<long long> prefetch_advised_bytes{0};
    std::atomic<long long> prefetch_advice_errors{0};
    std::atomic<int> prefetch_high_water{0};
    std::atomic<long long> heap_updates{0};
    std::atomic<long long> unchanged_heap_lists{0};
    std::atomic<int> scheduled_high_water{0};
    std::atomic<int> computed_high_water{0};
    std::atomic<int> budget{0};
    std::atomic<bool> finished{false};
    std::string terminal_reason;
    // Wall-clock intervals for actual IVF list scanning only.  These are
    // intentionally narrower than a native Session lifetime: permit waits,
    // pause/gate waits, polling, and cancellation latency are excluded.  The
    // wall-clock domain matches Python's time.time_ns() generation trace so
    // retrieval/generation active-compute overlap can be computed directly.
    std::vector<std::pair<int64_t, int64_t>> scan_active_intervals_ns;
};

struct Session {
    int id = 0;
    int n = 0;
    int d = 0;
    std::vector<float> embeddings;
    std::vector<float> coarse_scores;
    std::vector<faiss::idx_t> coarse_ids;
    bool coarse_supplied = false;
    bool pause_after_bootstrap = true;
    bool persistent_dispatch = false;
    std::atomic<int> remaining_bootstrap_queries{0};
    std::atomic<int> remaining_queries{0};
    Clock::time_point started = Clock::now();

    std::vector<std::shared_ptr<QueryState>> queries;

    // One counter set per session survives removal from the limiter's live
    // demand map, so close()/wait() can report its complete concurrency cost.
    harmorag::ScanPermitCounters scan_permits;

    std::mutex task_mutex;
    std::condition_variable task_cv;
    std::thread controller;

    std::atomic<bool> cancel_all{false};
    std::mutex error_mutex;
    std::mutex event_mutex;
    std::condition_variable event_cv;
    std::condition_variable done_cv;
    std::vector<std::shared_ptr<Event>> events;
    uint64_t next_sequence = 0;
    bool done = false;
    std::string error;
};

bool higher_is_better(faiss::MetricType metric) {
    return metric == faiss::METRIC_INNER_PRODUCT;
}

void update_max(std::atomic<int>& value, int candidate) {
    int current = value.load(std::memory_order_relaxed);
    while (current < candidate &&
           !value.compare_exchange_weak(current, candidate, std::memory_order_relaxed)) {
    }
}

std::vector<double> rank_weights(int k, double alpha) {
    std::vector<double> weights(static_cast<size_t>(k));
    double total = 0.0;
    for (int i = 0; i < k; ++i) {
        weights[static_cast<size_t>(i)] = std::pow(static_cast<double>(i + 1), -alpha);
        total += weights[static_cast<size_t>(i)];
    }
    for (double& value : weights) {
        value /= total;
    }
    return weights;
}

double weighted_rank_delta(
        WrdState& state,
        const std::vector<faiss::idx_t>& previous,
        const std::vector<faiss::idx_t>& current,
        const std::vector<double>& weights) {
    size_t table_size = 1;
    while (table_size < weights.size() * 2) table_size <<= 1;
    table_size = std::max<size_t>(table_size, 2);
    if (state.rank_hash_keys.size() != table_size) {
        state.rank_hash_keys.resize(table_size);
        state.rank_hash_positions.resize(table_size);
    }
    std::fill(
            state.rank_hash_positions.begin(),
            state.rank_hash_positions.end(),
            -1);
    const size_t mask = table_size - 1;
    const auto hash_slot = [mask](faiss::idx_t label) {
        uint64_t value = static_cast<uint64_t>(label);
        // SplitMix64 finalizer: deterministic and inexpensive for sequential
        // vector identifiers.
        value ^= value >> 30;
        value *= UINT64_C(0xbf58476d1ce4e5b9);
        value ^= value >> 27;
        value *= UINT64_C(0x94d049bb133111eb);
        value ^= value >> 31;
        return static_cast<size_t>(value) & mask;
    };

    double unmatched_previous = 0.0;
    const size_t previous_size = std::min(weights.size(), previous.size());
    for (size_t rank = 0; rank < previous_size; ++rank) {
        const faiss::idx_t label = previous[rank];
        if (label < 0) continue;
        unmatched_previous += weights[rank];
        size_t slot = hash_slot(label);
        while (state.rank_hash_positions[slot] >= 0) {
            slot = (slot + 1) & mask;
        }
        state.rank_hash_keys[slot] = label;
        state.rank_hash_positions[slot] = static_cast<int>(rank);
    }

    double new_current = 0.0;
    double moved = 0.0;
    const size_t current_size = std::min(weights.size(), current.size());
    for (size_t rank = 0; rank < current_size; ++rank) {
        const faiss::idx_t label = current[rank];
        if (label < 0) continue;
        size_t slot = hash_slot(label);
        int previous_rank = -1;
        while (state.rank_hash_positions[slot] >= 0) {
            if (state.rank_hash_keys[slot] == label) {
                previous_rank = state.rank_hash_positions[slot];
                break;
            }
            slot = (slot + 1) & mask;
        }
        if (previous_rank < 0) {
            new_current += weights[rank];
            continue;
        }
        unmatched_previous -= weights[static_cast<size_t>(previous_rank)];
        if (previous_rank != static_cast<int>(rank)) {
            moved += std::abs(
                    weights[static_cast<size_t>(previous_rank)] -
                    weights[rank]);
        }
    }

    // Equivalent to 0.5 * (left + right) in the two-map implementation:
    // each moved item contributes the same absolute rank-weight difference to
    // both sides; unmatched old/new items contribute to one side each.
    const double delta =
            moved + 0.5 * (unmatched_previous + new_current);
    return std::min(1.0, std::max(0.0, delta));
}

double ewma(const std::vector<double>& values, double beta) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        const double weight = std::pow(beta, static_cast<double>(values.size() - 1 - i));
        numerator += values[i] * weight;
        denominator += weight;
    }
    return denominator == 0.0 ? 0.0 : numerator / denominator;
}

class ContinuousIVFScanner {
public:
    ContinuousIVFScanner(
            const std::string& index_path,
            int worker_threads = 24,
            int initial_nprobe = 28,
            int max_nprobe = 128,
            int heap_topk = 100,
            int lookahead = 8,
            bool enable_wrd = true,
            double wrd_alpha = 1.0,
            double wrd_threshold = 0.0025,
            int wrd_window = 3,
            double wrd_beta = 0.4,
            double wrd_mu = 0.5,
            int wrd_patience = 16,
            int global_scan_permits = 0,
            int wrd_topk = 0,
            bool use_mmap = true,
            bool use_scan_permits = true,
            int wrd_check_interval = 1,
            int continuation_quantum = 0,
            bool request_aligned_priority = false,
            int request_aligned_priority_burst = 0,
            int request_aligned_max_fifo_wait_ms = 0)
            : index_path_(index_path),
              worker_threads_(worker_threads),
              initial_nprobe_(initial_nprobe),
              max_nprobe_(max_nprobe),
              heap_topk_(heap_topk),
              lookahead_(lookahead),
              enable_wrd_(enable_wrd),
              wrd_alpha_(wrd_alpha),
              wrd_threshold_(wrd_threshold),
              wrd_window_(wrd_window),
              wrd_beta_(wrd_beta),
              wrd_mu_(wrd_mu),
              wrd_patience_(wrd_patience),
              wrd_check_interval_(wrd_check_interval),
              wrd_topk_(wrd_topk == 0 ? heap_topk : wrd_topk),
              use_mmap_(use_mmap),
              use_scan_permits_(use_scan_permits),
              continuation_quantum_(continuation_quantum),
              request_aligned_priority_(request_aligned_priority),
              request_aligned_priority_burst_(request_aligned_priority_burst),
              request_aligned_max_fifo_wait_ms_(
                      request_aligned_max_fifo_wait_ms),
              global_scan_permits_(
                      global_scan_permits == 0
                              ? worker_threads
                              : global_scan_permits),
              scan_permits_(global_scan_permits_),
              weights_(rank_weights(wrd_topk_, wrd_alpha)) {
        if (worker_threads < 1 || initial_nprobe < 1 || max_nprobe < initial_nprobe ||
            heap_topk < 1 || lookahead < 1 || wrd_window < 2 ||
            wrd_patience < 1 || wrd_check_interval < 1 ||
            (wrd_topk != 0 && (wrd_topk < 1 || wrd_topk > heap_topk)) ||
            wrd_alpha <= 0.0 || wrd_threshold < 0.0 || wrd_beta < 0.0 ||
            wrd_beta >= 1.0 || wrd_mu < 0.0 || global_scan_permits < 0 ||
            continuation_quantum < 0 || request_aligned_priority_burst < 0 ||
            request_aligned_max_fifo_wait_ms < 0) {
            throw std::invalid_argument("invalid native scanner configuration");
        }
        const int flags = use_mmap_
                ? faiss::IO_FLAG_MMAP | faiss::IO_FLAG_READ_ONLY
                : 0;
        index_.reset(faiss::read_index(index_path.c_str(), flags));
        ivf_ = faiss::ivflib::extract_index_ivf(index_.get());
        if (!ivf_) throw std::runtime_error("index does not contain an IVF index");
        if (ivf_->code_size != static_cast<size_t>(ivf_->d) * sizeof(float)) {
            throw std::runtime_error("native scanner currently requires IVFFlat float32 codes");
        }
        if (max_nprobe_ > static_cast<int>(ivf_->nlist)) {
            throw std::invalid_argument("max_nprobe exceeds index nlist");
        }
        high_ = higher_is_better(ivf_->metric_type);
        ondisk_invlists_ = dynamic_cast<const faiss::OnDiskInvertedLists*>(
                ivf_->invlists);
        if (ondisk_invlists_) {
            prefetch_fd_ = ::open(index_path_.c_str(), O_RDONLY);
        }
        persistent_workers_.reserve(static_cast<size_t>(worker_threads_));
        for (int worker = 0; worker < worker_threads_; ++worker) {
            persistent_workers_.emplace_back(
                    [this]() { persistent_worker_loop(); });
        }
    }

    ~ContinuousIVFScanner() {
        shutdown_all();
        if (prefetch_fd_ >= 0) ::close(prefetch_fd_);
    }

    int start(
            const py::array& embedding_object,
            const py::object& coarse_ids_object = py::none(),
            const py::object& coarse_scores_object = py::none(),
            bool pause_after_bootstrap = true) {
        const Clock::time_point start_called = Clock::now();
        auto embeddings = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(
                embedding_object);
        if (!embeddings) throw std::invalid_argument("embeddings must be numeric C-order array");
        const auto info = embeddings.request();
        int n = 0;
        int d = 0;
        if (info.ndim == 1) {
            n = 1;
            d = static_cast<int>(info.shape[0]);
        } else if (info.ndim == 2) {
            n = static_cast<int>(info.shape[0]);
            d = static_cast<int>(info.shape[1]);
        } else {
            throw std::invalid_argument("embeddings must have shape [D] or [N,D]");
        }
        if (n < 1 || d != ivf_->d) throw std::invalid_argument("embedding shape mismatch");

        auto session = std::make_shared<Session>();
        session->id = next_session_id_.fetch_add(1);
        session->n = n;
        session->d = d;
        session->pause_after_bootstrap = pause_after_bootstrap;
        // Per-query terminal events are measured from API entry, so native
        // validation, embedding ownership, state allocation, coarse search,
        // list scanning, WRD, and result materialization share one boundary.
        session->started = start_called;
        const float* embedding_data = static_cast<const float*>(info.ptr);
        session->embeddings.assign(embedding_data, embedding_data + static_cast<size_t>(n) * d);

        const bool has_ids = !coarse_ids_object.is_none();
        const bool has_scores = !coarse_scores_object.is_none();
        if (has_ids != has_scores) {
            throw std::invalid_argument("coarse_ids and coarse_scores must be supplied together");
        }
        if (has_ids) {
            auto ids = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(
                    coarse_ids_object);
            auto scores = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(
                    coarse_scores_object);
            if (!ids || !scores) throw std::invalid_argument("invalid coarse arrays");
            const auto ids_info = ids.request();
            const auto scores_info = scores.request();
            if (ids_info.ndim != 2 || scores_info.ndim != 2 ||
                ids_info.shape[0] != n || scores_info.shape[0] != n ||
                ids_info.shape[1] < max_nprobe_ || scores_info.shape[1] < max_nprobe_) {
                throw std::invalid_argument("coarse arrays must have shape [N, >=max_nprobe]");
            }
            session->coarse_ids.resize(static_cast<size_t>(n) * max_nprobe_);
            session->coarse_scores.resize(static_cast<size_t>(n) * max_nprobe_);
            const auto* id_data = static_cast<const int64_t*>(ids_info.ptr);
            const auto* score_data = static_cast<const float*>(scores_info.ptr);
            const size_t id_stride = static_cast<size_t>(ids_info.shape[1]);
            const size_t score_stride = static_cast<size_t>(scores_info.shape[1]);
            for (int qi = 0; qi < n; ++qi) {
                std::copy_n(id_data + static_cast<size_t>(qi) * id_stride,
                            max_nprobe_,
                            session->coarse_ids.begin() + static_cast<size_t>(qi) * max_nprobe_);
                std::copy_n(score_data + static_cast<size_t>(qi) * score_stride,
                            max_nprobe_,
                            session->coarse_scores.begin() + static_cast<size_t>(qi) * max_nprobe_);
            }
            session->coarse_supplied = true;
        }

        session->queries.reserve(static_cast<size_t>(n));
        session->events.reserve(
                static_cast<size_t>(n) *
                static_cast<size_t>(max_nprobe_ - initial_nprobe_ + 2));
        for (int qi = 0; qi < n; ++qi) {
            auto state = std::make_shared<QueryState>();
            state->heap_distances.resize(static_cast<size_t>(heap_topk_));
            state->heap_labels.resize(static_cast<size_t>(heap_topk_));
            state->sorted_distances.resize(static_cast<size_t>(heap_topk_));
            state->sorted_labels.resize(static_cast<size_t>(heap_topk_));
            state->wrd.recent_scratch.reserve(static_cast<size_t>(wrd_window_));
            state->wrd.volatility_scratch.reserve(
                    static_cast<size_t>(wrd_window_ - 1));
            heapify(*state);
            state->paused.store(pause_after_bootstrap);
            state->scheduled_high_water.store(initial_nprobe_);
            session->queries.push_back(std::move(state));
        }
        if (!pause_after_bootstrap) {
            // Retrieval-only/open-loop sessions use the scanner-wide
            // persistent pool. Coarse assignment is complete before jobs
            // become visible, so every admitted query can transition
            // independently without creating a controller and an OpenMP team
            // for each 100 ms window.
            prepare_coarse(session);
            session->persistent_dispatch = true;
            session->remaining_bootstrap_queries.store(
                    n, std::memory_order_release);
            session->remaining_queries.store(n, std::memory_order_release);
        }
        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            sessions_.emplace(session->id, session);
        }
        if (session->persistent_dispatch) {
            {
                std::lock_guard<std::mutex> lock(persistent_mutex_);
                for (int qi = 0; qi < n; ++qi) {
                    persistent_jobs_.push_back(
                            {session, qi, false, Clock::now()});
                }
            }
            persistent_cv_.notify_all();
        } else {
            session->controller =
                    std::thread([this, session]() { run_session(session); });
        }
        return session->id;
    }

    py::dict poll(
            int session_id,
            long long after_sequence = -1,
            int timeout_ms = 0,
            bool latest = true) {
        auto session = get_session(session_id);
        std::shared_ptr<Event> selected;
        bool done = false;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(session->event_mutex);
            auto has_event = [&]() {
                const uint64_t next = after_sequence < 0
                        ? 0
                        : static_cast<uint64_t>(after_sequence) + 1;
                return next < session->events.size();
            };
            if (!has_event() && !session->done) {
                if (timeout_ms < 0) {
                    session->event_cv.wait(lock, [&]() { return has_event() || session->done; });
                } else if (timeout_ms > 0) {
                    session->event_cv.wait_for(
                            lock, std::chrono::milliseconds(timeout_ms),
                            [&]() { return has_event() || session->done; });
                }
            }
            if (latest) {
                for (auto it = session->events.rbegin(); it != session->events.rend(); ++it) {
                    if (static_cast<long long>((*it)->sequence) > after_sequence) {
                        selected = *it;
                        break;
                    }
                }
            } else {
                const uint64_t next = after_sequence < 0
                        ? 0
                        : static_cast<uint64_t>(after_sequence) + 1;
                if (next < session->events.size()) {
                    selected = session->events[static_cast<size_t>(next)];
                }
            }
            done = session->done;
        }
        if (!selected) {
            py::dict output;
            output["ready"] = false;
            output["done"] = done;
            output["session_id"] = session_id;
            return output;
        }
        return event_to_python(session_id, *selected, done);
    }

    void pause(int session_id, int local_index = -1) {
        auto session = get_session(session_id);
        std::vector<int> targets;
        if (local_index < -1 || local_index >= session->n) {
            throw std::out_of_range("local_index out of range");
        }
        {
            std::lock_guard<std::mutex> lock(session->task_mutex);
            const int begin = local_index < 0 ? 0 : local_index;
            const int end = local_index < 0 ? session->n : local_index + 1;
            for (int qi = begin; qi < end; ++qi) {
                QueryState& state = *session->queries[static_cast<size_t>(qi)];
                if (!state.finished.load(std::memory_order_acquire)) {
                    state.paused.store(true, std::memory_order_seq_cst);
                    targets.push_back(qi);
                }
            }
        }
        // A worker can be queued outside task_cv while waiting for a global
        // scan slot.  Wake it so the pause predicate removes its ticket before
        // pause() establishes the quiescence barrier.
        scan_permits_.notify_all();
        session->task_cv.notify_all();
        // pause() remains a blocking quiescence barrier for compatibility.
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(session->task_mutex);
            session->task_cv.wait(lock, [&]() {
                for (const int qi : targets) {
                    const QueryState& state =
                            *session->queries[static_cast<size_t>(qi)];
                    if (!state.finished.load(std::memory_order_acquire) &&
                        (state.active_scan.load(std::memory_order_seq_cst) ||
                         state.budget.load(std::memory_order_acquire) <
                                 initial_nprobe_)) {
                        return false;
                    }
                }
                return true;
            });
        }
    }

    void resume(int session_id, int local_index = -1) {
        auto session = get_session(session_id);
        {
            std::lock_guard<std::mutex> lock(session->task_mutex);
            for_each_query(session, local_index, [](QueryState& state) {
                if (!state.finished.load(std::memory_order_acquire)) {
                    state.paused.store(false, std::memory_order_release);
                }
            });
        }
        session->task_cv.notify_all();
    }

    void set_generation_active(
            int session_id,
            int local_index,
            bool active = true) {
        auto session = get_session(session_id);
        if (local_index < 0 || local_index >= session->n) {
            throw std::out_of_range("local_index out of range");
        }
        session->queries[static_cast<size_t>(local_index)]
                ->generation_active.store(active, std::memory_order_release);
        // A priority transition should be visible promptly even when all
        // persistent workers are sleeping between short continuation quanta.
        persistent_cv_.notify_all();
    }

    void cancel(int session_id, int local_index = -1) {
        auto session = get_session(session_id);
        {
            std::lock_guard<std::mutex> lock(session->task_mutex);
            if (local_index < 0) {
                session->cancel_all.store(true, std::memory_order_release);
            }
            for_each_query(session, local_index, [](QueryState& state) {
                state.external_cancel.store(true, std::memory_order_release);
            });
        }
        scan_permits_.notify_all();
        session->task_cv.notify_all();
        session->event_cv.notify_all();
    }

    py::dict wait(int session_id, int timeout_ms = -1) {
        auto session = get_session(session_id);
        bool done = false;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(session->event_mutex);
            if (timeout_ms < 0) {
                session->done_cv.wait(lock, [&]() { return session->done; });
            } else {
                session->done_cv.wait_for(
                        lock, std::chrono::milliseconds(timeout_ms),
                        [&]() { return session->done; });
            }
            done = session->done;
        }
        return summary_to_python(*session, done);
    }

    py::dict stats(int session_id) {
        auto session = get_session(session_id);
        bool done = false;
        {
            std::lock_guard<std::mutex> lock(session->event_mutex);
            done = session->done;
        }
        py::dict output;
        output["session_id"] = session_id;
        output["done"] = done;
        output["scheduler"] = kScheduler;
        output["continuation_quantum"] = continuation_quantum_;
        output["request_aligned_priority"] = request_aligned_priority_;
        output["request_aligned_priority_burst"] =
                request_aligned_priority_burst_;
        output["request_aligned_max_fifo_wait_ms"] =
                request_aligned_max_fifo_wait_ms_;
        output["persistent_bootstrap_dispatches"] =
                persistent_bootstrap_dispatches_.load();
        output["persistent_continuation_dispatches"] =
                persistent_continuation_dispatches_.load();
        output["persistent_priority_dispatches"] =
                persistent_priority_dispatches_.load();
        output["persistent_fairness_dispatches"] =
                persistent_fairness_dispatches_.load();
        output["persistent_continuation_requeues"] =
                persistent_continuation_requeues_.load();
        output["prefetch_backend"] = prefetch_backend();
        output["prefetch_lookahead"] = lookahead_;
        output["scan_permit_global"] = global_scan_permit_stats_to_python();
        output["scan_permit_session"] = session_scan_permit_stats_to_python(
                session->scan_permits);
        py::list queries;
        for (int qi = 0; qi < session->n; ++qi) {
            const QueryState& state = *session->queries[static_cast<size_t>(qi)];
            py::dict item;
            item["local_index"] = qi;
            item["actual_scanned_lists"] = state.actual_scanned_lists.load();
            item["completed_list_count"] = state.actual_scanned_lists.load();
            item["actual_scanned_documents"] =
                    state.actual_scanned_documents.load();
            item["prefetch_advised_lists"] = state.prefetch_advised_lists.load();
            item["prefetch_advised_bytes"] = state.prefetch_advised_bytes.load();
            item["prefetch_advice_errors"] = state.prefetch_advice_errors.load();
            item["prefetch_high_water"] = state.prefetch_high_water.load();
            item["heap_updates"] = state.heap_updates.load();
            item["unchanged_heap_lists"] = state.unchanged_heap_lists.load();
            item["scheduled_high_water"] = state.scheduled_high_water.load();
            item["computed_high_water"] = state.computed_high_water.load();
            item["paused"] = state.paused.load();
            item["generation_active"] = state.generation_active.load();
            item["continuation_quanta"] = state.continuation_quanta.load();
            item["priority_dispatches"] = state.priority_dispatches.load();
            item["finished"] = state.finished.load();
            queries.append(std::move(item));
        }
        output["queries"] = std::move(queries);
        return output;
    }

    py::dict close(int session_id) {
        auto session = get_session(session_id);
        cancel(session_id);
        {
            py::gil_scoped_release release;
            if (session->controller.joinable()) {
                session->controller.join();
            } else {
                std::unique_lock<std::mutex> lock(session->event_mutex);
                session->done_cv.wait(lock, [&]() { return session->done; });
            }
        }
        py::dict summary = summary_to_python(*session, true);
        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            sessions_.erase(session_id);
        }
        return summary;
    }

    int dimension() const { return ivf_->d; }
    int nlist() const { return static_cast<int>(ivf_->nlist); }
    std::string scheduler() const { return kScheduler; }
    int global_scan_permits() const { return global_scan_permits_; }
    int continuation_quantum() const { return continuation_quantum_; }
    bool request_aligned_priority() const {
        return request_aligned_priority_;
    }
    int request_aligned_priority_burst() const {
        return request_aligned_priority_burst_;
    }
    int request_aligned_max_fifo_wait_ms() const {
        return request_aligned_max_fifo_wait_ms_;
    }
    py::dict permit_stats() const {
        return global_scan_permit_stats_to_python();
    }
    std::string prefetch_backend() const {
        return ondisk_invlists_ && prefetch_fd_ >= 0
                ? "ondisk_posix_fadvise"
                : "disabled";
    }
    int prefetch_lookahead() const { return lookahead_; }

private:
    std::string index_path_;
    int worker_threads_;
    int initial_nprobe_;
    int max_nprobe_;
    int heap_topk_;
    int lookahead_;
    bool enable_wrd_;
    double wrd_alpha_;
    double wrd_threshold_;
    int wrd_window_;
    double wrd_beta_;
    double wrd_mu_;
    int wrd_patience_;
    int wrd_check_interval_;
    int wrd_topk_;
    bool use_mmap_;
    bool use_scan_permits_;
    int continuation_quantum_;
    bool request_aligned_priority_;
    int request_aligned_priority_burst_;
    int request_aligned_max_fifo_wait_ms_;
    int global_scan_permits_;
    harmorag::ScanPermitLimiter scan_permits_;
    std::vector<double> weights_;
    bool high_ = true;
    std::unique_ptr<faiss::Index> index_;
    faiss::IndexIVF* ivf_ = nullptr;
    const faiss::OnDiskInvertedLists* ondisk_invlists_ = nullptr;
    int prefetch_fd_ = -1;
    std::atomic<int> next_session_id_{1};
    std::mutex sessions_mutex_;
    std::unordered_map<int, std::shared_ptr<Session>> sessions_;
    struct PersistentJob {
        std::shared_ptr<Session> session;
        int local_index;
        bool continuation;
        Clock::time_point enqueued_at;
    };
    std::vector<std::thread> persistent_workers_;
    std::deque<PersistentJob> persistent_jobs_;
    std::mutex persistent_mutex_;
    std::condition_variable persistent_cv_;
    int consecutive_priority_dispatches_ = 0;
    bool persistent_stopping_ = false;
    std::atomic<long long> persistent_bootstrap_dispatches_{0};
    std::atomic<long long> persistent_continuation_dispatches_{0};
    std::atomic<long long> persistent_priority_dispatches_{0};
    std::atomic<long long> persistent_fairness_dispatches_{0};
    std::atomic<long long> persistent_continuation_requeues_{0};

    py::dict global_scan_permit_stats_to_python() const {
        const harmorag::ScanPermitSnapshot snapshot = scan_permits_.snapshot();
        py::dict output;
        output["capacity"] = snapshot.capacity;
        output["in_use"] = snapshot.in_use;
        output["peak_in_use"] = snapshot.peak_in_use;
        output["waiting"] = snapshot.waiting;
        output["peak_waiting"] = snapshot.peak_waiting;
        output["active_sessions"] = snapshot.active_sessions;
        output["acquires"] = snapshot.acquires;
        output["waits"] = snapshot.waits;
        output["aborted_waits"] = snapshot.aborted_waits;
        output["wait_ns"] = snapshot.wait_ns;
        output["hold_ns"] = snapshot.hold_ns;
        return output;
    }

    py::dict session_scan_permit_stats_to_python(
            const harmorag::ScanPermitCounters& counters) const {
        py::dict output;
        output["in_use"] = counters.in_use.load(std::memory_order_relaxed);
        output["peak_in_use"] = counters.peak_in_use.load(std::memory_order_relaxed);
        output["waiting"] = counters.waiting.load(std::memory_order_relaxed);
        output["peak_waiting"] = counters.peak_waiting.load(std::memory_order_relaxed);
        output["acquires"] = counters.acquires.load(std::memory_order_relaxed);
        output["waits"] = counters.waits.load(std::memory_order_relaxed);
        output["aborted_waits"] = counters.aborted_waits.load(
                std::memory_order_relaxed);
        output["wait_ns"] = counters.wait_ns.load(std::memory_order_relaxed);
        output["hold_ns"] = counters.hold_ns.load(std::memory_order_relaxed);
        return output;
    }

    std::shared_ptr<Session> get_session(int id) {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        const auto found = sessions_.find(id);
        if (found == sessions_.end()) throw std::out_of_range("unknown native session id");
        return found->second;
    }

    template <typename Function>
    void for_each_query(const std::shared_ptr<Session>& session, int local_index, Function fn) {
        if (local_index < -1 || local_index >= session->n) {
            throw std::out_of_range("local_index out of range");
        }
        if (local_index >= 0) {
            fn(*session->queries[static_cast<size_t>(local_index)]);
            return;
        }
        for (const auto& state : session->queries) fn(*state);
    }

    void heapify(QueryState& state) const {
        if (high_) {
            faiss::heap_heapify<faiss::CMin<float, faiss::idx_t>>(
                    heap_topk_, state.heap_distances.data(), state.heap_labels.data());
        } else {
            faiss::heap_heapify<faiss::CMax<float, faiss::idx_t>>(
                    heap_topk_, state.heap_distances.data(), state.heap_labels.data());
        }
    }

    void sorted_snapshot(
            const QueryState& state,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const {
        distances = state.heap_distances;
        labels = state.heap_labels;
        if (high_) {
            faiss::heap_reorder<faiss::CMin<float, faiss::idx_t>>(
                    distances.size(), distances.data(), labels.data());
        } else {
            faiss::heap_reorder<faiss::CMax<float, faiss::idx_t>>(
                    distances.size(), distances.data(), labels.data());
        }
    }

    size_t scan_list(
            faiss::InvertedListScanner* scanner,
            faiss::idx_t list_no,
            float coarse_score,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const {
        if (list_no < 0) return 0;
        const size_t list_size = ivf_->invlists->list_size(list_no);
        if (list_size == 0) return 0;
        scanner->set_list(list_no, coarse_score);
        faiss::InvertedLists::ScopedCodes codes(ivf_->invlists, list_no);
        faiss::InvertedLists::ScopedIds ids(ivf_->invlists, list_no);
        return scanner->scan_codes(
                list_size, codes.get(), ids.get(), distances.data(), labels.data(), heap_topk_);
    }

    bool observe_wrd_delta(QueryState& state, double delta) const {
        state.wrd.last_delta = delta;
        state.wrd.deltas.push_back(state.wrd.last_delta);
        while (static_cast<int>(state.wrd.deltas.size()) > wrd_window_) {
            state.wrd.deltas.pop_front();
        }
        if (static_cast<int>(state.wrd.deltas.size()) < wrd_window_) {
            state.wrd.stable_streak = 0;
            return false;
        }
        state.wrd.recent_scratch.assign(
                state.wrd.deltas.begin(), state.wrd.deltas.end());
        state.wrd.volatility_scratch.clear();
        for (size_t i = 1; i < state.wrd.recent_scratch.size(); ++i) {
            state.wrd.volatility_scratch.push_back(std::abs(
                    state.wrd.recent_scratch[i] -
                    state.wrd.recent_scratch[i - 1]));
        }
        state.wrd.last_risk = ewma(state.wrd.recent_scratch, wrd_beta_) +
                wrd_mu_ * ewma(state.wrd.volatility_scratch, wrd_beta_);
        state.wrd.stable_streak = state.wrd.last_risk <= wrd_threshold_
                ? state.wrd.stable_streak + 1
                : 0;
        return state.wrd.stable_streak >= wrd_patience_;
    }

    bool observe_wrd(
            QueryState& state,
            const std::vector<faiss::idx_t>& current,
            bool heap_unchanged) const {
        if (state.wrd.previous_ids.empty()) {
            state.wrd.previous_ids = current;
            return false;
        }
        if (heap_unchanged) {
            return observe_wrd_delta(state, 0.0);
        }
        const double delta = weighted_rank_delta(
                state.wrd, state.wrd.previous_ids, current, weights_);
        state.wrd.previous_ids = current;
        return observe_wrd_delta(state, delta);
    }

    void publish(
            const std::shared_ptr<Session>& session,
            int local_index,
            const std::string& phase,
            bool terminal,
            const std::string& reason = std::string(),
            const std::vector<float>* sorted_distances = nullptr,
            const std::vector<faiss::idx_t>* sorted_labels = nullptr) {
        QueryState& state = *session->queries[static_cast<size_t>(local_index)];
        auto event = std::make_shared<Event>();
        event->local_index = local_index;
        event->budget = state.budget.load(std::memory_order_acquire);
        event->phase = phase;
        event->terminal = terminal;
        event->parked = state.paused.load(std::memory_order_acquire) && !terminal;
        event->terminal_reason = reason;
        event->wrd_delta = state.wrd.last_delta;
        event->wrd_risk = state.wrd.last_risk;
        event->wrd_stable_streak = state.wrd.stable_streak;
        event->actual_scanned_lists = state.actual_scanned_lists.load();
        event->actual_scanned_documents =
                state.actual_scanned_documents.load();
        event->prefetch_advised_lists = state.prefetch_advised_lists.load();
        event->prefetch_advised_bytes = state.prefetch_advised_bytes.load();
        event->prefetch_advice_errors = state.prefetch_advice_errors.load();
        event->prefetch_high_water = state.prefetch_high_water.load();
        event->heap_updates = state.heap_updates.load();
        event->unchanged_heap_lists = state.unchanged_heap_lists.load();
        event->scheduled_high_water = state.scheduled_high_water.load();
        event->computed_high_water = state.computed_high_water.load();
        if (sorted_distances && sorted_labels) {
            event->distances = *sorted_distances;
            event->labels = *sorted_labels;
        } else {
            sorted_snapshot(state, event->distances, event->labels);
        }
        {
            std::lock_guard<std::mutex> lock(session->event_mutex);
            event->sequence = session->next_sequence++;
            session->events.push_back(event);
            // Stamp completion only after the result payload is materialized
            // and queued. Poll cadence therefore never enters request
            // latency, while result-copy cost does.
            const Clock::time_point published = Clock::now();
            event->elapsed_ms =
                    std::chrono::duration<double, std::milli>(
                            published - session->started)
                            .count();
            event->terminal_steady_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                            published.time_since_epoch())
                            .count();
        }
        session->event_cv.notify_one();
    }

    bool cancellation_requested(
            const std::shared_ptr<Session>& session,
            QueryState& state) const {
        if (session->cancel_all.load(std::memory_order_acquire)) {
            state.external_cancel.store(true, std::memory_order_release);
        }
        return state.external_cancel.load(std::memory_order_acquire);
    }

    bool advise_file_range(
            QueryState& state,
            size_t offset,
            size_t length) const {
        if (length == 0) return false;
        if (prefetch_fd_ < 0) {
            state.prefetch_advice_errors.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const int error = ::posix_fadvise(
                prefetch_fd_,
                static_cast<off_t>(offset),
                static_cast<off_t>(length),
                POSIX_FADV_WILLNEED);
        if (error != 0) {
            state.prefetch_advice_errors.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        state.prefetch_advised_bytes.fetch_add(
                static_cast<long long>(length), std::memory_order_relaxed);
        return true;
    }

    void advise_rank(
            const std::shared_ptr<Session>& session,
            int local_index,
            int rank) const {
        QueryState& state = *session->queries[static_cast<size_t>(local_index)];
        const size_t coarse_offset =
                static_cast<size_t>(local_index) * max_nprobe_ + rank;
        const faiss::idx_t list_no = session->coarse_ids[coarse_offset];
        bool advised = false;
        if (ondisk_invlists_ && list_no >= 0) {
            const size_t list_index = static_cast<size_t>(list_no);
            const size_t list_size = ondisk_invlists_->list_size(list_index);
            if (list_size > 0) {
                const auto& list = ondisk_invlists_->lists[list_index];
                const size_t code_bytes = list_size * ondisk_invlists_->code_size;
                const size_t id_offset = list.offset +
                        list.capacity * ondisk_invlists_->code_size;
                const size_t id_bytes = list_size * sizeof(faiss::idx_t);
                advised = advise_file_range(
                                  state, list.offset, code_bytes) ||
                        advised;
                advised = advise_file_range(state, id_offset, id_bytes) ||
                        advised;
            }
        }
        if (advised) {
            state.prefetch_advised_lists.fetch_add(
                    1, std::memory_order_relaxed);
        }
        state.prefetch_high_water.store(rank + 1, std::memory_order_release);
    }

    void advise_through(
            const std::shared_ptr<Session>& session,
            int local_index,
            int exclusive_end,
            bool honor_pause) const {
        QueryState& state = *session->queries[static_cast<size_t>(local_index)];
        const int bounded_end = std::min(max_nprobe_, exclusive_end);
        if (!ondisk_invlists_) {
            // An in-memory index has no ranges to advise. Avoid walking and
            // atomically publishing every synthetic prefetch rank.
            state.prefetch_high_water.store(
                    bounded_end, std::memory_order_release);
            return;
        }
        int rank = state.prefetch_high_water.load(std::memory_order_acquire);
        while (rank < bounded_end) {
            if (cancellation_requested(session, state) ||
                state.finished.load(std::memory_order_acquire) ||
                (honor_pause &&
                 state.paused.load(std::memory_order_acquire))) {
                return;
            }
            advise_rank(session, local_index, rank);
            ++rank;
        }
    }

    bool scan_rank(
            const std::shared_ptr<Session>& session,
            int local_index,
            int rank,
            faiss::InvertedListScanner* scanner,
            bool honor_pause,
            size_t* heap_updates = nullptr) {
        QueryState& state = *session->queries[static_cast<size_t>(local_index)];
        harmorag::ScanPermitLimiter::Lease scan_permit;
        while (true) {
            if (honor_pause &&
                state.paused.load(std::memory_order_acquire)) {
                std::unique_lock<std::mutex> lock(session->task_mutex);
                session->task_cv.wait(lock, [&]() {
                    return cancellation_requested(session, state) ||
                            !state.paused.load(std::memory_order_acquire);
                });
            }
            if (cancellation_requested(session, state)) return false;

            if (use_scan_permits_) {
                scan_permit = scan_permits_.acquire(
                        session->id,
                        session->scan_permits,
                        [&]() {
                            return cancellation_requested(session, state) ||
                                    (honor_pause &&
                                     state.paused.load(std::memory_order_acquire));
                        });
                if (!scan_permit) {
                    // pause()/cancel() woke and removed our queued ticket.  A
                    // paused continuation waits on task_cv before submitting
                    // a new ticket; cancellation remains terminal.
                    session->task_cv.notify_all();
                    if (cancellation_requested(session, state)) return false;
                    continue;
                }
            }

            if (honor_pause) {
                state.active_scan.store(true, std::memory_order_seq_cst);
            }
            // Close the race where pause() published its flag between the
            // permit grant and active_scan becoming visible. In that ordering
            // pause may already have returned, but this second check prevents
            // any list from starting behind its barrier and Lease releases the
            // unused permit on continue/return.
            if (cancellation_requested(session, state) ||
                (honor_pause &&
                 state.paused.load(std::memory_order_seq_cst))) {
                if (honor_pause) {
                    state.active_scan.store(
                            false, std::memory_order_seq_cst);
                }
                scan_permit.reset();
                session->task_cv.notify_all();
                if (cancellation_requested(session, state)) return false;
                continue;
            }
            state.scheduled_high_water.store(rank + 1, std::memory_order_release);
            break;
        }

        try {
            const size_t offset =
                    static_cast<size_t>(local_index) * max_nprobe_ + rank;
            const faiss::idx_t list_no = session->coarse_ids[offset];
            const size_t scanned_documents = list_no >= 0
                    ? ivf_->invlists->list_size(
                              static_cast<size_t>(list_no))
                    : 0;
            const int64_t scan_started_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                            WallClock::now().time_since_epoch())
                            .count();
            const size_t updates = scan_list(
                    scanner,
                    list_no,
                    session->coarse_scores[offset],
                    state.heap_distances,
                    state.heap_labels);
            const int64_t scan_stopped_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                            WallClock::now().time_since_epoch())
                            .count();
            state.scan_active_intervals_ns.emplace_back(
                    scan_started_ns, scan_stopped_ns);
            state.heap_updates.fetch_add(
                    static_cast<long long>(updates), std::memory_order_relaxed);
            if (updates == 0) {
                state.unchanged_heap_lists.fetch_add(
                        1, std::memory_order_relaxed);
            }
            if (heap_updates) *heap_updates = updates;
            state.actual_scanned_lists.fetch_add(1, std::memory_order_relaxed);
            state.actual_scanned_documents.fetch_add(
                    static_cast<long long>(scanned_documents),
                    std::memory_order_relaxed);
            state.computed_high_water.store(rank + 1, std::memory_order_release);
            state.budget.store(rank + 1, std::memory_order_release);
        } catch (...) {
            if (honor_pause) {
                state.active_scan.store(false, std::memory_order_seq_cst);
            }
            session->task_cv.notify_all();
            throw;
        }
        if (honor_pause) {
            state.active_scan.store(false, std::memory_order_seq_cst);
        }
        if (scan_permit) scan_permit.reset();
        if ((honor_pause &&
             state.paused.load(std::memory_order_acquire)) ||
            cancellation_requested(session, state)) {
            session->task_cv.notify_all();
        }
        return true;
    }

    void finish_query(
            const std::shared_ptr<Session>& session,
            int local_index,
            const std::string& reason,
            const std::vector<float>* sorted_distances = nullptr,
            const std::vector<faiss::idx_t>* sorted_labels = nullptr) {
        QueryState& state = *session->queries[static_cast<size_t>(local_index)];
        if (state.finished.exchange(true, std::memory_order_acq_rel)) return;
        state.terminal_reason = reason;
        publish(
                session,
                local_index,
                "terminal",
                true,
                reason,
                sorted_distances,
                sorted_labels);
        session->task_cv.notify_all();
    }

    void bootstrap_query(
            const std::shared_ptr<Session>& session,
            int local_index) {
        QueryState& state = *session->queries[static_cast<size_t>(local_index)];
        // A parked lookahead session must not pull any continuation pages into
        // cache.  The fast prefix is scanned by one query worker without
        // publishing per-list Python events.  WRD is nevertheless observed
        // after every physical list, beginning with the @1 heap, so its
        // statistical history is independent of the generation latch.
        advise_through(session, local_index, initial_nprobe_, false);
        state.scanner.reset(ivf_->get_InvertedListScanner(false, nullptr));
        state.scanner->set_query(
                session->embeddings.data() +
                static_cast<size_t>(local_index) * session->d);

        // The fast prefix is phase one for the entire 2-D handle.  No query may
        // scan beyond it until every query has published the generation-latch
        // snapshot.  This preserves the batch barrier without making that
        // snapshot the first WRD observation.
        for (int rank = 0; rank < initial_nprobe_; ++rank) {
            size_t heap_updates = 0;
            if (!scan_rank(
                        session,
                        local_index,
                        rank,
                        state.scanner.get(),
                        false,
                        &heap_updates)) {
                finish_query(session, local_index, "cancelled");
                return;
            }
            if (heap_updates != 0) {
                state.sorted_valid = false;
                state.wrd.heap_changed_since_observation = true;
            }
            const bool at_max = rank + 1 >= max_nprobe_;
            const bool check_wrd =
                    (rank + 1) % wrd_check_interval_ == 0 || at_max;
            if (check_wrd && !state.sorted_valid) {
                sorted_snapshot(
                        state, state.sorted_distances, state.sorted_labels);
                state.sorted_valid = true;
            }
            bool stable = false;
            if (enable_wrd_ && check_wrd) {
                stable = observe_wrd(
                        state,
                        state.sorted_labels,
                        !state.wrd.heap_changed_since_observation);
                state.wrd.heap_changed_since_observation = false;
            }
            if (stable || at_max) {
                // A genuinely stable terminal before the generation target is
                // still a valid frozen snapshot.  In that case M2 consumes
                // the terminal heap immediately; retrieval must not resume
                // merely to manufacture a snapshot at the configured target.
                finish_query(
                        session,
                        local_index,
                        at_max ? "max_nprobe" : "wrd_stable",
                        &state.sorted_distances,
                        &state.sorted_labels);
                return;
            }
        }
        if (cancellation_requested(session, state)) {
            finish_query(session, local_index, "cancelled");
            return;
        }
        if (!state.sorted_valid) {
            sorted_snapshot(
                    state, state.sorted_distances, state.sorted_labels);
            state.sorted_valid = true;
        }

        publish(
                session,
                local_index,
                session->pause_after_bootstrap ? "bootstrap_parked" : "bootstrap",
                false,
                std::string(),
                &state.sorted_distances,
                &state.sorted_labels);
    }

    void continue_query(
            const std::shared_ptr<Session>& session,
            int local_index,
            bool honor_pause = true,
            int max_lists = 0) {
        QueryState& state = *session->queries[static_cast<size_t>(local_index)];
        if (state.finished.load(std::memory_order_acquire)) return;
        if (cancellation_requested(session, state)) {
            finish_query(session, local_index, "cancelled");
            return;
        }
        if (!state.scanner) {
            throw std::runtime_error(
                    "native scanner continuation started before bootstrap");
        }

        const int first_rank = std::max(
                initial_nprobe_, state.budget.load(std::memory_order_acquire));
        if (!state.sorted_valid) {
            sorted_snapshot(
                    state, state.sorted_distances, state.sorted_labels);
            state.sorted_valid = true;
        }
        // Keep at most lookahead_ list ranks advised.  Each successful scan
        // refills one rank only after WRD/cancellation decides that retrieval
        // remains live.
        advise_through(
                session,
                local_index,
                first_rank + lookahead_,
                honor_pause);
        int lists_scanned_this_call = 0;
        for (int rank = first_rank; rank < max_nprobe_; ++rank) {
            size_t heap_updates = 0;
            if (!scan_rank(
                        session,
                        local_index,
                        rank,
                        state.scanner.get(),
                        honor_pause,
                        &heap_updates)) {
                finish_query(
                        session,
                        local_index,
                        "cancelled",
                        &state.sorted_distances,
                        &state.sorted_labels);
                return;
            }
            if (heap_updates != 0) {
                state.sorted_valid = false;
                state.wrd.heap_changed_since_observation = true;
            }
            const bool at_max = rank + 1 >= max_nprobe_;
            const bool check_wrd =
                    (rank + 1) % wrd_check_interval_ == 0 || at_max;
            if (check_wrd && !state.sorted_valid) {
                sorted_snapshot(
                        state, state.sorted_distances, state.sorted_labels);
                state.sorted_valid = true;
            }
            bool stable = false;
            if (enable_wrd_ && check_wrd) {
                stable = observe_wrd(
                        state,
                        state.sorted_labels,
                        !state.wrd.heap_changed_since_observation);
                state.wrd.heap_changed_since_observation = false;
            }
            if (cancellation_requested(session, state)) {
                finish_query(
                        session,
                        local_index,
                        "cancelled",
                        &state.sorted_distances,
                        &state.sorted_labels);
                return;
            }
            if (stable || at_max) {
                // The hard ceiling wins over a simultaneous WRD signal.
                finish_query(
                        session,
                        local_index,
                        at_max ? "max_nprobe" : "wrd_stable",
                        &state.sorted_distances,
                        &state.sorted_labels);
                return;
            }
            advise_through(
                    session,
                    local_index,
                    rank + 1 + lookahead_,
                    honor_pause);
#if HARMORAG_PUBLISH_EVERY_BUDGET
            publish(
                    session,
                    local_index,
                    "searching",
                    false,
                    std::string(),
                    &state.sorted_distances,
                    &state.sorted_labels);
#endif
            ++lists_scanned_this_call;
            if (max_lists > 0 && lists_scanned_this_call >= max_lists) {
                // Yield only after the same per-list WRD, cancellation, heap,
                // event publication, accounting, and rolling-prefetch work
                // used by the unlimited path.  The persistent worker will
                // re-enqueue this query; no list is repeated because
                // state.budget is the next cursor.
                state.continuation_quanta.fetch_add(
                        1, std::memory_order_relaxed);
                return;
            }
        }
    }

    void record_worker_error(
            const std::shared_ptr<Session>& session,
            int local_index,
            const std::exception& error) {
        {
            std::lock_guard<std::mutex> lock(session->error_mutex);
            if (session->error.empty()) session->error = error.what();
        }
        QueryState& state = *session->queries[static_cast<size_t>(local_index)];
        state.active_scan.store(false, std::memory_order_seq_cst);
        finish_query(session, local_index, "error");
    }

    template <typename Function>
    void run_parallel_jobs(
            const std::shared_ptr<Session>& session,
            const std::vector<int>& local_indices,
            Function function) {
        if (local_indices.empty()) return;
        const int places = omp_get_num_places();
        const int available_workers = places > 1 ? places - 1 : worker_threads_;
        const int count = std::min(
                std::min(
                        worker_threads_,
                        static_cast<int>(local_indices.size())),
                std::min(available_workers, scan_permits_.capacity()));
        std::atomic<int> next_query{0};
        omp_set_dynamic(0);
        // Use FAISS' OpenMP runtime for the query workers.  With
        // OMP_PROC_BIND=spread this creates the same core-distributed team as
        // stock IndexIVF::search.  The controller/master is deliberately kept
        // out of scanning: in a Python process it shares the main OpenMP place
        // with runtime helper threads and otherwise becomes a multi-second
        // straggler.  At most places-1 query workers occupy distinct cores.
#pragma omp parallel num_threads(count + 1)
        {
            if (omp_get_thread_num() != 0) {
                while (true) {
                    const int position = next_query.fetch_add(
                            1, std::memory_order_relaxed);
                    if (position >= static_cast<int>(local_indices.size())) break;
                    const int local_index =
                            local_indices[static_cast<size_t>(position)];
                    try {
                        function(local_index);
                    } catch (const std::exception& error) {
                        record_worker_error(session, local_index, error);
                    }
                }
            }
        }
    }

    bool all_queries_finished(
            const std::shared_ptr<Session>& session) const {
        for (const auto& state : session->queries) {
            if (!state->finished.load(std::memory_order_acquire)) return false;
        }
        return true;
    }

    void run_queries_parallel(const std::shared_ptr<Session>& session) {
        std::vector<int> all_indices(static_cast<size_t>(session->n));
        std::iota(all_indices.begin(), all_indices.end(), 0);

        // A non-parked session does not need a batch-wide prefix barrier.
        // Keep each query on one worker from its exact bootstrap snapshot
        // through terminal WRD/max-nprobe completion.  The bootstrap event is
        // still published, so an M2 consumer can begin generation while that
        // same query continues scanning.
        if (!session->pause_after_bootstrap) {
            run_parallel_jobs(session, all_indices, [&](int local_index) {
                bootstrap_query(session, local_index);
                QueryState& state = *session->queries[
                        static_cast<size_t>(local_index)];
                if (!state.finished.load(std::memory_order_acquire)) {
                    continue_query(session, local_index, false);
                }
            });
            return;
        }

        // Phase 1 is a true fast-prefix batch barrier.  In particular, a
        // current batch cannot scan beyond the latch before every query has
        // emitted its exact latch snapshot.
        run_parallel_jobs(session, all_indices, [&](int local_index) {
            bootstrap_query(session, local_index);
        });

        // Phase 2 runs only queries which are explicitly runnable.  Lookahead
        // sessions therefore consume zero scan workers while parked at the
        // configured fast prefix;
        // current sessions enter here immediately and keep scanning without
        // any artificial pause/resume handshake.
        while (!all_queries_finished(session)) {
            std::vector<int> runnable;
            {
                std::unique_lock<std::mutex> lock(session->task_mutex);
                session->task_cv.wait(lock, [&]() {
                    if (all_queries_finished(session) ||
                        session->cancel_all.load(std::memory_order_acquire)) {
                        return true;
                    }
                    for (const auto& state : session->queries) {
                        if (!state->finished.load(std::memory_order_acquire) &&
                            (state->external_cancel.load(
                                     std::memory_order_acquire) ||
                             !state->paused.load(std::memory_order_acquire))) {
                            return true;
                        }
                    }
                    return false;
                });

                const bool cancel_all =
                        session->cancel_all.load(std::memory_order_acquire);
                for (int local_index = 0;
                     local_index < session->n;
                     ++local_index) {
                    const QueryState& state = *session->queries[
                            static_cast<size_t>(local_index)];
                    if (!state.finished.load(std::memory_order_acquire) &&
                        (cancel_all ||
                         state.external_cancel.load(std::memory_order_acquire) ||
                         !state.paused.load(std::memory_order_acquire))) {
                        runnable.push_back(local_index);
                    }
                }
            }
            if (runnable.empty()) continue;
            run_parallel_jobs(session, runnable, [&](int local_index) {
                continue_query(session, local_index);
            });
        }
    }

    void prepare_coarse(const std::shared_ptr<Session>& session) {
        if (session->coarse_supplied) return;
        session->coarse_scores.resize(
                static_cast<size_t>(session->n) * max_nprobe_);
        session->coarse_ids.resize(
                static_cast<size_t>(session->n) * max_nprobe_);
        // IndexFlat's coarse search consults the calling thread's OpenMP
        // maximum. Match the team to actual admitted query parallelism.
        const int coarse_threads =
                std::max(1, std::min(worker_threads_, session->n));
        omp_set_dynamic(0);
        omp_set_num_threads(coarse_threads);
        ivf_->quantizer->search(
                session->n,
                session->embeddings.data(),
                max_nprobe_,
                session->coarse_scores.data(),
                session->coarse_ids.data());
    }

    void complete_session(const std::shared_ptr<Session>& session) {
        {
            std::lock_guard<std::mutex> lock(session->event_mutex);
            session->done = true;
        }
        session->event_cv.notify_all();
        session->done_cv.notify_all();
    }

    void persistent_worker_loop() {
        while (true) {
            PersistentJob job;
            bool priority_dispatch = false;
            {
                std::unique_lock<std::mutex> lock(persistent_mutex_);
                persistent_cv_.wait(lock, [&]() {
                    return persistent_stopping_ || !persistent_jobs_.empty();
                });
                if (persistent_stopping_ && persistent_jobs_.empty()) return;
                auto selected = persistent_jobs_.begin();
                const bool burst_fairness_dispatch =
                        request_aligned_priority_ &&
                        request_aligned_priority_burst_ > 0 &&
                        consecutive_priority_dispatches_ >=
                                request_aligned_priority_burst_;
                const bool age_fairness_dispatch =
                        request_aligned_priority_ &&
                        request_aligned_max_fifo_wait_ms_ > 0 &&
                        Clock::now() - persistent_jobs_.front().enqueued_at >=
                                std::chrono::milliseconds(
                                        request_aligned_max_fifo_wait_ms_);
                const bool fairness_dispatch =
                        burst_fairness_dispatch || age_fairness_dispatch;
                if (request_aligned_priority_ && !fairness_dispatch) {
                    // Prefer continuation work whose generation request is
                    // currently in flight.  Bootstrap and inactive
                    // continuations remain FIFO fallbacks, so the scheduler is
                    // work-conserving and never manufactures overlap by idling.
                    for (auto it = persistent_jobs_.begin();
                         it != persistent_jobs_.end(); ++it) {
                        if (!it->continuation) continue;
                        const QueryState& candidate =
                                *it->session->queries[static_cast<size_t>(
                                        it->local_index)];
                        if (candidate.generation_active.load(
                                    std::memory_order_acquire)) {
                            selected = it;
                            priority_dispatch = true;
                            break;
                        }
                    }
                }
                if (priority_dispatch) {
                    ++consecutive_priority_dispatches_;
                } else {
                    if (fairness_dispatch) {
                        persistent_fairness_dispatches_.fetch_add(
                                1, std::memory_order_relaxed);
                    }
                    consecutive_priority_dispatches_ = 0;
                }
                job = std::move(*selected);
                persistent_jobs_.erase(selected);
            }
            bool query_terminal = false;
            try {
                QueryState& state = *job.session->queries[
                        static_cast<size_t>(job.local_index)];
                if (job.continuation) {
                    persistent_continuation_dispatches_.fetch_add(
                            1, std::memory_order_relaxed);
                    if (priority_dispatch) {
                        persistent_priority_dispatches_.fetch_add(
                                1, std::memory_order_relaxed);
                        state.priority_dispatches.fetch_add(
                                1, std::memory_order_relaxed);
                    }
                    continue_query(
                            job.session,
                            job.local_index,
                            false,
                            continuation_quantum_);
                } else {
                    persistent_bootstrap_dispatches_.fetch_add(
                            1, std::memory_order_relaxed);
                    bootstrap_query(job.session, job.local_index);
                }
                query_terminal = state.finished.load(std::memory_order_acquire);
            } catch (const std::exception& error) {
                record_worker_error(
                        job.session, job.local_index, error);
                query_terminal = true;
            }

            if (!job.continuation) {
                // The final bootstrap completion releases every unfinished
                // query in this microbatch into the scanner-wide continuation
                // queue. The initial evidence barrier is therefore preserved
                // without a batch completion barrier during list expansion.
                const bool bootstrap_barrier_reached =
                        job.session->remaining_bootstrap_queries.fetch_sub(
                                1, std::memory_order_acq_rel) == 1;
                if (query_terminal &&
                    job.session->remaining_queries.fetch_sub(
                            1, std::memory_order_acq_rel) == 1) {
                    complete_session(job.session);
                }
                if (bootstrap_barrier_reached) {
                    {
                        std::lock_guard<std::mutex> lock(persistent_mutex_);
                        for (int qi = 0; qi < job.session->n; ++qi) {
                            QueryState& state = *job.session->queries[
                                    static_cast<size_t>(qi)];
                            if (!state.finished.load(std::memory_order_acquire)) {
                                persistent_jobs_.push_back(
                                        {job.session, qi, true, Clock::now()});
                            }
                        }
                    }
                    persistent_cv_.notify_all();
                }
            } else if (query_terminal) {
                if (job.session->remaining_queries.fetch_sub(
                            1, std::memory_order_acq_rel) == 1) {
                    complete_session(job.session);
                }
            } else {
                // Only the persistent full-continuous path uses bounded
                // continuation quanta.  Exactly one job exists per live query,
                // so re-enqueueing cannot duplicate physical list work.
                {
                    std::lock_guard<std::mutex> lock(persistent_mutex_);
                    job.enqueued_at = Clock::now();
                    persistent_jobs_.push_back(std::move(job));
                }
                persistent_continuation_requeues_.fetch_add(
                        1, std::memory_order_relaxed);
                persistent_cv_.notify_one();
            }
        }
    }

    void run_session(const std::shared_ptr<Session>& session) {
        try {
            prepare_coarse(session);
            // Do not invoke OnDiskInvertedLists::prefetch_lists for all
            // n*max_nprobe lists.  FAISS 1.8's implementation owns one global
            // queue, clears it on every call, and eagerly touches every queued
            // byte.  Per-query rolling advice is issued by bootstrap_query /
            // continue_query instead, so WRD/cancellation bounds future I/O.
            run_queries_parallel(session);
        } catch (const std::exception& error) {
            {
                std::lock_guard<std::mutex> lock(session->error_mutex);
                if (session->error.empty()) session->error = error.what();
            }
            for (int qi = 0; qi < session->n; ++qi) {
                QueryState& state = *session->queries[static_cast<size_t>(qi)];
                if (!state.finished.load(std::memory_order_acquire)) {
                    state.external_cancel.store(true, std::memory_order_release);
                }
            }
            scan_permits_.notify_all();
            session->task_cv.notify_all();
            for (int qi = 0; qi < session->n; ++qi) {
                if (!session->queries[static_cast<size_t>(qi)]->finished.load()) {
                    finish_query(session, qi, "error");
                }
            }
        }
        complete_session(session);
    }

    py::dict event_to_python(int session_id, const Event& event, bool session_done) const {
        py::dict output;
        output["ready"] = true;
        output["done"] = session_done;
        output["session_id"] = session_id;
        output["prefetch_backend"] = prefetch_backend();
        output["prefetch_lookahead"] = lookahead_;
        output["sequence"] = event.sequence;
        output["local_index"] = event.local_index;
        output["budget"] = event.budget;
        output["merged_budget"] = event.budget;
        output["phase"] = event.phase;
        output["terminal"] = event.terminal;
        output["terminal_reason"] = event.terminal_reason;
        output["parked"] = event.parked;
        output["scheduler"] = kScheduler;
        output["continuation_quantum"] = continuation_quantum_;
        output["request_aligned_priority"] = request_aligned_priority_;
        output["request_aligned_priority_burst"] =
                request_aligned_priority_burst_;
        output["request_aligned_max_fifo_wait_ms"] =
                request_aligned_max_fifo_wait_ms_;
        output["wrd_delta"] = std::isfinite(event.wrd_delta) ? py::cast(event.wrd_delta) : py::none();
        output["wrd_risk"] = std::isfinite(event.wrd_risk) ? py::cast(event.wrd_risk) : py::none();
        output["wrd_stable_streak"] = event.wrd_stable_streak;
        output["actual_scanned_lists"] = event.actual_scanned_lists;
        output["completed_list_count"] = event.actual_scanned_lists;
        output["actual_scanned_documents"] = event.actual_scanned_documents;
        output["prefetch_advised_lists"] = event.prefetch_advised_lists;
        output["prefetch_advised_bytes"] = event.prefetch_advised_bytes;
        output["prefetch_advice_errors"] = event.prefetch_advice_errors;
        output["prefetch_high_water"] = event.prefetch_high_water;
        output["heap_updates"] = event.heap_updates;
        output["unchanged_heap_lists"] = event.unchanged_heap_lists;
        output["scheduled_high_water"] = event.scheduled_high_water;
        output["computed_high_water"] = event.computed_high_water;
        output["elapsed_ms"] = event.elapsed_ms;
        output["terminal_steady_ns"] = event.terminal_steady_ns;
        py::array_t<float> distances(event.distances.size());
        py::array_t<int64_t> labels(event.labels.size());
        std::memcpy(distances.mutable_data(), event.distances.data(),
                    event.distances.size() * sizeof(float));
        std::memcpy(labels.mutable_data(), event.labels.data(),
                    event.labels.size() * sizeof(int64_t));
        output["distances"] = std::move(distances);
        output["labels"] = std::move(labels);
        return output;
    }

    py::dict summary_to_python(const Session& session, bool done) const {
        py::dict output;
        output["session_id"] = session.id;
        output["done"] = done;
        output["scheduler"] = kScheduler;
        output["continuation_quantum"] = continuation_quantum_;
        output["request_aligned_priority"] = request_aligned_priority_;
        output["request_aligned_priority_burst"] =
                request_aligned_priority_burst_;
        output["request_aligned_max_fifo_wait_ms"] =
                request_aligned_max_fifo_wait_ms_;
        output["persistent_bootstrap_dispatches"] =
                persistent_bootstrap_dispatches_.load();
        output["persistent_continuation_dispatches"] =
                persistent_continuation_dispatches_.load();
        output["persistent_priority_dispatches"] =
                persistent_priority_dispatches_.load();
        output["persistent_fairness_dispatches"] =
                persistent_fairness_dispatches_.load();
        output["persistent_continuation_requeues"] =
                persistent_continuation_requeues_.load();
        output["prefetch_backend"] = prefetch_backend();
        output["prefetch_lookahead"] = lookahead_;
        output["scan_permit_global"] = global_scan_permit_stats_to_python();
        output["scan_permit_session"] = session_scan_permit_stats_to_python(
                session.scan_permits);
        output["error"] = session.error;
        long long total = 0;
        long long total_documents = 0;
        long long total_advised_lists = 0;
        long long total_advised_bytes = 0;
        long long total_advice_errors = 0;
        py::list queries;
        for (int qi = 0; qi < session.n; ++qi) {
            const QueryState& state = *session.queries[static_cast<size_t>(qi)];
            py::dict item;
            const long long actual = state.actual_scanned_lists.load();
            const long long actual_documents =
                    state.actual_scanned_documents.load();
            total += actual;
            total_documents += actual_documents;
            total_advised_lists += state.prefetch_advised_lists.load();
            total_advised_bytes += state.prefetch_advised_bytes.load();
            total_advice_errors += state.prefetch_advice_errors.load();
            item["local_index"] = qi;
            item["budget"] = state.budget.load(std::memory_order_acquire);
            item["merged_budget"] = state.budget.load(std::memory_order_acquire);
            item["terminal_reason"] = state.terminal_reason;
            item["actual_scanned_lists"] = actual;
            item["completed_list_count"] = actual;
            item["actual_scanned_documents"] = actual_documents;
            item["prefetch_advised_lists"] = state.prefetch_advised_lists.load();
            item["prefetch_advised_bytes"] = state.prefetch_advised_bytes.load();
            item["prefetch_advice_errors"] = state.prefetch_advice_errors.load();
            item["prefetch_high_water"] = state.prefetch_high_water.load();
            item["heap_updates"] = state.heap_updates.load();
            item["unchanged_heap_lists"] = state.unchanged_heap_lists.load();
            item["scheduled_high_water"] = state.scheduled_high_water.load();
            item["computed_high_water"] = state.computed_high_water.load();
            item["generation_active"] = state.generation_active.load();
            item["continuation_quanta"] = state.continuation_quanta.load();
            item["priority_dispatches"] = state.priority_dispatches.load();
            py::list query_scan_active_intervals;
            for (const auto& interval : state.scan_active_intervals_ns) {
                query_scan_active_intervals.append(
                        py::make_tuple(interval.first, interval.second));
            }
            item["scan_active_clock"] = "unix_time_ns";
            item["scan_active_interval_count"] =
                    static_cast<long long>(state.scan_active_intervals_ns.size());
            item["scan_active_intervals_ns"] =
                    std::move(query_scan_active_intervals);
            queries.append(std::move(item));
        }
        output["actual_scanned_lists"] = total;
        output["completed_list_count"] = total;
        output["actual_scanned_documents"] = total_documents;
        output["prefetch_advised_lists"] = total_advised_lists;
        output["prefetch_advised_bytes"] = total_advised_bytes;
        output["prefetch_advice_errors"] = total_advice_errors;
        py::list scan_active_intervals;
        long long scan_active_interval_count = 0;
        for (const auto& query : session.queries) {
            for (const auto& interval : query->scan_active_intervals_ns) {
                scan_active_intervals.append(
                        py::make_tuple(interval.first, interval.second));
                ++scan_active_interval_count;
            }
        }
        output["scan_active_clock"] = "unix_time_ns";
        output["scan_active_interval_count"] = scan_active_interval_count;
        output["scan_active_intervals_ns"] = std::move(scan_active_intervals);
        output["queries"] = std::move(queries);
        return output;
    }

    void shutdown_all() {
        std::vector<std::shared_ptr<Session>> sessions;
        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            for (const auto& entry : sessions_) sessions.push_back(entry.second);
        }
        for (const auto& session : sessions) {
            {
                std::lock_guard<std::mutex> lock(session->task_mutex);
                session->cancel_all.store(true);
                for (auto& state : session->queries) {
                    state->external_cancel.store(true);
                }
            }
            session->task_cv.notify_all();
        }
        // Some workers may be blocked in the scanner-global queue rather than
        // on a session task_cv. Their stop predicates now observe cancel_all.
        scan_permits_.notify_all();
        {
            std::lock_guard<std::mutex> lock(persistent_mutex_);
            persistent_stopping_ = true;
        }
        persistent_cv_.notify_all();
        for (auto& worker : persistent_workers_) {
            if (worker.joinable()) worker.join();
        }
        for (const auto& session : sessions) {
            if (session->controller.joinable()) session->controller.join();
        }
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        sessions_.clear();
    }
};

} // namespace

PYBIND11_MODULE(harmorag_native_scanner, module) {
    module.doc() = "Asynchronous query-parallel direct IVFFlat scanner for HarmoRAG";
    py::class_<ContinuousIVFScanner>(module, "ContinuousIVFScanner")
            .def(py::init<
                         const std::string&, int, int, int, int, int, bool,
                         double, double, int, double, double, int, int, int,
                         bool, bool, int, int, bool, int, int>(),
                 py::arg("index_path"),
                 py::arg("worker_threads") = 24,
                 py::arg("initial_nprobe") = 28,
                 py::arg("max_nprobe") = 128,
                 py::arg("heap_topk") = 100,
                 py::arg("lookahead") = 8,
                 py::arg("enable_wrd") = true,
                 py::arg("wrd_alpha") = 1.0,
                 py::arg("wrd_threshold") = 0.0025,
                 py::arg("wrd_window") = 3,
                 py::arg("wrd_beta") = 0.4,
                 py::arg("wrd_mu") = 0.5,
                 py::arg("wrd_patience") = 16,
                 // Zero preserves source compatibility and resolves to
                 // worker_threads. Python production config passes an
                 // explicit scanner-global capacity.
                 py::arg("global_scan_permits") = 0,
                 // Zero preserves the historical heap_topk behavior.
                 py::arg("wrd_topk") = 0,
                 py::arg("use_mmap") = true,
                 py::arg("use_scan_permits") = true,
                 py::arg("wrd_check_interval") = 1,
                 py::arg("continuation_quantum") = 0,
                 py::arg("request_aligned_priority") = false,
                 py::arg("request_aligned_priority_burst") = 0,
                 py::arg("request_aligned_max_fifo_wait_ms") = 0)
            .def("start", &ContinuousIVFScanner::start,
                 py::arg("embeddings"),
                 py::arg("coarse_ids") = py::none(),
                 py::arg("coarse_scores") = py::none(),
                 py::arg("pause_after_bootstrap") = true)
            .def("poll", &ContinuousIVFScanner::poll,
                 py::arg("session_id"),
                 py::arg("after_sequence") = -1,
                 py::arg("timeout_ms") = 0,
                 py::arg("latest") = true)
            .def("pause", &ContinuousIVFScanner::pause,
                 py::arg("session_id"), py::arg("local_index") = -1)
            .def("resume", &ContinuousIVFScanner::resume,
                 py::arg("session_id"), py::arg("local_index") = -1)
            .def("set_generation_active",
                 &ContinuousIVFScanner::set_generation_active,
                 py::arg("session_id"),
                 py::arg("local_index"),
                 py::arg("active") = true)
            .def("cancel", &ContinuousIVFScanner::cancel,
                 py::arg("session_id"), py::arg("local_index") = -1)
            .def("wait", &ContinuousIVFScanner::wait,
                 py::arg("session_id"), py::arg("timeout_ms") = -1)
            .def("stats", &ContinuousIVFScanner::stats, py::arg("session_id"))
            .def("permit_stats", &ContinuousIVFScanner::permit_stats)
            .def("close", &ContinuousIVFScanner::close, py::arg("session_id"))
            .def_property_readonly("dimension", &ContinuousIVFScanner::dimension)
            .def_property_readonly("nlist", &ContinuousIVFScanner::nlist)
            .def_property_readonly("scheduler", &ContinuousIVFScanner::scheduler)
            .def_property_readonly(
                    "global_scan_permits",
                    &ContinuousIVFScanner::global_scan_permits)
            .def_property_readonly(
                    "continuation_quantum",
                    &ContinuousIVFScanner::continuation_quantum)
            .def_property_readonly(
                    "request_aligned_priority",
                    &ContinuousIVFScanner::request_aligned_priority)
            .def_property_readonly(
                    "request_aligned_priority_burst",
                    &ContinuousIVFScanner::request_aligned_priority_burst)
            .def_property_readonly(
                    "request_aligned_max_fifo_wait_ms",
                    &ContinuousIVFScanner::request_aligned_max_fifo_wait_ms)
            .def_property_readonly(
                    "prefetch_backend", &ContinuousIVFScanner::prefetch_backend)
            .def_property_readonly(
                    "prefetch_lookahead", &ContinuousIVFScanner::prefetch_lookahead);
}
