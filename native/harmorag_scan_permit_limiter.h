#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace harmorag {

inline void update_atomic_max(std::atomic<int>& value, int candidate) {
    int current = value.load(std::memory_order_relaxed);
    while (current < candidate &&
           !value.compare_exchange_weak(
                   current, candidate, std::memory_order_relaxed)) {
    }
}

// Counters are owned by a native scanner Session.  They deliberately remain
// independent from the limiter so a completed/closed session can still report
// its own permit history after the limiter has removed its live demand state.
struct ScanPermitCounters {
    std::atomic<int> in_use{0};
    std::atomic<int> peak_in_use{0};
    std::atomic<int> waiting{0};
    std::atomic<int> peak_waiting{0};
    std::atomic<long long> acquires{0};
    std::atomic<long long> waits{0};
    std::atomic<long long> aborted_waits{0};
    std::atomic<long long> wait_ns{0};
    std::atomic<long long> hold_ns{0};
};

struct ScanPermitSnapshot {
    int capacity = 0;
    int in_use = 0;
    int peak_in_use = 0;
    int waiting = 0;
    int peak_waiting = 0;
    int active_sessions = 0;
    long long acquires = 0;
    long long waits = 0;
    long long aborted_waits = 0;
    long long wait_ns = 0;
    long long hold_ns = 0;
};

// Scanner-global, work-conserving fair limiter for physical IVF list scans.
//
// A session with no competitor may borrow every permit.  When multiple
// sessions queue work, the next free permit goes to the waiting session with
// the fewest permits currently in use; FIFO ticket order breaks ties.  Thus a
// newly-started lookahead session converges toward an equal share without
// permanently reducing the single-session worker team.
class ScanPermitLimiter {
public:
    using Clock = std::chrono::steady_clock;

    class Lease {
    public:
        Lease() = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        Lease(Lease&& other) noexcept {
            move_from(std::move(other));
        }

        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                reset();
                move_from(std::move(other));
            }
            return *this;
        }

        ~Lease() {
            reset();
        }

        explicit operator bool() const noexcept {
            return owner_ != nullptr;
        }

        void reset() noexcept {
            if (!owner_) return;
            ScanPermitLimiter* owner = owner_;
            ScanPermitCounters* counters = counters_;
            const int session_id = session_id_;
            const Clock::time_point acquired_at = acquired_at_;
            owner_ = nullptr;
            counters_ = nullptr;
            session_id_ = 0;
            owner->release(session_id, *counters, acquired_at);
        }

    private:
        friend class ScanPermitLimiter;

        Lease(
                ScanPermitLimiter* owner,
                int session_id,
                ScanPermitCounters* counters,
                Clock::time_point acquired_at)
                : owner_(owner),
                  session_id_(session_id),
                  counters_(counters),
                  acquired_at_(acquired_at) {}

        void move_from(Lease&& other) noexcept {
            owner_ = other.owner_;
            session_id_ = other.session_id_;
            counters_ = other.counters_;
            acquired_at_ = other.acquired_at_;
            other.owner_ = nullptr;
            other.counters_ = nullptr;
            other.session_id_ = 0;
        }

        ScanPermitLimiter* owner_ = nullptr;
        int session_id_ = 0;
        ScanPermitCounters* counters_ = nullptr;
        Clock::time_point acquired_at_{};
    };

    explicit ScanPermitLimiter(int capacity) : capacity_(capacity) {
        if (capacity < 1) {
            throw std::invalid_argument("scan permit capacity must be positive");
        }
    }

    ScanPermitLimiter(const ScanPermitLimiter&) = delete;
    ScanPermitLimiter& operator=(const ScanPermitLimiter&) = delete;

    int capacity() const noexcept {
        return capacity_;
    }

    // Returns an empty lease if stop_requested becomes true while queued.
    // Callers must invoke notify_all() after changing state observed by the
    // predicate (pause/cancel/shutdown), otherwise no permit release may occur
    // to wake an otherwise idle queue.
    template <typename StopPredicate>
    Lease acquire(
            int session_id,
            ScanPermitCounters& counters,
            StopPredicate stop_requested) {
        Waiter waiter;
        waiter.session_id = session_id;
        const Clock::time_point started = Clock::now();
        std::unique_lock<std::mutex> lock(mutex_);
        waiter.ticket = next_ticket_++;
        waiters_.push_back(&waiter);
        ++waiting_by_session_[session_id];
        const int session_waiting =
                counters.waiting.fetch_add(1, std::memory_order_relaxed) + 1;
        update_atomic_max(counters.peak_waiting, session_waiting);
        peak_waiting_ = std::max(
                peak_waiting_, static_cast<int>(waiters_.size()));

        const bool contended =
                in_use_ >= capacity_ || selected_waiter_locked() != &waiter;
        if (contended) {
            ++waits_;
            counters.waits.fetch_add(1, std::memory_order_relaxed);
        }

        while (true) {
            bool stop = false;
            try {
                stop = static_cast<bool>(stop_requested());
            } catch (...) {
                abort_wait_locked(waiter, counters, started, contended);
                lock.unlock();
                cv_.notify_all();
                throw;
            }
            if (stop) {
                abort_wait_locked(waiter, counters, started, contended);
                lock.unlock();
                cv_.notify_all();
                return Lease();
            }
            if (in_use_ < capacity_ && selected_waiter_locked() == &waiter) {
                remove_waiter_locked(waiter);
                counters.waiting.fetch_sub(1, std::memory_order_relaxed);
                ++in_use_;
                peak_in_use_ = std::max(peak_in_use_, in_use_);
                ++in_use_by_session_[session_id];
                ++acquires_;
                const int session_in_use =
                        counters.in_use.fetch_add(1, std::memory_order_relaxed) + 1;
                update_atomic_max(counters.peak_in_use, session_in_use);
                counters.acquires.fetch_add(1, std::memory_order_relaxed);
                if (contended) {
                    const long long elapsed = elapsed_ns(started, Clock::now());
                    wait_ns_ += elapsed;
                    counters.wait_ns.fetch_add(elapsed, std::memory_order_relaxed);
                }
                const Clock::time_point acquired_at = Clock::now();
                lock.unlock();
                // More than one permit may be free.  Wake the next selected
                // waiter immediately instead of waiting for this scan to end.
                cv_.notify_all();
                return Lease(this, session_id, &counters, acquired_at);
            }
            cv_.wait(lock);
        }
    }

    void notify_all() noexcept {
        cv_.notify_all();
    }

    ScanPermitSnapshot snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        ScanPermitSnapshot result;
        result.capacity = capacity_;
        result.in_use = in_use_;
        result.peak_in_use = peak_in_use_;
        result.waiting = static_cast<int>(waiters_.size());
        result.peak_waiting = peak_waiting_;
        result.active_sessions = static_cast<int>(active_sessions_locked());
        result.acquires = acquires_;
        result.waits = waits_;
        result.aborted_waits = aborted_waits_;
        result.wait_ns = wait_ns_;
        result.hold_ns = hold_ns_;
        return result;
    }

private:
    struct Waiter {
        std::uint64_t ticket = 0;
        int session_id = 0;
    };

    static long long elapsed_ns(Clock::time_point start, Clock::time_point end) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();
    }

    int in_use_for_session_locked(int session_id) const {
        const auto found = in_use_by_session_.find(session_id);
        return found == in_use_by_session_.end() ? 0 : found->second;
    }

    Waiter* selected_waiter_locked() const {
        if (waiters_.empty()) return nullptr;
        int minimum_in_use = std::numeric_limits<int>::max();
        for (const Waiter* waiter : waiters_) {
            minimum_in_use = std::min(
                    minimum_in_use,
                    in_use_for_session_locked(waiter->session_id));
        }
        // The deque is ticket ordered, so this is FIFO among sessions with the
        // same current allocation and FIFO within a session.
        for (Waiter* waiter : waiters_) {
            if (in_use_for_session_locked(waiter->session_id) == minimum_in_use) {
                return waiter;
            }
        }
        return nullptr;
    }

    void decrement_waiting_session_locked(int session_id) {
        auto found = waiting_by_session_.find(session_id);
        if (found == waiting_by_session_.end() || found->second < 1) {
            std::terminate();
        }
        if (--found->second == 0) waiting_by_session_.erase(found);
    }

    void remove_waiter_locked(Waiter& waiter) {
        const auto found = std::find(waiters_.begin(), waiters_.end(), &waiter);
        if (found == waiters_.end()) std::terminate();
        waiters_.erase(found);
        decrement_waiting_session_locked(waiter.session_id);
    }

    void abort_wait_locked(
            Waiter& waiter,
            ScanPermitCounters& counters,
            Clock::time_point started,
            bool contended) {
        remove_waiter_locked(waiter);
        counters.waiting.fetch_sub(1, std::memory_order_relaxed);
        ++aborted_waits_;
        counters.aborted_waits.fetch_add(1, std::memory_order_relaxed);
        if (contended) {
            const long long elapsed = elapsed_ns(started, Clock::now());
            wait_ns_ += elapsed;
            counters.wait_ns.fetch_add(elapsed, std::memory_order_relaxed);
        }
    }

    std::size_t active_sessions_locked() const {
        std::size_t active = in_use_by_session_.size();
        for (const auto& entry : waiting_by_session_) {
            if (in_use_by_session_.find(entry.first) == in_use_by_session_.end()) {
                ++active;
            }
        }
        return active;
    }

    void release(
            int session_id,
            ScanPermitCounters& counters,
            Clock::time_point acquired_at) noexcept {
        const long long held = elapsed_ns(acquired_at, Clock::now());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (in_use_ < 1) std::terminate();
            auto found = in_use_by_session_.find(session_id);
            if (found == in_use_by_session_.end() || found->second < 1) {
                std::terminate();
            }
            --in_use_;
            if (--found->second == 0) in_use_by_session_.erase(found);
            hold_ns_ += held;
            counters.in_use.fetch_sub(1, std::memory_order_relaxed);
            counters.hold_ns.fetch_add(held, std::memory_order_relaxed);
        }
        cv_.notify_all();
    }

    const int capacity_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Waiter*> waiters_;
    std::unordered_map<int, int> in_use_by_session_;
    std::unordered_map<int, int> waiting_by_session_;
    std::uint64_t next_ticket_ = 0;
    int in_use_ = 0;
    int peak_in_use_ = 0;
    int peak_waiting_ = 0;
    long long acquires_ = 0;
    long long waits_ = 0;
    long long aborted_waits_ = 0;
    long long wait_ns_ = 0;
    long long hold_ns_ = 0;
};

} // namespace harmorag
