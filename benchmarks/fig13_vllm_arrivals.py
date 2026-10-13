"""Figure 13 arrivals matching the installed vLLM 0.11.2 serve benchmark.

Adapted from vLLM 0.11.2's ``vllm/benchmarks/serve.py``,
``get_request`` at source SHA-256 e8d1a0f7502a07ea6e549476fe896f53e73311819493b37eb04b7e0dcfa3c107.
The default burstiness=1 makes gamma intervals exponential. vLLM samples
one interval per request, including the first, then normalizes cumulative
delays to num_requests/request_rate when ramp-up is disabled.
"""

from __future__ import annotations

import hashlib

import numpy as np


def arrivals_for_rate(count: int, rate: float, seed: int) -> np.ndarray:
    if count < 1 or rate <= 0:
        raise ValueError("count and rate must be positive")
    rng = np.random.RandomState(seed)
    gaps = rng.gamma(shape=1.0, scale=1.0 / float(rate), size=count)
    arrivals = np.cumsum(gaps, dtype=np.float64)
    arrivals *= (count / float(rate)) / float(arrivals[-1])
    return arrivals


def trace_sha256(arrivals: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(arrivals, dtype="<f8").tobytes()).hexdigest()
