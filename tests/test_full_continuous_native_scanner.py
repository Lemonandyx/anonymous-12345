#!/usr/bin/env python3
"""Standalone contract test for scanner-wide full continuous batching."""

from __future__ import annotations

import tempfile
from pathlib import Path

import faiss
import numpy as np

import harmorag_native_scanner


def normalized_random(rng: np.random.Generator, rows: int, dim: int) -> np.ndarray:
    values = rng.normal(size=(rows, dim)).astype(np.float32)
    faiss.normalize_L2(values)
    return values


def test_full_continuous_native_scanner() -> None:
    rng = np.random.default_rng(20260831)
    dim = 32
    nlist = 32
    base = normalized_random(rng, 4096, dim)
    queries = normalized_random(rng, 18, dim)

    with tempfile.TemporaryDirectory(prefix="harmorag-full-cb-") as directory:
        index_path = Path(directory) / "tiny_ivf.index"
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(dim),
            dim,
            nlist,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(base)
        index.add(base)
        faiss.write_index(index, str(index_path))

        scanner = harmorag_native_scanner.ContinuousIVFScanner(
            str(index_path),
            worker_threads=4,
            global_scan_permits=4,
            initial_nprobe=4,
            max_nprobe=16,
            heap_topk=10,
            lookahead=4,
            enable_wrd=False,
            use_mmap=False,
        )
        assert scanner.scheduler == "scanner_wide_full_continuous_batching"

        sessions = [
            scanner.start(queries[:9], pause_after_bootstrap=False),
            scanner.start(queries[9:], pause_after_bootstrap=False),
        ]
        cursors = [-1, -1]
        bootstrap_seen = [set(), set()]
        terminal_seen = [set(), set()]

        while any(len(rows) < 9 for rows in terminal_seen):
            progress = False
            for session_index, session in enumerate(sessions):
                if len(terminal_seen[session_index]) == 9:
                    continue
                event = dict(
                    scanner.poll(
                        session,
                        after_sequence=cursors[session_index],
                        timeout_ms=100,
                        latest=False,
                    )
                )
                if not event.get("ready"):
                    continue
                progress = True
                cursors[session_index] = int(event["sequence"])
                query_index = int(event["local_index"])
                budget = int(event["budget"])
                if budget == 4:
                    bootstrap_seen[session_index].add(query_index)
                else:
                    assert len(bootstrap_seen[session_index]) == 9, (
                        session_index,
                        budget,
                        bootstrap_seen[session_index],
                    )
                if event["terminal"]:
                    terminal_seen[session_index].add(query_index)
            if not progress:
                assert any(
                    not bool(scanner.wait(session, 0).get("done"))
                    for session in sessions
                )

        for session_index, session in enumerate(sessions):
            summary = dict(scanner.wait(session, 30_000))
            assert summary["done"] and not summary["error"], summary
            assert bootstrap_seen[session_index] == set(range(9))
            scanner.close(session)

        permits = dict(scanner.permit_stats())
        assert int(permits["capacity"]) == 4
        assert int(permits["peak_in_use"]) <= 4

    print("full continuous native scanner contract: PASS")


if __name__ == "__main__":
    test_full_continuous_native_scanner()
