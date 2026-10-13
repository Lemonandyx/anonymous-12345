"""Contract tests for continuous-scan progressive RAG.

These tests intentionally describe the new public behavior rather than the
retired staged-nprobe/verifier/fallback implementation.  The deterministic fake
retriever exposes continuous scan checkpoints for observability; they are not a
user-configured nprobe schedule.
"""

from types import SimpleNamespace
import queue
import re
import threading
import time
import unittest

from harmorag.pipeline.progressive_pipeline import (
    AppendOnlyEvidencePrompt,
    COMPACT_RAG_SYSTEM_PROMPT,
    COMPACT_RAG_USER_TAIL_TEMPLATE,
    MINIMAL_CONTROL_RAG_SYSTEM_PROMPT,
    MINIMAL_CONTROL_RAG_USER_TAIL_TEMPLATE,
    ProgressiveAsyncPipeline,
    _QueryPipelineState,
    answer_is_grounded,
    is_no_answer,
    parse_answerability_decision,
    parse_quoted_answerability_decision,
    semantic_high_risk_question,
    semantic_quote_support_features,
)


BOOTSTRAP_NPROBE = 4
MAX_NPROBE = 128
FAST_MAX_NPROBE = MAX_NPROBE  # compatibility field; there is no FAST cap
HEAP_TOPK = 100
WRD_TOPK = 100
FINAL_TOPK = 5
WRD_EPSILON = 0.0025
WRD_WINDOW = 5
WRD_BETA = 0.6
WRD_MU = 0.5
WRD_PATIENCE = 20


class _Dataset:
    def __init__(self, questions):
        self.question = list(questions)
        self.outputs = {}

    def update_output(self, key, values):
        if len(values) != len(self.question):
            raise AssertionError(f"{key} is not per-query")
        self.outputs[key] = values


def _output(dataset, *names):
    for name in names:
        if name in dataset.outputs:
            return dataset.outputs[name]
    raise AssertionError(f"none of the compatible output fields exists: {names}")


def _attempted_ids(dataset, query_index=0):
    value = _output(
        dataset,
        "docs_attempted",
        "attempted_document_ids",
        "evidence_doc_ids",
    )[query_index]
    if isinstance(value, int):
        return list(range(value))
    return [int(item) for item in value]


class _ContinuousSession:
    def __init__(self, batch_id, query_indices, pause_after_bootstrap=False):
        self.batch_id = batch_id
        self.query_indices = tuple(query_indices)
        self.cursor = 0
        self.cancelled = set()
        self.fast_stop_budget = None
        self.fast_stop_reason = None
        self.resume_from_cursor = None
        self.stable_streak = 0
        self.paused = bool(pause_after_bootstrap)
        self.bootstrap_emitted = False
        self.draining = False
        self.stable_terminal = False

    @property
    def terminal(self):
        return self.stable_terminal or self.cursor >= MAX_NPROBE or set(self.query_indices).issubset(
            self.cancelled
        )

    @property
    def has_runnable(self):
        return not self.terminal

    @property
    def complete(self):
        return self.terminal


class _ContinuousRetriever:
    """Deterministic continuous scanner with one query-global 100-entry heap."""

    bootstrap_nprobe = BOOTSTRAP_NPROBE
    fast_max_nprobe = FAST_MAX_NPROBE
    max_nprobe = MAX_NPROBE
    heap_topk = HEAP_TOPK
    wrd_topk = WRD_TOPK
    topk = FINAL_TOPK

    def __init__(
        self,
        stable_at=None,
        scan_delay=0.0,
        long_documents=False,
        common_support=False,
    ):
        self.stable_at = stable_at
        self.scan_delay = scan_delay
        self.long_documents = long_documents
        self.common_support = common_support
        self.timestamps = {}
        self.scan_history = {}
        self.sessions = []
        self.resume_calls = 0
        self.close_calls = {}
        self.materialize_calls = []

    def start_batch(
        self,
        queries,
        batch_id,
        query_indices,
        pause_after_bootstrap=True,
    ):
        session = _ContinuousSession(
            batch_id,
            query_indices,
            pause_after_bootstrap=pause_after_bootstrap,
        )
        self.sessions.append(session)
        self.timestamps[f"batch_{batch_id}_start"] = time.monotonic()
        for query_index in query_indices:
            self.scan_history[int(query_index)] = []
        return session

    @staticmethod
    def _next_cursor(cursor):
        if cursor == 0:
            return BOOTSTRAP_NPROBE
        return min(MAX_NPROBE, cursor + 1)

    def advance_batch(self, session):
        if not session.has_runnable:
            raise StopIteration("scanner is paused or terminal")
        if session.paused and session.bootstrap_emitted:
            raise StopIteration("bootstrap-parked scanner was advanced before promotion")
        previous = session.cursor
        current = self._next_cursor(previous)
        # One delayed post-bootstrap quantum is enough for the overlap tests;
        # keeping all 124 one-list advances delayed would make this fake slow.
        if self.scan_delay and previous == BOOTSTRAP_NPROBE:
            time.sleep(self.scan_delay)
        session.cursor = current
        if previous == 0:
            session.bootstrap_emitted = True

        snapshots = []
        for query_index in session.query_indices:
            if query_index in session.cancelled:
                continue
            query_index = int(query_index)
            self.scan_history[query_index].append(current)
            base = query_index * 1000
            heap_ids = tuple(base + rank for rank in range(HEAP_TOPK))
            stable_observation = (
                not session.draining
                and self.stable_at is not None
                and current >= self.stable_at
            )
            session.stable_streak = (
                session.stable_streak + 1 if stable_observation else 0
            )
            stable = session.stable_streak >= WRD_PATIENCE
            if session.fast_stop_budget is None and stable:
                session.fast_stop_budget = current
                session.fast_stop_reason = "stable"
                session.stable_terminal = True
            snapshots.append(
                SimpleNamespace(
                    query_index=query_index,
                    batch_id=session.batch_id,
                    budget=current,
                    nprobe=current,
                    scanned_nprobe=current,
                    actual_scanned_nprobe=current,
                    document_ids=heap_ids[:FINAL_TOPK],
                    heap_document_ids=heap_ids,
                    scores=tuple(1.0 - rank / 1000 for rank in range(FINAL_TOPK)),
                    delta=None if previous == 0 else (0.0 if stable else 0.1),
                    average_delta=None,
                    volatility=None,
                    risk=0.0 if stable else 1.0,
                    stable_streak=session.stable_streak,
                    top1_changed=previous == 0,
                    top1_is_new=previous == 0,
                    top1_document=self.materialize_documents([base])[0]
                    if previous == 0
                    else None,
                    fast_phase_stopped=session.fast_stop_budget == current,
                    fast_stop_budget=session.fast_stop_budget,
                    fast_stop_reason=session.fast_stop_reason,
                    resumed_from_nprobe=session.resume_from_cursor,
                    phase=(
                        "FINISHED"
                        if current == MAX_NPROBE or session.stable_terminal
                        else "DRAINING"
                        if session.draining
                        else "FAST"
                    ),
                    pause_reason=None,
                    scanned_budget=current,
                    finished=current == MAX_NPROBE or session.stable_terminal,
                    stop_reason=(
                        "max_budget"
                        if current == MAX_NPROBE
                        else "wrd_stable"
                        if session.stable_terminal
                        else None
                    ),
                    final_top5_ready=current == MAX_NPROBE,
                )
            )
        self.timestamps[f"batch_{session.batch_id}_nprobe_{current}"] = time.monotonic()
        return SimpleNamespace(
            batch_id=session.batch_id,
            nprobe=current,
            budget=current,
            snapshots=tuple(snapshots),
            batch_complete=not session.has_runnable,
            has_runnable=session.has_runnable,
            terminal=session.terminal,
        )

    def cancel_queries(self, session, query_indices):
        for query_index in query_indices:
            session.cancelled.add(int(query_index))
            self.timestamps[f"query_{int(query_index)}_cancel"] = time.monotonic()

    def resume_queries(self, session, query_indices):
        self.resume_calls += 1
        return ()

    def resume_batch(self, session):
        if session.terminal or not session.paused:
            return False
        self.resume_calls += 1
        self.timestamps[f"batch_{session.batch_id}_resume"] = time.monotonic()
        self.timestamps[f"batch_{session.batch_id}_cursor_before_resume"] = (
            session.cursor
        )
        session.paused = False
        return True

    def close_batch(self, session):
        batch_id = int(session.batch_id)
        self.close_calls[batch_id] = self.close_calls.get(batch_id, 0) + 1
        return {
            int(query_index): int(session.cursor)
            for query_index in session.query_indices
        }

    def materialize_documents(self, document_ids):
        self.materialize_calls.append(tuple(int(value) for value in document_ids))
        documents = []
        for document_id in document_ids:
            suffix = " x" * 4000 if self.long_documents else ""
            support = (
                " independently confirms evidence-0"
                if self.common_support and int(document_id) != 0
                else ""
            )
            documents.append(
                {
                    "id": str(document_id),
                    "contents": f"evidence-{int(document_id)}{support}{suffix}",
                }
            )
        return documents


class _RecordingStopEvent:
    """Deterministic Event double that records scheduler idle waits."""

    def __init__(self):
        self._set = False
        self.waits = []

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def wait(self, timeout=None):
        self.waits.append(timeout)
        self._set = True
        return True


class _RoundPollingNativeRetriever:
    """Two-session native double for production round-poll scheduling tests."""

    use_native_continuous = True

    def __init__(self, stop_event, *, event_on_first=False):
        self.stop_event = stop_event
        self.event_on_first = bool(event_on_first)
        self.event_emitted = False
        self.poll_order = []
        self.pause_after_bootstrap = []
        self.closed_batch_ids = []

    def start_batch(
        self,
        queries,
        batch_id,
        query_indices,
        pause_after_bootstrap=True,
    ):
        self.pause_after_bootstrap.append(bool(pause_after_bootstrap))
        return SimpleNamespace(
            batch_id=int(batch_id),
            query_indices=tuple(int(value) for value in query_indices),
            terminal=False,
            has_runnable=True,
            complete=False,
        )

    def try_advance_batch(self, session, timeout_ms=0):
        self.poll_order.append(int(session.batch_id))
        event = None
        if (
            self.event_on_first
            and not self.event_emitted
            and int(session.batch_id) == 0
        ):
            self.event_emitted = True
            query_index = int(session.query_indices[0])
            snapshot = SimpleNamespace(
                query_index=query_index,
                budget=BOOTSTRAP_NPROBE,
                actual_scanned_nprobe=BOOTSTRAP_NPROBE,
            )
            event = SimpleNamespace(snapshots=(snapshot,))
        # End the event-bearing test after both sessions have been polled.  If
        # the scheduler incorrectly waits after the empty second peer, the
        # recording event will expose that call even though it is already set.
        if self.event_on_first and len(self.poll_order) == 2:
            self.stop_event.set()
        return event

    def cancel_queries(self, session, query_indices):
        return None

    def close_batch(self, session):
        self.closed_batch_ids.append(int(session.batch_id))
        return {int(index): 0 for index in session.query_indices}


class _NoGenerator:
    def __init__(self, first_delay=0.0):
        self.first_delay = first_delay
        self.calls = 0
        self.first_start = None
        self.first_end = None
        self.call_starts = []
        self.prompts = []
        self.params = []

    def prompt_text(self, prompt):
        if isinstance(prompt, dict) and "prompt_token_ids" in prompt:
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None:
                return str(prompt)
            token_ids = [int(value) for value in prompt["prompt_token_ids"]]
            if hasattr(tokenizer, "decode"):
                return tokenizer.decode(token_ids, skip_special_tokens=False)
            # The character tokenizer's reversible test encoding uses 100+ord.
            return "".join(chr(value - 100) for value in token_ids if value >= 100)
        return str(prompt)

    def generate(self, prompts, **params):
        self.calls += 1
        self.call_starts.append(time.monotonic())
        self.prompts.append(list(prompts))
        self.params.append(dict(params))
        if self.calls == 1:
            self.first_start = time.monotonic()
            if self.first_delay:
                time.sleep(self.first_delay)
            self.first_end = time.monotonic()
        return ["No"] * len(prompts)


class _GroundedGenerator(_NoGenerator):
    def generate(self, prompts, **params):
        self.calls += 1
        self.call_starts.append(time.monotonic())
        self.prompts.append(list(prompts))
        self.params.append(dict(params))
        if self.calls == 1:
            self.first_start = time.monotonic()
            if self.first_delay:
                time.sleep(self.first_delay)
            self.first_end = time.monotonic()
        outputs = []
        for prompt in prompts:
            text = str(prompt)
            matches = re.findall(r"evidence-(\d+)", text)
            outputs.append(f"evidence-{matches[-1]}" if matches else "No")
        return outputs


class _ConsistentGroundedGenerator(_GroundedGenerator):
    """Always repeat the first grounded span in the accumulated prefix."""

    def generate(self, prompts, **params):
        self.calls += 1
        self.call_starts.append(time.monotonic())
        self.prompts.append(list(prompts))
        self.params.append(dict(params))
        outputs = []
        for prompt in prompts:
            matches = re.findall(r"evidence-(\d+)", str(prompt))
            outputs.append(f"evidence-{matches[0]}" if matches else "No")
        return outputs


class _GateNoNativeAnswerGenerator(_NoGenerator):
    def __init__(self, native_answer="native-final-answer", first_delay=0.0):
        super().__init__(first_delay=first_delay)
        self.native_answer = native_answer

    def generate(self, prompts, **params):
        self.calls += 1
        self.call_starts.append(time.monotonic())
        self.prompts.append(list(prompts))
        self.params.append(dict(params))
        if self.calls == 1:
            self.first_start = time.monotonic()
            if self.first_delay:
                time.sleep(self.first_delay)
            self.first_end = time.monotonic()
        return [
            self.native_answer
            if "RETRIEVAL_STATUS=FINAL" in self.prompt_text(prompt)
            else "No"
            for prompt in prompts
        ]


class _TrailingGroundedGenerator(_GroundedGenerator):
    def __init__(self):
        super().__init__()
        self.params = []

    def generate(self, prompts, **params):
        outputs = _ConsistentGroundedGenerator.generate(self, prompts, **params)
        return [f"{output}\nThis line must never affect the decision." for output in outputs]


class _TruncatedGroundedGenerator(_GroundedGenerator):
    def generate(self, prompts, **params):
        outputs = super().generate(prompts, **params)
        return outputs, [
            {
                "finish_reason": "length",
                "prompt_tokens": 32,
                "num_cached_tokens": 16,
            }
            for _ in outputs
        ]


class _FailingGenerator(_NoGenerator):
    def generate(self, prompts, **params):
        raise ValueError("generation failed")


class _QuoteGateGenerator(_NoGenerator):
    def __init__(self, output, margin=15.0):
        super().__init__()
        self.output = str(output)
        self.margin = float(margin)

    def generate(self, prompts, **params):
        self.calls += 1
        self.prompts.append(list(prompts))
        self.params.append(dict(params))
        return [self.output for _ in prompts], [
            {
                "finish_reason": "stop",
                "prompt_tokens": 64,
                "num_cached_tokens": 32,
                "answerability_gate": {
                    "logprob_margin": self.margin,
                    "answer_probability": 0.999,
                    "complete": True,
                },
            }
            for _ in prompts
        ]


class _SteppedEventGenerator(_NoGenerator):
    """Small vLLM event-API double with explicitly controlled completions."""

    def __init__(self, active_request_ids=()):
        super().__init__()
        self.active_request_ids = {
            str(request_id) for request_id in active_request_ids
        }
        self.added_batches = []
        self.added_params = []
        self.step_calls = 0
        self._next_request_id = 0

    def add_requests(self, prompts, request_metadata=None, **params):
        prompts = list(prompts)
        metadata = list(request_metadata or ({} for _ in prompts))
        if len(metadata) != len(prompts):
            raise AssertionError("metadata must remain aligned with prompts")
        request_ids = []
        for _ in prompts:
            request_id = f"new-{self._next_request_id}"
            self._next_request_id += 1
            request_ids.append(request_id)
            self.active_request_ids.add(request_id)
        self.added_batches.append((prompts, metadata, request_ids))
        self.added_params.append(dict(params))
        return request_ids

    def step_requests(self):
        self.step_calls += 1
        return []

    def has_unfinished_requests(self):
        return bool(self.active_request_ids)

    def abort_requests(self, request_ids):
        self.active_request_ids.difference_update(
            str(request_id) for request_id in request_ids
        )


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        prefix = [1] if add_special_tokens else []
        return prefix + [100 + ord(character) for character in text]


class _ChatCharacterTokenizer(_CharacterTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize
        parts = ["<BOS>"]
        for message in messages:
            role = str(message["role"]).upper()
            if role == "ASSISTANT":
                role = "ASSIST"
            parts.extend(
                [
                    f"<{role}>",
                    str(message["content"]),
                    "<EOT>",
                ]
            )
        if add_generation_prompt:
            parts.append("<ASSIST>")
        return "".join(parts)


def _pipeline(retriever, generator, queue_size=64):
    """Build a pipeline against the new compatibility field contract."""

    pipeline = ProgressiveAsyncPipeline.__new__(ProgressiveAsyncPipeline)
    pipeline.retriever = retriever
    pipeline.generator = generator
    pipeline.pipeline_batch_size = 1
    pipeline.prefetch_batches = 1
    pipeline.event_queue_size = queue_size
    pipeline.coalesce_ranking_events = False
    pipeline.continuation_priority_scheduler = False
    pipeline.no_answer_sentinel = "No"
    pipeline.bootstrap_nprobe = BOOTSTRAP_NPROBE
    pipeline.m2_first_snapshot_nprobe = BOOTSTRAP_NPROBE
    pipeline.fast_max_nprobe = FAST_MAX_NPROBE
    pipeline.max_nprobe = MAX_NPROBE
    pipeline.heap_topk = HEAP_TOPK
    pipeline.wrd_topk = WRD_TOPK
    pipeline.wrd_alpha = 1.0
    pipeline.wrd_epsilon = WRD_EPSILON
    pipeline.wrd_window = WRD_WINDOW
    pipeline.wrd_beta = WRD_BETA
    pipeline.wrd_mu = WRD_MU
    pipeline.wrd_patience = WRD_PATIENCE
    pipeline.final_topk = FINAL_TOPK
    pipeline.best_unsent_after_no = True
    pipeline.attempt_all_final_topk = True
    pipeline.stop_on_context_full = True
    pipeline.max_evidence_documents = HEAP_TOPK
    pipeline.max_input_tokens = 10000
    pipeline.generation_tokens = 16
    pipeline.max_answer_tokens = 5
    pipeline.final_max_answer_tokens = 10
    pipeline.final_generation_max_tokens = 32
    pipeline.enable_incremental_generation = True
    pipeline.answer_consistency_observations = 2
    pipeline.answer_retention_topk = 5
    pipeline.answer_fast_supporting_documents = 2
    pipeline.answer_stable_supporting_documents = 1
    pipeline.require_grounded_answer = True
    pipeline.store_prompt_trajectory = True
    pipeline.use_token_prompts = False
    pipeline.capture_prefix_cache_stats = True
    pipeline.compact_answerability_prompt = False
    pipeline.single_snapshot_gate = False
    pipeline.semantic_quote_gate = False
    pipeline.semantic_quote_gate_shadow_only = True
    pipeline.search_document_limit = FINAL_TOPK
    pipeline.search_gate_ordinals = tuple(range(1, FINAL_TOPK))
    pipeline.bulk_final_on_retrieval_terminal = False
    pipeline._configure_evidence_prompt_frame()
    pipeline.evaluate = lambda dataset, do_eval=True, pred_process_fun=None: dataset
    return pipeline


def _enable_semantic_quote_gate(pipeline, *, shadow_only):
    pipeline.compact_answerability_prompt = True
    pipeline.single_snapshot_gate = True
    pipeline.answerability_logprob_margin = 0.0
    pipeline.answerability_gate_logprobs = 100
    pipeline.answerability_require_logprobs = True
    pipeline.answerability_retention_topk = 5
    pipeline.answerability_structured_output = True
    pipeline.semantic_quote_gate = True
    pipeline.semantic_quote_gate_shadow_only = bool(shadow_only)
    pipeline.semantic_quote_margin = 14.0
    pipeline.semantic_quote_snapshot_rank_max = 2
    pipeline.semantic_quote_max_answer_words = 3
    pipeline.semantic_quote_max_span_words = 32
    pipeline.semantic_quote_min_question_coverage = 0.6
    pipeline.semantic_quote_require_entity_anchor = True
    pipeline.semantic_quote_require_relation_anchor = True
    pipeline.semantic_quote_exclude_high_risk = True
    pipeline.semantic_quote_generation_max_tokens = 48
    pipeline._system_prompt_template = COMPACT_RAG_SYSTEM_PROMPT
    pipeline._user_tail_template = COMPACT_RAG_USER_TAIL_TEMPLATE
    pipeline._configure_evidence_prompt_frame()


class PromptContractTest(unittest.TestCase):
    def test_grounding_and_no_sentinel(self):
        self.assertTrue(answer_is_grounded("Wilhelm Röntgen", ["Prize: Wilhelm Röntgen."]))
        self.assertFalse(answer_is_grounded("York", ["Yorkshire"]))
        self.assertTrue(is_no_answer(" `No.` "))
        self.assertFalse(is_no_answer("Nobel"))
        control = "[INSUFFICIENT_EVIDENCE]"
        self.assertTrue(is_no_answer(" `[INSUFFICIENT_EVIDENCE].` ", control))
        self.assertFalse(is_no_answer("No", control))
        self.assertEqual(parse_answerability_decision("A No"), ("A", "No"))

    def test_quoted_answerability_protocol_fails_closed(self):
        self.assertEqual(
            parse_quoted_answerability_decision(
                "A Billy Hill || written by Billy Hill"
            ),
            ("A", "Billy Hill", "written by Billy Hill"),
        )
        self.assertEqual(
            parse_quoted_answerability_decision("N"), ("N", "", "")
        )
        self.assertEqual(
            parse_quoted_answerability_decision("A Billy Hill"),
            ("INVALID", "", ""),
        )

    def test_semantic_quote_features_use_verbatim_online_evidence(self):
        document = (
            'The Glory of Love\n"The Glory of Love" is a song written by '
            "Billy Hill, recorded by Benny Goodman."
        )
        quote = '"The Glory of Love" is a song written by Billy Hill'
        features = semantic_quote_support_features(
            "who wrote the song the glory of love",
            "Billy Hill",
            quote,
            document,
        )

        self.assertTrue(features["quote_is_verbatim"])
        self.assertTrue(features["answer_in_quote"])
        self.assertGreaterEqual(features["question_window_coverage"], 0.6)
        self.assertTrue(features["entity_anchor_present"])
        self.assertTrue(
            features["relation_anchor_present"] is not False
        )
        self.assertTrue(
            semantic_high_risk_question(
                "who is the owner of reading football club"
            )
        )
        self.assertFalse(
            semantic_high_risk_question(
                "who wrote the song the glory of love"
            )
        )

    def test_exact_append_only_token_prefix(self):
        pipeline = ProgressiveAsyncPipeline.__new__(ProgressiveAsyncPipeline)
        pipeline.generator = SimpleNamespace(tokenizer=_CharacterTokenizer())
        pipeline.use_token_prompts = True
        pipeline.no_answer_sentinel = "No"
        pipeline.max_input_tokens = 10000
        pipeline.generation_tokens = 8
        pipeline.store_prompt_trajectory = False

        chain = AppendOnlyEvidencePrompt("Q", max_documents=5)
        state = _QueryPipelineState(question="Q", prompt_chain=chain)
        first = SimpleNamespace(
            budget=BOOTSTRAP_NPROBE,
            document_ids=(1,),
            top1_document={"id": "1", "contents": "first"},
        )
        state.pending_documents.append((1, first.top1_document, first))
        _, first_prompts = pipeline._ready_generation_batch([state])
        first_evidence_ids = tuple(state.evidence_token_ids)
        suffix_ids = tuple(
            pipeline._encode_text(
                pipeline._render_generation_prompt("", question="Q"),
                add_special_tokens=False,
            )
        )
        self.assertEqual(
            tuple(first_prompts[0]["prompt_token_ids"]),
            first_evidence_ids + suffix_ids,
        )

        second = SimpleNamespace(
            budget=FAST_MAX_NPROBE,
            document_ids=(2,),
            top1_document={"id": "2", "contents": "second"},
        )
        state.pending_documents.append((2, second.top1_document, second))
        _, second_prompts = pipeline._ready_generation_batch([state])

        self.assertEqual(
            tuple(second_prompts[0]["prompt_token_ids"][: len(first_evidence_ids)]),
            first_evidence_ids,
        )
        self.assertEqual(
            tuple(state.evidence_token_ids[: len(first_evidence_ids)]),
            first_evidence_ids,
        )
        self.assertEqual(
            tuple(second_prompts[0]["prompt_token_ids"][-len(suffix_ids) :]),
            suffix_ids,
        )
        self.assertNotEqual(
            tuple(second_prompts[0]["prompt_token_ids"][: len(first_prompts[0]["prompt_token_ids"])]),
            tuple(first_prompts[0]["prompt_token_ids"]),
        )
        self.assertGreater(
            len(second_prompts[0]["prompt_token_ids"]),
            len(first_prompts[0]["prompt_token_ids"]),
        )

    @staticmethod
    def _conversational_prefix_pipeline():
        pipeline = ProgressiveAsyncPipeline.__new__(ProgressiveAsyncPipeline)
        pipeline.generator = SimpleNamespace(tokenizer=_ChatCharacterTokenizer())
        pipeline.use_token_prompts = True
        pipeline.conversational_prefix = True
        pipeline.compact_answerability_prompt = True
        pipeline.single_snapshot_gate = True
        pipeline.semantic_quote_gate = False
        pipeline.no_answer_sentinel = "No"
        pipeline.max_input_tokens = 100000
        pipeline.generation_tokens = 8
        pipeline.final_generation_max_tokens = 32
        pipeline.store_prompt_trajectory = True
        pipeline.search_document_limit = FINAL_TOPK
        pipeline.bootstrap_nprobe = BOOTSTRAP_NPROBE
        pipeline.max_evidence_documents = 0
        pipeline.final_topk = FINAL_TOPK
        pipeline.enable_incremental_generation = True
        pipeline.require_grounded_answer = True
        pipeline.max_answer_tokens = 5
        pipeline.answerability_logprob_margin = 16.0
        pipeline.answerability_require_logprobs = True
        pipeline.answerability_retention_topk = FINAL_TOPK
        pipeline.semantic_quote_gate_shadow_only = False
        return pipeline

    @staticmethod
    def _conversation_snapshot(budget, document_ids):
        document_ids = tuple(int(value) for value in document_ids)
        return SimpleNamespace(
            budget=int(budget),
            scanned_budget=int(budget),
            document_ids=document_ids,
            heap_document_ids=document_ids,
            scores=tuple(
                1.0 - (index * 0.01)
                for index in range(len(document_ids))
            ),
        )

    def test_conversational_request_contains_previous_full_prompt_prefix(self):
        pipeline = self._conversational_prefix_pipeline()
        state = _QueryPipelineState(
            question="Who?",
            prompt_chain=AppendOnlyEvidencePrompt("Who?", max_documents=0),
        )
        first_snapshot = self._conversation_snapshot(
            BOOTSTRAP_NPROBE, (11,)
        )
        state.bootstrap_snapshot = first_snapshot
        state.bootstrap_document_id = 11
        state.pending_documents.append(
            (
                11,
                {"contents": "First title\nfirst evidence"},
                first_snapshot,
            )
        )

        first_indices, first_prompts = pipeline._ready_generation_batch([state])
        self.assertEqual(first_indices, [0])
        first_ids = list(first_prompts[0]["prompt_token_ids"])
        self.assertEqual(state.first_search_generation_budget, BOOTSTRAP_NPROBE)
        pipeline._complete_conversation_assistant(state, "N")

        second_snapshot = self._conversation_snapshot(9, (22, 11))
        state.pending_documents.append(
            (
                22,
                {"contents": "Second title\nsecond evidence"},
                second_snapshot,
            )
        )
        second_indices, second_prompts = pipeline._ready_generation_batch([state])
        self.assertEqual(second_indices, [0])
        second_ids = list(second_prompts[0]["prompt_token_ids"])

        self.assertEqual(second_ids[: len(first_ids)], first_ids)
        self.assertEqual(state.conversation_lcp_tokens[-1], len(first_ids))
        self.assertTrue(all(state.conversation_prefix_checks))
        self.assertEqual(state.conversation_document_ids, [11, 22])
        second_text = pipeline.generator.tokenizer.apply_chat_template(
            state.conversation_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertIn("<ASSIST><EOT><USER>", second_text)
        self.assertNotIn("<ASSIST>N<EOT><USER>", second_text)
        self.assertEqual(state.conversation_assistant_outputs, ["N"])
        self.assertEqual(second_text.count("first evidence"), 1)
        self.assertEqual(second_text.count("second evidence"), 1)
        self.assertEqual(second_text.count("Question: Who?"), 1)
        self.assertEqual(second_text.count("SEARCHING decision:"), 2)

    def test_conversational_fifth_document_is_direct_final_prefix(self):
        pipeline = self._conversational_prefix_pipeline()
        state = _QueryPipelineState(
            question="Who?",
            prompt_chain=AppendOnlyEvidencePrompt("Who?", max_documents=0),
        )
        cancelled = set()
        lock = threading.Lock()
        fourth_prompt_ids = None

        for document_id in range(1, FINAL_TOPK + 1):
            snapshot = self._conversation_snapshot(
                BOOTSTRAP_NPROBE if document_id == 1 else 8 + document_id,
                tuple(range(1, document_id + 1)),
            )
            state.latest_snapshot = snapshot
            state.candidate_ids = snapshot.document_ids
            state.candidate_scores = snapshot.scores
            if document_id == 1:
                state.bootstrap_snapshot = snapshot
                state.bootstrap_document_id = 1
            state.pending_documents.append(
                (
                    document_id,
                    {
                        "contents": (
                            f"Title {document_id}\nevidence-{document_id}"
                        )
                    },
                    snapshot,
                )
            )
            indices, prompts = pipeline._ready_generation_batch(
                [state], cancelled, lock
            )
            if document_id < FINAL_TOPK:
                self.assertEqual(indices, [0])
                self.assertEqual(len(prompts), 1)
                fourth_prompt_ids = list(prompts[0]["prompt_token_ids"])
                pipeline._complete_conversation_assistant(state, "N")
            else:
                self.assertEqual(indices, [])
                self.assertEqual(prompts, [])

        self.assertTrue(state.final_generation_pending)
        self.assertEqual(len(state.search_generation_budgets), FINAL_TOPK)
        final_prompt = pipeline._build_final_prompt(state)
        final_ids = list(final_prompt["prompt_token_ids"])
        self.assertEqual(
            final_ids[: len(fourth_prompt_ids)], fourth_prompt_ids
        )
        self.assertEqual(state.conversation_lcp_tokens[-1], len(fourth_prompt_ids))
        self.assertEqual(state.conversation_document_ids, [1, 2, 3, 4, 5])
        self.assertEqual(state.final_prompt_document_ids, (1, 2, 3, 4, 5))
        self.assertEqual(state.last_prompt_status, "FINAL")
        self.assertIn("FINAL\nDoc 5", state.final_prompt)
        self.assertNotIn("SEARCHING\nDoc 5", state.final_prompt)
        self.assertTrue(state.final_prompt_token_prefix_aligned)

    def test_conversational_low_confidence_abort_can_keep_exact_prefix(self):
        pipeline = self._conversational_prefix_pipeline()
        pipeline.answerability_early_abort_below_margin = True
        pipeline.answerability_structured_output = True
        pipeline.answerability_require_logprobs = True
        pipeline.semantic_quote_gate = False
        pipeline.capture_prefix_cache_stats = False
        kwargs = pipeline._search_generation_kwargs()
        self.assertNotIn("gate_abort_margin", kwargs)

        # The event submitter attaches the abort margin; an aborted A is the
        # actual assistant prefix and must remain safe to replay in round two.
        state = _QueryPipelineState(
            question="Who?",
            prompt_chain=AppendOnlyEvidencePrompt("Who?", max_documents=0),
        )
        first = self._conversation_snapshot(BOOTSTRAP_NPROBE, (11,))
        state.bootstrap_snapshot = first
        state.bootstrap_document_id = 11
        state.pending_documents.append(
            (11, {"contents": "First title\nfirst evidence"}, first)
        )
        _, prompts = pipeline._ready_generation_batch([state])
        first_ids = list(prompts[0]["prompt_token_ids"])
        pipeline._complete_conversation_assistant(state, "A")
        second = self._conversation_snapshot(9, (22, 11))
        state.pending_documents.append(
            (22, {"contents": "Second title\nsecond evidence"}, second)
        )
        _, prompts = pipeline._ready_generation_batch([state])
        self.assertEqual(
            prompts[0]["prompt_token_ids"][: len(first_ids)], first_ids
        )

    def test_prefix_cache_audit_separates_logical_lcp_from_physical_miss(self):
        pipeline = ProgressiveAsyncPipeline.__new__(ProgressiveAsyncPipeline)
        pipeline.prefix_cache_block_size = 16
        state = _QueryPipelineState(
            question="Q",
            prompt_chain=AppendOnlyEvidencePrompt("Q", max_documents=5),
        )
        first = {"prompt_token_ids": list(range(40))}
        pipeline._stage_prefix_cache_audit(state, first, "SEARCHING")
        first_row = pipeline._complete_prefix_cache_audit(
            state, {"num_cached_tokens": 0, "prompt_tokens": 40}
        )
        self.assertEqual(first_row["logical_chain_expected_cached_tokens"], 0)

        second = {"prompt_token_ids": list(range(32)) + list(range(100, 124))}
        pipeline._stage_prefix_cache_audit(state, second, "SEARCHING")
        second_row = pipeline._complete_prefix_cache_audit(
            state, {"num_cached_tokens": 16, "prompt_tokens": 56}
        )
        self.assertEqual(second_row["logical_chain_lcp_tokens"], 32)
        self.assertEqual(
            second_row["logical_chain_expected_cached_tokens"], 32
        )
        self.assertEqual(second_row["logical_chain_reused_tokens"], 16)
        self.assertEqual(
            second_row["logical_chain_cache_shortfall_tokens"], 16
        )
        self.assertEqual(second_row["logical_chain_reuse_efficiency"], 0.5)

    def test_ranking_burst_keeps_latest_snapshot_per_query(self):
        pipeline = ProgressiveAsyncPipeline.__new__(ProgressiveAsyncPipeline)
        pipeline.coalesce_ranking_events = True
        consumed = []
        pipeline._consume_event = lambda event, states: consumed.append(event)
        query0_at5 = SimpleNamespace(query_index=0, budget=5)
        query1_terminal = SimpleNamespace(query_index=1, budget=5)
        query0_at6 = SimpleNamespace(query_index=0, budget=6)
        events = [
            SimpleNamespace(
                batch_id=7,
                budget=5,
                snapshots=(query0_at5, query1_terminal),
                batch_complete=False,
                has_runnable=True,
                terminal=False,
            ),
            SimpleNamespace(
                batch_id=7,
                budget=6,
                snapshots=(query0_at6,),
                batch_complete=False,
                has_runnable=True,
                terminal=False,
            ),
        ]
        pipeline._consume_ranking_event_burst(events, [object(), object()])
        self.assertEqual(len(consumed), 1)
        by_query = {
            snapshot.query_index: snapshot for snapshot in consumed[0].snapshots
        }
        self.assertIs(by_query[0], query0_at6)
        self.assertIs(by_query[1], query1_terminal)
        self.assertEqual(pipeline._ranking_events_received, 2)
        self.assertEqual(pipeline._ranking_events_dispatched, 1)

    def test_ranking_burst_default_does_not_coalesce_without_init(self):
        pipeline = ProgressiveAsyncPipeline.__new__(ProgressiveAsyncPipeline)
        consumed = []
        pipeline._consume_event = lambda event, states: consumed.append(event)
        events = [
            SimpleNamespace(batch_id=7, budget=5, snapshots=()),
            SimpleNamespace(batch_id=7, budget=6, snapshots=()),
        ]

        pipeline._consume_ranking_event_burst(events, [])

        self.assertEqual(consumed, events)
        self.assertEqual(pipeline._ranking_events_received, 2)
        self.assertEqual(pipeline._ranking_events_dispatched, 2)

    def test_conversational_answer_uses_single_snapshot_without_confirmation(self):
        pipeline = self._conversational_prefix_pipeline()
        pipeline._evaluate_consistency_candidate = lambda *args, **kwargs: (
            self.fail("cross-snapshot consistency must not run")
        )
        snapshot = self._conversation_snapshot(BOOTSTRAP_NPROBE, (11,))
        state = _QueryPipelineState(
            question="Who?",
            prompt_chain=AppendOnlyEvidencePrompt("Who?", max_documents=0),
        )
        state.bootstrap_snapshot = snapshot
        state.bootstrap_document_id = 11
        state.latest_snapshot = snapshot
        state.candidate_ids = snapshot.document_ids
        state.candidate_scores = snapshot.scores
        state.pending_documents.append(
            (
                11,
                {"contents": "First title\nevidence-11"},
                snapshot,
            )
        )
        indices, prompts = pipeline._ready_generation_batch([state])
        self.assertEqual(indices, [0])
        self.assertEqual(len(prompts), 1)
        state.generation_inflight = True
        cancelled = set()
        pipeline._handle_event_search_completion(
            [state],
            0,
            "A evidence-11",
            {
                "finish_reason": "stop",
                "answerability_gate": {
                    "complete": True,
                    "logprob_margin": 20.0,
                    "answer_probability": 0.999,
                },
            },
            cancelled,
            threading.Lock(),
        )

        self.assertIsNone(state.answer)
        self.assertEqual(state.pending_single_snapshot_answer, "evidence-11")
        pipeline._commit_pending_single_snapshot_answers(
            [state], cancelled, threading.Lock()
        )

        self.assertEqual(state.answer, "evidence-11")
        self.assertEqual(state.answer_source, "early_single_snapshot_confident")
        self.assertEqual(state.answer_accept_budget, BOOTSTRAP_NPROBE)
        self.assertEqual(cancelled, {0})
        self.assertFalse(state.conversation_pending_assistant)

    def test_single_snapshot_commit_rechecks_latest_heap_without_second_decode(self):
        pipeline = self._conversational_prefix_pipeline()
        snapshot = self._conversation_snapshot(BOOTSTRAP_NPROBE, (11,))
        state = _QueryPipelineState(
            question="Who?",
            prompt_chain=AppendOnlyEvidencePrompt("Who?", max_documents=0),
        )
        state.bootstrap_snapshot = snapshot
        state.bootstrap_document_id = 11
        state.latest_snapshot = snapshot
        state.candidate_ids = snapshot.document_ids
        state.candidate_scores = snapshot.scores
        state.pending_documents.append(
            (
                11,
                {"contents": "First title\nevidence-11"},
                snapshot,
            )
        )
        pipeline._ready_generation_batch([state])
        state.generation_inflight = True
        cancelled = set()
        pipeline._handle_event_search_completion(
            [state],
            0,
            "A evidence-11",
            {
                "finish_reason": "stop",
                "answerability_gate": {
                    "complete": True,
                    "logprob_margin": 20.0,
                    "answer_probability": 0.999,
                },
            },
            cancelled,
            threading.Lock(),
        )

        newer = self._conversation_snapshot(9, (22, 33, 44, 55, 66))
        state.latest_snapshot = newer
        state.candidate_ids = newer.document_ids
        state.candidate_scores = newer.scores
        pipeline._commit_pending_single_snapshot_answers(
            [state], cancelled, threading.Lock()
        )

        self.assertIsNone(state.answer)
        self.assertTrue(state.need_document)
        self.assertEqual(cancelled, set())
        self.assertEqual(
            state.answer_verification_trace[-1]["reason"],
            "support_document_left_retained_heap_during_decode",
        )

    def test_chat_template_suffix_is_not_persisted_between_documents(self):
        pipeline = ProgressiveAsyncPipeline.__new__(ProgressiveAsyncPipeline)
        pipeline.generator = SimpleNamespace(tokenizer=_ChatCharacterTokenizer())
        pipeline.no_answer_sentinel = "No"
        pipeline.max_evidence_documents = 5
        pipeline._configure_evidence_prompt_frame()

        chain = pipeline._new_prompt_chain("Who?")
        chain.append_document(1, {"contents": "first evidence"})
        first_evidence_prefix = chain.prompt
        first_request = pipeline._render_generation_prompt(
            first_evidence_prefix, question="Who?"
        )
        chain.append_document(2, {"contents": "second evidence"})
        second_request = pipeline._render_generation_prompt(
            chain.prompt, question="Who?"
        )

        self.assertTrue(second_request.startswith(first_evidence_prefix))
        self.assertFalse(second_request.startswith(first_request))
        self.assertEqual(first_request.count("<ASSIST>"), 1)
        self.assertEqual(second_request.count("<ASSIST>"), 1)
        self.assertNotIn("Answer:\n\n<DOC", second_request)
        self.assertNotIn("Answer:\nDoc", second_request)
        self.assertIn("Doc 1(Title: first evidence)", second_request)
        self.assertIn("Doc 2(Title: second evidence)", second_request)
        self.assertIn("Return only one concise answer span", second_request)
        self.assertIn("Question: Who?", second_request)
        self.assertLess(
            second_request.index("Question: Who?"),
            second_request.index("first evidence"),
        )
        self.assertNotIn("Question: Who?", pipeline._generation_suffix_template)

    def test_searching_and_final_are_one_template_with_only_status_changed(self):
        pipeline = _pipeline(_ContinuousRetriever(), _NoGenerator())
        chain = pipeline._new_prompt_chain("Who?")
        chain.append_document(7, {"contents": "evidence-7"})
        searching = pipeline._render_generation_prompt(
            chain.prompt, "SEARCHING", "Who?"
        )
        final = pipeline._render_generation_prompt(chain.prompt, "FINAL", "Who?")

        self.assertEqual(
            searching.replace("RETRIEVAL_STATUS=SEARCHING", "RETRIEVAL_STATUS=FINAL"),
            final,
        )
        self.assertIn("If the evidence is insufficient, output exactly No", searching)
        self.assertIn("never output No", final)

    def test_compact_final_is_stock_flashrag_prompt_layout(self):
        pipeline = ProgressiveAsyncPipeline.__new__(ProgressiveAsyncPipeline)
        tokenizer = _ChatCharacterTokenizer()
        pipeline.generator = SimpleNamespace(tokenizer=tokenizer)
        pipeline.no_answer_sentinel = "No"
        pipeline.max_evidence_documents = 5
        pipeline.compact_answerability_prompt = True
        pipeline._system_prompt_template = COMPACT_RAG_SYSTEM_PROMPT
        pipeline._user_tail_template = COMPACT_RAG_USER_TAIL_TEMPLATE
        pipeline._configure_evidence_prompt_frame()

        chain = pipeline._new_prompt_chain("Who?")
        chain.append_document(
            7, {"contents": "Example title\nExample evidence text."}
        )
        final = pipeline._render_generation_prompt(
            chain.prompt, "FINAL", "Who?"
        )
        stock = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        COMPACT_RAG_SYSTEM_PROMPT
                        + "Doc 1(Title: Example title) Example evidence text.\n"
                    ),
                },
                {"role": "user", "content": "Question: Who?\nAnswer:"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        searching = pipeline._render_generation_prompt(
            chain.prompt, "SEARCHING", "Who?"
        )

        self.assertEqual(final, stock)
        self.assertIn("Retrieval is still searching", searching)
        self.assertNotIn("Retrieval is still searching", final)

    def test_minimal_control_tail_preserves_native_system_and_final(self):
        pipeline = ProgressiveAsyncPipeline.__new__(ProgressiveAsyncPipeline)
        tokenizer = _ChatCharacterTokenizer()
        pipeline.generator = SimpleNamespace(tokenizer=tokenizer)
        pipeline.no_answer_sentinel = "No"
        pipeline.max_evidence_documents = 5
        pipeline.compact_answerability_prompt = True
        pipeline.minimal_control_tail = True
        pipeline._system_prompt_template = MINIMAL_CONTROL_RAG_SYSTEM_PROMPT
        pipeline._user_tail_template = MINIMAL_CONTROL_RAG_USER_TAIL_TEMPLATE
        pipeline._configure_evidence_prompt_frame()

        chain = pipeline._new_prompt_chain("Who?")
        chain.append_document(7, {"contents": "First title\nfirst evidence"})
        first = pipeline._render_generation_prompt(
            chain.prompt, "SEARCHING", "Who?"
        )
        chain.append_document(8, {"contents": "Second title\nsecond evidence"})
        second = pipeline._render_generation_prompt(
            chain.prompt, "SEARCHING", "Who?"
        )
        final = pipeline._render_generation_prompt(
            chain.prompt, "FINAL", "Who?"
        )

        first_evidence_end = first.index("<EOT><USER>")
        self.assertEqual(second[:first_evidence_end], first[:first_evidence_end])
        self.assertTrue(first.startswith("<BOS><SYSTEM>" + COMPACT_RAG_SYSTEM_PROMPT))
        self.assertIn("Question: Who?", first[first.index("first evidence") :])
        self.assertIn("Question: {question}", pipeline._generation_suffix_template)
        self.assertEqual(first.count("Question: Who?"), 1)
        self.assertEqual(second.count("Question: Who?"), 1)
        self.assertNotIn("<ASSIST>N", second)
        self.assertIn("Retrieval is searching. Use only the newest Doc.", second)
        self.assertNotIn("Retrieval is still searching. Answer only", second)
        stock_final = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        COMPACT_RAG_SYSTEM_PROMPT
                        + "Doc 1(Title: First title) first evidence\n"
                        + "Doc 2(Title: Second title) second evidence\n"
                    ),
                },
                {"role": "user", "content": "Question: Who?\nAnswer:"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertEqual(final, stock_final)

    def test_quote_protocol_changes_searching_but_not_stock_final(self):
        pipeline = ProgressiveAsyncPipeline.__new__(ProgressiveAsyncPipeline)
        tokenizer = _ChatCharacterTokenizer()
        pipeline.generator = SimpleNamespace(tokenizer=tokenizer)
        pipeline.no_answer_sentinel = "No"
        pipeline.max_evidence_documents = 5
        pipeline.compact_answerability_prompt = True
        pipeline.semantic_quote_gate = True
        pipeline._system_prompt_template = COMPACT_RAG_SYSTEM_PROMPT
        pipeline._user_tail_template = COMPACT_RAG_USER_TAIL_TEMPLATE
        pipeline._configure_evidence_prompt_frame()
        chain = pipeline._new_prompt_chain("Who?")
        chain.append_document(
            7, {"contents": "Example title\nExample evidence text."}
        )

        searching = pipeline._render_generation_prompt(
            chain.prompt, "SEARCHING", "Who?"
        )
        final = pipeline._render_generation_prompt(
            chain.prompt, "FINAL", "Who?"
        )
        stock = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        COMPACT_RAG_SYSTEM_PROMPT
                        + "Doc 1(Title: Example title) Example evidence text.\n"
                    ),
                },
                {"role": "user", "content": "Question: Who?\nAnswer:"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        self.assertIn("A <answer> || <verbatim minimal support span", searching)
        self.assertEqual(final, stock)


class ContinuousPipelineContractTest(unittest.TestCase):
    @staticmethod
    def _terminal_bulk_state(pipeline, *, generation_inflight=False):
        snapshot = SimpleNamespace(
            budget=MAX_NPROBE,
            scanned_budget=MAX_NPROBE,
            document_ids=(0, 1, 2, 3, 4),
            heap_document_ids=(0, 1, 2, 3, 4),
            scores=(1.0, 0.9, 0.8, 0.7, 0.6),
        )
        state = _QueryPipelineState(
            question="q",
            prompt_chain=pipeline._new_prompt_chain("q"),
        )
        self_document = {"contents": "evidence-0"}
        state.prompt_chain.append_document(0, self_document)
        state.evidence_texts.append(state.prompt_chain.document_texts[-1])
        state.sent_document_ids.add(0)
        state.search_document_ids.append(0)
        state.search_generation_budgets.append(BOOTSTRAP_NPROBE)
        state.latest_snapshot = snapshot
        state.candidate_ids = snapshot.heap_document_ids
        state.candidate_scores = snapshot.scores
        state.generation_inflight = bool(generation_inflight)
        state.need_document = not state.generation_inflight
        return state, snapshot

    def test_terminal_bulk_final_appends_all_remaining_docs_without_searching(self):
        generator = _GateNoNativeAnswerGenerator()
        generator.tokenizer = _CharacterTokenizer()
        retriever = _ContinuousRetriever()
        pipeline = _pipeline(retriever, generator)
        pipeline.use_token_prompts = True
        pipeline.bulk_final_on_retrieval_terminal = True
        state, snapshot = self._terminal_bulk_state(pipeline)
        state.last_search_rejected = True
        cancelled = set()
        lock = threading.Lock()

        self.assertTrue(pipeline._mark_terminal(state, snapshot, "wrd_stable"))
        self.assertTrue(state.terminal_bulk_final_armed)
        self.assertTrue(
            pipeline._queue_ready_documents([state], cancelled, lock)
        )
        self.assertEqual(
            [document_id for document_id, _, _ in state.pending_documents],
            [1, 2, 3, 4],
        )
        self.assertEqual(retriever.materialize_calls[-1], (1, 2, 3, 4))

        self.assertTrue(pipeline._generate_once([state], cancelled, lock))

        self.assertEqual(generator.calls, 1)
        self.assertEqual(state.answer, "native-final-answer")
        self.assertEqual(state.answer_source, "final_document_limit")
        self.assertEqual(state.prompt_chain.document_ids, [0, 1, 2, 3, 4])
        self.assertEqual(state.terminal_bulk_final_appended_documents, 4)
        self.assertEqual(
            [row["reason"] for row in state.answerability_skip_trace],
            ["terminal_bulk_final_append"] * 3,
        )
        self.assertIn(
            "RETRIEVAL_STATUS=FINAL",
            generator.prompt_text(generator.prompts[0][0]),
        )
        self.assertEqual(cancelled, {0})

    def test_terminal_during_search_decode_arms_bulk_after_rejected_completion(self):
        pipeline = _pipeline(_ContinuousRetriever(), _GateNoNativeAnswerGenerator())
        pipeline.bulk_final_on_retrieval_terminal = True
        state, snapshot = self._terminal_bulk_state(
            pipeline, generation_inflight=True
        )

        self.assertTrue(pipeline._mark_terminal(state, snapshot, "max_nprobe"))
        self.assertFalse(state.terminal_bulk_final_armed)
        state.generation_inflight = False

        self.assertTrue(
            pipeline._record_rejected_search_for_terminal_bulk_final(
                state, "model_no"
            )
        )
        self.assertTrue(state.terminal_bulk_final_armed)
        self.assertEqual(state.terminal_bulk_final_arm_budget, MAX_NPROBE)
        self.assertEqual(state.terminal_bulk_final_arm_reason, "model_no")

    def test_rejected_search_does_not_bulk_append_while_retrieval_is_live(self):
        retriever = _ContinuousRetriever()
        pipeline = _pipeline(retriever, _GateNoNativeAnswerGenerator())
        pipeline.bulk_final_on_retrieval_terminal = True
        state, snapshot = self._terminal_bulk_state(pipeline)
        snapshot.budget = BOOTSTRAP_NPROBE + 1
        snapshot.scanned_budget = BOOTSTRAP_NPROBE + 1
        state.need_document = True

        self.assertFalse(
            pipeline._record_rejected_search_for_terminal_bulk_final(
                state, "model_no"
            )
        )
        self.assertFalse(state.terminal_bulk_final_armed)
        self.assertTrue(
            pipeline._queue_ready_documents([state], set(), threading.Lock())
        )
        self.assertEqual(len(state.pending_documents), 1)
        self.assertEqual(state.pending_documents[0][0], 1)
        self.assertEqual(retriever.materialize_calls[-1], (1,))

    @staticmethod
    def _quote_gate_state(pipeline):
        question = "who wrote the song the glory of love"
        document = {
            "contents": (
                'The Glory of Love\n"The Glory of Love" is a song written '
                "by Billy Hill, recorded by Benny Goodman."
            )
        }
        snapshot = SimpleNamespace(
            budget=BOOTSTRAP_NPROBE,
            scanned_budget=BOOTSTRAP_NPROBE,
            document_ids=(7, 8, 9, 10, 11),
            heap_document_ids=(7, 8, 9, 10, 11),
            scores=(1.0, 0.9, 0.8, 0.7, 0.6),
        )
        state = _QueryPipelineState(
            question=question,
            prompt_chain=pipeline._new_prompt_chain(question),
        )
        state.latest_snapshot = snapshot
        state.candidate_ids = snapshot.heap_document_ids
        state.pending_documents.append((7, document, snapshot))
        return state

    def test_semantic_quote_shadow_pass_never_changes_answer(self):
        output = (
            'A Billy Hill || "The Glory of Love" is a song written by Billy Hill'
        )
        generator = _QuoteGateGenerator(output)
        pipeline = _pipeline(_ContinuousRetriever(), generator)
        _enable_semantic_quote_gate(pipeline, shadow_only=True)
        state = self._quote_gate_state(pipeline)
        cancelled = set()

        self.assertTrue(
            pipeline._generate_once([state], cancelled, threading.Lock())
        )

        self.assertIsNone(state.answer)
        self.assertEqual(cancelled, set())
        self.assertTrue(state.need_document)
        self.assertEqual(
            state.answer_verification_trace[-1]["reason"],
            "semantic_quote_shadow_pass",
        )
        semantic = state.answerability_gate_trace[-1]["semantic_quote_gate"]
        self.assertTrue(semantic["passed"])
        self.assertTrue(semantic["shadow_only"])
        self.assertEqual(semantic["snapshot_rank"], 1)
        self.assertEqual(generator.params[0]["max_tokens"], 48)
        self.assertEqual(
            generator.params[0]["structured_outputs"]["regex"],
            r"(?:N|A [^\n\r|]+ \|\| [^\n\r]+)",
        )

    def test_semantic_quote_active_gate_accepts_without_second_call(self):
        output = (
            'A Billy Hill || "The Glory of Love" is a song written by Billy Hill'
        )
        generator = _QuoteGateGenerator(output)
        pipeline = _pipeline(_ContinuousRetriever(), generator)
        _enable_semantic_quote_gate(pipeline, shadow_only=False)
        state = self._quote_gate_state(pipeline)
        cancelled = set()

        self.assertTrue(
            pipeline._generate_once([state], cancelled, threading.Lock())
        )
        self.assertIsNone(state.answer)
        pipeline._commit_pending_single_snapshot_answers(
            [state], cancelled, threading.Lock()
        )

        self.assertEqual(state.answer, "Billy Hill")
        self.assertEqual(state.answer_source, "early_single_snapshot_confident")
        self.assertEqual(cancelled, {0})
        self.assertEqual(generator.calls, 1)
        accepted = state.answer_verification_trace[-1]
        self.assertEqual(accepted["decision"], "accepted")
        self.assertEqual(
            accepted["reason"], "single_snapshot_quote_anchor_gate"
        )

    def test_event_semantic_quote_shadow_uses_the_same_gate(self):
        pipeline = _pipeline(_ContinuousRetriever(), _NoGenerator())
        _enable_semantic_quote_gate(pipeline, shadow_only=True)
        state = self._quote_gate_state(pipeline)
        document_id, document, snapshot = state.pending_documents.popleft()
        visible = document["contents"]
        state.prompt_chain.append_document(document_id, document)
        state.evidence_texts.append(visible)
        state.generation_document_id = document_id
        state.generation_document_text = visible
        state.generation_snapshot = snapshot
        state.generation_inflight = True
        cancelled = set()
        cache_stat = {
            "finish_reason": "stop",
            "answerability_gate": {
                "logprob_margin": 15.0,
                "answer_probability": 0.999,
                "complete": True,
            },
        }

        pipeline._handle_event_search_completion(
            [state],
            0,
            'A Billy Hill || "The Glory of Love" is a song written by Billy Hill',
            cache_stat,
            cancelled,
            threading.Lock(),
        )

        self.assertIsNone(state.answer)
        self.assertFalse(state.generation_inflight)
        self.assertEqual(cancelled, set())
        self.assertEqual(
            state.answer_verification_trace[-1]["reason"],
            "semantic_quote_shadow_pass",
        )

    def test_semantic_quote_online_risk_gate_rejects_former_owner(self):
        pipeline = _pipeline(_ContinuousRetriever(), _NoGenerator())
        _enable_semantic_quote_gate(pipeline, shadow_only=False)
        state = _QueryPipelineState(
            question="who is the owner of reading football club",
            prompt_chain=pipeline._new_prompt_chain(
                "who is the owner of reading football club"
            ),
        )
        state.generation_document_text = (
            "Anton Zingarevich is the former owner of Reading Football Club."
        )
        snapshot = SimpleNamespace(
            budget=4,
            document_ids=(7, 8, 9),
            heap_document_ids=(7, 8, 9),
        )

        record = pipeline._semantic_quote_gate_evaluation(
            state,
            snapshot,
            7,
            "Anton Zingarevich",
            "Anton Zingarevich is the former owner of Reading Football Club",
            "A",
            True,
            True,
            True,
            18.0,
        )

        self.assertFalse(record["passed"])
        self.assertEqual(record["reason"], "risk")
        self.assertTrue(record["high_risk_question"])

    def _run_native_poll_round(self, *, event_on_first):
        stop_event = _RecordingStopEvent()
        retriever = _RoundPollingNativeRetriever(
            stop_event, event_on_first=event_on_first
        )
        pipeline = _pipeline(retriever, _NoGenerator())
        pipeline.event_driven_generation = True
        pipeline.pipeline_batch_size = 1
        pipeline.prefetch_batches = 1
        pipeline._runtime_actual_scanned_nprobe = [0, 0]
        pipeline._retrieval_worker(
            ["q0", "q1"],
            queue.Queue(),
            set(),
            threading.Lock(),
            stop_event,
        )
        return retriever, stop_event

    def test_full_continuous_retrieval_admits_bounded_nonparked_window(self):
        stop_event = _RecordingStopEvent()
        retriever = _RoundPollingNativeRetriever(
            stop_event, event_on_first=False
        )
        pipeline = _pipeline(retriever, _NoGenerator())
        pipeline.event_driven_generation = True
        pipeline.full_retrieval_continuous_batching = True
        pipeline.max_active_retrieval_sessions = 3
        pipeline.pipeline_batch_size = 1
        pipeline.prefetch_batches = 0
        pipeline._runtime_actual_scanned_nprobe = [0, 0, 0]

        pipeline._retrieval_worker(
            ["q0", "q1", "q2"],
            queue.Queue(),
            set(),
            threading.Lock(),
            stop_event,
        )

        self.assertEqual(retriever.pause_after_bootstrap, [False, False, False])
        self.assertEqual(retriever.poll_order, [0, 1, 2])
        self.assertEqual(retriever.closed_batch_ids, [0, 1, 2])
        self.assertEqual(len(stop_event.waits), 1)

    def test_native_m3_two_empty_sessions_wait_once_after_a_full_round(self):
        retriever, stop_event = self._run_native_poll_round(
            event_on_first=False
        )

        self.assertEqual(retriever.poll_order, [0, 1])
        self.assertEqual(len(stop_event.waits), 1)
        self.assertAlmostEqual(stop_event.waits[0], 0.00005)
        self.assertEqual(retriever.pause_after_bootstrap, [False, False])
        self.assertEqual(retriever.closed_batch_ids, [0, 1])

    def test_native_m3_any_event_in_round_skips_idle_wait(self):
        retriever, stop_event = self._run_native_poll_round(
            event_on_first=True
        )

        self.assertEqual(retriever.poll_order, [0, 1])
        self.assertEqual(stop_event.waits, [])
        self.assertEqual(retriever.pause_after_bootstrap, [False, False])
        self.assertEqual(retriever.closed_batch_ids, [0, 1])

    def test_bootstrap_top1_is_immutably_latched_while_scan_advances(self):
        retriever = _ContinuousRetriever()
        pipeline = _pipeline(retriever, _NoGenerator())
        state = _QueryPipelineState(
            question="q0", prompt_chain=pipeline._new_prompt_chain("q0")
        )
        session = retriever.start_batch(
            ["q0"], batch_id=0, query_indices=(0,), pause_after_bootstrap=False
        )

        event4 = retriever.advance_batch(session)
        pipeline._consume_event(event4, [state])
        event5 = retriever.advance_batch(session)
        pipeline._consume_event(event5, [state])

        self.assertEqual(state.latest_snapshot.budget, 5)
        self.assertEqual(state.bootstrap_snapshot.budget, BOOTSTRAP_NPROBE)
        self.assertTrue(
            pipeline._queue_ready_documents([state], set(), threading.Lock())
        )
        document_id, _, generation_snapshot = state.pending_documents[0]
        self.assertEqual(document_id, state.bootstrap_document_id)
        self.assertEqual(generation_snapshot.budget, BOOTSTRAP_NPROBE)

    def test_m2_target12_is_exact_and_survives_ranking_coalescing(self):
        retriever = _ContinuousRetriever()
        pipeline = _pipeline(retriever, _NoGenerator())
        pipeline.m2_first_snapshot_nprobe = 12
        pipeline.coalesce_ranking_events = True
        state = _QueryPipelineState(
            question="q0", prompt_chain=pipeline._new_prompt_chain("q0")
        )
        session = retriever.start_batch(
            ["q0"], batch_id=0, query_indices=(0,), pause_after_bootstrap=False
        )

        event4 = retriever.advance_batch(session)
        pipeline._consume_event(event4, [state])
        post_bootstrap = [retriever.advance_batch(session) for _ in range(9)]
        self.assertEqual(post_bootstrap[-1].budget, 13)
        pipeline._consume_ranking_event_burst(post_bootstrap, [state])

        self.assertEqual(state.latest_snapshot.budget, 13)
        self.assertEqual(state.bootstrap_snapshot.budget, 12)
        self.assertEqual(state.m2_first_snapshot_reason, "target_reached")
        self.assertTrue(pipeline._queue_bootstrap_documents([state]))
        _, _, generation_snapshot = state.pending_documents[0]
        self.assertEqual(generation_snapshot.budget, 12)

    def test_m2_target_batch_barrier_does_not_release_partial_batch(self):
        retriever = _ContinuousRetriever()
        pipeline = _pipeline(retriever, _NoGenerator())
        pipeline.m2_first_snapshot_nprobe = 12
        states = [
            _QueryPipelineState(
                question=f"q{index}",
                prompt_chain=pipeline._new_prompt_chain(f"q{index}"),
            )
            for index in range(2)
        ]
        session = retriever.start_batch(
            ["q0", "q1"],
            batch_id=0,
            query_indices=(0, 1),
            pause_after_bootstrap=False,
        )
        pipeline._consume_event(retriever.advance_batch(session), states)
        for _ in range(7):
            pipeline._consume_event(retriever.advance_batch(session), states)
        event12 = retriever.advance_batch(session)
        first_only = SimpleNamespace(
            batch_id=event12.batch_id,
            budget=event12.budget,
            snapshots=(event12.snapshots[0],),
        )
        second_only = SimpleNamespace(
            batch_id=event12.batch_id,
            budget=event12.budget,
            snapshots=(event12.snapshots[1],),
        )

        pipeline._consume_event(first_only, states)
        self.assertIsNotNone(states[0].bootstrap_snapshot)
        self.assertIsNone(states[1].bootstrap_snapshot)
        self.assertFalse(pipeline._queue_bootstrap_documents(states))
        self.assertFalse(states[0].pending_documents)

        pipeline._consume_event(second_only, states)
        self.assertTrue(pipeline._queue_bootstrap_documents(states))
        self.assertTrue(all(state.pending_documents for state in states))
        self.assertEqual(
            [state.pending_documents[0][2].budget for state in states],
            [12, 12],
        )

    def test_m2_target_batch_barrier_streams_after_batch_is_ready(self):
        retriever = _ContinuousRetriever()
        pipeline = _pipeline(retriever, _NoGenerator())
        pipeline.m2_first_snapshot_nprobe = 12
        pipeline.max_generation_chains = 1
        states = [
            _QueryPipelineState(
                question=f"q{index}",
                prompt_chain=pipeline._new_prompt_chain(f"q{index}"),
            )
            for index in range(2)
        ]
        session = retriever.start_batch(
            ["q0", "q1"],
            batch_id=0,
            query_indices=(0, 1),
            pause_after_bootstrap=False,
        )
        for _ in range(9):
            pipeline._consume_event(retriever.advance_batch(session), states)

        self.assertEqual(
            [state.bootstrap_snapshot.budget for state in states], [12, 12]
        )
        self.assertTrue(pipeline._queue_bootstrap_documents(states))
        self.assertEqual(
            [bool(state.pending_documents) for state in states], [True, False]
        )

        pipeline.max_generation_chains = 2
        self.assertTrue(pipeline._queue_bootstrap_documents(states))
        self.assertTrue(all(state.pending_documents for state in states))

    def test_m2_terminal_before_target_latches_without_resuming_retrieval(self):
        retriever = _ContinuousRetriever()
        pipeline = _pipeline(retriever, _NoGenerator())
        pipeline.m2_first_snapshot_nprobe = 12
        state = _QueryPipelineState(
            question="q0", prompt_chain=pipeline._new_prompt_chain("q0")
        )
        session = retriever.start_batch(
            ["q0"], batch_id=0, query_indices=(0,), pause_after_bootstrap=False
        )
        pipeline._consume_event(retriever.advance_batch(session), [state])
        for _ in range(5):
            event = retriever.advance_batch(session)
            if event.budget == 9:
                snapshot = event.snapshots[0]
                snapshot.finished = True
                snapshot.stop_reason = "wrd_stable"
                snapshot.phase = "FINISHED"
            pipeline._consume_event(event, [state])

        self.assertEqual(state.bootstrap_snapshot.budget, 9)
        self.assertEqual(
            state.m2_first_snapshot_reason,
            "retrieval_terminal_before_target",
        )
        self.assertEqual(state.terminal_reason, "wrd_stable")
        self.assertTrue(pipeline._queue_bootstrap_documents([state]))
        self.assertEqual(state.pending_documents[0][2].budget, 9)
        self.assertEqual(retriever.resume_calls, 0)

    def test_tuned_wrd_and_heap_contract(self):
        pipeline = _pipeline(_ContinuousRetriever(), _NoGenerator())
        self.assertEqual(pipeline.bootstrap_nprobe, 4)
        self.assertEqual(pipeline.fast_max_nprobe, 128)
        self.assertEqual(pipeline.max_nprobe, 128)
        self.assertEqual(pipeline.heap_topk, 100)
        self.assertEqual(pipeline.wrd_topk, 100)
        self.assertEqual(
            (
                pipeline.wrd_alpha,
                pipeline.wrd_epsilon,
                pipeline.wrd_window,
                pipeline.wrd_beta,
                pipeline.wrd_mu,
                pipeline.wrd_patience,
            ),
            (1.0, 0.0025, 5, 0.6, 0.5, 20),
        )
        self.assertEqual(pipeline.final_topk, 5)
        self.assertEqual(pipeline.max_answer_tokens, 5)
        self.assertEqual(pipeline.final_max_answer_tokens, 10)
        self.assertEqual(pipeline.final_generation_max_tokens, 32)
        self.assertTrue(pipeline.enable_incremental_generation)
        self.assertEqual(pipeline.answer_consistency_observations, 2)
        self.assertEqual(pipeline.answer_retention_topk, 5)
        self.assertEqual(pipeline.answer_fast_supporting_documents, 2)
        self.assertEqual(pipeline.answer_stable_supporting_documents, 1)

    def test_final_top5_uses_wider_grounded_answer_limit(self):
        pipeline = _pipeline(_ContinuousRetriever(), _NoGenerator())
        state = _QueryPipelineState(
            question="q",
            prompt_chain=AppendOnlyEvidencePrompt("q"),
        )
        state.evidence_texts = ["alpha beta gamma delta epsilon zeta"]

        self.assertFalse(
            pipeline._valid_grounded_answer(
                "alpha beta gamma delta epsilon zeta", state
            )
        )
        self.assertTrue(
            pipeline._valid_grounded_answer(
                "alpha beta gamma delta epsilon zeta",
                state,
                pipeline.final_max_answer_tokens,
            )
        )

    def test_m1_terminal_top5_runs_exactly_one_unified_final_generation(self):
        generator = _GateNoNativeAnswerGenerator()
        pipeline = _pipeline(
            _ContinuousRetriever(scan_delay=0.05), generator
        )
        pipeline.enable_incremental_generation = False
        state = _QueryPipelineState(
            question="q",
            prompt_chain=pipeline._new_prompt_chain("q"),
        )
        state.actual_scanned_nprobe = MAX_NPROBE
        state.latest_snapshot = SimpleNamespace(
            budget=MAX_NPROBE,
            scanned_budget=MAX_NPROBE,
            document_ids=(1, 2, 3, 4, 5),
            heap_document_ids=(1, 2, 3, 4, 5),
            scores=(1.0, 0.9, 0.8, 0.7, 0.6),
        )
        cancelled = set()
        original_prompt = state.prompt_chain.prompt

        changed = pipeline._mark_terminal(
            state, state.latest_snapshot, "max_nprobe"
        )

        self.assertTrue(changed)
        self.assertIsNone(state.answer)
        self.assertTrue(state.final_generation_pending)
        self.assertTrue(
            pipeline._generate_once([state], cancelled, threading.Lock())
        )
        self.assertEqual(state.answer, "native-final-answer")
        self.assertEqual(state.answer_accept_budget, MAX_NPROBE)
        self.assertEqual(state.answer_source, "final_max_nprobe")
        self.assertEqual(cancelled, {0})
        self.assertEqual(generator.calls, 1)
        final_prompt = generator.prompts[0][0]
        self.assertTrue(final_prompt.startswith(original_prompt))
        self.assertIn("RETRIEVAL_STATUS=FINAL", final_prompt)
        self.assertIn("evidence-1", final_prompt)
        self.assertIn("evidence-5", final_prompt)
        self.assertEqual(generator.params[0]["max_tokens"], 32)
        self.assertEqual(generator.params[0]["min_tokens"], 1)
        final_bad_words = set(generator.params[0]["bad_words"])
        self.assertTrue({"NO", "No", "nO", "no"}.issubset(final_bad_words))
        for quote in ('"', "'", "`"):
            self.assertIn(f"{quote}No", final_bad_words)
            self.assertIn(f"{quote}No{quote}", final_bad_words)
        self.assertTrue(generator.params[0]["return_cache_stats"])
        self.assertNotIn("stop", generator.params[0])
        self.assertFalse(
            pipeline._final_generation_once(
                [state], cancelled, threading.Lock()
            )
        )
        self.assertEqual(generator.calls, 1)

    def test_no_waits_for_a_strictly_newer_heap_snapshot(self):
        pipeline = _pipeline(_ContinuousRetriever(), _NoGenerator())
        state = _QueryPipelineState(
            question="q",
            prompt_chain=pipeline._new_prompt_chain("q"),
        )
        budget4 = SimpleNamespace(budget=4, scanned_budget=4)
        state.latest_snapshot = budget4
        state.generation_snapshot = budget4
        state.candidate_ids = (1, 2, 3)
        state.sent_document_ids.add(1)
        state.need_document = True

        changed = pipeline._queue_ready_documents(
            [state], set(), threading.Lock()
        )

        self.assertFalse(changed)
        self.assertEqual(list(state.pending_documents), [])

        budget5 = SimpleNamespace(budget=5, scanned_budget=5)
        state.latest_snapshot = budget5
        changed = pipeline._queue_ready_documents(
            [state], set(), threading.Lock()
        )

        self.assertTrue(changed)
        self.assertEqual(len(state.pending_documents), 1)
        document_id, _, generation_snapshot = state.pending_documents[0]
        self.assertEqual(document_id, 2)
        self.assertIs(generation_snapshot, budget5)

    def test_ready_documents_are_materialized_once_per_scheduler_batch(self):
        retriever = _ContinuousRetriever()
        pipeline = _pipeline(retriever, _NoGenerator())
        pipeline._runtime_promoted_batch_ids = frozenset({0})
        states = [
            _QueryPipelineState(
                question=f"q{query_index}",
                prompt_chain=pipeline._new_prompt_chain(f"q{query_index}"),
            )
            for query_index in range(2)
        ]
        for query_index, state in enumerate(states):
            base = query_index * 1000
            snapshot = SimpleNamespace(
                query_index=query_index,
                batch_id=0,
                budget=BOOTSTRAP_NPROBE,
                scanned_budget=BOOTSTRAP_NPROBE,
            )
            state.bootstrap_snapshot = snapshot
            state.bootstrap_document_id = base

        retriever.materialize_calls.clear()
        self.assertTrue(pipeline._queue_bootstrap_documents(states))
        self.assertEqual(retriever.materialize_calls, [(0, 1000)])

        for query_index, state in enumerate(states):
            base = query_index * 1000
            state.pending_documents.clear()
            state.search_document_ids.append(base)
            state.sent_document_ids.add(base)
            state.need_document = True
            state.generation_snapshot = state.bootstrap_snapshot
            state.latest_snapshot = SimpleNamespace(
                query_index=query_index,
                batch_id=0,
                budget=BOOTSTRAP_NPROBE + 1,
                scanned_budget=BOOTSTRAP_NPROBE + 1,
            )
            state.candidate_ids = (base, base + 1, base + 2)

        retriever.materialize_calls.clear()
        self.assertTrue(
            pipeline._queue_ready_documents(states, set(), threading.Lock())
        )
        self.assertEqual(retriever.materialize_calls, [(1, 1001)])
        pipeline._runtime_promoted_batch_ids = None

    def test_wrd_stable_is_terminal_and_no_never_resumes_to_128(self):
        retriever = _ContinuousRetriever(stable_at=8)
        output = _pipeline(retriever, _GateNoNativeAnswerGenerator()).run(
            _Dataset(["q0"]), do_eval=False
        )

        fast_stop = _output(output, "fast_stop_budget", "fast_stop_nprobe")[0]
        self.assertEqual(_output(output, "fast_stop_reason")[0], "stable")
        self.assertGreater(fast_stop, 12)
        self.assertLess(fast_stop, MAX_NPROBE)
        self.assertEqual(retriever.resume_calls, 0)
        self.assertEqual(retriever.scan_history[0][-1], fast_stop)
        self.assertNotIn(fast_stop + 1, retriever.scan_history[0])
        self.assertEqual(
            _output(output, "actual_scanned_nprobe", "scanned_nprobe")[0],
            fast_stop,
        )
        self.assertEqual(_output(output, "terminal_reason")[0], "wrd_stable")
        self.assertEqual(_output(output, "answer_source")[0], "final_document_limit")
        self.assertEqual(_output(output, "pred")[0], "native-final-answer")
        self.assertTrue(_output(output, "final_generation_done")[0])
        self.assertEqual(len(_output(output, "retrieval_idxs")[0]), FINAL_TOPK)

    def test_mechanism1_only_skips_searching_generation(self):
        retriever = _ContinuousRetriever(stable_at=8)
        generator = _GateNoNativeAnswerGenerator()
        pipeline = _pipeline(retriever, generator)
        pipeline.enable_incremental_generation = False
        pipeline.prefetch_batches = 0

        output = pipeline.run(_Dataset(["q0"]), do_eval=False)

        self.assertEqual(generator.calls, 1)
        self.assertIn("RETRIEVAL_STATUS=FINAL", generator.prompts[0][0])
        self.assertNotIn("RETRIEVAL_STATUS=SEARCHING", generator.prompts[0][0])
        self.assertEqual(_attempted_ids(output), [])
        self.assertEqual(_output(output, "answer_source")[0], "final_wrd_stable")
        self.assertEqual(retriever.resume_calls, 0)

    def test_final_rebuilds_when_search_evidence_is_not_top5_prefix(self):
        pipeline = _pipeline(_ContinuousRetriever(), _GateNoNativeAnswerGenerator())
        pipeline.enable_incremental_generation = False
        state = _QueryPipelineState(
            question="q", prompt_chain=pipeline._new_prompt_chain("q")
        )
        state.prompt_chain.append_document(99, {"contents": "stale evidence-99"})
        snapshot = SimpleNamespace(
            budget=20,
            scanned_budget=20,
            document_ids=(1, 2, 3, 4, 5),
            heap_document_ids=(1, 2, 3, 4, 5),
            scores=(1.0, 0.9, 0.8, 0.7, 0.6),
        )
        pipeline._mark_terminal(state, snapshot, "wrd_stable")
        prompt = pipeline._build_final_prompt(state)

        self.assertNotIn("stale evidence-99", prompt)
        self.assertEqual(state.final_prompt_document_ids, (1, 2, 3, 4, 5))
        self.assertFalse(state.final_prompt_reused_evidence_prefix)
        self.assertIn("RETRIEVAL_STATUS=FINAL", prompt)

    def test_without_wrd_stability_scans_directly_to_128(self):
        retriever = _ContinuousRetriever(stable_at=None)
        output = _pipeline(retriever, _GateNoNativeAnswerGenerator()).run(
            _Dataset(["q0"]), do_eval=False
        )

        self.assertIsNone(_output(output, "fast_stop_budget", "fast_stop_nprobe")[0])
        self.assertIsNone(_output(output, "fast_stop_reason")[0])
        self.assertIsNone(retriever.sessions[0].resume_from_cursor)
        self.assertEqual(retriever.resume_calls, 0)
        self.assertEqual(retriever.scan_history[0][0], BOOTSTRAP_NPROBE)
        self.assertEqual(retriever.scan_history[0][-1], MAX_NPROBE)
        self.assertEqual(
            _output(output, "actual_scanned_nprobe", "scanned_nprobe")[0],
            MAX_NPROBE,
        )
        self.assertEqual(_output(output, "terminal_reason")[0], "max_nprobe")
        self.assertEqual(_output(output, "answer_source")[0], "final_document_limit")

    def test_fast_stable_branch_remains_defined_for_preloaded_history(self):
        retriever = _ContinuousRetriever(stable_at=5)
        session = retriever.start_batch(["q0"], batch_id=0, query_indices=(0,))
        session.cursor = BOOTSTRAP_NPROBE
        session.stable_streak = WRD_PATIENCE - 1

        event = retriever.advance_batch(session)

        self.assertEqual(event.snapshots[0].fast_stop_reason, "stable")
        self.assertEqual(event.snapshots[0].fast_stop_budget, 5)

    def test_no_selects_best_unsent_then_final_uses_latest_top5(self):
        retriever = _ContinuousRetriever(scan_delay=0.02)
        generator = _GateNoNativeAnswerGenerator(first_delay=0.08)
        output = _pipeline(retriever, generator).run(
            _Dataset(["q0"]), do_eval=False
        )

        attempted = _attempted_ids(output)
        self.assertEqual(attempted, [0, 1, 2, 3, 4])
        self.assertEqual(
            _output(output, "search_generation_budgets")[0][0],
            BOOTSTRAP_NPROBE,
        )
        # The same-batch scanner resumes immediately after emitting @4 and
        # advances while the first LLM call is still running.
        self.assertLess(
            retriever.timestamps["batch_0_nprobe_5"], generator.first_end
        )
        self.assertEqual(_output(output, "pred")[0], "native-final-answer")
        self.assertEqual(_output(output, "answer_source")[0], "final_document_limit")
        self.assertTrue(_output(output, "final_generation_done")[0])
        self.assertEqual(
            _output(output, "final_prompt_document_ids")[0], [0, 1, 2, 3, 4]
        )
        self.assertEqual(
            _output(output, "actual_scanned_nprobe", "scanned_nprobe")[0],
            MAX_NPROBE,
        )
        self.assertTrue(_output(output, "final_top5_ready")[0])
        # Four answerability checks plus one forced FINAL on the fifth append:
        # never a fifth SEARCHING followed by a duplicate sixth decode.
        self.assertEqual(generator.calls, FINAL_TOPK)
        searching_prompts = [call[0] for call in generator.prompts[:-1]]
        self.assertTrue(
            all("RETRIEVAL_STATUS=SEARCHING" in prompt for prompt in searching_prompts)
        )
        self.assertEqual(
            [len(set(re.findall(r"evidence-(\d+)", prompt))) for prompt in searching_prompts],
            [1, 2, 3, 4],
        )
        self.assertIn("RETRIEVAL_STATUS=FINAL", generator.prompts[-1][0])
        self.assertEqual(
            len(set(re.findall(r"evidence-(\d+)", generator.prompts[-1][0]))),
            FINAL_TOPK,
        )
        self.assertEqual(generator.params[-1]["max_tokens"], 32)
        self.assertEqual(generator.params[-1]["min_tokens"], 1)
        self.assertIn("bad_words", generator.params[-1])
        self.assertNotIn("stop", generator.params[-1])
        self.assertEqual(len(_output(output, "generation_trace")[0]), FINAL_TOPK)
        self.assertTrue(all(_output(output, "prefix_append_invariant")[0]))

    def test_fifth_token_prompt_append_queues_final_without_search_decode(self):
        generator = _GateNoNativeAnswerGenerator()
        generator.tokenizer = _CharacterTokenizer()
        pipeline = _pipeline(_ContinuousRetriever(), generator)
        pipeline.use_token_prompts = True
        state = _QueryPipelineState(
            question="q",
            prompt_chain=pipeline._new_prompt_chain("q"),
        )
        for document_id in range(4):
            self.assertTrue(
                state.prompt_chain.append_document(
                    document_id, {"contents": f"evidence-{document_id}"}
                )
            )
        state.evidence_token_ids = pipeline._encode_text(
            state.prompt_chain.prompt, add_special_tokens=False
        )
        snapshot = SimpleNamespace(
            budget=MAX_NPROBE,
            scanned_budget=MAX_NPROBE,
            document_ids=(0, 1, 2, 3, 4),
            heap_document_ids=(0, 1, 2, 3, 4),
            scores=(1.0, 0.9, 0.8, 0.7, 0.6),
        )
        state.latest_snapshot = snapshot
        state.pending_documents.append(
            (4, {"contents": "evidence-4"}, snapshot)
        )
        cancelled = set()
        lock = threading.Lock()

        query_indices, prompts = pipeline._ready_generation_batch(
            [state], cancelled, lock
        )

        self.assertEqual(query_indices, [])
        self.assertEqual(prompts, [])
        self.assertTrue(state.final_generation_pending)
        self.assertEqual(state.final_trigger_reason, "document_limit")
        self.assertEqual(cancelled, {0})
        self.assertEqual(state.prompt_chain.document_ids, [0, 1, 2, 3, 4])
        self.assertEqual(
            state.evidence_token_ids,
            pipeline._encode_text(
                state.prompt_chain.prompt, add_special_tokens=False
            ),
        )
        self.assertTrue(pipeline._generate_once([state], cancelled, lock))
        self.assertEqual(generator.calls, 1)
        self.assertIn(
            "RETRIEVAL_STATUS=FINAL",
            generator.prompt_text(generator.prompts[0][0]),
        )
        self.assertTrue(state.final_prompt_token_prefix_aligned)
        self.assertEqual(
            generator.prompts[0][0]["prompt_token_ids"][
                : len(state.evidence_token_ids)
            ],
            state.evidence_token_ids,
        )
        self.assertEqual(generator.params[0]["max_tokens"], 32)
        self.assertNotIn("stop", generator.params[0])

    def test_scheduled_penultimate_gate_skip_still_appends_then_finalizes(self):
        generator = _GateNoNativeAnswerGenerator()
        generator.tokenizer = _CharacterTokenizer()
        pipeline = _pipeline(_ContinuousRetriever(), generator)
        pipeline.use_token_prompts = True
        pipeline.search_gate_ordinals = (1, 2, 3)
        state = _QueryPipelineState(
            question="q",
            prompt_chain=pipeline._new_prompt_chain("q"),
        )
        snapshot = SimpleNamespace(
            budget=MAX_NPROBE,
            scanned_budget=MAX_NPROBE,
            document_ids=(0, 1, 2, 3, 4),
            heap_document_ids=(0, 1, 2, 3, 4),
            scores=(1.0, 0.9, 0.8, 0.7, 0.6),
        )
        state.latest_snapshot = snapshot
        for document_id in range(3):
            self.assertTrue(
                state.prompt_chain.append_document(
                    document_id, {"contents": f"evidence-{document_id}"}
                )
            )
            state.sent_document_ids.add(document_id)
            state.evidence_texts.append(state.prompt_chain.document_texts[-1])
            state.search_generation_budgets.append(MAX_NPROBE)
        state.evidence_token_ids = pipeline._encode_text(
            state.prompt_chain.prompt, add_special_tokens=False
        )
        state.pending_documents.append(
            (3, {"contents": "evidence-3"}, snapshot)
        )

        query_indices, prompts = pipeline._ready_generation_batch([state])

        self.assertEqual(query_indices, [])
        self.assertEqual(prompts, [])
        self.assertEqual(state.prompt_chain.document_ids, [0, 1, 2, 3])
        self.assertTrue(state.need_document)
        self.assertEqual(
            state.answerability_skip_trace[-1]["reason"],
            "scheduled_search_gate_skip",
        )
        self.assertEqual(
            state.answerability_skip_trace[-1]["document_ordinal"], 4
        )

        state.need_document = False
        state.pending_documents.append(
            (4, {"contents": "evidence-4"}, snapshot)
        )
        cancelled = set()
        lock = threading.Lock()
        query_indices, prompts = pipeline._ready_generation_batch(
            [state], cancelled, lock
        )

        self.assertEqual(query_indices, [])
        self.assertEqual(prompts, [])
        self.assertTrue(state.final_generation_pending)
        self.assertEqual(state.prompt_chain.document_ids, [0, 1, 2, 3, 4])
        self.assertEqual(cancelled, {0})

    def test_cross_snapshot_answer_requires_a_new_snapshot_confirmation(self):
        retriever = _ContinuousRetriever(scan_delay=0.05, common_support=True)
        pipeline = _pipeline(retriever, _ConsistentGroundedGenerator())
        pipeline.answer_fast_supporting_documents = 1
        output = pipeline.run(
            _Dataset(["q0"]), do_eval=False
        )

        self.assertEqual(_output(output, "pred")[0], "evidence-0")
        self.assertGreater(
            _output(output, "answer_accept_budget", "answer_accept_nprobe")[0],
            BOOTSTRAP_NPROBE,
        )
        self.assertLess(
            _output(output, "actual_scanned_nprobe", "scanned_nprobe")[0],
            MAX_NPROBE,
        )
        decisions = [
            record["decision"]
            for record in _output(output, "answer_verification_trace")[0]
        ]
        self.assertIn("provisional", decisions)
        self.assertIn("cross_snapshot_confirmation", decisions)
        self.assertIn("accepted", decisions)
        self.assertEqual(
            _output(output, "answer_source")[0],
            "early_cross_snapshot_consistent",
        )
        # With a single-query workload the pipeline can stop the retrieval
        # actor globally before its next per-session cancel callback.  The
        # observable contract is that no scan reaches the max budget.

    def test_cross_snapshot_consistency_does_not_require_verbatim_new_doc_support(self):
        pipeline = _pipeline(
            _ContinuousRetriever(), _ConsistentGroundedGenerator()
        )
        state = _QueryPipelineState(
            question="q",
            prompt_chain=AppendOnlyEvidencePrompt("q", max_documents=5),
        )
        state.candidate_ids = tuple(range(HEAP_TOPK))
        cancelled = set()
        lock = threading.Lock()

        state.pending_documents.append(
            (0, {"contents": "evidence-0"}, SimpleNamespace(budget=4))
        )
        self.assertTrue(pipeline._generate_once([state], cancelled, lock))
        self.assertIsNone(state.answer)

        # The accumulated prefix still supports the same answer, but this new
        # document deliberately does not repeat it verbatim.  Agreement across
        # the newer snapshot is the verifier; lexical copying is not.
        state.pending_documents.append(
            (1, {"contents": "a second relevant passage"}, SimpleNamespace(budget=5))
        )
        self.assertTrue(pipeline._generate_once([state], cancelled, lock))
        self.assertEqual(state.answer, "evidence-0")
        accepted = [
            row
            for row in state.answer_verification_trace
            if row.get("decision") == "accepted"
        ]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(
            accepted[0]["reason"],
            "cross_snapshot_consistent_with_retained_observation_documents",
        )
        self.assertEqual(accepted[0]["observation_document_ids"], [0, 1])

    def test_generation_stops_at_first_line_with_defensive_parse(self):
        generator = _TrailingGroundedGenerator()
        pipeline = _pipeline(_ContinuousRetriever(), generator)
        pipeline.answer_fast_supporting_documents = 1
        state = _QueryPipelineState(
            question="q0",
            prompt_chain=AppendOnlyEvidencePrompt("q0", max_documents=5),
        )
        state.candidate_ids = tuple(range(HEAP_TOPK))
        state.pending_documents.append(
            (0, {"contents": "evidence-0"}, SimpleNamespace(budget=4))
        )
        cancelled = set()
        lock = threading.Lock()
        self.assertTrue(pipeline._generate_once([state], cancelled, lock))
        state.pending_documents.append(
            (
                1,
                {"contents": "evidence-1 independently confirms evidence-0"},
                SimpleNamespace(budget=5),
            )
        )
        self.assertTrue(pipeline._generate_once([state], cancelled, lock))

        self.assertEqual(state.answer, "evidence-0")
        self.assertEqual(generator.params[0]["stop"], ["\n", "\r"])
        self.assertFalse(generator.params[0]["include_stop_str_in_output"])
        self.assertEqual(generator.params[0]["max_tokens"], 16)
        self.assertEqual(len(state.generation_trace), 2)
        self.assertTrue(
            all(
                value == "evidence-0"
                for value in state.generation_trace
            )
        )

    def test_max_token_truncation_is_never_accepted_as_an_answer(self):
        output = _pipeline(
            _ContinuousRetriever(scan_delay=0.05), _TruncatedGroundedGenerator()
        ).run(_Dataset(["q0"]), do_eval=False)

        self.assertEqual(_output(output, "answer_source")[0], "final_document_limit")
        reasons = [
            record["reason"]
            for record in _output(output, "answer_verification_trace")[0]
        ]
        self.assertIn("max_tokens_truncated", reasons)

    def test_same_snapshot_does_not_count_as_cross_snapshot_confirmation(self):
        pipeline = _pipeline(
            _ContinuousRetriever(), _ConsistentGroundedGenerator()
        )
        pipeline.answer_fast_supporting_documents = 1
        state = _QueryPipelineState(
            question="q",
            prompt_chain=AppendOnlyEvidencePrompt("q", max_documents=5),
        )
        state.candidate_ids = tuple(range(HEAP_TOPK))
        cancelled = set()
        lock = threading.Lock()

        first = SimpleNamespace(budget=4)
        state.pending_documents.append(
            (0, {"contents": "evidence-0"}, first)
        )
        self.assertTrue(pipeline._generate_once([state], cancelled, lock))
        self.assertEqual(state.provisional_observations, 1)
        self.assertIsNone(state.answer)

        same_snapshot = SimpleNamespace(budget=4)
        state.pending_documents.append(
            (
                1,
                {"contents": "evidence-1 independently confirms evidence-0"},
                same_snapshot,
            )
        )
        self.assertTrue(pipeline._generate_once([state], cancelled, lock))
        self.assertEqual(state.provisional_observations, 1)
        self.assertIsNone(state.answer)

        newer_snapshot = SimpleNamespace(budget=5)
        state.pending_documents.append(
            (
                2,
                {"contents": "evidence-2 independently confirms evidence-0"},
                newer_snapshot,
            )
        )
        self.assertTrue(pipeline._generate_once([state], cancelled, lock))
        self.assertEqual(state.provisional_observations, 2)
        self.assertEqual(state.answer, "evidence-0")

    def test_conflicting_answers_reach_unified_final_instead_of_early_exit(self):
        output = _pipeline(
            _ContinuousRetriever(scan_delay=0.05), _GroundedGenerator()
        ).run(_Dataset(["q0"]), do_eval=False)

        self.assertEqual(_output(output, "answer_source")[0], "final_document_limit")
        reasons = [
            record["reason"]
            for record in _output(output, "answer_verification_trace")[0]
        ]
        self.assertIn("conflicting_answer_across_snapshots", reasons)
        self.assertIn("document_limit", reasons)

    def test_cross_snapshot_agreement_can_finish_before_wrd_final(self):
        retriever = _ContinuousRetriever(stable_at=5)
        output = _pipeline(
            retriever, _ConsistentGroundedGenerator()
        ).run(_Dataset(["q0"]), do_eval=False)

        self.assertEqual(
            _output(output, "answer_source")[0],
            "early_cross_snapshot_consistent",
        )
        self.assertGreater(
            _output(output, "answer_accept_budget")[0], BOOTSTRAP_NPROBE
        )
        accepted = [
            record
            for record in _output(output, "answer_verification_trace")[0]
            if record.get("decision") == "accepted"
        ]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(
            accepted[0]["reason"],
            "cross_snapshot_consistent_with_retained_observation_documents",
        )
        forced = [
            record
            for record in _output(output, "answer_verification_trace")[0]
            if record.get("decision") == "forced_final"
        ]
        self.assertEqual(forced, [])

    def test_context_full_stops_before_attempting_all_documents(self):
        retriever = _ContinuousRetriever(long_documents=True, scan_delay=0.02)
        pipeline = _pipeline(retriever, _GateNoNativeAnswerGenerator())
        pipeline.max_input_tokens = 128
        output = pipeline.run(_Dataset(["q0"]), do_eval=False)

        self.assertEqual(_output(output, "pred")[0], "native-final-answer")
        self.assertEqual(
            _output(output, "retrieval_stop_reason")[0],
            "context_full",
        )
        self.assertEqual(_output(output, "answer_source")[0], "final_context_full")
        self.assertTrue(_output(output, "final_generation_done")[0])
        self.assertEqual(
            _output(output, "final_prompt_document_ids")[0],
            _attempted_ids(output),
        )
        self.assertTrue(
            _output(output, "final_prompt_reused_evidence_prefix")[0]
        )
        self.assertLess(len(_attempted_ids(output)), FINAL_TOPK)
        self.assertLess(
            _output(output, "actual_scanned_nprobe", "scanned_nprobe")[0],
            MAX_NPROBE,
        )

    @staticmethod
    def _coalescing_pipeline(generator):
        pipeline = _pipeline(_ContinuousRetriever(), generator)
        pipeline.event_driven_generation = True
        pipeline.single_snapshot_gate = True
        pipeline.compact_answerability_prompt = True
        pipeline.answerability_structured_output = True
        pipeline.semantic_quote_gate = False
        pipeline.answerability_gate_logprobs = 100
        pipeline.minimal_structured_gate_logprobs = True
        pipeline.answerability_require_logprobs = True
        pipeline.answerability_logprob_margin = 16.0
        pipeline.answerability_retention_topk = 5
        pipeline._answerability_gate_token_ids = {
            "answer": (1,),
            "no": (2,),
        }
        pipeline._system_prompt_template = COMPACT_RAG_SYSTEM_PROMPT
        pipeline._user_tail_template = COMPACT_RAG_USER_TAIL_TEMPLATE
        pipeline._configure_evidence_prompt_frame()
        pipeline.generation_coalesce_ms = 100.0
        pipeline.generation_coalesce_max_batch = 8
        pipeline.generation_coalesce_max_steps = 1
        pipeline._reset_event_search_coalescer(clear_trace=True)
        return pipeline

    @staticmethod
    def _pending_event_state(document_id=0, *, continuation=True):
        state = _QueryPipelineState(
            question=f"q{document_id}",
            prompt_chain=AppendOnlyEvidencePrompt(
                f"q{document_id}", max_documents=FINAL_TOPK
            ),
        )
        if continuation:
            prior_id = 1000 + int(document_id)
            state.prompt_chain.append_document(
                prior_id,
                {"contents": f"prior-title-{document_id}\nprior-{document_id}"},
            )
            state.sent_document_ids.add(prior_id)
            state.completed_document_ids.add(prior_id)
            state.evidence_texts.append(
                state.prompt_chain.document_texts[-1]
            )
            state.search_generation_budgets.append(BOOTSTRAP_NPROBE)
        state.candidate_ids = tuple(range(HEAP_TOPK))
        snapshot = SimpleNamespace(budget=BOOTSTRAP_NPROBE)
        if not continuation:
            state.bootstrap_snapshot = snapshot
            state.bootstrap_document_id = int(document_id)
        state.pending_documents.append(
            (
                int(document_id),
                {"contents": f"title-{document_id}\nevidence-{document_id}"},
                snapshot,
            )
        )
        return state

    def test_event_coalescer_defers_only_while_engine_does_useful_work(self):
        generator = _SteppedEventGenerator(active_request_ids=("existing",))
        pipeline = self._coalescing_pipeline(generator)
        state = self._pending_event_state()
        cancelled = set()
        lock = threading.Lock()

        # The continuation remains untouched for one useful engine step.  No
        # sleep is used and the independent retrieval actor would keep running.
        self.assertTrue(
            pipeline._drive_event_generation_once([state], cancelled, lock)
        )
        self.assertEqual(generator.step_calls, 1)
        self.assertEqual(generator.added_batches, [])
        self.assertEqual(len(state.pending_documents), 1)
        self.assertEqual(state.prompt_chain.document_ids, [1000])

        # The engine-step bound now forces admission even though the generous
        # wall-clock deadline has not elapsed.
        self.assertTrue(
            pipeline._drive_event_generation_once([state], cancelled, lock)
        )
        self.assertEqual(len(generator.added_batches), 1)
        self.assertEqual(len(generator.added_batches[0][0]), 1)
        self.assertEqual(list(state.pending_documents), [])
        self.assertEqual(state.prompt_chain.document_ids, [1000, 0])
        self.assertEqual(
            pipeline._event_search_coalesce_trace[-1]["reason"],
            "engine_step_bound",
        )
        self.assertEqual(
            pipeline._event_search_coalesce_trace[-1][
                "deferred_engine_steps"
            ],
            1,
        )

    def test_event_coalescer_never_delays_an_idle_engine(self):
        generator = _SteppedEventGenerator()
        pipeline = self._coalescing_pipeline(generator)
        state = self._pending_event_state()

        self.assertTrue(
            pipeline._drive_event_generation_once(
                [state], set(), threading.Lock()
            )
        )
        self.assertEqual(len(generator.added_batches), 1)
        self.assertEqual(generator.step_calls, 1)
        self.assertEqual(pipeline._event_search_coalesce_trace, [])

    def test_event_coalescer_never_delays_nprobe4_bootstrap(self):
        generator = _SteppedEventGenerator(active_request_ids=("existing",))
        pipeline = self._coalescing_pipeline(generator)
        state = self._pending_event_state(continuation=False)

        self.assertTrue(
            pipeline._drive_event_generation_once(
                [state], set(), threading.Lock()
            )
        )
        self.assertEqual(len(generator.added_batches), 1)
        self.assertEqual(state.first_search_generation_budget, BOOTSTRAP_NPROBE)
        self.assertEqual(pipeline._event_search_coalesce_trace, [])

    def test_event_coalescer_flushes_ready_siblings_as_one_dynamic_batch(self):
        generator = _SteppedEventGenerator(active_request_ids=("existing",))
        pipeline = self._coalescing_pipeline(generator)
        pipeline.generation_coalesce_max_batch = 2
        states = [self._pending_event_state(0), self._pending_event_state(1)]

        self.assertTrue(
            pipeline._drive_event_generation_once(
                states, set(), threading.Lock()
            )
        )
        self.assertEqual(len(generator.added_batches), 1)
        self.assertEqual(len(generator.added_batches[0][0]), 2)
        self.assertEqual(
            pipeline._event_search_coalesce_trace[-1]["reason"],
            "batch_threshold",
        )

    def test_structured_gate_requests_only_the_two_allowed_logprobs(self):
        pipeline = self._coalescing_pipeline(_SteppedEventGenerator())
        self.assertEqual(
            pipeline._search_generation_kwargs()["gate_logprobs"], 2
        )
        pipeline.minimal_structured_gate_logprobs = False
        self.assertEqual(
            pipeline._search_generation_kwargs()["gate_logprobs"], 100
        )

    def test_priority_scheduler_uses_post_build_search_state_and_metadata(self):
        cases = ((False, 0), (True, -1))
        for continuation, expected_priority in cases:
            with self.subTest(continuation=continuation):
                generator = _SteppedEventGenerator()
                pipeline = self._coalescing_pipeline(generator)
                pipeline.continuation_priority_scheduler = True
                state = self._pending_event_state(
                    continuation=continuation
                )

                submitted = pipeline._submit_event_search_requests(
                    [state], set(), threading.Lock()
                )

                self.assertEqual(submitted, 1)
                self.assertEqual(
                    generator.added_params[-1]["priorities"],
                    [expected_priority],
                )
                metadata = generator.added_batches[-1][1][0]
                self.assertEqual(metadata["kind"], "SEARCHING")
                self.assertEqual(metadata["priority"], expected_priority)
                self.assertEqual(
                    len(state.search_generation_budgets),
                    2 if continuation else 1,
                )

    def test_priority_scheduler_final_and_disabled_control_metadata(self):
        generator = _SteppedEventGenerator()
        pipeline = self._coalescing_pipeline(generator)
        pipeline.continuation_priority_scheduler = True
        state = self._pending_event_state()
        state.pending_documents.clear()
        state.final_generation_pending = True
        pipeline._build_final_prompt = lambda _state: "forced-final"

        submitted = pipeline._submit_event_final_requests([state])

        self.assertEqual(submitted, 1)
        self.assertEqual(generator.added_params[-1]["priorities"], [-2])
        self.assertEqual(
            generator.added_batches[-1][1][0],
            {"kind": "FINAL", "query_index": 0, "priority": -2},
        )

        control_generator = _SteppedEventGenerator()
        control = self._coalescing_pipeline(control_generator)
        control.continuation_priority_scheduler = False
        control_state = self._pending_event_state(continuation=True)
        control._submit_event_search_requests(
            [control_state], set(), threading.Lock()
        )
        self.assertEqual(
            control_generator.added_params[-1]["priorities"], [0]
        )
        self.assertEqual(
            control_generator.added_batches[-1][1][0]["priority"], 0
        )

    @staticmethod
    def _length_repair_pipeline_and_state():
        generator = _SteppedEventGenerator()
        pipeline = ContinuousPipelineContractTest._coalescing_pipeline(generator)
        pipeline.final_generation_stop = ["\n", "\r"]
        pipeline.final_length_repair_stop = ["\n", "\r", "."]
        pipeline.final_generation_bad_words = []
        pipeline.capture_prefix_cache_stats = True
        pipeline._build_final_prompt = lambda _state: "same-frozen-evidence-prompt"
        pipeline._stage_prefix_cache_audit = lambda *_args, **_kwargs: None
        pipeline._complete_prefix_cache_audit = (
            lambda _state, cache_stat: cache_stat
        )
        state = ContinuousPipelineContractTest._pending_event_state()
        state.pending_documents.clear()
        state.final_generation_pending = True
        state.final_trigger_reason = "max_budget"
        state.terminal_reason = "max_budget"
        state.terminal_budget = MAX_NPROBE
        state.terminal_top5_ids = tuple(range(FINAL_TOPK))
        return pipeline, generator, state

    def test_event_final_length_completion_is_accepted_as_truncated(self):
        pipeline, generator, state = self._length_repair_pipeline_and_state()
        cancelled = set()
        lock = threading.Lock()

        self.assertEqual(pipeline._submit_event_final_requests([state]), 1)
        self.assertEqual(generator.added_batches[-1][1][0]["kind"], "FINAL")
        self.assertEqual(generator.added_params[-1]["stop"], ["\n", "\r"])
        pipeline._handle_event_final_completion(
            [state],
            0,
            "truncated draft",
            {"finish_reason": "length"},
            "FINAL",
            cancelled,
            lock,
        )

        self.assertEqual(state.answer, "truncated draft")
        self.assertTrue(state.final_generation_done)
        self.assertFalse(state.final_generation_pending)
        self.assertFalse(state.final_length_repair_pending)
        self.assertIn(0, cancelled)
        completed = [
            row
            for row in state.final_generation_trace
            if row.get("event") == "completed"
        ]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["finish_reason"], "length")
        self.assertTrue(completed[0]["output_truncated"])
        self.assertFalse(completed[0]["length_repair"])

    def test_event_final_length_repair_still_length_fails_closed(self):
        pipeline, _generator, state = self._length_repair_pipeline_and_state()
        cancelled = set()
        lock = threading.Lock()
        state.final_length_repair_attempted = True
        with self.assertRaisesRegex(RuntimeError, "still truncated"):
            pipeline._handle_event_final_completion(
                [state],
                0,
                "second truncated draft",
                {"finish_reason": "length"},
                "FINAL_LENGTH_REPAIR",
                cancelled,
                lock,
            )

    def test_event_final_sentinel_repair_is_one_shot_and_audited(self):
        pipeline, generator, state = self._length_repair_pipeline_and_state()
        pipeline.no_answer_sentinel = "[INSUFFICIENT_EVIDENCE]"
        cancelled = set()
        lock = threading.Lock()

        self.assertEqual(pipeline._submit_event_final_requests([state]), 1)
        pipeline._handle_event_final_completion(
            [state],
            0,
            "[INSUFFICIENT_EVIDENCE]?",
            {"finish_reason": "stop"},
            "FINAL",
            cancelled,
            lock,
        )

        self.assertIsNone(state.answer)
        self.assertTrue(state.final_generation_pending)
        self.assertTrue(state.final_sentinel_repair_pending)
        self.assertTrue(state.final_sentinel_repair_attempted)
        self.assertNotIn(0, cancelled)

        self.assertEqual(pipeline._submit_event_final_requests([state]), 1)
        self.assertEqual(
            generator.added_batches[-1][1][0]["kind"],
            "FINAL_SENTINEL_REPAIR",
        )
        self.assertIn(
            "invalid retrieval-control sentinel",
            generator.added_batches[-1][0][0],
        )
        self.assertIn(
            "A valid answer may be Yes or No",
            generator.added_batches[-1][0][0],
        )
        pipeline._handle_event_final_completion(
            [state],
            0,
            "supported answer",
            {"finish_reason": "stop"},
            "FINAL_SENTINEL_REPAIR",
            cancelled,
            lock,
        )

        self.assertEqual(state.answer, "supported answer")
        self.assertTrue(state.final_generation_done)
        self.assertIn(0, cancelled)
        completed = [
            row
            for row in state.final_generation_trace
            if row.get("event") == "completed"
        ]
        self.assertEqual(len(completed), 1)
        self.assertTrue(completed[0]["sentinel_repair"])

    def test_event_final_sentinel_repair_still_no_fails_closed(self):
        pipeline, _generator, state = self._length_repair_pipeline_and_state()
        pipeline.no_answer_sentinel = "[INSUFFICIENT_EVIDENCE]"
        state.final_sentinel_repair_attempted = True
        with self.assertRaisesRegex(RuntimeError, "still returned"):
            pipeline._handle_event_final_completion(
                [state],
                0,
                "[INSUFFICIENT_EVIDENCE],",
                {"finish_reason": "stop"},
                "FINAL_SENTINEL_REPAIR",
                set(),
                threading.Lock(),
            )

    def test_event_final_allows_legitimate_no_answer(self):
        pipeline, _generator, state = self._length_repair_pipeline_and_state()
        pipeline.no_answer_sentinel = "[INSUFFICIENT_EVIDENCE]"
        cancelled = set()
        pipeline._handle_event_final_completion(
            [state],
            0,
            "No",
            {"finish_reason": "stop"},
            "FINAL",
            cancelled,
            threading.Lock(),
        )
        self.assertEqual(state.answer, "No")
        self.assertTrue(state.final_generation_done)
        self.assertFalse(state.final_sentinel_repair_attempted)
        self.assertIn(0, cancelled)

    def test_grounded_search_allows_legitimate_a_no_payload(self):
        pipeline, _generator, state = self._length_repair_pipeline_and_state()
        pipeline.no_answer_sentinel = "[INSUFFICIENT_EVIDENCE]"
        state.evidence_texts = [
            "The sources explicitly state that the two people did not do "
            "the same kind of work, so the answer is no."
        ]
        self.assertTrue(pipeline._valid_grounded_answer("No", state))
        self.assertFalse(
            pipeline._valid_grounded_answer(
                "[INSUFFICIENT_EVIDENCE]", state
            )
        )

    def test_event_final_stop_completes_without_length_repair(self):
        pipeline, _generator, state = self._length_repair_pipeline_and_state()
        cancelled = set()
        pipeline._handle_event_final_completion(
            [state],
            0,
            "direct supported answer",
            {"finish_reason": "stop"},
            "FINAL",
            cancelled,
            threading.Lock(),
        )
        self.assertEqual(state.answer, "direct supported answer")
        self.assertFalse(state.final_length_repair_attempted)
        self.assertTrue(state.final_generation_done)

    def test_next_batch_bootstrap_overlaps_current_generation(self):
        retriever = _ContinuousRetriever(scan_delay=0.1)
        generator = _GateNoNativeAnswerGenerator(first_delay=0.2)
        output = _pipeline(retriever, generator).run(
            _Dataset(["q0", "q1"]), do_eval=False
        )

        self.assertLess(retriever.timestamps["batch_1_start"], generator.first_end)
        self.assertLess(
            retriever.timestamps[f"batch_1_nprobe_{BOOTSTRAP_NPROBE}"],
            generator.first_end,
        )
        self.assertEqual(
            _output(output, "pred"),
            ["native-final-answer", "native-final-answer"],
        )

    def test_production_m3_parks_generation_and_scan_until_ordered_promotion(self):
        retriever = _ContinuousRetriever(scan_delay=0.03)
        generator = _GateNoNativeAnswerGenerator(first_delay=0.12)
        pipeline = _pipeline(retriever, generator)
        pipeline.pipeline_batch_size = 1
        pipeline.prefetch_batches = 1

        output = pipeline.run(_Dataset(["q0", "q1"]), do_eval=False)

        resume = retriever.timestamps["batch_1_resume"]
        self.assertEqual(
            retriever.timestamps["batch_1_cursor_before_resume"],
            BOOTSTRAP_NPROBE,
        )
        self.assertLess(
            retriever.timestamps[f"batch_1_nprobe_{BOOTSTRAP_NPROBE}"],
            generator.first_end,
        )
        self.assertGreaterEqual(
            retriever.timestamps[f"batch_1_nprobe_{BOOTSTRAP_NPROBE + 1}"],
            resume,
        )

        q1_generation_starts = []
        for started, prompts in zip(generator.call_starts, generator.prompts):
            prompt_texts = [generator.prompt_text(prompt) for prompt in prompts]
            if any("Question: q1" in prompt for prompt in prompt_texts):
                q1_generation_starts.append(started)
        self.assertTrue(q1_generation_starts)
        self.assertGreaterEqual(min(q1_generation_starts), resume)
        for started, prompts in zip(generator.call_starts, generator.prompts):
            if started >= resume:
                continue
            self.assertFalse(
                any(
                    "Question: q1" in generator.prompt_text(prompt)
                    for prompt in prompts
                )
            )

        self.assertEqual(retriever.close_calls, {0: 1, 1: 1})
        self.assertEqual(
            _output(output, "pred"),
            ["native-final-answer", "native-final-answer"],
        )

    def test_generator_failure_unblocks_full_event_queue(self):
        pipeline = _pipeline(_ContinuousRetriever(), _FailingGenerator(), queue_size=1)
        with self.assertRaisesRegex(ValueError, "generation failed"):
            pipeline.run(_Dataset([f"q{index}" for index in range(12)]), do_eval=False)
        self.assertFalse(
            any(
                thread.name == "harmorag-progressive-retrieval" and thread.is_alive()
                for thread in threading.enumerate()
            )
        )


if __name__ == "__main__":
    unittest.main()
