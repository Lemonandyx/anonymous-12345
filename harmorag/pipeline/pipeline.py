"""Minimal pipeline base retained from the FlashRAG integration layer.

Only the evaluator/cache contract required by HarmoRAG is kept.  The many
unrelated upstream RAG pipeline implementations are intentionally excluded
from this release.
"""

from harmorag.evaluator import Evaluator
from harmorag.prompt import PromptTemplate


class BasicPipeline:
    """Base class shared by HarmoRAG pipeline implementations."""

    def __init__(self, config, prompt_template=None):
        self.config = config
        self.device = config["device"]
        self.retriever = None
        self.evaluator = Evaluator(config)
        self.save_retrieval_cache = config["save_retrieval_cache"]
        self.prompt_template = prompt_template or PromptTemplate(config)

    def run(self, dataset):  # pragma: no cover - abstract interface
        raise NotImplementedError

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """Optionally post-process, evaluate, and persist a completed dataset."""

        if pred_process_fun is not None:
            dataset = pred_process_fun(dataset)

        if do_eval:
            print(self.evaluator.evaluate(dataset))

        if self.save_retrieval_cache:
            if self.retriever is None:
                raise RuntimeError("retrieval cache requested before a retriever was attached")
            self.retriever._save_cache()

        return dataset


__all__ = ["BasicPipeline"]
