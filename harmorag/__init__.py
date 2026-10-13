"""HarmoRAG public Python API.

HarmoRAG extends IVF retrieval with rank-aware early termination and overlaps
incremental evidence publication with vLLM generation.  The package keeps the
public surface deliberately small; research-only baselines from the upstream
FlashRAG tree are not re-exported here.
"""

from harmorag.config import Config
from harmorag.dataset import Dataset, Item
from harmorag.pipeline import ProgressiveAsyncPipeline
from harmorag.retriever import ProgressiveIVFRetriever

__all__ = [
    "Config",
    "Dataset",
    "Item",
    "ProgressiveAsyncPipeline",
    "ProgressiveIVFRetriever",
]

__version__ = "0.1.0"
