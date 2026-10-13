"""Runtime factories for the trimmed HarmoRAG distribution."""

from __future__ import annotations


def get_device() -> str:
    """Return the default PyTorch device without selecting a CUDA device."""

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def get_generator(config, **params):
    """Construct the vLLM adapter supported by the HarmoRAG pipeline."""

    framework = str(config["framework"]).lower()
    if framework != "vllm":
        raise ValueError(
            "HarmoRAG's progressive serving path requires framework='vllm'; "
            f"received {framework!r}."
        )
    from harmorag.generator import VLLMGenerator

    return VLLMGenerator(config, **params)


def get_reranker(config):
    """Reject unsupported upstream rerankers with an actionable error."""

    raise NotImplementedError(
        "The trimmed HarmoRAG release does not bundle FlashRAG rerankers. "
        "Set use_reranker: false (the configuration used in the paper)."
    )


__all__ = ["get_device", "get_generator", "get_reranker"]
