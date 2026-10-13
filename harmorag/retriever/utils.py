"""Text-retrieval helpers required by HarmoRAG."""

from __future__ import annotations

import json
import re
import warnings
from typing import Any, Dict, List, Union

import datasets
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer

from harmorag.utils import get_device


_has_printed_instruction = False


def convert_numpy(obj: Union[Dict, list, np.ndarray, np.generic]) -> Any:
    """Recursively convert NumPy values into JSON-serializable values."""

    if isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def load_model(model_path: str, use_fp16: bool = False):
    """Load the dense retriever and tokenizer."""

    AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.to(get_device())
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True
    )
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    """Pool a transformer representation using the configured dense method."""

    if last_hidden_state is None and pooling_method in {"mean", "cls"}:
        warnings.warn("last_hidden_state is unavailable; using pooler_output")
        pooling_method = "pooler"
    if pooling_method == "mean":
        masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if pooling_method == "cls":
        return last_hidden_state[:, 0]
    if pooling_method == "pooler":
        return pooler_output
    raise NotImplementedError(f"unsupported pooling method: {pooling_method}")


def _looks_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def set_default_instruction(model_name, is_query=True, is_zh=False):
    """Return the model-specific retrieval prefix used by the paper."""

    lowered = str(model_name).lower()
    if "e5" in lowered:
        return "query: " if is_query else "passage: "
    if "bge" in lowered and is_query:
        if "zh" in lowered or is_zh:
            return "为这个句子生成表示以用于检索相关文章："
        return "Represent this sentence for searching relevant passages: "
    return ""


def parse_query(model_name, query_list, instruction=None, is_query=True):
    """Apply the dense retriever instruction without modifying query order."""

    global _has_printed_instruction
    if isinstance(query_list, str):
        query_list = [query_list]
    if instruction is not None:
        prefix = instruction.strip() + " "
    else:
        prefix = set_default_instruction(
            model_name,
            is_query=is_query,
            is_zh=_looks_chinese(query_list[0]) if query_list else False,
        )
    if not _has_printed_instruction:
        if prefix:
            print(f"Use `{prefix}` as retrieval instruction")
        else:
            warnings.warn("retrieval instruction is not set")
        _has_printed_instruction = True
    return [prefix + query for query in query_list]


def load_corpus(corpus_path: str):
    """Load a JSONL or Parquet corpus through Hugging Face Datasets."""

    if corpus_path.endswith(".jsonl"):
        corpus = datasets.load_dataset("json", data_files=corpus_path, split="train")
    elif corpus_path.endswith(".parquet"):
        corpus = datasets.load_dataset("parquet", data_files=corpus_path, split="train")
    else:
        raise NotImplementedError("corpus must be JSONL or Parquet")
    if "contents" not in corpus.features:
        if "text" not in corpus.features:
            raise ValueError("corpus must contain a 'contents' or 'text' field")
        corpus = corpus.map(lambda row: {"contents": row["text"]})
    return corpus


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_docs(corpus, doc_idxs: List[int]):
    """Resolve integer document identifiers against the loaded corpus."""

    return [corpus[int(index)] for index in doc_idxs]


__all__ = [
    "convert_numpy",
    "load_corpus",
    "load_docs",
    "load_model",
    "parse_query",
    "pooling",
    "read_jsonl",
    "set_default_instruction",
]
