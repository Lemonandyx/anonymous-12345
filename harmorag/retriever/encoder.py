from typing import List, Union

import torch
import numpy as np
from tqdm import tqdm

from harmorag.retriever.utils import load_model, parse_query, pooling
from harmorag.utils import get_device


class Encoder:
    """
    Encoder class for encoding queries using a specified model.

    Attributes:
        model_name (str): The name of the model.
        model_path (str): The path to the model.
        pooling_method (str): The method used for pooling.
        max_length (int): The maximum length of the input sequences.
        use_fp16 (bool): Whether to use FP16 precision.
        instruction (str): Additional instructions for parsing queries.

    Methods:
        encode(query_list: List[str], is_query=True) -> np.ndarray:
            Encodes a list of queries into embeddings.
    """

    def __init__(
        self,
        model_name,
        model_path,
        pooling_method,
        max_length,
        use_fp16=True,
        instruction=None,
        silent=False,
        device=None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.instruction = instruction
        self.silent = silent
        self.device = device or get_device()
        self.gpu_num = torch.cuda.device_count()
        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.to(self.device)

    @torch.inference_mode()
    def single_batch_encode(self, query_list: Union[List[str], str], is_query=True) -> np.ndarray:
        query_list = parse_query(self.model_name, query_list, self.instruction, is_query)

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "T5" in type(self.model).__name__ or (
            isinstance(self.model, torch.nn.DataParallel) and "T5" in type(self.model.module).__name__
        ):
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            output = self.model(**inputs, return_dict=True)
            pooler_output = output.get("pooler_output", None)
            last_hidden_state = output.get("last_hidden_state", None)
            query_emb = pooling(pooler_output, last_hidden_state, inputs["attention_mask"], self.pooling_method)
        if "dpr" not in self.model_name:
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        return query_emb
    @torch.inference_mode()
    def encode(self, query_list: List[str], batch_size=64, is_query=True) -> np.ndarray:
        query_emb = []
        for i in tqdm(range(0, len(query_list), batch_size), desc="Encoding process: ", disable=self.silent):
            query_emb.append(self.single_batch_encode(query_list[i : i + batch_size], is_query))
        query_emb = np.concatenate(query_emb, axis=0)
        return query_emb

    @torch.inference_mode()
    def multi_gpu_encode(self, query_list: Union[List[str], str], batch_size=64, is_query=True) -> np.ndarray:
        if self.gpu_num > 1:
            self.model = torch.nn.DataParallel(self.model)
        query_emb = self.encode(query_list, batch_size, is_query)
        return query_emb


class STEncoder:
    """
    STEncoder class for encoding queries using SentenceTransformers.

    Attributes:
        model_name (str): The name of the model.
        model_path (str): The path to the model.
        max_length (int): The maximum length of the input sequences.
        use_fp16 (bool): Whether to use FP16 precision.
        instruction (str): Additional instructions for parsing queries.

    Methods:
        encode(query_list: List[str], batch_size=64, is_query=True) -> np.ndarray:
            Encodes a list of queries into embeddings.
        multi_gpu_encode(query_list: List[str], is_query=True, batch_size=None) -> np.ndarray:
            Encodes a list of queries into embeddings using multiple GPUs.
    """

    def __init__(
        self,
        model_name,
        model_path,
        max_length,
        use_fp16,
        instruction,
        silent=False,
        device=None,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model_path = model_path
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.instruction = instruction
        self.silent = silent
        self.device = device or get_device()
        self.model = SentenceTransformer(
            model_path,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16 if use_fp16 else torch.float},
            device=self.device,
        )

    @torch.inference_mode()
    def encode(self, query_list: Union[List[str], str], batch_size=64, is_query=True) -> np.ndarray:
        query_list = parse_query(self.model_name, query_list, self.instruction, is_query)
        query_emb = self.model.encode(
            query_list,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=not self.silent,
        )
        query_emb = query_emb.astype(np.float32, order="C")

        return query_emb

    @torch.inference_mode()
    def multi_gpu_encode(self, query_list: Union[List[str], str], batch_size=None, is_query=True) -> np.ndarray:
        query_list = parse_query(self.model_name, query_list, self.instruction, is_query)
        pool = self.model.start_multi_process_pool()
        query_emb = self.model.encode_multi_process(
            query_list,
            pool,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=not self.silent,
        )
        self.model.stop_multi_process_pool(pool)
        query_emb = query_emb.astype(np.float32, order="C")

        return query_emb
