import math
from typing import Any, Dict, Iterable, List

import requests

from app.conf.embedding_config import embedding_config
from app.core.logger import logger
from app.lm.dashscope_http_utils import post_dashscope_json
from app.utils.normalize_sparse_vector import normalize_sparse_vector

_bge_m3_ef = None
_EMBEDDING_ENDPOINT = "/services/embeddings/text-embedding/text-embedding"


def _normalize_dense_vector(dense_vector: Iterable[Any]) -> List[float]:
    values = [float(value) for value in dense_vector or []]
    if not values:
        return []

    l2_norm = math.sqrt(sum(value * value for value in values))
    if l2_norm < 1e-12:
        return values
    return [value / l2_norm for value in values]


def _parse_sparse_embedding(raw_sparse: Any) -> Dict[int, float]:
    if not raw_sparse:
        return {}

    if isinstance(raw_sparse, dict):
        if "indices" in raw_sparse and "values" in raw_sparse:
            return {
                int(index): float(value)
                for index, value in zip(raw_sparse.get("indices", []), raw_sparse.get("values", []))
            }
        return {int(index): float(value) for index, value in raw_sparse.items()}

    if isinstance(raw_sparse, list):
        parsed: Dict[int, float] = {}
        for item in raw_sparse:
            if isinstance(item, dict):
                index = item.get("index")
                value = item.get("value")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                index, value = item
            else:
                continue

            if index is None or value is None:
                continue
            parsed[int(index)] = float(value)
        return parsed

    return {}


class DashScopeBGEM3EmbeddingFunction:
    def __init__(self):
        self.base_url = embedding_config.base_url
        self.api_key = embedding_config.api_key
        self.model = embedding_config.model
        self.output_type = embedding_config.output_type or "dense&sparse"
        self.dimensions = embedding_config.dimensions
        self.timeout_seconds = embedding_config.timeout_seconds
        self.max_retries = embedding_config.max_retries
        self.retry_backoff_seconds = embedding_config.retry_backoff_seconds

    def _encode(self, texts: List[str], text_type: str) -> Dict[str, List[Any]]:
        if not isinstance(texts, list) or not texts:
            raise ValueError("参数 texts 必须是包含文本的非空列表")

        normalized_texts = [str(text) for text in texts]
        payload = {
            "model": self.model,
            "input": {"texts": normalized_texts},
            "parameters": {
                "text_type": text_type,
                "output_type": self.output_type,
                "dimension": self.dimensions,
            },
        }
        logger.info(
            f"调用 DashScope Embedding API: model={self.model}, text_type={text_type}, count={len(normalized_texts)}"
        )
        response = post_dashscope_json(
            base_url=self.base_url,
            endpoint_path=_EMBEDDING_ENDPOINT,
            api_key=self.api_key,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            operation_name="DashScope Embedding API",
        )

        embeddings = (response.get("output") or {}).get("embeddings") or []
        if len(embeddings) != len(normalized_texts):
            raise RuntimeError(
                f"DashScope Embedding API 返回数量异常，期望 {len(normalized_texts)} 条，实际 {len(embeddings)} 条"
            )

        dense_vectors: List[List[float]] = []
        sparse_vectors: List[Dict[int, float]] = []
        for embedding in embeddings:
            dense_vector = _normalize_dense_vector(embedding.get("embedding") or embedding.get("dense_embedding"))
            if self.dimensions and dense_vector and len(dense_vector) != self.dimensions:
                raise RuntimeError(
                    f"Embedding 维度不匹配，期望 {self.dimensions}，实际 {len(dense_vector)}。"
                    " 请检查 EMBEDDING_DIM 与现有 Milvus 集合配置。"
                )

            sparse_vector = _parse_sparse_embedding(embedding.get("sparse_embedding"))
            normalized_sparse = normalize_sparse_vector(sparse_vector)
            sparse_vectors.append({int(index): float(value) for index, value in normalized_sparse.items()})
            dense_vectors.append(dense_vector)

        return {"dense": dense_vectors, "sparse": sparse_vectors}

    def encode_documents(self, texts: List[str]) -> Dict[str, List[Any]]:
        return self._encode(texts, "document")

    def encode_queries(self, texts: List[str]) -> Dict[str, List[Any]]:
        return self._encode(texts, "query")


def get_bge_m3_ef():
    global _bge_m3_ef
    if _bge_m3_ef is None:
        logger.info(
            f"初始化 DashScope Embedding 适配器: model={embedding_config.model}, dim={embedding_config.dimensions}"
        )
        _bge_m3_ef = DashScopeBGEM3EmbeddingFunction()
    return _bge_m3_ef


def generate_embeddings(texts):
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts 必须为非空列表")
        raise ValueError("参数 texts 必须是包含文本的非空列表")

    logger.info(f"开始为 {len(texts)} 条文本生成混合向量")
    try:
        model = get_bge_m3_ef()
        result = model.encode_documents(texts)
        logger.success(f"{len(texts)} 条文本向量生成完成")
        return result
    except Exception as exc:
        logger.error(f"文本向量生成失败: {exc}", exc_info=True)
        raise
