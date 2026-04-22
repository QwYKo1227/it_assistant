from collections import OrderedDict
from typing import List, Sequence, Tuple

import requests

from app.conf.reranker_config import reranker_config
from app.core.logger import logger
from app.lm.dashscope_http_utils import post_dashscope_json

_reranker_model = None
_RERANK_ENDPOINT = "/services/rerank/text-rerank/text-rerank"


class DashScopeReranker:
    def __init__(self):
        self.base_url = reranker_config.base_url
        self.api_key = reranker_config.api_key
        self.model = reranker_config.model
        self.timeout_seconds = reranker_config.timeout_seconds
        self.max_retries = reranker_config.max_retries
        self.retry_backoff_seconds = reranker_config.retry_backoff_seconds

    def _group_sentence_pairs(self, sentence_pairs: Sequence[Sequence[str]]) -> OrderedDict[str, List[Tuple[int, str]]]:
        grouped: OrderedDict[str, List[Tuple[int, str]]] = OrderedDict()
        for index, pair in enumerate(sentence_pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("sentence_pairs 必须是形如 [[query, document], ...] 的非空列表")
            query, document = str(pair[0]), str(pair[1])
            grouped.setdefault(query, []).append((index, document))
        return grouped

    def _call_rerank(self, query: str, documents: List[str]) -> List[float]:
        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "top_n": len(documents),
                "return_documents": False,
            },
        }
        logger.info(f"调用 DashScope Rerank API: model={self.model}, docs={len(documents)}")
        response = post_dashscope_json(
            base_url=self.base_url,
            endpoint_path=_RERANK_ENDPOINT,
            api_key=self.api_key,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            operation_name="DashScope Rerank API",
        )
        results = (response.get("output") or {}).get("results") or []
        scores = [0.0] * len(documents)
        for position, result in enumerate(results):
            result_index = result.get("index", position)
            if 0 <= int(result_index) < len(documents):
                scores[int(result_index)] = float(result.get("relevance_score", 0.0))
        return scores

    def compute_score(self, sentence_pairs: Sequence[Sequence[str]]) -> List[float]:
        if not sentence_pairs:
            raise ValueError("sentence_pairs 不能为空")

        grouped = self._group_sentence_pairs(sentence_pairs)
        scores = [0.0] * len(sentence_pairs)

        for query, query_documents in grouped.items():
            documents = [document for _, document in query_documents]
            group_scores = self._call_rerank(query, documents)
            if len(group_scores) != len(documents):
                raise RuntimeError(
                    f"DashScope Rerank API 返回数量异常，期望 {len(documents)} 条，实际 {len(group_scores)} 条"
                )
            for (original_index, _), score in zip(query_documents, group_scores):
                scores[original_index] = float(score)

        return scores


def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        logger.info(f"初始化 DashScope Rerank 适配器: model={reranker_config.model}")
        _reranker_model = DashScopeReranker()
    return _reranker_model
