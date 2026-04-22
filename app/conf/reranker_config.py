from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


@dataclass
class RerankerConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float


reranker_config = RerankerConfig(
    base_url=(
        os.getenv("RERANK_API_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
        or "https://dashscope.aliyuncs.com/api/v1"
    ),
    api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"),
    model=os.getenv("RERANK_API_MODEL") or "gte-rerank-v2",
    timeout_seconds=_read_float("RERANK_API_TIMEOUT_SECONDS", 30.0),
    max_retries=_read_int("RERANK_API_MAX_RETRIES", 2),
    retry_backoff_seconds=_read_float("RERANK_API_RETRY_BACKOFF_SECONDS", 1.0),
)
