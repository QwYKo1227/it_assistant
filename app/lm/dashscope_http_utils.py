import time
from typing import Any, Dict
from urllib.parse import urljoin

import requests

from app.core.logger import logger


def post_dashscope_json(
    *,
    base_url: str,
    endpoint_path: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    operation_name: str,
) -> Dict[str, Any]:
    if not api_key:
        raise ValueError(f"{operation_name} 缺少 API Key，请配置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
    if not base_url:
        raise ValueError(f"{operation_name} 缺少 Base URL，请配置对应的 API BASE URL")

    url = urljoin(base_url.rstrip("/") + "/", endpoint_path.lstrip("/"))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    attempts = max(1, int(max_retries))

    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get("code") not in (None, 200, "200"):
                raise RuntimeError(f"{operation_name} 返回业务错误: {data.get('code')} - {data.get('message')}")
            return data
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            logger.warning(
                f"{operation_name} 调用失败，第 {attempt}/{attempts} 次尝试: {exc}",
                exc_info=attempt == attempts,
            )
            if attempt >= attempts:
                raise RuntimeError(f"{operation_name} 调用失败: {exc}") from exc
            if retry_backoff_seconds > 0:
                time.sleep(retry_backoff_seconds * attempt)

    raise RuntimeError(f"{operation_name} 调用失败: 未知错误")
