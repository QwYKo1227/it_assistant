from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


@dataclass
class McpConfig:
    mcp_base_url: str
    api_key: str
    response_base_url: str
    response_model: str
    max_results: int


mcp_config = McpConfig(
    mcp_base_url=os.getenv("MCP_DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"),
    response_base_url=os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
    response_model=os.getenv("MCP_RESPONSE_MODEL") or "qwen3.5-flash",
    max_results=_read_int("MCP_WEB_SEARCH_COUNT", 5),
)
