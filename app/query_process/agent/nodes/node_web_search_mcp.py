import json
import re
import sys
from typing import Any, Dict, List
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[4]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from openai import OpenAI

from app.conf.bailian_mcp_config import mcp_config
from app.core.logger import logger
from app.utils.task_utils import add_done_task, add_running_task

_mcp_response_client = None


def get_mcp_response_client() -> OpenAI:
    global _mcp_response_client
    if _mcp_response_client is None:
        if not mcp_config.api_key:
            raise ValueError("MCP WebSearch 缺少 API Key，请配置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
        if not mcp_config.response_base_url:
            raise ValueError("MCP WebSearch 缺少 Responses API Base URL，请配置 OPENAI_BASE_URL")
        _mcp_response_client = OpenAI(
            api_key=mcp_config.api_key,
            base_url=mcp_config.response_base_url,
        )
    return _mcp_response_client


def _build_mcp_tool_config() -> Dict[str, Any]:
    if not mcp_config.mcp_base_url:
        raise ValueError("MCP WebSearch 缺少服务地址，请配置 MCP_DASHSCOPE_BASE_URL")
    return {
        "type": "mcp",
        "server_protocol": "sse",
        "server_label": "bailian_web_search",
        "server_url": mcp_config.mcp_base_url,
        "require_approval": "never",
        "headers": {
            "Authorization": f"Bearer {mcp_config.api_key}",
        },
    }


def _extract_json_payload(raw_text: str) -> Dict[str, Any]:
    if not raw_text:
        return {}

    candidates = [raw_text.strip()]
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue

    logger.error(f"MCP 返回结果解析 JSON 失败: {raw_text[:200]}...")
    return {}


def _normalize_pages(data: Dict[str, Any]) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for item in data.get("pages") or []:
        if not isinstance(item, dict):
            continue
        snippet = (item.get("snippet") or "").strip()
        url = (item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        if not snippet:
            continue
        docs.append({"title": title, "url": url, "snippet": snippet})
    return docs


def mcp_call(query: str) -> List[Dict[str, str]]:
    client = get_mcp_response_client()
    logger.info(f"[MCP] 调用 Responses API + WebSearch MCP, query={query}")

    prompt = (
        "请使用已配置的 WebSearch MCP 工具搜索用户问题，并仅返回 JSON。"
        "返回格式必须严格为："
        '{"pages":[{"title":"标题","url":"链接","snippet":"摘要"}]}。'
        f"最多返回 {mcp_config.max_results} 条结果，不要输出 markdown，不要输出解释。"
        f"\n用户问题：{query}"
    )

    response = client.responses.create(
        model=mcp_config.response_model,
        input=prompt,
        tools=[_build_mcp_tool_config()],
        temperature=0,
    )
    raw_text = getattr(response, "output_text", "") or ""
    data = _extract_json_payload(raw_text)
    docs = _normalize_pages(data)
    logger.info(f"[MCP] 结构化搜索结果数量: {len(docs)}")
    return docs


def node_web_search_mcp(state):
    logger.info("---node_web_search_mcp 开始处理--")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    query = state.get("rewritten_query", "") or state.get("original_query", "")
    docs: List[Dict[str, str]] = []

    if query:
        try:
            docs = mcp_call(query)
        except Exception as exc:
            logger.error(f"MCP 搜索节点执行异常: {exc}", exc_info=True)
    else:
        logger.warning("查询词为空，跳过 MCP 搜索")

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.info("---node_web_search_mcp 处理结束---")

    if docs:
        return {"web_search_docs": docs}
    return {}


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_web_search_mcp 本地测试")
    print("=" * 50)

    test_state = {
        "session_id": "test_mcp_session",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm 到 70 mm 的局部区域，应如何设置？",
        "is_stream": False,
    }

    try:
        result_state = node_web_search_mcp(test_state)
        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        search_results = result_state.get("web_search_docs", [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
        print("=" * 50)
    except Exception as exc:
        logger.exception(f"测试运行期间发生未捕获异常: {exc}")
