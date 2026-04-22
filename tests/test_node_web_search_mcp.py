import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class NodeWebSearchMcpImportTests(unittest.TestCase):
    def test_module_imports_without_third_party_agents_conflict(self):
        repo_root = Path(__file__).resolve().parent.parent
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import app.query_process.agent.nodes.node_web_search_mcp; print('ok')",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

    def test_module_file_runs_directly_from_repo_root(self):
        repo_root = Path(__file__).resolve().parent.parent
        proc = subprocess.run(
            [
                sys.executable,
                "app/query_process/agent/nodes/node_web_search_mcp.py",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)


class NodeWebSearchMcpBehaviorTests(unittest.TestCase):
    def setUp(self):
        from app.query_process.agent.nodes import node_web_search_mcp as module

        module._mcp_response_client = None
        module.mcp_config.api_key = "test-key"
        module.mcp_config.mcp_base_url = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse"
        module.mcp_config.response_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        module.mcp_config.response_model = "qwen-flash"
        module.mcp_config.max_results = 3
        self.module = module

    def test_build_mcp_tool_config_uses_bearer_auth(self):
        tool = self.module._build_mcp_tool_config()

        self.assertEqual(tool["type"], "mcp")
        self.assertEqual(tool["server_protocol"], "sse")
        self.assertEqual(tool["server_label"], "bailian_web_search")
        self.assertEqual(tool["headers"]["Authorization"], "Bearer test-key")

    @patch("app.query_process.agent.nodes.node_web_search_mcp.OpenAI")
    def test_mcp_call_returns_structured_docs(self, mock_openai):
        fake_client = Mock()
        fake_response = Mock()
        fake_response.output_text = (
            '{"pages":['
            '{"title":"Result 1","url":"https://example.com/1","snippet":"Snippet 1"},'
            '{"title":"Result 2","url":"https://example.com/2","snippet":"Snippet 2"}'
            ']}'
        )
        fake_client.responses.create.return_value = fake_response
        mock_openai.return_value = fake_client

        docs = self.module.mcp_call("test query")

        self.assertEqual(
            docs,
            [
                {"title": "Result 1", "url": "https://example.com/1", "snippet": "Snippet 1"},
                {"title": "Result 2", "url": "https://example.com/2", "snippet": "Snippet 2"},
            ],
        )
        create_kwargs = fake_client.responses.create.call_args.kwargs
        self.assertEqual(create_kwargs["model"], "qwen-flash")
        self.assertEqual(create_kwargs["tools"][0]["headers"]["Authorization"], "Bearer test-key")


if __name__ == "__main__":
    unittest.main()
