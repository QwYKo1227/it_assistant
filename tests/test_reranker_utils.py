import unittest
from unittest.mock import Mock, patch

from app.lm import reranker_utils


class DashScopeRerankerTests(unittest.TestCase):
    def setUp(self):
        reranker_utils._reranker_model = None
        reranker_utils.reranker_config.api_key = "test-key"
        reranker_utils.reranker_config.model = "gte-rerank-v2"
        reranker_utils.reranker_config.base_url = "https://dashscope.aliyuncs.com/api/v1"
        reranker_utils.reranker_config.timeout_seconds = 5.0
        reranker_utils.reranker_config.max_retries = 1
        reranker_utils.reranker_config.retry_backoff_seconds = 0.0

    @patch("app.lm.dashscope_http_utils.requests.post")
    def test_compute_score_returns_float_scores_in_order(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "output": {
                "results": [
                    {"index": 0, "relevance_score": 0.91},
                    {"index": 1, "relevance_score": 0.37},
                ]
            }
        }
        mock_post.return_value = response

        reranker = reranker_utils.get_reranker_model()
        scores = reranker.compute_score([["q1", "doc1"], ["q1", "doc2"]])

        self.assertEqual(scores, [0.91, 0.37])
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "gte-rerank-v2")
        self.assertEqual(payload["input"]["query"], "q1")
        self.assertEqual(payload["input"]["documents"], ["doc1", "doc2"])

    def test_compute_score_rejects_empty_pairs(self):
        reranker = reranker_utils.get_reranker_model()

        with self.assertRaises(ValueError):
            reranker.compute_score([])

    @patch("app.lm.dashscope_http_utils.time.sleep")
    @patch("app.lm.dashscope_http_utils.requests.post")
    def test_compute_score_retries_then_raises_helpful_error(self, mock_post, mock_sleep):
        mock_post.side_effect = reranker_utils.requests.RequestException("boom")
        reranker = reranker_utils.get_reranker_model()

        with self.assertRaises(RuntimeError) as ctx:
            reranker.compute_score([["q1", "doc1"]])

        self.assertIn("DashScope Rerank API", str(ctx.exception))
        self.assertEqual(mock_post.call_count, 1)
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
