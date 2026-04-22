import unittest
from unittest.mock import Mock, patch

from app.lm import embedding_utils


class DashScopeEmbeddingTests(unittest.TestCase):
    def setUp(self):
        embedding_utils._bge_m3_ef = None
        embedding_utils.embedding_config.api_key = "test-key"
        embedding_utils.embedding_config.model = "text-embedding-v4"
        embedding_utils.embedding_config.base_url = "https://dashscope.aliyuncs.com/api/v1"
        embedding_utils.embedding_config.output_type = "dense&sparse"
        embedding_utils.embedding_config.dimensions = 2
        embedding_utils.embedding_config.timeout_seconds = 5.0
        embedding_utils.embedding_config.max_retries = 1
        embedding_utils.embedding_config.retry_backoff_seconds = 0.0

    @patch("app.lm.dashscope_http_utils.requests.post")
    def test_generate_embeddings_preserves_dense_sparse_shape(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "output": {
                "embeddings": [
                    {
                        "embedding": [3.0, 4.0],
                        "sparse_embedding": {"2": 6.0, "9": 8.0},
                    },
                    {
                        "embedding": [0.0, 5.0],
                        "sparse_embedding": {"3": 2.0},
                    },
                ]
            }
        }
        mock_post.return_value = response

        result = embedding_utils.generate_embeddings(["doc-1", "doc-2"])

        self.assertEqual(len(result["dense"]), 2)
        self.assertEqual(len(result["sparse"]), 2)
        self.assertAlmostEqual(result["dense"][0][0], 0.6, places=6)
        self.assertAlmostEqual(result["dense"][0][1], 0.8, places=6)
        self.assertEqual(result["sparse"][0], {2: 0.6, 9: 0.8})
        self.assertEqual(result["sparse"][1], {3: 1.0})

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "text-embedding-v4")
        self.assertEqual(payload["parameters"]["text_type"], "document")
        self.assertEqual(payload["parameters"]["output_type"], "dense&sparse")
        self.assertEqual(payload["parameters"]["dimension"], 2)
        self.assertEqual(payload["input"]["texts"], ["doc-1", "doc-2"])

    @patch("app.lm.dashscope_http_utils.requests.post")
    def test_encode_queries_switches_to_query_mode(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "output": {
                "embeddings": [
                    {
                        "embedding": [1.0, 0.0],
                        "sparse_embedding": {"7": 1.0},
                    }
                ]
            }
        }
        mock_post.return_value = response

        model = embedding_utils.get_bge_m3_ef()
        result = model.encode_queries(["what is rerank"])

        self.assertEqual(result["dense"], [[1.0, 0.0]])
        self.assertEqual(result["sparse"], [{7: 1.0}])
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["parameters"]["text_type"], "query")

    def test_generate_embeddings_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            embedding_utils.generate_embeddings([])

    @patch("app.lm.dashscope_http_utils.time.sleep")
    @patch("app.lm.dashscope_http_utils.requests.post")
    def test_generate_embeddings_retries_then_raises_helpful_error(self, mock_post, mock_sleep):
        mock_post.side_effect = embedding_utils.requests.RequestException("boom")

        with self.assertRaises(RuntimeError) as ctx:
            embedding_utils.generate_embeddings(["doc"])

        self.assertIn("DashScope Embedding API", str(ctx.exception))
        self.assertEqual(mock_post.call_count, 1)
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
