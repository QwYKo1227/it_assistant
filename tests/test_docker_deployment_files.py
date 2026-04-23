from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DockerDeploymentFilesTest(unittest.TestCase):
    def test_dockerfile_and_compose_define_two_app_services(self):
        dockerfile_path = PROJECT_ROOT / "Dockerfile"
        compose_path = PROJECT_ROOT / "docker-compose.yml"

        self.assertTrue(dockerfile_path.exists(), "Dockerfile should exist")
        self.assertTrue(compose_path.exists(), "docker-compose.yml should exist")

        dockerfile = dockerfile_path.read_text(encoding="utf-8")
        compose = compose_path.read_text(encoding="utf-8")

        self.assertIn("python:3.12-slim", dockerfile)
        self.assertIn("uv sync --frozen --no-dev", dockerfile)
        self.assertIn('PATH="/app/.venv/bin:$PATH"', dockerfile)

        self.assertIn("import-service:", compose)
        self.assertIn("query-service:", compose)
        self.assertIn("uvicorn app.import_process.api.file_import_service:app --host 0.0.0.0 --port 8000", compose)
        self.assertIn("uvicorn app.query_process.api.query_service:app --host 0.0.0.0 --port 8001", compose)
        self.assertIn("./.env:/app/.env:ro", compose)
        self.assertIn("./output:/app/output", compose)
        self.assertIn("./logs:/app/logs", compose)
        self.assertIn("PROJECT_ROOT: /app", compose)
        self.assertIn('TF_CPP_MIN_LOG_LEVEL: "3"', compose)


if __name__ == "__main__":
    unittest.main()
