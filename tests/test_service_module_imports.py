import subprocess
import sys
import unittest
import os
from pathlib import Path


class ServiceModuleImportTests(unittest.TestCase):
    def test_file_import_service_imports_local_app_package(self):
        repo_root = Path(__file__).resolve().parent.parent
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import app.import_process.api.file_import_service; print('ok')",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

    def test_query_service_imports_local_app_package(self):
        repo_root = Path(__file__).resolve().parent.parent
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import app.query_process.api.query_service; print('ok')",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

    def test_file_import_service_can_be_executed_directly(self):
        repo_root = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env["IT_ASSISTANT_IMPORT_ONLY"] = "1"
        proc = subprocess.run(
            [
                sys.executable,
                "app/import_process/api/file_import_service.py",
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

    def test_query_service_can_be_executed_directly(self):
        repo_root = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env["IT_ASSISTANT_IMPORT_ONLY"] = "1"
        proc = subprocess.run(
            [
                sys.executable,
                "app/query_process/api/query_service.py",
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)


if __name__ == "__main__":
    unittest.main()
