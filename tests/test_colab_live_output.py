"""Exercise notebook output forwarding with real child processes."""

import ast
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.build_colab_4_members_notebook import build_cells


class Progress:
    def __init__(self):
        self.statuses = []

    def set_postfix_str(self, text):
        self.statuses.append(text)

    def refresh(self):
        pass


class ColabOutputTests(unittest.TestCase):
    def setUp(self):
        source = next(
            "".join(cell["source"])
            for cell in build_cells()
            if "def run_with_live_output" in "".join(cell["source"])
        )
        node = next(
            node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "run_with_live_output"
        )
        namespace = {"os": os, "sys": sys, "subprocess": subprocess, "PROJECT": Path.cwd()}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook", "exec"), namespace)
        self.run_child = namespace["run_with_live_output"]

    def test_live_stdout_stderr_carriage_return_and_idle_status(self):
        writes = []

        class Capture(io.StringIO):
            def write(self, text):
                writes.append((time.monotonic(), text))
                return super().write(text)

        progress = Progress()
        script = (
            "import sys,time; "
            "sys.stdout.write('batch 1\\r'); sys.stdout.flush(); "
            "sys.stderr.write('stderr visible\\n'); sys.stderr.flush(); "
            "time.sleep(2.5); print('finished',flush=True)"
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.stdout", Capture()):
                self.run_child([sys.executable, "-u", "-c", script], Path(directory), progress)
            log = (Path(directory) / "console.log").read_bytes()
        first = next(at for at, text in writes if "batch 1" in text)
        last = next(at for at, text in writes if "finished" in text)
        self.assertGreater(last - first, 1)
        self.assertIn(b"batch 1\r", log)
        self.assertIn(b"stderr visible", log)
        self.assertTrue(progress.statuses)

    def test_failure_preserves_error_log(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.stdout", io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError) as caught:
                    self.run_child(
                        [sys.executable, "-c", "import sys; print('bad argument',file=sys.stderr); sys.exit(2)"],
                        Path(directory), Progress(),
                    )
            self.assertEqual(caught.exception.returncode, 2)
            self.assertIn("bad argument", (Path(directory) / "console.log").read_text())


if __name__ == "__main__":
    unittest.main()
