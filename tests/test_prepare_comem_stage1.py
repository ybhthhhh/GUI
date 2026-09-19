import base64
import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


class PrepareCoMEMStageOneTests(unittest.TestCase):
    def test_success_archive_becomes_multimodal_history_samples(self):
        script = Path(__file__).parents[1] / "examples" / "prepare_comem_stage1.py"
        spec = importlib.util.spec_from_file_location("prepare_comem_stage1", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        pixel = "data:image/png;base64," + base64.b64encode(b"not-a-real-png-but-a-stable-test-payload").decode()
        source = {
            "task_description": "Buy the red mug",
            "conversation_id": "episode-1",
            "rounds": [
                {"response": "click(search)", "image_url": {"url": pixel}},
                {"response": "type(red mug)", "image_url": {"url": pixel}},
                {"response": "click(cart)", "image_url": {"url": pixel}},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "shopping.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("expand_memory/shopping/qwen2.5-vl-32b/test_1/success/task.jsonl", json.dumps(source))
                zipped.writestr("expand_memory/shopping/qwen2.5-vl-32b/test_1/fail/task.jsonl", json.dumps(source))
            manifest = root / "manifest.jsonl"
            old_argv = sys.argv
            try:
                sys.argv = [str(script), str(archive), str(manifest), "--image-root", str(root / "images")]
                module.main()
            finally:
                sys.argv = old_argv
            records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record["history"] for record in records))
            self.assertTrue(all(Path(record["current_image"]).is_file() for record in records))


if __name__ == "__main__":
    unittest.main()
