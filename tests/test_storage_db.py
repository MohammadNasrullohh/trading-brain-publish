from __future__ import annotations

import unittest
from pathlib import Path

from trading_brain.storage_db import default_data_dir


class StorageDbPathTest(unittest.TestCase):
    def test_default_data_dir_uses_shared_folder_for_release_layout(self) -> None:
        root = Path("/home/asrulvps/trading-brain/releases/trading-brain-20260401-134548")
        self.assertEqual(default_data_dir(root), Path("/home/asrulvps/trading-brain/shared"))

    def test_default_data_dir_falls_back_to_logs_for_local_workspace(self) -> None:
        root = Path("/workspace/brain")
        self.assertEqual(default_data_dir(root), root / "logs")


if __name__ == "__main__":
    unittest.main()
