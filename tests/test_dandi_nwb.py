from __future__ import annotations

import tempfile
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumulant_dynamics.dandi_nwb import DandiNWBDownloader


class DandiNWBDownloaderTest(unittest.TestCase):
    def test_downloads_only_matching_nwb_assets_in_parallel_groups(self) -> None:
        pages = iter(
            [
                {
                    "results": [
                        {"asset_id": "1", "path": "mouse1/spike_times.nwb", "download_url": "https://example/spike"},
                        {"asset_id": "2", "path": "mouse1/lfp_traces.nwb", "download_url": "https://example/lfp"},
                        {"asset_id": "3", "path": "mouse1/readme.txt", "download_url": "https://example/readme"},
                    ],
                    "next": None,
                }
            ]
        )
        downloaded: list[tuple[str, Path]] = []

        def fetch_json(_url: str) -> dict:
            return next(pages)

        def download_file(url: str, destination: Path) -> Path:
            downloaded.append((url, destination))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(url)
            return destination

        downloader = DandiNWBDownloader(
            "000001",
            fetch_json=fetch_json,
            download_file=download_file,
            max_workers=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            grouped = downloader.download_grouped_assets(
                {"spiking": ["spike"], "lpf": ["lfp"]},
                tmpdir,
            )

        self.assertEqual(set(grouped), {"spiking", "lpf"})
        self.assertEqual(len(grouped["spiking"]), 1)
        self.assertEqual(len(grouped["lpf"]), 1)
        self.assertTrue(grouped["spiking"][0].name.endswith("spike_times.nwb"))
        self.assertTrue(grouped["lpf"][0].name.endswith("lfp_traces.nwb"))
        self.assertEqual(len(downloaded), 2)
        self.assertTrue(all(path.suffix == ".nwb" for _, path in downloaded))


if __name__ == "__main__":
    unittest.main()
