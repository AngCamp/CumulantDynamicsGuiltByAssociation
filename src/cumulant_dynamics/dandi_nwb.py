from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import json


JsonFetcher = Callable[[str], dict]
FileDownloader = Callable[[str, Path], Path]


@dataclass(frozen=True)
class DandiAsset:
    """A DANDI asset that may contain NWB spike or LFP data."""

    identifier: str
    filename: str
    download_url: str
    path: str = ""
    metadata: dict = field(default_factory=dict)


class DandiNWBDownloader:
    """Download NWB files from DANDI and fan them out in parallel."""

    def __init__(
        self,
        dandiset_id: str,
        version: str = "draft",
        api_base: str = "https://api.dandiarchive.org/api",
        max_workers: int | None = None,
        fetch_json: JsonFetcher | None = None,
        download_file: FileDownloader | None = None,
    ) -> None:
        self.dandiset_id = dandiset_id
        self.version = version
        self.api_base = api_base.rstrip("/")
        self.max_workers = max_workers
        self._fetch_json = fetch_json or self._default_fetch_json
        self._download_file = download_file or self._default_download_file

    def iter_assets(self) -> list[DandiAsset]:
        """Return all assets in the Dandiset, following pagination."""
        url = f"{self.api_base}/dandisets/{self.dandiset_id}/versions/{self.version}/assets/"
        assets: list[DandiAsset] = []
        while url:
            payload = self._fetch_json(url)
            for item in payload.get("results", []):
                filename = item.get("path") or item.get("filename") or ""
                assets.append(
                    DandiAsset(
                        identifier=str(item.get("asset_id") or item.get("identifier") or filename),
                        filename=Path(filename).name,
                        download_url=item.get("download_url") or item.get("url") or "",
                        path=filename,
                        metadata=item,
                    )
                )
            next_url = payload.get("next") or ""
            url = urljoin(url, next_url) if next_url else ""
        return assets

    def iter_nwb_assets(self) -> list[DandiAsset]:
        """Return only NWB assets."""
        return [asset for asset in self.iter_assets() if asset.filename.lower().endswith(".nwb")]

    def download_grouped_assets(
        self,
        group_patterns: Mapping[str, Iterable[str]],
        output_dir: str | Path,
    ) -> dict[str, list[Path]]:
        """Download matching NWB files in parallel and group them by label."""
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        grouped_assets: dict[str, list[DandiAsset]] = {label: [] for label in group_patterns}
        for asset in self.iter_nwb_assets():
            asset_text = f"{asset.filename} {asset.path}".lower()
            for label, patterns in group_patterns.items():
                if any(pattern.lower() in asset_text for pattern in patterns):
                    grouped_assets[label].append(asset)

        downloads: dict[str, list[Path]] = {label: [] for label in grouped_assets}
        tasks: list[tuple[str, DandiAsset]] = [
            (label, asset) for label, assets in grouped_assets.items() for asset in assets
        ]

        if not tasks:
            return downloads

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._download_file,
                    asset.download_url,
                    output_root / label / asset.filename,
                ): label
                for label, asset in tasks
            }
            for future in futures:
                downloaded_path = future.result()
                downloads[futures[future]].append(downloaded_path)

        return downloads

    @staticmethod
    def _default_fetch_json(url: str) -> dict:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _default_download_file(url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url) as response, destination.open("wb") as sink:
            sink.write(response.read())
        return destination
