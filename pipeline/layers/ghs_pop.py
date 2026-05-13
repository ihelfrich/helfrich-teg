"""GHSL global population-layer helpers.

The exposure pipeline uses the Global Human Settlement Layer population raster
as its default world-coverage population denominator. The 2020 1 km World
Mollweide product is equal-area, aligns conceptually with the global GHS-SMOD
grid, and is small enough to scan block-wise on a laptop when kept on the
external drive.
"""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import certifi
import requests


GHS_POP_2020_1KM_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_54009_1000/"
    "V1-0/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.zip"
)

GHS_POP_CACHE_DIR = Path("/Volumes/HELFRICH-GD/TEG_data/inputs/ghsl/pop")
GHS_POP_2020_1KM_ZIP = GHS_POP_CACHE_DIR / "GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.zip"
GHS_POP_2020_1KM_TIF = GHS_POP_CACHE_DIR / "GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.tif"


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60, verify=certifi.where()) as response:
        response.raise_for_status()
        with path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _extract_first_tif(zip_path: Path, out_path: Path) -> None:
    with ZipFile(zip_path) as zf:
        tif_members = [name for name in zf.namelist() if name.lower().endswith(".tif")]
        if not tif_members:
            raise FileNotFoundError(f"No GeoTIFF member found inside {zip_path}")
        with zf.open(tif_members[0]) as src, out_path.open("wb") as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(chunk)


def ensure_ghs_pop_2020_1km(*, download: bool = True) -> Path:
    """Return the local GHSL 2020 global 1 km population GeoTIFF path.

    The checked-in code never stores this raster in git. It lives under
    `/Volumes/HELFRICH-GD/TEG_data/inputs/ghsl/pop/`.
    """
    if GHS_POP_2020_1KM_TIF.exists():
        return GHS_POP_2020_1KM_TIF
    if not GHS_POP_2020_1KM_ZIP.exists():
        if not download:
            raise FileNotFoundError(GHS_POP_2020_1KM_ZIP)
        _download(GHS_POP_2020_1KM_URL, GHS_POP_2020_1KM_ZIP)
    _extract_first_tif(GHS_POP_2020_1KM_ZIP, GHS_POP_2020_1KM_TIF)
    return GHS_POP_2020_1KM_TIF
