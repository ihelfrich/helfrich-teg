"""GHS-SMOD urban/rural settlement classification helpers.

The Global Human Settlement Layer's Settlement Model Grid (SMOD) classifies
each ~1km cell as one of:
    30 — urban centre
    23 — dense urban cluster
    22 — semi-dense urban cluster
    21 — suburban / peri-urban
    13 — rural cluster
    12 — low-density rural
    11 — very low density rural
    10 — water
    NoData — non-land

The exposure pipeline uses these classes for data-driven stratification of
exposed population. The legacy `SMOD_TO_WEIGHT` mapping remains only for graph
smoke diagnostics; it is not a calibrated SES model.
"""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import certifi
import numpy as np
import rasterio
import requests
from rasterio.warp import reproject, Resampling, calculate_default_transform


GHS_SMOD_2020_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000/"
    "V2-0/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.zip"
)

GHS_SMOD_2020_PATH = Path(
    "/Volumes/HELFRICH-GD/KatiaBlendedFinance/raster_cache/ghs_smod/"
    "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.tif"
)
GHS_SMOD_CACHE_DIR = Path("/Volumes/HELFRICH-GD/TEG_data/inputs/ghsl/smod")
GHS_SMOD_2020_ZIP = GHS_SMOD_CACHE_DIR / "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.zip"
GHS_SMOD_2020_TIF = GHS_SMOD_CACHE_DIR / "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.tif"

SMOD_CLASS_LABELS = {
    30: "urban_centre",
    23: "dense_urban_cluster",
    22: "semi_dense_urban_cluster",
    21: "suburban_or_peri_urban",
    13: "rural_cluster",
    12: "low_density_rural",
    11: "very_low_density_rural",
    10: "water",
}

SMOD_GROUPS = {
    "urban": [21, 22, 23, 30],
    "rural": [11, 12, 13],
    "water": [10],
}

# Class → detection-capacity weight (rough; replaced by DHS/OSM-calibrated values later)
SMOD_TO_WEIGHT = {
    30: 1.00,   # urban centre
    23: 0.85,
    22: 0.70,
    21: 0.55,
    13: 0.35,
    12: 0.20,
    11: 0.10,
    10: 0.05,   # water (negligible)
}


def load_smod_to_grid(
    smod_path: Path | str,
    target_transform,
    target_shape: tuple[int, int],
    target_crs,
) -> np.ndarray:
    """Reproject GHS-SMOD to align with a target grid (e.g. WorldPop coarsened).

    Returns an integer-class array; pass through `smod_to_weight()` to get the
    detection-capacity multiplier.
    """
    with rasterio.open(smod_path) as src:
        dst = np.zeros(target_shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.mode,  # categorical
        )
    return dst


def smod_to_weight(classes: np.ndarray) -> np.ndarray:
    """Map SMOD class array to detection-capacity weights in (0, 1]."""
    out = np.zeros_like(classes, dtype=np.float32)
    for cls, w in SMOD_TO_WEIGHT.items():
        out[classes == cls] = w
    # Fallback for unmapped values: treat as very low density rural
    out[out == 0] = 0.05
    return out


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
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(tif_members[0]) as src, out_path.open("wb") as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(chunk)


def ensure_ghs_smod_2020_1km(*, download: bool = True) -> Path:
    """Return a local GHSL 2020 global 1 km SMOD GeoTIFF path.

    Ian already has the global SMOD raster in the KatiaBlendedFinance cache.
    If that cache is absent, the function falls back to the project external
    cache under `/Volumes/HELFRICH-GD/TEG_data/inputs/ghsl/smod/`.
    """
    if GHS_SMOD_2020_PATH.exists():
        return GHS_SMOD_2020_PATH
    if GHS_SMOD_2020_TIF.exists():
        return GHS_SMOD_2020_TIF
    if not GHS_SMOD_2020_ZIP.exists():
        if not download:
            raise FileNotFoundError(GHS_SMOD_2020_ZIP)
        _download(GHS_SMOD_2020_URL, GHS_SMOD_2020_ZIP)
    _extract_first_tif(GHS_SMOD_2020_ZIP, GHS_SMOD_2020_TIF)
    return GHS_SMOD_2020_TIF
