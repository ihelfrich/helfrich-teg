"""GHS-SMOD urban/rural settlement classification → cell capacity weight.

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

For SES-weighted effective distance, we map these classes to a detection-
capacity weight in (0, 1]: urban centres ≈ 1 (high detection probability),
rural ≈ low. This is the simplest defensible mapping; a calibrated mapping
from DHS wealth + healthcare-facility density is in pipeline/layers/ses_*.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform


GHS_SMOD_2020_PATH = Path(
    "/Volumes/HELFRICH-GD/KatiaBlendedFinance/raster_cache/ghs_smod/"
    "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.tif"
)

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
