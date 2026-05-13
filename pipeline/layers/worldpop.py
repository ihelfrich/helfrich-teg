"""WorldPop population-grid loader.

Reads a country WorldPop GeoTIFF and produces a coarsened, masked population
grid suitable for graph construction. Default coarsening: 10km, which trades
spatial detail for tractable graph sizes (a 1M-cell graph is the upper bound
for the SciPy sparse-Dijkstra step in distance/effdist_grid.py).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


WORLDPOP_BR_2020_1KM = Path(
    "/Volumes/HELFRICH-GD/KatiaBlendedFinance/raster_cache/worldpop/"
    "bra_ppp_2020_1km_UNadj.tif"
)


def load_population_grid(
    src_path: Path | str,
    bbox: tuple[float, float, float, float] | None = None,
    coarsen_factor: int = 10,
) -> dict:
    """Read a WorldPop GeoTIFF and coarsen to a tractable resolution.

    Args:
        src_path: Path to the WorldPop GeoTIFF (e.g., bra_ppp_2020_1km_UNadj.tif).
        bbox: Optional (xmin, ymin, xmax, ymax) in the raster's native CRS.
              If None, the full raster extent is read.
        coarsen_factor: Decimation factor applied via Resampling.sum aggregation,
                        so a coarsen_factor of 10 on a 1km grid yields a 10km
                        population sum per cell.

    Returns:
        dict with keys:
            pop:        (H, W) ndarray of population counts per coarsened cell.
            transform:  Affine transform of the coarsened grid (rasterio).
            crs:        Coordinate reference system.
            bbox:       (xmin, ymin, xmax, ymax) of the coarsened grid.
            shape:      (H, W).
    """
    with rasterio.open(src_path) as r:
        if bbox is not None:
            window = from_bounds(*bbox, r.transform)
        else:
            window = None
        if window is not None:
            full = r.read(1, window=window)
            full_transform = r.window_transform(window)
        else:
            full = r.read(1)
            full_transform = r.transform
        nodata = r.nodata if r.nodata is not None else -99999
        full = np.where(full == nodata, 0, full)
        full = np.maximum(full, 0)

    # Aggregate by sum over (coarsen_factor x coarsen_factor) blocks
    H, W = full.shape
    Hc = (H // coarsen_factor) * coarsen_factor
    Wc = (W // coarsen_factor) * coarsen_factor
    cropped = full[:Hc, :Wc]
    coarse = cropped.reshape(
        Hc // coarsen_factor, coarsen_factor,
        Wc // coarsen_factor, coarsen_factor,
    ).sum(axis=(1, 3))

    # Build a new affine transform for the coarsened grid
    a, b, c, d, e, f = (
        full_transform.a * coarsen_factor,
        full_transform.b,
        full_transform.c,
        full_transform.d,
        full_transform.e * coarsen_factor,
        full_transform.f,
    )
    from affine import Affine
    new_transform = Affine(a, b, c, d, e, f)

    new_h, new_w = coarse.shape
    xmin = new_transform.c
    ymax = new_transform.f
    xmax = xmin + a * new_w
    ymin = ymax + e * new_h

    return {
        "pop": coarse.astype(np.float32),
        "transform": new_transform,
        "crs": r.crs,
        "bbox": (xmin, ymin, xmax, ymax),
        "shape": coarse.shape,
        "cell_size_native_units": abs(a),
    }


def populated_cell_mask(pop: np.ndarray, min_pop: float = 1.0) -> np.ndarray:
    """Boolean mask of cells with at least `min_pop` people.

    Used downstream to restrict graph construction to populated cells only,
    which is the dominant cost reducer for continental-scale grids.
    """
    return pop >= min_pop


if __name__ == "__main__":
    out = load_population_grid(WORLDPOP_BR_2020_1KM, coarsen_factor=10)
    pop = out["pop"]
    print(f"Brazil 2020 WorldPop → {out['cell_size_native_units']:.4f}-degree cells")
    print(f"  shape:           {pop.shape}")
    print(f"  bbox:            {out['bbox']}")
    print(f"  total population: {pop.sum():,.0f}")
    print(f"  populated cells:  {populated_cell_mask(pop).sum():,}")
