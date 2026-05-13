"""Estimate global population exposure footprints from reported case locations.

This is not an outbreak-origin model. It takes the reported-case evidence panel,
keeps only records that are local enough to support granular mapping, and asks:
how many people live within specified distance footprints of reported cases?

Low-confidence country/admin centroid records are not converted into fake local
hotspots. They are counted in the diagnostics as unlocalizable evidence until a
better geocoding layer is available.

Default output:
    * data/outputs/global_exposure_v1.json
    * data/outputs/global_exposure_locations_v1.csv
    * /Volumes/HELFRICH-GD/TEG_data/outputs/global_exposure_point_supported_100km_v1.tif
      with a pointer in data/outputs/_external_index.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.layers.ghs_pop import (
    GHS_POP_2020_1KM_TIF,
    GHS_POP_2020_1KM_URL,
    GHS_POP_2020_1KM_ZIP,
    ensure_ghs_pop_2020_1km,
)
from pipeline.layers.ghs_smod import (
    GHS_SMOD_2020_URL,
    SMOD_CLASS_LABELS,
    SMOD_GROUPS,
    ensure_ghs_smod_2020_1km,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = REPO_ROOT / "data" / "outputs" / "cases_panel_v1.parquet"
OUTPUT_JSON = REPO_ROOT / "data" / "outputs" / "global_exposure_v1.json"
OUTPUT_LOCATIONS = REPO_ROOT / "data" / "outputs" / "global_exposure_locations_v1.csv"
EXTERNAL_INDEX = REPO_ROOT / "data" / "outputs" / "_external_index.json"
EXTERNAL_OUTPUT_DIR = Path("/Volumes/HELFRICH-GD/TEG_data/outputs")
DEFAULT_SURFACE = EXTERNAL_OUTPUT_DIR / "global_exposure_point_supported_100km_v1.tif"
GEOD = pyproj.Geod(ellps="WGS84")
SMOD_CLASS_KEYS = sorted(SMOD_CLASS_LABELS)
SMOD_UNCLASSIFIED_INDEX = len(SMOD_CLASS_KEYS)
SMOD_LOOKUP_OFFSET = 10000
SMOD_LOOKUP = np.full(SMOD_LOOKUP_OFFSET + max(SMOD_CLASS_KEYS) + 1, SMOD_UNCLASSIFIED_INDEX, dtype=np.int16)
for _smod_idx, _smod_cls in enumerate(SMOD_CLASS_KEYS):
    SMOD_LOOKUP[SMOD_LOOKUP_OFFSET + _smod_cls] = _smod_idx


@dataclass
class EvidenceLocation:
    location_id: int
    country: str
    lon: float
    lat: float
    x: float
    y: float
    row: float
    col: float
    records: int
    count: int
    evidence_weight: float
    first_date: str
    last_date: str


def _now_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_external_index(entry: dict[str, Any]) -> None:
    EXTERNAL_INDEX.parent.mkdir(parents=True, exist_ok=True)
    if EXTERNAL_INDEX.exists():
        payload = json.loads(EXTERNAL_INDEX.read_text(encoding="utf-8"))
    else:
        payload = {"artifacts": []}
    artifacts = [item for item in payload.get("artifacts", []) if item.get("path") != entry["path"]]
    artifacts.append(entry)
    payload["artifacts"] = sorted(artifacts, key=lambda item: item["path"])
    payload["updated_at_utc"] = _now_utc()
    EXTERNAL_INDEX.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _assert_same_grid(a, b, *, label_a: str, label_b: str) -> None:
    if a.crs != b.crs:
        raise ValueError(f"{label_a} CRS {a.crs} does not match {label_b} CRS {b.crs}")
    if (a.width, a.height) != (b.width, b.height):
        raise ValueError(
            f"{label_a} shape {(a.width, a.height)} does not match "
            f"{label_b} shape {(b.width, b.height)}"
        )
    if a.transform != b.transform:
        raise ValueError(f"{label_a} transform does not match {label_b} transform")


def _empty_settlement_accumulator() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {
        str(cls): {
            "label": label,
            "total_population": 0.0,
            "exposed_population": 0.0,
            "total_cells": 0,
            "exposed_cells": 0,
        }
        for cls, label in sorted(SMOD_CLASS_LABELS.items())
    }
    out["unclassified"] = {
        "label": "no_data_or_unclassified",
        "total_population": 0.0,
        "exposed_population": 0.0,
        "total_cells": 0,
        "exposed_cells": 0,
    }
    return out


def _smod_class_indices(smod: np.ndarray) -> np.ndarray:
    lookup_idx = smod.astype(np.int32, copy=False) + SMOD_LOOKUP_OFFSET
    valid = (lookup_idx >= 0) & (lookup_idx < len(SMOD_LOOKUP))
    out = np.full(smod.shape, SMOD_UNCLASSIFIED_INDEX, dtype=np.int16)
    out[valid] = SMOD_LOOKUP[lookup_idx[valid]]
    return out


def _finalize_settlement_summary(class_acc: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_class = {}
    for key, item in class_acc.items():
        total_population = float(item["total_population"])
        exposed_population = float(item["exposed_population"])
        by_class[key] = {
            "label": item["label"],
            "total_population": total_population,
            "exposed_population": exposed_population,
            "share_of_class_population": (
                exposed_population / total_population if total_population else None
            ),
            "total_cells": int(item["total_cells"]),
            "exposed_cells": int(item["exposed_cells"]),
        }

    by_group = {}
    for group, classes in SMOD_GROUPS.items():
        keys = [str(cls) for cls in classes]
        total_population = sum(by_class[key]["total_population"] for key in keys)
        exposed_population = sum(by_class[key]["exposed_population"] for key in keys)
        total_cells = sum(by_class[key]["total_cells"] for key in keys)
        exposed_cells = sum(by_class[key]["exposed_cells"] for key in keys)
        by_group[group] = {
            "classes": classes,
            "total_population": total_population,
            "exposed_population": exposed_population,
            "share_of_class_population": (
                exposed_population / total_population if total_population else None
            ),
            "total_cells": int(total_cells),
            "exposed_cells": int(exposed_cells),
        }

    return {"by_class": by_class, "by_group": by_group}


def _load_point_supported_locations(
    cases_path: Path,
    *,
    min_confidence: float,
    coordinate_precision: int,
    raster_crs,
    raster_transform,
    raster_shape: tuple[int, int],
) -> tuple[list[EvidenceLocation], dict[str, Any], pd.DataFrame]:
    cases = gpd.read_parquet(cases_path)
    cases["date"] = pd.to_datetime(cases["date"], errors="coerce")
    valid = cases.loc[
        cases["date"].notna()
        & cases["lon"].between(-180, 180)
        & cases["lat"].between(-90, 90)
    ].copy()
    point_supported = valid.loc[valid["confidence"] >= min_confidence].copy()
    unlocalizable = valid.loc[valid["confidence"] < min_confidence].copy()

    point_supported["lon_round"] = point_supported["lon"].round(coordinate_precision)
    point_supported["lat_round"] = point_supported["lat"].round(coordinate_precision)
    point_supported["case_count"] = point_supported["count"].fillna(1).astype(int)
    point_supported["evidence_weight"] = point_supported["case_count"] * point_supported["confidence"]

    grouped = (
        point_supported.groupby(["country", "lon_round", "lat_round"], dropna=False)
        .agg(
            records=("source", "size"),
            count=("case_count", "sum"),
            evidence_weight=("evidence_weight", "sum"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .reset_index()
        .sort_values(["country", "lon_round", "lat_round"])
        .reset_index(drop=True)
    )

    transformer = pyproj.Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    height, width = raster_shape
    locations: list[EvidenceLocation] = []
    outside_raster = 0
    for location_id, row in grouped.iterrows():
        lon = float(row.lon_round)
        lat = float(row.lat_round)
        x, y = transformer.transform(lon, lat)
        col, rr = ~raster_transform * (x, y)
        if not (0 <= rr < height and 0 <= col < width):
            outside_raster += 1
            continue
        locations.append(
            EvidenceLocation(
                location_id=int(location_id),
                country=str(row.country),
                lon=lon,
                lat=lat,
                x=float(x),
                y=float(y),
                row=float(rr),
                col=float(col),
                records=int(row.records),
                count=int(row["count"]),
                evidence_weight=float(row.evidence_weight),
                first_date=pd.Timestamp(row.first_date).date().isoformat(),
                last_date=pd.Timestamp(row.last_date).date().isoformat(),
            )
        )

    diagnostics = {
        "case_records_total": int(len(cases)),
        "case_records_with_valid_coordinates": int(len(valid)),
        "point_supported_records": int(len(point_supported)),
        "unlocalizable_records": int(len(unlocalizable)),
        "point_supported_countries": int(point_supported["country"].nunique()),
        "unlocalizable_countries": int(unlocalizable["country"].nunique()),
        "unique_point_supported_locations": int(len(grouped)),
        "locations_inside_population_raster": int(len(locations)),
        "locations_outside_population_raster": int(outside_raster),
        "unlocalizable_by_source": {
            str(k): int(v) for k, v in unlocalizable["source"].value_counts().sort_index().items()
        },
        "top_unlocalizable_countries": {
            str(k): int(v) for k, v in unlocalizable["country"].value_counts().head(20).items()
        },
    }
    return locations, diagnostics, grouped


def _geodesic_footprint_polygon(
    location: EvidenceLocation,
    *,
    radius_m: float,
    transformer: pyproj.Transformer,
    vertices: int,
) -> Polygon | None:
    azimuths = np.linspace(0.0, 360.0, vertices, endpoint=False)
    lons = np.full(vertices, location.lon, dtype=np.float64)
    lats = np.full(vertices, location.lat, dtype=np.float64)
    distances = np.full(vertices, radius_m, dtype=np.float64)
    ring_lons, ring_lats, _ = GEOD.fwd(lons, lats, azimuths, distances)
    xs, ys = transformer.transform(ring_lons, ring_lats)
    coords = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if np.isfinite(x) and np.isfinite(y)
    ]
    if len(coords) < 3:
        return None
    polygon = Polygon(coords)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None
    return polygon


def _mark_radius(
    mask: np.ndarray,
    locations: list[EvidenceLocation],
    *,
    radius_m: float,
    transform,
    raster_crs,
    vertices: int,
) -> int:
    transformer = pyproj.Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    shapes = []
    skipped = 0
    for location in locations:
        polygon = _geodesic_footprint_polygon(
            location,
            radius_m=radius_m,
            transformer=transformer,
            vertices=vertices,
        )
        if polygon is None:
            skipped += 1
            continue
        shapes.append((polygon, 1))
    if shapes:
        rasterize(
            shapes,
            out=mask,
            transform=transform,
            fill=0,
            default_value=1,
            dtype="uint8",
            all_touched=False,
        )
    return skipped


def _sum_population_and_write_surface(
    src,
    mask: np.ndarray,
    *,
    surface_path: Path | None,
    smod_src=None,
) -> tuple[float, float, int, dict[str, Any] | None]:
    exposed_population = 0.0
    total_population = 0.0
    exposed_cells = 0
    settlement = _empty_settlement_accumulator() if smod_src is not None else None
    dst = None
    if surface_path is not None:
        surface_path.parent.mkdir(parents=True, exist_ok=True)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            nodata=0,
            compress="DEFLATE",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )
        dst = rasterio.open(surface_path, "w", **profile)

    try:
        for window in _iter_windows(src.height, src.width, block_size=512):
            row0 = int(window.row_off)
            col0 = int(window.col_off)
            row1 = row0 + int(window.height)
            col1 = col0 + int(window.width)
            pop = src.read(1, window=window, masked=True).filled(0).astype(np.float64)
            pop[pop < 0] = 0
            local_mask = np.asarray(mask[row0:row1, col0:col1], dtype=np.uint8)
            total_population += float(pop.sum())
            exposed_cells += int(local_mask.sum())
            exposed_population += float(pop[local_mask == 1].sum())
            if smod_src is not None and settlement is not None:
                smod = smod_src.read(1, window=window, masked=True).filled(-9999).astype(np.int16)
                class_idx = _smod_class_indices(smod).ravel()
                pop_flat = pop.ravel()
                exposed_flat = local_mask.ravel() == 1
                minlength = SMOD_UNCLASSIFIED_INDEX + 1
                total_pop_by_class = np.bincount(class_idx, weights=pop_flat, minlength=minlength)
                exposed_pop_by_class = np.bincount(
                    class_idx[exposed_flat],
                    weights=pop_flat[exposed_flat],
                    minlength=minlength,
                )
                total_cells_by_class = np.bincount(class_idx, minlength=minlength)
                exposed_cells_by_class = np.bincount(class_idx[exposed_flat], minlength=minlength)
                for idx, cls in enumerate(SMOD_CLASS_KEYS):
                    key = str(cls)
                    settlement[key]["total_population"] += float(total_pop_by_class[idx])
                    settlement[key]["exposed_population"] += float(exposed_pop_by_class[idx])
                    settlement[key]["total_cells"] += int(total_cells_by_class[idx])
                    settlement[key]["exposed_cells"] += int(exposed_cells_by_class[idx])
                settlement["unclassified"]["total_population"] += float(total_pop_by_class[SMOD_UNCLASSIFIED_INDEX])
                settlement["unclassified"]["exposed_population"] += float(exposed_pop_by_class[SMOD_UNCLASSIFIED_INDEX])
                settlement["unclassified"]["total_cells"] += int(total_cells_by_class[SMOD_UNCLASSIFIED_INDEX])
                settlement["unclassified"]["exposed_cells"] += int(exposed_cells_by_class[SMOD_UNCLASSIFIED_INDEX])
            if dst is not None:
                dst.write(local_mask, 1, window=window)
    finally:
        if dst is not None:
            dst.close()
    settlement_summary = _finalize_settlement_summary(settlement) if settlement is not None else None
    return exposed_population, total_population, exposed_cells, settlement_summary


def _iter_windows(height: int, width: int, *, block_size: int) -> list[Window]:
    windows = []
    for row_off in range(0, height, block_size):
        for col_off in range(0, width, block_size):
            windows.append(
                Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=min(block_size, width - col_off),
                    height=min(block_size, height - row_off),
                )
            )
    return windows


def _radius_exposure(
    src,
    locations: list[EvidenceLocation],
    *,
    radius_km: float,
    surface_path: Path | None,
    geodesic_vertices: int,
    smod_src=None,
) -> dict[str, Any]:
    height, width = src.height, src.width
    radius_m = float(radius_km) * 1000.0
    t0 = time.perf_counter()
    mask = np.zeros((height, width), dtype=np.uint8)
    print(f"marking {radius_km:g} km geodesic exposure footprint for {len(locations)} locations", flush=True)
    skipped_polygons = _mark_radius(
        mask,
        locations,
        radius_m=radius_m,
        transform=src.transform,
        raster_crs=src.crs,
        vertices=geodesic_vertices,
    )
    mark_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    print(f"scanning population raster for {radius_km:g} km footprint", flush=True)
    exposed_population, total_population, exposed_cells, settlement_summary = _sum_population_and_write_surface(
        src,
        mask,
        surface_path=surface_path,
        smod_src=smod_src,
    )
    scan_seconds = time.perf_counter() - t1
    del mask

    out: dict[str, Any] = {
        "radius_km": float(radius_km),
        "exposed_population": exposed_population,
        "total_population_in_raster": total_population,
        "share_of_population": exposed_population / total_population if total_population else None,
        "exposed_cells": exposed_cells,
        "skipped_footprints": skipped_polygons,
        "timing_seconds": {
            "mark": mark_seconds,
            "scan_and_write": scan_seconds,
        },
    }
    if surface_path is not None:
        out["surface"] = {
            "path": str(surface_path),
            "sha256": _sha256(surface_path),
            "size_bytes": surface_path.stat().st_size,
        }
    if settlement_summary is not None:
        out["settlement"] = settlement_summary
    return out


def build_global_exposure(args: argparse.Namespace) -> dict[str, Any]:
    pop_path = ensure_ghs_pop_2020_1km(download=not args.no_download)
    smod_path = None if args.no_smod else ensure_ghs_smod_2020_1km(download=not args.no_download)
    radii = sorted({float(radius) for radius in args.radii_km})
    surface_radius = float(args.surface_radius_km)
    if surface_radius not in radii:
        radii.append(surface_radius)
        radii = sorted(radii)

    with rasterio.open(pop_path) as src:
        smod_src = rasterio.open(smod_path) if smod_path is not None else None
        if smod_src is not None:
            _assert_same_grid(src, smod_src, label_a="GHSL population", label_b="GHS-SMOD")
        locations, case_diagnostics, grouped = _load_point_supported_locations(
            CASES_PATH,
            min_confidence=args.min_confidence,
            coordinate_precision=args.coordinate_precision,
            raster_crs=src.crs,
            raster_transform=src.transform,
            raster_shape=(src.height, src.width),
        )
        if not locations:
            raise RuntimeError("No point-supported locations fall inside the population raster")

        location_frame = pd.DataFrame([asdict(location) for location in locations])
        OUTPUT_LOCATIONS.parent.mkdir(parents=True, exist_ok=True)
        location_frame.to_csv(OUTPUT_LOCATIONS, index=False)

        radius_results = []
        for radius in radii:
            surface_path = args.surface_path if math.isclose(radius, surface_radius) else None
            radius_results.append(
                _radius_exposure(
                    src,
                    locations,
                    radius_km=radius,
                    surface_path=surface_path,
                    geodesic_vertices=args.geodesic_vertices,
                    smod_src=smod_src,
                )
            )
        if smod_src is not None:
            smod_src.close()

    surface_result = next(
        (result for result in radius_results if math.isclose(result["radius_km"], surface_radius)),
        None,
    )
    if surface_result and "surface" in surface_result:
        _write_external_index(
            {
                **surface_result["surface"],
                "description": (
                    f"Global 1 km point-supported geodesic exposure mask within {surface_radius:g} km "
                    "of high-confidence reported hantavirus case locations."
                ),
                "created_at_utc": _now_utc(),
                "source_script": "pipeline/exposure/exposed_population.py",
            }
        )

    output = {
        "schema_version": "global_exposure_v1",
        "created_at_utc": _now_utc(),
        "interpretation": (
            "Population residing within distance footprints of high-confidence reported case "
            "locations. This is exposure accounting from reported evidence, not origin inference."
        ),
        "case_evidence": case_diagnostics,
        "parameters": {
            "min_confidence_for_granular_surface": float(args.min_confidence),
            "coordinate_precision": int(args.coordinate_precision),
            "radii_km": radii,
            "surface_radius_km": surface_radius,
            "distance_footprint": "geodesic circle rasterized to the GHSL World Mollweide grid",
            "geodesic_vertices": int(args.geodesic_vertices),
            "low_confidence_policy": (
                "country/admin centroid records are reported as unlocalizable and excluded "
                "from granular local exposure masks"
            ),
        },
        "population_raster": {
            "path": str(pop_path),
            "zip_path": str(GHS_POP_2020_1KM_ZIP),
            "source_url": GHS_POP_2020_1KM_URL,
            "sha256": _sha256(pop_path),
            "size_bytes": pop_path.stat().st_size,
            "license": "European Commission reuse policy / GHSL terms",
            "citation": "Schiavina, M., Freire, S., MacManus, K. et al. GHS-POP R2023A.",
        },
        "settlement_raster": (
            {
                "path": str(smod_path),
                "source_url": GHS_SMOD_2020_URL,
                "sha256": _sha256(smod_path),
                "size_bytes": smod_path.stat().st_size,
                "license": "European Commission reuse policy / GHSL terms",
                "citation": "Pesaresi, M., Politis, P. et al. GHS-SMOD R2023A.",
                "class_labels": {str(k): v for k, v in sorted(SMOD_CLASS_LABELS.items())},
                "groups": SMOD_GROUPS,
            }
            if smod_path is not None
            else None
        ),
        "outputs": {
            "locations_csv": {
                "path": str(OUTPUT_LOCATIONS.relative_to(REPO_ROOT)),
                "sha256": _sha256(OUTPUT_LOCATIONS),
                "size_bytes": OUTPUT_LOCATIONS.stat().st_size,
            },
            "external_index": str(EXTERNAL_INDEX.relative_to(REPO_ROOT)),
        },
        "exposure": radius_results,
        "warnings": [
            "Point-supported exposure is a lower-bound local footprint because low-confidence records are not locally mapped.",
            "Distance footprints are evidence footprints, not transmission probabilities.",
            "Rasterized geodesic footprints assign whole 1 km cells by the rasterization rule rather than sub-cell population fractions.",
            "SMOD settlement strata describe where exposed population lives; they are not a reservoir-habitat model.",
            "GHSL 2020 population is used for all case years, so historical exposure is not population-year specific.",
        ],
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build global point-supported exposure accounting outputs.")
    parser.add_argument("--radii-km", nargs="+", type=float, default=[25, 50, 100])
    parser.add_argument("--surface-radius-km", type=float, default=100)
    parser.add_argument("--min-confidence", type=float, default=0.95)
    parser.add_argument("--coordinate-precision", type=int, default=5)
    parser.add_argument("--geodesic-vertices", type=int, default=96)
    parser.add_argument("--surface-path", type=Path, default=DEFAULT_SURFACE)
    parser.add_argument("--no-smod", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    result = build_global_exposure(args)
    print(
        "global exposure: locations={locations}, point_records={point_records}, "
        "unlocalizable_records={unlocalizable}".format(
            locations=result["case_evidence"]["locations_inside_population_raster"],
            point_records=result["case_evidence"]["point_supported_records"],
            unlocalizable=result["case_evidence"]["unlocalizable_records"],
        )
    )
    for item in result["exposure"]:
        print(
            "{radius:g} km: exposed_population={pop:,.0f}, share={share:.4%}, cells={cells:,}".format(
                radius=item["radius_km"],
                pop=item["exposed_population"],
                share=item["share_of_population"] or 0,
                cells=item["exposed_cells"],
            )
        )
    print(f"Saved: {OUTPUT_JSON}")
    print(f"Locations: {OUTPUT_LOCATIONS}")


if __name__ == "__main__":
    main()
