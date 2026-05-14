"""Estimate global population exposure footprints from reported case evidence.

This is not an outbreak-origin model. It takes the reported-case evidence panel,
keeps point records local enough to support granular distance footprints, and
asks: how many people live in population cells plausibly covered by the reported
exposure evidence?

Low-confidence country/admin centroid records are not converted into fake local
hotspots. Country-only records stay unlocalizable. Admin1-supported records are
rasterized as a separate broad areal tier when their country/admin1 labels match
Natural Earth Admin 1 boundaries without ambiguity.

Default output:
    * data/outputs/global_exposure_v1.json
    * data/outputs/global_exposure_locations_v1.csv
    * data/outputs/global_exposure_admin_v1.csv
    * /Volumes/HELFRICH-GD/TEG_data/outputs/global_exposure_point_supported_100km_v1.tif
    * /Volumes/HELFRICH-GD/TEG_data/outputs/global_exposure_admin_supported_v1.tif
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

from pipeline.layers.admin_boundaries import (
    NE_ADMIN1_CACHE,
    NE_ADMIN1_URL,
    build_admin1_index,
    load_admin1_boundaries,
    match_admin1,
    normalize_label,
)
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
OUTPUT_ADMIN = REPO_ROOT / "data" / "outputs" / "global_exposure_admin_v1.csv"
EXTERNAL_INDEX = REPO_ROOT / "data" / "outputs" / "_external_index.json"
EXTERNAL_OUTPUT_DIR = Path("/Volumes/HELFRICH-GD/TEG_data/outputs")
DEFAULT_SURFACE = EXTERNAL_OUTPUT_DIR / "global_exposure_point_supported_100km_v1.tif"
DEFAULT_ADMIN_SURFACE = EXTERNAL_OUTPUT_DIR / "global_exposure_admin_supported_v1.tif"
ADMIN_MATCH_COLUMNS = [
    "country",
    "admin1",
    "normalized_country",
    "normalized_admin1",
    "records",
    "count",
    "evidence_weight",
    "first_date",
    "last_date",
    "sources",
    "matched",
    "match_status",
    "match_key",
    "matched_name",
    "matched_name_en",
    "iso_3166_2",
    "adm1_code",
    "boundary_index",
]
GEOD = pyproj.Geod(ellps="WGS84")
SMOD_CLASS_KEYS = sorted(SMOD_CLASS_LABELS)
SMOD_UNCLASSIFIED_INDEX = len(SMOD_CLASS_KEYS)
SMOD_LOOKUP_OFFSET = 10000
SMOD_LOOKUP = np.full(SMOD_LOOKUP_OFFSET + max(SMOD_CLASS_KEYS) + 1, SMOD_UNCLASSIFIED_INDEX, dtype=np.int16)
for _smod_idx, _smod_cls in enumerate(SMOD_CLASS_KEYS):
    SMOD_LOOKUP[SMOD_LOOKUP_OFFSET + _smod_cls] = _smod_idx
POPULATION_SMOD_CLASSES = [11, 12, 13, 21, 22, 23, 30]


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


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_external_index(entry: dict[str, Any], *, index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        payload = {"artifacts": []}
    artifacts = [item for item in payload.get("artifacts", []) if item.get("path") != entry["path"]]
    artifacts.append(entry)
    payload["artifacts"] = sorted(artifacts, key=lambda item: item["path"])
    payload["updated_at_utc"] = _now_utc()
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    dated = cases.loc[cases["date"].notna()].copy()
    valid_coords = dated["lon"].between(-180, 180) & dated["lat"].between(-90, 90)
    valid = dated.loc[valid_coords].copy()
    point_supported = valid.loc[valid["confidence"] >= min_confidence].copy()
    below_point_confidence = dated.loc[dated["confidence"] < min_confidence].copy()

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
        "case_records_with_valid_dates": int(len(dated)),
        "case_records_with_valid_coordinates": int(len(valid)),
        "point_supported_records": int(len(point_supported)),
        "point_surface_records": int(sum(location.records for location in locations)),
        "below_point_confidence_records": int(len(below_point_confidence)),
        "unlocalizable_records": int(len(below_point_confidence)),
        "point_supported_countries": int(point_supported["country"].nunique()),
        "below_point_confidence_countries": int(below_point_confidence["country"].nunique()),
        "unique_point_supported_locations": int(len(grouped)),
        "locations_inside_population_raster": int(len(locations)),
        "locations_outside_population_raster": int(outside_raster),
        "below_point_confidence_by_source": {
            str(k): int(v)
            for k, v in below_point_confidence["source"].value_counts().sort_index().items()
        },
        "top_below_point_confidence_countries": {
            str(k): int(v)
            for k, v in below_point_confidence["country"].value_counts().head(20).items()
        },
    }
    return locations, diagnostics, grouped


def _load_admin_supported_evidence(
    cases_path: Path,
    *,
    point_confidence: float,
    admin_min_confidence: float,
    output_path: Path,
    download: bool,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, Any]]:
    cases = gpd.read_parquet(cases_path)
    cases["date"] = pd.to_datetime(cases["date"], errors="coerce")
    has_point_coordinates = cases["lon"].between(-180, 180) & cases["lat"].between(-90, 90)
    consumed_by_point_tier = has_point_coordinates & (cases["confidence"] >= point_confidence)
    valid = cases.loc[
        cases["date"].notna()
        & cases["admin1"].notna()
        & (cases["admin1"].astype(str).str.strip() != "")
        & (cases["confidence"] >= admin_min_confidence)
        & ~consumed_by_point_tier
    ].copy()
    if valid.empty:
        empty = pd.DataFrame(columns=ADMIN_MATCH_COLUMNS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(output_path, index=False)
        return gpd.GeoDataFrame(), empty, {
            "candidate_records": 0,
            "candidate_count": 0,
            "candidate_countries": 0,
            "unique_admin_labels": 0,
            "matched_records": 0,
            "matched_count": 0,
            "matched_unique_admin_labels": 0,
            "matched_unique_admin_units": 0,
            "unmatched_records": 0,
            "unmatched_count": 0,
            "unmatched_unique_admin_labels": 0,
            "top_unmatched_admin_labels": [],
        }

    valid["case_count"] = valid["count"].fillna(1).astype(int)
    valid["evidence_weight"] = valid["case_count"] * valid["confidence"]
    grouped = (
        valid.groupby(["country", "admin1"], dropna=False)
        .agg(
            records=("source", "size"),
            count=("case_count", "sum"),
            evidence_weight=("evidence_weight", "sum"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            sources=("source", lambda x: ",".join(sorted(set(map(str, x))))),
        )
        .reset_index()
        .sort_values(["country", "admin1"])
        .reset_index(drop=True)
    )

    boundaries = load_admin1_boundaries(download=download)
    index = build_admin1_index(boundaries)
    rows = []
    matched_indices: set[int] = set()
    for _, row in grouped.iterrows():
        match = match_admin1(row.country, row.admin1, boundaries, index)
        matched = match.status == "matched" and match.index is not None
        if matched:
            matched_indices.add(int(match.index))
        rows.append(
            {
                "country": str(row.country),
                "admin1": str(row.admin1),
                "normalized_country": normalize_label(row.country),
                "normalized_admin1": normalize_label(row.admin1),
                "records": int(row.records),
                "count": int(row["count"]),
                "evidence_weight": float(row.evidence_weight),
                "first_date": pd.Timestamp(row.first_date).date().isoformat(),
                "last_date": pd.Timestamp(row.last_date).date().isoformat(),
                "sources": str(row.sources),
                "matched": bool(matched),
                "match_status": match.status,
                "match_key": match.match_key,
                "matched_name": match.matched_name,
                "matched_name_en": match.matched_name_en,
                "iso_3166_2": match.iso_3166_2,
                "adm1_code": match.adm1_code,
                "boundary_index": match.index,
            }
        )

    match_frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    match_frame.to_csv(output_path, index=False)
    matched_boundaries = boundaries.loc[sorted(matched_indices)].copy()

    diagnostics = {
        "candidate_records": int(valid.shape[0]),
        "candidate_count": int(valid["case_count"].sum()),
        "candidate_countries": int(valid["country"].nunique()),
        "unique_admin_labels": int(grouped.shape[0]),
        "matched_records": int(match_frame.loc[match_frame["matched"], "records"].sum()),
        "matched_count": int(match_frame.loc[match_frame["matched"], "count"].sum()),
        "matched_unique_admin_labels": int(match_frame["matched"].sum()),
        "matched_unique_admin_units": int(len(matched_indices)),
        "unmatched_records": int(match_frame.loc[~match_frame["matched"], "records"].sum()),
        "unmatched_count": int(match_frame.loc[~match_frame["matched"], "count"].sum()),
        "unmatched_unique_admin_labels": int((~match_frame["matched"]).sum()),
        "top_unmatched_admin_labels": match_frame.loc[
            ~match_frame["matched"],
            ["country", "admin1", "records"],
        ]
        .sort_values("records", ascending=False)
        .head(25)
        .to_dict(orient="records"),
    }
    return matched_boundaries, match_frame, diagnostics


def _admin_supported_exposure(
    src,
    admin_boundaries: gpd.GeoDataFrame,
    *,
    surface_path: Path | None,
    smod_src=None,
    allowed_smod_classes: list[int] | None = None,
) -> dict[str, Any] | None:
    if admin_boundaries.empty:
        return None
    t0 = time.perf_counter()
    projected = admin_boundaries.to_crs(src.crs)
    shapes = [(geometry, 1) for geometry in projected.geometry if geometry is not None and not geometry.is_empty]
    mask = np.zeros((src.height, src.width), dtype=np.uint8)
    rasterize(
        shapes,
        out=mask,
        transform=src.transform,
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=False,
    )
    mark_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    print(f"scanning population raster for {len(shapes)} matched admin polygons", flush=True)
    exposed_population, total_population, exposed_cells, settlement_summary = _sum_population_and_write_surface(
        src,
        mask,
        surface_path=surface_path,
        smod_src=smod_src,
        allowed_smod_classes=allowed_smod_classes,
    )
    scan_seconds = time.perf_counter() - t1
    del mask

    out: dict[str, Any] = {
        "interpretation": (
            "Population inside matched admin-1 polygons for records with admin1 evidence. "
            "This broad areal tier is reported separately from point-supported distance footprints."
        ),
        "exposed_population": exposed_population,
        "total_population_in_raster": total_population,
        "share_of_population": exposed_population / total_population if total_population else None,
        "exposed_cells": exposed_cells,
        "matched_admin_units": int(len(admin_boundaries)),
        "timing_seconds": {
            "rasterize": mark_seconds,
            "scan_and_write": scan_seconds,
        },
    }
    if settlement_summary is not None:
        out["settlement"] = settlement_summary
    if surface_path is not None:
        out["surface"] = {
            "path": str(surface_path),
            "sha256": _sha256(surface_path),
            "size_bytes": surface_path.stat().st_size,
        }
    return out


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
    allowed_smod_classes: list[int] | None = None,
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

            smod = None
            if smod_src is not None:
                smod = (
                    smod_src.read(1, window=window, masked=True)
                    .filled(-9999)
                    .astype(np.int16)
                )

            if allowed_smod_classes is not None and smod is not None:
                allowed_mask = np.isin(smod, allowed_smod_classes)
                local_mask[~allowed_mask] = 0

            total_population += float(pop.sum())
            exposed_cells += int(local_mask.sum())
            exposed_population += float(pop[local_mask == 1].sum())
            if smod is not None and settlement is not None:
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


def _resolve_allowed_smod_classes(
    args: argparse.Namespace,
    smod_path: Path | None,
) -> tuple[list[int] | None, str]:
    """Resolve optional settlement-class masking without changing the default metric."""
    if args.no_smod and (args.allowed_smod_classes or args.smod_population_only):
        raise ValueError("SMOD filtering requires the GHS-SMOD raster; remove --no-smod.")
    if args.allowed_smod_classes and args.smod_population_only:
        raise ValueError("Use either --allowed-smod-classes or --smod-population-only, not both.")
    if smod_path is None:
        return None, "none"
    if args.smod_population_only:
        return POPULATION_SMOD_CLASSES.copy(), "population_settlement_classes"
    if args.allowed_smod_classes:
        allowed = sorted({int(cls) for cls in args.allowed_smod_classes})
        unknown = [cls for cls in allowed if cls not in SMOD_CLASS_LABELS]
        if unknown:
            raise ValueError(f"Unknown GHS-SMOD classes: {unknown}")
        return allowed, "explicit"
    return None, "none"


def _validate_filtered_output_paths(
    args: argparse.Namespace,
    allowed_smod_classes: list[int] | None,
) -> None:
    """Prevent filtered sensitivity runs from overwriting canonical outputs."""
    if allowed_smod_classes is None:
        return

    defaults = {
        "--output-json": (args.output_json, OUTPUT_JSON),
        "--surface-path": (args.surface_path, DEFAULT_SURFACE),
    }
    if not args.no_admin:
        defaults["--admin-surface-path"] = (args.admin_surface_path, DEFAULT_ADMIN_SURFACE)

    conflicting = [
        flag
        for flag, (chosen, default) in defaults.items()
        if Path(chosen).resolve() == Path(default).resolve()
    ]
    if conflicting:
        raise ValueError(
            "SMOD-filtered runs are sensitivity analyses and cannot overwrite canonical "
            f"baseline outputs. Provide variant paths for: {', '.join(conflicting)}."
        )


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
    allowed_smod_classes: list[int] | None = None,
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
        allowed_smod_classes=allowed_smod_classes,
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
    allowed_smod_classes, smod_filter_policy = _resolve_allowed_smod_classes(args, smod_path)
    _validate_filtered_output_paths(args, allowed_smod_classes)
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
        args.output_locations.parent.mkdir(parents=True, exist_ok=True)
        location_frame.to_csv(args.output_locations, index=False)

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
                    allowed_smod_classes=allowed_smod_classes,
                )
            )

        admin_boundaries = gpd.GeoDataFrame()
        admin_diagnostics = {
            "candidate_records": 0,
            "candidate_count": 0,
            "candidate_countries": 0,
            "unique_admin_labels": 0,
            "matched_records": 0,
            "matched_count": 0,
            "matched_unique_admin_labels": 0,
            "matched_unique_admin_units": 0,
            "unmatched_records": 0,
            "unmatched_count": 0,
            "unmatched_unique_admin_labels": 0,
            "top_unmatched_admin_labels": [],
            "disabled": bool(args.no_admin),
        }
        if not args.no_admin:
            admin_boundaries, _, admin_diagnostics = _load_admin_supported_evidence(
                CASES_PATH,
                point_confidence=args.min_confidence,
                admin_min_confidence=args.admin_min_confidence,
                output_path=args.output_admin_matches,
                download=not args.no_download,
            )
        admin_result = None
        if not args.no_admin and not admin_boundaries.empty:
            admin_result = _admin_supported_exposure(
                src,
                admin_boundaries,
                surface_path=args.admin_surface_path,
                smod_src=smod_src,
                allowed_smod_classes=allowed_smod_classes,
            )
        if smod_src is not None:
            smod_src.close()

    valid_dated_records = int(case_diagnostics.get("case_records_with_valid_dates", 0))
    point_surface_records = int(case_diagnostics.get("point_surface_records", 0))
    matched_admin_records = int(admin_diagnostics.get("matched_records", 0))
    not_surface_records = max(0, valid_dated_records - point_surface_records - matched_admin_records)
    case_diagnostics.update(
        {
            "admin_candidate_records": int(admin_diagnostics.get("candidate_records", 0)),
            "admin_matched_records": matched_admin_records,
            "admin_unmatched_records": int(admin_diagnostics.get("unmatched_records", 0)),
            "records_not_used_in_any_surface": not_surface_records,
            "unlocalizable_records": not_surface_records,
        }
    )

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
            },
            index_path=args.external_index,
        )
    if admin_result and "surface" in admin_result:
        _write_external_index(
            {
                **admin_result["surface"],
                "description": (
                    "Global 1 km admin-supported areal exposure mask for matched "
                    "Natural Earth Admin 1 reported-case evidence."
                ),
                "created_at_utc": _now_utc(),
                "source_script": "pipeline/exposure/exposed_population.py",
            },
            index_path=args.external_index,
        )

    output_entries = {
        "locations_csv": {
            "path": _display_path(args.output_locations),
            "sha256": _sha256(args.output_locations),
            "size_bytes": args.output_locations.stat().st_size,
        },
        "external_index": _display_path(args.external_index),
    }
    if not args.no_admin and args.output_admin_matches.exists():
        output_entries["admin_matches_csv"] = {
            "path": _display_path(args.output_admin_matches),
            "sha256": _sha256(args.output_admin_matches),
            "size_bytes": args.output_admin_matches.stat().st_size,
        }

    output = {
        "schema_version": "global_exposure_v1_1",
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
            "admin_min_confidence": float(args.admin_min_confidence),
            "low_confidence_policy": (
                "country-only records remain unlocalizable; matched admin1 records are reported "
                "as a separate broad areal exposure tier, not as point hotspots"
            ),
            "smod_filter_policy": smod_filter_policy,
            "allowed_smod_classes": allowed_smod_classes,
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
        "admin_boundaries": {
            "path": str(NE_ADMIN1_CACHE),
            "source_url": NE_ADMIN1_URL,
            "sha256": _sha256(NE_ADMIN1_CACHE) if NE_ADMIN1_CACHE.exists() else None,
            "size_bytes": NE_ADMIN1_CACHE.stat().st_size if NE_ADMIN1_CACHE.exists() else None,
            "license": "Natural Earth public domain",
            "citation": "Natural Earth. 1:10m Cultural Vectors, Admin 1 States and Provinces.",
        },
        "outputs": output_entries,
        "exposure": radius_results,
        "admin_supported": {
            "diagnostics": admin_diagnostics,
            "exposure": admin_result,
        },
        "warnings": [
            "Point-supported exposure is a lower-bound local footprint because low-confidence records are not locally mapped.",
            "Admin-supported exposure is a broad areal evidence tier and is not additive with the point-supported footprint.",
            "Admin label matching is conservative; unmatched locality, county, and ambiguous labels stay out of the admin mask.",
            "Distance footprints are evidence footprints, not transmission probabilities.",
            "Rasterized geodesic footprints assign whole 1 km cells by the rasterization rule rather than sub-cell population fractions.",
            "Admin polygons use center-cell rasterization at 1 km and can undercount very small boundary units.",
            "SMOD settlement strata describe where exposed population lives; they are not a reservoir-habitat model.",
            "GHSL 2020 population is used for all case years, so historical exposure is not population-year specific.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build global point-supported exposure accounting outputs.")
    parser.add_argument("--radii-km", nargs="+", type=float, default=[25, 50, 100])
    parser.add_argument("--surface-radius-km", type=float, default=100)
    parser.add_argument("--min-confidence", type=float, default=0.95)
    parser.add_argument("--coordinate-precision", type=int, default=5)
    parser.add_argument("--geodesic-vertices", type=int, default=96)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-locations", type=Path, default=OUTPUT_LOCATIONS)
    parser.add_argument("--output-admin-matches", type=Path, default=OUTPUT_ADMIN)
    parser.add_argument("--external-index", type=Path, default=EXTERNAL_INDEX)
    parser.add_argument("--surface-path", type=Path, default=DEFAULT_SURFACE)
    parser.add_argument("--admin-surface-path", type=Path, default=DEFAULT_ADMIN_SURFACE)
    parser.add_argument("--admin-min-confidence", type=float, default=0.35)
    parser.add_argument("--no-admin", action="store_true")
    parser.add_argument("--no-smod", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--allowed-smod-classes",
        type=int,
        nargs="+",
        default=None,
        help="Strictly filter exposed population to these explicit GHS-SMOD classes.",
    )
    parser.add_argument(
        "--smod-population-only",
        action="store_true",
        help="Filter exposure to GHS-SMOD populated settlement classes: 11, 12, 13, 21, 22, 23, and 30.",
    )
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
    admin = result.get("admin_supported", {})
    if admin.get("exposure"):
        print(
            "admin tier: matched_records={records}, exposed_population={pop:,.0f}, cells={cells:,}".format(
                records=admin["diagnostics"]["matched_records"],
                pop=admin["exposure"]["exposed_population"],
                cells=admin["exposure"]["exposed_cells"],
            )
        )
    print(f"Saved: {args.output_json}")
    print(f"Locations: {args.output_locations}")


if __name__ == "__main__":
    main()
