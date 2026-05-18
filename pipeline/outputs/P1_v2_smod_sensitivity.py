"""Build SMOD-filtered sensitivity totals from the canonical P1 v2 masks.

This is a derived-output step, not a new exposure model. It scans the validated
point-supported and admin-supported exposure masks against the GHSL population
and GHS-SMOD rasters, then reports how exposed-population totals change when
the accounting denominator is restricted to settlement classes such as urban or
rural cells.

Assumptions:
    * `global_exposure_v1.json` remains the canonical all-population baseline.
    * The point-supported mask is the 100 km geodesic footprint recorded in that
      JSON, and the admin-supported mask is the separate broad areal tier.
    * SMOD filters are sensitivity definitions only. They do not overwrite,
      replace, or revalidate the canonical masks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.layers.ghs_pop import ensure_ghs_pop_2020_1km
from pipeline.layers.ghs_smod import (
    SMOD_CLASS_LABELS,
    SMOD_GROUPS,
    ensure_ghs_smod_2020_1km,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
EXPOSURE_JSON = OUTPUT_DIR / "global_exposure_v1.json"
OUTPUT_JSON = OUTPUT_DIR / "P1_v2_smod_sensitivity_v1.json"
OUTPUT_CSV = OUTPUT_DIR / "P1_v2_smod_sensitivity_v1.csv"

SCHEMA_VERSION = "P1_v2_smod_sensitivity_v1"
POPULATION_SETTLEMENT_CLASSES = sorted(SMOD_GROUPS["rural"] + SMOD_GROUPS["urban"])
DEFAULT_VARIANTS = {
    "all_population": None,
    "population_settlement": POPULATION_SETTLEMENT_CLASSES,
    "urban": SMOD_GROUPS["urban"],
    "rural": SMOD_GROUPS["rural"],
}
TIERS = {
    "point_supported_100km": "point-supported 100 km geodesic footprint",
    "admin_supported": "admin-supported areal footprint",
    "union": "non-additive union of point-supported and admin-supported footprints",
}


@dataclass
class TierAccumulator:
    exposed_population: float = 0.0
    exposed_cells: int = 0


@dataclass
class VariantAccumulator:
    allowed_smod_classes: list[int] | None
    total_population_in_variant: float = 0.0
    total_cells_in_variant: int = 0
    point_supported_100km: TierAccumulator = field(default_factory=TierAccumulator)
    admin_supported: TierAccumulator = field(default_factory=TierAccumulator)
    union: TierAccumulator = field(default_factory=TierAccumulator)


def _now_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _surface_for_radius(report: dict[str, Any], radius_km: float) -> dict[str, Any]:
    for item in report["exposure"]:
        if math.isclose(float(item["radius_km"]), float(radius_km)):
            if "surface" not in item:
                raise KeyError(f"No surface recorded for {radius_km:g} km exposure")
            return item
    raise KeyError(f"No exposure entry for radius {radius_km:g} km")


def _resolve_reported_input_path(
    report: dict[str, Any],
    key: str,
    fallback: Path,
) -> Path:
    value = report.get(key, {}).get("path")
    if value:
        path = _path(value)
        if path.exists():
            return path
    return fallback


def _assert_same_grid(reference, other, *, reference_label: str, other_label: str) -> None:
    if reference.crs != other.crs:
        raise ValueError(f"{other_label} CRS {other.crs} does not match {reference_label} CRS {reference.crs}")
    if reference.transform != other.transform:
        raise ValueError(f"{other_label} transform does not match {reference_label}")
    if reference.shape != other.shape:
        raise ValueError(f"{other_label} shape {other.shape} does not match {reference_label} shape {reference.shape}")


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


def _new_accumulators() -> dict[str, VariantAccumulator]:
    return {
        name: VariantAccumulator(
            allowed_smod_classes=None if classes is None else sorted(classes),
            point_supported_100km=TierAccumulator(),
            admin_supported=TierAccumulator(),
            union=TierAccumulator(),
        )
        for name, classes in DEFAULT_VARIANTS.items()
    }


def _allowed_mask(smod: np.ndarray, allowed_smod_classes: list[int] | None) -> np.ndarray:
    if allowed_smod_classes is None:
        return np.ones(smod.shape, dtype=bool)
    return np.isin(smod, allowed_smod_classes)


def _accumulate_variant(
    acc: VariantAccumulator,
    *,
    pop: np.ndarray,
    point_mask: np.ndarray,
    admin_mask: np.ndarray,
    allowed: np.ndarray,
) -> None:
    point = point_mask & allowed
    admin = admin_mask & allowed
    union = point | admin
    acc.total_population_in_variant += float(pop[allowed].sum())
    acc.total_cells_in_variant += int(allowed.sum())
    acc.point_supported_100km.exposed_population += float(pop[point].sum())
    acc.point_supported_100km.exposed_cells += int(point.sum())
    acc.admin_supported.exposed_population += float(pop[admin].sum())
    acc.admin_supported.exposed_cells += int(admin.sum())
    acc.union.exposed_population += float(pop[union].sum())
    acc.union.exposed_cells += int(union.sum())


def _scan_sensitivity(
    pop_path: Path,
    smod_path: Path,
    point_surface: Path,
    admin_surface: Path,
    *,
    block_size: int,
) -> dict[str, VariantAccumulator]:
    accumulators = _new_accumulators()
    with (
        rasterio.open(pop_path) as pop_src,
        rasterio.open(smod_path) as smod_src,
        rasterio.open(point_surface) as point_src,
        rasterio.open(admin_surface) as admin_src,
    ):
        for label, src in [
            ("GHS-SMOD", smod_src),
            ("point-supported mask", point_src),
            ("admin-supported mask", admin_src),
        ]:
            _assert_same_grid(pop_src, src, reference_label="GHSL population", other_label=label)

        for window in _iter_windows(pop_src.height, pop_src.width, block_size=block_size):
            pop = pop_src.read(1, window=window, masked=True).filled(0).astype(np.float64)
            pop[pop < 0] = 0
            smod = smod_src.read(1, window=window, masked=True).filled(-9999).astype(np.int16)
            point_mask = point_src.read(1, window=window, masked=True).filled(0).astype(np.uint8) > 0
            admin_mask = admin_src.read(1, window=window, masked=True).filled(0).astype(np.uint8) > 0

            for acc in accumulators.values():
                allowed = _allowed_mask(smod, acc.allowed_smod_classes)
                _accumulate_variant(
                    acc,
                    pop=pop,
                    point_mask=point_mask,
                    admin_mask=admin_mask,
                    allowed=allowed,
                )
    return accumulators


def _class_labels(classes: list[int] | None) -> dict[str, str] | None:
    if classes is None:
        return None
    return {str(cls): SMOD_CLASS_LABELS[int(cls)] for cls in classes}


def _tier_dict(
    tier: TierAccumulator,
    *,
    global_population: float,
    variant_population: float,
    all_population_tier: TierAccumulator | None,
) -> dict[str, Any]:
    delta = None
    delta_pct = None
    if all_population_tier is not None:
        baseline = float(all_population_tier.exposed_population)
        delta = float(tier.exposed_population - baseline)
        delta_pct = delta / baseline if baseline else None
    return {
        "exposed_population": float(tier.exposed_population),
        "exposed_cells": int(tier.exposed_cells),
        "share_of_global_population": (
            float(tier.exposed_population) / global_population if global_population else None
        ),
        "share_of_variant_population": (
            float(tier.exposed_population) / variant_population if variant_population else None
        ),
        "delta_from_all_population": delta,
        "delta_pct_from_all_population": delta_pct,
    }


def _build_report(
    *,
    exposure_report: dict[str, Any],
    accumulators: dict[str, VariantAccumulator],
    exposure_json: Path,
    pop_path: Path,
    smod_path: Path,
    point_surface: Path,
    admin_surface: Path,
) -> dict[str, Any]:
    all_population = accumulators["all_population"]
    global_population = float(all_population.total_population_in_variant)
    variants = {}
    for name, acc in accumulators.items():
        all_acc = None if name == "all_population" else all_population
        variants[name] = {
            "allowed_smod_classes": acc.allowed_smod_classes,
            "class_labels": _class_labels(acc.allowed_smod_classes),
            "total_population_in_variant": float(acc.total_population_in_variant),
            "total_cells_in_variant": int(acc.total_cells_in_variant),
            "share_of_global_population_in_variant": (
                float(acc.total_population_in_variant) / global_population if global_population else None
            ),
            "tiers": {
                "point_supported_100km": _tier_dict(
                    acc.point_supported_100km,
                    global_population=global_population,
                    variant_population=float(acc.total_population_in_variant),
                    all_population_tier=None if all_acc is None else all_acc.point_supported_100km,
                ),
                "admin_supported": _tier_dict(
                    acc.admin_supported,
                    global_population=global_population,
                    variant_population=float(acc.total_population_in_variant),
                    all_population_tier=None if all_acc is None else all_acc.admin_supported,
                ),
                "union": _tier_dict(
                    acc.union,
                    global_population=global_population,
                    variant_population=float(acc.total_population_in_variant),
                    all_population_tier=None if all_acc is None else all_acc.union,
                ),
            },
        }

    surface_radius = float(exposure_report["parameters"]["surface_radius_km"])
    canonical_point = _surface_for_radius(exposure_report, surface_radius)
    canonical_admin = exposure_report["admin_supported"]["exposure"]
    checks = {
        "canonical_smod_policy_is_none": exposure_report["parameters"].get("smod_filter_policy") == "none",
        "canonical_allowed_smod_classes_is_null": exposure_report["parameters"].get("allowed_smod_classes") is None,
        "point_population_matches_canonical": math.isclose(
            variants["all_population"]["tiers"]["point_supported_100km"]["exposed_population"],
            float(canonical_point["exposed_population"]),
            rel_tol=1e-9,
            abs_tol=1.0,
        ),
        "point_cells_match_canonical": (
            variants["all_population"]["tiers"]["point_supported_100km"]["exposed_cells"]
            == int(canonical_point["exposed_cells"])
        ),
        "admin_population_matches_canonical": math.isclose(
            variants["all_population"]["tiers"]["admin_supported"]["exposed_population"],
            float(canonical_admin["exposed_population"]),
            rel_tol=1e-9,
            abs_tol=1.0,
        ),
        "admin_cells_match_canonical": (
            variants["all_population"]["tiers"]["admin_supported"]["exposed_cells"]
            == int(canonical_admin["exposed_cells"])
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"SMOD sensitivity baseline does not reconcile with canonical output: {failed}")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now_utc(),
        "interpretation": (
            "SMOD sensitivity report derived from the canonical P1 v2 exposure masks. "
            "Variants restrict which settlement classes are counted; they do not alter "
            "the reported-evidence footprints."
        ),
        "source_exposure": {
            "path": _display_path(exposure_json),
            "schema_version": exposure_report.get("schema_version"),
            "sha256": _sha256(exposure_json),
            "surface_radius_km": surface_radius,
            "smod_filter_policy": exposure_report["parameters"].get("smod_filter_policy"),
            "allowed_smod_classes": exposure_report["parameters"].get("allowed_smod_classes"),
        },
        "inputs": {
            "population_raster": {
                "path": str(pop_path),
                "sha256": _sha256(pop_path),
                "size_bytes": pop_path.stat().st_size,
            },
            "settlement_raster": {
                "path": str(smod_path),
                "sha256": _sha256(smod_path),
                "size_bytes": smod_path.stat().st_size,
                "class_labels": {str(k): v for k, v in sorted(SMOD_CLASS_LABELS.items())},
                "groups": SMOD_GROUPS,
            },
            "point_surface": {
                "path": str(point_surface),
                "sha256": _sha256(point_surface),
                "size_bytes": point_surface.stat().st_size,
            },
            "admin_surface": {
                "path": str(admin_surface),
                "sha256": _sha256(admin_surface),
                "size_bytes": admin_surface.stat().st_size,
            },
        },
        "tiers": TIERS,
        "variants": variants,
        "checks": checks,
        "warnings": [
            "Urban and rural variants are sensitivity definitions, not evidence that exposure is absent outside those classes.",
            "The admin-supported tier is broad areal evidence and should not be added to the point-supported tier; use the union tier when a single footprint total is needed.",
        ],
    }


def _rows_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    source = report["source_exposure"]
    inputs = report["inputs"]
    for variant_name, variant in report["variants"].items():
        for tier_name, tier in variant["tiers"].items():
            rows.append(
                {
                    "variant": variant_name,
                    "tier": tier_name,
                    "allowed_smod_classes": (
                        "" if variant["allowed_smod_classes"] is None else ",".join(map(str, variant["allowed_smod_classes"]))
                    ),
                    "total_population_in_variant": variant["total_population_in_variant"],
                    "share_of_global_population_in_variant": variant["share_of_global_population_in_variant"],
                    "exposed_population": tier["exposed_population"],
                    "exposed_cells": tier["exposed_cells"],
                    "share_of_global_population": tier["share_of_global_population"],
                    "share_of_variant_population": tier["share_of_variant_population"],
                    "delta_from_all_population": tier["delta_from_all_population"],
                    "delta_pct_from_all_population": tier["delta_pct_from_all_population"],
                    "source_exposure_json_sha256": source["sha256"],
                    "population_raster_sha256": inputs["population_raster"]["sha256"],
                    "settlement_raster_sha256": inputs["settlement_raster"]["sha256"],
                    "point_surface_sha256": inputs["point_surface"]["sha256"],
                    "admin_surface_sha256": inputs["admin_surface"]["sha256"],
                    "created_at_utc": report["created_at_utc"],
                }
            )
    return rows


def build_smod_sensitivity(args: argparse.Namespace) -> dict[str, Any]:
    exposure_report = _load_json(args.exposure_json)
    fallback_pop = ensure_ghs_pop_2020_1km(download=not args.no_download)
    fallback_smod = ensure_ghs_smod_2020_1km(download=not args.no_download)
    pop_path = _resolve_reported_input_path(exposure_report, "population_raster", fallback_pop)
    smod_path = _resolve_reported_input_path(exposure_report, "settlement_raster", fallback_smod)
    surface_radius = float(exposure_report["parameters"]["surface_radius_km"])
    point_surface = _path(_surface_for_radius(exposure_report, surface_radius)["surface"]["path"])
    admin_surface = _path(exposure_report["admin_supported"]["exposure"]["surface"]["path"])

    print("scanning GHSL population, GHS-SMOD, and canonical exposure masks", flush=True)
    accumulators = _scan_sensitivity(
        pop_path,
        smod_path,
        point_surface,
        admin_surface,
        block_size=args.block_size,
    )
    report = _build_report(
        exposure_report=exposure_report,
        accumulators=accumulators,
        exposure_json=args.exposure_json,
        pop_path=pop_path,
        smod_path=smod_path,
        point_surface=point_surface,
        admin_surface=admin_surface,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(_rows_from_report(report)).to_csv(args.output_csv, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build P1 v2 SMOD-filtered exposure sensitivity totals.")
    parser.add_argument("--exposure-json", type=Path, default=EXPOSURE_JSON)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    report = build_smod_sensitivity(args)
    all_point = report["variants"]["all_population"]["tiers"]["point_supported_100km"]["exposed_population"]
    urban_point = report["variants"]["urban"]["tiers"]["point_supported_100km"]["exposed_population"]
    rural_point = report["variants"]["rural"]["tiers"]["point_supported_100km"]["exposed_population"]
    union_all = report["variants"]["all_population"]["tiers"]["union"]["exposed_population"]
    print(f"variants: {len(report['variants'])}")
    print(f"point-supported all-population: {all_point:,.0f}")
    print(f"point-supported urban/rural: {urban_point:,.0f} / {rural_point:,.0f}")
    print(f"non-additive union all-population: {union_all:,.0f}")
    print(f"saved: {args.output_json}")
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
