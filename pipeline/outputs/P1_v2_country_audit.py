"""Build a country-level audit table for P1 v2 exposure accounting.

This script overlays the global point-supported and admin-supported exposure
masks onto Natural Earth Admin 0 countries and scans the GHSL population raster
block by block. The output is an inspection table, not a new model: it helps
check which countries carry the reported-evidence exposure totals and where
case records were surfaced or left unused.

Assumptions:
    * GHSL 2020 population remains the population denominator.
    * Natural Earth Admin 0 boundaries are used only for audit aggregation.
    * The point-supported and admin-supported masks remain separate evidence
      tiers and are not additive.
    * The `union_exposed_population` field is the non-additive spatial union of
      those two masks by country. Use that field when a single country total is
      needed.
    * The `__unassigned_geometry__` row is the portion of the GHSL grid that did
      not fall inside a Natural Earth 110m country polygon.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.layers.cases import EXTERNAL_CACHE
from pipeline.layers.ghs_pop import ensure_ghs_pop_2020_1km


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
EXPOSURE_JSON = OUTPUT_DIR / "global_exposure_v1.json"
CASES_PATH = OUTPUT_DIR / "cases_panel_v1.parquet"
LOCATIONS_CSV = OUTPUT_DIR / "global_exposure_locations_v1.csv"
ADMIN_MATCHES_CSV = OUTPUT_DIR / "global_exposure_admin_v1.csv"
COUNTRIES_ZIP = EXTERNAL_CACHE / "boundaries" / "ne_110m_admin_0_countries.zip"
OUTPUT_CSV = OUTPUT_DIR / "P1_v2_country_audit_v1.csv"
COUNTRY_NAME_ALIASES = {
    "Serbia": "Republic of Serbia",
    "Tanzania": "United Republic of Tanzania",
}


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


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing exposure summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_country_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "__missing_country__"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "__missing_country__"
    return COUNTRY_NAME_ALIASES.get(text, text)


def _surface_for_radius(report: dict[str, Any], radius_km: float) -> dict[str, Any]:
    for item in report["exposure"]:
        if float(item["radius_km"]) == float(radius_km):
            if "surface" not in item:
                raise KeyError(f"No surface recorded for {radius_km:g} km exposure")
            return item
    raise KeyError(f"No exposure entry for {radius_km:g} km")


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


def _load_countries(crs) -> gpd.GeoDataFrame:
    if not COUNTRIES_ZIP.exists():
        raise FileNotFoundError(COUNTRIES_ZIP)
    countries = gpd.read_file(f"zip://{COUNTRIES_ZIP}")
    countries = countries.loc[countries.geometry.notna() & ~countries.geometry.is_empty].copy()
    countries = countries[["ADMIN", "NAME", "ISO_A3", "ADM0_A3", "geometry"]].copy()
    countries["country"] = countries["ADMIN"].astype(str)
    countries["iso_a3"] = countries["ISO_A3"].where(countries["ISO_A3"].astype(str) != "-99", countries["ADM0_A3"])
    countries = countries.sort_values(["country", "iso_a3"]).reset_index(drop=True)
    countries["country_id"] = np.arange(1, len(countries) + 1, dtype=np.int32)
    return countries.to_crs(crs)


def _country_grid(countries: gpd.GeoDataFrame, src) -> np.ndarray:
    shapes = [
        (geometry, int(country_id))
        for geometry, country_id in zip(countries.geometry, countries["country_id"])
        if geometry is not None and not geometry.is_empty
    ]
    return rasterize(
        shapes,
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="uint16",
        all_touched=False,
    )


def _scan_population_by_country(
    pop_path: Path,
    point_mask_path: Path,
    admin_mask_path: Path,
    countries: gpd.GeoDataFrame,
    *,
    block_size: int,
) -> pd.DataFrame:
    with rasterio.open(pop_path) as pop_src, rasterio.open(point_mask_path) as point_src, rasterio.open(admin_mask_path) as admin_src:
        for label, src in [("point mask", point_src), ("admin mask", admin_src)]:
            if src.crs != pop_src.crs or src.transform != pop_src.transform or src.shape != pop_src.shape:
                raise ValueError(f"{label} is not aligned to GHSL population")

        country_ids = _country_grid(countries, pop_src)
        n = int(countries["country_id"].max()) + 1
        total_pop = np.zeros(n, dtype=np.float64)
        point_pop = np.zeros(n, dtype=np.float64)
        admin_pop = np.zeros(n, dtype=np.float64)
        union_pop = np.zeros(n, dtype=np.float64)
        point_cells = np.zeros(n, dtype=np.int64)
        admin_cells = np.zeros(n, dtype=np.int64)
        union_cells = np.zeros(n, dtype=np.int64)

        for window in _iter_windows(pop_src.height, pop_src.width, block_size=block_size):
            row0 = int(window.row_off)
            col0 = int(window.col_off)
            row1 = row0 + int(window.height)
            col1 = col0 + int(window.width)
            ids = country_ids[row0:row1, col0:col1].ravel()
            pop = pop_src.read(1, window=window, masked=True).filled(0).astype(np.float64).ravel()
            pop[pop < 0] = 0
            point = point_src.read(1, window=window, masked=True).filled(0).ravel() > 0
            admin = admin_src.read(1, window=window, masked=True).filled(0).ravel() > 0
            union = point | admin
            total_pop += np.bincount(ids, weights=pop, minlength=n)
            point_pop += np.bincount(ids[point], weights=pop[point], minlength=n)
            admin_pop += np.bincount(ids[admin], weights=pop[admin], minlength=n)
            union_pop += np.bincount(ids[union], weights=pop[union], minlength=n)
            point_cells += np.bincount(ids[point], minlength=n).astype(np.int64)
            admin_cells += np.bincount(ids[admin], minlength=n).astype(np.int64)
            union_cells += np.bincount(ids[union], minlength=n).astype(np.int64)

    countries_out = countries.drop(columns="geometry").copy()
    countries_out["country_id"] = countries_out["country_id"].astype(int)
    indexed = countries_out.set_index("country_id")
    out = indexed.reindex(range(1, n)).reset_index()
    out["country"] = out["country"].fillna("__unassigned__")
    out["iso_a3"] = out["iso_a3"].fillna("UNK")
    out["population_2020"] = total_pop[1:]
    out["point_exposed_population_100km"] = point_pop[1:]
    out["admin_exposed_population"] = admin_pop[1:]
    out["union_exposed_population"] = union_pop[1:]
    out["point_exposed_cells_100km"] = point_cells[1:]
    out["admin_exposed_cells"] = admin_cells[1:]
    out["union_exposed_cells"] = union_cells[1:]

    unassigned = pd.DataFrame(
        [
            {
                "country_id": 0,
                "ADMIN": "__unassigned_geometry__",
                "NAME": "__unassigned_geometry__",
                "ISO_A3": "UNK",
                "ADM0_A3": "UNK",
                "country": "__unassigned_geometry__",
                "iso_a3": "UNK",
                "population_2020": total_pop[0],
                "point_exposed_population_100km": point_pop[0],
                "admin_exposed_population": admin_pop[0],
                "union_exposed_population": union_pop[0],
                "point_exposed_cells_100km": point_cells[0],
                "admin_exposed_cells": admin_cells[0],
                "union_exposed_cells": union_cells[0],
            }
        ]
    )
    return pd.concat([out, unassigned], ignore_index=True)


def _case_evidence_by_country() -> pd.DataFrame:
    cases = gpd.read_parquet(CASES_PATH)
    locations = pd.read_csv(LOCATIONS_CSV)
    admin = pd.read_csv(ADMIN_MATCHES_CSV)

    cases["case_count"] = cases["count"].fillna(1).astype(int)
    cases["audit_country"] = cases["country"].map(_audit_country_label)
    locations["audit_country"] = locations["country"].map(_audit_country_label)
    admin["audit_country"] = admin["country"].map(_audit_country_label)
    cases_by_country = (
        cases.groupby("audit_country", dropna=False)
        .agg(
            case_records_total=("source", "size"),
            case_count_total=("case_count", "sum"),
            mean_confidence=("confidence", "mean"),
            sources=("source", lambda x: ",".join(sorted(set(map(str, x))))),
        )
        .reset_index()
        .rename(columns={"audit_country": "country"})
    )

    point_by_country = (
        locations.groupby("audit_country", dropna=False)
        .agg(
            point_surface_records=("records", "sum"),
            point_surface_count=("count", "sum"),
            point_unique_locations=("location_id", "size"),
        )
        .reset_index()
        .rename(columns={"audit_country": "country"})
    )

    admin_base = admin.groupby("audit_country", dropna=False).agg(
        admin_candidate_records=("records", "sum"),
        admin_candidate_count=("count", "sum"),
        admin_unique_labels=("admin1", "size"),
        admin_matched_labels=("matched", "sum"),
    )
    admin_matched = admin.loc[admin["matched"] == True].groupby("audit_country", dropna=False).agg(
        admin_matched_records=("records", "sum"),
        admin_matched_count=("count", "sum"),
    )
    admin_unmatched = admin.loc[admin["match_status"] == "unmatched"].groupby("audit_country", dropna=False).agg(
        admin_unmatched_records=("records", "sum"),
    )
    admin_ambiguous = admin.loc[admin["match_status"] == "ambiguous"].groupby("audit_country", dropna=False).agg(
        admin_ambiguous_records=("records", "sum"),
    )
    admin_by_country = (
        admin_base.join(admin_matched, how="left")
        .join(admin_unmatched, how="left")
        .join(admin_ambiguous, how="left")
        .reset_index()
        .rename(columns={"audit_country": "country"})
    )

    out = cases_by_country.merge(point_by_country, on="country", how="outer")
    out = out.merge(admin_by_country, on="country", how="outer")
    return out


def build_country_audit(args: argparse.Namespace) -> pd.DataFrame:
    report = _load_report(args.exposure_json)
    pop_path = ensure_ghs_pop_2020_1km(download=not args.no_download)
    point_surface = Path(_surface_for_radius(report, report["parameters"]["surface_radius_km"])["surface"]["path"])
    admin_surface = Path(report["admin_supported"]["exposure"]["surface"]["path"])

    with rasterio.open(pop_path) as pop_src:
        countries = _load_countries(pop_src.crs)

    print("rasterizing Natural Earth Admin 0 countries to the GHSL grid", flush=True)
    exposure = _scan_population_by_country(
        pop_path,
        point_surface,
        admin_surface,
        countries,
        block_size=args.block_size,
    )
    evidence = _case_evidence_by_country()
    audit = exposure.merge(evidence, on="country", how="outer")
    audit["country"] = audit["country"].fillna("__unassigned_geometry__")
    audit["iso_a3"] = audit["iso_a3"].fillna("UNK")

    numeric_cols = [
        "population_2020",
        "point_exposed_population_100km",
        "admin_exposed_population",
        "union_exposed_population",
        "point_exposed_cells_100km",
        "admin_exposed_cells",
        "union_exposed_cells",
        "case_records_total",
        "case_count_total",
        "point_surface_records",
        "point_surface_count",
        "point_unique_locations",
        "admin_candidate_records",
        "admin_candidate_count",
        "admin_matched_records",
        "admin_matched_count",
        "admin_unmatched_records",
        "admin_ambiguous_records",
        "admin_unique_labels",
        "admin_matched_labels",
    ]
    for col in numeric_cols:
        if col in audit:
            audit[col] = audit[col].fillna(0)

    audit["point_exposed_share_of_country"] = np.where(
        audit["population_2020"] > 0,
        audit["point_exposed_population_100km"] / audit["population_2020"],
        np.nan,
    )
    audit["admin_exposed_share_of_country"] = np.where(
        audit["population_2020"] > 0,
        audit["admin_exposed_population"] / audit["population_2020"],
        np.nan,
    )
    audit["union_exposed_share_of_country"] = np.where(
        audit["population_2020"] > 0,
        audit["union_exposed_population"] / audit["population_2020"],
        np.nan,
    )
    audit["records_not_surfaced"] = (
        audit["case_records_total"] - audit["point_surface_records"] - audit["admin_matched_records"]
    ).clip(lower=0)
    unresolved = audit.loc[
        (audit["case_records_total"] > 0)
        & (audit["iso_a3"] == "UNK")
        & ~audit["country"].isin(["__missing_country__", "__unassigned_geometry__"]),
        "country",
    ].dropna().astype(str).tolist()
    if unresolved:
        raise ValueError(
            "Case-evidence countries did not match Natural Earth Admin 0 names: "
            + ", ".join(sorted(unresolved))
        )
    audit["created_at_utc"] = _now_utc()
    audit["exposure_json_sha256"] = _sha256(args.exposure_json)
    audit["case_panel_sha256"] = _sha256(CASES_PATH)
    audit["locations_csv_sha256"] = _sha256(LOCATIONS_CSV)
    audit["admin_matches_csv_sha256"] = _sha256(ADMIN_MATCHES_CSV)
    audit["point_surface_sha256"] = _sha256(point_surface)
    audit["admin_surface_sha256"] = _sha256(admin_surface)
    audit["country_boundaries_sha256"] = _sha256(COUNTRIES_ZIP)

    ordered = [
        "country",
        "iso_a3",
        "population_2020",
        "point_exposed_population_100km",
        "point_exposed_share_of_country",
        "point_exposed_cells_100km",
        "admin_exposed_population",
        "admin_exposed_share_of_country",
        "admin_exposed_cells",
        "union_exposed_population",
        "union_exposed_share_of_country",
        "union_exposed_cells",
        "case_records_total",
        "case_count_total",
        "point_surface_records",
        "point_surface_count",
        "point_unique_locations",
        "admin_candidate_records",
        "admin_candidate_count",
        "admin_matched_records",
        "admin_matched_count",
        "admin_unmatched_records",
        "admin_ambiguous_records",
        "admin_unique_labels",
        "admin_matched_labels",
        "records_not_surfaced",
        "mean_confidence",
        "sources",
        "created_at_utc",
        "exposure_json_sha256",
        "case_panel_sha256",
        "locations_csv_sha256",
        "admin_matches_csv_sha256",
        "point_surface_sha256",
        "admin_surface_sha256",
        "country_boundaries_sha256",
    ]
    for col in ordered:
        if col not in audit:
            audit[col] = np.nan
    audit = audit[ordered].sort_values(
        ["point_exposed_population_100km", "admin_exposed_population", "case_records_total"],
        ascending=False,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_csv, index=False)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build P1 v2 country-level exposure audit table.")
    parser.add_argument("--exposure-json", type=Path, default=EXPOSURE_JSON)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    audit = build_country_audit(args)
    point_total = audit["point_exposed_population_100km"].sum()
    admin_total = audit["admin_exposed_population"].sum()
    print(f"countries: {len(audit):,}")
    print(f"point-supported exposed population assigned by country: {point_total:,.0f}")
    print(f"admin-supported exposed population assigned by country: {admin_total:,.0f}")
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
