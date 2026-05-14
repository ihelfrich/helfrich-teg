"""Create P1 v2 global exposure figures and compact dashboard arrays.

This script is an output renderer for the data-driven exposure pipeline. It does
not estimate an outbreak origin or a transmission parameter. It reads the
reported-evidence exposure summary from `pipeline/exposure/exposed_population.py`,
renders the point-supported and admin-supported masks, and writes a compressed
NPZ with downsampled masks and summary arrays for dashboard ingestion.

Assumptions:
    * The GHSL 1 km raster grid remains the accounting denominator.
    * The point-supported 100 km mask and admin-supported mask are separate
      evidence tiers and are not additive.
    * Raster masks are downsampled with average resampling for display and
      dashboard preview arrays, then thresholded at > 0 so any exposed source
      cell in a preview block is retained. The authoritative exposed-population
      totals remain in `data/outputs/global_exposure_v1.json`.
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.layers.cases import EXTERNAL_CACHE


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
EXPOSURE_JSON = OUTPUT_DIR / "global_exposure_v1.json"
ADMIN_MATCHES_CSV = OUTPUT_DIR / "global_exposure_admin_v1.csv"
COUNTRIES_ZIP = EXTERNAL_CACHE / "boundaries" / "ne_110m_admin_0_countries.zip"

SURFACE_FIG = OUTPUT_DIR / "P1_v2_exposure_surfaces.png"
SUMMARY_FIG = OUTPUT_DIR / "P1_v2_exposure_summary.png"
NPZ_OUTPUT = OUTPUT_DIR / "P1_v2_exposure_v1.npz"


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing exposure summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def _surface_for_radius(report: dict[str, Any], radius_km: float) -> dict[str, Any]:
    for item in report["exposure"]:
        if float(item["radius_km"]) == float(radius_km):
            if "surface" not in item:
                raise KeyError(f"No surface recorded for {radius_km:g} km exposure")
            return item
    raise KeyError(f"No exposure entry for {radius_km:g} km")


def _read_mask_preview(path: Path, *, max_width: int) -> tuple[np.ndarray, tuple[float, float, float, float], Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing external mask: {path}")
    with rasterio.open(path) as src:
        factor = max(1, int(np.ceil(src.width / max_width)))
        out_height = int(np.ceil(src.height / factor))
        out_width = int(np.ceil(src.width / factor))
        mask = src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.average,
            masked=False,
        )
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        return (mask > 0).astype(np.uint8), extent, src.crs


def _load_country_outlines(crs) -> gpd.GeoDataFrame | None:
    if not COUNTRIES_ZIP.exists():
        return None
    countries = gpd.read_file(f"zip://{COUNTRIES_ZIP}")
    countries = countries.loc[countries.geometry.notna() & ~countries.geometry.is_empty].copy()
    return countries.to_crs(crs)


def _fmt_people(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    return f"{value:,.0f}"


def _tier_smod_totals(exposure: dict[str, Any]) -> dict[str, float]:
    by_group = exposure.get("settlement", {}).get("by_group", {})
    urban = float(by_group.get("urban", {}).get("exposed_population", 0.0))
    rural = float(by_group.get("rural", {}).get("exposed_population", 0.0))
    total = float(exposure.get("exposed_population", 0.0))
    other = max(0.0, total - urban - rural)
    return {"rural": rural, "urban": urban, "other": other}


def _draw_mask_panel(
    ax,
    mask: np.ndarray,
    extent: tuple[float, float, float, float],
    countries: gpd.GeoDataFrame | None,
    *,
    color: tuple[float, float, float, float],
    title: str,
    subtitle: str,
) -> None:
    ax.set_facecolor("#f8fafc")
    if countries is not None:
        countries.boundary.plot(ax=ax, color="#64748b", linewidth=0.25, alpha=0.55)
    cmap = ListedColormap([(0, 0, 0, 0), color])
    ax.imshow(mask, cmap=cmap, vmin=0, vmax=1, extent=extent, origin="upper", interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.text(
        0.0,
        0.99,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#334155",
    )
    ax.set_axis_off()
    ax.set_aspect("equal")


def plot_surfaces(
    report: dict[str, Any],
    *,
    point_mask: np.ndarray,
    admin_mask: np.ndarray,
    extent: tuple[float, float, float, float],
    crs,
    output_path: Path,
) -> None:
    countries = _load_country_outlines(crs)
    point = _surface_for_radius(report, report["parameters"]["surface_radius_km"])
    admin = report["admin_supported"]["exposure"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    _draw_mask_panel(
        axes[0],
        point_mask,
        extent,
        countries,
        color=(0.88, 0.08, 0.24, 0.82),
        title="Point-supported 100 km footprint",
        subtitle=(
            f"{_fmt_people(point['exposed_population'])} people, "
            f"{point['share_of_population']:.2%} of GHSL 2020 population"
        ),
    )
    _draw_mask_panel(
        axes[1],
        admin_mask,
        extent,
        countries,
        color=(0.10, 0.36, 0.78, 0.72),
        title="Admin-supported areal tier",
        subtitle=(
            f"{_fmt_people(admin['exposed_population'])} people, "
            f"{admin['share_of_population']:.2%} of GHSL 2020 population"
        ),
    )
    fig.suptitle(
        "P1 v2 global reported-evidence exposure surfaces",
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_summary(report: dict[str, Any], admin_matches: pd.DataFrame, *, output_path: Path) -> None:
    radius_df = pd.DataFrame(report["exposure"]).sort_values("radius_km")
    point = _surface_for_radius(report, report["parameters"]["surface_radius_km"])
    admin = report["admin_supported"]["exposure"]
    point_smod = _tier_smod_totals(point)
    admin_smod = _tier_smod_totals(admin)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    bars = ax.bar(
        radius_df["radius_km"].astype(str),
        radius_df["exposed_population"] / 1_000_000,
        color=["#fb7185", "#e11d48", "#be123c"],
    )
    ax.set_title("Point footprint sensitivity", loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("Exposed population, millions")
    ax.set_xlabel("Geodesic radius, km")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, share in zip(bars, radius_df["share_of_population"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{share:.2%}", ha="center", va="bottom", fontsize=8)

    ax = axes[0, 1]
    categories = ["rural", "urban", "other"]
    bottom = np.zeros(2)
    colors = {"rural": "#65a30d", "urban": "#2563eb", "other": "#94a3b8"}
    for category in categories:
        values = np.array([point_smod[category], admin_smod[category]]) / 1_000_000
        ax.bar(["point 100 km", "admin tier"], values, bottom=bottom, color=colors[category], label=category)
        bottom += values
    ax.set_title("Settlement composition", loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("Exposed population, millions")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 0]
    evidence_counts = pd.Series(
        {
            "point surface": report["case_evidence"]["point_surface_records"],
            "admin surface": report["case_evidence"]["admin_matched_records"],
            "not surfaced": report["case_evidence"]["records_not_used_in_any_surface"],
        }
    )
    ax.bar(evidence_counts.index, evidence_counts.values, color=["#e11d48", "#2563eb", "#64748b"])
    ax.set_title("Case-evidence accounting", loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("Records")
    ax.tick_params(axis="x", rotation=15)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    status = (
        admin_matches.groupby("match_status", dropna=False)["records"]
        .sum()
        .reindex(["matched", "unmatched", "ambiguous"])
        .fillna(0)
    )
    ax.bar(status.index, status.values, color=["#2563eb", "#64748b", "#f59e0b"])
    ax.set_title("Admin label matching audit", loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("Records")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "P1 v2 exposure accounting summary",
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_npz(
    report: dict[str, Any],
    *,
    point_mask: np.ndarray,
    admin_mask: np.ndarray,
    extent: tuple[float, float, float, float],
    exposure_json: Path,
    admin_matches_csv: Path,
    point_surface: Path,
    admin_surface: Path,
    output_path: Path,
) -> None:
    radius_df = pd.DataFrame(report["exposure"]).sort_values("radius_km")
    evidence_counts = np.array(
        [
            report["case_evidence"]["point_surface_records"],
            report["case_evidence"]["admin_matched_records"],
            report["case_evidence"]["records_not_used_in_any_surface"],
        ],
        dtype=np.int64,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.array(["P1_v2_exposure_v1"]),
        exposure_schema_version=np.array([report["schema_version"]]),
        created_at_utc=np.array([_now_utc()]),
        mask_preview_resampling=np.array(["average_threshold_positive"]),
        exposure_json_sha256=np.array([_sha256(exposure_json)]),
        admin_matches_csv_sha256=np.array([_sha256(admin_matches_csv)]),
        point_surface_sha256=np.array([_sha256(point_surface)]),
        admin_surface_sha256=np.array([_sha256(admin_surface)]),
        point_mask_100km=point_mask,
        admin_mask=admin_mask,
        extent=np.array(extent, dtype=np.float64),
        radii_km=radius_df["radius_km"].to_numpy(dtype=np.float64),
        radius_exposed_population=radius_df["exposed_population"].to_numpy(dtype=np.float64),
        radius_exposed_share=radius_df["share_of_population"].to_numpy(dtype=np.float64),
        evidence_counts=evidence_counts,
        evidence_count_labels=np.array(["point_surface", "admin_surface", "not_surfaced"]),
    )


def build_outputs(args: argparse.Namespace) -> dict[str, Path]:
    report = _load_report(args.exposure_json)
    admin_matches = pd.read_csv(args.admin_matches_csv)
    point_surface = Path(_surface_for_radius(report, report["parameters"]["surface_radius_km"])["surface"]["path"])
    admin_surface = Path(report["admin_supported"]["exposure"]["surface"]["path"])

    point_mask, extent, crs = _read_mask_preview(point_surface, max_width=args.max_width)
    admin_mask, admin_extent, admin_crs = _read_mask_preview(admin_surface, max_width=args.max_width)
    if admin_extent != extent or admin_crs != crs:
        raise ValueError("Point and admin masks are not on the same preview grid")
    if admin_mask.shape != point_mask.shape:
        raise ValueError(f"Preview mask shapes differ: {point_mask.shape} vs {admin_mask.shape}")

    plot_surfaces(report, point_mask=point_mask, admin_mask=admin_mask, extent=extent, crs=crs, output_path=args.surface_fig)
    plot_summary(report, admin_matches, output_path=args.summary_fig)
    write_npz(
        report,
        point_mask=point_mask,
        admin_mask=admin_mask,
        extent=extent,
        exposure_json=args.exposure_json,
        admin_matches_csv=args.admin_matches_csv,
        point_surface=point_surface,
        admin_surface=admin_surface,
        output_path=args.npz_output,
    )

    return {
        "surface_fig": args.surface_fig,
        "summary_fig": args.summary_fig,
        "npz": args.npz_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render P1 v2 global exposure outputs.")
    parser.add_argument("--exposure-json", type=Path, default=EXPOSURE_JSON)
    parser.add_argument("--admin-matches-csv", type=Path, default=ADMIN_MATCHES_CSV)
    parser.add_argument("--surface-fig", type=Path, default=SURFACE_FIG)
    parser.add_argument("--summary-fig", type=Path, default=SUMMARY_FIG)
    parser.add_argument("--npz-output", type=Path, default=NPZ_OUTPUT)
    parser.add_argument("--max-width", type=int, default=2200)
    args = parser.parse_args()

    outputs = build_outputs(args)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
