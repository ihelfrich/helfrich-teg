"""Pilot estimate of the population-gravity distance-decay exponent beta.

The current v2 repository has a Brazil WorldPop grid wired into the graph
builder, so this script estimates a Brazil-only beta from the unified case
panel. Argentina and Chile can be added by passing additional country grids
once their tiles are available.

For each candidate beta, the script builds a population-only gravity graph,
precomputes Dijkstra effective-distance vectors from each unique observed case
cell, then scores each held-out event against all strictly earlier events as
sources. The held-out likelihood is a log-softmax over populated cells:

    log P(observed cell | pre-t sources, beta)

This is now explicitly a pilot/smoke diagnostic. The main project objective is
global exposed-population accounting from reported case evidence, not
pinpointing origins from Brazil. Most Brazil records in cases_panel_v1 are
country-centroid GenBank records with low confidence, so the output JSON
includes diagnostics and an identification warning when the spatial support is
weak.
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
from scipy.special import logsumexp
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.distance.effdist_grid import build_grid_graph, grid_index_for_lonlat
from pipeline.layers.worldpop import WORLDPOP_BR_2020_1KM, load_population_grid


REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = REPO_ROOT / "data" / "outputs" / "cases_panel_v1.parquet"
OUTPUT_PATH = REPO_ROOT / "data" / "outputs" / "beta_fit_v1.json"

COUNTRY_GRIDS = {
    "Brazil": {
        "country_code": "BR",
        "worldpop_path": WORLDPOP_BR_2020_1KM,
        "default_coarsen_factor": 10,
    }
}


@dataclass
class MappedEvent:
    event_id: int
    date: str
    source: str
    country: str
    admin1: str | None
    lon: float
    lat: float
    count: int
    confidence: float
    event_weight: float
    flat_index: int
    node: int
    snapped_km: float
    point_level: bool


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


def _candidate_betas(start: float, stop: float, step: float) -> np.ndarray:
    n = int(round((stop - start) / step)) + 1
    betas = np.round(start + np.arange(n) * step, 10)
    return betas[(betas >= start - 1e-9) & (betas <= stop + 1e-9)]


def _node_lookup(node_idx: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rev = -np.ones(shape[0] * shape[1], dtype=np.int64)
    rev[node_idx] = np.arange(len(node_idx))
    return rev


def _flat_to_row_col(flat: int, width: int) -> tuple[int, int]:
    return int(flat // width), int(flat % width)


def _cell_center_lonlat(flat: int, transform, width: int) -> tuple[float, float]:
    row, col = _flat_to_row_col(flat, width)
    lon, lat = transform * (col + 0.5, row + 0.5)
    return float(lon), float(lat)


def _approx_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    mean_lat = math.radians((lat1 + lat2) / 2)
    dx = (lon2 - lon1) * 111.32 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 110.57
    return float(math.hypot(dx, dy))


def _map_flat_to_node(
    flat_index: int,
    lon: float,
    lat: float,
    *,
    node_rev: np.ndarray,
    node_idx: np.ndarray,
    node_tree: cKDTree,
    transform,
    width: int,
) -> tuple[int, int, float]:
    node = int(node_rev[flat_index]) if 0 <= flat_index < len(node_rev) else -1
    if node >= 0:
        return flat_index, node, 0.0
    row, col = _flat_to_row_col(flat_index, width)
    _, nearest_node = node_tree.query([row, col], k=1)
    nearest_node = int(nearest_node)
    nearest_flat = int(node_idx[nearest_node])
    near_lon, near_lat = _cell_center_lonlat(nearest_flat, transform, width)
    return nearest_flat, nearest_node, _approx_km(lon, lat, near_lon, near_lat)


def _prepare_events(
    cases_path: Path,
    *,
    country: str,
    min_confidence: float,
    point_confidence: float,
    transform,
    shape: tuple[int, int],
    node_idx: np.ndarray,
) -> tuple[list[MappedEvent], dict[str, Any]]:
    cases = gpd.read_parquet(cases_path)
    cases["date"] = pd.to_datetime(cases["date"], errors="coerce")
    cases = cases.loc[
        (cases["country"] == country)
        & cases["date"].notna()
        & cases["lon"].between(-180, 180)
        & cases["lat"].between(-90, 90)
        & (cases["confidence"] >= min_confidence)
    ].copy()
    cases = cases.sort_values(["date", "source", "lon", "lat"]).reset_index(drop=True)

    node_rev = _node_lookup(node_idx, shape)
    width = shape[1]
    node_rows = node_idx // width
    node_cols = node_idx % width
    node_tree = cKDTree(np.column_stack([node_rows, node_cols]))

    raw_mapped: list[MappedEvent] = []
    skipped_outside_grid = 0
    for event_id, row in cases.iterrows():
        try:
            raw_flat = grid_index_for_lonlat(float(row.lon), float(row.lat), transform, shape)
        except ValueError:
            skipped_outside_grid += 1
            continue
        flat, node, snapped_km = _map_flat_to_node(
            raw_flat,
            float(row.lon),
            float(row.lat),
            node_rev=node_rev,
            node_idx=node_idx,
            node_tree=node_tree,
            transform=transform,
            width=width,
        )
        count = int(row.get("count", 1))
        confidence = float(row.get("confidence", 1.0))
        raw_mapped.append(
            MappedEvent(
                event_id=int(event_id),
                date=pd.Timestamp(row.date).date().isoformat(),
                source=str(row.source),
                country=str(row.country),
                admin1=None if pd.isna(row.admin1) else str(row.admin1),
                lon=float(row.lon),
                lat=float(row.lat),
                count=count,
                confidence=confidence,
                event_weight=float(count * confidence),
                flat_index=int(flat),
                node=int(node),
                snapped_km=float(snapped_km),
                point_level=bool(confidence >= point_confidence),
            )
        )

    if not raw_mapped:
        raise RuntimeError(f"No {country} case-panel events survived the calibration filters")

    mapped = _aggregate_events(raw_mapped)
    nodes = [event.node for event in mapped]
    dominant_share = max(pd.Series(nodes).value_counts(normalize=True).to_list())
    diagnostics = {
        "candidate_events": int(len(cases)),
        "raw_mapped_records": int(len(raw_mapped)),
        "mapped_events": int(len(mapped)),
        "aggregation": "summed by date, source, and populated grid cell",
        "skipped_outside_grid": int(skipped_outside_grid),
        "unique_case_cells": int(len(set(nodes))),
        "dominant_cell_share": float(dominant_share),
        "point_level_events": int(sum(event.point_level for event in mapped)),
        "low_confidence_events": int(sum(not event.point_level for event in mapped)),
        "max_snapped_km": float(max(event.snapped_km for event in mapped)),
    }
    return mapped, diagnostics


def _aggregate_events(events: list[MappedEvent]) -> list[MappedEvent]:
    frame = pd.DataFrame([asdict(event) for event in events])
    group_cols = ["date", "source", "country", "flat_index", "node", "point_level"]
    agg = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            admin1=("admin1", lambda values: next((value for value in values if pd.notna(value)), None)),
            lon=("lon", "mean"),
            lat=("lat", "mean"),
            count=("count", "sum"),
            event_weight=("event_weight", "sum"),
            snapped_km=("snapped_km", "max"),
            component_records=("event_id", "count"),
        )
        .reset_index()
        .sort_values(["date", "source", "node"])
        .reset_index(drop=True)
    )
    agg["confidence"] = (agg["event_weight"] / agg["count"]).clip(0, 1)
    out: list[MappedEvent] = []
    for event_id, row in agg.iterrows():
        out.append(
            MappedEvent(
                event_id=int(event_id),
                date=str(row.date),
                source=str(row.source),
                country=str(row.country),
                admin1=None if pd.isna(row.admin1) else str(row.admin1),
                lon=float(row.lon),
                lat=float(row.lat),
                count=int(row["count"]),
                confidence=float(row.confidence),
                event_weight=float(row.event_weight),
                flat_index=int(row.flat_index),
                node=int(row.node),
                snapped_km=float(row.snapped_km),
                point_level=bool(row.point_level),
            )
        )
    return out


def _precompute_distances(
    pop: np.ndarray,
    betas: np.ndarray,
    source_nodes: list[int],
    *,
    radius_cells: int,
    min_pop: float,
) -> tuple[dict[float, dict[int, np.ndarray]], np.ndarray, dict[str, Any]]:
    distances: dict[float, dict[int, np.ndarray]] = {}
    node_idx_ref: np.ndarray | None = None
    timings: list[dict[str, float]] = []

    for beta in betas:
        t0 = time.perf_counter()
        graph, node_idx = build_grid_graph(
            pop,
            beta=float(beta),
            radius_cells=radius_cells,
            min_pop=min_pop,
            ses_weight=None,
        )
        graph_seconds = time.perf_counter() - t0
        if node_idx_ref is None:
            node_idx_ref = node_idx
        elif not np.array_equal(node_idx_ref, node_idx):
            raise RuntimeError("Populated-node mask changed across beta candidates")

        beta_distances: dict[int, np.ndarray] = {}
        dijkstra_seconds = 0.0
        for node in source_nodes:
            t1 = time.perf_counter()
            from pipeline.distance.effdist_grid import effective_distance_from

            beta_distances[int(node)] = effective_distance_from(graph, int(node)).astype(np.float64)
            dijkstra_seconds += time.perf_counter() - t1
        distances[float(beta)] = beta_distances
        timings.append(
            {
                "beta": float(beta),
                "graph_seconds": float(graph_seconds),
                "dijkstra_seconds": float(dijkstra_seconds),
                "edges": int(graph.nnz),
            }
        )
    assert node_idx_ref is not None
    return distances, node_idx_ref, {"per_beta": timings}


def _risk_logprob_min_source(
    source_distances: list[np.ndarray],
    target_node: int,
) -> float:
    stacked = np.vstack(source_distances)
    d_eff = np.min(stacked, axis=0)
    finite = np.isfinite(d_eff)
    if not finite[target_node] or not finite.any():
        return float("-inf")
    log_risk = -d_eff[finite]
    target_log_risk = -float(d_eff[target_node])
    return float(target_log_risk - logsumexp(log_risk))


def score_beta_profile(
    events: list[MappedEvent],
    distances: dict[float, dict[int, np.ndarray]],
    betas: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events_sorted = sorted(events, key=lambda event: (event.date, event.event_id))
    event_rows: list[dict[str, Any]] = []
    contributions: list[list[float]] = []

    for event in events_sorted:
        prior_nodes = sorted({other.node for other in events_sorted if other.date < event.date})
        if not prior_nodes:
            continue
        row_scores: list[float] = []
        reachable_all = True
        for beta in betas:
            source_distances = [distances[float(beta)][node] for node in prior_nodes]
            logprob = _risk_logprob_min_source(source_distances, event.node)
            if not np.isfinite(logprob):
                reachable_all = False
            row_scores.append(float(event.event_weight * logprob))
        if not reachable_all:
            continue
        contributions.append(row_scores)
        event_rows.append(
            {
                "event_id": event.event_id,
                "date": event.date,
                "node": event.node,
                "flat_index": event.flat_index,
                "source": event.source,
                "count": event.count,
                "confidence": event.confidence,
                "event_weight": event.event_weight,
                "point_level": event.point_level,
                "prior_source_cells": len(prior_nodes),
            }
        )

    if not contributions:
        raise RuntimeError("No held-out events had strictly earlier source events")

    matrix = np.asarray(contributions, dtype=np.float64)
    profile = pd.DataFrame(
        {
            "beta": betas.astype(float),
            "weighted_log_likelihood": matrix.sum(axis=0),
            "mean_weighted_log_likelihood": matrix.mean(axis=0),
        }
    )
    event_scores = pd.DataFrame(event_rows)
    for col_idx, beta in enumerate(betas):
        event_scores[f"ll_beta_{beta:.2f}"] = matrix[:, col_idx]
    return profile, event_scores


def bootstrap_ci(
    event_scores: pd.DataFrame,
    betas: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    stratified: bool,
) -> dict[str, Any]:
    score_cols = [f"ll_beta_{beta:.2f}" for beta in betas]
    matrix = event_scores[score_cols].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    beta_hats: list[float] = []

    if stratified:
        strata = []
        for _, idx in event_scores.groupby(["point_level", "node"]).groups.items():
            idx_arr = np.asarray(list(idx), dtype=np.int64)
            if len(idx_arr):
                strata.append(idx_arr)
    else:
        strata = [np.arange(len(event_scores), dtype=np.int64)]

    for _ in range(n_bootstrap):
        draw_parts = []
        for idx in strata:
            draw_parts.append(rng.choice(idx, size=len(idx), replace=True))
        draw = np.concatenate(draw_parts)
        summed = matrix[draw].sum(axis=0)
        beta_hats.append(float(betas[int(np.argmax(summed))]))

    lower, upper = np.quantile(beta_hats, [0.025, 0.975])
    return {
        "method": "stratified event bootstrap by point_level and target cell"
        if stratified
        else "event bootstrap",
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "level": 0.95,
        "lower": float(lower),
        "upper": float(upper),
        "beta_hats": beta_hats,
    }


def _profile_summary(
    events: list[MappedEvent],
    distances: dict[float, dict[int, np.ndarray]],
    betas: np.ndarray,
    *,
    baseline_beta: float,
    bootstrap: int,
    seed: int,
    stratified: bool,
) -> dict[str, Any]:
    try:
        profile, event_scores = score_beta_profile(events, distances, betas)
    except RuntimeError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "events": int(len(events)),
        }
    best_idx = int(profile["weighted_log_likelihood"].to_numpy().argmax())
    beta_hat = float(profile.loc[best_idx, "beta"])
    ll_hat = float(profile.loc[best_idx, "weighted_log_likelihood"])
    baseline_idx = int(np.abs(profile["beta"].to_numpy() - baseline_beta).argmin())
    baseline_ll = float(profile.loc[baseline_idx, "weighted_log_likelihood"])
    ci = None
    if len(event_scores) >= 2 and bootstrap > 0:
        ci = bootstrap_ci(
            event_scores,
            betas,
            n_bootstrap=bootstrap,
            seed=seed,
            stratified=stratified,
        )
        ci.pop("beta_hats", None)
    return {
        "available": True,
        "events": int(len(events)),
        "scored_events": int(len(event_scores)),
        "unique_target_cells": int(event_scores["node"].nunique()),
        "beta_hat": beta_hat,
        "confidence_interval": ci,
        "weighted_log_likelihood": ll_hat,
        "baseline_beta": float(profile.loc[baseline_idx, "beta"]),
        "weighted_log_likelihood_at_baseline": baseline_ll,
        "improvement_vs_baseline": float(ll_hat - baseline_ll),
        "profile": profile.to_dict(orient="records"),
    }


def fit_beta(args: argparse.Namespace) -> dict[str, Any]:
    country_cfg = COUNTRY_GRIDS.get(args.country)
    if country_cfg is None:
        raise ValueError(f"No configured population grid for country={args.country!r}")

    grid = load_population_grid(
        country_cfg["worldpop_path"],
        coarsen_factor=args.coarsen_factor or country_cfg["default_coarsen_factor"],
    )
    graph_probe, node_idx = build_grid_graph(
        grid["pop"],
        beta=args.baseline_beta,
        radius_cells=args.radius_cells,
        min_pop=args.min_pop,
        ses_weight=None,
    )
    del graph_probe
    events, event_diagnostics = _prepare_events(
        CASES_PATH,
        country=args.country,
        min_confidence=args.min_confidence,
        point_confidence=args.point_confidence,
        transform=grid["transform"],
        shape=grid["shape"],
        node_idx=node_idx,
    )
    betas = _candidate_betas(args.beta_min, args.beta_max, args.beta_step)
    unique_source_nodes = sorted({event.node for event in events})
    distances, checked_node_idx, timing = _precompute_distances(
        grid["pop"],
        betas,
        unique_source_nodes,
        radius_cells=args.radius_cells,
        min_pop=args.min_pop,
    )
    if not np.array_equal(node_idx, checked_node_idx):
        raise RuntimeError("Calibration graph node mask changed unexpectedly")

    profile, event_scores = score_beta_profile(events, distances, betas)
    best_idx = int(profile["weighted_log_likelihood"].to_numpy().argmax())
    beta_hat = float(profile.loc[best_idx, "beta"])
    ll_hat = float(profile.loc[best_idx, "weighted_log_likelihood"])

    baseline_idx = int(np.abs(profile["beta"].to_numpy() - args.baseline_beta).argmin())
    baseline_beta = float(profile.loc[baseline_idx, "beta"])
    baseline_ll = float(profile.loc[baseline_idx, "weighted_log_likelihood"])
    ci = bootstrap_ci(
        event_scores,
        betas,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        stratified=not args.no_stratified_bootstrap,
    )
    ci_for_output = dict(ci)
    ci_for_output.pop("beta_hats", None)

    point_events = [event for event in events if event.point_level]
    point_sensitivity = _profile_summary(
        point_events,
        distances,
        betas,
        baseline_beta=args.baseline_beta,
        bootstrap=args.bootstrap,
        seed=args.seed + 1,
        stratified=not args.no_stratified_bootstrap,
    )

    scored_nodes = event_scores["node"].value_counts(normalize=True)
    identification_warnings = []
    if event_diagnostics["unique_case_cells"] < 10:
        identification_warnings.append(
            "Fewer than 10 unique Brazil case cells are available; beta is weakly identified."
        )
    if event_diagnostics["point_level_events"] < 20:
        identification_warnings.append(
            "Fewer than 20 point-level Brazil events are available; most spatial signal comes from lower-confidence locations."
        )
    if event_diagnostics["dominant_cell_share"] > 0.5:
        identification_warnings.append(
            "More than half of mapped events occupy one grid cell, mainly country-centroid records."
        )
    if math.isclose(beta_hat, float(betas.min())) or math.isclose(beta_hat, float(betas.max())):
        identification_warnings.append(
            "The likelihood profile is maximized at the edge of the searched beta interval."
        )
    if math.isclose(ci_for_output["lower"], ci_for_output["upper"]):
        identification_warnings.append(
            "The bootstrap interval is degenerate under the current spatial support."
        )

    result = {
        "schema_version": "beta_fit_v1",
        "status": "pilot_smoke_diagnostic",
        "not_core_objective": (
            "Brazil-only beta fitting is retained as a graph-calibration smoke test. "
            "The core TEG v2 objective is global exposed-population accounting from reported cases."
        ),
        "created_at_utc": _now_utc(),
        "beta_hat": beta_hat,
        "confidence_interval": ci_for_output,
        "log_likelihood": {
            "weighted_at_beta_hat": ll_hat,
            "baseline_beta": baseline_beta,
            "weighted_at_baseline_beta": baseline_ll,
            "improvement_vs_baseline": float(ll_hat - baseline_ll),
        },
        "profile": profile.to_dict(orient="records"),
        "study_area": {
            "country": args.country,
            "country_code": country_cfg["country_code"],
            "grid_shape": list(grid["shape"]),
            "grid_bbox": [float(x) for x in grid["bbox"]],
            "grid_crs": str(grid["crs"]),
            "populated_nodes": int(len(node_idx)),
            "graph_radius_cells": int(args.radius_cells),
            "min_pop": float(args.min_pop),
            "coarsen_factor": int(args.coarsen_factor or country_cfg["default_coarsen_factor"]),
        },
        "inputs": {
            "cases_panel": {
                "path": str(CASES_PATH.relative_to(REPO_ROOT)),
                "sha256": _sha256(CASES_PATH),
                "size_bytes": CASES_PATH.stat().st_size,
            },
            "worldpop": {
                "path": str(country_cfg["worldpop_path"]),
                "sha256": _sha256(country_cfg["worldpop_path"]),
                "size_bytes": country_cfg["worldpop_path"].stat().st_size,
            },
        },
        "calibration": {
            "candidate_betas": [float(beta) for beta in betas],
            "scoring": "log-softmax over populated cells using minimum effective distance from strictly pre-t source cells",
            "event_weight": "count * confidence",
            "min_confidence": float(args.min_confidence),
            "point_confidence": float(args.point_confidence),
            "scored_events": int(len(event_scores)),
            "scored_weight": float(event_scores["event_weight"].sum()),
            "scored_unique_target_cells": int(event_scores["node"].nunique()),
            "scored_dominant_cell_share": float(scored_nodes.iloc[0]),
        },
        "sensitivity": {
            "point_level_only": point_sensitivity,
        },
        "diagnostics": {
            **event_diagnostics,
            "unique_source_cells": int(len(unique_source_nodes)),
            "timing": timing,
            "identification_warnings": identification_warnings,
            "recommended_for_substantive_inference": bool(not identification_warnings),
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "geopandas": gpd.__version__,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the TEG gravity beta parameter.")
    parser.add_argument("--country", default="Brazil", choices=sorted(COUNTRY_GRIDS))
    parser.add_argument("--coarsen-factor", type=int, default=10)
    parser.add_argument("--radius-cells", type=int, default=3)
    parser.add_argument("--min-pop", type=float, default=1.0)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--point-confidence", type=float, default=0.95)
    parser.add_argument("--beta-min", type=float, default=0.5)
    parser.add_argument("--beta-max", type=float, default=3.0)
    parser.add_argument("--beta-step", type=float, default=0.1)
    parser.add_argument("--baseline-beta", type=float, default=1.5)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--no-stratified-bootstrap", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    result = fit_beta(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    ci = result["confidence_interval"]
    ll = result["log_likelihood"]
    print(
        "beta_hat={:.2f}, 95% CI=[{:.2f}, {:.2f}], "
        "ll_improvement_vs_beta_{:.1f}={:.3f}".format(
            result["beta_hat"],
            ci["lower"],
            ci["upper"],
            ll["baseline_beta"],
            ll["improvement_vs_baseline"],
        )
    )
    for warning in result["diagnostics"]["identification_warnings"]:
        print(f"warning: {warning}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
