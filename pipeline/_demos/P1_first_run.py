"""Deprecated v0.1 smoke test: SES-weighted effective distance for Brazil.

This script is intentionally kept out of `pipeline/outputs/` because it uses a
hardcoded source point and arbitrary parameters. It is useful only for checking
that the old graph machinery still runs; it is not part of the data-driven
global exposure accounting pipeline.

Source point: Foz do Iguaçu, Paraná (BR) — the seeded ARAV (Araraquara virus)
cluster region. We compute the effective distance from this point to every
populated cell in Brazil under two scenarios:

    (1) Population-only gravity (baseline: Brockmann-Helbing analogue)
    (2) Population × SMOD detection-capacity (TEG layer 4 + layer 5 stub)

and produce a side-by-side figure plus the absolute and relative differences.
This is the minimum viable demonstration that SES weighting changes the
geometry — the empirical-calibration step against observed cases follows
once we wire CDC NNDSS / WHO DON / PAHO point data into the pipeline.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.layers.worldpop import (
    load_population_grid, populated_cell_mask, WORLDPOP_BR_2020_1KM,
)
from pipeline.layers.ghs_smod import (
    load_smod_to_grid, smod_to_weight, GHS_SMOD_2020_PATH,
)
from pipeline.distance.effdist_grid import (
    build_grid_graph, effective_distance_from,
    grid_index_for_lonlat, graph_to_grid,
)

# Outbreak origin: Foz do Iguaçu (Paraná, BR) — ARAV reporting region
ORIGIN_LON, ORIGIN_LAT = -54.5854, -25.5478
ORIGIN_LABEL = "Foz do Iguaçu, Paraná (ARAV reporting region)"

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("→ loading WorldPop BR 1km, coarsening to ~10km")
    grid = load_population_grid(WORLDPOP_BR_2020_1KM, coarsen_factor=10)
    pop = grid["pop"]
    print(f"   shape={pop.shape}  total_pop={pop.sum():,.0f}  "
          f"populated_cells={populated_cell_mask(pop).sum():,}")

    print("→ aligning GHS-SMOD to coarsened grid (mode resampling)")
    smod_cls = load_smod_to_grid(
        GHS_SMOD_2020_PATH,
        target_transform=grid["transform"],
        target_shape=grid["shape"],
        target_crs=grid["crs"],
    )
    smod_w = smod_to_weight(smod_cls)
    print(f"   SMOD weight summary: min={smod_w.min():.2f}  "
          f"median={np.median(smod_w):.2f}  max={smod_w.max():.2f}")

    print("→ locating outbreak source cell")
    flat_src = grid_index_for_lonlat(
        ORIGIN_LON, ORIGIN_LAT, grid["transform"], grid["shape"]
    )
    print(f"   source flat index: {flat_src}")

    results = {}
    for label, ses in (("baseline", None), ("ses_weighted", smod_w)):
        print(f"→ building graph + Dijkstra ({label})")
        csgraph, node_idx = build_grid_graph(
            pop, radius_cells=3, beta=1.5, ses_weight=ses, min_pop=1.0,
        )
        src_node = int(np.searchsorted(node_idx, flat_src))
        if node_idx[src_node] != flat_src:
            raise RuntimeError("Source cell is below min_pop threshold")
        d_eff = effective_distance_from(csgraph, src_node)
        d_eff_grid = graph_to_grid(d_eff, node_idx, grid["shape"])
        results[label] = d_eff_grid
        reachable = np.isfinite(d_eff).sum()
        print(f"   reachable nodes: {reachable:,} / {len(node_idx):,}")

    # Output: 3-panel figure
    print("→ writing figure")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    vmax = np.nanpercentile(results["baseline"], 99)

    for ax, key, title in (
        (axes[0], "baseline", "Baseline d_eff\n(population gravity only)"),
        (axes[1], "ses_weighted", "TEG d_eff\n(× SMOD detection capacity)"),
    ):
        im = ax.imshow(results[key], cmap="magma_r", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        # Mark source
        col = int(round((ORIGIN_LON - grid["bbox"][0]) /
                  (grid["bbox"][2] - grid["bbox"][0]) * grid["shape"][1]))
        row = int(round((grid["bbox"][3] - ORIGIN_LAT) /
                  (grid["bbox"][3] - grid["bbox"][1]) * grid["shape"][0]))
        ax.plot(col, row, marker="o", mfc="white", mec="crimson", ms=10, mew=1.5)
        fig.colorbar(im, ax=ax, shrink=0.7, label="−log P (cumulative)")

    diff = results["ses_weighted"] - results["baseline"]
    dmax = np.nanpercentile(np.abs(diff), 99)
    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    axes[2].set_title("SES uplift\n(TEG − baseline)", fontsize=11)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(im, ax=axes[2], shrink=0.7, label="Δ effective distance")

    fig.suptitle(
        f"TEG v0.1 · BR effective distance from {ORIGIN_LABEL}\n"
        f"WorldPop 2020 × GHS-SMOD 2020 · gravity β=1.5 · 10 km cells",
        fontsize=11, y=1.04,
    )

    out_path = OUTPUT_DIR / "P1_BR_effdist_v1.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"   wrote {out_path}")

    # Also save the raw effective-distance arrays for dashboard ingestion
    npz_path = OUTPUT_DIR / "P1_BR_effdist_v1.npz"
    np.savez_compressed(
        npz_path,
        baseline=results["baseline"],
        ses_weighted=results["ses_weighted"],
        bbox=np.array(grid["bbox"]),
        origin=np.array([ORIGIN_LON, ORIGIN_LAT]),
    )
    print(f"   wrote {npz_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
