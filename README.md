# Topological Epidemic Geometry (TEG)

A research framework and code base for modelling zoonotic-spillover and pandemic propagation as a
**multi-layer heterogeneous-spatial network**, with population, mobility, ecology, and socioeconomic
detection-capacity layers carried at the resolution of the data that produces them.

**Author:** Ian T. Helfrich, PhD &nbsp;·&nbsp; [ianhelfrich.com](https://ianhelfrich.com) &nbsp;·&nbsp; ianthelfrich@gmail.com

This repository is the methodological sibling of
[The Hantavirus Observatory](https://ihelfrich.github.io/hantavirus-observatory/). The Observatory
is the live demonstration; this repo is the peer-reviewable methods code, paper sources, and
replication materials.

## What this is

The classical effective-distance framework (Brockmann & Helbing 2013) collapses global epidemic
geometry onto the air-passenger network, with airports as nodes and flow-derived probabilities as
edge weights. It works — predicts H1N1 and SARS arrival times within days — but inherits the
network's coarseness. Importation risk to a 100m grid cell in rural Patagonia is identical to a
100m cell in central Buenos Aires under that model. The detection probability, healthcare capacity,
and reservoir-ecology suitability of those two cells differ by orders of magnitude.

TEG carries the heterogeneity:

- **Layer 1** — international mobility (air, sea, rail)
- **Layer 2** — intra-country mobility (gravity model on population grids)
- **Layer 3** — reservoir habitat suitability (WorldClim × GBIF × land cover)
- **Layer 4** — population density and urbanisation (WorldPop × GHS-SMOD, 100m)
- **Layer 5** — socioeconomic detection capacity (gridded GDP, DHS wealth, healthcare-facility density)
- **Layer 6** — admin-level healthcare load (HHS Protect, ECDC TESSy, KDCA, MSal)

The core v2 output is an exposed-population accounting surface: given reported
case evidence, how many people live in the granular population cells plausibly
inside the reported exposure footprint? Effective distance remains one optional
geometry for exposure decay, but the project is not trying to infer a single
outbreak origin from reported cases.

## Paper series

| # | Title | Status |
|---|---|---|
| **P0** | Socioeconomically-weighted effective distance on gridded heterogeneous populations | drafting |
| **P1** | Predicting Andes-virus importation from socioeconomic topology: the 2026 Patagonian cluster | planned |
| **P2** | Mast-year climate forcing × detection asymmetry in European hantavirus surveillance (Puumala) | planned |
| **P3** | A unified socioeconomic-topological framework for zoonotic spillover risk across N priority pathogens | planned |
| **P4** | Trade-network exposure to outbreak-origin economies: supply-chain vulnerability as effective distance | planned |

Each paper carries its own Zenodo concept-DOI; the parent series brand is *Topological Epidemic Geometry*.

## Repository structure

```
pipeline/
  layers/      — input-layer loaders (WorldPop, GHS-SMOD, WorldClim, GBIF, OSM, DHS, HHS)
  suitability/ — reservoir habitat-suitability models (random forest, MaxEnt)
  distance/    — effective-distance computation on gridded heterogeneous networks
  outputs/     — runnable scripts that produce paper figures and dashboard tiles
papers/        — Quarto sources for each paper in the series
data/
  inputs/      — gitignored; pulled from HELFRICH-GD or upstream APIs
  outputs/     — checked in; what the dashboard consumes
notebooks/     — exploratory analyses; one per paper
site_layers/   — PMTiles, GeoJSON, and JSON metric files served to the Observatory dashboard
```

## Reproduction

```bash
pip install -e .
python3 pipeline/layers/cases.py --refresh
```

### Step 1: data-driven hantavirus case panel

Builds `data/outputs/cases_panel_v1.parquet` from NCBI GenBank/BioSample
metadata, CDC NNDSS Socrata tables, and WHO Disease Outbreak News.

```bash
python3 pipeline/layers/cases.py --refresh
```

Expected wall-time on Ian's Mac: 5-7 minutes on a cold cache, mostly NCBI
GenBank flatfile fetches; under 1 minute when `/Volumes/HELFRICH-GD/TEG_data/`
already has the caches.

Expected outputs:

- `data/outputs/cases_panel_v1.parquet` (~58 KB)
- `data/outputs/cases_panel_v1_manifest.json` (~98 KB)

The script prints a one-line panel summary and the point-coordinate fraction.

### Step 2: Brazil beta calibration smoke diagnostic

Runs a Brazil-only graph-calibration smoke diagnostic on the currently available
Brazil grid. This is retained to test the graph machinery, not as the core
epidemiological objective.

```bash
python3 pipeline/calibration/fit_beta.py
```

Expected wall-time on Ian's Mac: 75-90 seconds with the current 10 km Brazil
grid. The step uses Python/SciPy sparse graph construction and Dijkstra; Go or
Rust is not needed at this scale.

Expected output:

- `data/outputs/beta_fit_v1.json` (~16 KB)

Current result: `beta_hat=0.50`, 95% bootstrap CI `[0.50, 0.50]`, with
weighted log-likelihood improvement of `19.588` over the v0.1 baseline
`beta=1.5`. The JSON intentionally records weak-identification warnings because
the Brazil subset has only four unique mapped case cells and three point-level
events.

### Step 3: global point-supported exposed population

Builds a world-coverage exposed-population accounting output from high-confidence
reported case coordinates and GHSL 2020 global 1 km population. Low-confidence
country/admin centroid records are reported as unlocalizable evidence and are
not turned into fake local hotspots.

```bash
python3 pipeline/exposure/exposed_population.py
```

Expected wall-time on Ian's Mac: 4-8 minutes on a cold GHSL cache, including
the first GHSL download/extract; about 3 minutes on a warm cache because the
script scans the global 1 km population raster for each exposure radius. The
script keeps the per-radius geodesic mask in memory and only writes the final
compressed 100 km surface to the external drive.

Expected checked-in outputs:

- `data/outputs/global_exposure_v1.json` (~4.5 KB)
- `data/outputs/global_exposure_locations_v1.csv` (~28 KB)
- `data/outputs/_external_index.json` (~550 B)

Expected external output:

- `/Volumes/HELFRICH-GD/TEG_data/outputs/global_exposure_point_supported_100km_v1.tif` (~768 KB compressed)

Current result: 217 unique point-supported locations from 882 high-confidence
records across 26 countries. The 100 km geodesic footprint covers 358.8 million
people, or 4.58% of the GHSL 2020 population raster. The remaining 8,157 records are
reported as unlocalizable at this stage because they are country/admin centroids
or otherwise too coarse for granular exposure mapping.

## Data Provenance

Case-panel source caches are stored outside git at
`/Volumes/HELFRICH-GD/TEG_data/inputs/cases/`. The file-level provenance,
including source URLs, pull time, SHA-256 hashes, licenses/terms, and citations,
is checked in at `data_provenance.md`. The exact machine-readable manifest for
the current panel is `data/outputs/cases_panel_v1_manifest.json`.

## Citation

If you cite this code, use the Zenodo DOI for the version you used. For the parent series:

> Helfrich, I. T. (2026). *Topological Epidemic Geometry: a research code base for multi-layer
> heterogeneous-spatial pandemic modelling.* https://github.com/ihelfrich/helfrich-teg

## Licence

Code: MIT. Paper sources and figures: CC-BY-4.0.
