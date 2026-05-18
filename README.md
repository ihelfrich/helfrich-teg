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
  _demos/      — deprecated smoke tests retained outside the main reproduction path
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

### Step 3: global reported-evidence exposed population

Builds a world-coverage exposed-population accounting output from high-confidence
reported case coordinates, GHSL 2020 global 1 km population, and GHS-SMOD
settlement classes. Records with admin1 evidence that are not already consumed
by the point-coordinate tier are matched conservatively to Natural Earth Admin 1
boundaries and reported as a separate broad areal tier. Country-only records and
unmatched or ambiguous admin labels remain unlocalizable evidence.

```bash
python3 pipeline/exposure/exposed_population.py
```

By default this is an all-population accounting run: GHS-SMOD is used for
stratification, not for excluding people from the denominator. SMOD-filtered
sensitivity runs require explicit non-canonical output paths via
`--output-json`, `--surface-path`, and `--admin-surface-path`, so they do not
overwrite `global_exposure_v1`.

Expected wall-time on Ian's Mac: 15-18 minutes on a warm GHSL/GHS-SMOD/Natural
Earth cache. The script scans the global 1 km population and settlement rasters
for each exposure radius, keeps the per-radius geodesic mask in memory, and
writes the final compressed 100 km point-supported mask plus the admin-supported
mask to the external drive.

Expected checked-in outputs:

- `data/outputs/global_exposure_v1.json` (~26 KB)
- `data/outputs/global_exposure_locations_v1.csv` (~28 KB)
- `data/outputs/global_exposure_admin_v1.csv` (~46 KB)
- `data/outputs/_external_index.json` (~1 KB)

Expected external output:

- `/Volumes/HELFRICH-GD/TEG_data/outputs/global_exposure_point_supported_100km_v1.tif` (~768 KB compressed)
- `/Volumes/HELFRICH-GD/TEG_data/outputs/global_exposure_admin_supported_v1.tif` (~983 KB compressed)

Current result: 217 unique point-supported locations from 882 high-confidence
records across 26 countries. The 100 km geodesic footprint covers 358.8 million
people, or 4.58% of the GHSL 2020 population raster. Of that 100 km footprint,
312.5 million people are in GHS-SMOD urban classes and 46.3 million are in rural
classes. The admin-supported tier matched 1,779 lower-confidence records to 110
unique admin1 units, covering 1.46 billion people as a broad areal evidence tier.
This tier is not additive with the point-supported footprint.

### Step 4: P1 v2 exposure figures and dashboard arrays

Renders the checked-in P1 v2 figures and compact NPZ from the global exposure
summary and the off-repo point/admin masks. This is a visualization and export
step only; it does not re-estimate exposure.

```bash
python3 pipeline/outputs/P1_v2_exposure.py
```

Expected wall-time on Ian's Mac: under 30 seconds on a warm cache. The script
reads the two external GeoTIFF masks at preview resolution, overlays Natural
Earth country outlines, and writes small checked-in artifacts.

Expected outputs:

- `data/outputs/P1_v2_exposure_surfaces.png` (~283 KB)
- `data/outputs/P1_v2_exposure_summary.png` (~139 KB)
- `data/outputs/P1_v2_exposure_v1.npz` (~18 KB)

The older hardcoded Brazil first run is retained at
`pipeline/_demos/P1_first_run.py` as a smoke test, but it is no longer part of
the main P1 reproduction path.

### Step 5: P1 v2 country audit

Builds a country-level audit table by overlaying the point-supported and
admin-supported masks on Natural Earth Admin 0 countries and summing GHSL 2020
population by country. The table also joins case-panel evidence counts, surfaced
records, admin-label match counts, records not used in either surface, and the
non-additive union of point/admin exposure.

```bash
python3 pipeline/outputs/P1_v2_country_audit.py
python3 pipeline/outputs/P1_v2_plot_audit.py
```

Expected wall-time on Ian's Mac: about 1 minute for the audit table and under
10 seconds for the map on a warm cache. The audit script rasterizes country IDs
on the GHSL grid and scans the population, point mask, and admin mask once in
windows.

Expected output:

- `data/outputs/P1_v2_country_audit_v1.csv` (~114 KB)
- `data/outputs/P1_v2_country_audit_map.png` (~455 KB)

Current result: 179 audit rows. Country-assigned point-supported exposure sums
to 358.8 million people and country-assigned admin-supported exposure sums to
1.46 billion people, matching `global_exposure_v1.json`. The table includes
`union_exposed_population` so users do not sum overlapping evidence tiers. It
also separates `__unassigned_geometry__` raster cells from
`__missing_country__` case records so missing case metadata is not confused with
boundary-overlay gaps.

### Step 6: P1 v2 validation gate

Checks that the canonical exposure JSON, external GeoTIFF masks, dashboard NPZ,
and country audit describe the same all-population baseline run. This is the
guardrail that catches stale hashes, accidental SMOD-filtered overwrites, and
audit totals that no longer reconcile.

```bash
python3 pipeline/validation/validate_p1_v2.py
```

Expected wall-time on Ian's Mac: under 10 seconds on a warm cache. The script
hashes small checked-in artifacts, hashes the two external masks, verifies the
expected `smod_filter_policy`, and reconciles JSON totals against audit sums.

Expected output:

- `data/outputs/P1_v2_validation_v1.json` (~10 KB)

Current result: passed 42 checks with `smod_filter_policy=none`,
358.8 million people in the 100 km point-supported footprint, and 1.46 billion
people in the admin-supported areal tier.

### Step 7: P1 v2 SMOD sensitivity totals

Builds a derived sensitivity table from the validated point-supported and
admin-supported masks. This step scans GHSL population, GHS-SMOD, and the
canonical masks once, then reports exposed-population totals under all
population, population-settlement, urban, and rural SMOD definitions. It does
not rerasterize exposure footprints and does not overwrite the canonical
`global_exposure_v1` outputs.

```bash
python3 pipeline/outputs/P1_v2_smod_sensitivity.py --no-download
```

Expected wall-time on Ian's Mac: about 75 seconds on a warm cache. The script
uses windowed raster reads and fails if the all-population sensitivity totals do
not reconcile with `global_exposure_v1.json`.

Expected output:

- `data/outputs/P1_v2_smod_sensitivity_v1.json` (~9 KB)
- `data/outputs/P1_v2_smod_sensitivity_v1.csv` (~6 KB)

Current result: the canonical all-population 100 km point footprint is 358.8
million people, with 312.5 million in GHS-SMOD urban classes and 46.3 million
in rural classes. The non-additive union of point-supported and admin-supported
evidence covers 1.69 billion people.

### Step 8: P0 method-paper skeleton

Creates the Quarto source scaffold for the methods paper. The file contains
section stubs and links to pipeline figures in `data/outputs/`; it does not yet
contain full paper prose.

```bash
quarto render papers/P0_method/paper.qmd --to html
```

Expected wall-time on Ian's Mac: under 10 seconds with Quarto installed.

Expected source:

- `papers/P0_method/paper.qmd` (~2 KB)

Expected render output if the optional command is run:

- `papers/P0_method/paper.html` (not checked in)

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
