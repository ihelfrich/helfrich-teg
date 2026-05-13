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

The output is an effective-distance surface from any outbreak origin that integrates over all six
layers, against which observed case appearance can be calibrated and forward-predictions made.

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

## Replication

```bash
pip install -e .
python pipeline/outputs/P1_bariloche_v1.py
```

## Citation

If you cite this code, use the Zenodo DOI for the version you used. For the parent series:

> Helfrich, I. T. (2026). *Topological Epidemic Geometry: a research code base for multi-layer
> heterogeneous-spatial pandemic modelling.* https://github.com/ihelfrich/helfrich-teg

## Licence

Code: MIT. Paper sources and figures: CC-BY-4.0.
