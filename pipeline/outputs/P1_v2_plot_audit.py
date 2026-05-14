"""Plot a high-quality global choropleth map of the country-level exposure audit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.layers.cases import EXTERNAL_CACHE

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
AUDIT_CSV = OUTPUT_DIR / "P1_v2_country_audit_v1.csv"
COUNTRIES_ZIP = EXTERNAL_CACHE / "boundaries" / "ne_110m_admin_0_countries.zip"
OUTPUT_FIG = OUTPUT_DIR / "P1_v2_country_audit_map.png"


def plot_country_audit(audit_csv: Path, boundaries_zip: Path, output_fig: Path) -> None:
    if not audit_csv.exists():
        raise FileNotFoundError(f"Missing audit CSV: {audit_csv}")
    if not boundaries_zip.exists():
        raise FileNotFoundError(f"Missing boundaries: {boundaries_zip}")

    audit = pd.read_csv(audit_csv)
    world = gpd.read_file(f"zip://{boundaries_zip}")
    world["iso_a3"] = world["ISO_A3"].where(world["ISO_A3"].astype(str) != "-99", world["ADM0_A3"])

    # Project to Robinson for global display.
    world = world.to_crs("ESRI:54030")

    merged = world.merge(audit, on="iso_a3", how="left")
    # Use percentage for readability
    merged["plot_value_pct"] = merged["union_exposed_share_of_country"].fillna(0) * 100

    # Publication-ready typography and styling
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "axes.edgecolor": "#333333",
        "text.color": "#111111",
        "axes.labelcolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
    })

    fig, ax = plt.subplots(1, 1, figsize=(14, 7), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Base map: pale gray for missing/zero data, very thin white borders
    world.plot(ax=ax, color="#e5e7eb", edgecolor="#ffffff", linewidth=0.3)

    # Exposed data
    exposed_mask = merged["plot_value_pct"] > 0
    vmax = merged["plot_value_pct"].max()
    if pd.isna(vmax) or vmax == 0:
        vmax = 100

    cmap = mcolors.LinearSegmentedColormap.from_list("pub_reds", ["#fee2e2", "#ef4444", "#9f1239", "#4c0519"])

    merged[exposed_mask].plot(
        ax=ax,
        column="plot_value_pct",
        cmap=cmap,
        edgecolor="#ffffff",
        linewidth=0.3,
        vmin=0,
        vmax=vmax
    )

    # Clean, horizontal colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.03, pad=0.02, aspect=45)
    cbar.set_label("Share of National Population Exposed (%)", fontsize=11, labelpad=8, fontweight="bold")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=0, labelsize=10)

    ax.set_title("Global Population Exposure to Hantavirus", fontsize=18, fontweight="bold", pad=25)
    plt.figtext(0.5, 0.86, "Union of point-supported (100km) and admin-supported areal tiers", ha="center", fontsize=12, color="#475569")

    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fig, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {output_fig}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", type=Path, default=AUDIT_CSV)
    parser.add_argument("--boundaries-zip", type=Path, default=COUNTRIES_ZIP)
    parser.add_argument("--output-fig", type=Path, default=OUTPUT_FIG)
    args = parser.parse_args()
    plot_country_audit(args.audit_csv, args.boundaries_zip, args.output_fig)

if __name__ == "__main__":
    main()
