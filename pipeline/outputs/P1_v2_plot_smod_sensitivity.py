"""Render the P1 v2 SMOD sensitivity figure.

This script is a lightweight renderer for
`data/outputs/P1_v2_smod_sensitivity_v1.json`. It does not rescan rasters or
change exposure totals. The figure shows how the canonical point-supported,
admin-supported, and union tiers split into urban and rural GHS-SMOD
settlement definitions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
SENSITIVITY_JSON = OUTPUT_DIR / "P1_v2_smod_sensitivity_v1.json"
OUTPUT_FIG = OUTPUT_DIR / "P1_v2_smod_sensitivity.png"
TIER_LABELS = {
    "point_supported_100km": "Point\n100 km",
    "admin_supported": "Admin\nareal",
    "union": "Union",
}


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing SMOD sensitivity JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _people_millions(value: float) -> float:
    return float(value) / 1_000_000.0


def _fmt_people(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    return f"{value / 1_000_000:.0f}M"


def _tier_values(report: dict[str, Any], variant: str, tier_names: list[str]) -> np.ndarray:
    return np.array(
        [
            float(report["variants"][variant]["tiers"][tier]["exposed_population"])
            for tier in tier_names
        ],
        dtype=float,
    )


def _tier_rates(report: dict[str, Any], variant: str, tier_names: list[str]) -> np.ndarray:
    return np.array(
        [
            float(report["variants"][variant]["tiers"][tier]["share_of_variant_population"])
            for tier in tier_names
        ],
        dtype=float,
    )


def plot_smod_sensitivity(report: dict[str, Any], output_fig: Path) -> None:
    tier_names = ["point_supported_100km", "admin_supported", "union"]
    labels = [TIER_LABELS[tier] for tier in tier_names]

    all_values = _tier_values(report, "all_population", tier_names)
    urban_values = _tier_values(report, "urban", tier_names)
    rural_values = _tier_values(report, "rural", tier_names)
    other_values = np.maximum(0.0, all_values - urban_values - rural_values)

    urban_rates = _tier_rates(report, "urban", tier_names)
    rural_rates = _tier_rates(report, "rural", tier_names)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "text.color": "#111111",
        "axes.labelcolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    x = np.arange(len(tier_names))
    urban_color = "#2563eb"
    rural_color = "#65a30d"
    other_color = "#94a3b8"
    urban_m = np.array([_people_millions(value) for value in urban_values])
    rural_m = np.array([_people_millions(value) for value in rural_values])
    other_m = np.array([_people_millions(value) for value in other_values])
    all_m = np.array([_people_millions(value) for value in all_values])

    ax.bar(x, urban_m, color=urban_color, label="Urban")
    ax.bar(x, rural_m, bottom=urban_m, color=rural_color, label="Rural")
    if other_m.max() > 0.01:
        ax.bar(x, other_m, bottom=urban_m + rural_m, color=other_color, label="Other")
    for idx, total in enumerate(all_values):
        ax.text(
            idx,
            all_m[idx] + max(all_m) * 0.02,
            _fmt_people(total),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Exposed population, millions")
    ax.set_title("Exposed population by settlement class", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)

    ax = axes[1]
    width = 0.36
    ax.bar(x - width / 2, urban_rates * 100, width=width, color=urban_color, label="Urban")
    ax.bar(x + width / 2, rural_rates * 100, width=width, color=rural_color, label="Rural")
    for idx, value in enumerate(urban_rates * 100):
        ax.text(idx - width / 2, value + 0.4, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(rural_rates * 100):
        ax.text(idx + width / 2, value + 0.4, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of settlement population")
    ax.set_title("Settlement-specific exposure rate", loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.suptitle(
        "P1 v2 SMOD sensitivity of reported-evidence exposure",
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fig, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render P1 v2 SMOD sensitivity figure.")
    parser.add_argument("--sensitivity-json", type=Path, default=SENSITIVITY_JSON)
    parser.add_argument("--output-fig", type=Path, default=OUTPUT_FIG)
    args = parser.parse_args()

    report = _load_report(args.sensitivity_json)
    plot_smod_sensitivity(report, args.output_fig)
    print(f"saved: {args.output_fig}")


if __name__ == "__main__":
    main()
