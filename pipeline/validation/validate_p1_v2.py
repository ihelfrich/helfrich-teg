"""Validate the P1 v2 global exposure output bundle.

This is a reproducibility gate for the reported-evidence exposure pipeline. It
does not recompute exposure. It checks that the canonical JSON, external masks,
dashboard NPZ, and country audit all describe the same all-population baseline
run. Optional SMOD-filtered sensitivity runs should use separate output paths
and their own validation expectations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"

EXPOSURE_JSON = OUTPUT_DIR / "global_exposure_v1.json"
EXTERNAL_INDEX = OUTPUT_DIR / "_external_index.json"
COUNTRY_AUDIT = OUTPUT_DIR / "P1_v2_country_audit_v1.csv"
NPZ_OUTPUT = OUTPUT_DIR / "P1_v2_exposure_v1.npz"
VALIDATION_JSON = OUTPUT_DIR / "P1_v2_validation_v1.json"

EXPECTED_SCHEMA = "global_exposure_v1_1"
EXPECTED_NPZ_SCHEMA = "P1_v2_exposure_v1"


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


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _surface_for_radius(report: dict[str, Any], radius_km: float) -> dict[str, Any]:
    for item in report["exposure"]:
        if math.isclose(float(item["radius_km"]), float(radius_km)):
            return item
    raise KeyError(f"No exposure entry for radius {radius_km:g} km")


def _scalar_string(npz: np.lib.npyio.NpzFile, key: str) -> str:
    return str(npz[key][0])


def _float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def _single_value(frame: pd.DataFrame, column: str) -> Any:
    values = frame[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{column} has {len(values)} distinct non-null values")
    return values[0]


def _record(checks: list[dict[str, Any]], name: str, ok: bool, details: dict[str, Any] | None = None) -> None:
    checks.append({"name": name, "ok": bool(ok), "details": details or {}})


def _close_enough(a: float, b: float, *, rel_tol: float = 1e-9, abs_tol: float = 1e-3) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def validate_outputs(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    exposure_json = args.exposure_json
    external_index = args.external_index
    country_audit = args.country_audit
    npz_output = args.npz_output

    for label, path in {
        "exposure_json": exposure_json,
        "external_index": external_index,
        "country_audit": country_audit,
        "npz_output": npz_output,
    }.items():
        _record(checks, f"{label}_exists", path.exists(), {"path": _display_path(path)})

    report = _load_json(exposure_json)
    index = _load_json(external_index)
    audit = pd.read_csv(country_audit)
    npz = np.load(npz_output, allow_pickle=False)

    report_sha = _sha256(exposure_json)
    audit_sha = _sha256(country_audit)
    npz_sha = _sha256(npz_output)

    _record(checks, "exposure_schema", report.get("schema_version") == EXPECTED_SCHEMA, {
        "actual": report.get("schema_version"),
        "expected": EXPECTED_SCHEMA,
    })

    parameters = report.get("parameters", {})
    expected_classes = args.expected_allowed_smod_classes
    actual_classes = parameters.get("allowed_smod_classes")
    _record(checks, "smod_filter_policy", parameters.get("smod_filter_policy") == args.expected_smod_policy, {
        "actual": parameters.get("smod_filter_policy"),
        "expected": args.expected_smod_policy,
    })
    _record(checks, "allowed_smod_classes", actual_classes == expected_classes, {
        "actual": actual_classes,
        "expected": expected_classes,
    })

    output_entries = report.get("outputs", {})
    locations_entry = output_entries.get("locations_csv", {})
    admin_entry = output_entries.get("admin_matches_csv", {})
    locations_path = _path(locations_entry["path"])
    admin_matches_path = _path(admin_entry["path"])

    for label, entry, path in [
        ("locations_csv", locations_entry, locations_path),
        ("admin_matches_csv", admin_entry, admin_matches_path),
    ]:
        actual_sha = _sha256(path)
        actual_size = path.stat().st_size
        _record(checks, f"{label}_hash", actual_sha == entry.get("sha256"), {
            "path": _display_path(path),
            "actual": actual_sha,
            "expected": entry.get("sha256"),
        })
        _record(checks, f"{label}_size", actual_size == entry.get("size_bytes"), {
            "path": _display_path(path),
            "actual": actual_size,
            "expected": entry.get("size_bytes"),
        })

    surface_radius = float(parameters["surface_radius_km"])
    point_surface = _surface_for_radius(report, surface_radius)["surface"]
    admin_surface = report["admin_supported"]["exposure"]["surface"]
    surface_entries = {
        str(point_surface["path"]): point_surface,
        str(admin_surface["path"]): admin_surface,
    }
    index_entries = {str(item["path"]): item for item in index.get("artifacts", [])}
    for surface_path_string, surface_entry in surface_entries.items():
        path = Path(surface_path_string)
        index_entry = index_entries.get(surface_path_string)
        _record(checks, f"{path.name}_in_external_index", index_entry is not None, {
            "path": surface_path_string,
        })
        actual_sha = _sha256(path)
        actual_size = path.stat().st_size
        _record(checks, f"{path.name}_hash", actual_sha == surface_entry.get("sha256"), {
            "actual": actual_sha,
            "expected": surface_entry.get("sha256"),
        })
        _record(checks, f"{path.name}_size", actual_size == surface_entry.get("size_bytes"), {
            "actual": actual_size,
            "expected": surface_entry.get("size_bytes"),
        })
        if index_entry is not None:
            _record(checks, f"{path.name}_external_index_hash", index_entry.get("sha256") == surface_entry.get("sha256"), {
                "index": index_entry.get("sha256"),
                "report": surface_entry.get("sha256"),
            })
            _record(checks, f"{path.name}_external_index_size", index_entry.get("size_bytes") == surface_entry.get("size_bytes"), {
                "index": index_entry.get("size_bytes"),
                "report": surface_entry.get("size_bytes"),
            })

    point100 = _surface_for_radius(report, surface_radius)
    admin_exposure = report["admin_supported"]["exposure"]
    audit_point_pop = float(audit["point_exposed_population_100km"].sum())
    audit_admin_pop = float(audit["admin_exposed_population"].sum())
    audit_point_cells = int(audit["point_exposed_cells_100km"].sum())
    audit_admin_cells = int(audit["admin_exposed_cells"].sum())
    _record(checks, "audit_point_population_reconciles", _close_enough(audit_point_pop, point100["exposed_population"]), {
        "audit": audit_point_pop,
        "json": float(point100["exposed_population"]),
    })
    _record(checks, "audit_admin_population_reconciles", _close_enough(audit_admin_pop, admin_exposure["exposed_population"]), {
        "audit": audit_admin_pop,
        "json": float(admin_exposure["exposed_population"]),
    })
    _record(checks, "audit_point_cells_reconcile", audit_point_cells == int(point100["exposed_cells"]), {
        "audit": audit_point_cells,
        "json": int(point100["exposed_cells"]),
    })
    _record(checks, "audit_admin_cells_reconcile", audit_admin_cells == int(admin_exposure["exposed_cells"]), {
        "audit": audit_admin_cells,
        "json": int(admin_exposure["exposed_cells"]),
    })
    union_ok = bool((audit["union_exposed_population"] <= audit["point_exposed_population_100km"] + audit["admin_exposed_population"] + 1e-6).all())
    _record(checks, "audit_union_not_additive_overrun", union_ok)

    for column, expected in {
        "exposure_json_sha256": report_sha,
        "locations_csv_sha256": locations_entry.get("sha256"),
        "admin_matches_csv_sha256": admin_entry.get("sha256"),
        "point_surface_sha256": point_surface.get("sha256"),
        "admin_surface_sha256": admin_surface.get("sha256"),
    }.items():
        try:
            actual = _single_value(audit, column)
            ok = actual == expected
        except ValueError as exc:
            actual = str(exc)
            ok = False
        _record(checks, f"audit_{column}", ok, {"actual": actual, "expected": expected})

    required_npz_keys = {
        "schema_version",
        "exposure_schema_version",
        "exposure_json_sha256",
        "admin_matches_csv_sha256",
        "point_surface_sha256",
        "admin_surface_sha256",
        "point_mask_100km",
        "admin_mask",
        "radii_km",
        "radius_exposed_population",
        "radius_exposed_share",
        "evidence_counts",
        "evidence_count_labels",
    }
    missing_npz_keys = sorted(required_npz_keys.difference(npz.files))
    _record(checks, "npz_required_keys", not missing_npz_keys, {"missing": missing_npz_keys})
    _record(checks, "npz_schema", _scalar_string(npz, "schema_version") == EXPECTED_NPZ_SCHEMA, {
        "actual": _scalar_string(npz, "schema_version"),
        "expected": EXPECTED_NPZ_SCHEMA,
    })
    _record(checks, "npz_exposure_hash", _scalar_string(npz, "exposure_json_sha256") == report_sha, {
        "actual": _scalar_string(npz, "exposure_json_sha256"),
        "expected": report_sha,
    })
    _record(checks, "npz_admin_hash", _scalar_string(npz, "admin_matches_csv_sha256") == admin_entry.get("sha256"), {
        "actual": _scalar_string(npz, "admin_matches_csv_sha256"),
        "expected": admin_entry.get("sha256"),
    })
    _record(checks, "npz_point_surface_hash", _scalar_string(npz, "point_surface_sha256") == point_surface.get("sha256"), {
        "actual": _scalar_string(npz, "point_surface_sha256"),
        "expected": point_surface.get("sha256"),
    })
    _record(checks, "npz_admin_surface_hash", _scalar_string(npz, "admin_surface_sha256") == admin_surface.get("sha256"), {
        "actual": _scalar_string(npz, "admin_surface_sha256"),
        "expected": admin_surface.get("sha256"),
    })

    expected_radii = _float_list([item["radius_km"] for item in sorted(report["exposure"], key=lambda item: item["radius_km"])])
    expected_pop = _float_list([item["exposed_population"] for item in sorted(report["exposure"], key=lambda item: item["radius_km"])])
    expected_share = _float_list([item["share_of_population"] for item in sorted(report["exposure"], key=lambda item: item["radius_km"])])
    _record(checks, "npz_radii_match", np.allclose(npz["radii_km"], expected_radii), {
        "actual": npz["radii_km"].tolist(),
        "expected": expected_radii,
    })
    _record(checks, "npz_radius_population_match", np.allclose(npz["radius_exposed_population"], expected_pop), {
        "actual": npz["radius_exposed_population"].tolist(),
        "expected": expected_pop,
    })
    _record(checks, "npz_radius_share_match", np.allclose(npz["radius_exposed_share"], expected_share), {
        "actual": npz["radius_exposed_share"].tolist(),
        "expected": expected_share,
    })

    evidence_counts = report["case_evidence"]
    expected_evidence = [
        int(evidence_counts["point_surface_records"]),
        int(evidence_counts["admin_matched_records"]),
        int(evidence_counts["records_not_used_in_any_surface"]),
    ]
    _record(checks, "npz_evidence_counts_match", np.array_equal(npz["evidence_counts"], np.array(expected_evidence, dtype=np.int64)), {
        "actual": npz["evidence_counts"].tolist(),
        "expected": expected_evidence,
    })
    _record(checks, "npz_preview_shapes_match", npz["point_mask_100km"].shape == npz["admin_mask"].shape, {
        "point": list(npz["point_mask_100km"].shape),
        "admin": list(npz["admin_mask"].shape),
    })

    failed = [check for check in checks if not check["ok"]]
    report_out = {
        "schema_version": "P1_v2_validation_v1",
        "created_at_utc": _now_utc(),
        "status": "passed" if not failed else "failed",
        "checks_passed": len(checks) - len(failed),
        "checks_failed": len(failed),
        "summary": {
            "smod_filter_policy": parameters.get("smod_filter_policy"),
            "allowed_smod_classes": actual_classes,
            "point_supported_population_100km": float(point100["exposed_population"]),
            "admin_supported_population": float(admin_exposure["exposed_population"]),
            "audit_union_population": float(audit["union_exposed_population"].sum()),
            "country_audit_rows": int(len(audit)),
        },
        "artifacts": {
            "exposure_json": {"path": _display_path(exposure_json), "sha256": report_sha, "size_bytes": exposure_json.stat().st_size},
            "country_audit": {"path": _display_path(country_audit), "sha256": audit_sha, "size_bytes": country_audit.stat().st_size},
            "npz": {"path": _display_path(npz_output), "sha256": npz_sha, "size_bytes": npz_output.stat().st_size},
        },
        "checks": checks,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report_out, indent=2), encoding="utf-8")
    return report_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate P1 v2 exposure output consistency.")
    parser.add_argument("--exposure-json", type=Path, default=EXPOSURE_JSON)
    parser.add_argument("--external-index", type=Path, default=EXTERNAL_INDEX)
    parser.add_argument("--country-audit", type=Path, default=COUNTRY_AUDIT)
    parser.add_argument("--npz-output", type=Path, default=NPZ_OUTPUT)
    parser.add_argument("--output-json", type=Path, default=VALIDATION_JSON)
    parser.add_argument("--expected-smod-policy", default="none")
    parser.add_argument("--expected-allowed-smod-classes", type=int, nargs="*", default=None)
    parser.add_argument("--no-fail", action="store_true", help="Write the report but do not exit nonzero on failed checks.")
    args = parser.parse_args()

    report = validate_outputs(args)
    print(
        "P1 v2 validation: {status} ({passed} passed, {failed} failed) -> {path}".format(
            status=report["status"],
            passed=report["checks_passed"],
            failed=report["checks_failed"],
            path=args.output_json,
        )
    )
    if report["status"] != "passed" and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
