package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestDesktopDataFromFixture(t *testing.T) {
	report := &sensitivityReport{
		SchemaVersion: "P1_v2_smod_sensitivity_v1",
		CreatedAtUTC:  "2026-05-18T13:59:17Z",
		SourceExposure: sourceExposure{
			Path:             "data/outputs/global_exposure_v1.json",
			SchemaVersion:    "global_exposure_v1_1",
			SHA256:           "abc",
			SurfaceRadiusKM:  100,
			SMODFilterPolicy: "none",
		},
		Inputs: map[string]artifact{},
		Checks: map[string]bool{"baseline": true},
		Variants: map[string]sensitivityVariant{
			"all_population": variantFixture(1000, 4000, 4500, 10_000),
			"urban":          variantFixture(700, 3000, 3400, 7_000),
			"rural":          variantFixture(300, 1000, 1100, 3_000),
		},
	}
	data, err := makeDesktopData(report)
	if err != nil {
		t.Fatal(err)
	}
	if len(data.Tiers) != 3 {
		t.Fatalf("expected 3 tiers, got %d", len(data.Tiers))
	}
	if data.Tiers[0].UrbanShareOfTier != 0.7 {
		t.Fatalf("unexpected urban share: %v", data.Tiers[0].UrbanShareOfTier)
	}
}

func TestRunSummaryJSON(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "data", "outputs"), 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "data", "outputs", "P1_v2_smod_sensitivity_v1.json")
	report := sensitivityReport{
		SchemaVersion:  "P1_v2_smod_sensitivity_v1",
		CreatedAtUTC:   "2026-05-18T13:59:17Z",
		SourceExposure: sourceExposure{Path: "data/outputs/global_exposure_v1.json"},
		Inputs:         map[string]artifact{},
		Checks:         map[string]bool{"baseline": true},
		Variants: map[string]sensitivityVariant{
			"all_population": variantFixture(1000, 4000, 4500, 10_000),
			"urban":          variantFixture(700, 3000, 3400, 7_000),
			"rural":          variantFixture(300, 1000, 1100, 3_000),
		},
	}
	payload, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, payload, 0o644); err != nil {
		t.Fatal(err)
	}

	var out bytes.Buffer
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	err = run([]string{"summary", "--repo", dir, "--format", "json"})
	w.Close()
	os.Stdout = oldStdout
	if err != nil {
		t.Fatal(err)
	}
	if _, err := out.ReadFrom(r); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out.Bytes(), []byte(`"schema_version": "tegscan_desktop_v1"`)) {
		t.Fatalf("unexpected output: %s", out.String())
	}
}

func variantFixture(point float64, admin float64, union float64, total float64) sensitivityVariant {
	return sensitivityVariant{
		TotalPopulationInVariant: total,
		Tiers: map[string]sensitivityTier{
			"point_supported_100km": {ExposedPopulation: point, ShareOfVariantPopulation: point / total},
			"admin_supported":       {ExposedPopulation: admin, ShareOfVariantPopulation: admin / total},
			"union":                 {ExposedPopulation: union, ShareOfVariantPopulation: union / total},
		},
	}
}
