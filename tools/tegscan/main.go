package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const version = "0.1.0"

type sensitivityReport struct {
	SchemaVersion  string                        `json:"schema_version"`
	CreatedAtUTC   string                        `json:"created_at_utc"`
	Interpretation string                        `json:"interpretation"`
	SourceExposure sourceExposure                `json:"source_exposure"`
	Inputs         map[string]artifact           `json:"inputs"`
	Tiers          map[string]string             `json:"tiers"`
	Variants       map[string]sensitivityVariant `json:"variants"`
	Checks         map[string]bool               `json:"checks"`
	Warnings       []string                      `json:"warnings"`
}

type sourceExposure struct {
	Path               string  `json:"path"`
	SchemaVersion      string  `json:"schema_version"`
	SHA256             string  `json:"sha256"`
	SurfaceRadiusKM    float64 `json:"surface_radius_km"`
	SMODFilterPolicy   string  `json:"smod_filter_policy"`
	AllowedSMODClasses any     `json:"allowed_smod_classes"`
}

type artifact struct {
	Path      string `json:"path"`
	SHA256    string `json:"sha256"`
	SizeBytes int64  `json:"size_bytes"`
}

type sensitivityVariant struct {
	AllowedSMODClasses             []int                      `json:"allowed_smod_classes"`
	ClassLabels                    map[string]string          `json:"class_labels"`
	TotalPopulationInVariant       float64                    `json:"total_population_in_variant"`
	TotalCellsInVariant            int64                      `json:"total_cells_in_variant"`
	ShareOfGlobalPopulationVariant float64                    `json:"share_of_global_population_in_variant"`
	Tiers                          map[string]sensitivityTier `json:"tiers"`
}

type sensitivityTier struct {
	ExposedPopulation         float64  `json:"exposed_population"`
	ExposedCells              int64    `json:"exposed_cells"`
	ShareOfGlobalPopulation   float64  `json:"share_of_global_population"`
	ShareOfVariantPopulation  float64  `json:"share_of_variant_population"`
	DeltaFromAllPopulation    *float64 `json:"delta_from_all_population"`
	DeltaPctFromAllPopulation *float64 `json:"delta_pct_from_all_population"`
}

type appMetric struct {
	ID          string  `json:"id"`
	Label       string  `json:"label"`
	Value       float64 `json:"value"`
	Display     string  `json:"display"`
	Description string  `json:"description"`
}

type appTier struct {
	ID                     string  `json:"id"`
	Label                  string  `json:"label"`
	AllPopulation          float64 `json:"all_population"`
	UrbanPopulation        float64 `json:"urban_population"`
	RuralPopulation        float64 `json:"rural_population"`
	UrbanShareOfTier       float64 `json:"urban_share_of_tier"`
	RuralShareOfTier       float64 `json:"rural_share_of_tier"`
	UrbanExposureRate      float64 `json:"urban_exposure_rate"`
	RuralExposureRate      float64 `json:"rural_exposure_rate"`
	AllPopulationDisplay   string  `json:"all_population_display"`
	UrbanPopulationDisplay string  `json:"urban_population_display"`
	RuralPopulationDisplay string  `json:"rural_population_display"`
}

type desktopData struct {
	SchemaVersion string              `json:"schema_version"`
	GeneratedAt   string              `json:"generated_at"`
	Project       string              `json:"project"`
	Source        sourceExposure      `json:"source"`
	Metrics       []appMetric         `json:"metrics"`
	Tiers         []appTier           `json:"tiers"`
	Checks        map[string]bool     `json:"checks"`
	Warnings      []string            `json:"warnings"`
	Inputs        map[string]artifact `json:"inputs"`
}

type doctorCheck struct {
	Name    string `json:"name"`
	OK      bool   `json:"ok"`
	Details string `json:"details,omitempty"`
}

type doctorReport struct {
	SchemaVersion string        `json:"schema_version"`
	Status        string        `json:"status"`
	Checks        []doctorCheck `json:"checks"`
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "tegscan:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		printUsage(os.Stdout)
		return nil
	}
	switch args[0] {
	case "version":
		fmt.Println(version)
		return nil
	case "summary":
		return runSummary(args[1:])
	case "doctor":
		return runDoctor(args[1:])
	case "desktop-data":
		return runDesktopData(args[1:])
	case "help", "-h", "--help":
		printUsage(os.Stdout)
		return nil
	default:
		printUsage(os.Stderr)
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func printUsage(w io.Writer) {
	fmt.Fprintln(w, `tegscan is the TEG exposure-accounting command line tool.

Usage:
  tegscan summary [--repo PATH] [--smod-json PATH] [--format text|json]
  tegscan doctor [--repo PATH] [--format text|json]
  tegscan desktop-data [--repo PATH]
  tegscan version

Current scope:
  Summarize and validate the checked-in P1 v2 exposure sensitivity outputs.
  Raster scanning remains in the Python pipeline until the native raster engine
  lands behind this CLI.`)
}

func runSummary(args []string) error {
	fs := flag.NewFlagSet("summary", flag.ContinueOnError)
	repo := fs.String("repo", ".", "repository root")
	smodJSON := fs.String("smod-json", "", "SMOD sensitivity JSON")
	format := fs.String("format", "text", "output format: text or json")
	if err := fs.Parse(args); err != nil {
		return err
	}
	report, err := loadSensitivity(resolveSmodPath(*repo, *smodJSON))
	if err != nil {
		return err
	}
	data, err := makeDesktopData(report)
	if err != nil {
		return err
	}
	if *format == "json" {
		return writeJSON(os.Stdout, data)
	}
	if *format != "text" {
		return fmt.Errorf("unsupported format %q", *format)
	}
	printSummary(os.Stdout, data)
	return nil
}

func runDesktopData(args []string) error {
	fs := flag.NewFlagSet("desktop-data", flag.ContinueOnError)
	repo := fs.String("repo", ".", "repository root")
	if err := fs.Parse(args); err != nil {
		return err
	}
	report, err := loadSensitivity(resolveSmodPath(*repo, ""))
	if err != nil {
		return err
	}
	data, err := makeDesktopData(report)
	if err != nil {
		return err
	}
	return writeJSON(os.Stdout, data)
}

func runDoctor(args []string) error {
	fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
	repo := fs.String("repo", ".", "repository root")
	format := fs.String("format", "text", "output format: text or json")
	if err := fs.Parse(args); err != nil {
		return err
	}
	report := buildDoctorReport(*repo)
	if *format == "json" {
		return writeJSON(os.Stdout, report)
	}
	if *format != "text" {
		return fmt.Errorf("unsupported format %q", *format)
	}
	printDoctor(os.Stdout, report)
	if report.Status != "passed" {
		return errors.New("doctor checks failed")
	}
	return nil
}

func resolveSmodPath(repo string, explicit string) string {
	if explicit != "" {
		if filepath.IsAbs(explicit) {
			return explicit
		}
		return filepath.Join(repo, explicit)
	}
	return filepath.Join(repo, "data", "outputs", "P1_v2_smod_sensitivity_v1.json")
}

func loadSensitivity(path string) (*sensitivityReport, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var report sensitivityReport
	if err := json.NewDecoder(f).Decode(&report); err != nil {
		return nil, err
	}
	if report.SchemaVersion == "" || len(report.Variants) == 0 {
		return nil, fmt.Errorf("%s is not a TEG SMOD sensitivity report", path)
	}
	return &report, nil
}

func makeDesktopData(report *sensitivityReport) (*desktopData, error) {
	for _, required := range []string{"all_population", "urban", "rural"} {
		if _, ok := report.Variants[required]; !ok {
			return nil, fmt.Errorf("missing variant %q", required)
		}
	}
	tierIDs := []string{"point_supported_100km", "admin_supported", "union"}
	tierLabels := map[string]string{
		"point_supported_100km": "Point-supported 100 km",
		"admin_supported":       "Admin-supported areal",
		"union":                 "Non-additive union",
	}
	var tiers []appTier
	for _, id := range tierIDs {
		all := report.Variants["all_population"].Tiers[id]
		urban := report.Variants["urban"].Tiers[id]
		rural := report.Variants["rural"].Tiers[id]
		allPop := all.ExposedPopulation
		tiers = append(tiers, appTier{
			ID:                     id,
			Label:                  tierLabels[id],
			AllPopulation:          allPop,
			UrbanPopulation:        urban.ExposedPopulation,
			RuralPopulation:        rural.ExposedPopulation,
			UrbanShareOfTier:       safeDivide(urban.ExposedPopulation, allPop),
			RuralShareOfTier:       safeDivide(rural.ExposedPopulation, allPop),
			UrbanExposureRate:      urban.ShareOfVariantPopulation,
			RuralExposureRate:      rural.ShareOfVariantPopulation,
			AllPopulationDisplay:   formatPeople(allPop),
			UrbanPopulationDisplay: formatPeople(urban.ExposedPopulation),
			RuralPopulationDisplay: formatPeople(rural.ExposedPopulation),
		})
	}
	metrics := []appMetric{
		{
			ID:          "point_supported_100km",
			Label:       "Point-supported",
			Value:       report.Variants["all_population"].Tiers["point_supported_100km"].ExposedPopulation,
			Display:     formatPeople(report.Variants["all_population"].Tiers["point_supported_100km"].ExposedPopulation),
			Description: "People inside 100 km of high-confidence reported case coordinates.",
		},
		{
			ID:          "admin_supported",
			Label:       "Admin-supported",
			Value:       report.Variants["all_population"].Tiers["admin_supported"].ExposedPopulation,
			Display:     formatPeople(report.Variants["all_population"].Tiers["admin_supported"].ExposedPopulation),
			Description: "People inside matched administrative evidence units.",
		},
		{
			ID:          "union",
			Label:       "Union",
			Value:       report.Variants["all_population"].Tiers["union"].ExposedPopulation,
			Display:     formatPeople(report.Variants["all_population"].Tiers["union"].ExposedPopulation),
			Description: "Non-additive union of point and admin evidence tiers.",
		},
	}
	return &desktopData{
		SchemaVersion: "tegscan_desktop_v1",
		GeneratedAt:   report.CreatedAtUTC,
		Project:       "Topological Epidemic Geometry",
		Source:        report.SourceExposure,
		Metrics:       metrics,
		Tiers:         tiers,
		Checks:        report.Checks,
		Warnings:      report.Warnings,
		Inputs:        report.Inputs,
	}, nil
}

func buildDoctorReport(repo string) doctorReport {
	checks := []doctorCheck{}
	add := func(name string, ok bool, details string) {
		checks = append(checks, doctorCheck{Name: name, OK: ok, Details: details})
	}
	smodPath := filepath.Join(repo, "data", "outputs", "P1_v2_smod_sensitivity_v1.json")
	validationPath := filepath.Join(repo, "data", "outputs", "P1_v2_validation_v1.json")
	figurePath := filepath.Join(repo, "data", "outputs", "P1_v2_smod_sensitivity.png")
	globalExposurePath := filepath.Join(repo, "data", "outputs", "global_exposure_v1.json")

	report, err := loadSensitivity(smodPath)
	add("smod_sensitivity_json", err == nil, detailErr(err, smodPath))
	if err == nil {
		failed := failedSensitivityChecks(report.Checks)
		add("smod_reconciliation_checks", len(failed) == 0, strings.Join(failed, ","))
		if report.SourceExposure.Path != "" && report.SourceExposure.SHA256 != "" {
			sourcePath := filepath.Join(repo, report.SourceExposure.Path)
			hash, err := sha256File(sourcePath)
			add("source_exposure_hash", err == nil && hash == report.SourceExposure.SHA256, detailHash(err, hash, report.SourceExposure.SHA256))
		}
	}
	addFileCheck(&checks, "smod_sensitivity_figure", figurePath)
	addValidationCheck(&checks, validationPath)
	addFileCheck(&checks, "global_exposure_json", globalExposurePath)

	status := "passed"
	for _, check := range checks {
		if !check.OK {
			status = "failed"
			break
		}
	}
	return doctorReport{
		SchemaVersion: "tegscan_doctor_v1",
		Status:        status,
		Checks:        checks,
	}
}

func addFileCheck(checks *[]doctorCheck, name string, path string) {
	info, err := os.Stat(path)
	if err != nil {
		*checks = append(*checks, doctorCheck{Name: name, OK: false, Details: err.Error()})
		return
	}
	*checks = append(*checks, doctorCheck{Name: name, OK: true, Details: fmt.Sprintf("%s (%d bytes)", path, info.Size())})
}

func addValidationCheck(checks *[]doctorCheck, path string) {
	f, err := os.Open(path)
	if err != nil {
		*checks = append(*checks, doctorCheck{Name: "p1_v2_validation", OK: false, Details: err.Error()})
		return
	}
	defer f.Close()
	var payload struct {
		Status       string `json:"status"`
		ChecksPass   int    `json:"checks_passed"`
		ChecksFail   int    `json:"checks_failed"`
		Schema       string `json:"schema_version"`
		GeneratedUTC string `json:"created_at_utc"`
	}
	if err := json.NewDecoder(f).Decode(&payload); err != nil {
		*checks = append(*checks, doctorCheck{Name: "p1_v2_validation", OK: false, Details: err.Error()})
		return
	}
	ok := payload.Status == "passed" && payload.ChecksFail == 0
	*checks = append(*checks, doctorCheck{
		Name:    "p1_v2_validation",
		OK:      ok,
		Details: fmt.Sprintf("%s: %d passed, %d failed", payload.GeneratedUTC, payload.ChecksPass, payload.ChecksFail),
	})
}

func failedSensitivityChecks(checks map[string]bool) []string {
	failed := []string{}
	for name, ok := range checks {
		if !ok {
			failed = append(failed, name)
		}
	}
	sort.Strings(failed)
	return failed
}

func detailErr(err error, path string) string {
	if err == nil {
		return path
	}
	return err.Error()
}

func detailHash(err error, actual string, expected string) string {
	if err != nil {
		return err.Error()
	}
	return fmt.Sprintf("actual=%s expected=%s", actual, expected)
}

func sha256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	digest := sha256.New()
	if _, err := io.Copy(digest, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

func writeJSON(w io.Writer, value any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(value)
}

func printSummary(w io.Writer, data *desktopData) {
	fmt.Fprintf(w, "%s\n", data.Project)
	fmt.Fprintf(w, "generated: %s\n", data.GeneratedAt)
	fmt.Fprintln(w)
	for _, metric := range data.Metrics {
		fmt.Fprintf(w, "%-18s %s\n", metric.Label+":", metric.Display)
	}
	fmt.Fprintln(w)
	for _, tier := range data.Tiers {
		fmt.Fprintf(w, "%s\n", tier.Label)
		fmt.Fprintf(w, "  all:   %s\n", tier.AllPopulationDisplay)
		fmt.Fprintf(w, "  urban: %s (%s of tier, %s of urban population)\n",
			tier.UrbanPopulationDisplay,
			formatPercent(tier.UrbanShareOfTier),
			formatPercent(tier.UrbanExposureRate),
		)
		fmt.Fprintf(w, "  rural: %s (%s of tier, %s of rural population)\n",
			tier.RuralPopulationDisplay,
			formatPercent(tier.RuralShareOfTier),
			formatPercent(tier.RuralExposureRate),
		)
	}
}

func printDoctor(w io.Writer, report doctorReport) {
	fmt.Fprintf(w, "tegscan doctor: %s\n", report.Status)
	for _, check := range report.Checks {
		status := "ok"
		if !check.OK {
			status = "fail"
		}
		if check.Details != "" {
			fmt.Fprintf(w, "  [%s] %s: %s\n", status, check.Name, check.Details)
		} else {
			fmt.Fprintf(w, "  [%s] %s\n", status, check.Name)
		}
	}
}

func safeDivide(num float64, denom float64) float64 {
	if denom == 0 || math.IsNaN(denom) {
		return 0
	}
	return num / denom
}

func formatPeople(value float64) string {
	switch {
	case value >= 1_000_000_000:
		return fmt.Sprintf("%.2fB", value/1_000_000_000)
	case value >= 1_000_000:
		return fmt.Sprintf("%.1fM", value/1_000_000)
	default:
		return fmt.Sprintf("%.0f", value)
	}
}

func formatPercent(value float64) string {
	return fmt.Sprintf("%.1f%%", value*100)
}
