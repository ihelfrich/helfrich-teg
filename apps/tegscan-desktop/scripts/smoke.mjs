import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..", "..", "..");
const bin = path.join(
  repoRoot,
  "tools",
  "tegscan",
  "bin",
  process.platform === "win32" ? "tegscan.exe" : "tegscan"
);

if (!existsSync(bin)) {
  throw new Error(`Missing tegscan binary at ${bin}`);
}

const stdout = execFileSync(bin, ["desktop-data", "--repo", repoRoot], {
  cwd: repoRoot,
  encoding: "utf8"
});
const payload = JSON.parse(stdout);

if (payload.schema_version !== "tegscan_desktop_v1") {
  throw new Error(`Unexpected schema: ${payload.schema_version}`);
}
if (!Array.isArray(payload.metrics) || payload.metrics.length !== 3) {
  throw new Error("Expected three dashboard metrics");
}
if (!Array.isArray(payload.tiers) || payload.tiers.length !== 3) {
  throw new Error("Expected three evidence tiers");
}

console.log(`desktop smoke passed: ${payload.metrics.map((m) => m.display).join(", ")}`);
