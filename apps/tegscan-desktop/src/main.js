const { app, BrowserWindow, ipcMain } = require("electron");
const { execFile } = require("node:child_process");
const path = require("node:path");

const repoRoot = path.resolve(__dirname, "..", "..", "..");

function tegscanBinary() {
  if (process.env.TEGSCAN_BIN) {
    return process.env.TEGSCAN_BIN;
  }
  const name = process.platform === "win32" ? "tegscan.exe" : "tegscan";
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "bin", name);
  }
  return path.join(repoRoot, "tools", "tegscan", "bin", name);
}

function runTegscan(args) {
  return new Promise((resolve, reject) => {
    execFile(
      tegscanBinary(),
      args,
      {
        cwd: repoRoot,
        maxBuffer: 8 * 1024 * 1024
      },
      (error, stdout, stderr) => {
        if (error) {
          const detail = stderr ? `${error.message}\n${stderr}` : error.message;
          reject(new Error(detail));
          return;
        }
        resolve(stdout);
      }
    );
  });
}

async function loadDashboard() {
  const stdout = await runTegscan(["desktop-data", "--repo", repoRoot]);
  return JSON.parse(stdout);
}

async function runDoctor() {
  const stdout = await runTegscan(["doctor", "--repo", repoRoot, "--format", "json"]);
  return JSON.parse(stdout);
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1240,
    height: 820,
    minWidth: 980,
    minHeight: 680,
    title: "TEG Scan",
    backgroundColor: "#f7f8fb",
    titleBarStyle: "hiddenInset",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  win.loadFile(path.join(__dirname, "index.html"));
}

ipcMain.handle("tegscan:dashboard", loadDashboard);
ipcMain.handle("tegscan:doctor", runDoctor);

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
