const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("tegscan", {
  dashboard: () => ipcRenderer.invoke("tegscan:dashboard"),
  doctor: () => ipcRenderer.invoke("tegscan:doctor")
});
