/* Edify AI — Preload script.
 *
 * Secure bridge between the renderer (React app) and the main process.
 * Exposes a minimal, typed API via contextBridge — never exposes Node.js
 * or Electron directly to the renderer.
 */

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("edify", {
  // Secure credential storage
  credentials: {
    set: (key, value) => ipcRenderer.invoke("credentials:set", key, value),
    get: (key) => ipcRenderer.invoke("credentials:get", key),
    delete: (key) => ipcRenderer.invoke("credentials:delete", key),
  },
  // File dialogs
  dialog: {
    openFile: () => ipcRenderer.invoke("dialog:openFile"),
    saveFile: (defaultName) => ipcRenderer.invoke("dialog:saveFile", defaultName),
  },
  // File system
  fs: {
    readFile: (filePath) => ipcRenderer.invoke("fs:readFile", filePath),
    writeFile: (filePath, data) => ipcRenderer.invoke("fs:writeFile", filePath, data),
  },
  // App info
  app: {
    getVersion: () => ipcRenderer.invoke("app:getVersion"),
    openExternal: (url) => ipcRenderer.invoke("app:openExternal", url),
  },
  // Auto-update
  updater: {
    check: () => ipcRenderer.invoke("update:check"),
    download: () => ipcRenderer.invoke("update:download"),
    install: () => ipcRenderer.invoke("update:install"),
    onAvailable: (callback) => ipcRenderer.on("update:available", (_e, info) => callback(info)),
    onProgress: (callback) => ipcRenderer.on("update:progress", (_e, progress) => callback(progress)),
    onDownloaded: (callback) => ipcRenderer.on("update:downloaded", (_e, info) => callback(info)),
    onError: (callback) => ipcRenderer.on("update:error", (_e, err) => callback(err)),
  },
});
