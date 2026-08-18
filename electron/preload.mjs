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
    set: (key: string, value: string) => ipcRenderer.invoke("credentials:set", key, value),
    get: (key: string) => ipcRenderer.invoke("credentials:get", key),
    delete: (key: string) => ipcRenderer.invoke("credentials:delete", key),
  },
  // File dialogs
  dialog: {
    openFile: () => ipcRenderer.invoke("dialog:openFile"),
    saveFile: (defaultName: string) => ipcRenderer.invoke("dialog:saveFile", defaultName),
  },
  // File system
  fs: {
    readFile: (filePath: string) => ipcRenderer.invoke("fs:readFile", filePath),
    writeFile: (filePath: string, data: Uint8Array) => ipcRenderer.invoke("fs:writeFile", filePath, data),
  },
  // App info
  app: {
    getVersion: () => ipcRenderer.invoke("app:getVersion"),
    openExternal: (url: string) => ipcRenderer.invoke("app:openExternal", url),
  },
  // Platform
  platform: process.platform,
});
