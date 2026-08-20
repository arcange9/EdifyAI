/* Edify AI — Electron main process.
 *
 * Secure architecture: contextIsolation enabled, nodeIntegration disabled.
 * All native operations go through the preload bridge → IPC channels.
 *
 * The app serves the built Vite SPA locally and provides IPC handlers for:
 *   - Secure credential storage (safeStorage / keytar-equivalent)
 *   - File system operations (open/save dialogs, file reading)
 *   - YouTube transcript extraction (via local server endpoint)
 *   - Auto-update via GitHub releases (electron-updater)
 */

import { app, BrowserWindow, ipcMain, shell, dialog, safeStorage } from "electron";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import http from "node:http";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isDev = !app.isPackaged;

let mainWindow: BrowserWindow | null = null;
let serverPort = 0;

/* ------------------------------------------------------------------ */
/* Auto-updater — checks GitHub releases for new versions             */
/* ------------------------------------------------------------------ */

let autoUpdater = null;

// electron-updater is only available in packaged builds (not dev)
if (!isDev) {
  try {
    // Dynamic import because electron-updater uses __dirname at top level
    const { autoUpdater: updater } = await import("electron-updater");
    autoUpdater = updater;

    // Configure update checking
    autoUpdater.autoDownload = false;     // Don't auto-download — let user choose
    autoUpdater.autoInstallOnAppQuit = true; // Install on quit after download

    autoUpdater.on("update-available", (info) => {
      console.log(`[Updater] Update available: ${info.version}`);
      if (mainWindow) {
        mainWindow.webContents.send("update:available", { version: info.version, releaseDate: info.releaseDate });
      }
    });

    autoUpdater.on("update-not-available", (info) => {
      console.log(`[Updater] No update available (current: ${info.version})`);
    });

    autoUpdater.on("download-progress", (progress) => {
      if (mainWindow) {
        mainWindow.webContents.send("update:progress", {
          percent: progress.percent,
          transferred: progress.transferred,
          total: progress.total,
        });
      }
    });

    autoUpdater.on("update-downloaded", (info) => {
      console.log(`[Updater] Update downloaded: ${info.version}`);
      if (mainWindow) {
        mainWindow.webContents.send("update:downloaded", { version: info.version });
      }
    });

    autoUpdater.on("error", (err) => {
      console.error("[Updater] Error:", err?.message || err);
      if (mainWindow) {
        mainWindow.webContents.send("update:error", { message: err?.message || "Update check failed" });
      }
    });
  } catch (err) {
    console.log("[Updater] electron-updater not available in dev mode:", err?.message);
  }
}

/* ------------------------------------------------------------------ */
/* Local static server — serves the built SPA to the webview           */
/* ------------------------------------------------------------------ */

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
  ".wasm": "application/wasm",
};

function startStaticServer(distDir: string): Promise<number> {
  const server = http.createServer((req, res) => {
    const urlPath = (req.url ?? "/").split("?")[0];
    let filePath = path.join(distDir, decodeURIComponent(urlPath));

    if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
      filePath = path.join(distDir, "index.html");
    }

    if (!fs.existsSync(filePath)) {
      res.writeHead(404).end("Not found");
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, {
      "content-type": MIME[ext] ?? "application/octet-stream",
      "cache-control": ext === ".html" ? "no-store" : "public, max-age=31536000",
    });
    fs.createReadStream(filePath).pipe(res);
  });

  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      resolve(server.address().port);
    });
  });
}

/* ------------------------------------------------------------------ */
/* Window creation                                                     */
/* ------------------------------------------------------------------ */

async function createWindow() {
  const distDir = path.join(__dirname, "..", "dist");

  let url: string;
  if (isDev) {
    url = "http://localhost:5173";
  } else {
    serverPort = await startStaticServer(distDir);
    url = `http://127.0.0.1:${serverPort}`;
  }

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 640,
    backgroundColor: "#0f1117",
    title: "Edify AI",
    icon: path.join(__dirname, "..", "build-resources", "icon.png"),
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.mjs"),
      sandbox: true,
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({ url: targetUrl }) => {
    if (/^https?:\/\//.test(targetUrl)) {
      shell.openExternal(targetUrl);
      return { action: "deny" };
    }
    return { action: "allow" };
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  await mainWindow.loadURL(url);

  if (isDev) {
    mainWindow.webContents.openDevTools({ mode: "detach" });
  }
}

/* ------------------------------------------------------------------ */
/* IPC Handlers                                                        */
/* ------------------------------------------------------------------ */

// --- Credential storage via safeStorage ---
ipcMain.handle("credentials:set", (_event, key: string, value: string) => {
  if (!safeStorage.isEncryptionAvailable()) {
    return false;
  }
  const encrypted = safeStorage.encryptString(value);
  const credPath = path.join(app.getPath("userData"), "credentials");
  fs.mkdirSync(credPath, { recursive: true });
  fs.writeFileSync(path.join(credPath, `${key}.cred`), encrypted);
  return true;
});

ipcMain.handle("credentials:get", (_event, key: string): string | null => {
  if (!safeStorage.isEncryptionAvailable()) {
    return null;
  }
  const credFile = path.join(app.getPath("userData"), "credentials", `${key}.cred`);
  if (!fs.existsSync(credFile)) return null;
  try {
    const encrypted = fs.readFileSync(credFile);
    return safeStorage.decryptString(encrypted);
  } catch {
    return null;
  }
});

ipcMain.handle("credentials:delete", (_event, key: string): boolean => {
  const credFile = path.join(app.getPath("userData"), "credentials", `${key}.cred`);
  if (fs.existsSync(credFile)) {
    fs.unlinkSync(credFile);
    return true;
  }
  return false;
});

// --- File dialogs ---
ipcMain.handle("dialog:openFile", async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    properties: ["openFile"],
    filters: [
      { name: "Documents", extensions: ["pdf", "docx", "doc", "txt", "md", "markdown"] },
      { name: "All Files", extensions: ["*"] },
    ],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  const filePath = result.filePaths[0];
  const buffer = fs.readFileSync(filePath);
  return {
    name: path.basename(filePath),
    path: filePath,
    buffer: buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
  };
});

ipcMain.handle("dialog:saveFile", async (_event, defaultName: string) => {
  const result = await dialog.showSaveDialog(mainWindow!, {
    defaultPath: defaultName,
    filters: [{ name: "All Files", extensions: ["*"] }],
  });
  return result.canceled ? null : result.filePath;
});

// --- File system ---
ipcMain.handle("fs:readFile", (_event, filePath: string) => {
  const buffer = fs.readFileSync(filePath);
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
});

ipcMain.handle("fs:writeFile", (_event, filePath: string, data: Uint8Array) => {
  fs.writeFileSync(filePath, data);
  return true;
});

ipcMain.handle("app:getVersion", () => app.getVersion());

ipcMain.handle("app:openExternal", (_event, url: string) => {
  shell.openExternal(url);
});

// --- Auto-update handlers ---
ipcMain.handle("update:check", async () => {
  if (!autoUpdater) return { available: false, message: "Updates not available in dev mode" };
  try {
    const result = await autoUpdater.checkForUpdates();
    return { available: result?.updateInfo != null, message: "Check complete" };
  } catch (err) {
    return { available: false, message: err?.message || "Check failed" };
  }
});

ipcMain.handle("update:download", async () => {
  if (!autoUpdater) return { ok: false, message: "Not available" };
  try {
    await autoUpdater.downloadUpdate();
    return { ok: true, message: "Download started" };
  } catch (err) {
    return { ok: false, message: err?.message || "Download failed" };
  }
});

ipcMain.handle("update:install", () => {
  if (!autoUpdater) return;
  // quitAndInstall will close the app and install the update
  autoUpdater.quitAndInstall(false, true);
});

ipcMain.handle("update:status", () => {
  if (!autoUpdater) return { available: false, downloaded: false, version: app.getVersion() };
  return { available: false, downloaded: false, version: app.getVersion() };
});

/* ------------------------------------------------------------------ */
/* App lifecycle                                                        */
/* ------------------------------------------------------------------ */

app.whenReady().then(async () => {
  await createWindow();
  // Check for updates after a short delay (only in packaged builds)
  if (autoUpdater && !isDev) {
    setTimeout(() => {
      autoUpdater.checkForUpdates().catch((err) => {
        console.error("[Updater] Auto-check failed:", err?.message);
      });
    }, 3000);
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
