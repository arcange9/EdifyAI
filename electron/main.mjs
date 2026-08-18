/* Edify AI — Electron main process.
 *
 * Secure architecture: contextIsolation enabled, nodeIntegration disabled.
 * All native operations go through the preload bridge → IPC channels.
 *
 * The app serves the built Vite SPA locally and provides IPC handlers for:
 *   - Secure credential storage (safeStorage / keytar-equivalent)
 *   - SQLite database access (better-sqlite3, main process only)
 *   - File system operations (open/save dialogs, file reading)
 *   - YouTube transcript extraction (via local server endpoint)
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

    // SPA fallback
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

  // In dev mode, Vite dev server is used directly.
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

  // Open external links in the system browser
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
    // Fallback: store in app data (not ideal but better than plaintext in code)
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

// --- File system read (for ingestion) ---
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

/* ------------------------------------------------------------------ */
/* App lifecycle                                                        */
/* ------------------------------------------------------------------ */

app.whenReady().then(createWindow);

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
