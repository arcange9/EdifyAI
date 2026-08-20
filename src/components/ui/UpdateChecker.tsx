import { useState, useEffect } from "react";
import { Download, RefreshCw, Check, X, AlertCircle } from "lucide-react";

type EdifyAPI = {
  updater: {
    check: () => Promise<{ available: boolean; message: string }>;
    download: () => Promise<{ ok: boolean; message: string }>;
    install: () => Promise<void>;
    onAvailable: (cb: (info: { version: string; releaseDate?: string }) => void) => void;
    onProgress: (cb: (p: { percent: number; transferred: number; total: number }) => void) => void;
    onDownloaded: (cb: (info: { version: string }) => void) => void;
    onError: (cb: (err: { message: string }) => void) => void;
  };
};

function getAPI(): EdifyAPI | null {
  if (typeof window !== "undefined" && (window as unknown as Record<string, unknown>).edify) {
    return (window as unknown as Record<string, unknown>).edify as EdifyAPI;
  }
  return null;
}

export function UpdateChecker() {
  const [hasUpdate, setHasUpdate] = useState(false);
  const [version, setVersion] = useState("");
  const [downloading, setDownloading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [downloaded, setDownloaded] = useState(false);
  const [error, setError] = useState("");
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    const api = getAPI();
    if (!api) return;

    // Listen for update availability (auto-check runs on startup)
    api.updater.onAvailable((info) => {
      setHasUpdate(true);
      setVersion(info.version);
      setDismissed(false);
    });

    api.updater.onProgress((p) => {
      setProgress(Math.round(p.percent));
    });

    api.updater.onDownloaded((info) => {
      setDownloaded(true);
      setDownloading(false);
      setVersion(info.version);
    });

    api.updater.onError((err) => {
      setError(err.message);
      setDownloading(false);
    });
  }, []);

  if (!hasUpdate || dismissed) return null;

  async function handleDownload() {
    const api = getAPI();
    if (!api) return;
    setDownloading(true);
    setError("");
    await api.updater.download();
  }

  async function handleInstall() {
    const api = getAPI();
    if (!api) return;
    await api.updater.install();
  }

  return (
    <div className="card slide-up" style={{
      position: "fixed", bottom: 16, right: 16, zIndex: 9000,
      padding: 16, width: 340, maxWidth: "calc(100vw - 32px)",
      boxShadow: "var(--shadow-xl)", border: "1px solid var(--accent)",
    }}>
      {/* Close button */}
      <button
        onClick={() => setDismissed(true)}
        style={{ position: "absolute", top: 10, right: 10, background: "none", border: "none", cursor: "pointer", color: "var(--text-faint)" }}
      >
        <X size={16} />
      </button>

      {error ? (
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <AlertCircle size={20} style={{ color: "var(--error)" }} />
          <div>
            <div style={{ fontWeight: 700, fontSize: 14 }}>Update check failed</div>
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{error}</div>
          </div>
        </div>
      ) : downloaded ? (
        <>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
            <div style={{ width: 32, height: 32, borderRadius: "50%", background: "var(--success-bg)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Check size={18} style={{ color: "var(--success)" }} />
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 14 }}>Update ready to install</div>
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Restart to install version {version}</div>
            </div>
          </div>
          <button className="btn btn-primary btn-block" onClick={handleInstall}>
            <RefreshCw size={16} /> Restart & Install
          </button>
        </>
      ) : downloading ? (
        <>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
            <Download size={20} style={{ color: "var(--accent)" }} />
            <div>
              <div style={{ fontWeight: 700, fontSize: 14 }}>Downloading update...</div>
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{progress}% complete</div>
            </div>
          </div>
          <div style={{ height: 6, borderRadius: 3, background: "var(--surface-3)", overflow: "hidden" }}>
            <div style={{ height: "100%", borderRadius: 3, background: "var(--accent)", width: `${progress}%`, transition: "width 300ms ease" }} />
          </div>
        </>
      ) : (
        <>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
            <div style={{ width: 32, height: 32, borderRadius: "50%", background: "var(--accent-light)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <RefreshCw size={18} style={{ color: "var(--accent)" }} />
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 14 }}>Update available</div>
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Version {version} is ready to download</div>
            </div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn btn-ghost btn-sm" onClick={() => setDismissed(true)}>Later</button>
            <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={handleDownload}>
              <Download size={14} /> Download Update
            </button>
          </div>
        </>
      )}
    </div>
  );
}
