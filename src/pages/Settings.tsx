import { useState } from "react";
import { useApp } from "../lib/app-context";
import { v4 as uuid } from "uuid";
import {
  Settings as SettingsIcon, KeyRound, Palette, Database, Eye, EyeOff,
  Check, X, Trash2, Info, Monitor, Moon, Sun, Keyboard, Shield,
  Zap, BookOpen, RefreshCw, Download, AlertCircle, Loader2,
  Cpu, Layers, Globe,
} from "lucide-react";
import type { ProviderConfig, ProviderType } from "../lib/types";
import { providerDisplayName, providerDescription, defaultBaseUrl } from "../ai/providers/manager";

type Tab = "general" | "appearance" | "providers" | "models" | "storage" | "privacy" | "shortcuts" | "about";

export default function Settings() {
  const { preferences, setPreferences, providers, saveProviderConfig, setActiveProvider } = useApp();
  const [tab, setTab] = useState<Tab>("providers");
  const [updateState, setUpdateState] = useState<"idle" | "checking" | "available" | "downloading" | "ready" | "error" | "latest">("idle");
  const [updateMsg, setUpdateMsg] = useState("");

  const tabs = [
    { key: "providers" as Tab, label: "AI Providers", icon: KeyRound },
    { key: "models" as Tab, label: "Models", icon: Cpu },
    { key: "appearance" as Tab, label: "Appearance", icon: Palette },
    { key: "general" as Tab, label: "General", icon: SettingsIcon },
    { key: "storage" as Tab, label: "Storage", icon: Database },
    { key: "privacy" as Tab, label: "Privacy", icon: Shield },
    { key: "shortcuts" as Tab, label: "Shortcuts", icon: Keyboard },
    { key: "about" as Tab, label: "About", icon: Info },
  ];

  async function checkForUpdates() {
    const edify = (window as unknown as Record<string, { updater: { check: () => Promise<{ available: boolean; message: string }> } }>).edify;
    if (!edify?.updater) {
      setUpdateState("error");
      setUpdateMsg("Updates not available — running in development mode.");
      return;
    }
    setUpdateState("checking");
    setUpdateMsg("");
    try {
      const result = await edify.updater.check();
      if (result.available) {
        setUpdateState("available");
        setUpdateMsg(result.message || "A new version is available.");
      } else {
        setUpdateState("latest");
        setUpdateMsg("You are running the latest version.");
      }
    } catch (err) {
      setUpdateState("error");
      setUpdateMsg(err instanceof Error ? err.message : "Update check failed.");
    }
  }

  async function downloadUpdate() {
    const edify = (window as unknown as Record<string, { updater: { download: () => Promise<{ ok: boolean; message: string }> } }>).edify;
    if (!edify?.updater) return;
    setUpdateState("downloading");
    try {
      const result = await edify.updater.download();
      if (result.ok) {
        setUpdateState("ready");
        setUpdateMsg("Update downloaded. Click to install and restart.");
      } else {
        setUpdateState("error");
        setUpdateMsg(result.message || "Download failed.");
      }
    } catch (err) {
      setUpdateState("error");
      setUpdateMsg(err instanceof Error ? err.message : "Download failed.");
    }
  }

  async function installUpdate() {
    const edify = (window as unknown as Record<string, { updater: { install: () => Promise<void> } }>).edify;
    if (!edify?.updater) return;
    await edify.updater.install();
  }

  return (
    <div style={{ flex: 1, overflow: "hidden", display: "flex" }}>
      {/* Settings nav */}
      <div style={{
        width: 220, minWidth: 220,
        background: "var(--bg-sidebar)",
        borderRight: "1px solid var(--border)",
        padding: "24px 12px",
        overflowY: "auto",
      }}>
        <h2 style={{ fontSize: 18, fontWeight: 800, padding: "0 8px 16px" }}>Settings</h2>
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`nav-item ${tab === t.key ? "nav-item-active" : ""}`}
          >
            <t.icon size={18} style={{ flexShrink: 0 }} />
            <span>{t.label}</span>
          </button>
        ))}
      </div>

      {/* Settings content */}
      <div className="scroll-container" style={{ padding: "32px 40px", flex: 1 }}>
        <div style={{ maxWidth: 640, margin: "0 auto" }}>
          {tab === "providers" && (
            <ProvidersTab
              providers={providers}
              preferences={preferences}
              saveProviderConfig={saveProviderConfig}
              setActiveProvider={setActiveProvider}
            />
          )}

          {tab === "models" && (
            <ModelsTab providers={providers} preferences={preferences} setActiveProvider={setActiveProvider} />
          )}

          {tab === "appearance" && (
            <div className="fade-in">
              <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>Appearance</h3>
              <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 24 }}>
                Customize how Edify AI looks.
              </p>
              <div style={{ marginBottom: 28 }}>
                <label style={{ fontSize: 14, fontWeight: 600, display: "block", marginBottom: 10 }}>Theme</label>
                <div style={{ display: "flex", gap: 10 }}>
                  {([
                    { val: "light", icon: Sun, label: "Light" },
                    { val: "dark", icon: Moon, label: "Dark" },
                    { val: "system", icon: Monitor, label: "System" },
                  ] as const).map((t) => (
                    <button
                      key={t.val}
                      onClick={() => setPreferences({ theme: t.val })}
                      className={preferences.theme === t.val ? "btn btn-primary" : "btn btn-outline"}
                      style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, padding: "16px 20px", minWidth: 80 }}
                    >
                      <t.icon size={20} />
                      {t.label}
                    </button>
                  ))}
                </div>
              </div>
              <div style={{ marginBottom: 28 }}>
                <label style={{ fontSize: 14, fontWeight: 600, display: "block", marginBottom: 10 }}>Language</label>
                <input
                  className="input"
                  style={{ maxWidth: 300 }}
                  defaultValue={preferences.language}
                  onBlur={(e) => setPreferences({ language: e.target.value })}
                  placeholder="English"
                />
              </div>
            </div>
          )}

          {tab === "general" && (
            <div className="fade-in">
              <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>General</h3>
              <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 24 }}>
                Core application settings.
              </p>
              <div className="card" style={{ padding: 20, marginBottom: 16 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                  <Zap size={18} style={{ color: "var(--accent)" }} />
                  <span style={{ fontWeight: 600, fontSize: 15 }}>Fallback Provider</span>
                </div>
                <label style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12, cursor: "pointer" }}>
                  <input type="checkbox" className="checkbox" checked={preferences.fallbackEnabled}
                    onChange={(e) => setPreferences({ fallbackEnabled: e.target.checked })} />
                  <span style={{ fontSize: 14 }}>Enable fallback to secondary provider on failure</span>
                </label>
                <div style={{ fontSize: 12, color: "var(--text-muted)", lineHeight: 1.5 }}>
                  When the primary provider fails with a temporary error, Edify AI will try your fallback provider.
                </div>
              </div>
            </div>
          )}

          {tab === "storage" && (
            <div className="fade-in">
              <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>Storage & Data</h3>
              <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 24 }}>
                Manage your local data. All data is stored on your device.
              </p>
              <div className="card" style={{ padding: 20, marginBottom: 16 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                  <Database size={18} style={{ color: "var(--accent)" }} />
                  <span style={{ fontWeight: 600, fontSize: 15 }}>Export Data</span>
                </div>
                <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 12 }}>
                  Download all your projects, documents, notes, and settings as a JSON file.
                </p>
                <button className="btn btn-outline" onClick={() => alert("Export feature coming soon")}>
                  <Database size={16} /> Export Data
                </button>
              </div>
              <div className="card" style={{ padding: 20, border: "1px solid var(--error-light)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                  <Shield size={18} style={{ color: "var(--error)" }} />
                  <span style={{ fontWeight: 600, fontSize: 15, color: "var(--error)" }}>Danger Zone</span>
                </div>
                <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 12 }}>
                  Delete ALL local data including projects, documents, notes, flashcards, and quizzes. This cannot be undone.
                </p>
                <button className="btn btn-danger" onClick={() => {
                  if (confirm("Delete ALL local data? This cannot be undone.")) {
                    indexedDB.deleteDatabase("edifyai");
                    location.reload();
                  }
                }}>
                  <Trash2 size={16} /> Delete Local Data
                </button>
              </div>
            </div>
          )}

          {tab === "privacy" && (
            <div className="fade-in">
              <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>Privacy</h3>
              <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 24 }}>
                Your data privacy and security settings.
              </p>
              <div className="card" style={{ padding: 20, marginBottom: 16 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                  <Shield size={18} style={{ color: "var(--accent)" }} />
                  <span style={{ fontWeight: 600, fontSize: 15 }}>API Key Storage</span>
                </div>
                <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 12, lineHeight: 1.6 }}>
                  Your API keys are stored using your operating system's secure credential storage (Electron safeStorage). They never leave your device and are never sent to any server except the AI provider you've configured.
                </p>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8 }}>
                  <span className="status-dot success" />
                  <span style={{ fontSize: 13, color: "var(--success)", fontWeight: 600 }}>Secure storage active</span>
                </div>
              </div>
              <div className="card" style={{ padding: 20, marginBottom: 16 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                  <Database size={18} style={{ color: "var(--accent)" }} />
                  <span style={{ fontWeight: 600, fontSize: 15 }}>Data Storage</span>
                </div>
                <p style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.6 }}>
                  All your study data (documents, notes, flashcards, quizzes, chat history) is stored locally in your browser's IndexedDB. No data is uploaded to any cloud service.
                </p>
              </div>
              <div className="card" style={{ padding: 20 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                  <Globe size={18} style={{ color: "var(--accent)" }} />
                  <span style={{ fontWeight: 600, fontSize: 15 }}>AI Provider Communication</span>
                </div>
                <p style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.6 }}>
                  When you use AI features, your study materials are sent directly to your configured AI provider (OpenRouter, Google AI, Groq, or your custom endpoint) for processing. Edify AI does not intercept, log, or store this communication.
                </p>
              </div>
            </div>
          )}

          {tab === "shortcuts" && (
            <div className="fade-in">
              <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>Keyboard Shortcuts</h3>
              <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 24 }}>
                Speed up your workflow with these shortcuts.
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {[
                  { keys: "Ctrl + N", action: "New Study Project" },
                  { keys: "Ctrl + K", action: "Quick Search" },
                  { keys: "Ctrl + ,", action: "Open Settings" },
                  { keys: "Ctrl + B", action: "Toggle Sidebar" },
                  { keys: "Esc", action: "Close Modal / Dialog" },
                ].map((s) => (
                  <div key={s.keys} className="card" style={{ padding: "12px 16px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <span style={{ fontSize: 14 }}>{s.action}</span>
                    <kbd style={{
                      padding: "4px 10px", borderRadius: "var(--radius-sm)",
                      background: "var(--surface-3)", border: "1px solid var(--border)",
                      fontSize: 12, fontFamily: "var(--font-mono)", fontWeight: 600,
                    }}>
                      {s.keys}
                    </kbd>
                  </div>
                ))}
              </div>
            </div>
          )}

          {tab === "about" && (
            <div className="fade-in">
              <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 24 }}>
                <div style={{
                  width: 56, height: 56, borderRadius: "var(--radius-lg)",
                  background: "var(--brand-gradient)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  boxShadow: "var(--shadow-brand)",
                }}>
                  <BookOpen size={28} color="white" />
                </div>
                <div>
                  <h2 style={{ fontSize: 22, fontWeight: 800 }}>Edify AI</h2>
                  <p style={{ fontSize: 14, color: "var(--text-muted)" }}>Turn Knowledge Into Understanding.</p>
                </div>
              </div>

              <div className="card" style={{ padding: 20, marginBottom: 16 }}>
                <div style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 4 }}>Version 1.1.1</div>
                <div style={{ fontSize: 14, color: "var(--text-muted)" }}>Built for students, learners, and educators.</div>
              </div>

              {/* Update section */}
              <div className="card" style={{ padding: 20, marginBottom: 16 }}>
                <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 12, display: "flex", alignItems: "center", gap: 8 }}>
                  <RefreshCw size={16} style={{ color: "var(--accent)" }} /> Updates
                </h3>
                {updateState === "idle" && (
                  <button className="btn btn-outline" onClick={checkForUpdates}>
                    <RefreshCw size={16} /> Check for Updates
                  </button>
                )}
                {updateState === "checking" && (
                  <div style={{ display: "flex", alignItems: "center", gap: 10, color: "var(--text-muted)" }}>
                    <Loader2 size={18} className="animate-spin" style={{ color: "var(--accent)" }} />
                    <span style={{ fontSize: 14 }}>Checking for updates...</span>
                  </div>
                )}
                {updateState === "latest" && (
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{ width: 28, height: 28, borderRadius: "50%", background: "var(--success-bg)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <Check size={16} style={{ color: "var(--success)" }} />
                    </div>
                    <span style={{ fontSize: 14, color: "var(--text-secondary)" }}>{updateMsg}</span>
                  </div>
                )}
                {updateState === "available" && (
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                      <div style={{ width: 28, height: 28, borderRadius: "50%", background: "var(--accent-light)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <Download size={16} style={{ color: "var(--accent)" }} />
                      </div>
                      <span style={{ fontSize: 14, color: "var(--text-secondary)" }}>{updateMsg}</span>
                    </div>
                    <button className="btn btn-primary" onClick={downloadUpdate}>
                      <Download size={16} /> Download Update
                    </button>
                  </div>
                )}
                {updateState === "downloading" && (
                  <div style={{ display: "flex", alignItems: "center", gap: 10, color: "var(--text-muted)" }}>
                    <Loader2 size={18} className="animate-spin" style={{ color: "var(--accent)" }} />
                    <span style={{ fontSize: 14 }}>Downloading update...</span>
                  </div>
                )}
                {updateState === "ready" && (
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                      <div style={{ width: 28, height: 28, borderRadius: "50%", background: "var(--success-bg)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <Check size={16} style={{ color: "var(--success)" }} />
                      </div>
                      <span style={{ fontSize: 14, color: "var(--text-secondary)" }}>{updateMsg}</span>
                    </div>
                    <button className="btn btn-primary" onClick={installUpdate}>
                      <RefreshCw size={16} /> Restart & Install
                    </button>
                  </div>
                )}
                {updateState === "error" && (
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <AlertCircle size={18} style={{ color: "var(--error)" }} />
                    <span style={{ fontSize: 14, color: "var(--text-muted)" }}>{updateMsg}</span>
                    <button className="btn btn-ghost btn-sm" onClick={checkForUpdates}>Retry</button>
                  </div>
                )}
              </div>

              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 24 }}>
                <button className="btn btn-outline" onClick={() => window.open("https://github.com/arcange9/EdifyAI", "_blank")}>GitHub</button>
                <button className="btn btn-outline" onClick={() => window.open("https://github.com/arcange9/EdifyAI#readme", "_blank")}>Documentation</button>
                <button className="btn btn-outline" onClick={() => window.open("https://github.com/arcange9/EdifyAI/issues", "_blank")}>Feedback</button>
              </div>

              <div className="card" style={{ padding: 20, marginBottom: 16 }}>
                <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
                  <BookOpen size={16} style={{ color: "var(--accent)" }} /> Study Books
                </h3>
                <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 12 }}>
                  Free textbooks by the same author — download as DOCX:
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <button className="btn btn-outline btn-block" style={{ justifyContent: "flex-start" }} onClick={() => window.open("https://github.com/arcange9/python-mastery-notes/raw/main/docx/Python_Mastery_Notes_Complete_Book.docx", "_blank")}>
                    Python Mastery Notes — Complete Book
                  </button>
                  <button className="btn btn-outline btn-block" style={{ justifyContent: "flex-start" }} onClick={() => window.open("https://github.com/arcange9/ai-engineering-mastery/raw/main/docx/AI-Engineering-Mastery-90-Day-Journey.docx", "_blank")}>
                    AI Engineering Mastery — Complete Book
                  </button>
                  <button className="btn btn-outline btn-block" style={{ justifyContent: "flex-start" }} onClick={() => window.open("https://github.com/arcange9/ethical-hacking-book/raw/main/docx/Ethical_Hacking_Complete_Book.docx", "_blank")}>
                    Ethical Hacking — Complete Book
                  </button>
                </div>
              </div>

              <div style={{ textAlign: "center" }}>
                <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 4 }}>
                  Designed by Mukamyi Izere Arcange
                </p>
                <p style={{ fontSize: 12, color: "var(--text-faint)" }}>
                  Edify AI is an independent project and is not affiliated with NitroAI.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ── MODELS TAB ───────────────────────────────────────────────────── */

function ModelsTab({ providers, preferences, setActiveProvider }: {
  providers: ProviderConfig[];
  preferences: { defaultProviderId?: string };
  setActiveProvider: (id: string) => Promise<void>;
}) {
  const configuredProviders = providers.filter((p) => p.apiKey);
  const [search, setSearch] = useState("");

  if (configuredProviders.length === 0) {
    return (
      <div className="fade-in">
        <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>Models</h3>
        <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 24 }}>
          Select and manage AI models for your configured providers.
        </p>
        <div className="empty-state">
          <div className="empty-state-icon"><Cpu size={28} strokeWidth={1.8} /></div>
          <div className="empty-state-title">No providers configured</div>
          <div className="empty-state-desc">Configure an AI provider first to see and select available models.</div>
        </div>
      </div>
    );
  }

  return (
    <div className="fade-in">
      <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>Models</h3>
      <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 20 }}>
        Select which AI model each provider uses.
      </p>

      {/* Search */}
      <div style={{ position: "relative", marginBottom: 20, maxWidth: 400 }}>
        <input className="input" style={{ paddingLeft: 36 }} placeholder="Search models..." value={search} onChange={(e) => setSearch(e.target.value)} />
        <div style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: "var(--text-faint)" }}>
          <Cpu size={16} />
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        {configuredProviders.map((provider) => {
          const isActive = provider.id === preferences.defaultProviderId;
          const modelMatches = !search || (provider.model || "").toLowerCase().includes(search.toLowerCase());
          return (
            <div key={provider.id} style={{ display: modelMatches ? "block" : "none" }}>
              {/* Provider header */}
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                <div style={{
                  width: 36, height: 36, borderRadius: "var(--radius)",
                  background: "var(--accent-light)", display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                  <KeyRound size={18} style={{ color: "var(--accent)" }} />
                </div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 14 }}>{provider.name}</div>
                  {isActive && <span className="badge badge-success" style={{ marginTop: 2 }}><Check size={10} /> Active</span>}
                </div>
              </div>

              {/* Model card */}
              <div className="card" style={{ padding: 16 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <Cpu size={18} style={{ color: "var(--text-muted)" }} />
                    <div>
                      <div style={{ fontFamily: "var(--font-mono)", fontSize: 13, fontWeight: 600 }}>
                        {provider.model || "Auto-select"}
                      </div>
                      <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
                        <span className="badge badge-neutral"><Layers size={9} /> Text</span>
                        <span className="badge badge-neutral"><Globe size={9} /> Streaming</span>
                      </div>
                    </div>
                  </div>
                  <button
                    className="btn btn-outline btn-sm"
                    onClick={() => setActiveProvider(provider.id)}
                    disabled={isActive}
                  >
                    {isActive ? "Active" : "Set Active"}
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── PROVIDERS TAB ──────────────────────────────────────────────── */

const PROVIDER_TYPES: ProviderType[] = ["openrouter", "google", "groq", "custom"];

function ProvidersTab({ providers, preferences, saveProviderConfig, setActiveProvider }: {
  providers: ProviderConfig[];
  preferences: { defaultProviderId?: string; fallbackProviderId?: string; fallbackEnabled: boolean };
  saveProviderConfig: (config: ProviderConfig, apiKey?: string) => Promise<void>;
  setActiveProvider: (id: string) => Promise<void>;
}) {
  const [editing, setEditing] = useState<ProviderConfig | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [modelName, setModelName] = useState("");
  const [customName, setCustomName] = useState("");
  const [customBaseUrl, setCustomBaseUrl] = useState("");
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);

  function startEdit(type: ProviderType) {
    const existing = providers.find((p) => p.type === type);
    setEditing(existing || {
      id: uuid(), type, name: providerDisplayName(type),
      apiKey: "", baseUrl: defaultBaseUrl(type), model: "",
      enabled: true, createdAt: Date.now(),
    });
    setApiKey(existing?.apiKey || "");
    setModelName(existing?.model || "");
    setCustomName(existing?.name || "");
    setCustomBaseUrl(existing?.baseUrl || "");
    setShowKey(false);
    setTestResult(null);
  }

  async function handleSave() {
    if (!editing) return;
    const config = { ...editing, apiKey, model: modelName, name: customName || providerDisplayName(editing.type), baseUrl: customBaseUrl || defaultBaseUrl(editing.type) };
    await saveProviderConfig(config, apiKey);
    await setActiveProvider(config.id);
    setEditing(null);
  }

  async function handleTest() {
    if (!editing || !apiKey) return;
    setTesting(true);
    setTestResult(null);
    try {
      const { createProvider } = await import("../ai/providers/manager");
      const config = { ...editing, apiKey, model: modelName || "gpt-4o-mini" };
      const provider = await createProvider(config);
      if (provider) {
        const health = await provider.healthCheck();
        setTestResult({ ok: health.ok, message: health.message });
      } else {
        setTestResult({ ok: false, message: "Could not create provider" });
      }
    } catch (err) {
      setTestResult({ ok: false, message: err instanceof Error ? err.message : "Connection failed" });
    }
    setTesting(false);
  }

  return (
    <div className="fade-in">
      <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>AI Providers</h3>
      <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 24 }}>
        Configure how Edify AI powers your learning. API keys are stored securely on your device.
      </p>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {PROVIDER_TYPES.map((type) => {
          const configs = providers.filter((p) => p.type === type);
          const configured = configs.length > 0;
          const isActive = configs.some((c) => c.id === preferences.defaultProviderId);
          return (
            <div key={type} className="card" style={{ padding: 18 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div style={{
                    width: 44, height: 44, borderRadius: "var(--radius)",
                    background: configured ? "var(--accent-light)" : "var(--surface-3)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                  }}>
                    <KeyRound size={20} style={{ color: configured ? "var(--accent)" : "var(--text-faint)" }} />
                  </div>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 15 }}>{providerDisplayName(type)}</div>
                    <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>{providerDescription(type)}</div>
                  </div>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  {isActive && <span className="badge badge-success"><Check size={12} /> Active</span>}
                  {configured && !isActive && <span className="status-dot success" />}
                  {!configured && <span className="status-dot neutral" />}
                  <button className="btn btn-outline btn-sm" onClick={() => startEdit(type)}>
                    {configured ? "Configure" : "Setup"}
                  </button>
                </div>
              </div>
              {configured && configs[0].model && (
                <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--border)", display: "flex", alignItems: "center", gap: 8 }}>
                  <span className="badge badge-neutral">Model</span>
                  <span style={{ fontSize: 13, color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>{configs[0].model}</span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Edit modal */}
      {editing && (
        <div className="modal-overlay" onClick={() => setEditing(null)}>
          <div className="modal" style={{ width: 480 }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Configure {providerDisplayName(editing.type)}</h3>
              <button className="btn btn-ghost btn-icon" onClick={() => setEditing(null)}>✕</button>
            </div>
            <div className="modal-body">
              {editing.type === "custom" && (
                <>
                  <div style={{ marginBottom: 16 }}>
                    <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Provider Name</label>
                    <input className="input" placeholder="My AI Provider" value={customName} onChange={(e) => setCustomName(e.target.value)} />
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Base URL</label>
                    <input className="input" placeholder="https://example.com/v1" value={customBaseUrl} onChange={(e) => setCustomBaseUrl(e.target.value)} />
                  </div>
                </>
              )}
              <div style={{ marginBottom: 16 }}>
                <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>API Key</label>
                <div style={{ position: "relative" }}>
                  <input
                    className="input"
                    type={showKey ? "text" : "password"}
                    placeholder="••••••••••••"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                  />
                  <button onClick={() => setShowKey(!showKey)}
                    style={{ position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)" }}>
                    {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>
              <div style={{ marginBottom: 16 }}>
                <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Model (optional)</label>
                <input className="input" placeholder="Auto-select if empty" value={modelName} onChange={(e) => setModelName(e.target.value)} />
              </div>
              <p style={{ fontSize: 12, color: "var(--text-faint)", marginBottom: 16 }}>
                Your API key is stored locally using secure OS-level storage. It never leaves your device.
              </p>
              {testResult && (
                <div style={{
                  padding: "12px 16px", borderRadius: "var(--radius)", marginBottom: 16,
                  background: testResult.ok ? "var(--success-bg)" : "var(--error-bg)",
                  display: "flex", alignItems: "center", gap: 8,
                }}>
                  {testResult.ok ? <Check size={16} style={{ color: "var(--success)" }} /> : <X size={16} style={{ color: "var(--error)" }} />}
                  <span style={{ fontSize: 13, color: testResult.ok ? "var(--success)" : "var(--error)" }}>{testResult.message}</span>
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn btn-outline" onClick={handleTest} disabled={!apiKey.trim() || testing}>
                {testing ? "Testing..." : "Test Connection"}
              </button>
              <button className="btn btn-primary" onClick={handleSave} disabled={!apiKey.trim()}>
                <Check size={16} /> Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
