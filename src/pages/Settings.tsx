import { useState } from "react";
import { useApp } from "../lib/app-context";
import { v4 as uuid } from "uuid";
import { Settings as SettingsIcon, KeyRound, Palette, Database, Eye, EyeOff, Check, X, Trash2, Info } from "lucide-react";
import type { ProviderConfig, ProviderType } from "../lib/types";
import { providerDisplayName, providerDescription, defaultBaseUrl } from "../ai/providers/manager";

type Tab = "general" | "providers" | "appearance" | "about";
const PROVIDER_TYPES: ProviderType[] = ["openrouter", "google", "groq", "custom"];

export default function Settings() {
  const { preferences, setPreferences, providers, saveProviderConfig, deleteProviderConfig, setActiveProvider } = useApp();
  const [tab, setTab] = useState<Tab>("providers");

  const tabs = [
    { key: "providers" as Tab, label: "AI Providers", icon: KeyRound },
    { key: "appearance" as Tab, label: "Appearance", icon: Palette },
    { key: "general" as Tab, label: "General", icon: SettingsIcon },
    { key: "about" as Tab, label: "About", icon: Info },
  ];

  return (
    <div style={{ flex: 1, overflowY: "auto", padding: "32px 48px" }}>
      <h1 style={{ fontSize: 24, fontWeight: 800, marginBottom: 24 }}>Settings</h1>
      <div style={{ display: "flex", gap: 4, marginBottom: 24, borderBottom: "1px solid var(--border)" }}>
        {tabs.map((t) => (
          <button key={t.key} onClick={() => setTab(t.key)} style={{
            display: "flex", alignItems: "center", gap: 8, padding: "10px 16px", border: "none", background: "none",
            fontSize: 14, fontWeight: 600, cursor: "pointer",
            color: tab === t.key ? "var(--accent)" : "var(--text-muted)",
            borderBottom: tab === t.key ? "2px solid var(--accent)" : "2px solid transparent",
          }}>
            <t.icon size={16} /> {t.label}
          </button>
        ))}
      </div>

      {tab === "providers" && (
        <ProvidersTab
          providers={providers}
          preferences={preferences}
          saveProviderConfig={saveProviderConfig}
          deleteProviderConfig={deleteProviderConfig}
          setActiveProvider={setActiveProvider}
        />
      )}

      {tab === "appearance" && (
        <div className="fade-in" style={{ maxWidth: 500 }}>
          <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>Theme</h3>
          <div style={{ display: "flex", gap: 10 }}>
            {(["light", "dark", "system"] as const).map((t) => (
              <button key={t} onClick={() => setPreferences({ theme: t })}
                className={preferences.theme === t ? "btn btn-primary" : "btn btn-outline"}
                style={{ textTransform: "capitalize" }}>{t}</button>
            ))}
          </div>
          <h3 style={{ fontSize: 16, fontWeight: 700, marginTop: 28, marginBottom: 16 }}>Language</h3>
          <input className="input" style={{ maxWidth: 300 }} defaultValue={preferences.language}
            onBlur={(e) => setPreferences({ language: e.target.value })} />
        </div>
      )}

      {tab === "general" && (
        <div className="fade-in" style={{ maxWidth: 500 }}>
          <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>Fallback Provider</h3>
          <label style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12, cursor: "pointer" }}>
            <input type="checkbox" checked={preferences.fallbackEnabled}
              onChange={(e) => setPreferences({ fallbackEnabled: e.target.checked })} style={{ width: 18, height: 18 }} />
            <span style={{ fontSize: 14 }}>Enable fallback to secondary provider on failure</span>
          </label>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 12 }}>
            When the primary provider fails with a temporary error, Edify AI will try your fallback provider.
          </div>
          <h3 style={{ fontSize: 16, fontWeight: 700, marginTop: 28, marginBottom: 16 }}>Storage</h3>
          <button className="btn btn-outline" onClick={() => { if (confirm("Export all data? This will download a JSON file.")) {} }}>
            <Database size={16} /> Export Data
          </button>
          <div style={{ marginTop: 8 }}>
            <button className="btn btn-danger" onClick={() => { if (confirm("Delete ALL local data? This cannot be undone.")) { indexedDB.deleteDatabase("edifyai"); location.reload(); } }}>
              <Trash2 size={16} /> Delete Local Data
            </button>
          </div>
        </div>
      )}

      {tab === "about" && (
        <div className="fade-in" style={{ maxWidth: 500 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 20 }}>
            <div style={{ width: 56, height: 56, borderRadius: 14, background: "linear-gradient(135deg, #6366f1, #8b5cf6)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Info size={28} color="white" />
            </div>
            <div>
              <h2 style={{ fontSize: 22, fontWeight: 800 }}>Edify AI</h2>
              <p style={{ fontSize: 14, color: "var(--text-muted)" }}>Turn Knowledge Into Understanding.</p>
            </div>
          </div>
          <div style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 8 }}>Version 1.0.0</div>
          <div style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 20 }}>Built for students, learners, and educators.</div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button className="btn btn-outline" onClick={() => window.open("https://github.com/arcange9/EdifyAI", "_blank")}>GitHub</button>
            <button className="btn btn-outline" onClick={() => window.open("https://github.com/arcange9/EdifyAI#readme", "_blank")}>Documentation</button>
            <button className="btn btn-outline" onClick={() => window.open("https://github.com/arcange9/EdifyAI/issues", "_blank")}>Feedback</button>
          </div>
          <div style={{ marginTop: 24, padding: 16, borderRadius: 10, background: "var(--bg-elevated)", border: "1px solid var(--border)" }}>
            <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 10 }}>Study Books</h3>
            <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 12 }}>Free textbooks by the same author — download as DOCX:</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <button className="btn btn-outline" style={{ textAlign: "left", justifyContent: "flex-start" }} onClick={() => window.open("https://github.com/arcange9/python-mastery-notes/raw/main/docx/Python_Mastery_Notes_Complete_Book.docx", "_blank")}>
                Python Mastery Notes — Complete Book
              </button>
              <button className="btn btn-outline" style={{ textAlign: "left", justifyContent: "flex-start" }} onClick={() => window.open("https://github.com/arcange9/ai-engineering-mastery/raw/main/docx/AI_Engineering_Mastery_Complete_Book.docx", "_blank")}>
                AI Engineering Mastery — Complete Book
              </button>
              <button className="btn btn-outline" style={{ textAlign: "left", justifyContent: "flex-start" }} onClick={() => window.open("https://github.com/arcange9/python-mastery-notes", "_blank")}>
                Python Notes (individual chapters)
              </button>
              <button className="btn btn-outline" style={{ textAlign: "left", justifyContent: "flex-start" }} onClick={() => window.open("https://github.com/arcange9/ai-engineering-mastery", "_blank")}>
                AI Engineering (individual chapters)
              </button>
              <button className="btn btn-outline" style={{ textAlign: "left", justifyContent: "flex-start" }} onClick={() => window.open("https://github.com/arcange9/ethical-hacking-book/raw/main/docx/Ethical_Hacking_Complete_Book.docx", "_blank")}>
                Ethical Hacking — Complete Book
              </button>
              <button className="btn btn-outline" style={{ textAlign: "left", justifyContent: "flex-start" }} onClick={() => window.open("https://github.com/arcange9/ethical-hacking-book", "_blank")}>
                Ethical Hacking (individual chapters)
              </button>
            </div>
          </div>
          <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 24, textAlign: "center" }}>
            Designed by Mukamyi Izere Arcange
          </p>
          <p style={{ fontSize: 12, color: "var(--text-faint)", marginTop: 8, textAlign: "center" }}>
            Edify AI is an independent project and is not affiliated with NitroAI.
          </p>
        </div>
      )}
    </div>
  );
}

function ProvidersTab({ providers, preferences, saveProviderConfig, deleteProviderConfig, setActiveProvider }: {
  providers: ProviderConfig[];
  preferences: { defaultProviderId?: string; fallbackProviderId?: string; fallbackEnabled: boolean };
  saveProviderConfig: (config: ProviderConfig, apiKey?: string) => Promise<void>;
  deleteProviderConfig: (id: string) => Promise<void>;
  setActiveProvider: (id: string) => Promise<void>;
}) {
  const [editing, setEditing] = useState<ProviderConfig | null>(null);

  return (
    <div className="fade-in">
      <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>AI Providers</h3>
      <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 20 }}>
        Configure how Edify AI powers your learning. API keys are stored securely on your device.
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: 12, maxWidth: 600 }}>
        {PROVIDER_TYPES.map((type) => {
          const configs = providers.filter((p) => p.type === type);
          const configured = configs.length > 0;
          return (
            <div key={type} className="card" style={{ padding: 18 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 15 }}>{providerDisplayName(type)}</div>
                  <div style={{ fontSize: 13, color: "var(--text-muted)" }}>{providerDescription(type)}</div>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: configured ? "var(--success)" : "var(--text-faint)" }}>
                    {configured ? "✓ Configured" : "Not configured"}
                  </span>
                  <button className="btn btn-outline" style={{ padding: "6px 14px", fontSize: 13 }}
                    onClick={() => {
                      const c = configs[0];
                      if (c) setEditing(c);
                      else {
                        const nc: ProviderConfig = { id: uuid(), type, name: providerDisplayName(type), apiKey: "", baseUrl: defaultBaseUrl(type), model: "", enabled: true, createdAt: Date.now() };
                        setEditing(nc);
                      }
                    }}>
                    {configured ? "Manage" : "Configure"}
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {providers.length > 0 && (
        <div style={{ marginTop: 24, maxWidth: 600 }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>Default Provider</h3>
          <select className="input" style={{ maxWidth: 300 }} value={preferences.defaultProviderId ?? ""}
            onChange={(e) => setActiveProvider(e.target.value)}>
            <option value="">— Select —</option>
            {providers.filter((p) => p.enabled).map((p) => (
              <option key={p.id} value={p.id}>{p.name} / {p.model || "auto"}</option>
            ))}
          </select>
        </div>
      )}

      {editing && (
        <ProviderEditor
          config={editing}
          onClose={() => setEditing(null)}
          onSave={async (config, key) => { await saveProviderConfig(config, key); setEditing(null); }}
          onDelete={async () => { await deleteProviderConfig(editing.id); setEditing(null); }}
        />
      )}
    </div>
  );
}

function ProviderEditor({ config, onClose, onSave, onDelete }: {
  config: ProviderConfig;
  onClose: () => void;
  onSave: (config: ProviderConfig, apiKey?: string) => Promise<void>;
  onDelete: () => Promise<void>;
}) {
  const [name, setName] = useState(config.name);
  const [apiKey, setApiKey] = useState(config.apiKey);
  const [showKey, setShowKey] = useState(false);
  const [baseUrl, setBaseUrl] = useState(config.baseUrl ?? "");
  const [model, setModel] = useState(config.model);
  const [fallbackModel, setFallbackModel] = useState(config.fallbackModel ?? "");
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);

  async function handleTest() {
    setTesting(true);
    setTestResult(null);
    try {
      const updated: ProviderConfig = { ...config, name, apiKey, baseUrl, model, enabled: true };
      const { createProvider } = await import("../ai/providers/manager");
      const provider = await createProvider(updated);
      if (provider) {
        const health = await provider.healthCheck();
        setTestResult({ ok: health.ok, message: health.message });
      } else {
        setTestResult({ ok: false, message: "Could not create provider." });
      }
    } catch (err) {
      setTestResult({ ok: false, message: err instanceof Error ? err.message : "Failed" });
    }
    setTesting(false);
  }

  async function handleSave() {
    const updated: ProviderConfig = {
      ...config, name, apiKey,
      baseUrl: baseUrl || defaultBaseUrl(config.type),
      model, fallbackModel: fallbackModel || undefined,
      enabled: true,
    };
    await onSave(updated, apiKey);
  }

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000 }} onClick={onClose}>
      <div className="card fade-in" style={{ padding: 28, width: 480, maxWidth: "95vw", maxHeight: "90vh", overflowY: "auto" }} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 20 }}>{providerDisplayName(config.type)}</h3>
        {config.type === "custom" && (
          <div style={{ marginBottom: 12 }}>
            <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Provider Name</label>
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
        )}
        <div style={{ marginBottom: 12 }}>
          <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>API Key</label>
          <div style={{ position: "relative" }}>
            <input className="input" type={showKey ? "text" : "password"} placeholder="••••••••••••" value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
            <button onClick={() => setShowKey(!showKey)} style={{ position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer" }}>
              {showKey ? <EyeOff size={16} color="var(--text-muted)" /> : <Eye size={16} color="var(--text-muted)" />}
            </button>
          </div>
          <div style={{ fontSize: 11, color: "var(--text-faint)", marginTop: 4 }}>Your API key is stored locally on this device.</div>
        </div>
        {(config.type === "openrouter" || config.type === "groq" || config.type === "custom") && (
          <div style={{ marginBottom: 12 }}>
            <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Base URL</label>
            <input className="input" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder={defaultBaseUrl(config.type)} />
          </div>
        )}
        <div style={{ marginBottom: 12 }}>
          <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Default Model</label>
          <input className="input" value={model} onChange={(e) => setModel(e.target.value)} placeholder="Auto-select" />
        </div>
        <div style={{ marginBottom: 20 }}>
          <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Fallback Model (optional)</label>
          <input className="input" value={fallbackModel} onChange={(e) => setFallbackModel(e.target.value)} placeholder="Used if primary fails" />
        </div>
        {testResult && (
          <div style={{ padding: 12, borderRadius: 10, marginBottom: 16, background: testResult.ok ? "var(--success-light)" : "var(--danger-light)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              {testResult.ok ? <Check size={18} color="var(--success)" /> : <X size={18} color="var(--danger)" />}
              <span style={{ fontSize: 13, fontWeight: 600, color: testResult.ok ? "var(--success)" : "var(--danger)" }}>{testResult.message}</span>
            </div>
          </div>
        )}
        <div style={{ display: "flex", gap: 8, justifyContent: "space-between" }}>
          <div>
            <button className="btn btn-outline" onClick={handleTest} disabled={testing || !apiKey}>{testing ? "Testing…" : "Test Connection"}</button>
            {config.type === "custom" && <button className="btn btn-danger" style={{ marginLeft: 8 }} onClick={onDelete}><Trash2 size={14} /></button>}
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn btn-ghost" onClick={onClose}>Cancel</button>
            <button className="btn btn-primary" onClick={handleSave} disabled={!apiKey}>Save</button>
          </div>
        </div>
      </div>
    </div>
  );
}
