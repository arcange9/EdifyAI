import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useApp } from "../lib/app-context";
import { v4 as uuid } from "uuid";
import { BookOpen, ArrowRight, ArrowLeft, Check, X } from "lucide-react";
import type { ProviderConfig, ProviderType } from "../lib/types";
import { providerDisplayName, providerDescription, defaultBaseUrl } from "../ai/providers/manager";

type Step = "welcome" | "provider" | "key" | "test" | "model" | "done";

const PROVIDER_TYPES: ProviderType[] = ["openrouter", "google", "groq", "custom"];

export default function Onboarding() {
  const { saveProviderConfig, setActiveProvider, setPreferences } = useApp();
  const navigate = useNavigate();
  const [step, setStep] = useState<Step>("welcome");
  const [selectedType, setSelectedType] = useState<ProviderType | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);
  const [modelName, setModelName] = useState("");
  const [customName, setCustomName] = useState("");
  const [customBaseUrl, setCustomBaseUrl] = useState("");

  async function handleTest() {
    if (!selectedType || !apiKey) return;
    setTesting(true);
    setTestResult(null);
    try {
      const config: ProviderConfig = {
        id: uuid(),
        type: selectedType,
        name: selectedType === "custom" ? (customName || "Custom Provider") : providerDisplayName(selectedType),
        apiKey,
        baseUrl: selectedType === "custom" ? customBaseUrl : defaultBaseUrl(selectedType),
        model: modelName || "",
        enabled: true,
        createdAt: Date.now(),
      };
      // Create a temporary provider to test
      const { createProvider } = await import("../ai/providers/manager");
      const provider = await createProvider(config);
      if (provider) {
        // Set a default model if none specified
        if (!config.model) {
          const defaults: Record<string, string> = {
            openrouter: "openai/gpt-4o-mini",
            google: "gemini-2.0-flash",
            groq: "llama-3.3-70b-versatile",
            custom: "gpt-4o-mini",
          };
          config.model = defaults[selectedType] || "gpt-4o-mini";
          setModelName(config.model);
        }
        const retest = await createProvider(config);
        if (retest) {
          const health = await retest.healthCheck();
          setTestResult({ ok: health.ok, message: health.message });
          if (health.ok) {
            await saveProviderConfig(config, apiKey);
            await setActiveProvider(config.id);
          }
        }
      }
    } catch (err) {
      setTestResult({ ok: false, message: err instanceof Error ? err.message : "Connection failed" });
    }
    setTesting(false);
  }

  async function finish() {
    await setPreferences({ onboarded: true });
    navigate("/");
  }

  const stepOrder: Step[] = ["welcome", "provider", "key", "test", "model", "done"];
  const stepIndex = stepOrder.indexOf(step);

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "var(--bg)", padding: 20 }}>
      <div className="card fade-in" style={{ width: 520, maxWidth: "95vw", padding: 40 }}>
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 32 }}>
          <div style={{ width: 48, height: 48, borderRadius: 12, background: "linear-gradient(135deg, #6366f1, #8b5cf6)", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <BookOpen size={26} color="white" strokeWidth={2.2} />
          </div>
          <div>
            <div style={{ fontWeight: 800, fontSize: 22 }}>Edify AI</div>
            <div style={{ fontSize: 13, color: "var(--text-muted)" }}>Turn Knowledge Into Understanding</div>
          </div>
        </div>

        {/* Progress dots */}
        <div style={{ display: "flex", gap: 6, marginBottom: 28 }}>
          {stepOrder.map((s, i) => (
            <div key={s} style={{ flex: 1, height: 3, borderRadius: 2, background: i <= stepIndex ? "var(--accent)" : "var(--border-strong)" }} />
          ))}
        </div>

        {/* Welcome */}
        {step === "welcome" && (
          <div>
            <h2 style={{ fontSize: 24, fontWeight: 800, marginBottom: 8 }}>Welcome to Edify AI</h2>
            <p style={{ fontSize: 15, color: "var(--text-muted)", marginBottom: 24 }}>
              Your personal AI-powered study workspace. Import documents, generate notes, flashcards, quizzes, and chat with your learning materials.
            </p>
            <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 24 }}>
              To get started, you'll need to configure an AI provider. This powers all AI features — notes generation, flashcards, quizzes, and chat.
            </p>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn btn-ghost" onClick={finish}>Skip for now</button>
              <button className="btn btn-primary" onClick={() => setStep("provider")} style={{ marginLeft: "auto" }}>
                Get Started <ArrowRight size={18} />
              </button>
            </div>
          </div>
        )}

        {/* Provider selection */}
        {step === "provider" && (
          <div>
            <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>Choose AI Provider</h2>
            <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 20 }}>
              How should Edify AI power your learning experience?
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {PROVIDER_TYPES.map((type) => (
                <button
                  key={type}
                  onClick={() => { setSelectedType(type); setStep("key"); }}
                  className="card"
                  style={{ padding: 16, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "space-between", textAlign: "left", border: "1px solid var(--border)" }}
                  onMouseEnter={(e) => e.currentTarget.style.borderColor = "var(--accent)"}
                  onMouseLeave={(e) => e.currentTarget.style.borderColor = "var(--border)"}
                >
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 15 }}>{providerDisplayName(type)}</div>
                    <div style={{ fontSize: 13, color: "var(--text-muted)" }}>{providerDescription(type)}</div>
                  </div>
                  <ArrowRight size={18} color="var(--text-faint)" />
                </button>
              ))}
            </div>
            <button className="btn btn-ghost" onClick={() => setStep("welcome")} style={{ marginTop: 16 }}>
              <ArrowLeft size={16} /> Back
            </button>
          </div>
        )}

        {/* API Key */}
        {step === "key" && selectedType && (
          <div>
            <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>{providerDisplayName(selectedType)}</h2>
            <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 20 }}>
              Enter your API key to connect.
            </p>
            {selectedType === "custom" && (
              <div style={{ marginBottom: 12 }}>
                <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Provider Name</label>
                <input className="input" placeholder="My AI Provider" value={customName} onChange={(e) => setCustomName(e.target.value)} />
              </div>
            )}
            {selectedType === "custom" && (
              <div style={{ marginBottom: 12 }}>
                <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Base URL</label>
                <input className="input" placeholder="https://example.com/v1" value={customBaseUrl} onChange={(e) => setCustomBaseUrl(e.target.value)} />
              </div>
            )}
            <div style={{ marginBottom: 12 }}>
              <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>API Key</label>
              <div style={{ position: "relative" }}>
                <input
                  className="input"
                  type={showKey ? "text" : "password"}
                  placeholder="••••••••••••"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                />
                <button
                  onClick={() => setShowKey(!showKey)}
                  style={{ position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", fontSize: 12, color: "var(--text-muted)" }}
                >
                  {showKey ? "Hide" : "Show"}
                </button>
              </div>
            </div>
            <div style={{ marginBottom: 20 }}>
              <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Model (optional)</label>
              <input className="input" placeholder="Auto-select if empty" value={modelName} onChange={(e) => setModelName(e.target.value)} />
            </div>
            <p style={{ fontSize: 12, color: "var(--text-faint)", marginBottom: 16 }}>
              Your API key is stored locally on this device using secure OS-level storage.
            </p>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn btn-ghost" onClick={() => setStep("provider")}><ArrowLeft size={16} /> Back</button>
              <button className="btn btn-primary" onClick={() => { handleTest(); setStep("test"); }} disabled={!apiKey.trim()} style={{ marginLeft: "auto" }}>
                Test Connection
              </button>
            </div>
          </div>
        )}

        {/* Test result */}
        {step === "test" && (
          <div>
            <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 20 }}>Testing Connection</h2>
            {testing && <p style={{ fontSize: 15, color: "var(--text-muted)" }}>Connecting to {selectedType && providerDisplayName(selectedType)}…</p>}
            {testResult && (
              <div style={{ padding: 16, borderRadius: 10, marginBottom: 20, background: testResult.ok ? "var(--success-light)" : "var(--danger-light)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                  {testResult.ok ? <Check size={20} color="var(--success)" /> : <X size={20} color="var(--danger)" />}
                  <span style={{ fontWeight: 700, color: testResult.ok ? "var(--success)" : "var(--danger)" }}>
                    {testResult.ok ? "Connection successful" : "Connection failed"}
                  </span>
                </div>
                <p style={{ fontSize: 13, color: "var(--text-muted)" }}>{testResult.message}</p>
              </div>
            )}
            {!testing && testResult && (
              <div style={{ display: "flex", gap: 8 }}>
                {!testResult.ok && <button className="btn btn-ghost" onClick={() => setStep("key")}><ArrowLeft size={16} /> Back to edit</button>}
                {testResult.ok && <button className="btn btn-primary" onClick={() => setStep("model")} style={{ marginLeft: "auto" }}>Continue <ArrowRight size={16} /></button>}
                {testResult.ok && <button className="btn btn-outline" onClick={finish}>Skip to finish</button>}
              </div>
            )}
          </div>
        )}

        {/* Model selection */}
        {step === "model" && (
          <div>
            <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>Model Selected</h2>
            <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 20 }}>
              Your default model is set. You can change it anytime in Settings.
            </p>
            <div className="card" style={{ padding: 16, marginBottom: 20 }}>
              <div style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 4 }}>Default Model</div>
              <div style={{ fontSize: 16, fontWeight: 700 }}>{modelName || "Auto-selected"}</div>
            </div>
            <button className="btn btn-primary" onClick={finish} style={{ width: "100%", justifyContent: "center" }}>
              Start Using Edify AI <ArrowRight size={18} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
