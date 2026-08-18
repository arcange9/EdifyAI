import { describe, it, expect } from "vitest";
import { providerDisplayName, providerDescription, defaultBaseUrl } from "../src/ai/providers/manager";
import type { ProviderType } from "../src/lib/types";

describe("providerDisplayName", () => {
  it("returns correct display names", () => {
    expect(providerDisplayName("openrouter")).toBe("OpenRouter");
    expect(providerDisplayName("google")).toBe("Google AI Studio");
    expect(providerDisplayName("groq")).toBe("Groq");
    expect(providerDisplayName("custom")).toBe("Custom Provider");
  });
});

describe("providerDescription", () => {
  it("returns non-empty descriptions", () => {
    const types: ProviderType[] = ["openrouter", "google", "groq", "custom"];
    for (const t of types) {
      expect(providerDescription(t).length).toBeGreaterThan(0);
    }
  });
});

describe("defaultBaseUrl", () => {
  it("returns correct URLs", () => {
    expect(defaultBaseUrl("openrouter")).toBe("https://openrouter.ai/api/v1");
    expect(defaultBaseUrl("google")).toContain("googleapis.com");
    expect(defaultBaseUrl("groq")).toContain("groq.com");
    expect(defaultBaseUrl("custom")).toBe("");
  });
});
