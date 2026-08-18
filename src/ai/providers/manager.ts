/* Edify AI — Provider Manager
 * Creates AIProvider instances from stored configs, manages credentials
 * via Electron safeStorage (or localStorage fallback), and handles
 * fallback provider logic.
 */

import type { AIProvider } from "./interface";
import { OpenRouterProvider } from "./openrouter";
import { GoogleAIStudioProvider } from "./google";
import { GroqProvider } from "./groq";
import { CustomProvider } from "./custom";
import type { ProviderConfig, ProviderType } from "../../lib/types";

export async function createProvider(config: ProviderConfig): Promise<AIProvider | null> {
  if (!config.apiKey || !config.enabled) return null;
  switch (config.type) {
    case "openrouter":
      return new OpenRouterProvider(config.apiKey, config.model, config.name, config.baseUrl);
    case "google":
      return new GoogleAIStudioProvider(config.apiKey, config.model, config.name);
    case "groq":
      return new GroqProvider(config.apiKey, config.model, config.name, config.baseUrl);
    case "custom":
      return new CustomProvider(config.apiKey, config.model, config.name, config.baseUrl ?? "");
    default:
      return null;
  }
}

export function defaultBaseUrl(type: ProviderType): string {
  switch (type) {
    case "openrouter": return "https://openrouter.ai/api/v1";
    case "google": return "https://generativelanguage.googleapis.com/v1beta";
    case "groq": return "https://api.groq.com/openai/v1";
    case "custom": return "";
  }
}

export function providerDisplayName(type: ProviderType): string {
  switch (type) {
    case "openrouter": return "OpenRouter";
    case "google": return "Google AI Studio";
    case "groq": return "Groq";
    case "custom": return "Custom Provider";
  }
}

export function providerDescription(type: ProviderType): string {
  switch (type) {
    case "openrouter": return "Access many AI models through one API.";
    case "google": return "Connect Google's AI models.";
    case "groq": return "Fast AI inference.";
    case "custom": return "Connect an OpenAI-compatible endpoint.";
  }
}
