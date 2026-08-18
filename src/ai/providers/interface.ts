/* Edify AI — Unified AI Provider Interface.
 * All providers implement this interface. The rest of the app
 * never depends on provider-specific code.
 */

import type { ModelInfo, ProviderHealth, Citation } from "../../lib/types";

export interface ChatMessageInput {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface CompletionOptions {
  system?: string;
  messages: ChatMessageInput[];
  temperature?: number;
  maxTokens?: number;
  signal?: AbortSignal;
  context?: string; // RAG context to inject
  citations?: Citation[];
}

export type TokenHandler = (token: string) => void;

export interface StructuredOptions<_T> extends CompletionOptions {
  schema: Record<string, unknown>;
  schemaName?: string;
}

export interface AIProvider {
  readonly type: string;
  readonly name: string;

  // Core methods
  chat(opts: CompletionOptions, onToken?: TokenHandler): Promise<string>;
  streamChat(opts: CompletionOptions, onToken: TokenHandler): Promise<string>;
  generateText(opts: CompletionOptions): Promise<string>;
  generateStructured<T>(opts: StructuredOptions<T>): Promise<T>;

  // Study-specific methods
  summarize(text: string, style: "short" | "detailed"): Promise<string>;
  generateNotes(text: string, language: string): Promise<string>;
  generateFlashcards(text: string, topics: string[]): Promise<{ front: string; back: string; topic: string }[]>;
  generateQuiz(text: string, opts: { count: number; difficulty: string; types: string[] }): Promise<unknown[]>;
  explainConcept(text: string, level: "beginner" | "intermediate" | "advanced"): Promise<string>;
  generateStudyPlan(opts: { subject: string; goal: string; days: number; minutesPerDay: number; difficulty: string }): Promise<unknown>;

  // Provider management
  healthCheck(): Promise<ProviderHealth>;
  listModels(): Promise<ModelInfo[]>;

  // Capabilities
  capabilities(): string[];
}

export class ProviderError extends Error {
  constructor(
    message: string,
    public kind: "auth" | "rate_limit" | "network" | "model_not_found" | "quota" | "context_too_long" | "unsupported" | "unknown",
  ) {
    super(message);
    this.name = "ProviderError";
  }
}

export function userFriendlyError(err: ProviderError): string {
  switch (err.kind) {
    case "auth":
      return "Invalid API key. Check your provider settings and try again.";
    case "rate_limit":
      return "Rate limit reached. Wait a moment and try again.";
    case "network":
      return `Edify AI couldn't contact the provider. Check your internet connection.`;
    case "model_not_found":
      return "The selected model is no longer available. Choose a different model in Settings.";
    case "quota":
      return "Your API quota has been exceeded. Check your provider account.";
    case "context_too_long":
      return "This document is too large for the selected model's context window. Try a model with a larger context or import a smaller file.";
    case "unsupported":
      return err.message;
    default:
      return err.message;
  }
}
