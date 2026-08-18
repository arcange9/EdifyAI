/* Edify AI — Groq Provider
 * Uses Groq's OpenAI-compatible API for fast inference.
 */

import type { AIProvider, CompletionOptions, StructuredOptions, TokenHandler, ChatMessageInput } from "./interface";
import { ProviderError } from "./interface";
import type { ModelInfo, ProviderHealth } from "../../lib/types";

const BASE_URL = "https://api.groq.com/openai/v1";

export class GroqProvider implements AIProvider {
  readonly type = "groq";
  readonly name: string;

  constructor(
    private apiKey: string,
    private model: string,
    name = "Groq",
    private baseUrl: string = BASE_URL,
  ) {
    this.name = name;
  }

  capabilities(): string[] {
    return ["text", "streaming", "structured_output", "tool_calling"];
  }

  async chat(opts: CompletionOptions, onToken?: TokenHandler): Promise<string> {
    return this.streamChat(opts, onToken ?? (() => {}));
  }

  async streamChat(opts: CompletionOptions, onToken: TokenHandler): Promise<string> {
    const messages = this.buildMessages(opts);
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/chat/completions`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({ model: this.model, messages, stream: true, temperature: opts.temperature ?? 0.7, max_tokens: opts.maxTokens ?? 4096 }),
        signal: opts.signal,
      });
    } catch (err) { throw toNetworkError(err); }
    if (!res.ok) throw await mapError(res);
    if (!res.body) throw new ProviderError("Empty response.", "unknown");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "", full = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        if (trimmed.slice(5).trim() === "[DONE]") continue;
        try {
          const json = JSON.parse(trimmed.slice(5).trim());
          const text = json.choices?.[0]?.delta?.content ?? "";
          if (text) { full += text; onToken(text); }
        } catch { /* skip */ }
      }
    }
    return full;
  }

  async generateText(opts: CompletionOptions): Promise<string> {
    return this.chat(opts);
  }

  async generateStructured<T>(opts: StructuredOptions<T>): Promise<T> {
    const messages = this.buildMessages(opts);
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/chat/completions`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({ model: this.model, messages, temperature: opts.temperature ?? 0.3, max_tokens: opts.maxTokens ?? 4096, response_format: { type: "json_object" } }),
        signal: opts.signal,
      });
    } catch (err) { throw toNetworkError(err); }
    if (!res.ok) throw await mapError(res);
    const json = await res.json();
    const content = json.choices?.[0]?.message?.content ?? "{}";
    return JSON.parse(content) as T;
  }

  async summarize(text: string, style: "short" | "detailed"): Promise<string> {
    const p = style === "short" ? "Summarize in 2-3 sentences." : "Detailed summary with all major points.";
    return this.chat({ system: "Expert summarizer.", messages: [{ role: "user", content: `${p}\n\n${text}` }], temperature: 0.3 });
  }

  async generateNotes(text: string, language: string): Promise<string> {
    return this.chat({ system: `Expert study-note writer. Markdown notes with headings, bullets, **bold** terms, tables, KaTeX. End with "## Key Takeaways". Write in ${language}.`, messages: [{ role: "user", content: `Source:\n\n${text}` }], temperature: 0.4, maxTokens: 8192 });
  }

  async generateFlashcards(text: string, topics: string[]): Promise<{ front: string; back: string; topic: string }[]> {
    const schema = { type: "object", properties: { cards: { type: "array", items: { type: "object", properties: { front: { type: "string" }, back: { type: "string" }, topic: { type: "string" } }, required: ["front", "back", "topic"] } } }, required: ["cards"] };
    const result = await this.generateStructured<{ cards: { front: string; back: string; topic: string }[] }>({ system: `Create flashcards. Topics: ${topics.join(", ")}.`, messages: [{ role: "user", content: text }], schema, temperature: 0.4 });
    return result.cards;
  }

  async generateQuiz(text: string, opts: { count: number; difficulty: string; types: string[] }): Promise<unknown[]> {
    const schema = { type: "object", properties: { questions: { type: "array", items: { type: "object", properties: { type: { type: "string" }, topic: { type: "string" }, question: { type: "string" }, options: { type: "array", items: { type: "string" } }, correctIndex: { type: "number" }, explanation: { type: "string" } }, required: ["type", "topic", "question", "options", "correctIndex", "explanation"] } } }, required: ["questions"] };
    const result = await this.generateStructured<{ questions: unknown[] }>({ system: `Create ${opts.count} ${opts.difficulty} questions. Types: ${opts.types.join(", ")}.`, messages: [{ role: "user", content: text }], schema, temperature: 0.4 });
    return result.questions;
  }

  async explainConcept(text: string, level: "beginner" | "intermediate" | "advanced"): Promise<string> {
    const d = level === "beginner" ? "simply with analogies" : level === "intermediate" ? "undergraduate level" : "expert level";
    return this.chat({ system: `AI tutor. Explain ${d}.`, messages: [{ role: "user", content: text }], temperature: 0.5 });
  }

  async generateStudyPlan(opts: { subject: string; goal: string; days: number; minutesPerDay: number; difficulty: string }): Promise<unknown> {
    const schema = { type: "object", properties: { tasks: { type: "array", items: { type: "object", properties: { day: { type: "number" }, topic: { type: "string" }, objectives: { type: "array", items: { type: "string" } }, reading: { type: "string" }, practice: { type: "string" } }, required: ["day", "topic", "objectives"] } } }, required: ["tasks"] };
    const result = await this.generateStructured<{ tasks: unknown[] }>({ system: `${opts.days}-day plan for "${opts.subject}". ${opts.minutesPerDay} min/day. Difficulty: ${opts.difficulty}.`, messages: [{ role: "user", content: `Goal: ${opts.goal}` }], schema, temperature: 0.4 });
    return result.tasks;
  }

  async healthCheck(): Promise<ProviderHealth> {
    const start = Date.now();
    try {
      const res = await fetch(`${this.baseUrl}/chat/completions`, {
        method: "POST", headers: this.headers(),
        body: JSON.stringify({ model: this.model, messages: [{ role: "user", content: "hi" }], max_tokens: 1 }),
        signal: AbortSignal.timeout(10000),
      });
      if (res.status === 401) return { ok: false, message: "Invalid API key.", latencyMs: Date.now() - start };
      if (res.status === 429) return { ok: false, message: "Rate limit reached.", latencyMs: Date.now() - start };
      if (!res.ok) { const b = await res.json().catch(() => null); return { ok: false, message: b?.error?.message ?? `HTTP ${res.status}`, latencyMs: Date.now() - start }; }
      return { ok: true, message: "Connection successful", latencyMs: Date.now() - start };
    } catch (err) { return { ok: false, message: err instanceof Error ? err.message : "Failed", latencyMs: Date.now() - start }; }
  }

  async listModels(): Promise<ModelInfo[]> {
    try {
      const res = await fetch(`${this.baseUrl}/models`, { headers: this.headers() });
      if (!res.ok) return [];
      const json = await res.json();
      return (json.data ?? []).map((m: Record<string, unknown>) => ({
        id: m.id as string, name: m.id as string, provider: "groq",
        contextLength: m.context_window as number | undefined,
        capabilities: ["text", "streaming", "structured_output"],
      }));
    } catch { return []; }
  }

  private headers(): Record<string, string> {
    return { "Content-Type": "application/json", Authorization: `Bearer ${this.apiKey}` };
  }

  private buildMessages(opts: CompletionOptions): ChatMessageInput[] {
    const msgs: ChatMessageInput[] = [];
    if (opts.system) msgs.push({ role: "system", content: opts.system });
    if (opts.context) msgs.push({ role: "system", content: `Context:\n${opts.context}` });
    msgs.push(...opts.messages);
    return msgs;
  }
}

function toNetworkError(err: unknown): ProviderError {
  if (err instanceof Error && err.name === "AbortError") throw err;
  return new ProviderError(err instanceof Error ? err.message : "Network failed.", "network");
}

async function mapError(res: Response): Promise<ProviderError> {
  let msg = res.statusText || "Request failed.";
  try { const b = await res.json(); if (b?.error?.message) msg = b.error.message; } catch {}
  if (res.status === 401) return new ProviderError(msg, "auth");
  if (res.status === 429) return new ProviderError(msg, "rate_limit");
  if (res.status === 404) return new ProviderError(msg, "model_not_found");
  return new ProviderError(msg, "unknown");
}
