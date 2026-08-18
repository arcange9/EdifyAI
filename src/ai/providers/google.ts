/* Edify AI — Google AI Studio Provider
 * Uses Google's Generative Language API (Gemini models).
 */

import type { AIProvider, CompletionOptions, StructuredOptions, TokenHandler } from "./interface";
import { ProviderError } from "./interface";
import type { ModelInfo, ProviderHealth } from "../../lib/types";

const BASE_URL = "https://generativelanguage.googleapis.com/v1beta";

export class GoogleAIStudioProvider implements AIProvider {
  readonly type = "google";
  readonly name: string;

  constructor(
    private apiKey: string,
    private model: string,
    name = "Google AI Studio",
  ) {
    this.name = name;
  }

  capabilities(): string[] {
    return ["text", "streaming", "structured_output", "vision", "embeddings"];
  }

  async chat(opts: CompletionOptions, onToken?: TokenHandler): Promise<string> {
    return this.streamChat(opts, onToken ?? (() => {}));
  }

  async streamChat(opts: CompletionOptions, onToken: TokenHandler): Promise<string> {
    const contents = this.buildContents(opts);
    let res: Response;
    try {
      res = await fetch(`${BASE_URL}/models/${this.model}:streamGenerateContent?key=${this.apiKey}&alt=sse`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents,
          systemInstruction: opts.system ? { parts: [{ text: opts.system }] } : undefined,
          generationConfig: {
            temperature: opts.temperature ?? 0.7,
            maxOutputTokens: opts.maxTokens ?? 4096,
          },
        }),
        signal: opts.signal,
      });
    } catch (err) {
      throw toNetworkError(err);
    }
    if (!res.ok) throw await mapError(res);
    if (!res.body) throw new ProviderError("Empty response.", "unknown");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let full = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        try {
          const json = JSON.parse(trimmed.slice(5).trim());
          const text = json.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
          if (text) { full += text; onToken(text); }
        } catch { /* skip */ }
      }
    }
    return full;
  }

  async generateText(opts: CompletionOptions): Promise<string> {
    const contents = this.buildContents(opts);
    let res: Response;
    try {
      res = await fetch(`${BASE_URL}/models/${this.model}:generateContent?key=${this.apiKey}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents,
          systemInstruction: opts.system ? { parts: [{ text: opts.system }] } : undefined,
          generationConfig: { temperature: opts.temperature ?? 0.7, maxOutputTokens: opts.maxTokens ?? 4096 },
        }),
        signal: opts.signal,
      });
    } catch (err) { throw toNetworkError(err); }
    if (!res.ok) throw await mapError(res);
    const json = await res.json();
    return json.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  }

  async generateStructured<T>(opts: StructuredOptions<T>): Promise<T> {
    const contents = this.buildContents(opts);
    let res: Response;
    try {
      res = await fetch(`${BASE_URL}/models/${this.model}:generateContent?key=${this.apiKey}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents,
          systemInstruction: opts.system ? { parts: [{ text: opts.system }] } : undefined,
          generationConfig: {
            temperature: opts.temperature ?? 0.3,
            maxOutputTokens: opts.maxTokens ?? 4096,
            responseMimeType: "application/json",
            responseSchema: opts.schema,
          },
        }),
        signal: opts.signal,
      });
    } catch (err) { throw toNetworkError(err); }
    if (!res.ok) throw await mapError(res);
    const json = await res.json();
    const text = json.candidates?.[0]?.content?.parts?.[0]?.text ?? "{}";
    return JSON.parse(text) as T;
  }

  async summarize(text: string, style: "short" | "detailed"): Promise<string> {
    const prompt = style === "short" ? "Summarize in 2-3 sentences." : "Provide a detailed summary with all major points.";
    return this.generateText({ system: "You are an expert summarizer.", messages: [{ role: "user", content: `${prompt}\n\n${text}` }], temperature: 0.3 });
  }

  async generateNotes(text: string, language: string): Promise<string> {
    return this.generateText({
      system: `You are an expert study-note writer. Create well-structured Markdown notes. Use headings, bullet lists, **bold** key terms, tables, KaTeX math. End with "## Key Takeaways". Write in ${language}.`,
      messages: [{ role: "user", content: `Source material:\n\n${text}` }],
      temperature: 0.4, maxTokens: 8192,
    });
  }

  async generateFlashcards(text: string, topics: string[]): Promise<{ front: string; back: string; topic: string }[]> {
    const schema = { type: "object", properties: { cards: { type: "array", items: { type: "object", properties: { front: { type: "string" }, back: { type: "string" }, topic: { type: "string" } }, required: ["front", "back", "topic"] } } }, required: ["cards"] };
    const result = await this.generateStructured<{ cards: { front: string; back: string; topic: string }[] }>({
      system: `Create flashcards. One concept per card. Tag with topics: ${topics.join(", ")}.`,
      messages: [{ role: "user", content: text }], schema, temperature: 0.4,
    });
    return result.cards;
  }

  async generateQuiz(text: string, opts: { count: number; difficulty: string; types: string[] }): Promise<unknown[]> {
    const schema = { type: "object", properties: { questions: { type: "array", items: { type: "object", properties: { type: { type: "string" }, topic: { type: "string" }, question: { type: "string" }, options: { type: "array", items: { type: "string" } }, correctIndex: { type: "number" }, explanation: { type: "string" } }, required: ["type", "topic", "question", "options", "correctIndex", "explanation"] } } }, required: ["questions"] };
    const result = await this.generateStructured<{ questions: unknown[] }>({
      system: `Create ${opts.count} ${opts.difficulty} questions. Types: ${opts.types.join(", ")}. Include explanations.`,
      messages: [{ role: "user", content: text }], schema, temperature: 0.4,
    });
    return result.questions;
  }

  async explainConcept(text: string, level: "beginner" | "intermediate" | "advanced"): Promise<string> {
    const desc = level === "beginner" ? "simply, with analogies" : level === "intermediate" ? "at an undergraduate level" : "at an expert level";
    return this.generateText({ system: `You are an AI tutor. Explain ${desc}.`, messages: [{ role: "user", content: text }], temperature: 0.5 });
  }

  async generateStudyPlan(opts: { subject: string; goal: string; days: number; minutesPerDay: number; difficulty: string }): Promise<unknown> {
    const schema = { type: "object", properties: { tasks: { type: "array", items: { type: "object", properties: { day: { type: "number" }, topic: { type: "string" }, objectives: { type: "array", items: { type: "string" } }, reading: { type: "string" }, practice: { type: "string" } }, required: ["day", "topic", "objectives"] } } }, required: ["tasks"] };
    const result = await this.generateStructured<{ tasks: unknown[] }>({
      system: `Create a ${opts.days}-day study plan for "${opts.subject}". ${opts.minutesPerDay} min/day. Difficulty: ${opts.difficulty}.`,
      messages: [{ role: "user", content: `Goal: ${opts.goal}` }], schema, temperature: 0.4,
    });
    return result.tasks;
  }

  async healthCheck(): Promise<ProviderHealth> {
    const start = Date.now();
    try {
      const res = await fetch(`${BASE_URL}/models/${this.model}:generateContent?key=${this.apiKey}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ contents: [{ parts: [{ text: "hi" }] }], generationConfig: { maxOutputTokens: 1 } }),
        signal: AbortSignal.timeout(10000),
      });
      if (res.status === 400 || res.status === 403) return { ok: false, message: "Invalid API key or model.", latencyMs: Date.now() - start };
      if (res.status === 429) return { ok: false, message: "Rate limit reached.", latencyMs: Date.now() - start };
      if (!res.ok) return { ok: false, message: `HTTP ${res.status}`, latencyMs: Date.now() - start };
      return { ok: true, message: "Connection successful", latencyMs: Date.now() - start };
    } catch (err) {
      return { ok: false, message: err instanceof Error ? err.message : "Failed", latencyMs: Date.now() - start };
    }
  }

  async listModels(): Promise<ModelInfo[]> {
    try {
      const res = await fetch(`${BASE_URL}/models?key=${this.apiKey}`);
      if (!res.ok) return [];
      const json = await res.json();
      return (json.models ?? []).map((m: Record<string, unknown>) => ({
        id: m.name as string,
        name: m.displayName as string,
        provider: "google",
        contextLength: m.inputTokenLimit as number | undefined,
        capabilities: ["text", "streaming", "structured_output"],
        description: m.description as string | undefined,
      }));
    } catch { return []; }
  }

  private buildContents(opts: CompletionOptions): Array<{ role: string; parts: Array<{ text: string }> }> {
    const contents: Array<{ role: string; parts: Array<{ text: string }> }> = [];
    if (opts.context) {
      contents.push({ role: "user", parts: [{ text: `Context:\n${opts.context}` }] });
      contents.push({ role: "model", parts: [{ text: "I'll use this context." }] });
    }
    for (const m of opts.messages) {
      contents.push({ role: m.role === "assistant" ? "model" : "user", parts: [{ text: m.content }] });
    }
    return contents;
  }
}

function toNetworkError(err: unknown): ProviderError {
  if (err instanceof Error && err.name === "AbortError") throw err;
  return new ProviderError(err instanceof Error ? err.message : "Network failed.", "network");
}

async function mapError(res: Response): Promise<ProviderError> {
  let message = res.statusText || "Request failed.";
  try {
    const body = await res.json();
    if (body?.error?.message) message = body.error.message;
  } catch { /* not JSON */ }
  if (res.status === 400 || res.status === 403) return new ProviderError(message, "auth");
  if (res.status === 429) return new ProviderError(message, "rate_limit");
  if (res.status === 404) return new ProviderError(message, "model_not_found");
  return new ProviderError(message, "unknown");
}
