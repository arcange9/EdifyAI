/* Edify AI — OpenRouter Provider
 * First-class OpenRouter integration with dynamic model listing,
 * streaming, and structured output via OpenAI-compatible API.
 */

import type { AIProvider, CompletionOptions, StructuredOptions, TokenHandler, ChatMessageInput } from "./interface";
import { ProviderError } from "./interface";
import type { ModelInfo, ProviderHealth } from "../../lib/types";

const BASE_URL = "https://openrouter.ai/api/v1";

export class OpenRouterProvider implements AIProvider {
  readonly type = "openrouter";
  readonly name: string;

  constructor(
    private apiKey: string,
    private model: string,
    name = "OpenRouter",
    private baseUrl: string = BASE_URL,
  ) {
    this.name = name;
  }

  capabilities(): string[] {
    return ["text", "streaming", "structured_output", "tool_calling", "vision"];
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
        body: JSON.stringify({
          model: this.model,
          messages,
          stream: true,
          temperature: opts.temperature ?? 0.7,
          max_tokens: opts.maxTokens ?? 4096,
        }),
        signal: opts.signal,
      });
    } catch (err) {
      throw toNetworkError(err);
    }

    if (!res.ok) throw await mapError(res);
    if (!res.body) throw new ProviderError("Empty response stream.", "unknown");

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
        const data = trimmed.slice(5).trim();
        if (data === "[DONE]") continue;
        try {
          const json = JSON.parse(data);
          const text = json.choices?.[0]?.delta?.content ?? "";
          if (text) {
            full += text;
            onToken(text);
          }
        } catch { /* skip malformed */ }
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
        body: JSON.stringify({
          model: this.model,
          messages,
          temperature: opts.temperature ?? 0.3,
          max_tokens: opts.maxTokens ?? 4096,
          response_format: { type: "json_schema", json_schema: { name: opts.schemaName ?? "output", schema: opts.schema, strict: true } },
        }),
        signal: opts.signal,
      });
    } catch (err) {
      throw toNetworkError(err);
    }
    if (!res.ok) throw await mapError(res);
    const json = await res.json();
    const content = json.choices?.[0]?.message?.content ?? "";
    try {
      return JSON.parse(content) as T;
    } catch {
      throw new ProviderError("Provider returned invalid JSON.", "unknown");
    }
  }

  async summarize(text: string, style: "short" | "detailed"): Promise<string> {
    const prompt = style === "short"
      ? "Summarize the following in 2-3 sentences. Capture the main idea and key conclusions only."
      : "Provide a detailed summary of the following. Include all major points, definitions, and conclusions. Organize with clear paragraphs.";
    return this.chat({
      system: "You are an expert academic summarizer.",
      messages: [{ role: "user", content: `${prompt}\n\n${text}` }],
      temperature: 0.3,
    });
  }

  async generateNotes(text: string, language: string): Promise<string> {
    return this.chat({
      system: `You are an expert study-note writer. Turn the source material into clear, well-structured study notes in Markdown. Use headings (##, ###), bullet lists, **bold** key terms, tables, and KaTeX math where relevant. Genuinely synthesize — do not merely reorder. End with a "## Key Takeaways" section. Write in ${language}. Output only the Markdown.`,
      messages: [{ role: "user", content: `Source material:\n\n${text}` }],
      temperature: 0.4,
      maxTokens: 8192,
    });
  }

  async generateFlashcards(text: string, topics: string[]): Promise<{ front: string; back: string; topic: string }[]> {
    const schema = {
      type: "object",
      properties: {
        cards: {
          type: "array",
          items: {
            type: "object",
            properties: {
              front: { type: "string" },
              back: { type: "string" },
              topic: { type: "string" },
            },
            required: ["front", "back", "topic"],
          },
        },
      },
      required: ["cards"],
    };
    const result = await this.generateStructured<{ cards: { front: string; back: string; topic: string }[] }>({
      system: `You create study flashcards. Rules: one atomic concept per card; front is a question or term, back is a complete answer. Tag each card with the most relevant topic from: ${topics.join(", ")}. Aim for 2-4 cards per topic.`,
      messages: [{ role: "user", content: `Source material:\n\n${text}` }],
      schema,
      schemaName: "flashcards",
      temperature: 0.4,
    });
    return result.cards;
  }

  async generateQuiz(text: string, opts: { count: number; difficulty: string; types: string[] }): Promise<unknown[]> {
    const schema = {
      type: "object",
      properties: {
        questions: {
          type: "array",
          items: {
            type: "object",
            properties: {
              type: { type: "string", enum: opts.types },
              topic: { type: "string" },
              question: { type: "string" },
              options: { type: "array", items: { type: "string" } },
              correctIndex: { type: "number" },
              explanation: { type: "string" },
            },
            required: ["type", "topic", "question", "options", "correctIndex", "explanation"],
          },
        },
      },
      required: ["questions"],
    };
    const result = await this.generateStructured<{ questions: unknown[] }>({
      system: `You are a quiz generator. Create ${opts.count} ${opts.difficulty} difficulty questions from the source. Use these question types: ${opts.types.join(", ")}. For MCQ, provide 4 options and the correct index. For true_false, provide 2 options ["True","False"]. For short_answer, put the answer in options[0] and correctIndex=0. Always include an explanation.`,
      messages: [{ role: "user", content: `Source material:\n\n${text}` }],
      schema,
      schemaName: "quiz",
      temperature: 0.4,
    });
    return result.questions;
  }

  async explainConcept(text: string, level: "beginner" | "intermediate" | "advanced"): Promise<string> {
    const levelDesc = level === "beginner" ? "as if explaining to someone new to the subject, using simple analogies" :
      level === "intermediate" ? "at an undergraduate level, connecting concepts" :
      "at a graduate/expert level, exploring nuances and edge cases";
    return this.chat({
      system: `You are an expert AI tutor. Explain concepts ${levelDesc}. Use the provided material as context.`,
      messages: [{ role: "user", content: text }],
      temperature: 0.5,
    });
  }

  async generateStudyPlan(opts: { subject: string; goal: string; days: number; minutesPerDay: number; difficulty: string }): Promise<unknown> {
    const schema = {
      type: "object",
      properties: {
        tasks: {
          type: "array",
          items: {
            type: "object",
            properties: {
              day: { type: "number" },
              topic: { type: "string" },
              objectives: { type: "array", items: { type: "string" } },
              reading: { type: "string" },
              practice: { type: "string" },
            },
            required: ["day", "topic", "objectives"],
          },
        },
      },
      required: ["tasks"],
    };
    const result = await this.generateStructured<{ tasks: unknown[] }>({
      system: `You are a study plan generator. Create a ${opts.days}-day study plan for "${opts.subject}". Goal: ${opts.goal}. Difficulty: ${opts.difficulty}. Each day: ${opts.minutesPerDay} minutes. Break into manageable topics with objectives, reading, and practice.`,
      messages: [{ role: "user", content: `Subject: ${opts.subject}\nGoal: ${opts.goal}` }],
      schema,
      schemaName: "study_plan",
      temperature: 0.4,
    });
    return result.tasks;
  }

  async healthCheck(): Promise<ProviderHealth> {
    const start = Date.now();
    try {
      const res = await fetch(`${this.baseUrl}/chat/completions`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({
          model: this.model,
          messages: [{ role: "user", content: "hi" }],
          max_tokens: 1,
        }),
        signal: AbortSignal.timeout(10000),
      });
      if (res.status === 401) return { ok: false, message: "Invalid API key.", latencyMs: Date.now() - start };
      if (res.status === 429) return { ok: false, message: "Rate limit reached.", latencyMs: Date.now() - start };
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        return { ok: false, message: body?.error?.message ?? `HTTP ${res.status}`, latencyMs: Date.now() - start };
      }
      return { ok: true, message: "Connection successful", latencyMs: Date.now() - start };
    } catch (err) {
      return { ok: false, message: err instanceof Error ? err.message : "Connection failed", latencyMs: Date.now() - start };
    }
  }

  async listModels(): Promise<ModelInfo[]> {
    try {
      const res = await fetch(`${this.baseUrl}/models`, { headers: this.headers() });
      if (!res.ok) return [];
      const json = await res.json();
      return (json.data ?? []).map((m: Record<string, unknown>) => ({
        id: m.id as string,
        name: (m.name as string) || (m.id as string),
        provider: (m.id as string).split("/")[0] ?? "unknown",
        contextLength: m.context_length as number | undefined,
        capabilities: this.inferCapabilities(m),
        isFree: ((m as Record<string, unknown>).pricing as Record<string, string> | undefined)?.prompt?.startsWith("0"),
        description: m.description as string | undefined,
      }));
    } catch {
      return [];
    }
  }

  private inferCapabilities(model: Record<string, unknown>): string[] {
    const caps = ["text", "streaming"];
    const id = (model.id as string) ?? "";
    if (/vision|image|multimodal|gpt-4o|claude-3|gemini|llava/i.test(id)) caps.push("vision");
    if (/whisper|audio|voice/i.test(id)) caps.push("audio");
    caps.push("structured_output");
    return caps;
  }

  private headers(): Record<string, string> {
    return {
      "Content-Type": "application/json",
      Authorization: `Bearer ${this.apiKey}`,
      "HTTP-Referer": "https://edifyai.app",
      "X-Title": "Edify AI",
    };
  }

  private buildMessages(opts: CompletionOptions): ChatMessageInput[] {
    const messages: ChatMessageInput[] = [];
    if (opts.system) messages.push({ role: "system", content: opts.system });
    if (opts.context) {
      messages.push({ role: "system", content: `Use the following context to answer questions. Cite sources where possible.\n\n${opts.context}` });
    }
    messages.push(...opts.messages);
    return messages;
  }
}

function toNetworkError(err: unknown): ProviderError {
  if (err instanceof Error && err.name === "AbortError") throw err;
  return new ProviderError(err instanceof Error ? err.message : "Network request failed.", "network");
}

async function mapError(res: Response): Promise<ProviderError> {
  let message = res.statusText || "Request failed.";
  let type: string | undefined;
  try {
    const body = await res.json();
    if (body?.error?.message) message = body.error.message;
    type = body?.error?.type;
  } catch { /* not JSON */ }
  if (res.status === 401) return new ProviderError(message, "auth");
  if (res.status === 429) return new ProviderError(message, "rate_limit");
  if (res.status === 404) return new ProviderError(message, "model_not_found");
  if (type === "quota_exceeded" || type === "billing") return new ProviderError(message, "quota");
  return new ProviderError(message, "unknown");
}
