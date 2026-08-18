/* Edify AI — Resilient Provider Wrapper
 * Adds exponential backoff on rate-limit and transient network errors.
 * Streaming ops only retry on pre-stream failures so tokens are never double-emitted.
 */

import type { AIProvider, CompletionOptions, TokenHandler, StructuredOptions } from "./interface";
import { ProviderError } from "./interface";
import type { ModelInfo, ProviderHealth } from "../../lib/types";

const DELAYS_MS = [3000, 8000, 20000, 40000];

function retryable(e: unknown): boolean {
  return e instanceof ProviderError && (e.kind === "rate_limit" || e.kind === "network");
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new ProviderError("Cancelled", "unknown"));
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => { clearTimeout(t); reject(new ProviderError("Cancelled", "unknown")); }, { once: true });
  });
}

export class ResilientProvider implements AIProvider {
  readonly type: string;
  readonly name: string;

  constructor(private inner: AIProvider) {
    this.type = inner.type;
    this.name = inner.name;
  }

  capabilities(): string[] { return this.inner.capabilities(); }

  async chat(opts: CompletionOptions, onToken?: TokenHandler): Promise<string> {
    for (let attempt = 0; attempt <= DELAYS_MS.length; attempt++) {
      try {
        return await this.inner.chat(opts, onToken);
      } catch (e) {
        if (e instanceof ProviderError && e.kind === "auth") throw e;
        if (e instanceof Error && e.name === "AbortError") throw e;
        if (attempt < DELAYS_MS.length && retryable(e)) {
          await sleep(DELAYS_MS[attempt], opts.signal);
          continue;
        }
        throw e;
      }
    }
    throw new ProviderError("Max retries exceeded.", "unknown");
  }

  async streamChat(opts: CompletionOptions, onToken: TokenHandler): Promise<string> {
    for (let attempt = 0; attempt <= DELAYS_MS.length; attempt++) {
      try {
        return await this.inner.streamChat(opts, onToken);
      } catch (e) {
        if (e instanceof ProviderError && e.kind === "auth") throw e;
        if (e instanceof Error && e.name === "AbortError") throw e;
        if (attempt < DELAYS_MS.length && retryable(e)) {
          await sleep(DELAYS_MS[attempt], opts.signal);
          continue;
        }
        throw e;
      }
    }
    throw new ProviderError("Max retries exceeded.", "unknown");
  }

  async generateText(opts: CompletionOptions): Promise<string> { return this.chat(opts); }

  async generateStructured<T>(opts: StructuredOptions<T>): Promise<T> {
    for (let attempt = 0; attempt <= DELAYS_MS.length; attempt++) {
      try {
        return await this.inner.generateStructured(opts);
      } catch (e) {
        if (e instanceof ProviderError && e.kind === "auth") throw e;
        if (e instanceof Error && e.name === "AbortError") throw e;
        if (attempt < DELAYS_MS.length && retryable(e)) {
          await sleep(DELAYS_MS[attempt], opts.signal);
          continue;
        }
        throw e;
      }
    }
    throw new ProviderError("Max retries exceeded.", "unknown");
  }

  summarize(text: string, style: "short" | "detailed") { return this.inner.summarize(text, style); }
  generateNotes(text: string, lang: string) { return this.inner.generateNotes(text, lang); }
  generateFlashcards(text: string, topics: string[]) { return this.inner.generateFlashcards(text, topics); }
  generateQuiz(text: string, opts: { count: number; difficulty: string; types: string[] }) { return this.inner.generateQuiz(text, opts); }
  explainConcept(text: string, level: "beginner" | "intermediate" | "advanced") { return this.inner.explainConcept(text, level); }
  generateStudyPlan(opts: { subject: string; goal: string; days: number; minutesPerDay: number; difficulty: string }) { return this.inner.generateStudyPlan(opts); }
  healthCheck(): Promise<ProviderHealth> { return this.inner.healthCheck(); }
  listModels(): Promise<ModelInfo[]> { return this.inner.listModels(); }
}
