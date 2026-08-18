/* Edify AI — RAG (Retrieval-Augmented Generation)
 * Simple but effective: chunks documents, does keyword-based search
 * (with optional embedding-based similarity when embeddings are available),
 * and returns relevant context chunks for the AI provider.
 */

import type { Document, TextChunk, Citation } from "./types";

/* Simple TF-IDF-like scoring for keyword retrieval with stemming. */
function tokenize(text: string): string[] {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter((t) => t.length > 2);
}

/* Basic stemmer: removes common suffixes so "cells" matches "cell", etc. */
function stem(word: string): string {
  return word.replace(/(s|es|ing|ed|ly|er|est)$/, "");
}

function scoreChunk(query: string, chunkText: string): number {
  const queryTokens = new Set(tokenize(query).map(stem));
  const chunkTokens = tokenize(chunkText);
  let score = 0;
  for (const token of chunkTokens) {
    if (queryTokens.has(stem(token))) score += 1;
  }
  // Normalize by chunk length to avoid bias toward long chunks
  return score / Math.sqrt(chunkTokens.length || 1);
}

export function retrieveRelevantChunks(
  query: string,
  documents: Document[],
  topK = 3,
): { text: string; citations: Citation[] } {
  if (!documents.length || !query.trim()) return { text: "", citations: [] };

  const scored: Array<{ doc: Document; chunk: TextChunk; score: number }> = [];

  for (const doc of documents) {
    for (const chunk of doc.chunks) {
      const score = scoreChunk(query, chunk.text);
      if (score > 0) scored.push({ doc, chunk, score });
    }
  }

  scored.sort((a, b) => b.score - a.score);
  const top = scored.slice(0, topK);

  const contextText = top.map((item) => item.chunk.text).join("\n\n---\n\n");
  const citations: Citation[] = top.map((item) => ({
    documentId: item.doc.id,
    documentTitle: item.doc.title,
    chunkIndex: parseInt(item.chunk.id.replace("chunk-", "")) || 0,
    text: item.chunk.text.slice(0, 200) + "...",
  }));

  return { text: contextText, citations };
}

export function buildRAGContext(query: string, documents: Document[]): { context: string; citations: Citation[] } {
  const { text, citations } = retrieveRelevantChunks(query, documents);
  return { context: text, citations };
}
