import { describe, it, expect } from "vitest";
import { chunkText } from "../src/lib/ingestion";
import { retrieveRelevantChunks, buildRAGContext } from "../src/lib/rag";
import { youtubeId } from "../src/lib/ingestion";
import type { Document } from "../src/lib/types";

describe("chunkText", () => {
  it("chunks text into reasonable pieces", () => {
    const text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five. This is sentence six. This is sentence seven. This is sentence eight. This is sentence nine. This is sentence ten.";
    const chunks = chunkText(text, 100, 20);
    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks[0].id).toBeDefined();
    expect(chunks[0].text.length).toBeGreaterThan(0);
  });

  it("handles empty text", () => {
    const chunks = chunkText("");
    expect(chunks).toEqual([]);
  });

  it("handles short text", () => {
    const chunks = chunkText("Just one sentence.");
    expect(chunks.length).toBe(1);
  });
});

describe("retrieveRelevantChunks", () => {
  const mockDocs: Document[] = [
    {
      id: "doc1",
      projectId: "proj1",
      title: "Biology 101",
      sourceKind: "pdf",
      sourceText: "Cells are the basic unit of life. Mitochondria produce energy.",
      chunks: [
        { id: "chunk-0", text: "Cells are the basic unit of life." },
        { id: "chunk-1", text: "Mitochondria produce energy through ATP synthesis." },
      ],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    },
  ];

  it("returns relevant chunks for a query", () => {
    const { text, citations } = retrieveRelevantChunks("What is a cell?", mockDocs);
    expect(text).toContain("Cells");
    expect(citations.length).toBeGreaterThan(0);
    expect(citations[0].documentTitle).toBe("Biology 101");
  });

  it("returns empty for no matches", () => {
    const { text, citations } = retrieveRelevantChunks("quantum physics", mockDocs);
    expect(text).toBe("");
    expect(citations).toEqual([]);
  });
});

describe("youtubeId", () => {
  it("parses watch URLs", () => {
    expect(youtubeId("https://www.youtube.com/watch?v=dQw4w9WgXcQ")).toBe("dQw4w9WgXcQ");
  });

  it("parses youtu.be URLs", () => {
    expect(youtubeId("https://youtu.be/dQw4w9WgXcQ")).toBe("dQw4w9WgXcQ");
  });

  it("parses embed URLs", () => {
    expect(youtubeId("https://www.youtube.com/embed/dQw4w9WgXcQ")).toBe("dQw4w9WgXcQ");
  });

  it("returns null for invalid URLs", () => {
    expect(youtubeId("https://example.com")).toBeNull();
    expect(youtubeId("not a url")).toBeNull();
  });
});
