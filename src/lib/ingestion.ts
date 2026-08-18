/* Edify AI — Document Ingestion
 * Extracts text from PDF, DOCX, TXT, MD, URLs, and YouTube.
 * Chunks text for RAG retrieval.
 */

import type { SourceKind, TextChunk } from "./types";

export interface IngestResult {
  text: string;
  title?: string;
  meta?: Record<string, string | number | undefined>;
}

/* ---- Text chunking ---- */

export function chunkText(text: string, chunkSize = 1200, overlap = 150): TextChunk[] {
  const chunks: TextChunk[] = [];
  const sentences = text.split(/(?<=[.!?])\s+/);
  let current = "";
  let start = 0;

  for (const sentence of sentences) {
    if (current.length + sentence.length > chunkSize && current) {
      chunks.push({ id: `chunk-${chunks.length}`, text: current.trim(), metadata: { start, end: start + current.length } });
      current = current.slice(-overlap) + " " + sentence;
      start += current.length - sentence.length;
    } else {
      current += (current ? " " : "") + sentence;
    }
  }
  if (current.trim()) {
    chunks.push({ id: `chunk-${chunks.length}`, text: current.trim(), metadata: { start } });
  }
  return chunks;
}

/* ---- PDF extraction ---- */

export async function ingestPdf(file: File | Blob, filename?: string): Promise<IngestResult> {
  try {
    const pdfjs = await import("pdfjs-dist");
    // Use bundled worker
    const workerUrl = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).href;
    pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

    const data = await file.arrayBuffer();
    const doc = await pdfjs.getDocument({ data }).promise;
    const pages: string[] = [];
    for (let i = 1; i <= doc.numPages; i++) {
      const page = await doc.getPage(i);
      const content = await page.getTextContent();
      pages.push(content.items.map((item) => 'str' in item ? item.str : '').join(' ').trim());
      page.cleanup();
    }
    
    const text = pages.join("\n\n").trim();
    return { text, title: filename?.replace(/\.[^./\\]+$/, ""), meta: { filename, pages: pages.length } };
  } catch (err) {
    throw new Error(`Couldn't extract PDF: ${err instanceof Error ? err.message : "unknown error"}`);
  }
}

/* ---- DOCX extraction ---- */

export async function ingestDocx(file: File | Blob, filename?: string): Promise<IngestResult> {
  try {
    const mammoth = await import("mammoth");
    const arrayBuffer = await file.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer });
    return { text: result.value.trim(), title: filename?.replace(/\.[^./\\]+$/, ""), meta: { filename } };
  } catch (err) {
    throw new Error(`Couldn't extract DOCX: ${err instanceof Error ? err.message : "unknown error"}`);
  }
}

/* ---- Text/Markdown ---- */

export function ingestText(text: string): IngestResult {
  return { text: text.trim(), title: "Untitled" };
}

/* ---- URL ---- */

export async function ingestUrl(url: string): Promise<IngestResult> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch (err) {
    throw new Error(`Couldn't fetch URL: ${err instanceof Error ? err.message : "unknown error"}`);
  }
  if (!res.ok) throw new Error(`Server responded with ${res.status}`);
  const html = await res.text();
  const doc = new DOMParser().parseFromString(html, "text/html");
  doc.querySelectorAll("script, style, nav, noscript, iframe, svg, header, footer").forEach((el) => el.remove());
  const title = doc.querySelector("title")?.textContent?.trim() || doc.querySelector("h1")?.textContent?.trim();
  const text = (doc.body?.textContent ?? "").replace(/[ \t]+/g, " ").split("\n").map((l) => l.trim()).filter(Boolean).join("\n").trim();
  if (!text) throw new Error("No readable text found at that URL.");
  return { text, title, meta: { url } };
}

/* ---- YouTube ---- */

export function youtubeId(url: string): string | null {
  try {
    const u = new URL(url.startsWith("http") ? url : `https://${url}`);
    const host = u.hostname.replace(/^(www\.|m\.)/, "");
    if (host === "youtu.be") return u.pathname.slice(1).split("/")[0] || null;
    if (host === "youtube.com" || host === "youtube-nocookie.com") {
      if (u.pathname === "/watch") return u.searchParams.get("v");
      const m = u.pathname.match(/^\/(?:embed|shorts|live)\/([A-Za-z0-9_-]{11})/);
      return m ? m[1] : null;
    }
  } catch { return null; }
  return null;
}

export async function ingestYoutube(url: string): Promise<IngestResult> {
  const id = youtubeId(url);
  if (!id) throw new Error("Invalid YouTube URL.");

  // Try fetching transcript via YouTube timed text API
  for (const lang of ["en", "en-US", "en-GB"]) {
    try {
      const res = await fetch(`https://www.youtube.com/api/timedtext?lang=${lang}&v=${id}`);
      if (!res.ok) continue;
      const xml = await res.text();
      if (!xml.trim()) continue;
      const matches = [...xml.matchAll(/<text[^>]*>([\s\S]*?)<\/text>/g)];
      if (matches.length === 0) continue;
      const text = matches.map((m) => decodeEntities(m[1] ?? "").replace(/\s+/g, " ").trim()).filter(Boolean).join(" ");
      if (text) {
        // Get title
        let title: string | undefined;
        try {
          const pageRes = await fetch(`https://www.youtube.com/watch?v=${id}`);
          const html = await pageRes.text();
          const m = html.match(/<title>([^<]*)<\/title>/);
          title = m ? decodeEntities(m[1] ?? "").replace(/\s*-\s*YouTube\s*$/, "").trim() : undefined;
        } catch { /* ignore */ }
        return { text, title, meta: { url, videoId: id } };
      }
    } catch { continue; }
  }
  throw new Error("Couldn't extract transcript. The video may not have captions.");
}

function decodeEntities(s: string): string {
  return s.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&nbsp;/g, " ");
}

/* ---- Main ingest dispatcher ---- */

export async function ingest(
  kind: SourceKind,
  input: { file?: File | Blob; filename?: string; url?: string; text?: string },
): Promise<IngestResult> {
  switch (kind) {
    case "pdf": return ingestPdf(input.file!, input.filename);
    case "docx": return ingestDocx(input.file!, input.filename);
    case "txt":
    case "md": return ingestText(input.text ?? "");
    case "url": return ingestUrl(input.url!);
    case "youtube": return ingestYoutube(input.url!);
    case "blank": return { text: "" };
    case "audio": return { text: "", meta: { filename: input.filename } };
  }
}
