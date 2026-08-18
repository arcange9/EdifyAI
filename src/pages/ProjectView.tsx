import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { useApp } from "../lib/app-context";
import { db } from "../lib/db";
import { v4 as uuid } from "uuid";
import { ingest, chunkText } from "../lib/ingestion";
import { buildRAGContext } from "../lib/rag";
import type { Document, Note, Flashcard, QuizQuestion, ChatMessage, SourceKind } from "../lib/types";
import { FileText, MessageSquare, Brain, ListChecks, BookOpen, Upload, Link2, Video, Sparkles, Loader2, Send, Square, FileDown } from "lucide-react";

type Tab = "overview" | "documents" | "chat" | "notes" | "flashcards" | "quizzes" | "studyplan";

export default function ProjectView() {
  const { id } = useParams();
  
  const { activeProvider, preferences } = useApp();
  const [tab, setTab] = useState<Tab>("overview");
  const [project, setProject] = useState<{ id: string; name: string; description?: string; color: string } | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
  const [quizzes, setQuizzes] = useState<QuizQuestion[]>([]);
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    (async () => {
      const p = await db.get<{ id: string; name: string; description?: string; color: string }>("projects", id);
      setProject(p ?? null);
      await loadData(id);
    })();
  }, [id]);

  async function loadData(projectId: string) {
    setDocuments(await db.getDocuments(projectId));
    setNotes(await db.getNotes(projectId));
    setFlashcards(await db.getFlashcards(projectId));
    setQuizzes(await db.getQuizQuestions(projectId));
  }

  async function handleImport(kind: SourceKind, file?: File, url?: string) {
    if (!id || !activeProvider) return;
    setImporting("Importing…");
    try {
      const result = await ingest(kind, { file, url, filename: file?.name });
      const chunks = chunkText(result.text);
      const doc: Document = {
        id: uuid(),
        projectId: id,
        title: result.title ?? file?.name ?? "Untitled",
        sourceKind: kind,
        sourceText: result.text,
        sourceMeta: result.meta,
        chunks,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      await db.put("documents", doc);
      await loadData(id);
      setImporting(null);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Import failed");
      setImporting(null);
    }
  }

  async function generateNotes() {
    if (!id || !activeProvider || !documents.length) return;
    setLoading(true);
    try {
      const combinedText = documents.map((d) => d.sourceText).join("\n\n---\n\n");
      const content = await activeProvider.generateNotes(combinedText, preferences.language);
      const note: Note = {
        id: uuid(), projectId: id,
        title: "Study Notes",
        content, type: "notes",
        createdAt: Date.now(), updatedAt: Date.now(),
      };
      await db.put("notes", note);
      await loadData(id);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed");
    }
    setLoading(false);
  }

  async function generateFlashcards() {
    if (!id || !activeProvider || !documents.length) return;
    setLoading(true);
    try {
      const text = documents.map((d) => d.sourceText).join("\n\n");
      const cards = await activeProvider.generateFlashcards(text, ["General"]);
      const flashcardObjs: Flashcard[] = cards.map((c) => ({
        id: uuid(), projectId: id,
        front: c.front, back: c.back, topic: c.topic,
        difficulty: "new", due: Date.now(), reps: 0, lapses: 0,
        createdAt: Date.now(),
      }));
      for (const fc of flashcardObjs) await db.put("flashcards", fc);
      await loadData(id);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed");
    }
    setLoading(false);
  }

  async function generateQuiz() {
    if (!id || !activeProvider || !documents.length) return;
    setLoading(true);
    try {
      const text = documents.map((d) => d.sourceText).join("\n\n");
      const questions = await activeProvider.generateQuiz(text, { count: 10, difficulty: "medium", types: ["mcq", "true_false"] });
      for (const q of questions as Record<string, unknown>[]) {
        const quizQ: QuizQuestion = {
          id: uuid(), projectId: id,
          type: q.type as QuizQuestion["type"],
          topic: q.topic as string,
          difficulty: "medium",
          question: q.question as string,
          options: q.options as string[],
          correctIndex: q.correctIndex as number,
          explanation: q.explanation as string,
          createdAt: Date.now(),
        };
        await db.put("quizzes", quizQ);
      }
      await loadData(id);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed");
    }
    setLoading(false);
  }

  if (!project) return <div style={{ padding: 32 }}>Loading…</div>;

  const tabs = [
    { key: "overview" as Tab, label: "Overview", icon: BookOpen },
    { key: "documents" as Tab, label: "Documents", icon: FileText },
    { key: "chat" as Tab, label: "Chat", icon: MessageSquare },
    { key: "notes" as Tab, label: "Notes", icon: FileDown },
    { key: "flashcards" as Tab, label: "Flashcards", icon: Brain },
    { key: "quizzes" as Tab, label: "Quizzes", icon: ListChecks },
  ];

  return (
    <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <div style={{ padding: "20px 32px", borderBottom: `1px solid var(--border)`, display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{ width: 12, height: 12, borderRadius: "50%", background: project.color }} />
        <h2 style={{ fontSize: 20, fontWeight: 700 }}>{project.name}</h2>
        {project.description && <span style={{ fontSize: 14, color: "var(--text-muted)" }}>— {project.description}</span>}
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 2, padding: "0 32px", borderBottom: `1px solid var(--border)` }}>
        {tabs.map((t) => (
          <button key={t.key} onClick={() => setTab(t.key)} style={{
            display: "flex", alignItems: "center", gap: 8, padding: "10px 14px",
            border: "none", background: "none", cursor: "pointer",
            fontSize: 13, fontWeight: 600,
            color: tab === t.key ? "var(--accent)" : "var(--text-muted)",
            borderBottom: tab === t.key ? "2px solid var(--accent)" : "2px solid transparent",
          }}>
            <t.icon size={16} /> {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: "auto", padding: "24px 32px" }}>
        {tab === "overview" && (
          <div className="fade-in">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12, marginBottom: 24 }}>
              <Stat label="Documents" value={documents.length} />
              <Stat label="Notes" value={notes.length} />
              <Stat label="Flashcards" value={flashcards.length} />
              <Stat label="Quiz Questions" value={quizzes.length} />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12 }}>
              <button className="card" style={{ padding: 16, cursor: "pointer", textAlign: "left", border: "1px solid var(--border)" }} onClick={() => setTab("documents")}>
                <Upload size={20} color="var(--accent)" />
                <div style={{ fontWeight: 600, fontSize: 14, marginTop: 8 }}>Import Document</div>
                <div style={{ fontSize: 12, color: "var(--text-muted)" }}>PDF, DOCX, TXT, URL, YouTube</div>
              </button>
              <button className="card" style={{ padding: 16, cursor: "pointer", textAlign: "left", border: "1px solid var(--border)" }} onClick={generateNotes} disabled={loading || !activeProvider || !documents.length}>
                <Sparkles size={20} color="var(--accent)" />
                <div style={{ fontWeight: 600, fontSize: 14, marginTop: 8 }}>Generate Notes</div>
                <div style={{ fontSize: 12, color: "var(--text-muted)" }}>AI-powered study notes</div>
              </button>
              <button className="card" style={{ padding: 16, cursor: "pointer", textAlign: "left", border: "1px solid var(--border)" }} onClick={generateFlashcards} disabled={loading || !activeProvider || !documents.length}>
                <Brain size={20} color="var(--accent)" />
                <div style={{ fontWeight: 600, fontSize: 14, marginTop: 8 }}>Generate Flashcards</div>
                <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Spaced repetition cards</div>
              </button>
              <button className="card" style={{ padding: 16, cursor: "pointer", textAlign: "left", border: "1px solid var(--border)" }} onClick={generateQuiz} disabled={loading || !activeProvider || !documents.length}>
                <ListChecks size={20} color="var(--accent)" />
                <div style={{ fontWeight: 600, fontSize: 14, marginTop: 8 }}>Generate Quiz</div>
                <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Test your knowledge</div>
              </button>
            </div>
            {loading && <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 24, color: "var(--text-muted)" }}><Loader2 size={18} className="animate-spin" /> Generating…</div>}
            {importing && <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 24, color: "var(--text-muted)" }}><Loader2 size={18} className="animate-spin" /> {importing}</div>}
          </div>
        )}

        {tab === "documents" && (
          <div className="fade-in">
            <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
              <button className="btn btn-outline" onClick={() => document.getElementById("file-input")?.click()}>
                <Upload size={16} /> Import File
              </button>
              <input id="file-input" type="file" accept=".pdf,.docx,.doc,.txt,.md" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) handleImport(f.name.endsWith(".pdf") ? "pdf" : f.name.endsWith(".docx") || f.name.endsWith(".doc") ? "docx" : "txt", f); }} />
              <button className="btn btn-outline" onClick={() => { const url = prompt("Enter URL:"); if (url) handleImport("url", undefined, url); }}>
                <Link2 size={16} /> URL
              </button>
              <button className="btn btn-outline" onClick={() => { const url = prompt("Enter YouTube URL:"); if (url) handleImport("youtube", undefined, url); }}>
                <Video size={16} /> YouTube
              </button>
            </div>
            {documents.length === 0 ? (
              <p style={{ color: "var(--text-muted)" }}>No documents yet. Import a file, URL, or YouTube video to get started.</p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {documents.map((doc) => (
                  <div key={doc.id} className="card" style={{ padding: 14, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <div>
                      <div style={{ fontWeight: 600, fontSize: 14 }}>{doc.title}</div>
                      <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{doc.sourceKind} · {doc.chunks.length} chunks · {new Date(doc.createdAt).toLocaleDateString()}</div>
                    </div>
                    <button className="btn btn-danger" style={{ padding: "4px 8px" }} onClick={async () => { await db.delete("documents", doc.id); if (id) await loadData(id); }}><Trash2Icon /></button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {tab === "chat" && <ChatTab projectId={id!} documents={documents} />}

        {tab === "notes" && (
          <div className="fade-in">
            <button className="btn btn-primary" style={{ marginBottom: 16 }} onClick={generateNotes} disabled={loading || !activeProvider || !documents.length}>
              <Sparkles size={16} /> Generate Notes
            </button>
            {notes.length === 0 ? (
              <p style={{ color: "var(--text-muted)" }}>No notes yet. Generate study notes from your documents.</p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {notes.map((note) => (
                  <div key={note.id} className="card" style={{ padding: 20 }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                      <h3 style={{ fontWeight: 700, fontSize: 16 }}>{note.title}</h3>
                      <span style={{ fontSize: 12, color: "var(--text-faint)" }}>{new Date(note.createdAt).toLocaleString()}</span>
                    </div>
                    <div style={{ fontSize: 14, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>{note.content}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {tab === "flashcards" && (
          <div className="fade-in">
            <button className="btn btn-primary" style={{ marginBottom: 16 }} onClick={generateFlashcards} disabled={loading || !activeProvider || !documents.length}>
              <Sparkles size={16} /> Generate Flashcards
            </button>
            {flashcards.length === 0 ? (
              <p style={{ color: "var(--text-muted)" }}>No flashcards yet.</p>
            ) : (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 12 }}>
                {flashcards.map((card) => (
                  <div key={card.id} className="card" style={{ padding: 16 }}>
                    <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", color: "var(--accent)", marginBottom: 8 }}>{card.topic}</div>
                    <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 8 }}>{card.front}</div>
                    <div style={{ fontSize: 13, color: "var(--text-muted)", borderTop: "1px solid var(--border)", paddingTop: 8 }}>{card.back}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {tab === "quizzes" && (
          <div className="fade-in">
            <button className="btn btn-primary" style={{ marginBottom: 16 }} onClick={generateQuiz} disabled={loading || !activeProvider || !documents.length}>
              <Sparkles size={16} /> Generate Quiz
            </button>
            {quizzes.length === 0 ? (
              <p style={{ color: "var(--text-muted)" }}>No quizzes yet.</p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {quizzes.map((q, i) => (
                  <div key={q.id} className="card" style={{ padding: 16 }}>
                    <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", color: "var(--accent)", marginBottom: 6 }}>Q{i + 1} · {q.type}</div>
                    <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 10 }}>{q.question}</div>
                    {q.options.map((opt, j) => (
                      <div key={j} style={{ padding: "6px 12px", borderRadius: 8, marginBottom: 4, fontSize: 13, background: j === q.correctIndex ? "var(--success-light)" : "var(--bg-hover)", fontWeight: j === q.correctIndex ? 600 : 400 }}>
                        {String.fromCharCode(65 + j)}. {opt} {j === q.correctIndex && "✓"}
                      </div>
                    ))}
                    <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 8, fontStyle: "italic" }}>{q.explanation}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* Chat Tab */
function ChatTab({ projectId, documents }: { projectId: string; documents: Document[] }) {
  const { activeProvider } = useApp();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const abortRef = { current: null as AbortController | null };

  useEffect(() => {
    (async () => setMessages(await db.getChatMessages(projectId)))();
  }, [projectId]);

  async function send() {
    if (!input.trim() || !activeProvider) return;
    const userMsg: ChatMessage = { id: uuid(), projectId, role: "user", content: input, at: Date.now() };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setStreaming(true);
    setStreamText("");

    const { context, citations } = buildRAGContext(input, documents);

    const controller = new AbortController();
    abortRef.current = controller;

    let fullText = "";
    try {
      await activeProvider.streamChat(
        {
          system: "You are Edify AI, a helpful study tutor. Answer questions using the provided context when available. Be clear and educational.",
          messages: [{ role: "user", content: input }],
          context,
          signal: controller.signal,
        },
        (token) => {
          fullText += token;
          setStreamText(fullText);
        }
      );
    } catch (err) {
      fullText = `Error: ${err instanceof Error ? err.message : "Failed"}`;
    }

    const aiMsg: ChatMessage = { id: uuid(), projectId, role: "assistant", content: fullText, citations: citations.length ? citations : undefined, at: Date.now() };
    await db.put("chats", userMsg);
    await db.put("chats", aiMsg);
    setMessages((m) => [...m, aiMsg]);
    setStreaming(false);
    setStreamText("");
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ flex: 1, overflowY: "auto" }}>
        {messages.length === 0 && !streaming && (
          <div style={{ textAlign: "center", padding: "40px 20px", color: "var(--text-muted)" }}>
            <MessageSquare size={32} style={{ opacity: 0.4, marginBottom: 12 }} />
            <p>Ask a question about your study materials.</p>
          </div>
        )}
        {messages.map((msg) => (
          <div key={msg.id} style={{ marginBottom: 16, display: "flex", justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}>
            <div style={{ maxWidth: "70%", padding: "12px 16px", borderRadius: 12, background: msg.role === "user" ? "var(--accent)" : "var(--bg-elevated)", color: msg.role === "user" ? "white" : "var(--text)", border: msg.role === "assistant" ? "1px solid var(--border)" : "none", fontSize: 14, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
              {msg.content}
              {msg.citations && msg.citations.length > 0 && (
                <div style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid rgba(255,255,255,0.2)", fontSize: 11, opacity: 0.7 }}>
                  Sources: {msg.citations.map((c, i) => `[${i + 1}] ${c.documentTitle}`).join(", ")}
                </div>
              )}
            </div>
          </div>
        ))}
        {streaming && (
          <div style={{ marginBottom: 16, display: "flex", justifyContent: "flex-start" }}>
            <div style={{ maxWidth: "70%", padding: "12px 16px", borderRadius: 12, background: "var(--bg-elevated)", border: "1px solid var(--border)", fontSize: 14, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
              {streamText || "Thinking…"}
            </div>
          </div>
        )}
      </div>
      <div style={{ display: "flex", gap: 8, padding: "16px 0", borderTop: `1px solid var(--border)` }}>
        <input className="input" placeholder="Ask about your materials…" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send()} disabled={streaming} />
        {streaming ? (
          <button className="btn btn-ghost" onClick={() => abortRef.current?.abort()}><Square size={16} /></button>
        ) : (
          <button className="btn btn-primary" onClick={send} disabled={!input.trim()}><Send size={16} /></button>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="card" style={{ padding: 14 }}>
      <div style={{ fontSize: 22, fontWeight: 800 }}>{value}</div>
      <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{label}</div>
    </div>
  );
}

function Trash2Icon() {
  return <span style={{ fontSize: 16 }}>🗑</span>;
}
