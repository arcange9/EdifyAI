import { useState, useEffect, useRef } from "react";
import { useParams } from "react-router-dom";
import { useApp } from "../lib/app-context";
import { db } from "../lib/db";
import { v4 as uuid } from "uuid";
import { ingest, chunkText } from "../lib/ingestion";
import { buildRAGContext } from "../lib/rag";
import type { Document, Note, Flashcard, QuizQuestion, ChatMessage, SourceKind } from "../lib/types";
import {
  FileText, MessageSquare, Brain, ListChecks, BookOpen, Upload,
  Link2, Video, Sparkles, Loader2, Send, Square, FileDown, Trash2,
  ChevronLeft, ChevronRight, RotateCw, Check, X, Award,
  AlertCircle, Clock,
} from "lucide-react";
import { marked } from "marked";
import { EmptyState } from "../components/ui/EmptyState";
import { LoadingState } from "../components/ui/LoadingState";
import { StatCard } from "../components/ui/StatCard";
import { ActionCard } from "../components/ui/ActionCard";

type Tab = "overview" | "documents" | "chat" | "notes" | "flashcards" | "quizzes";

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
  const [loadingMsg, setLoadingMsg] = useState("");
  const [importing, setImporting] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!id) return;
    (async () => {
      const p = await db.get<{ id: string; name: string; description?: string; color: string }>("projects", id);
      setProject(p ?? null);
      await loadData(id);
      setReady(true);
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
    setImporting("Importing your document...");
    try {
      const result = await ingest(kind, { file, url, filename: file?.name });
      const chunks = chunkText(result.text);
      const doc: Document = {
        id: uuid(), projectId: id,
        title: result.title ?? file?.name ?? "Untitled",
        sourceKind: kind, sourceText: result.text,
        sourceMeta: result.meta, chunks,
        createdAt: Date.now(), updatedAt: Date.now(),
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
    setLoading(true); setLoadingMsg("Analyzing your material...");
    setTimeout(() => setLoadingMsg("Generating your notes..."), 1500);
    try {
      const combinedText = documents.map((d) => d.sourceText).join("\n\n---\n\n");
      const content = await activeProvider.generateNotes(combinedText, preferences.language);
      const note: Note = { id: uuid(), projectId: id, title: "Study Notes", content, type: "notes", createdAt: Date.now(), updatedAt: Date.now() };
      await db.put("notes", note);
      await loadData(id);
    } catch (err) { alert(err instanceof Error ? err.message : "Failed"); }
    setLoading(false);
  }

  async function generateFlashcards() {
    if (!id || !activeProvider || !documents.length) return;
    setLoading(true); setLoadingMsg("Creating flashcards...");
    try {
      const text = documents.map((d) => d.sourceText).join("\n\n");
      const cards = await activeProvider.generateFlashcards(text, ["General"]);
      const flashcardObjs: Flashcard[] = cards.map((c) => ({
        id: uuid(), projectId: id, front: c.front, back: c.back, topic: c.topic,
        difficulty: "new", due: Date.now(), reps: 0, lapses: 0, createdAt: Date.now(),
      }));
      for (const fc of flashcardObjs) await db.put("flashcards", fc);
      await loadData(id);
    } catch (err) { alert(err instanceof Error ? err.message : "Failed"); }
    setLoading(false);
  }

  async function generateQuiz() {
    if (!id || !activeProvider || !documents.length) return;
    setLoading(true); setLoadingMsg("Preparing your quiz...");
    try {
      const text = documents.map((d) => d.sourceText).join("\n\n");
      const questions = await activeProvider.generateQuiz(text, { count: 10, difficulty: "medium", types: ["mcq", "true_false"] });
      for (const q of questions as Record<string, unknown>[]) {
        const quizQ: QuizQuestion = {
          id: uuid(), projectId: id,
          type: q.type as QuizQuestion["type"], topic: q.topic as string,
          difficulty: "medium", question: q.question as string,
          options: q.options as string[], correctIndex: q.correctIndex as number,
          explanation: q.explanation as string, createdAt: Date.now(),
        };
        await db.put("quizzes", quizQ);
      }
      await loadData(id);
    } catch (err) { alert(err instanceof Error ? err.message : "Failed"); }
    setLoading(false);
  }

  if (!ready || !project) return <LoadingState text="Loading project..." />;

  const tabs = [
    { key: "overview" as Tab, label: "Overview", icon: BookOpen },
    { key: "documents" as Tab, label: "Documents", icon: FileText, count: documents.length },
    { key: "chat" as Tab, label: "Chat", icon: MessageSquare },
    { key: "notes" as Tab, label: "Notes", icon: FileDown, count: notes.length },
    { key: "flashcards" as Tab, label: "Flashcards", icon: Brain, count: flashcards.length },
    { key: "quizzes" as Tab, label: "Quizzes", icon: ListChecks, count: quizzes.length },
  ];

  return (
    <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
      {/* Header with color accent */}
      <div style={{
        padding: "16px 28px", borderBottom: "1px solid var(--border)",
        display: "flex", alignItems: "center", gap: 12, background: "var(--bg-elevated)",
        borderLeft: `3px solid ${project.color}`,
      }}>
        <div style={{
          width: 32, height: 32, borderRadius: "var(--radius-sm)",
          background: `color-mix(in srgb, ${project.color} 15%, transparent)`,
          display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
        }}>
          <div style={{ width: 10, height: 10, borderRadius: "50%", background: project.color }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <h2 style={{ fontSize: 18, fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{project.name}</h2>
          {project.description && <div style={{ fontSize: 13, color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{project.description}</div>}
        </div>
        {!activeProvider && (
          <span className="badge badge-warning">No AI Provider</span>
        )}
      </div>

      {/* Tabs with counts */}
      <div className="tab-bar" style={{ padding: "0 28px" }}>
        {tabs.map((t) => (
          <button key={t.key} onClick={() => setTab(t.key)} className={`tab ${tab === t.key ? "tab-active" : ""}`}>
            <t.icon size={15} /> {t.label}
            {t.count !== undefined && t.count > 0 && (
              <span className="badge badge-neutral" style={{ marginLeft: 4, padding: "0 6px", fontSize: 10 }}>{t.count}</span>
            )}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="scroll-container" style={{ padding: "24px 28px" }}>
        <div style={{ maxWidth: "var(--max-content)", margin: "0 auto" }}>
          {tab === "overview" && (
            <OverviewTab
              documents={documents} notes={notes} flashcards={flashcards} quizzes={quizzes}
              loading={loading} loadingMsg={loadingMsg} importing={importing}
              onImport={() => setTab("documents")} onNotes={generateNotes}
              onFlashcards={generateFlashcards} onQuiz={generateQuiz}
            />
          )}

          {tab === "documents" && (
            <DocumentsTab documents={documents} projectId={id!} onImport={handleImport} onReload={() => loadData(id!)} />
          )}

          {tab === "chat" && <ChatTab projectId={id!} documents={documents} />}

          {tab === "notes" && (
            <NotesTab notes={notes} loading={loading} loadingMsg={loadingMsg} onGenerate={generateNotes}
              hasProvider={!!activeProvider} hasDocuments={documents.length > 0} />
          )}

          {tab === "flashcards" && (
            <FlashcardsTab flashcards={flashcards} loading={loading} loadingMsg={loadingMsg}
              onGenerate={generateFlashcards} hasProvider={!!activeProvider} hasDocuments={documents.length > 0} />
          )}

          {tab === "quizzes" && (
            <QuizTab quizzes={quizzes} loading={loading} loadingMsg={loadingMsg}
              onGenerate={generateQuiz} hasProvider={!!activeProvider} hasDocuments={documents.length > 0} />
          )}
        </div>
      </div>
    </div>
  );
}

/* ── OVERVIEW TAB ────────────────────────────────────────────────── */

function OverviewTab({ documents, notes, flashcards, quizzes, loading, loadingMsg, importing, onImport, onNotes, onFlashcards, onQuiz }: {
  documents: Document[]; notes: Note[]; flashcards: Flashcard[]; quizzes: QuizQuestion[];
  loading: boolean; loadingMsg: string; importing: string | null;
  onImport: () => void; onNotes: () => void; onFlashcards: () => void; onQuiz: () => void;
}) {
  const hasContent = documents.length > 0;
  return (
    <div className="fade-in">
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 12, marginBottom: 24 }}>
        <StatCard icon={FileText} label="Documents" value={documents.length} color="var(--accent)" />
        <StatCard icon={FileDown} label="Notes" value={notes.length} color="var(--accent-violet)" />
        <StatCard icon={Brain} label="Flashcards" value={flashcards.length} color="var(--success)" />
        <StatCard icon={ListChecks} label="Quiz Questions" value={quizzes.length} color="var(--warning)" />
      </div>

      <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 12 }}>Study Tools</h3>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12, marginBottom: 20 }}>
        <ActionCard icon={Upload} title="Import Document" description="PDF, DOCX, TXT, URL, YouTube" onClick={onImport} color="var(--accent)" />
        <ActionCard icon={Sparkles} title="Generate Notes" description="AI-powered study notes" onClick={onNotes} color="var(--accent-violet)" />
        <ActionCard icon={Brain} title="Generate Flashcards" description="Spaced repetition cards" onClick={onFlashcards} color="var(--success)" />
        <ActionCard icon={ListChecks} title="Generate Quiz" description="Test your knowledge" onClick={onQuiz} color="var(--warning)" />
      </div>

      {!hasContent && (
        <div className="card-inner" style={{ padding: 16, display: "flex", alignItems: "center", gap: 10 }}>
          <Sparkles size={18} style={{ color: "var(--accent)", flexShrink: 0 }} />
          <span style={{ fontSize: 13, color: "var(--text-muted)" }}>
            Start by importing a document, then generate notes, flashcards, and quizzes from it.
          </span>
        </div>
      )}

      {(loading || importing) && (
        <div className="card-inner" style={{ padding: 16, marginTop: 16, display: "flex", alignItems: "center", gap: 10 }}>
          <Loader2 size={18} className="animate-spin" style={{ color: "var(--accent)" }} />
          <span style={{ fontSize: 14, color: "var(--text-secondary)" }}>{importing || loadingMsg}</span>
        </div>
      )}
    </div>
  );
}

/* ── DOCUMENTS TAB ──────────────────────────────────────────────── */

function DocumentsTab({ documents, projectId: _projectId, onImport, onReload }: {
  documents: Document[]; projectId: string;
  onImport: (kind: SourceKind, file?: File, url?: string) => void; onReload: () => void;
}) {
  void _projectId;
  if (documents.length === 0) {
    return (
      <div className="fade-in">
        <EmptyState
          icon={FileText}
          title="No documents yet"
          description="Import your first learning material and let Edify turn it into notes, flashcards, quizzes, and more."
          action={
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn btn-primary" onClick={() => document.getElementById("file-input")?.click()}>
                <Upload size={16} /> Import File
              </button>
              <input id="file-input" type="file" accept=".pdf,.docx,.doc,.txt,.md" style={{ display: "none" }}
                onChange={(e) => { const f = e.target.files?.[0]; if (f) onImport(f.name.endsWith(".pdf") ? "pdf" : f.name.endsWith(".docx") || f.name.endsWith(".doc") ? "docx" : "txt", f); }} />
              <button className="btn btn-outline" onClick={() => { const url = prompt("Enter URL:"); if (url) onImport("url", undefined, url); }}>
                <Link2 size={16} /> URL
              </button>
            </div>
          }
        />
      </div>
    );
  }
  return (
    <div className="fade-in">
      {/* Import bar */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
        <button className="btn btn-outline" onClick={() => document.getElementById("file-input")?.click()}>
          <Upload size={16} /> Import File
        </button>
        <input id="file-input" type="file" accept=".pdf,.docx,.doc,.txt,.md" style={{ display: "none" }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) onImport(f.name.endsWith(".pdf") ? "pdf" : f.name.endsWith(".docx") || f.name.endsWith(".doc") ? "docx" : "txt", f); }} />
        <button className="btn btn-outline" onClick={() => { const url = prompt("Enter URL:"); if (url) onImport("url", undefined, url); }}>
          <Link2 size={16} /> URL
        </button>
        <button className="btn btn-outline" onClick={() => { const url = prompt("Enter YouTube URL:"); if (url) onImport("youtube", undefined, url); }}>
          <Video size={16} /> YouTube
        </button>
      </div>
      {/* Documents list */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {documents.map((doc) => (
          <div key={doc.id} className="card" style={{ padding: 14, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, flex: 1, minWidth: 0 }}>
              <div style={{ width: 36, height: 36, borderRadius: "var(--radius-sm)", background: "var(--accent-light)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                <FileText size={18} style={{ color: "var(--accent)" }} />
              </div>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontWeight: 600, fontSize: 14, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{doc.title}</div>
                <div style={{ fontSize: 12, color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 6 }}>
                  <span className="badge badge-neutral" style={{ padding: "1px 6px", fontSize: 10 }}>{doc.sourceKind}</span>
                  {doc.chunks.length} chunks · {new Date(doc.createdAt).toLocaleDateString()}
                </div>
              </div>
            </div>
            <button className="btn btn-danger btn-icon btn-sm" onClick={async () => { await db.delete("documents", doc.id); onReload(); }}>
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── NOTES TAB ──────────────────────────────────────────────────── */

function NotesTab({ notes, loading, loadingMsg, onGenerate, hasProvider, hasDocuments }: {
  notes: Note[]; loading: boolean; loadingMsg: string; onGenerate: () => void; hasProvider: boolean; hasDocuments: boolean;
}) {
  const [activeNote, setActiveNote] = useState<string | null>(null);
  const selectedNote = notes.find((n) => n.id === activeNote) || notes[0];

  if (notes.length === 0 && !loading) {
    return (
      <div className="fade-in">
        <EmptyState
          icon={FileDown}
          title="No notes yet"
          description={hasDocuments ? "Generate study notes from your documents and Edify will summarize the key concepts for you." : "Import a document first, then generate AI-powered study notes from it."}
          action={hasProvider && hasDocuments && <button className="btn btn-primary" onClick={onGenerate}><Sparkles size={16} /> Generate Notes</button>}
        />
      </div>
    );
  }

  return (
    <div className="fade-in">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <h3 style={{ fontSize: 16, fontWeight: 700 }}>Study Notes</h3>
        <button className="btn btn-primary btn-sm" onClick={onGenerate} disabled={loading || !hasProvider || !hasDocuments}>
          <Sparkles size={14} /> Generate
        </button>
      </div>

      {loading && <LoadingState text={loadingMsg} />}

      {!loading && notes.length > 0 && (
        <div style={{ display: "flex", gap: 20 }}>
          {/* Note list */}
          {notes.length > 1 && (
            <div style={{ width: 200, flexShrink: 0, display: "flex", flexDirection: "column", gap: 6 }}>
              {notes.map((note) => (
                <button
                  key={note.id}
                  onClick={() => setActiveNote(note.id)}
                  className={`nav-item ${selectedNote?.id === note.id ? "nav-item-active" : ""}`}
                  style={{ marginBottom: 0 }}
                >
                  <FileDown size={14} />
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{note.title}</span>
                </button>
              ))}
            </div>
          )}

          {/* Active note */}
          {selectedNote && (
            <div className="card" style={{ flex: 1, padding: 24, minWidth: 0 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                <h3 style={{ fontSize: 17, fontWeight: 700 }}>{selectedNote.title}</h3>
                <span className="badge badge-neutral">{selectedNote.type}</span>
              </div>
              <div style={{ fontSize: 12, color: "var(--text-faint)", marginBottom: 16, display: "flex", alignItems: "center", gap: 6 }}>
                <Clock size={12} /> {new Date(selectedNote.createdAt).toLocaleString()}
              </div>
              <div className="divider" style={{ marginBottom: 16 }} />
              <div
                style={{ fontSize: 14, lineHeight: 1.7 }}
                dangerouslySetInnerHTML={{ __html: marked.parse(selectedNote.content) as string }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── FLASHCARDS TAB ─────────────────────────────────────────────── */

function FlashcardsTab({ flashcards, loading, loadingMsg, onGenerate, hasProvider, hasDocuments }: {
  flashcards: Flashcard[]; loading: boolean; loadingMsg: string; onGenerate: () => void; hasProvider: boolean; hasDocuments: boolean;
}) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [reviewing, setReviewing] = useState(false);
  const [reviewed, setReviewed] = useState<Record<string, "easy" | "good" | "hard">>({});
  void reviewed;

  if (flashcards.length === 0 && !loading) {
    return (
      <div className="fade-in">
        <EmptyState
          icon={Brain}
          title="No flashcards yet"
          description={hasDocuments ? "Generate flashcards from your documents and study with spaced repetition." : "Import a document first, then generate flashcards to study."}
          action={hasProvider && hasDocuments && <button className="btn btn-primary" onClick={onGenerate}><Sparkles size={16} /> Generate Flashcards</button>}
        />
      </div>
    );
  }

  if (loading) return <LoadingState text={loadingMsg} />;

  // Review mode
  if (reviewing) {
    const card = flashcards[currentIndex];
    const progress = ((currentIndex + 1) / flashcards.length) * 100;

    function nextCard(rating: "easy" | "good" | "hard") {
      setReviewed((prev) => ({ ...prev, [card.id]: rating }));
      if (currentIndex < flashcards.length - 1) {
        setCurrentIndex(currentIndex + 1);
        setFlipped(false);
      } else {
        setReviewing(false);
        setCurrentIndex(0);
        setFlipped(false);
      }
    }

    return (
      <div className="fade-in" style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
        {/* Progress */}
        <div style={{ width: "100%", maxWidth: 500, marginBottom: 24 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
            <span style={{ fontSize: 13, color: "var(--text-muted)" }}>
              Card {currentIndex + 1} of {flashcards.length}
            </span>
            <button className="btn btn-ghost btn-sm" onClick={() => { setReviewing(false); setCurrentIndex(0); setFlipped(false); }}>
              Exit
            </button>
          </div>
          <div style={{ height: 4, borderRadius: 2, background: "var(--surface-3)" }}>
            <div style={{ height: "100%", borderRadius: 2, background: "var(--accent)", width: `${progress}%`, transition: "width 300ms ease" }} />
          </div>
        </div>

        {/* Flashcard with 3D flip */}
        <div className="flashcard-container">
          <div
            className={`flashcard-inner ${flipped ? "flipped" : ""}`}
            onClick={() => setFlipped(!flipped)}
            style={{ cursor: "pointer" }}
          >
            <div className="flashcard-face flashcard-front">
              <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", color: "var(--accent)", marginBottom: 12 }}>
                {card.topic}
              </div>
              <div style={{ fontSize: 18, fontWeight: 600, textAlign: "center", lineHeight: 1.5 }}>
                {card.front}
              </div>
              <div style={{ position: "absolute", bottom: 16, fontSize: 11, color: "var(--text-faint)" }}>
                Click to flip
              </div>
            </div>
            <div className="flashcard-face flashcard-back">
              <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", color: "var(--accent-violet)", marginBottom: 12 }}>
                Answer
              </div>
              <div style={{ fontSize: 16, textAlign: "center", lineHeight: 1.6, color: "var(--text)" }}>
                {card.back}
              </div>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div style={{ display: "flex", gap: 8, marginTop: 24 }}>
          <button className="btn btn-ghost btn-icon tooltip" data-tooltip="Previous" onClick={() => { if (currentIndex > 0) { setCurrentIndex(currentIndex - 1); setFlipped(false); } }} disabled={currentIndex === 0}>
            <ChevronLeft size={18} />
          </button>
          <button className="btn btn-outline" onClick={() => setFlipped(!flipped)}>
            <RotateCw size={14} /> Flip
          </button>
          {flipped ? (
            <>
              <button className="btn btn-danger" onClick={() => nextCard("hard")} style={{ marginLeft: 8 }}>
                Hard
              </button>
              <button className="btn btn-secondary" onClick={() => nextCard("good")}>
                Good
              </button>
              <button className="btn btn-primary" onClick={() => nextCard("easy")}>
                Easy
              </button>
            </>
          ) : (
            <button className="btn btn-ghost btn-icon tooltip" data-tooltip="Next" onClick={() => { if (currentIndex < flashcards.length - 1) { setCurrentIndex(currentIndex + 1); setFlipped(false); } }} disabled={currentIndex === flashcards.length - 1}>
              <ChevronRight size={18} />
            </button>
          )}
        </div>
      </div>
    );
  }

  // Browse mode — grid view
  return (
    <div className="fade-in">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <h3 style={{ fontSize: 16, fontWeight: 700 }}>{flashcards.length} Flashcards</h3>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn btn-outline" onClick={() => setReviewing(true)}>
            <Brain size={14} /> Start Review
          </button>
          <button className="btn btn-primary btn-sm" onClick={onGenerate} disabled={loading || !hasProvider || !hasDocuments}>
            <Sparkles size={14} /> Generate
          </button>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 12 }}>
        {flashcards.map((card) => (
          <div key={card.id} className="card" style={{ padding: 16 }}>
            <div style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase", color: "var(--accent)", marginBottom: 8 }}>{card.topic}</div>
            <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 8, lineHeight: 1.4 }}>{card.front}</div>
            <div style={{ fontSize: 13, color: "var(--text-muted)", borderTop: "1px solid var(--border)", paddingTop: 8, lineHeight: 1.5 }}>{card.back}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── QUIZ TAB ───────────────────────────────────────────────────── */

function QuizTab({ quizzes, loading, loadingMsg, onGenerate, hasProvider, hasDocuments }: {
  quizzes: QuizQuestion[]; loading: boolean; loadingMsg: string; onGenerate: () => void; hasProvider: boolean; hasDocuments: boolean;
}) {
  const [quizMode, setQuizMode] = useState(false);
  const [currentQ, setCurrentQ] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showResult, setShowResult] = useState(false);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState<boolean[]>([]);
  const [finished, setFinished] = useState(false);

  if (quizzes.length === 0 && !loading) {
    return (
      <div className="fade-in">
        <EmptyState
          icon={ListChecks}
          title="No quizzes yet"
          description={hasDocuments ? "Generate a quiz from your documents and test your knowledge." : "Import a document first, then generate a quiz to test yourself."}
          action={hasProvider && hasDocuments && <button className="btn btn-primary" onClick={onGenerate}><Sparkles size={16} /> Generate Quiz</button>}
        />
      </div>
    );
  }

  if (loading) return <LoadingState text={loadingMsg} />;

  // Quiz finished — results screen
  if (finished) {
    const percentage = Math.round((score / quizzes.length) * 100);
    const incorrect = answers.filter((a) => !a).length;
    return (
      <div className="fade-in" style={{ display: "flex", flexDirection: "column", alignItems: "center", maxWidth: 500, margin: "0 auto", paddingTop: 40 }}>
        <div style={{
          width: 80, height: 80, borderRadius: "50%",
          display: "flex", alignItems: "center", justifyContent: "center",
          background: percentage >= 70 ? "var(--success-bg)" : percentage >= 50 ? "var(--warning-bg)" : "var(--error-bg)",
          marginBottom: 20,
        }}>
          <Award size={40} style={{ color: percentage >= 70 ? "var(--success)" : percentage >= 50 ? "var(--warning)" : "var(--error)" }} />
        </div>
        <h2 style={{ fontSize: 28, fontWeight: 800, marginBottom: 8 }}>{percentage}%</h2>
        <p style={{ fontSize: 16, color: "var(--text-muted)", marginBottom: 24 }}>
          {score} correct · {incorrect} incorrect · {quizzes.length} total
        </p>

        {/* Question breakdown */}
        <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 8, marginBottom: 24 }}>
          {quizzes.map((q, i) => (
            <div key={q.id} className="card-inner" style={{ padding: 12, display: "flex", alignItems: "center", gap: 10 }}>
              {answers[i] ? <Check size={16} style={{ color: "var(--success)", flexShrink: 0 }} /> : <X size={16} style={{ color: "var(--error)", flexShrink: 0 }} />}
              <span style={{ fontSize: 13, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{q.question}</span>
            </div>
          ))}
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn btn-outline" onClick={() => { setQuizMode(false); setFinished(false); setCurrentQ(0); setSelectedAnswer(null); setShowResult(false); setScore(0); setAnswers([]); }}>
            Back to Questions
          </button>
          <button className="btn btn-primary" onClick={() => { setFinished(false); setCurrentQ(0); setSelectedAnswer(null); setShowResult(false); setScore(0); setAnswers([]); setQuizMode(true); }}>
            <RotateCw size={14} /> Retry Quiz
          </button>
        </div>
      </div>
    );
  }

  // Interactive quiz mode
  if (quizMode) {
    const question = quizzes[currentQ];
    const progress = ((currentQ + 1) / quizzes.length) * 100;

    function submitAnswer() {
      if (selectedAnswer === null) return;
      const correct = selectedAnswer === question.correctIndex;
      if (correct) setScore(score + 1);
      setAnswers([...answers, correct]);
      setShowResult(true);
    }

    function nextQuestion() {
      if (currentQ < quizzes.length - 1) {
        setCurrentQ(currentQ + 1);
        setSelectedAnswer(null);
        setShowResult(false);
      } else {
        setFinished(true);
      }
    }

    return (
      <div className="fade-in" style={{ maxWidth: 600, margin: "0 auto" }}>
        {/* Progress bar */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
            <span style={{ fontSize: 13, color: "var(--text-muted)" }}>Question {currentQ + 1} of {quizzes.length}</span>
            <button className="btn btn-ghost btn-sm" onClick={() => setQuizMode(false)}>Exit</button>
          </div>
          <div style={{ height: 4, borderRadius: 2, background: "var(--surface-3)" }}>
            <div style={{ height: "100%", borderRadius: 2, background: "var(--accent)", width: `${progress}%`, transition: "width 300ms ease" }} />
          </div>
        </div>

        {/* Question card */}
        <div className="card" style={{ padding: 24, marginBottom: 20 }}>
          <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
            <span className="badge badge-brand">{question.type}</span>
            <span className="badge badge-neutral">{question.topic}</span>
          </div>
          <h3 style={{ fontSize: 18, fontWeight: 600, lineHeight: 1.5, marginBottom: 20 }}>{question.question}</h3>

          {/* Options */}
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {question.options.map((opt, j) => {
              let bg = "var(--surface-2)";
              let border = "1px solid var(--border)";
              let fontWeight = 400;
              if (showResult) {
                if (j === question.correctIndex) {
                  bg = "var(--success-bg)"; border = "1px solid var(--success)"; fontWeight = 600;
                } else if (j === selectedAnswer) {
                  bg = "var(--error-bg)"; border = "1px solid var(--error)";
                }
              } else if (j === selectedAnswer) {
                bg = "var(--accent-light)"; border = "1px solid var(--accent)"; fontWeight = 500;
              }
              return (
                <button
                  key={j}
                  onClick={() => !showResult && setSelectedAnswer(j)}
                  disabled={showResult}
                  style={{
                    padding: "12px 16px", borderRadius: "var(--radius)", background: bg, border,
                    fontSize: 14, fontWeight, cursor: showResult ? "default" : "pointer",
                    transition: "all var(--transition)", textAlign: "left",
                    display: "flex", alignItems: "center", gap: 10,
                  }}
                >
                  <span style={{ width: 24, height: 24, borderRadius: "50%", border: "1px solid var(--border-strong)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 600, flexShrink: 0 }}>
                    {String.fromCharCode(65 + j)}
                  </span>
                  <span style={{ flex: 1 }}>{opt}</span>
                  {showResult && j === question.correctIndex && <Check size={16} style={{ color: "var(--success)" }} />}
                  {showResult && j === selectedAnswer && j !== question.correctIndex && <X size={16} style={{ color: "var(--error)" }} />}
                </button>
              );
            })}
          </div>

          {/* Explanation */}
          {showResult && (
            <div className="card-inner" style={{ padding: 12, marginTop: 16, background: selectedAnswer === question.correctIndex ? "var(--success-bg)" : "var(--error-bg)" }}>
              <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                {selectedAnswer === question.correctIndex ? <Check size={16} style={{ color: "var(--success)", marginTop: 2 }} /> : <AlertCircle size={16} style={{ color: "var(--error)", marginTop: 2 }} />}
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
                    {selectedAnswer === question.correctIndex ? "Correct!" : "Incorrect"}
                  </div>
                  <div style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.5 }}>{question.explanation}</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Actions */}
        <div style={{ display: "flex", justifyContent: "center" }}>
          {!showResult ? (
            <button className="btn btn-primary btn-lg" onClick={submitAnswer} disabled={selectedAnswer === null}>
              Submit Answer
            </button>
          ) : (
            <button className="btn btn-primary btn-lg" onClick={nextQuestion}>
              {currentQ < quizzes.length - 1 ? "Next Question" : "See Results"} <ChevronRight size={16} />
            </button>
          )}
        </div>
      </div>
    );
  }

  // Browse mode — question list
  return (
    <div className="fade-in">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <h3 style={{ fontSize: 16, fontWeight: 700 }}>{quizzes.length} Questions</h3>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn btn-primary" onClick={() => { setQuizMode(true); setCurrentQ(0); setSelectedAnswer(null); setShowResult(false); setScore(0); setAnswers([]); setFinished(false); }}>
            <ListChecks size={16} /> Start Quiz
          </button>
          <button className="btn btn-outline btn-sm" onClick={onGenerate} disabled={loading || !hasProvider || !hasDocuments}>
            <Sparkles size={14} /> Generate
          </button>
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {quizzes.map((q, i) => (
          <div key={q.id} className="card" style={{ padding: 16 }}>
            <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
              <span className="badge badge-brand">Q{i + 1}</span>
              <span className="badge badge-neutral">{q.type}</span>
              <span className="badge badge-neutral">{q.topic}</span>
            </div>
            <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 10, lineHeight: 1.4 }}>{q.question}</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {q.options.map((opt, j) => (
                <div key={j} style={{
                  padding: "6px 12px", borderRadius: "var(--radius-sm)", fontSize: 13,
                  background: j === q.correctIndex ? "var(--success-bg)" : "transparent",
                  fontWeight: j === q.correctIndex ? 600 : 400,
                  color: j === q.correctIndex ? "var(--success)" : "var(--text-muted)",
                  display: "flex", alignItems: "center", gap: 6,
                }}>
                  <span>{String.fromCharCode(65 + j)}.</span> {opt}
                  {j === q.correctIndex && <Check size={12} />}
                </div>
              ))}
            </div>
            {q.explanation && <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 8, fontStyle: "italic", lineHeight: 1.5 }}>{q.explanation}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── CHAT TAB ───────────────────────────────────────────────────── */

function ChatTab({ projectId, documents }: { projectId: string; documents: Document[] }) {
  const { activeProvider } = useApp();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    (async () => setMessages(await db.getChatMessages(projectId)))();
  }, [projectId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, streamText]);

  const suggestions = [
    "Summarize the key concepts",
    "Explain the main ideas",
    "What are the most important points?",
    "Create a study guide",
  ];

  async function send(text: string) {
    if (!text.trim() || !activeProvider) return;
    const userMsg: ChatMessage = { id: uuid(), projectId, role: "user", content: text, at: Date.now() };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setStreaming(true);
    setStreamText("");

    const { context, citations } = buildRAGContext(text, documents);
    const controller = new AbortController();
    abortRef.current = controller;

    let fullText = "";
    try {
      await activeProvider.streamChat(
        {
          system: "You are Edify AI, a helpful study tutor. Answer questions using the provided context when available. Be clear and educational. Use markdown formatting.",
          messages: [{ role: "user", content: text }],
          context, signal: controller.signal,
        },
        (token) => { fullText += token; setStreamText(fullText); }
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
      {/* Context indicator */}
      <div style={{ padding: "8px 0", marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Using:</span>
        <span className="badge badge-brand">
          {documents.length > 0 ? `${documents.length} document${documents.length > 1 ? "s" : ""}` : "No documents"}
        </span>
      </div>

      <div ref={scrollRef} className="scroll-container" style={{ flex: 1, padding: "8px 0" }}>
        {messages.length === 0 && !streaming && (
          <div style={{ paddingTop: 20 }}>
            <EmptyState
              icon={MessageSquare}
              title="Ask about your materials"
              description="Chat with your study materials. Ask questions, get explanations, or request summaries."
            />
            {/* Suggestion prompts */}
            {documents.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", marginTop: 16 }}>
                {suggestions.map((s) => (
                  <button
                    key={s}
                    className="btn btn-outline btn-sm"
                    onClick={() => send(s)}
                    style={{ borderRadius: "var(--radius-full)" }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        <div style={{ maxWidth: 680, display: "flex", flexDirection: "column", gap: 14 }}>
          {messages.map((msg) => (
            <div key={msg.id} className="fade-in" style={{ display: "flex", justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}>
              {msg.role === "user" ? (
                <div className="chat-bubble-user">{msg.content}</div>
              ) : (
                <div className="chat-bubble-ai">
                  <div dangerouslySetInnerHTML={{ __html: marked.parse(msg.content) as string }} />
                  {msg.citations && msg.citations.length > 0 && (
                    <div style={{ marginTop: 10, paddingTop: 8, borderTop: "1px solid var(--border)", display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {msg.citations.map((c, i) => (
                        <span key={i} className="chat-citation" title={c.text}>[{i + 1}] {c.documentTitle}</span>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          {streaming && (
            <div style={{ display: "flex", justifyContent: "flex-start" }}>
              <div className="chat-bubble-ai">
                <div dangerouslySetInnerHTML={{ __html: marked.parse(streamText || "Thinking...") as string }} />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Composer */}
      <div style={{ display: "flex", gap: 8, padding: "12px 0", borderTop: "1px solid var(--border)" }}>
        <input
          className="input"
          placeholder="Ask about your materials..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send(input)}
          disabled={streaming}
          style={{ flex: 1 }}
        />
        {streaming ? (
          <button className="btn btn-ghost btn-icon" onClick={() => abortRef.current?.abort()} title="Stop">
            <Square size={16} />
          </button>
        ) : (
          <button className="btn btn-primary btn-icon" onClick={() => send(input)} disabled={!input.trim()} title="Send">
            <Send size={16} />
          </button>
        )}
      </div>
    </div>
  );
}
