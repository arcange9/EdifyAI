import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useApp } from "../lib/app-context";
import { db } from "../lib/db";
import type { Document, Note, Flashcard, QuizQuestion } from "../lib/types";
import {
  FileText, Brain, ListChecks, FileDown, Search, LayoutGrid, List,
  BookOpen,
} from "lucide-react";
import { EmptyState } from "../components/ui/EmptyState";
import { StatCard } from "../components/ui/StatCard";

export default function Library() {
  const { projects } = useApp();
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [viewMode, setViewMode] = useState<"grid" | "list">("list");
  const [docs, setDocs] = useState<Document[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [cards, setCards] = useState<Flashcard[]>([]);
  const [quizzes, setQuizzes] = useState<QuizQuestion[]>([]);

  useEffect(() => {
    (async () => {
      const aDocs: Document[] = [], aNotes: Note[] = [], aCards: Flashcard[] = [], aQuizzes: QuizQuestion[] = [];
      for (const p of projects) {
        aDocs.push(...await db.getDocuments(p.id));
        aNotes.push(...await db.getNotes(p.id));
        aCards.push(...await db.getFlashcards(p.id));
        aQuizzes.push(...await db.getQuizQuestions(p.id));
      }
      setDocs(aDocs); setNotes(aNotes); setCards(aCards); setQuizzes(aQuizzes);
    })();
  }, [projects]);

  const s = search.toLowerCase();
  const f = <T extends { title?: string; front?: string; question?: string }>(arr: T[]) =>
    arr.filter((item) => !s || (item.title ?? item.front ?? item.question ?? "").toLowerCase().includes(s));

  const filteredDocs = f(docs);
  const filteredNotes = f(notes);
  const filteredCards = f(cards);
  const filteredQuizzes = f(quizzes);
  const totalItems = filteredDocs.length + filteredNotes.length + filteredCards.length + filteredQuizzes.length;

  return (
    <div className="scroll-container" style={{ padding: "32px 40px" }}>
      <div style={{ maxWidth: "var(--max-content)", margin: "0 auto" }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
          <h1 style={{ fontSize: 24, fontWeight: 800 }}>Library</h1>
          <div style={{ display: "flex", gap: 4 }} className="card-inner">
            <button className={`btn btn-icon ${viewMode === "list" ? "btn-secondary" : "btn-ghost"}`} onClick={() => setViewMode("list")} title="List view">
              <List size={16} />
            </button>
            <button className={`btn btn-icon ${viewMode === "grid" ? "btn-secondary" : "btn-ghost"}`} onClick={() => setViewMode("grid")} title="Grid view">
              <LayoutGrid size={16} />
            </button>
          </div>
        </div>

        {/* Search */}
        <div style={{ position: "relative", marginBottom: 24, maxWidth: 500 }}>
          <Search size={18} style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "var(--text-faint)" }} />
          <input className="input" style={{ paddingLeft: 40 }} placeholder="Search documents, notes, flashcards..." value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>

        {/* Stats */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12, marginBottom: 28 }}>
          <StatCard icon={FileText} label="Documents" value={docs.length} color="var(--accent)" />
          <StatCard icon={FileDown} label="Notes" value={notes.length} color="var(--accent-violet)" />
          <StatCard icon={Brain} label="Flashcards" value={cards.length} color="var(--success)" />
          <StatCard icon={ListChecks} label="Quiz Questions" value={quizzes.length} color="var(--warning)" />
        </div>

        {/* Content */}
        {totalItems === 0 ? (
          <EmptyState
            icon={BookOpen}
            title={search ? "No results found" : "Your library is empty"}
            description={search ? "Try different search terms." : "Import your first learning material and let Edify turn it into notes, flashcards, quizzes and more."}
            action={!search && <button className="btn btn-primary" onClick={() => navigate("/")}>Import Document</button>}
          />
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            {filteredDocs.length > 0 && (
              <LibrarySection title="Documents" count={filteredDocs.length}>
                {viewMode === "list" ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {filteredDocs.map((doc) => (
                      <div key={doc.id} className="card card-hover" style={{ padding: 14, display: "flex", alignItems: "center", gap: 12 }} onClick={() => navigate(`/project/${doc.projectId}`)}>
                        <div style={{ width: 36, height: 36, borderRadius: "var(--radius-sm)", background: "var(--accent-light)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                          <FileText size={18} style={{ color: "var(--accent)" }} />
                        </div>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontWeight: 600, fontSize: 14 }}>{doc.title}</div>
                          <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{doc.sourceKind} · {doc.chunks.length} chunks · {new Date(doc.createdAt).toLocaleDateString()}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 12 }}>
                    {filteredDocs.map((doc) => (
                      <div key={doc.id} className="card card-hover" style={{ padding: 16, cursor: "pointer" }} onClick={() => navigate(`/project/${doc.projectId}`)}>
                        <div style={{ width: 40, height: 40, borderRadius: "var(--radius)", background: "var(--accent-light)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 10 }}>
                          <FileText size={20} style={{ color: "var(--accent)" }} />
                        </div>
                        <div style={{ fontWeight: 600, fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{doc.title}</div>
                        <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{doc.sourceKind} · {new Date(doc.createdAt).toLocaleDateString()}</div>
                      </div>
                    ))}
                  </div>
                )}
              </LibrarySection>
            )}

            {filteredNotes.length > 0 && (
              <LibrarySection title="Notes" count={filteredNotes.length}>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {filteredNotes.map((note) => (
                    <div key={note.id} className="card card-hover" style={{ padding: 14, display: "flex", alignItems: "center", gap: 12 }} onClick={() => navigate(`/project/${note.projectId}`)}>
                      <div style={{ width: 36, height: 36, borderRadius: "var(--radius-sm)", background: "var(--accent-light)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <FileDown size={18} style={{ color: "var(--accent-violet)" }} />
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontWeight: 600, fontSize: 14 }}>{note.title}</div>
                        <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{note.type} · {new Date(note.createdAt).toLocaleDateString()}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </LibrarySection>
            )}

            {filteredCards.length > 0 && (
              <LibrarySection title="Flashcards" count={filteredCards.length}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 10 }}>
                  {filteredCards.slice(0, 12).map((card) => (
                    <div key={card.id} className="card" style={{ padding: 14 }}>
                      <div style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase", color: "var(--accent)", marginBottom: 6 }}>{card.topic}</div>
                      <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 6 }}>{card.front}</div>
                      <div style={{ fontSize: 12, color: "var(--text-muted)", borderTop: "1px solid var(--border)", paddingTop: 6 }}>{card.back}</div>
                    </div>
                  ))}
                </div>
              </LibrarySection>
            )}

            {filteredQuizzes.length > 0 && (
              <LibrarySection title="Quiz Questions" count={filteredQuizzes.length}>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {filteredQuizzes.slice(0, 10).map((q) => (
                    <div key={q.id} className="card" style={{ padding: 14 }}>
                      <div style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase", color: "var(--accent)", marginBottom: 4 }}>{q.type} · {q.difficulty}</div>
                      <div style={{ fontWeight: 600, fontSize: 13 }}>{q.question}</div>
                    </div>
                  ))}
                </div>
              </LibrarySection>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function LibrarySection({ title, count, children }: { title: string; count: number; children: React.ReactNode }) {
  return (
    <div>
      <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12, display: "flex", alignItems: "center", gap: 8 }}>
        {title}
        <span className="badge badge-neutral">{count}</span>
      </h2>
      {children}
    </div>
  );
}
