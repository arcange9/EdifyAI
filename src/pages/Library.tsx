import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useApp } from "../lib/app-context";
import { db } from "../lib/db";
import type { Document, Note, Flashcard, QuizQuestion } from "../lib/types";
import { FileText, Brain, ListChecks, FileDown, Search } from "lucide-react";

export default function Library() {
  const { projects } = useApp();
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [docs, setDocs] = useState<Document[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [cards, setCards] = useState<Flashcard[]>([]);
  const [quizzes, setQuizzes] = useState<QuizQuestion[]>([]);

  useEffect(() => {
    (async () => {
      const aDocs: Document[] = [];
      const aNotes: Note[] = [];
      const aCards: Flashcard[] = [];
      const aQuizzes: QuizQuestion[] = [];
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

  return (
    <div style={{ flex: 1, overflowY: "auto", padding: "32px 48px" }}>
      <h1 style={{ fontSize: 24, fontWeight: 800, marginBottom: 16 }}>Library</h1>
      <div style={{ position: "relative", marginBottom: 28, maxWidth: 500 }}>
        <Search size={18} style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "var(--text-faint)" }} />
        <input className="input" style={{ paddingLeft: 40 }} placeholder="Search documents, notes, flashcards…" value={search} onChange={(e) => setSearch(e.target.value)} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12, marginBottom: 28 }}>
        <LibStat label="Documents" value={docs.length} icon={FileText} />
        <LibStat label="Notes" value={notes.length} icon={FileDown} />
        <LibStat label="Flashcards" value={cards.length} icon={Brain} />
        <LibStat label="Quiz Questions" value={quizzes.length} icon={ListChecks} />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {f(docs).map((doc) => (
          <div key={doc.id} className="card" style={{ padding: 14, display: "flex", alignItems: "center", gap: 12, cursor: "pointer" }} onClick={() => navigate(`/project/${doc.projectId}`)}>
            <FileText size={18} color="var(--accent)" />
            <div><div style={{ fontWeight: 600, fontSize: 14 }}>{doc.title}</div><div style={{ fontSize: 12, color: "var(--text-muted)" }}>{doc.sourceKind} · {new Date(doc.createdAt).toLocaleDateString()}</div></div>
          </div>
        ))}
        {f(notes).map((note) => (
          <div key={note.id} className="card" style={{ padding: 14, display: "flex", alignItems: "center", gap: 12, cursor: "pointer" }} onClick={() => navigate(`/project/${note.projectId}`)}>
            <FileDown size={18} color="var(--accent-violet)" />
            <div><div style={{ fontWeight: 600, fontSize: 14 }}>{note.title}</div><div style={{ fontSize: 12, color: "var(--text-muted)" }}>{note.type} · {new Date(note.createdAt).toLocaleDateString()}</div></div>
          </div>
        ))}
        {f(cards).map((card) => (
          <div key={card.id} className="card" style={{ padding: 14, display: "flex", alignItems: "center", gap: 12 }}>
            <Brain size={18} color="var(--success)" />
            <div><div style={{ fontWeight: 600, fontSize: 14 }}>{card.front}</div><div style={{ fontSize: 12, color: "var(--text-muted)" }}>{card.topic} · {card.difficulty}</div></div>
          </div>
        ))}
        {f(quizzes).map((q) => (
          <div key={q.id} className="card" style={{ padding: 14, display: "flex", alignItems: "center", gap: 12 }}>
            <ListChecks size={18} color="var(--warning)" />
            <div><div style={{ fontWeight: 600, fontSize: 14 }}>{q.question}</div><div style={{ fontSize: 12, color: "var(--text-muted)" }}>{q.type} · {q.difficulty}</div></div>
          </div>
        ))}
      </div>
    </div>
  );
}

function LibStat({ label, value, icon: Icon }: { label: string; value: number; icon: typeof FileText }) {
  return (
    <div className="card" style={{ padding: 16, display: "flex", alignItems: "center", gap: 10 }}>
      <Icon size={20} color="var(--accent)" />
      <div><div style={{ fontSize: 20, fontWeight: 800 }}>{value}</div><div style={{ fontSize: 12, color: "var(--text-muted)" }}>{label}</div></div>
    </div>
  );
}
