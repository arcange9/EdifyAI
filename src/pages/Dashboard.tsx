import { useApp } from "../lib/app-context";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  FileText, Sparkles, BookOpen, Zap, Clock,
  Award, Flame, ChevronRight, Plus, Brain, ListChecks,
  GraduationCap, Upload, Calendar,
} from "lucide-react";
import { db } from "../lib/db";
import { v4 as uuid } from "uuid";
import { StatCard } from "../components/ui/StatCard";
import { ActionCard } from "../components/ui/ActionCard";
import { EmptyState } from "../components/ui/EmptyState";

export default function Dashboard() {
  const { projects, activeProvider, refreshProjects } = useApp();
  const navigate = useNavigate();
  const [showCreate, setShowCreate] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [projectDesc, setProjectDesc] = useState("");
  const [recentItems, setRecentItems] = useState<{ type: string; title: string; projectId: string; date: number; icon: typeof FileText; color: string }[]>([]);

  // Gather recent activity across all projects
  useEffect(() => {
    (async () => {
      const items: { type: string; title: string; projectId: string; date: number; icon: typeof FileText; color: string }[] = [];
      for (const p of projects.slice(0, 5)) {
        const docs = await db.getDocuments(p.id);
        const notes = await db.getNotes(p.id);
        const cards = await db.getFlashcards(p.id);
        const quizzes = await db.getQuizQuestions(p.id);
        docs.slice(-3).forEach((d) => items.push({ type: "Document", title: d.title, projectId: p.id, date: d.createdAt, icon: FileText, color: "var(--accent)" }));
        notes.slice(-2).forEach((n) => items.push({ type: "Note", title: n.title, projectId: p.id, date: n.createdAt, icon: FileText, color: "var(--accent-violet)" }));
        if (cards.length) items.push({ type: "Flashcards", title: `${cards.length} cards`, projectId: p.id, date: cards[0].createdAt, icon: Brain, color: "var(--success)" });
        if (quizzes.length) items.push({ type: "Quiz", title: `${quizzes.length} questions`, projectId: p.id, date: quizzes[0].createdAt, icon: ListChecks, color: "var(--warning)" });
      }
      items.sort((a, b) => b.date - a.date);
      setRecentItems(items.slice(0, 6));
    })();
  }, [projects]);

  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";

  async function createProject(name: string, desc: string) {
    const project = {
      id: uuid(),
      name,
      description: desc,
      color: ["#4f46e5", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981", "#06b6d4"][Math.floor(Math.random() * 6)],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    await db.put("projects", project);
    await refreshProjects();
    setShowCreate(false);
    setProjectName("");
    setProjectDesc("");
    navigate(`/project/${project.id}`);
  }

  const hasProvider = !!activeProvider;

  return (
    <div className="scroll-container" style={{ padding: "32px 40px" }}>
      <div className="fade-in" style={{ maxWidth: "var(--max-content)", margin: "0 auto" }}>
        {/* Hero greeting */}
        <div style={{ marginBottom: 28 }}>
          <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4, letterSpacing: -0.03 }}>
            {greeting}.
          </h1>
          <p style={{ fontSize: 16, color: "var(--text-muted)" }}>
            What would you like to learn today?
          </p>
        </div>

        {/* Primary actions */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12, marginBottom: 28 }}>
          <ActionCard
            icon={Plus}
            title="New Study"
            description="Create a new study project"
            onClick={() => setShowCreate(true)}
            color="var(--accent)"
          />
          <ActionCard
            icon={Upload}
            title="Import Document"
            description="PDF, DOCX, TXT, Markdown"
            onClick={() => {
              if (!hasProvider) { navigate("/settings"); return; }
              createProject("New Study", "");
            }}
            color="var(--accent-violet)"
          />
          <ActionCard
            icon={Sparkles}
            title="Ask Edify"
            description="Chat with your materials"
            onClick={() => {
              if (!projects.length) { setShowCreate(true); return; }
              navigate(`/project/${projects[0].id}`);
            }}
            color="var(--brand-accent)"
          />
        </div>

        {/* Stats row */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 12, marginBottom: 28 }}>
          <StatCard icon={BookOpen} label="Projects" value={projects.length} color="var(--accent)" />
          <StatCard icon={Clock} label="Study Sessions" value={0} color="var(--accent-violet)" />
          <StatCard icon={Award} label="Quiz Score" value="—" color="var(--success)" />
          <StatCard icon={Flame} label="Study Streak" value={0} color="var(--warning)" />
        </div>

        {/* Quick Actions */}
        <h2 style={{ fontSize: 15, fontWeight: 700, marginBottom: 12, display: "flex", alignItems: "center", gap: 8 }}>
          <Zap size={16} style={{ color: "var(--accent)" }} /> Quick Actions
        </h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 10, marginBottom: 32 }}>
          <QuickAction icon={FileText} label="Summarize" onClick={() => navigate("/library")} />
          <QuickAction icon={Brain} label="Create Flashcards" onClick={() => navigate("/library")} />
          <QuickAction icon={ListChecks} label="Generate Quiz" onClick={() => navigate("/library")} />
          <QuickAction icon={GraduationCap} label="Explain a Concept" onClick={() => navigate("/tutor")} />
          <QuickAction icon={Calendar} label="Create Study Plan" onClick={() => navigate("/study-plans")} />
        </div>

        {/* Continue Learning + Recent Activity */}
        {projects.length > 0 ? (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 32 }}>
            {/* Continue Learning */}
            <div>
              <h2 style={{ fontSize: 15, fontWeight: 700, marginBottom: 12, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                Continue Learning
                <button className="btn btn-ghost btn-sm" onClick={() => navigate("/library")}>
                  View all <ChevronRight size={14} />
                </button>
              </h2>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {projects.slice(0, 4).map((project) => (
                  <div
                    key={project.id}
                    className="card card-hover"
                    onClick={() => navigate(`/project/${project.id}`)}
                    style={{ padding: 14, cursor: "pointer" }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <div style={{ width: 32, height: 32, borderRadius: "var(--radius-sm)", background: `color-mix(in srgb, ${project.color} 15%, transparent)`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                        <div style={{ width: 8, height: 8, borderRadius: "50%", background: project.color }} />
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontWeight: 600, fontSize: 14, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {project.name}
                        </div>
                        {project.description && (
                          <div style={{ fontSize: 12, color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {project.description}
                          </div>
                        )}
                      </div>
                      <ChevronRight size={14} style={{ color: "var(--text-faint)", flexShrink: 0 }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Recent Activity */}
            <div>
              <h2 style={{ fontSize: 15, fontWeight: 700, marginBottom: 12 }}>Recent Activity</h2>
              {recentItems.length > 0 ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {recentItems.map((item, i) => (
                    <div
                      key={i}
                      className="card card-hover"
                      onClick={() => navigate(`/project/${item.projectId}`)}
                      style={{ padding: 14, display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }}
                    >
                      <div style={{
                        width: 32, height: 32, borderRadius: "var(--radius-sm)",
                        background: `color-mix(in srgb, ${item.color} 12%, transparent)`,
                        display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                      }}>
                        <item.icon size={16} style={{ color: item.color }} />
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontWeight: 600, fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {item.title}
                        </div>
                        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                          {item.type} · {timeAgo(item.date)}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="card-inner" style={{ padding: 20, textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
                  No recent activity yet. Start studying to see your progress here.
                </div>
              )}
            </div>
          </div>
        ) : (
          <EmptyState
            icon={BookOpen}
            title="No projects yet"
            description="Create your first study project and let Edify turn your materials into notes, flashcards, quizzes, and more."
            action={<button className="btn btn-primary" onClick={() => setShowCreate(true)}><Plus size={16} /> Create Your First Project</button>}
          />
        )}
      </div>

      {/* Create modal */}
      {showCreate && (
        <div className="modal-overlay" onClick={() => setShowCreate(false)}>
          <div className="modal" style={{ width: 460 }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>New Study Project</h3>
              <button className="btn btn-ghost btn-icon" onClick={() => setShowCreate(false)}>✕</button>
            </div>
            <div className="modal-body">
              <div style={{ marginBottom: 16 }}>
                <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Project Name</label>
                <input
                  className="input"
                  placeholder="e.g., Computer Architecture"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  autoFocus
                />
              </div>
              <div>
                <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Description (optional)</label>
                <input
                  className="input"
                  placeholder="What are you studying?"
                  value={projectDesc}
                  onChange={(e) => setProjectDesc(e.target.value)}
                />
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-ghost" onClick={() => setShowCreate(false)}>Cancel</button>
              <button
                className="btn btn-primary"
                onClick={() => createProject(projectName || "New Study", projectDesc)}
                disabled={!projectName.trim()}
              >
                <Plus size={16} /> Create Project
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function QuickAction({ icon: Icon, label, onClick }: {
  icon: typeof FileText; label: string; onClick: () => void;
}) {
  return (
    <button
      className="card card-hover"
      onClick={onClick}
      style={{
        padding: "12px 16px", display: "flex", alignItems: "center", gap: 10,
        cursor: "pointer", textAlign: "left", border: "1px solid var(--border)",
      }}
    >
      <div style={{
        width: 32, height: 32, borderRadius: "var(--radius-sm)",
        background: "var(--accent-light)", display: "flex", alignItems: "center", justifyContent: "center",
        flexShrink: 0,
      }}>
        <Icon size={16} style={{ color: "var(--accent)" }} />
      </div>
      <span style={{ fontSize: 13, fontWeight: 600 }}>{label}</span>
      <ChevronRight size={14} style={{ marginLeft: "auto", color: "var(--text-faint)" }} />
    </button>
  );
}

function timeAgo(ts: number): string {
  const diff = Date.now() - ts;
  const min = Math.floor(diff / 60000);
  const hr = Math.floor(diff / 3600000);
  const day = Math.floor(diff / 86400000);
  if (day > 0) return `${day}d ago`;
  if (hr > 0) return `${hr}h ago`;
  if (min > 0) return `${min}m ago`;
  return "just now";
}
