import { useApp } from "../lib/app-context";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  FileText, Sparkles, BookOpen, Zap, Clock,
  Award, Flame, ChevronRight, Plus, Brain, ListChecks,
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
        <div style={{ marginBottom: 32 }}>
          <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
            {greeting} <span style={{ fontSize: 24 }}>👋</span>
          </h1>
          <p style={{ fontSize: 16, color: "var(--text-muted)" }}>
            What do you want to learn today?
          </p>
        </div>

        {/* Stats row */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12, marginBottom: 28 }}>
          <StatCard icon={BookOpen} label="Projects" value={projects.length} color="var(--accent)" />
          <StatCard icon={Clock} label="Study Sessions" value={0} color="var(--accent-violet)" />
          <StatCard icon={Award} label="Quiz Score" value="—" color="var(--success)" />
          <StatCard icon={Flame} label="Study Streak" value={0} color="var(--warning)" />
        </div>

        {/* Primary actions */}
        <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>Quick Start</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12, marginBottom: 32 }}>
          <ActionCard
            icon={Plus}
            title="New Study"
            description="Create a new study project"
            onClick={() => setShowCreate(true)}
            color="var(--accent)"
          />
          <ActionCard
            icon={FileText}
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

        {/* Quick Actions */}
        <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>Quick Actions</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12, marginBottom: 32 }}>
          <QuickAction icon={FileText} label="Summarize" onClick={() => navigate("/library")} />
          <QuickAction icon={Brain} label="Create Flashcards" onClick={() => navigate("/library")} />
          <QuickAction icon={ListChecks} label="Generate Quiz" onClick={() => navigate("/library")} />
          <QuickAction icon={Sparkles} label="Explain a Concept" onClick={() => navigate("/tutor")} />
          <QuickAction icon={Zap} label="Create Study Plan" onClick={() => navigate("/study-plans")} />
        </div>

        {/* Recent Projects */}
        {projects.length > 0 ? (
          <>
            <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              Continue Learning
              <button className="btn btn-ghost btn-sm" onClick={() => navigate("/library")}>
                View all <ChevronRight size={14} />
              </button>
            </h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16 }}>
              {projects.slice(0, 6).map((project) => (
                <div
                  key={project.id}
                  className="card card-hover"
                  onClick={() => navigate(`/project/${project.id}`)}
                  style={{ padding: 20, cursor: "pointer" }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                    <div style={{ width: 10, height: 10, borderRadius: "50%", background: project.color, flexShrink: 0 }} />
                    <div style={{ fontWeight: 700, fontSize: 15, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {project.name}
                    </div>
                  </div>
                  {project.description && (
                    <div style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 8, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {project.description}
                    </div>
                  )}
                  <div style={{ fontSize: 11, color: "var(--text-faint)" }}>
                    Updated {timeAgo(project.updatedAt)}
                  </div>
                </div>
              ))}
            </div>
          </>
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
              <h3 style={{ fontSize: 17, fontWeight: 700 }}>New Study Project</h3>
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
