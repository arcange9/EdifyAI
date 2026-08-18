import { useApp } from "../lib/app-context";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { FileText, Link2, Video, Sparkles, BookOpen, Zap, Clock, Award, Flame } from "lucide-react";
import { db } from "../lib/db";
import { v4 as uuid } from "uuid";

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
      color: ["#6366f1", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981", "#06b6d4"][Math.floor(Math.random() * 6)],
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
    <div style={{ flex: 1, overflowY: "auto", padding: "32px 48px" }}>
      <div className="fade-in">
        {/* Greeting */}
        <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
          {greeting} 👋
        </h1>
        <p style={{ fontSize: 16, color: "var(--text-muted)", marginBottom: 32 }}>
          What do you want to learn today?
        </p>

        {/* Stats */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16, marginBottom: 32 }}>
          <StatCard icon={BookOpen} label="Projects" value={projects.length} color="var(--accent)" />
          <StatCard icon={Clock} label="Study Sessions" value={0} color="var(--accent-violet)" />
          <StatCard icon={Award} label="Quiz Score" value="—" color="var(--success)" />
          <StatCard icon={Flame} label="Study Streak" value={0} color="var(--warning)" />
        </div>

        {/* Quick Actions */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16, marginBottom: 32 }}>
          <ActionCard icon={FileText} title="Import Document" desc="PDF, DOCX, TXT, Markdown" onClick={() => { if (!hasProvider) { navigate("/settings"); return; } createProject("New Study", ""); }} />
          <ActionCard icon={Link2} title="Add URL" desc="Import web content" onClick={() => { if (!hasProvider) { navigate("/settings"); return; } createProject("Web Study", ""); }} />
          <ActionCard icon={Video} title="YouTube" desc="Import video transcripts" onClick={() => { if (!hasProvider) { navigate("/settings"); return; } createProject("YouTube Study", ""); }} />
          <ActionCard icon={Sparkles} title="Ask Edify" desc="Chat with your materials" onClick={() => { if (!projects.length) { setShowCreate(true); return; } navigate(`/project/${projects[0].id}/chat`); }} />
        </div>

        {/* Recent Projects */}
        {projects.length > 0 && (
          <>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 12 }}>Recent Projects</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16 }}>
              {projects.slice(0, 6).map((project) => (
                <div
                  key={project.id}
                  onClick={() => navigate(`/project/${project.id}`)}
                  className="card"
                  style={{ padding: 20, cursor: "pointer", transition: "var(--transition)" }}
                  onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--accent)"; e.currentTarget.style.transform = "translateY(-2px)"; e.currentTarget.style.boxShadow = "var(--shadow)"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.transform = ""; e.currentTarget.style.boxShadow = "var(--shadow-sm)"; }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                    <div style={{ width: 10, height: 10, borderRadius: "50%", background: project.color }} />
                    <span style={{ fontWeight: 700, fontSize: 16 }}>{project.name}</span>
                  </div>
                  {project.description && <p style={{ fontSize: 13, color: "var(--text-muted)" }}>{project.description}</p>}
                  <p style={{ fontSize: 12, color: "var(--text-faint)", marginTop: 8 }}>
                    {new Date(project.updatedAt).toLocaleDateString()}
                  </p>
                </div>
              ))}
            </div>
          </>
        )}

        {/* No provider warning */}
        {!hasProvider && (
          <div className="card" style={{ padding: 20, marginTop: 24, borderColor: "var(--warning)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
              <Zap size={20} color="var(--warning)" />
              <span style={{ fontWeight: 700, fontSize: 15 }}>No AI Provider Configured</span>
            </div>
            <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 12 }}>
              AI generation requires a configured provider. You can still browse and manage projects, but generating notes, flashcards, and quizzes needs an API key.
            </p>
            <button className="btn btn-primary" onClick={() => navigate("/settings")}>Configure Provider</button>
          </div>
        )}
      </div>

      {/* Create Project Modal */}
      {showCreate && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000 }} onClick={() => setShowCreate(false)}>
          <div className="card fade-in" style={{ padding: 28, width: 420, maxWidth: "90vw" }} onClick={(e) => e.stopPropagation()}>
            <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Create Study Project</h3>
            <div style={{ marginBottom: 12 }}>
              <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Project Name</label>
              <input className="input" placeholder="e.g. Computer Architecture" value={projectName} onChange={(e) => setProjectName(e.target.value)} autoFocus />
            </div>
            <div style={{ marginBottom: 20 }}>
              <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Description (optional)</label>
              <textarea className="input" style={{ minHeight: 60, resize: "vertical" }} placeholder="What are you studying?" value={projectDesc} onChange={(e) => setProjectDesc(e.target.value)} />
            </div>
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button className="btn btn-ghost" onClick={() => setShowCreate(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={() => createProject(projectName || "Untitled Project", projectDesc)} disabled={!projectName.trim()}>Create</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({ icon: Icon, label, value, color }: { icon: typeof BookOpen; label: string; value: number | string; color: string }) {
  return (
    <div className="card" style={{ padding: 16, display: "flex", alignItems: "center", gap: 12 }}>
      <div style={{ width: 40, height: 40, borderRadius: 10, background: `${color}15`, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <Icon size={20} color={color} />
      </div>
      <div>
        <div style={{ fontSize: 22, fontWeight: 800 }}>{value}</div>
        <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{label}</div>
      </div>
    </div>
  );
}

function ActionCard({ icon: Icon, title, desc, onClick }: { icon: typeof BookOpen; title: string; desc: string; onClick: () => void }) {
  return (
    <div className="card" style={{ padding: 20, cursor: "pointer", transition: "var(--transition)" }} onClick={onClick}
      onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--accent)"; e.currentTarget.style.transform = "translateY(-2px)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.transform = ""; }}>
      <div style={{ width: 44, height: 44, borderRadius: 12, background: "var(--accent-light)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 12 }}>
        <Icon size={22} color="var(--accent)" />
      </div>
      <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 13, color: "var(--text-muted)" }}>{desc}</div>
    </div>
  );
}
