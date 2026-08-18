import { Outlet, NavLink, useNavigate } from "react-router-dom";
import { useApp } from "../lib/app-context";
import { BookOpen, LayoutDashboard, Library, Settings, Plus, HelpCircle } from "lucide-react";

const navItems = [
  { path: "/", label: "Dashboard", icon: LayoutDashboard },
  { path: "/library", label: "Library", icon: Library },
  { path: "/settings", label: "Settings", icon: Settings },
];

export default function AppShell() {
  const { projects } = useApp();
  const navigate = useNavigate();

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      {/* Sidebar */}
      <aside style={{
        width: 240,
        minWidth: 240,
        background: "var(--bg-sidebar)",
        borderRight: `1px solid var(--border)`,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}>
        {/* Logo */}
        <div style={{ padding: "20px 20px 16px", display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <BookOpen size={20} color="white" strokeWidth={2.2} />
          </div>
          <div>
            <div style={{ fontWeight: 800, fontSize: 16, letterSpacing: -0.5, color: "var(--text)" }}>Edify AI</div>
            <div style={{ fontSize: 10, color: "var(--text-faint)", marginTop: -2 }}>Turn Knowledge Into Understanding</div>
          </div>
        </div>

        {/* New Study button */}
        <div style={{ padding: "0 16px 8px" }}>
          <button
            onClick={() => navigate("/")}
            className="btn btn-primary"
            style={{ width: "100%", justifyContent: "center", padding: "10px" }}
          >
            <Plus size={18} /> New Study
          </button>
        </div>

        {/* Nav */}
        <nav style={{ padding: "4px 12px", flex: 1, overflowY: "auto" }}>
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              style={({ isActive }) => ({
                display: "flex", alignItems: "center", gap: 10,
                padding: "9px 12px", borderRadius: 8,
                fontSize: 14, fontWeight: 500,
                textDecoration: "none",
                marginBottom: 2,
                color: isActive ? "var(--accent)" : "var(--text-muted)",
                background: isActive ? "var(--accent-light)" : "transparent",
                transition: "var(--transition)",
              })}
            >
              <item.icon size={18} />
              {item.label}
            </NavLink>
          ))}

          {/* Projects */}
          {projects.length > 0 && (
            <>
              <div style={{ padding: "12px 12px 4px", fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.8, color: "var(--text-faint)" }}>
                Projects
              </div>
              {projects.map((project) => (
                <NavLink
                  key={project.id}
                  to={`/project/${project.id}`}
                  style={({ isActive }) => ({
                    display: "flex", alignItems: "center", gap: 10,
                    padding: "9px 12px", borderRadius: 8,
                    fontSize: 14, fontWeight: 500,
                    textDecoration: "none",
                    marginBottom: 2,
                    color: isActive ? "var(--accent)" : "var(--text-muted)",
                    background: isActive ? "var(--accent-light)" : "transparent",
                    transition: "var(--transition)",
                  })}
                >
                  <div style={{ width: 8, height: 8, borderRadius: "50%", background: project.color }} />
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{project.name}</span>
                </NavLink>
              ))}
            </>
          )}
        </nav>

        {/* Footer */}
        <div style={{ padding: "12px 16px", borderTop: `1px solid var(--border)` }}>
          <button
            onClick={() => window.open("https://github.com/arcange9/EdifyAI", "_blank")}
            style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "var(--text-faint)", background: "none", border: "none", cursor: "pointer", padding: 0 }}
          >
            <HelpCircle size={14} /> Help & Feedback
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        <Outlet />
      </main>
    </div>
  );
}
