import { useState, useEffect } from "react";
import { Outlet, NavLink, useNavigate } from "react-router-dom";
import { useApp } from "../lib/app-context";
import {
  BookOpen, LayoutDashboard, Library, Settings, Plus, HelpCircle,
  ChevronLeft, ChevronRight, GraduationCap, Brain, ListChecks,
  StickyNote, Calendar, FileText,
} from "lucide-react";

const navItems = [
  { path: "/", label: "Home", icon: LayoutDashboard },
  { path: "/library", label: "My Library", icon: Library },
  { path: "/tutor", label: "AI Tutor", icon: GraduationCap },
  { path: "/study-plans", label: "Study Plans", icon: Calendar },
];

const toolItems = [
  { path: "/library?tab=notes", label: "Notes", icon: StickyNote },
  { path: "/library?tab=flashcards", label: "Flashcards", icon: Brain },
  { path: "/library?tab=quizzes", label: "Quizzes", icon: ListChecks },
  { path: "/library?tab=documents", label: "Documents", icon: FileText },
];

export default function AppShell() {
  const { projects, activeProvider } = useApp();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(false);

  // Keyboard shortcut: Ctrl+B to toggle sidebar
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "b") {
        e.preventDefault();
        setCollapsed((c) => !c);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const sidebarWidth = collapsed ? "var(--sidebar-collapsed)" : "var(--sidebar-width)";

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      {/* Sidebar */}
      <aside style={{
        width: sidebarWidth,
        minWidth: sidebarWidth,
        background: "var(--bg-sidebar)",
        borderRight: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        transition: "width var(--transition), min-width var(--transition)",
        position: "relative",
        zIndex: 10,
      }}>
        {/* Logo + Collapse */}
        <div style={{
          padding: collapsed ? "16px 12px" : "20px 20px 16px",
          display: "flex", alignItems: "center", gap: 10,
          justifyContent: collapsed ? "center" : "flex-start",
        }}>
          <div style={{
            width: 36, height: 36, borderRadius: "var(--radius-md)",
            background: "var(--brand-gradient)",
            display: "flex", alignItems: "center", justifyContent: "center",
            flexShrink: 0,
            boxShadow: "var(--shadow-brand)",
          }}>
            <BookOpen size={20} color="white" strokeWidth={2.2} />
          </div>
          {!collapsed && (
            <div style={{ overflow: "hidden" }}>
              <div style={{ fontWeight: 800, fontSize: 15, letterSpacing: -0.03, color: "var(--text)" }}>Edify AI</div>
              <div style={{ fontSize: 10, color: "var(--text-faint)", marginTop: -1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                Turn Knowledge Into Understanding
              </div>
            </div>
          )}
        </div>

        {/* New Study button */}
        <div style={{ padding: collapsed ? "4px 8px" : "0 16px 8px" }}>
          <button
            onClick={() => navigate("/")}
            className="btn btn-primary"
            style={{
              width: "100%",
              justifyContent: "center",
              padding: collapsed ? "10px" : "10px",
            }}
            title="New Study"
          >
            <Plus size={18} />
            {!collapsed && <span>New Study</span>}
          </button>
        </div>

        {/* Nav */}
        <nav style={{ padding: "4px 12px", flex: 1, overflowY: "auto" }} className="no-scrollbar">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === "/"}
              className={({ isActive }) => `nav-item ${isActive ? "nav-item-active" : ""}`}
              style={{ justifyContent: collapsed ? "center" : "flex-start" }}
              title={collapsed ? item.label : undefined}
            >
              <item.icon size={18} style={{ flexShrink: 0 }} />
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          ))}

          {/* Tools section */}
          {!collapsed && <div className="nav-section-label">Study Tools</div>}
          {toolItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `nav-item ${isActive ? "nav-item-active" : ""}`}
              style={{ justifyContent: collapsed ? "center" : "flex-start" }}
              title={collapsed ? item.label : undefined}
            >
              <item.icon size={18} style={{ flexShrink: 0 }} />
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          ))}

          {/* Projects */}
          {projects.length > 0 && (
            <>
              {!collapsed && <div className="nav-section-label" style={{ marginTop: 8 }}>Recent Projects</div>}
              {projects.slice(0, 8).map((project) => (
                <NavLink
                  key={project.id}
                  to={`/project/${project.id}`}
                  className={({ isActive }) => `nav-item ${isActive ? "nav-item-active" : ""}`}
                  style={{ justifyContent: collapsed ? "center" : "flex-start" }}
                  title={collapsed ? project.name : undefined}
                >
                  <div style={{ width: 8, height: 8, borderRadius: "50%", background: project.color, flexShrink: 0 }} />
                  {!collapsed && (
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {project.name}
                    </span>
                  )}
                </NavLink>
              ))}
            </>
          )}
        </nav>

        {/* Footer */}
        <div style={{
          padding: collapsed ? "8px" : "8px 16px",
          borderTop: "1px solid var(--border)",
          display: "flex",
          flexDirection: collapsed ? "column" : "row",
          gap: 4,
          alignItems: "center",
        }}>
          <button
            onClick={() => navigate("/settings")}
            className="nav-item tooltip"
            data-tooltip="Settings"
            style={{ justifyContent: collapsed ? "center" : "flex-start", margin: 0, flex: 1 }}
          >
            <Settings size={18} style={{ flexShrink: 0 }} />
            {!collapsed && <span>Settings</span>}
          </button>
          <button
            onClick={() => window.open("https://github.com/arcange9/EdifyAI", "_blank")}
            className="nav-item tooltip"
            data-tooltip="Help & Feedback"
            style={{ justifyContent: collapsed ? "center" : "flex-start", margin: 0, flex: 1 }}
          >
            <HelpCircle size={18} style={{ flexShrink: 0 }} />
            {!collapsed && <span>Help</span>}
          </button>
        </div>

        {/* Provider status indicator */}
        {!collapsed && (
          <div style={{ padding: "0 16px 8px" }}>
            <div style={{
              padding: "8px 12px", borderRadius: "var(--radius-sm)",
              background: activeProvider ? "var(--success-bg)" : "var(--surface-3)",
              display: "flex", alignItems: "center", gap: 8, fontSize: 11,
            }}>
              <span className={`status-dot ${activeProvider ? "success" : "neutral"}`} />
              <span style={{ color: activeProvider ? "var(--success)" : "var(--text-muted)", fontWeight: 600 }}>
                {activeProvider ? "AI Connected" : "No Provider"}
              </span>
            </div>
          </div>
        )}

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          style={{
            display: "flex", alignItems: "center", justifyContent: "center",
            padding: "8px", background: "transparent", border: "none",
            cursor: "pointer", color: "var(--text-faint)",
            transition: "color var(--transition)",
            borderTop: "1px solid var(--border)",
          }}
          onMouseEnter={(e) => e.currentTarget.style.color = "var(--text)"}
          onMouseLeave={(e) => e.currentTarget.style.color = "var(--text-faint)"}
          title={collapsed ? "Expand sidebar (Ctrl+B)" : "Collapse sidebar (Ctrl+B)"}
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </aside>

      {/* Main content */}
      <main style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column", minWidth: 0 }}>
        <Outlet />
      </main>
    </div>
  );
}
