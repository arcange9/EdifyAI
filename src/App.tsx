import { Navigate, Route, Routes } from "react-router-dom";
import { AppProvider, useApp } from "./lib/app-context";
import AppShell from "./components/AppShell";
import Dashboard from "./pages/Dashboard";
import Onboarding from "./pages/Onboarding";
import Settings from "./pages/Settings";
import ProjectView from "./pages/ProjectView";
import ChatView from "./pages/ChatView";
import Library from "./pages/Library";
import AITutor from "./pages/AITutor";
import StudyPlans from "./pages/StudyPlans";
import { UpdateChecker } from "./components/ui/UpdateChecker";
import { Loader2 } from "lucide-react";

function AppRoutes() {
  const { ready, preferences } = useApp();

  if (!ready) {
    return (
      <div style={{ display: "flex", height: "100vh", alignItems: "center", justifyContent: "center", gap: 12, background: "var(--bg)", color: "var(--text-muted)" }}>
        <Loader2 size={24} className="animate-spin" style={{ color: "var(--accent)" }} />
        <span style={{ fontSize: 16, fontWeight: 600 }}>Loading Edify AI...</span>
      </div>
    );
  }

  if (!preferences.onboarded) {
    return (
      <Routes>
        <Route path="*" element={<Onboarding />} />
      </Routes>
    );
  }

  return (
    <>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/library" element={<Library />} />
          <Route path="/tutor" element={<AITutor />} />
          <Route path="/study-plans" element={<StudyPlans />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/project/:id" element={<ProjectView />} />
          <Route path="/project/:id/chat" element={<ChatView />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <UpdateChecker />
    </>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppRoutes />
    </AppProvider>
  );
}
