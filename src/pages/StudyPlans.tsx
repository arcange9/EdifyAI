import { useState, useEffect } from "react";
import { useApp } from "../lib/app-context";
import { db } from "../lib/db";
import { Calendar, Plus, Clock, Target, CheckCircle2, Circle, Loader2, Sparkles } from "lucide-react";
import type { StudyPlan } from "../lib/types";
import { EmptyState } from "../components/ui/EmptyState";

export default function StudyPlans() {
  const { projects, activeProvider } = useApp();
  const [plans, setPlans] = useState<StudyPlan[]>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [subject, setSubject] = useState("");
  const [goal, setGoal] = useState("");
  const [days, setDays] = useState("7");
  const [minutes, setMinutes] = useState("60");
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    (async () => {
      const all: StudyPlan[] = [];
      for (const p of projects) {
        const pPlans = await db.getAll<StudyPlan>("studyPlans");
        all.push(...pPlans.filter(sp => sp.projectId === p.id));
      }
      setPlans(all);
    })();
  }, [projects]);

  async function toggleTask(planId: string, taskId: string) {
    const plan = plans.find(p => p.id === planId);
    if (!plan) return;
    const updatedTasks = plan.tasks.map(t =>
      t.id === taskId ? { ...t, completed: !t.completed } : t
    );
    const updatedPlan = { ...plan, tasks: updatedTasks };
    await db.put("studyPlans", updatedPlan);
    setPlans(plans.map(p => p.id === planId ? updatedPlan : p));
  }

  async function generatePlan() {
    if (!activeProvider || !projects.length) return;
    setGenerating(true);
    try {
      const result = await activeProvider.generateStudyPlan({
        subject, goal, days: parseInt(days), minutesPerDay: parseInt(minutes), difficulty: "medium",
      });
      const plan = result as Record<string, unknown>;
      const studyPlan: StudyPlan = {
        id: crypto.randomUUID(),
        projectId: projects[0].id,
        subject,
        goal,
        daysPerWeek: parseInt(days),
        minutesPerDay: parseInt(minutes),
        difficulty: "medium",
        tasks: (plan.tasks as StudyPlan["tasks"]) || [],
        createdAt: Date.now(),
      };
      await db.put("studyPlans", studyPlan);
      setPlans([...plans, studyPlan]);
      setShowCreate(false);
      setSubject(""); setGoal("");
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to generate plan");
    }
    setGenerating(false);
  }

  return (
    <div className="scroll-container" style={{ padding: "32px 40px" }}>
      <div style={{ maxWidth: "var(--max-content)", margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
          <div>
            <h1 style={{ fontSize: 24, fontWeight: 800 }}>Study Plans</h1>
            <p style={{ fontSize: 14, color: "var(--text-muted)", marginTop: 4 }}>Plan your learning journey with AI-powered study schedules.</p>
          </div>
          <button className="btn btn-primary" onClick={() => setShowCreate(true)} disabled={!activeProvider}>
            <Plus size={16} /> Create Plan
          </button>
        </div>

        {!activeProvider && (
          <div className="card" style={{ padding: 16, marginBottom: 20, border: "1px solid var(--warning-light)", background: "var(--warning-bg)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <Sparkles size={18} style={{ color: "var(--warning)" }} />
              <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>
                Configure an AI provider in Settings to create study plans.
              </span>
            </div>
          </div>
        )}

        {plans.length === 0 ? (
          <EmptyState
            icon={Calendar}
            title="No study plans yet"
            description="Create an AI-powered study plan tailored to your goals, timeline, and availability."
            action={activeProvider && <button className="btn btn-primary" onClick={() => setShowCreate(true)}><Plus size={16} /> Create Your First Plan</button>}
          />
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {plans.map((plan) => {
              const completedTasks = plan.tasks.filter(t => t.completed).length;
              const progress = plan.tasks.length > 0 ? Math.round((completedTasks / plan.tasks.length) * 100) : 0;
              return (
                <div key={plan.id} className="card" style={{ padding: 20 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                    <div>
                      <h3 style={{ fontSize: 17, fontWeight: 700 }}>{plan.subject}</h3>
                      <div style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 2 }}>{plan.goal}</div>
                    </div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <span className="badge badge-brand"><Target size={11} /> {progress}% complete</span>
                      <span className="badge badge-neutral"><Clock size={11} /> {plan.minutesPerDay}m/day</span>
                    </div>
                  </div>
                  {/* Progress bar */}
                  <div style={{ height: 4, borderRadius: 2, background: "var(--surface-3)", marginBottom: 16 }}>
                    <div style={{ height: "100%", borderRadius: 2, background: "var(--accent)", width: `${progress}%`, transition: "width var(--transition)" }} />
                  </div>
                  {/* Tasks */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    {plan.tasks.slice(0, 8).map((task) => (
                      <div
                        key={task.id}
                        onClick={() => toggleTask(plan.id, task.id)}
                        style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", borderRadius: "var(--radius-sm)", background: "var(--surface-2)", cursor: "pointer", transition: "background var(--transition)" }}
                        onMouseEnter={(e) => e.currentTarget.style.background = "var(--bg-hover)"}
                        onMouseLeave={(e) => e.currentTarget.style.background = "var(--surface-2)"}
                      >
                        {task.completed ? <CheckCircle2 size={18} style={{ color: "var(--success)", flexShrink: 0 }} /> : <Circle size={18} style={{ color: "var(--text-faint)", flexShrink: 0 }} />}
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: 13, fontWeight: 500, textDecoration: task.completed ? "line-through" : "none", color: task.completed ? "var(--text-muted)" : "var(--text)" }}>
                            {task.topic}
                          </div>
                          {task.objectives.length > 0 && (
                            <div style={{ fontSize: 11, color: "var(--text-faint)", marginTop: 2 }}>
                              {task.objectives.join(" · ")}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                  {plan.tasks.length > 8 && (
                    <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 8, textAlign: "center" }}>
                      + {plan.tasks.length - 8} more tasks
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Create modal */}
      {showCreate && (
        <div className="modal-overlay" onClick={() => setShowCreate(false)}>
          <div className="modal" style={{ width: 460 }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Create Study Plan</h3>
              <button className="btn btn-ghost btn-icon" onClick={() => setShowCreate(false)}>✕</button>
            </div>
            <div className="modal-body">
              <div style={{ marginBottom: 16 }}>
                <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Subject</label>
                <input className="input" placeholder="e.g., Machine Learning" value={subject} onChange={(e) => setSubject(e.target.value)} />
              </div>
              <div style={{ marginBottom: 16 }}>
                <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Goal</label>
                <input className="input" placeholder="e.g., Pass the final exam" value={goal} onChange={(e) => setGoal(e.target.value)} />
              </div>
              <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
                <div style={{ flex: 1 }}>
                  <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Days</label>
                  <input className="input" type="number" value={days} onChange={(e) => setDays(e.target.value)} />
                </div>
                <div style={{ flex: 1 }}>
                  <label style={{ fontSize: 13, fontWeight: 600, display: "block", marginBottom: 6 }}>Min/Day</label>
                  <input className="input" type="number" value={minutes} onChange={(e) => setMinutes(e.target.value)} />
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-ghost" onClick={() => setShowCreate(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={generatePlan} disabled={!subject.trim() || !goal.trim() || generating}>
                {generating ? <><Loader2 size={16} className="animate-spin" /> Generating...</> : <><Sparkles size={16} /> Generate Plan</>}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
