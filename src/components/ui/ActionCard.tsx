import type { LucideIcon } from "lucide-react";

export function ActionCard({ icon: Icon, title, description, onClick, color }: {
  icon: LucideIcon;
  title: string;
  description: string;
  onClick: () => void;
  color?: string;
}) {
  return (
    <button
      className="card card-hover"
      onClick={onClick}
      style={{
        padding: 20, cursor: "pointer", textAlign: "left",
        border: "1px solid var(--border)", display: "flex", flexDirection: "column", gap: 10,
      }}
    >
      <div style={{
        width: 44, height: 44, borderRadius: "var(--radius-md)",
        display: "flex", alignItems: "center", justifyContent: "center",
        background: `color-mix(in srgb, ${color || "var(--accent)"} 12%, transparent)`,
      }}>
        <Icon size={22} style={{ color: color || "var(--accent)" }} />
      </div>
      <div>
        <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 2 }}>{title}</div>
        <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{description}</div>
      </div>
    </button>
  );
}
