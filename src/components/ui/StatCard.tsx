import type { LucideIcon } from "lucide-react";

export function StatCard({ icon: Icon, label, value, color }: {
  icon: LucideIcon;
  label: string;
  value: string | number;
  color: string;
}) {
  return (
    <div className="card" style={{ padding: 16, display: "flex", alignItems: "center", gap: 12 }}>
      <div style={{
        width: 40, height: 40, borderRadius: "var(--radius)",
        display: "flex", alignItems: "center", justifyContent: "center",
        background: `color-mix(in srgb, ${color} 12%, transparent)`,
      }}>
        <Icon size={20} style={{ color }} />
      </div>
      <div>
        <div style={{ fontSize: 22, fontWeight: 800, lineHeight: 1.1 }}>{value}</div>
        <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>{label}</div>
      </div>
    </div>
  );
}
