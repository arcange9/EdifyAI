import type { ReactNode } from "react";
import { AlertCircle } from "lucide-react";

export function ErrorState({ title, description, action }: {
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="error-state">
      <div className="error-state-icon">
        <AlertCircle size={28} />
      </div>
      <div className="error-state-title">{title}</div>
      <div className="error-state-desc">{description}</div>
      {action && <div style={{ display: "flex", gap: 8, marginTop: 8 }}>{action}</div>}
    </div>
  );
}
