import { Loader2 } from "lucide-react";

export function LoadingState({ text = "Loading..." }: { text?: string }) {
  return (
    <div className="loading-state">
      <Loader2 size={24} className="animate-spin" style={{ color: "var(--accent)" }} />
      <div className="loading-text">{text}</div>
    </div>
  );
}

export function SkeletonCard() {
  return (
    <div className="card" style={{ padding: 16 }}>
      <div className="skeleton" style={{ height: 12, width: "40%", marginBottom: 8 }} />
      <div className="skeleton" style={{ height: 20, width: "80%", marginBottom: 6 }} />
      <div className="skeleton" style={{ height: 12, width: "60%" }} />
    </div>
  );
}

export function SkeletonGrid({ count = 4 }: { count?: number }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 16 }}>
      {Array.from({ length: count }).map((_, i) => <SkeletonCard key={i} />)}
    </div>
  );
}
