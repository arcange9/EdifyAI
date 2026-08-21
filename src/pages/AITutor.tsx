import { useState, useRef } from "react";
import { useApp } from "../lib/app-context";
import { v4 as uuid } from "uuid";
import type { ChatMessage } from "../lib/types";
import {
  GraduationCap, Send, Lightbulb, BookOpen,
  ListChecks, Square, PenLine, HelpCircle,
} from "lucide-react";
import { EmptyState } from "../components/ui/EmptyState";
import { marked } from "marked";

const TUTOR_PROJECT_ID = "edify-tutor-global";

export default function AITutor() {
  const { activeProvider } = useApp();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const quickActions = [
    { label: "Explain simply", icon: Lightbulb, prompt: "Explain this concept simply: ", desc: "Get a clear, simple explanation" },
    { label: "Give an example", icon: BookOpen, prompt: "Give me a practical example of: ", desc: "See it in action" },
    { label: "Teach me step-by-step", icon: GraduationCap, prompt: "Teach me step by step: ", desc: "Learn at your own pace" },
    { label: "Test me", icon: ListChecks, prompt: "Test my knowledge on: ", desc: "Check your understanding" },
    { label: "Practice questions", icon: PenLine, prompt: "Give me practice questions about: ", desc: "Sharpen your skills" },
    { label: "Ask a question", icon: HelpCircle, prompt: "", desc: "Ask anything you want to learn" },
  ];

  async function send(text: string) {
    if (!text.trim() || !activeProvider) return;
    const userMsg: ChatMessage = { id: uuid(), projectId: TUTOR_PROJECT_ID, role: "user", content: text, at: Date.now() };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setStreaming(true);
    setStreamText("");

    const controller = new AbortController();
    abortRef.current = controller;

    let fullText = "";
    try {
      await activeProvider.streamChat(
        {
          system: "You are Edify AI Tutor, a patient and encouraging teacher. Break down complex concepts into simple, understandable parts. Use examples, analogies, and step-by-step explanations. Ask questions to check understanding. Be warm and supportive. Use markdown formatting for clarity.",
          messages: [{ role: "user", content: text }],
          signal: controller.signal,
        },
        (token) => { fullText += token; setStreamText(fullText); }
      );
    } catch (err) {
      fullText = `Error: ${err instanceof Error ? err.message : "Failed"}`;
    }

    const aiMsg: ChatMessage = { id: uuid(), projectId: TUTOR_PROJECT_ID, role: "assistant", content: fullText, at: Date.now() };
    setMessages((m) => [...m, aiMsg]);
    setStreaming(false);
    setStreamText("");
  }

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {/* Header */}
      <div style={{
        padding: "16px 24px", borderBottom: "1px solid var(--border)",
        background: "var(--bg-elevated)", flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 40, height: 40, borderRadius: "var(--radius-md)", background: "var(--brand-gradient)",
            display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "var(--shadow-brand)",
          }}>
            <GraduationCap size={20} color="white" />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 17 }}>Edify Tutor</div>
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Learn anything, one concept at a time.</div>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="scroll-container" style={{ flex: 1, padding: "24px" }}>
        {messages.length === 0 && !streaming ? (
          <div style={{ maxWidth: 640, margin: "0 auto", paddingTop: 24 }}>
            <EmptyState
              icon={GraduationCap}
              title="Your personal AI Tutor"
              description="Ask me to explain any concept, break down complex topics, give you examples, or test your knowledge."
            />
            {/* Quick action grid */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 10, marginTop: 28 }}>
              {quickActions.map((qa) => (
                <button
                  key={qa.label}
                  className="card card-hover"
                  onClick={() => send(qa.prompt + (input || "a topic of your choice"))}
                  style={{ padding: 16, display: "flex", flexDirection: "column", gap: 6, textAlign: "left" }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{
                      width: 32, height: 32, borderRadius: "var(--radius-sm)",
                      background: "var(--accent-light)", display: "flex", alignItems: "center", justifyContent: "center",
                    }}>
                      <qa.icon size={16} style={{ color: "var(--accent)" }} />
                    </div>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>{qa.label}</span>
                  </div>
                  <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{qa.desc}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div style={{ maxWidth: 640, margin: "0 auto", display: "flex", flexDirection: "column", gap: 16 }}>
            {messages.map((msg) => (
              <div key={msg.id} className="fade-in" style={{ display: "flex", justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}>
                {msg.role === "user" ? (
                  <div className="chat-bubble-user">{msg.content}</div>
                ) : (
                  <div className="chat-bubble-ai">
                    <div dangerouslySetInnerHTML={{ __html: marked.parse(msg.content) as string }} />
                  </div>
                )}
              </div>
            ))}
            {streaming && (
              <div style={{ display: "flex", justifyContent: "flex-start" }}>
                <div className="chat-bubble-ai">
                  <div dangerouslySetInnerHTML={{ __html: marked.parse(streamText || "Thinking...") as string }} />
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Composer */}
      <div style={{
        padding: "12px 24px 16px", borderTop: "1px solid var(--border)",
        background: "var(--bg-elevated)", flexShrink: 0,
      }}>
        <div style={{ maxWidth: 640, margin: "0 auto", display: "flex", gap: 8 }}>
          <input
            className="input"
            placeholder="Ask your tutor anything..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send(input)}
            disabled={streaming}
            style={{ flex: 1 }}
          />
          {streaming ? (
            <button className="btn btn-ghost btn-icon" onClick={() => abortRef.current?.abort()} title="Stop">
              <Square size={16} />
            </button>
          ) : (
            <button className="btn btn-primary btn-icon" onClick={() => send(input)} disabled={!input.trim()} title="Send">
              <Send size={16} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
