import { useState, useEffect, useRef } from "react";
import { useParams } from "react-router-dom";
import { useApp } from "../lib/app-context";
import { db } from "../lib/db";
import { v4 as uuid } from "uuid";
import { buildRAGContext } from "../lib/rag";
import type { ChatMessage, Document } from "../lib/types";
import {
  Send, Square, MessageSquare, Paperclip, Sparkles,
} from "lucide-react";
import { EmptyState } from "../components/ui/EmptyState";
import { marked } from "marked";

export default function ChatView() {
  const { id } = useParams();
  const { activeProvider } = useApp();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [documents, setDocuments] = useState<Document[]>([]);
  const [projectName, setProjectName] = useState("");
  
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!id) return;
    (async () => {
      setMessages(await db.getChatMessages(id));
      setDocuments(await db.getDocuments(id));
      const proj = await db.get<{ name: string }>("projects", id);
      setProjectName(proj?.name || "Chat");
    })();
  }, [id]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, streamText]);

  async function send() {
    if (!input.trim() || !activeProvider || !id) return;
    const userMsg: ChatMessage = { id: uuid(), projectId: id, role: "user", content: input, at: Date.now() };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setStreaming(true);
    setStreamText("");

    const { context, citations } = buildRAGContext(input, documents);
    const controller = new AbortController();
    abortRef.current = controller;

    let fullText = "";
    try {
      await activeProvider.streamChat(
        {
          system: "You are Edify AI, a helpful study tutor. Answer questions using the provided context when available. Be clear and educational. Use markdown formatting.",
          messages: [{ role: "user", content: input }],
          context, signal: controller.signal,
        },
        (token) => { fullText += token; setStreamText(fullText); }
      );
    } catch (err) {
      fullText = `Error: ${err instanceof Error ? err.message : "Failed"}`;
    }

    const aiMsg: ChatMessage = { id: uuid(), projectId: id, role: "assistant", content: fullText, citations: citations.length ? citations : undefined, at: Date.now() };
    await db.put("chats", userMsg);
    await db.put("chats", aiMsg);
    setMessages((m) => [...m, aiMsg]);
    setStreaming(false);
    setStreamText("");
  }

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {/* Header */}
      <div style={{
        padding: "14px 24px", borderBottom: "1px solid var(--border)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "var(--bg-elevated)", flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 32, height: 32, borderRadius: "var(--radius)", background: "var(--brand-gradient)", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <MessageSquare size={16} color="white" />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 15 }}>{projectName}</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
              {documents.length > 0 ? `Using ${documents.length} document${documents.length > 1 ? "s" : ""} as context` : "No documents attached"}
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className="badge badge-brand"><Sparkles size={11} /> {activeProvider?.name || "No provider"}</span>
        </div>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="scroll-container" style={{ flex: 1, padding: "24px" }}>
        {messages.length === 0 && !streaming && (
          <EmptyState
            icon={MessageSquare}
            title="Ask Edify anything"
            description="Chat with your study materials. Ask questions, get explanations, or request summaries."
          />
        )}
        <div style={{ maxWidth: 720, margin: "0 auto", display: "flex", flexDirection: "column", gap: 16 }}>
          {messages.map((msg) => (
            <div key={msg.id} className="fade-in" style={{ display: "flex", justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}>
              {msg.role === "user" ? (
                <div className="chat-bubble-user">{msg.content}</div>
              ) : (
                <div className="chat-bubble-ai">
                  <div dangerouslySetInnerHTML={{ __html: marked.parse(msg.content) as string }} />
                  {msg.citations && msg.citations.length > 0 && (
                    <div style={{ marginTop: 10, paddingTop: 8, borderTop: "1px solid var(--border)", display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {msg.citations.map((c, i) => (
                        <span key={i} className="chat-citation" title={c.text}>
                          [{i + 1}] {c.documentTitle}
                        </span>
                      ))}
                    </div>
                  )}
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
      </div>

      {/* Composer */}
      <div style={{
        padding: "12px 24px 16px", borderTop: "1px solid var(--border)",
        background: "var(--bg-elevated)", flexShrink: 0,
      }}>
        <div style={{ maxWidth: 720, margin: "0 auto", display: "flex", gap: 8, alignItems: "flex-end" }}>
          <button className="btn btn-ghost btn-icon tooltip" data-tooltip="Attach document" disabled>
            <Paperclip size={18} />
          </button>
          <input
            className="input"
            placeholder="Ask Edify anything..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
            disabled={streaming}
            style={{ flex: 1 }}
          />
          {streaming ? (
            <button className="btn btn-ghost btn-icon" onClick={() => abortRef.current?.abort()} title="Stop">
              <Square size={16} />
            </button>
          ) : (
            <button className="btn btn-primary btn-icon" onClick={send} disabled={!input.trim()} title="Send">
              <Send size={16} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
