import { useState, useEffect, useRef } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { cn } from "@/lib/utils";
import { Search, Phone, Send } from "lucide-react";
import { fetchConversations, fetchConversation, fetchFlags, toggleFlag, sendStaffReply } from "@/lib/api";

type Source = "whatsapp" | "voice" | "instagram";
type Message = { id: number; from: "guest" | "agent"; text: string; time: string };

type ConversationSummary = {
  id: string;
  name: string;
  initials: string;
  source: Source;
  status: "hot" | "warm" | "cold";
  preview: string;
  time: string;
  color: string;
};

type Conversation = ConversationSummary & {
  messages: Message[];
};

const statusStyles: Record<string, string> = {
  hot: "bg-hot-bg text-hot",
  warm: "bg-warm-bg text-warm",
  cold: "bg-cold-bg text-cold",
};

const statusLabel: Record<string, string> = {
  hot: "Hot",
  warm: "Warm",
  cold: "Cold",
};

const SourceIcon = ({ s }: { s: Source }) => {
  if (s === "whatsapp")
    return (
      <span className="inline-flex h-5 w-5 items-center justify-center rounded text-emerald-500">
        <svg viewBox="0 0 24 24" className="h-4 w-4 fill-current">
          <path d="M20.5 3.5A11 11 0 0 0 3.7 17.3L2 22l4.8-1.7A11 11 0 1 0 20.5 3.5Zm-3 13.9c-.3-.2-1.7-.8-1.9-.9-.3-.1-.5-.2-.7.2s-.8.9-1 1.1c-.2.2-.4.2-.7.1-.3-.2-1.2-.5-2.3-1.4-.9-.8-1.5-1.7-1.6-2-.2-.3 0-.5.1-.6l.5-.6.2-.4c.1-.2 0-.3 0-.5l-.7-1.7c-.2-.4-.4-.4-.6-.4h-.5c-.2 0-.5.1-.7.4s-.9.9-.9 2.2.9 2.6 1.1 2.8c.1.2 1.9 2.9 4.6 4 2.7 1 2.7.7 3.2.7.5 0 1.7-.7 1.9-1.4.2-.7.2-1.3.2-1.4-.1-.1-.3-.2-.6-.4Z" />
        </svg>
      </span>
    );
  if (s === "voice") return <Phone className="h-4 w-4 text-blue-500" />;
  return null;
};

type Tab = "all" | "whatsapp" | "voice";

const TABS: { id: Tab; label: string }[] = [
  { id: "all", label: "All" },
  { id: "whatsapp", label: "WhatsApp" },
  { id: "voice", label: "Voice" },
];

const Conversations = () => {
  const [tab, setTab] = useState<Tab>("all");
  const [search, setSearch] = useState("");
  const [activeId, setActiveId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversation, setActiveConversation] = useState<Conversation | null>(null);
  const [listLoading, setListLoading] = useState(true);
  const [chatLoading, setChatLoading] = useState(false);
  const [flags, setFlags] = useState<Record<string, boolean>>({});
  const [replyText, setReplyText] = useState("");
  const [sending, setSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeConversation?.messages]);

  // Load conversations on mount
  useEffect(() => {
    fetchConversations()
      .then((data) => {
        setConversations(data);
        if (data.length > 0) setActiveId(data[0].id);
      })
      .catch(console.error)
      .finally(() => setListLoading(false));
  }, []);

  // Load flags on mount
  useEffect(() => {
    fetchFlags().then(setFlags).catch(console.error);
  }, []);

  // Fetch full messages whenever activeId changes
  useEffect(() => {
    if (!activeId) return;
    setChatLoading(true);
    setActiveConversation(null);
    setReplyText("");
    setSendError(null);
    fetchConversation(activeId)
      .then(setActiveConversation)
      .catch(console.error)
      .finally(() => setChatLoading(false));
  }, [activeId]);

  const handleToggle = async (key: string) => {
    const newVal = !flags[key];
    setFlags((prev) => ({ ...prev, [key]: newVal }));
    await toggleFlag(key, newVal).catch(console.error);
  };

  const handleSend = async () => {
    if (!replyText.trim() || !activeId || sending) return;
    setSending(true);
    setSendError(null);
    try {
      await sendStaffReply(activeId, replyText.trim());
      setReplyText("");
      // Refresh messages so sent message appears
      fetchConversation(activeId).then(setActiveConversation);
    } catch (e) {
      setSendError("Failed to send. Please try again.");
    } finally {
      setSending(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const visible = conversations.filter((c) => tab === "all" || c.source === tab);
  const filtered = visible.filter((c) =>
    c.name.toLowerCase().includes(search.toLowerCase())
  );

  const counts = {
    all: conversations.length,
    whatsapp: conversations.filter((c) => c.source === "whatsapp").length,
    voice: conversations.filter((c) => c.source === "voice").length,
  };

  // Only show reply bar for whatsapp conversations
  const canReply = activeConversation?.source === "whatsapp";

  return (
    <DashboardLayout>
      {/* Header + toggles */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Conversations</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Chat history from your WhatsApp and voice agents.
          </p>
        </div>

        <div className="flex items-center gap-6">
          {[
            { key: "whatsapp_agent", label: "WhatsApp" },
            { key: "voice_agent", label: "Voice" },
          ].map(({ key, label }) => (
            <div key={key} className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">{label}</span>
              <button
                onClick={() => handleToggle(key)}
                className={cn(
                  "relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200",
                  flags[key] ? "bg-primary" : "bg-muted"
                )}
              >
                <span
                  className={cn(
                    "inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform duration-200",
                    flags[key] ? "translate-x-6" : "translate-x-1"
                  )}
                />
              </button>
              <span
                className={cn(
                  "text-xs font-medium",
                  flags[key] ? "text-primary" : "text-muted-foreground"
                )}
              >
                {flags[key] ? "On" : "Off"}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div className="mb-4 inline-flex rounded-lg border border-border bg-card p-1">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={cn(
              "flex items-center gap-2 rounded-md px-4 py-1.5 text-sm font-medium transition-colors",
              tab === t.id
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            {t.id === "whatsapp" && <SourceIcon s="whatsapp" />}
            {t.id === "voice" && <Phone className="h-4 w-4" />}
            {t.label}
            <span
              className={cn(
                "rounded px-1.5 text-xs",
                tab === t.id ? "bg-primary-foreground/20" : "bg-secondary"
              )}
            >
              {counts[t.id]}
            </span>
          </button>
        ))}
      </div>

      {/* Main grid */}
      <div className="grid h-[calc(100vh-12rem)] grid-cols-1 gap-4 rounded-xl border border-border bg-card lg:grid-cols-[320px_1fr]">

        {/* Conversation list */}
        <div className="flex flex-col border-r border-border">
          <div className="p-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                placeholder="Search guests..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full rounded-lg border border-border bg-background py-2 pl-9 pr-3 text-sm outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
          </div>

          <ul className="flex-1 overflow-y-auto">
            {listLoading && (
              <li className="px-4 py-8 text-center text-sm text-muted-foreground">
                Loading conversations...
              </li>
            )}
            {!listLoading && filtered.length === 0 && (
              <li className="px-4 py-8 text-center text-sm text-muted-foreground">
                No conversations found.
              </li>
            )}
            {filtered.map((c) => (
              <li key={c.id}>
                <button
                  onClick={() => setActiveId(c.id)}
                  className={cn(
                    "flex w-full items-start gap-3 border-l-2 px-4 py-3 text-left transition-colors",
                    activeId === c.id
                      ? "border-primary bg-accent/40"
                      : "border-transparent hover:bg-secondary"
                  )}
                >
                  <div
                    className={cn(
                      "flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-xs font-semibold",
                    )}
                    style={{ background: c.color }}
                  >
                    {c.initials}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-sm font-medium">{c.name}</span>
                      <span className="text-xs text-muted-foreground">{c.time}</span>
                    </div>
                    <div className="mt-0.5 truncate text-xs text-muted-foreground">
                      {c.preview}
                    </div>
                    <div className="mt-1 flex items-center gap-2">
                      <SourceIcon s={c.source} />
                      <span
                        className={cn(
                          "rounded px-1.5 py-0.5 text-[10px] font-medium",
                          statusStyles[c.status]
                        )}
                      >
                        {statusLabel[c.status]}
                      </span>
                    </div>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </div>

        {/* Chat panel */}
        <div className="flex min-h-0 flex-col">
          {chatLoading && (
            <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
              Loading messages...
            </div>
          )}

          {!chatLoading && !activeConversation && (
            <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
              {listLoading ? "Loading..." : "Select a conversation"}
            </div>
          )}

          {!chatLoading && activeConversation && (
            <>
              {/* Chat header */}
              <div className="flex items-center justify-between border-b border-border px-5 py-3">
                <div className="flex items-center gap-3">
                  <div
                    className="flex h-9 w-9 items-center justify-center rounded-full text-xs font-semibold"
                    style={{ background: activeConversation.color }}
                  >
                    {activeConversation.initials}
                  </div>
                  <div>
                    <div className="text-sm font-semibold">{activeConversation.name}</div>
                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                      <SourceIcon s={activeConversation.source} />
                      <span className="capitalize">{activeConversation.source} agent</span>
                    </div>
                  </div>
                </div>
                <span
                  className={cn(
                    "rounded-md px-2 py-0.5 text-xs font-medium",
                    statusStyles[activeConversation.status]
                  )}
                >
                  {statusLabel[activeConversation.status]}
                </span>
              </div>

              {/* Messages */}
              <div className="flex-1 space-y-3 overflow-y-auto bg-background p-5">
                {activeConversation.messages.length === 0 && (
                  <p className="text-center text-sm text-muted-foreground">
                    No messages yet.
                  </p>
                )}
                {activeConversation.messages.map((m) => (
                  <div
                    key={m.id}
                    className={cn(
                      "flex",
                      m.from === "agent" ? "justify-end" : "justify-start"
                    )}
                  >
                    <div
                      className={cn(
                        "max-w-[70%] rounded-2xl px-4 py-2 text-sm shadow-sm",
                        m.from === "agent"
                          ? "rounded-br-sm bg-primary text-primary-foreground"
                          : "rounded-bl-sm border border-border bg-card"
                      )}
                    >
                      <div>{m.text}</div>
                      {m.time && (
                        <div
                          className={cn(
                            "mt-1 text-[10px]",
                            m.from === "agent"
                              ? "text-primary-foreground/70"
                              : "text-muted-foreground"
                          )}
                        >
                          {m.time}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>

              {/* Input bar */}
              <div className="border-t border-border p-3">
                {sendError && (
                  <p className="mb-2 text-xs text-destructive">{sendError}</p>
                )}
                {canReply ? (
                  <div className={cn(
                    "flex items-center gap-2 rounded-lg border bg-background px-3 py-2 transition-colors",
                    sending ? "border-border opacity-60" : "border-border focus-within:ring-2 focus-within:ring-ring"
                  )}>
                    <input
                      placeholder="Reply as staff... (Enter to send)"
                      value={replyText}
                      onChange={(e) => setReplyText(e.target.value)}
                      onKeyDown={handleKeyDown}
                      disabled={sending}
                      className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
                    />
                    <button
                      onClick={handleSend}
                      disabled={sending || !replyText.trim()}
                      className={cn(
                        "transition-colors",
                        replyText.trim() && !sending
                          ? "text-primary hover:text-primary/80"
                          : "text-muted-foreground"
                      )}
                    >
                      <Send className="h-4 w-4" />
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-2 opacity-50">
                    <input
                      placeholder="Replies only available for WhatsApp conversations"
                      disabled
                      className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
                    />
                    <Send className="h-4 w-4 text-muted-foreground" />
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
};

export default Conversations;