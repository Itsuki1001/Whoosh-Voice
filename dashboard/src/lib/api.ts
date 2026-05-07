const BASE = import.meta.env.VITE_API_URL;

export async function fetchConversations() {
  const res = await fetch(`${BASE}/conversations`);
  if (!res.ok) throw new Error("Failed to fetch conversations");
  const data = await res.json();
  // Capitalise status to match UI expectations
  return data.map((c: any) => ({
    ...c,
    status: c.status.charAt(0).toUpperCase() + c.status.slice(1),
    messages: [], // placeholder — loaded on demand
  }));
}

export async function fetchConversation(id: string) {
  const res = await fetch(`${BASE}/conversations/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error("Failed to fetch conversation");
  const c = await res.json();
  return {
    ...c,
    status: c.status.charAt(0).toUpperCase() + c.status.slice(1),
  };
}

export async function fetchFlags() {
  const res = await fetch(`${BASE}/conversations/flags`);
  if (!res.ok) throw new Error("Failed to fetch flags");
  return res.json(); // { whatsapp_agent: true, voice_agent: true }
}

export async function toggleFlag(key: string, enabled: boolean) {
  const res = await fetch(`${BASE}/conversations/flags/${key}?enabled=${enabled}`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to toggle flag");
  return res.json();
}

export async function sendStaffReply(conversationId: string, message: string) {
  const res = await fetch(`${BASE}/conversations/${conversationId}/reply`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!res.ok) throw new Error("Failed to send");
  return res.json();
}