import { useEffect, useMemo, useState } from "react";
import { api } from "../services/api";
import { connectConversationSocket } from "../services/socket";

function toConversationMap(conversations = []) {
  const map = {};
  conversations.forEach((conversation) => {
    map[conversation.id] = {
      ...conversation,
      unread_count: conversation.unread_count || 0,
    };
  });
  return map;
}

function formatCounterpart(conversation) {
  const counterpart = conversation?.counterpart;
  if (!counterpart) return "Unknown user";
  return counterpart.email || counterpart.username || `User ${counterpart.id}`;
}

function shortName(value = "") {
  return String(value).trim().charAt(0).toUpperCase() || "?";
}

export default function ChatPanel({ me, assignments = [] }) {
  const [conversationId, setConversationId] = useState(null);
  const [conversationsById, setConversationsById] = useState({});
  const [messagesByConversation, setMessagesByConversation] = useState({});
  const [text, setText] = useState("");
  const [recipient, setRecipient] = useState("");
  const [loading, setLoading] = useState(true);
  const [sendState, setSendState] = useState("");

  function upsertIncomingMessage(message, activeConversationId) {
    if (!message?.id || !message?.conversation) return;
    setMessagesByConversation((prev) => {
      const existing = prev[message.conversation] || [];
      const foundIdx = existing.findIndex((item) => Number(item.id) === Number(message.id));
      if (foundIdx >= 0) {
        const next = [...existing];
        next[foundIdx] = message;
        return { ...prev, [message.conversation]: next };
      }
      return { ...prev, [message.conversation]: [...existing, message] };
    });

    setConversationsById((prev) => {
      const current = prev[message.conversation];
      if (!current) return prev;
      const isCurrentConversation = Number(activeConversationId) === Number(message.conversation);
      const isInbound = Number(message.sender) !== Number(me?.id);
      const unreadCount = isCurrentConversation ? 0 : (current.unread_count || 0) + (isInbound ? 1 : 0);
      return {
        ...prev,
        [message.conversation]: {
          ...current,
          unread_count: unreadCount,
          last_message: message,
        },
      };
    });
  }

  function applyReadReceipt(eventPayload) {
    const convId = Number(eventPayload?.conversation_id);
    const readAt = eventPayload?.read_at;
    const ids = new Set((eventPayload?.message_ids || []).map((id) => Number(id)));
    if (!convId || !ids.size) return;
    setMessagesByConversation((prev) => {
      const source = prev[convId] || [];
      return {
        ...prev,
        [convId]: source.map((message) =>
          ids.has(Number(message.id)) ? { ...message, read_at: readAt, status: "read" } : message
        ),
      };
    });
  }

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      try {
        const convs = await api.conversations();
        if (cancelled) return;
        const map = toConversationMap(convs);
        setConversationsById(map);
        if (convs.length) {
          setConversationId((prev) => prev || convs[0].id);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load().catch(() => setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [me?.id]);

  useEffect(() => {
    if (!conversationId) return undefined;
    let ws = null;

    api.messages(conversationId).then((initial) => {
      setMessagesByConversation((prev) => ({ ...prev, [conversationId]: initial }));
    }).catch(() => {});

    ws = connectConversationSocket(conversationId, {
      onMessage: (payload) => {
        if (payload?.type === "message.created" && payload.message) {
          upsertIncomingMessage(payload.message, conversationId);
          const inbound = Number(payload.message.sender) !== Number(me?.id);
          if (inbound && Number(conversationId) === Number(payload.message.conversation)) {
            api.markConversationRead(payload.message.conversation).catch(() => {});
          }
        }
        if (payload?.type === "message.read") {
          applyReadReceipt(payload);
        }
      },
    });

    return () => {
      if (ws) ws.close();
    };
  }, [conversationId, me?.id]);

  const conversationList = useMemo(
    () => Object.values(conversationsById).sort((a, b) => new Date(b.created_at) - new Date(a.created_at)),
    [conversationsById]
  );

  const activeMessages = conversationId ? (messagesByConversation[conversationId] || []) : [];
  const activeConversation = conversationId ? conversationsById[conversationId] : null;
  const recipients = useMemo(() => {
    if (!me?.role) return [];
    if (me.role === "doctor") {
      return assignments.map((assignment) => ({
        key: String(assignment.patient.id),
        payload: { patient: assignment.patient.id },
        label: assignment.patient.user.email || assignment.patient.user.username || `Patient ${assignment.patient.id}`,
      }));
    }
    return assignments.map((assignment) => ({
      key: String(assignment.doctor.id),
      payload: { doctor: assignment.doctor.id },
      label: assignment.doctor.user.email || assignment.doctor.user.username || `Doctor ${assignment.doctor.id}`,
    }));
  }, [assignments, me?.role]);

  useEffect(() => {
    if (!recipient && recipients.length) {
      setRecipient(recipients[0].key);
    }
  }, [recipients, recipient]);

  async function onSend(event) {
    event.preventDefault();
    if (!conversationId || !text.trim()) return;
    setSendState("");
    const tempId = `temp-${Date.now()}`;
    const optimisticMessage = {
      id: tempId,
      conversation: conversationId,
      sender: me?.id,
      sender_email: me?.email,
      sender_username: me?.username,
      sender_role: me?.role,
      body: text,
      sent_at: new Date().toISOString(),
      delivered_at: null,
      read_at: null,
      status: "sent",
    };
    setMessagesByConversation((prev) => ({
      ...prev,
      [conversationId]: [...(prev[conversationId] || []), optimisticMessage],
    }));

    const textSnapshot = text;
    setText("");
    try {
      const saved = await api.sendMessage(conversationId, textSnapshot);
      setMessagesByConversation((prev) => {
        const source = prev[conversationId] || [];
        const replaced = source.map((message) => (message.id === tempId ? saved : message));
        const hasSaved = replaced.some((message) => Number(message.id) === Number(saved.id));
        return {
          ...prev,
          [conversationId]: hasSaved ? replaced.filter((message) => message.id !== tempId) : replaced,
        };
      });
      setConversationsById((prev) => ({
        ...prev,
        [conversationId]: {
          ...prev[conversationId],
          last_message: saved,
        },
      }));
    } catch {
      setSendState("Message failed to send.");
      setMessagesByConversation((prev) => ({
        ...prev,
        [conversationId]: (prev[conversationId] || []).filter((message) => message.id !== tempId),
      }));
      setText(textSnapshot);
    }
  }

  async function onSelectConversation(nextId) {
    setConversationId(nextId);
    setConversationsById((prev) => ({
      ...prev,
      [nextId]: {
        ...prev[nextId],
        unread_count: 0,
      },
    }));
    try {
      await api.markConversationRead(nextId);
    } catch {
      // no-op; UI already handles optimistic unread reset
    }
  }

  async function onStartConversation() {
    if (!recipient) {
      setSendState("No linked counterpart available for chat.");
      return;
    }
    const choice = recipients.find((item) => item.key === recipient);
    if (!choice) {
      setSendState("Selected recipient is invalid.");
      return;
    }
    try {
      const created = await api.createConversation(choice.payload);
      setConversationsById((prev) => ({
        ...prev,
        [created.id]: {
          ...created,
          unread_count: created.unread_count || 0,
        },
      }));
      setConversationId(created.id);
      setSendState("");
    } catch (error) {
      setSendState(error.message || "Could not start conversation.");
    }
  }

  return (
    <section className="panel messaging-panel">
      <div className="panel-head">
        <h3>Secure Messaging</h3>
        <span className="panel-count">{conversationList.length}</span>
      </div>
      <div className="messaging-start-row">
        <select value={recipient} onChange={(event) => setRecipient(event.target.value)}>
          {recipients.length === 0 && <option value="">No linked users</option>}
          {recipients.map((option) => (
            <option key={option.key} value={option.key}>{option.label}</option>
          ))}
        </select>
        <button type="button" className="ghost-button" onClick={onStartConversation}>
          Start chat
        </button>
      </div>
      <div className="messaging-layout">
        <aside className="conversation-list">
          {loading && <p className="muted timeline-empty">Loading conversations...</p>}
          {!loading && conversationList.length === 0 && (
            <p className="muted timeline-empty">No conversations yet. Start one above.</p>
          )}
          {conversationList.map((conversation) => (
            <button
              type="button"
              key={conversation.id}
              className={`conversation-item ${Number(conversationId) === Number(conversation.id) ? "active" : ""}`}
              onClick={() => onSelectConversation(conversation.id)}
            >
              <div className="conversation-item-head">
                <div className="conversation-peer">
                  <span className="conversation-avatar">{shortName(formatCounterpart(conversation))}</span>
                  <strong>{formatCounterpart(conversation)}</strong>
                </div>
                {conversation.unread_count > 0 && <span className="status-chip pending">{conversation.unread_count}</span>}
              </div>
              <p className="muted conversation-preview">
                {conversation.last_message?.body || "No messages yet."}
              </p>
            </button>
          ))}
        </aside>

        <div className="chat-thread">
          {!conversationId && <p className="muted timeline-empty">Select a conversation to view messages.</p>}
          {conversationId && (
            <>
              <div className="chat-thread-head">
                <div className="conversation-peer">
                  <span className="conversation-avatar">{shortName(formatCounterpart(activeConversation))}</span>
                  <div>
                    <strong>{formatCounterpart(activeConversation)}</strong>
                    <p className="muted chat-thread-subtitle">Active conversation</p>
                  </div>
                </div>
              </div>
              <div className="chat-messages">
                {activeMessages.length === 0 ? <p className="muted timeline-empty">No messages yet. Start a secure conversation.</p> : null}
                {activeMessages.map((message) => (
                  <div className={`chat-bubble ${Number(message.sender) === Number(me?.id) ? "outbound" : "inbound"}`} key={message.id}>
                    <div className="chat-bubble-head">
                      <strong>{Number(message.sender) === Number(me?.id) ? "You" : (message.sender_email || message.sender_username || "User")}</strong>
                      <span className="muted">{message.status || (message.read_at ? "read" : message.delivered_at ? "delivered" : "sent")}</span>
                    </div>
                    <p>{message.body}</p>
                  </div>
                ))}
              </div>
              <form onSubmit={onSend} className="chat-input-row">
                <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Type your message..." />
                <button type="submit">Send</button>
              </form>
            </>
          )}
        </div>
      </div>
      {sendState && (
        <p className="muted timeline-empty">
          {sendState}
        </p>
      )}
    </section>
  );
}
