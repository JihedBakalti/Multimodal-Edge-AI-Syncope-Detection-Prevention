const WS_BASE = import.meta.env.VITE_PLATFORM_WS_BASE || "ws://127.0.0.1:8100";

export function connectConversationSocket(conversationId, handlers = {}) {
  const { onMessage, onOpen, onClose, onError } = handlers;
  const ws = new WebSocket(`${WS_BASE}/ws/chat/${conversationId}/`);
  ws.onopen = () => {
    if (onOpen) onOpen();
  };
  ws.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (onMessage) onMessage(payload);
    } catch (error) {
      if (onError) onError(error);
    }
  };
  ws.onerror = (error) => {
    if (onError) onError(error);
  };
  ws.onclose = () => {
    if (onClose) onClose();
  };
  return ws;
}
