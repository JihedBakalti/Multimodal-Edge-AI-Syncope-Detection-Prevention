const API_BASE = import.meta.env.VITE_PLATFORM_API_BASE || "http://127.0.0.1:8100/api";
const AUTH_STORAGE_KEY = "platform_basic_token";

function queryString(params) {
  const u = new URLSearchParams();
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      u.set(key, String(value));
    }
  });
  const s = u.toString();
  return s ? `?${s}` : "";
}

async function http(path, options = {}) {
  const token = localStorage.getItem(AUTH_STORAGE_KEY);
  const authHeader = token ? { Authorization: `Basic ${token}` } : {};
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...authHeader, ...(options.headers || {}) },
    ...options,
  });
  const raw = await response.text();
  let data = null;
  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch {
      data = raw;
    }
  }
  if (!response.ok) {
    const detail =
      (data && typeof data === "object" && (data.detail || data.error || data.message)) ||
      `Request failed (${response.status})`;
    const error = new Error(String(detail));
    error.status = response.status;
    error.payload = data;
    throw error;
  }
  return data ?? {};
}

export const api = {
  login: async (email, password) => {
    const data = await http("/accounts/login/", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    localStorage.setItem(AUTH_STORAGE_KEY, data.token);
    return data;
  },
  logout: async () => {
    try {
      await http("/accounts/logout/", { method: "POST" });
    } finally {
      localStorage.removeItem(AUTH_STORAGE_KEY);
    }
  },
  hasToken: () => Boolean(localStorage.getItem(AUTH_STORAGE_KEY)),
  me: () => http("/accounts/me/"),
  assignments: () => http("/accounts/assignments/"),
  createPatientAccount: (payload) =>
    http("/accounts/patients/create/", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updatePatientAccount: (patientId, payload) =>
    http(`/accounts/patients/${patientId}/`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  vitals: (params) => http(`/clinical/vitals/${queryString(params)}`),
  voiceChecks: (params) => http(`/clinical/voice-checks/${queryString(params)}`),
  triggers: (params) => http(`/clinical/triggers/${queryString(params)}`),
  incidents: (params) => http(`/clinical/incidents/${queryString(params)}`),
  alerts: (params) => http(`/clinical/alerts/${queryString(params)}`),
  patientSummary: (patientId) => http(`/clinical/patients/${patientId}/summary/`),
  conversations: () => http("/messaging/conversations/"),
  createConversation: (payload) =>
    http("/messaging/conversations/", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  messages: (conversationId) => http(`/messaging/conversations/${conversationId}/messages/`),
  sendMessage: (conversationId, body) =>
    http(`/messaging/conversations/${conversationId}/messages/`, {
      method: "POST",
      body: JSON.stringify({ body }),
    }),
  markConversationRead: (conversationId) =>
    http(`/messaging/conversations/${conversationId}/mark-read/`, {
      method: "POST",
    }),
  reports: () => http("/reports/"),
  createReport: (payload) =>
    http("/reports/", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
