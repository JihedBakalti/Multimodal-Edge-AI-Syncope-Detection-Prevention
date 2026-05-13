/**
 * Same live vitals source as `frontend/src/components/InterSenseSimulator.jsx`:
 * GET {VITE_API_BASE_URL}/api/live-vitals?user_id=&session_id=
 *
 * That endpoint (medical_assistant embedded API) merges Firebase RTDB with
 * vital_sample_session simulator state — identical effective_* fields as the simulator UI.
 */

export function assistantApiBaseFromEnv() {
  const b =
    import.meta.env.VITE_ORCHESTRATOR_API_BASE_URL ||
    import.meta.env.VITE_API_BASE_URL ||
    "";
  return String(b || "").replace(/\/+$/, "");
}

export async function fetchLiveVitalsAssistant(baseUrl, userId, sessionId) {
  const uid = String(userId ?? "").trim();
  const sid = String(sessionId ?? "default-session").trim() || "default-session";
  if (!baseUrl || !uid) {
    return { ok: false, message: "Missing assistant API base URL or Firebase user id." };
  }
  const q = new URLSearchParams({ user_id: uid, session_id: sid });
  try {
    const res = await fetch(`${baseUrl}/api/live-vitals?${q}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      return { ok: false, message: data?.detail || `HTTP ${res.status}`, raw: data };
    }
    if (!data?.ok) {
      return {
        ok: false,
        message: "No vitals yet (Firebase empty for this user and no simulator samples for this session).",
        raw: data,
      };
    }
    const hr = Number.isFinite(Number(data.effective_heart_rate)) ? Number(data.effective_heart_rate) : null;
    const spo2 = Number.isFinite(Number(data.effective_blood_oxygen)) ? Number(data.effective_blood_oxygen) : null;
    return {
      ok: true,
      heart_rate: hr,
      blood_oxygen: spo2,
      source: "/api/live-vitals",
      raw: data,
    };
  } catch (e) {
    return { ok: false, message: e?.message ? String(e.message) : String(e) };
  }
}
