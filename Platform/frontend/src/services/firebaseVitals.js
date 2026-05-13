/**
 * Direct Firebase RTDB reads from the browser (REST or SDK).
 * Prefer `liveVitalsAssistant.js` + `VITE_API_BASE_URL` when the medical assistant is running:
 * that path matches `InterSenseSimulator.jsx` and merges Firebase with simulator vitals server-side.
 * Primary path: /users/{user_id}/vitals with HeartRate, BloodOxygen.
 * Fallback: /users.json (array index or object key).
 *
 * Frontend env (Vite):
 *   VITE_FIREBASE_RTDB_URL — base RTDB URL (same role as FIREBASE_RTDB_URL in medical_assistant).
 *   Optional for Web SDK live listener: VITE_FIREBASE_API_KEY, VITE_FIREBASE_PROJECT_ID,
 *   VITE_FIREBASE_DATABASE_URL (or reuse RTDB URL), VITE_FIREBASE_AUTH_DOMAIN.
 */

export function rtdbBaseUrlFromEnv() {
  const u =
    import.meta.env.VITE_FIREBASE_RTDB_URL ||
    import.meta.env.VITE_FIREBASE_DATABASE_URL ||
    "";
  return String(u || "").replace(/\/+$/, "");
}

export function parseVitalsDict(direct) {
  if (!direct || typeof direct !== "object") return null;
  const hr = direct.HeartRate;
  const spo2 = direct.BloodOxygen;
  if (hr == null && spo2 == null) return null;
  return {
    heart_rate: hr != null && hr !== "" ? Number(hr) : null,
    blood_oxygen: spo2 != null && spo2 !== "" ? Number(spo2) : null,
    raw: direct,
  };
}

export async function fetchFirebaseVitalsRest(baseUrl, userId) {
  const uid = String(userId ?? "").trim();
  if (!baseUrl || !uid) {
    return { ok: false, message: "Missing RTDB base URL or patient Firebase user id." };
  }
  const root = baseUrl.replace(/\/+$/, "");
  const urlDirect = `${root}/users/${encodeURIComponent(uid)}/vitals.json`;
  try {
    const directResp = await fetch(urlDirect, { headers: { Accept: "application/json" } });
    const direct = await directResp.json();
    const parsed = parseVitalsDict(direct);
    if (parsed) {
      return { ok: true, ...parsed, source: `users/${uid}/vitals` };
    }

    const usersResp = await fetch(`${root}/users.json`, { headers: { Accept: "application/json" } });
    const usersPayload = await usersResp.json();
    let candidate = null;
    if (Array.isArray(usersPayload)) {
      const idx = /^\d+$/.test(uid) ? Number(uid) : null;
      if (idx != null && usersPayload.length > idx && usersPayload[idx] && typeof usersPayload[idx] === "object") {
        candidate = usersPayload[idx].vitals;
      }
    } else if (usersPayload && typeof usersPayload === "object") {
      const userEntry = usersPayload[uid];
      if (userEntry && typeof userEntry === "object") {
        candidate = userEntry.vitals;
      }
    }
    const p2 = parseVitalsDict(candidate);
    if (p2) {
      return { ok: true, ...p2, source: "users fallback" };
    }
    return {
      ok: false,
      message: `No vitals found at users/${uid}/vitals or users fallback.`,
      raw: { direct, users: usersPayload },
    };
  } catch (e) {
    return { ok: false, message: e?.message ? String(e.message) : String(e) };
  }
}
