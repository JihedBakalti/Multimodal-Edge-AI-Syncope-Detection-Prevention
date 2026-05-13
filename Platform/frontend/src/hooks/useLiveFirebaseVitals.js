import { useEffect, useState } from "react";
import { initializeApp, getApps } from "firebase/app";
import { getDatabase, ref, onValue } from "firebase/database";
import { fetchFirebaseVitalsRest, parseVitalsDict, rtdbBaseUrlFromEnv } from "../services/firebaseVitals";
import { assistantApiBaseFromEnv, fetchLiveVitalsAssistant } from "../services/liveVitalsAssistant";

function ensureFirebaseApp() {
  const existing = getApps();
  if (existing.length) return existing[0];
  const apiKey = import.meta.env.VITE_FIREBASE_API_KEY;
  const projectId = import.meta.env.VITE_FIREBASE_PROJECT_ID;
  const databaseURL =
    import.meta.env.VITE_FIREBASE_DATABASE_URL || import.meta.env.VITE_FIREBASE_RTDB_URL || "";
  if (!apiKey || !projectId || !String(databaseURL).trim()) {
    return null;
  }
  return initializeApp({
    apiKey,
    projectId,
    databaseURL: String(databaseURL).replace(/\/+$/, ""),
    authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || `${projectId}.firebaseapp.com`,
    storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || undefined,
    messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || undefined,
    appId: import.meta.env.VITE_FIREBASE_APP_ID || undefined,
  });
}

function firebasePollIntervalMs() {
  const n = Number(import.meta.env.VITE_FIREBASE_VITALS_POLL_MS);
  return Number.isFinite(n) && n >= 500 ? n : 2000;
}

/** Same interval as `frontend/src/components/InterSenseSimulator.jsx` live vitals poll. */
function assistantPollIntervalMs() {
  const n = Number(import.meta.env.VITE_LIVE_VITALS_POLL_MS);
  return Number.isFinite(n) && n >= 500 ? n : 3000;
}

/**
 * Live vitals for one Firebase user id (`PatientProfile.firebase_user_id` / API `user_id`).
 *
 * 1) If `VITE_API_BASE_URL` or `VITE_ORCHESTRATOR_API_BASE_URL` is set — polls `GET /api/live-vitals`
 *    (same contract as the medical assistant / InterSenseSimulator frontend; server reads Firebase).
 * 2) Else — browser Firebase RTDB (SDK or REST) using `VITE_FIREBASE_RTDB_URL`.
 */
export function useLiveFirebaseVitals(firebaseUserId, sessionId = "default-session") {
  const [state, setState] = useState({
    status: "idle",
    mode: "none",
    heart_rate: null,
    blood_oxygen: null,
    source: null,
    message: null,
    updatedAt: null,
    raw: null,
  });

  useEffect(() => {
    const uid = String(firebaseUserId ?? "").trim();
    if (!uid) {
      setState((s) => ({
        ...s,
        status: "no_patient_id",
        mode: "none",
        message: "No Firebase user id on this profile.",
      }));
      return undefined;
    }

    const assistantBase = assistantApiBaseFromEnv();
    if (assistantBase) {
      const sid = String(sessionId ?? "default-session").trim() || "default-session";
      let cancelled = false;
      setState((s) => ({ ...s, status: "polling", mode: "assistant" }));

      async function tick() {
        const result = await fetchLiveVitalsAssistant(assistantBase, uid, sid);
        if (cancelled) return;
        if (result.ok) {
          setState({
            status: "live",
            mode: "assistant",
            heart_rate: result.heart_rate,
            blood_oxygen: result.blood_oxygen,
            source: result.source || "/api/live-vitals",
            message: null,
            updatedAt: Date.now(),
            raw: result.raw ?? null,
          });
        } else {
          setState({
            status: "error",
            mode: "assistant",
            heart_rate: null,
            blood_oxygen: null,
            source: null,
            message: result.message || "live-vitals request failed",
            updatedAt: Date.now(),
            raw: result.raw ?? null,
          });
        }
      }

      tick();
      const id = setInterval(tick, assistantPollIntervalMs());
      return () => {
        cancelled = true;
        clearInterval(id);
      };
    }

    const baseUrl = rtdbBaseUrlFromEnv();
    if (!baseUrl) {
      setState((s) => ({
        ...s,
        status: "unconfigured",
        mode: "none",
        message:
          "Set VITE_API_BASE_URL to the medical assistant host (e.g. http://127.0.0.1:8000), or set VITE_FIREBASE_RTDB_URL for direct browser reads.",
      }));
      return undefined;
    }

    let app = null;
    try {
      app = ensureFirebaseApp();
    } catch {
      app = null;
    }
    if (app) {
      setState((s) => ({ ...s, status: "connecting", mode: "realtime" }));
      const db = getDatabase(app);
      const vitalsRef = ref(db, `users/${uid}/vitals`);
      const unsub = onValue(
        vitalsRef,
        (snapshot) => {
          const val = snapshot.val();
          const parsed = parseVitalsDict(val);
          if (parsed) {
            setState({
              status: "live",
              mode: "realtime",
              heart_rate: parsed.heart_rate,
              blood_oxygen: parsed.blood_oxygen,
              source: `users/${uid}/vitals`,
              message: null,
              updatedAt: Date.now(),
              raw: parsed.raw,
            });
          } else {
            setState({
              status: "empty",
              mode: "realtime",
              heart_rate: null,
              blood_oxygen: null,
              source: `users/${uid}/vitals`,
              message: "No HeartRate / BloodOxygen at this path yet.",
              updatedAt: Date.now(),
              raw: val,
            });
          }
        },
        (err) => {
          setState((s) => ({
            ...s,
            status: "error",
            mode: "realtime",
            message: err?.message ? String(err.message) : String(err),
          }));
        }
      );
      return () => unsub();
    }

    let cancelled = false;
    const intervalMs = firebasePollIntervalMs();
    setState((s) => ({ ...s, status: "polling", mode: "poll" }));

    async function tick() {
      const result = await fetchFirebaseVitalsRest(baseUrl, uid);
      if (cancelled) return;
      if (result.ok) {
        setState({
          status: "live",
          mode: "poll",
          heart_rate: result.heart_rate,
          blood_oxygen: result.blood_oxygen,
          source: result.source || null,
          message: null,
          updatedAt: Date.now(),
          raw: result.raw ?? null,
        });
      } else {
        setState({
          status: "error",
          mode: "poll",
          heart_rate: null,
          blood_oxygen: null,
          source: null,
          message: result.message || "Fetch failed",
          updatedAt: Date.now(),
          raw: result.raw ?? null,
        });
      }
    }

    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [firebaseUserId, sessionId]);

  return state;
}
