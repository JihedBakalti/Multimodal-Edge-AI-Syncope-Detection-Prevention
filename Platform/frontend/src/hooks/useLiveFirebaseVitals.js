import { useEffect, useState } from "react";
import { initializeApp, getApps } from "firebase/app";
import { getDatabase, ref, onValue } from "firebase/database";
import { api } from "../services/api";
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

function assistantPollIntervalMs() {
  const n = Number(import.meta.env.VITE_LIVE_VITALS_POLL_MS);
  return Number.isFinite(n) && n >= 500 ? n : 3000;
}

function platformProxyPollIntervalMs() {
  const n = Number(import.meta.env.VITE_PLATFORM_FIREBASE_POLL_MS);
  return Number.isFinite(n) && n >= 500 ? n : 2500;
}

/**
 * Live vitals for one patient (Firebase user id on profile).
 *
 * 1) VITE_API_BASE_URL / VITE_ORCHESTRATOR_API_BASE_URL → medical assistant GET /api/live-vitals
 * 2) Else VITE_PLATFORM_API_BASE + Django patient id → GET /api/clinical/firebase-live-vitals/ (Render reads RTDB; set FIREBASE_RTDB_URL on server)
 * 3) Else VITE_FIREBASE_RTDB_URL (+ optional web SDK env) → browser RTDB
 */
export function useLiveFirebaseVitals(firebaseUserId, sessionId = "default-session", patientId = null) {
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

    const pid = patientId != null && String(patientId).trim() !== "" ? Number(patientId) : null;
    if (pid != null && Number.isFinite(pid)) {
      let cancelled = false;
      setState((s) => ({ ...s, status: "polling", mode: "platform" }));

      async function tick() {
        try {
          const data = await api.firebaseLiveVitals(pid);
          if (cancelled) return;
          if (data?.ok) {
            setState({
              status: "live",
              mode: "platform",
              heart_rate: data.heart_rate ?? null,
              blood_oxygen: data.blood_oxygen ?? null,
              source: data.source || "/api/clinical/firebase-live-vitals/",
              message: null,
              updatedAt: Date.now(),
              raw: data.raw ?? null,
            });
          } else {
            setState({
              status: "error",
              mode: "platform",
              heart_rate: null,
              blood_oxygen: null,
              source: null,
              message: data?.message || "Platform proxy returned no vitals",
              updatedAt: Date.now(),
              raw: data?.raw ?? null,
            });
          }
        } catch (e) {
          if (cancelled) return;
          setState({
            status: "error",
            mode: "platform",
            heart_rate: null,
            blood_oxygen: null,
            source: null,
            message: e?.message ? String(e.message) : "Platform live vitals request failed",
            updatedAt: Date.now(),
            raw: null,
          });
        }
      }

      tick();
      const id = setInterval(tick, platformProxyPollIntervalMs());
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
          "On Render set FIREBASE_RTDB_URL (same RTDB URL as the medical assistant). Redeploy the backend. Optional: VITE_API_BASE_URL on Vercel for the assistant merge feed, or VITE_FIREBASE_RTDB_URL for direct browser reads.",
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
  }, [firebaseUserId, sessionId, patientId]);

  return state;
}
