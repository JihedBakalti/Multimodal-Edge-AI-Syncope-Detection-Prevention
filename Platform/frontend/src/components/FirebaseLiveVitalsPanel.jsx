import { useLiveFirebaseVitals } from "../hooks/useLiveFirebaseVitals";

function formatTime(ts) {
  if (!ts) return "—";
  try {
    return new Date(ts).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return "—";
  }
}

export default function FirebaseLiveVitalsPanel({
  firebaseUserId,
  sessionId = "default-session",
  title = "Live device vitals",
}) {
  const live = useLiveFirebaseVitals(firebaseUserId, sessionId);
  const uid = String(firebaseUserId ?? "").trim();
  const sid = String(sessionId ?? "default-session").trim() || "default-session";

  if (live.status === "no_patient_id") {
    return (
      <section className="panel firebase-live-panel">
        <div className="panel-head">
          <h3>{title}</h3>
        </div>
        <p className="muted">Set the patient&apos;s Firebase <code>user_id</code> on their profile to stream vitals from the same path as the medical assistant.</p>
      </section>
    );
  }

  if (live.status === "unconfigured") {
    return (
      <section className="panel firebase-live-panel">
        <div className="panel-head">
          <h3>{title}</h3>
        </div>
        <p className="muted">{live.message}</p>
      </section>
    );
  }

  const fbPollMs = Number(import.meta.env.VITE_FIREBASE_VITALS_POLL_MS);
  const fbPollLabel = Number.isFinite(fbPollMs) && fbPollMs >= 500 ? fbPollMs : 2000;
  const asstPollMs = Number(import.meta.env.VITE_LIVE_VITALS_POLL_MS);
  const asstPollLabel = Number.isFinite(asstPollMs) && asstPollMs >= 500 ? asstPollMs : 3000;
  const modeLabel =
    live.mode === "assistant"
      ? `Medical assistant API (${asstPollLabel} ms)`
      : live.mode === "realtime"
        ? "Realtime (Firebase SDK)"
        : live.mode === "poll"
          ? `Firebase REST (${fbPollLabel} ms)`
          : "—";
  const statusChip =
    live.status === "live" ? "live" : live.status === "connecting" || live.status === "polling" ? "pending" : live.status === "empty" ? "neutral" : "critical";

  return (
    <section className="panel firebase-live-panel">
      <div className="panel-head">
        <h3>{title}</h3>
        <div className="firebase-live-meta">
          <span className={`status-chip ${statusChip}`}>{live.status}</span>
          <span className="muted small-label">{modeLabel}</span>
        </div>
      </div>
      <p className="muted firebase-live-path">
        {live.mode === "assistant" ? (
          <>
            Same source as InterSense simulator: <code>{`/api/live-vitals?user_id=${uid}&session_id=${sid}`}</code>
          </>
        ) : (
          <>
            RTDB: <code>users/{uid}/vitals</code>
            {live.source ? <span className="muted"> · {live.source}</span> : null}
          </>
        )}
      </p>

      <div className="module-metrics-row firebase-live-metrics">
        <article className="module-metric-card">
          <p className="mini-label">Heart rate</p>
          <p className="mini-value">{live.heart_rate != null && live.heart_rate > 0 ? `${live.heart_rate} BPM` : "—"}</p>
        </article>
        <article className="module-metric-card">
          <p className="mini-label">SpO₂</p>
          <p className="mini-value">{live.blood_oxygen != null && live.blood_oxygen > 0 ? `${live.blood_oxygen}%` : "—"}</p>
        </article>
        <article className="module-metric-card">
          <p className="mini-label">Last update</p>
          <p className="mini-value">{formatTime(live.updatedAt)}</p>
        </article>
      </div>

      {live.message && live.status !== "live" ? <p className="muted firebase-live-msg">{live.message}</p> : null}
    </section>
  );
}
