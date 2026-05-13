import KpiCard from "../components/KpiCard";
import TimelineList from "../components/TimelineList";
import ChatPanel from "../components/ChatPanel";
import ClinicalLineChart from "../components/ClinicalLineChart";
import FirebaseLiveVitalsPanel from "../components/FirebaseLiveVitalsPanel";

export default function PatientDashboard({ me, assignments, activeSection, vitals, triggers, voiceChecks, incidents, alerts, onNavigateSection }) {
  const patientId = assignments[0]?.patient?.id;
  const myVitals = patientId ? vitals.filter((v) => Number(v.patient) === Number(patientId)) : vitals;
  const myVoice = patientId ? voiceChecks.filter((v) => Number(v.patient) === Number(patientId)) : voiceChecks;
  const myTriggers = patientId ? triggers.filter((v) => Number(v.patient) === Number(patientId)) : triggers;
  const myIncidents = patientId ? incidents.filter((v) => Number(v.patient) === Number(patientId)) : incidents;
  const latestVital = myVitals[0];
  const latestIncident = myIncidents[0];
  const warningAlerts = alerts.filter((item) => String(item.status || "").toLowerCase().includes("pending")).length;
  const criticalIncidents = incidents.filter((item) => String(item.state || "").toLowerCase().includes("critical")).length;

  return (
    <div className="dashboard">
      {activeSection === "overview" && (
        <>
          <section className="panel welcome-banner">
            <h2>Welcome back</h2>
            <p>Your latest monitoring summary is ready. Review incidents and messages below.</p>
          </section>

          <section className="panel action-panel">
            <div className="panel-head">
              <h3>Quick Actions</h3>
            </div>
            <div className="action-grid">
              <button type="button" className="action-btn" onClick={() => onNavigateSection("monitoring")}>View Monitoring</button>
              <button type="button" className="action-btn" onClick={() => onNavigateSection("alerts")}>Check Alerts</button>
              <button type="button" className="action-btn" onClick={() => onNavigateSection("messaging")}>Message Doctor</button>
              <button type="button" className="action-btn" onClick={() => onNavigateSection("overview")}>Download Summary</button>
            </div>
          </section>

          <section className="panel hero-panel">
            <p className="eyebrow">Patient Overview</p>
            <h2>Your long-term syncope monitoring timeline</h2>
            <p className="muted">Track trends, triggers, alerts, and incident history in one clear medical workspace.</p>
          </section>

          <section className="kpi-grid">
            <KpiCard label="Latest Heart Rate" value={latestVital ? `${latestVital.heart_rate} BPM` : "--"} hint="Most recent measured value" />
            <KpiCard label="Blood Oxygen" value={latestVital?.blood_oxygen ? `${latestVital.blood_oxygen}%` : "--"} hint="Peripheral oxygen saturation" />
            <KpiCard label="Current State" value={latestIncident?.state || "normal"} tone="warning" hint="Latest orchestrator decision" />
            <KpiCard label="Incidents" value={myIncidents.length} hint="Total logged incidents" />
            <KpiCard label="Voice checks" value={myVoice.length} hint="Safety-check voice scores" />
            <KpiCard label="Alerts" value={alerts.length} tone="warning" hint="Escalations and deliveries" />
          </section>
        </>
      )}

      {activeSection === "monitoring" && (
        <div className="content-grid">
          <div className="content-column">
            <section className="panel module-intro monitoring-intro">
              <p className="eyebrow">Monitoring Module</p>
              <h3>Signals and model decisions</h3>
              <p className="muted">Follow your recent vitals and model outcomes over time.</p>
            </section>
            <section className="module-metrics-row">
              <article className="module-metric-card">
                <p className="mini-label">Heart Rate</p>
                <p className="mini-value">{latestVital ? `${latestVital.heart_rate} bpm` : "--"}</p>
              </article>
              <article className="module-metric-card">
                <p className="mini-label">SpO2</p>
                <p className="mini-value">{latestVital?.blood_oxygen ? `${latestVital.blood_oxygen}%` : "--"}</p>
              </article>
              <article className="module-metric-card">
                <p className="mini-label">Triggers Logged</p>
                <p className="mini-value">{myTriggers.length}</p>
              </article>
            </section>
            <FirebaseLiveVitalsPanel
              firebaseUserId={assignments[0]?.patient?.user_id}
              sessionId={assignments[0]?.patient?.id ? `platform-patient-${assignments[0].patient.id}` : "default-session"}
              title="Live vitals from your device (Firebase)"
            />
            <section className="panel patient-graph-panel">
              <div className="panel-head">
                <h3>Vitals and voice risk over time</h3>
              </div>
              <ClinicalLineChart vitals={myVitals} voiceSeries={myVoice} height={240} />
            </section>
            <TimelineList
              title="Voice safety checks"
              items={myVoice.slice(0, 12)}
              renderItem={(item) => (
                <div>
                  <p className="timeline-title">
                    Attempt {item.attempt} · {(item.check_kind || "").replace("_", " ")}
                    <span className={`status-chip ${item.response_class || "neutral"}`}>{item.response_class}</span>
                  </p>
                  <p className="muted">
                    dysarthria risk {(Number(item.prob_dysarthric) * 100).toFixed(0)}% · HR {item.heart_rate ?? "--"} ·{" "}
                    {(item.transcript_preview || "").slice(0, 120)}
                    {(item.transcript_preview || "").length > 120 ? "…" : ""}
                  </p>
                </div>
              )}
            />
            <TimelineList
              title="Model triggers"
              items={myTriggers.slice(0, 8)}
              renderItem={(item) => (
                <div>
                  <p className="timeline-title">{item.model_name}</p>
                  <p className="muted">
                    score {Number(item.score).toFixed(2)} · {item.state}
                  </p>
                </div>
              )}
            />
            <TimelineList
              title="Monitoring Incidents"
              items={myIncidents.slice(0, 8)}
              renderItem={(item) => (
                <div>
                  <p className="timeline-title">
                    <span className={`status-chip ${item.state || "neutral"}`}>{item.state}</span>
                  </p>
                  <p className="muted">{item.message || "No detail"}</p>
                </div>
              )}
            />
          </div>
          <div className="content-column">
            <TimelineList
              title="Alert History"
              items={alerts.slice(0, 8)}
              renderItem={(item) => (
                <div>
                  <p className="timeline-title">
                    {item.alert_type} <span className={`status-chip ${item.status || "neutral"}`}>{item.status}</span>
                  </p>
                  <p className="muted">channel: {item.channel}</p>
                </div>
              )}
            />
          </div>
        </div>
      )}

      {activeSection === "alerts" && (
        <div className="content-grid">
          <div className="content-column">
            <section className="panel module-intro alerts-intro">
              <p className="eyebrow">Alerts Module</p>
              <h3>Escalations and notifications</h3>
              <p className="muted">Review every alert event and follow response actions.</p>
            </section>
            <section className="module-metrics-row">
              <article className="module-metric-card">
                <p className="mini-label">Total Alerts</p>
                <p className="mini-value">{alerts.length}</p>
              </article>
              <article className="module-metric-card">
                <p className="mini-label">Pending</p>
                <p className="mini-value">{warningAlerts}</p>
              </article>
              <article className="module-metric-card">
                <p className="mini-label">Critical Incidents</p>
                <p className="mini-value">{criticalIncidents}</p>
              </article>
            </section>
            <TimelineList
              title="Full Alert Timeline"
              items={alerts}
              renderItem={(item) => (
                <div>
                  <p className="timeline-title">
                    {item.alert_type} <span className={`status-chip ${item.status || "neutral"}`}>{item.status}</span>
                  </p>
                  <p className="muted">channel: {item.channel}</p>
                </div>
              )}
            />
          </div>
          <div className="content-column">
            <section className="panel">
              <div className="panel-head">
                <h3>Alert Actions</h3>
              </div>
              <div className="action-grid">
                <button type="button" className="action-btn" onClick={() => onNavigateSection("alerts")}>Acknowledge Alerts</button>
                <button type="button" className="action-btn" onClick={() => onNavigateSection("messaging")}>Notify Doctor</button>
                <button type="button" className="action-btn" onClick={() => onNavigateSection("overview")}>Export Alert Log</button>
                <button type="button" className="action-btn" onClick={() => onNavigateSection("messaging")}>Open Support Chat</button>
              </div>
            </section>
          </div>
        </div>
      )}

      {activeSection === "messaging" && (
        <>
          <section className="panel module-intro messaging-intro">
            <p className="eyebrow">Messaging Module</p>
            <h3>Secure contact with your care team</h3>
            <p className="muted">Messages remain stored and synchronized when doctor or patient reconnects later.</p>
          </section>
          <ChatPanel me={me} assignments={assignments} />
        </>
      )}
    </div>
  );
}
