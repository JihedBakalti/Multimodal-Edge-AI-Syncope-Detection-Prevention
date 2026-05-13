import { useEffect, useState } from "react";
import KpiCard from "../components/KpiCard";
import TimelineList from "../components/TimelineList";
import ChatPanel from "../components/ChatPanel";
import ClinicalLineChart from "../components/ClinicalLineChart";
import FirebaseLiveVitalsPanel from "../components/FirebaseLiveVitalsPanel";
import { api } from "../services/api";

export default function DoctorDashboard({ me, activeSection, assignments, vitals, triggers, voiceChecks, incidents, alerts, onNavigateSection }) {
  const [doctorAssignments, setDoctorAssignments] = useState(assignments);
  const [patientQuery, setPatientQuery] = useState("");
  const [patientFilter, setPatientFilter] = useState("all");
  const [patientsView, setPatientsView] = useState("list");
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [selectedPatientSummary, setSelectedPatientSummary] = useState(null);
  const [selectedPatientSummaryState, setSelectedPatientSummaryState] = useState("idle");
  const [reportForm, setReportForm] = useState({
    patient: "",
    period_start: "",
    period_end: "",
    summary: "",
    risk_notes: "",
    recommendations: "",
    source_log_ids: [],
  });
  const [reportState, setReportState] = useState("");
  const [patientForm, setPatientForm] = useState({
    email: "",
    username: "",
    password: "",
    user_id: "",
    emergency_contact: "",
  });
  const [patientCreateState, setPatientCreateState] = useState("");
  const [patientEditState, setPatientEditState] = useState("");
  const highRiskIncidents = incidents.filter((item) => String(item.state || "").toLowerCase().includes("critical")).length;
  const pendingAlerts = alerts.filter((item) => String(item.status || "").toLowerCase() !== "sent").length;
  const reportCards = [
    { key: "patients", label: "Patient", value: doctorAssignments.length || 0, icon: "☻" },
    { key: "consultations", label: "Consultation", value: Math.max(1, Math.floor((vitals.length || 0) / 2)), icon: "◌" },
    { key: "alerts", label: "Alert", value: alerts.length || 0, icon: "✦" },
    { key: "incidents", label: "Incident", value: incidents.length || 0, icon: "⚠" },
  ];

  useEffect(() => {
    setDoctorAssignments(assignments);
  }, [assignments]);

  useEffect(() => {
    if (!doctorAssignments.length) {
      setSelectedPatientId(null);
      return;
    }
    setSelectedPatientId((prev) => {
      if (prev && doctorAssignments.some((item) => Number(item.patient.id) === Number(prev))) {
        return prev;
      }
      return doctorAssignments[0].patient.id;
    });
  }, [doctorAssignments]);

  useEffect(() => {
    if (!selectedPatientId) {
      setSelectedPatientSummary(null);
      setSelectedPatientSummaryState("idle");
      return;
    }
    let cancelled = false;
    async function loadSummary() {
      setSelectedPatientSummaryState("loading");
      try {
        const summary = await api.patientSummary(selectedPatientId);
        if (cancelled) return;
        setSelectedPatientSummary(summary);
        setSelectedPatientSummaryState("success");
      } catch {
        if (cancelled) return;
        setSelectedPatientSummaryState("error");
      }
    }
    loadSummary().catch(() => setSelectedPatientSummaryState("error"));
    return () => {
      cancelled = true;
    };
  }, [selectedPatientId]);

  async function submitReport(event) {
    event.preventDefault();
    try {
      await api.createReport(reportForm);
      setReportState("Report created successfully.");
    } catch {
      setReportState("Could not create report.");
    }
  }

  async function submitPatientCreate(event) {
    event.preventDefault();
    setPatientCreateState("");
    try {
      await api.createPatientAccount(patientForm);
      const refreshed = await api.assignments();
      setDoctorAssignments(refreshed);
      setPatientForm({
        email: "",
        username: "",
        password: "",
        user_id: "",
        emergency_contact: "",
      });
      setPatientCreateState("Patient account created and linked to your account.");
      setPatientsView("list");
    } catch (error) {
      setPatientCreateState(error.message || "Could not create patient account.");
    }
  }

  async function submitPatientEdit(event) {
    event.preventDefault();
    if (!selectedPatientId) return;
    setPatientEditState("");
    try {
      await api.updatePatientAccount(selectedPatientId, {
        email: patientForm.email,
        username: patientForm.username,
        user_id: patientForm.user_id,
        emergency_contact: patientForm.emergency_contact,
      });
      const refreshed = await api.assignments();
      setDoctorAssignments(refreshed);
      setPatientEditState("Patient profile updated successfully.");
      setPatientsView("detail");
    } catch (error) {
      setPatientEditState(error.message || "Could not update patient profile.");
    }
  }

  const filteredAssignments = doctorAssignments.filter((assignment) => {
    const email = String(assignment.patient.user.email || "").toLowerCase();
    const username = String(assignment.patient.user.username || "").toLowerCase();
    const userId = String(assignment.patient.user_id || "").toLowerCase();
    const q = patientQuery.trim().toLowerCase();
    const matchesQuery = !q || email.includes(q) || username.includes(q) || userId.includes(q);
    if (!matchesQuery) return false;
    if (patientFilter === "all") return true;
    if (patientFilter === "with_user_id") {
      return Boolean(assignment.patient?.user_id);
    }
    if (patientFilter === "with_contact") {
      return Boolean(assignment.patient?.emergency_contact);
    }
    return true;
  });
  const selectedAssignment = doctorAssignments.find((assignment) => Number(assignment.patient.id) === Number(selectedPatientId));
  const selectedVitalsChart = vitals.filter((item) => Number(item.patient) === Number(selectedPatientId));
  const selectedVitals = selectedVitalsChart.slice(0, 12);
  const selectedVoiceChecks = voiceChecks.filter((item) => Number(item.patient) === Number(selectedPatientId));
  const selectedTriggers = triggers.filter((item) => Number(item.patient) === Number(selectedPatientId));
  const selectedIncidents = incidents
    .filter((item) => Number(item.patient) === Number(selectedPatientId))
    .slice(0, 8);
  const selectedAlerts = alerts
    .filter((item) => Number(item.patient) === Number(selectedPatientId))
    .slice(0, 8);

  function openAddPatientPage() {
    setPatientCreateState("");
    setPatientEditState("");
    setPatientForm({
      email: "",
      username: "",
      password: "",
      user_id: "",
      emergency_contact: "",
    });
    setPatientsView("add");
  }

  function openEditPatientPage(assignment) {
    setPatientCreateState("");
    setPatientEditState("");
    setSelectedPatientId(assignment.patient.id);
    setPatientForm({
      email: assignment.patient.user.email || "",
      username: assignment.patient.user.username || "",
      password: "",
      user_id: assignment.patient.user_id || "",
      emergency_contact: assignment.patient.emergency_contact || "",
    });
    setPatientsView("edit");
  }

  return (
    <div className="dashboard">
      {activeSection === "overview" && (
        <>
          <section className="panel welcome-banner">
            <h2>Welcome, Doctor</h2>
            <p>Have a focused day at work. Track patients and react quickly to alerts.</p>
          </section>

          <section className="panel action-panel">
            <div className="panel-head">
              <h3>Quick Actions</h3>
            </div>
            <div className="action-grid">
              <button type="button" className="action-btn" onClick={() => onNavigateSection("patients")}>Create Patient</button>
              <button type="button" className="action-btn" onClick={() => onNavigateSection("reports")}>Generate Report</button>
              <button type="button" className="action-btn" onClick={() => onNavigateSection("incidents")}>Review Alerts</button>
              <button type="button" className="action-btn" onClick={() => onNavigateSection("messaging")}>Open Messaging</button>
            </div>
          </section>

          <section className="panel hero-panel">
            <p className="eyebrow">Doctor Command Center</p>
            <h2>Patient safety oversight and reporting workspace</h2>
            <p className="muted">Executive view of patient health activity, incident load, and escalation performance.</p>
          </section>

          <section className="panel">
            <div className="panel-head">
              <h3>Report</h3>
              <span className="muted small-label">This month</span>
            </div>
            <div className="report-cards-grid">
              {reportCards.map((card) => (
                <article key={card.key} className="report-mini-card">
                  <div className="mini-icon">{card.icon}</div>
                  <p className="mini-label">{card.label}</p>
                  <p className="mini-value">{card.value}</p>
                </article>
              ))}
              <button type="button" className="report-mini-card add-card" onClick={() => onNavigateSection("reports")}>
                <div className="add-symbol">+</div>
              </button>
            </div>
          </section>

          <section className="kpi-grid">
            <KpiCard label="Assigned Patients" value={doctorAssignments.length} hint="Active care assignments" />
            <KpiCard label="Recent Vitals" value={vitals.length} hint="Logged samples available" />
            <KpiCard label="Recent Incidents" value={incidents.length} tone="warning" hint="Detected safety events" />
            <KpiCard label="Alert Events" value={alerts.length} tone="warning" hint="Escalations and delivery status" />
          </section>

          <div className="content-grid">
            <div className="content-column">
              <TimelineList
                title="Recent Incidents"
                items={incidents.slice(0, 8)}
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
                title="Alert Delivery Log"
                items={alerts.slice(0, 8)}
                renderItem={(item) => (
                  <div>
                    <p className="timeline-title">
                      {item.alert_type} <span className={`status-chip ${item.status || "neutral"}`}>{item.status}</span>
                    </p>
                    <p className="muted">channel: {item.channel} | session: {item.session_id || "n/a"}</p>
                  </div>
                )}
              />
            </div>
          </div>
        </>
      )}

      {activeSection === "patients" && (
        <>
          <section className="panel module-intro patients-intro">
            <p className="eyebrow">Patients Module</p>
            <h3>Patient list and profile management</h3>
            <p className="muted">
              Browse all assigned patients. Select one to open their statistics dashboard, or use Add/Edit actions for account management.
            </p>
          </section>

          {patientsView === "list" && (
            <section className="panel">
              <div className="panel-head">
                <h3>Assigned Patients</h3>
                <div className="topbar-actions">
                  <button type="button" className="ghost-button" onClick={openAddPatientPage}>Add Patient</button>
                  <span className="panel-count">{filteredAssignments.length}</span>
                </div>
              </div>
              <div className="patient-tools">
                <input
                  value={patientQuery}
                  onChange={(event) => setPatientQuery(event.target.value)}
                  placeholder="Search by email, username, or user_id"
                />
                <select value={patientFilter} onChange={(event) => setPatientFilter(event.target.value)}>
                  <option value="all">All patients</option>
                  <option value="with_user_id">With user_id</option>
                  <option value="with_contact">With emergency contact</option>
                </select>
              </div>
              <div className="list">
                {filteredAssignments.map((assignment) => (
                  <div key={assignment.id} className="list-row patient-list-item">
                    <button
                      type="button"
                      className="patient-list-main"
                      onClick={() => {
                        setSelectedPatientId(assignment.patient.id);
                        setPatientsView("detail");
                      }}
                    >
                      <strong>{assignment.patient.user.email}</strong>
                      <span className="muted">user_id: {assignment.patient.user_id}</span>
                    </button>
                    <button type="button" className="ghost-button tiny-btn" onClick={() => openEditPatientPage(assignment)}>
                      Edit
                    </button>
                  </div>
                ))}
                {!filteredAssignments.length && <p className="muted timeline-empty">No patient matches this filter.</p>}
              </div>
            </section>
          )}

          {patientsView === "detail" && (
            <section className="panel">
              <div className="panel-head">
                <h3>{selectedAssignment?.patient?.user?.email || "Patient"} - Statistics Dashboard</h3>
                <div className="topbar-actions">
                  <button type="button" className="ghost-button" onClick={() => setPatientsView("list")}>Back to List</button>
                  {selectedAssignment && (
                    <button type="button" className="ghost-button" onClick={() => openEditPatientPage(selectedAssignment)}>Edit Patient</button>
                  )}
                </div>
              </div>

              {selectedPatientSummaryState === "loading" && <p className="muted">Loading patient statistics...</p>}
              {selectedPatientSummaryState === "error" && <p className="muted">Could not load patient summary.</p>}
              {selectedPatientSummaryState === "success" && (
                <>
                  <section className="module-metrics-row">
                    <article className="module-metric-card">
                      <p className="mini-label">Vitals</p>
                      <p className="mini-value">{selectedPatientSummary?.vitals_count ?? 0}</p>
                    </article>
                    <article className="module-metric-card">
                      <p className="mini-label">Triggers</p>
                      <p className="mini-value">{selectedPatientSummary?.triggers_count ?? 0}</p>
                    </article>
                    <article className="module-metric-card">
                      <p className="mini-label">Incidents</p>
                      <p className="mini-value">{selectedPatientSummary?.incidents_count ?? 0}</p>
                    </article>
                    <article className="module-metric-card">
                      <p className="mini-label">Voice checks</p>
                      <p className="mini-value">{selectedPatientSummary?.voice_checks_count ?? 0}</p>
                    </article>
                  </section>

                  <FirebaseLiveVitalsPanel
                    firebaseUserId={selectedAssignment?.patient?.user_id}
                    sessionId={selectedPatientId ? `platform-patient-${selectedPatientId}` : "default-session"}
                    patientId={selectedPatientId}
                  />

                  <section className="panel patient-graph-panel">
                    <div className="panel-head">
                      <h3>Vitals, SpO2, and voice dysarthria risk</h3>
                    </div>
                    <ClinicalLineChart vitals={selectedVitalsChart} voiceSeries={selectedVoiceChecks} height={260} />
                    {selectedVitalsChart.length === 0 && <p className="muted">No vitals samples for this patient yet.</p>}
                  </section>

                  <div className="content-grid">
                    <TimelineList
                      title="Voice safety checks"
                      items={selectedVoiceChecks.slice(0, 12)}
                      renderItem={(item) => (
                        <div>
                          <p className="timeline-title">
                            #{item.attempt} {item.check_kind}{" "}
                            <span className={`status-chip ${item.response_class || "neutral"}`}>{item.response_class}</span>
                          </p>
                          <p className="muted">
                            risk {(Number(item.prob_dysarthric) * 100).toFixed(0)}% · HR {item.heart_rate ?? "--"} ·{" "}
                            {(item.transcript_preview || "").slice(0, 100)}
                            {(item.transcript_preview || "").length > 100 ? "…" : ""}
                          </p>
                        </div>
                      )}
                    />
                    <TimelineList
                      title="Model triggers"
                      items={selectedTriggers.slice(0, 12)}
                      renderItem={(item) => (
                        <div>
                          <p className="timeline-title">{item.model_name}</p>
                          <p className="muted">
                            score {Number(item.score).toFixed(2)} / {item.state}
                          </p>
                        </div>
                      )}
                    />
                  </div>

                  <div className="content-grid">
                    <TimelineList
                      title="Incident Logs"
                      items={selectedIncidents}
                      renderItem={(item) => (
                        <div>
                          <p className="timeline-title">
                            <span className={`status-chip ${item.state || "neutral"}`}>{item.state}</span>
                          </p>
                          <p className="muted">{item.message || "No detail"}</p>
                        </div>
                      )}
                    />
                    <TimelineList
                      title="Alert Logs"
                      items={selectedAlerts}
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
                </>
              )}
            </section>
          )}

          {patientsView === "add" && (
            <section className="panel">
              <div className="panel-head">
                <h3>Add New Patient</h3>
                <button type="button" className="ghost-button" onClick={() => setPatientsView("list")}>Back to List</button>
              </div>
              <p className="muted">Create a patient account and link it automatically to your doctor profile.</p>
              <form className="report-form" onSubmit={submitPatientCreate}>
                <input required type="email" placeholder="Patient email" value={patientForm.email} onChange={(e) => setPatientForm({ ...patientForm, email: e.target.value })} />
                <input required placeholder="Username" value={patientForm.username} onChange={(e) => setPatientForm({ ...patientForm, username: e.target.value })} />
                <input required type="password" placeholder="Temporary password" value={patientForm.password} onChange={(e) => setPatientForm({ ...patientForm, password: e.target.value })} />
                <input required placeholder="Firebase user_id" value={patientForm.user_id} onChange={(e) => setPatientForm({ ...patientForm, user_id: e.target.value })} />
                <input placeholder="Emergency contact" value={patientForm.emergency_contact} onChange={(e) => setPatientForm({ ...patientForm, emergency_contact: e.target.value })} />
                <button type="submit">Create Patient Account</button>
                {patientCreateState && <p className="muted">{patientCreateState}</p>}
              </form>
            </section>
          )}

          {patientsView === "edit" && (
            <section className="panel">
              <div className="panel-head">
                <h3>Edit Patient</h3>
                <button type="button" className="ghost-button" onClick={() => setPatientsView("detail")}>Back to Dashboard</button>
              </div>
              <p className="muted">Modify the selected patient profile.</p>
              <form className="report-form" onSubmit={submitPatientEdit}>
                <input required type="email" placeholder="Patient email" value={patientForm.email} onChange={(e) => setPatientForm({ ...patientForm, email: e.target.value })} />
                <input required placeholder="Username" value={patientForm.username} onChange={(e) => setPatientForm({ ...patientForm, username: e.target.value })} />
                <input required placeholder="Firebase user_id" value={patientForm.user_id} onChange={(e) => setPatientForm({ ...patientForm, user_id: e.target.value })} />
                <input placeholder="Emergency contact" value={patientForm.emergency_contact} onChange={(e) => setPatientForm({ ...patientForm, emergency_contact: e.target.value })} />
                <button type="submit">Save Changes</button>
                {patientEditState && <p className="muted">{patientEditState}</p>}
              </form>
            </section>
          )}
        </>
      )}

      {activeSection === "incidents" && (
        <div className="content-grid">
          <div className="content-column">
            <section className="panel module-intro incidents-intro">
              <p className="eyebrow">Incidents Module</p>
              <h3>Escalation and event oversight</h3>
              <p className="muted">Track incident intensity, escalation flow, and alert delivery quality in one place.</p>
            </section>
            <section className="module-metrics-row">
              <article className="module-metric-card">
                <p className="mini-label">Total Incidents</p>
                <p className="mini-value">{incidents.length}</p>
              </article>
              <article className="module-metric-card">
                <p className="mini-label">High Risk</p>
                <p className="mini-value">{highRiskIncidents}</p>
              </article>
              <article className="module-metric-card">
                <p className="mini-label">Pending Alerts</p>
                <p className="mini-value">{pendingAlerts}</p>
              </article>
            </section>
            <TimelineList
              title="Incident Timeline"
              items={incidents}
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
            <section className="panel">
              <div className="panel-head">
                <h3>Escalation Channels</h3>
              </div>
              <div className="list">
                {["SMS", "voice_call", "push", "email"].map((channel) => {
                  const count = alerts.filter((item) => String(item.channel).toLowerCase() === channel).length;
                  return (
                    <div className="list-row" key={channel}>
                      <strong>{channel}</strong>
                      <span className="status-chip neutral">{count}</span>
                    </div>
                  );
                })}
              </div>
            </section>
            <TimelineList
              title="Alert Timeline"
              items={alerts}
              renderItem={(item) => (
                <div>
                  <p className="timeline-title">
                    {item.alert_type} <span className={`status-chip ${item.status || "neutral"}`}>{item.status}</span>
                  </p>
                  <p className="muted">channel: {item.channel} | session: {item.session_id || "n/a"}</p>
                </div>
              )}
            />
          </div>
        </div>
      )}

      {activeSection === "messaging" && (
        <>
          <section className="panel module-intro messaging-intro">
            <p className="eyebrow">Messaging Module</p>
            <h3>Doctor-patient communication center</h3>
            <p className="muted">Asynchronous conversations stay stored so patients can respond whenever they reconnect.</p>
          </section>
          <ChatPanel me={me} assignments={doctorAssignments} />
        </>
      )}

      {activeSection === "reports" && (
        <section className="panel">
          <section className="panel module-intro reports-intro">
            <p className="eyebrow">Reports Module</p>
            <h3>Clinical report drafting and export</h3>
            <p className="muted">Build narrative reports with period filters, risk markers, and care recommendations.</p>
          </section>
          <section className="module-metrics-row">
            <article className="module-metric-card">
              <p className="mini-label">Reports Drafted</p>
              <p className="mini-value">{Math.max(1, Math.floor((incidents.length + alerts.length) / 2))}</p>
            </article>
            <article className="module-metric-card">
              <p className="mini-label">Recommended Review</p>
              <p className="mini-value">7 days</p>
            </article>
            <article className="module-metric-card">
              <p className="mini-label">Export Format</p>
              <p className="mini-value">PDF</p>
            </article>
          </section>
          <div className="panel-head">
            <h3>Generate Clinical Report</h3>
          </div>
          <div className="report-toolbar">
            <button type="button" className="ghost-button">Last 24h</button>
            <button type="button" className="ghost-button">Last 7 days</button>
            <button type="button" className="ghost-button">Last 30 days</button>
          </div>
          <form className="report-form" onSubmit={submitReport}>
            <input required placeholder="Patient ID" value={reportForm.patient} onChange={(e) => setReportForm({ ...reportForm, patient: e.target.value })} />
            <input required type="datetime-local" value={reportForm.period_start} onChange={(e) => setReportForm({ ...reportForm, period_start: e.target.value })} />
            <input required type="datetime-local" value={reportForm.period_end} onChange={(e) => setReportForm({ ...reportForm, period_end: e.target.value })} />
            <textarea required placeholder="Clinical summary" value={reportForm.summary} onChange={(e) => setReportForm({ ...reportForm, summary: e.target.value })} />
            <textarea placeholder="Risk notes" value={reportForm.risk_notes} onChange={(e) => setReportForm({ ...reportForm, risk_notes: e.target.value })} />
            <textarea placeholder="Recommendations" value={reportForm.recommendations} onChange={(e) => setReportForm({ ...reportForm, recommendations: e.target.value })} />
            <button type="submit">Generate Report</button>
            {reportState && <p className="muted">{reportState}</p>}
          </form>
        </section>
      )}
    </div>
  );
}
