import { useEffect, useRef, useState } from "react";
import { api } from "./services/api";
import PatientDashboard from "./pages/PatientDashboard";
import DoctorDashboard from "./pages/DoctorDashboard";

export default function App() {
  const [initializing, setInitializing] = useState(true);
  const [loginSubmitting, setLoginSubmitting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [me, setMe] = useState(null);
  const [authMode, setAuthMode] = useState("login");
  const [loginForm, setLoginForm] = useState({ email: "", password: "" });
  const [loginError, setLoginError] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const emailInputRef = useRef(null);
  const [assignments, setAssignments] = useState([]);
  const [vitals, setVitals] = useState([]);
  const [triggers, setTriggers] = useState([]);
  const [voiceChecks, setVoiceChecks] = useState([]);
  const [incidents, setIncidents] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [activeSection, setActiveSection] = useState("overview");

  async function bootstrap() {
    try {
      const [meData, assignmentData, vitalsData, triggerData, voiceData, incidentData, alertData] = await Promise.all([
        api.me(),
        api.assignments(),
        api.vitals(),
        api.triggers(),
        api.voiceChecks(),
        api.incidents(),
        api.alerts(),
      ]);
      setMe(meData);
      setAssignments(assignmentData);
      setVitals(vitalsData);
      setTriggers(triggerData);
      setVoiceChecks(voiceData);
      setIncidents(incidentData);
      setAlerts(alertData);
      setActiveSection("overview");
      setError("");
      setAuthMode("app");
    } catch (e) {
      if (e.status === 401 || !api.hasToken()) {
        setAuthMode("login");
        setError("");
      } else {
        setError("Unable to load platform data.");
      }
    } finally {
      setInitializing(false);
    }
  }

  useEffect(() => {
    bootstrap();
  }, []);

  useEffect(() => {
    if (!initializing && authMode === "login") {
      emailInputRef.current?.focus();
    }
  }, [initializing, authMode]);

  async function handleLogin(event) {
    event.preventDefault();
    setLoginError("");
    setLoginSubmitting(true);
    try {
      await api.login(loginForm.email, loginForm.password);
      await bootstrap();
    } catch (e) {
      if (e.status === 401) {
        setLoginError("Invalid email or password.");
      } else if (e.status === 403) {
        setLoginError(e.message || "Access denied (403). Check backend CORS and CSRF settings.");
      } else if (e.status === 0) {
        setLoginError(e.message || "Cannot reach the API. Check VITE_PLATFORM_API_BASE and redeploy.");
      } else {
        setLoginError(e.message || "Login failed. Check your connection and try again.");
      }
    } finally {
      setLoginSubmitting(false);
    }
  }

  async function handleLogout() {
    await api.logout();
    setMe(null);
    setAuthMode("login");
  }

  async function handleRefresh() {
    setRefreshing(true);
    try {
      await bootstrap();
    } finally {
      setRefreshing(false);
    }
  }

  if (initializing) {
    return (
      <div className="auth-boot-screen" role="status" aria-live="polite" aria-busy="true">
        <div className="auth-boot-card">
          <p className="auth-boot-brand">InterSense</p>
          <div className="auth-boot-spinner" aria-hidden />
          <p className="auth-boot-text">Preparing your clinical workspace…</p>
        </div>
      </div>
    );
  }

  if (authMode === "login") {
    const loginInvalid = Boolean(loginError);
    return (
      <main className="auth-shell">
        <section className="auth-side">
          <p className="eyebrow">InterSense Healthcare</p>
          <h1>Clinical Monitoring Platform</h1>
          <p>Securely monitor vitals, incidents, and alerts with doctor–patient coordination in real time.</p>
          <ul className="auth-side-points">
            <li>Role-based access for clinical staff and patients</li>
            <li>Encrypted session; sign out from Profile when finished</li>
          </ul>
        </section>
        <section className="panel auth-panel">
          <div className="auth-panel-head">
            <div className="auth-panel-icon" aria-hidden>
              <svg width="26" height="26" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z" />
              </svg>
            </div>
            <div>
              <h2>Sign in</h2>
              <p className="muted auth-panel-sub">Use the email and password issued for this portal.</p>
            </div>
          </div>

          <form className="auth-form" onSubmit={handleLogin}>
            {loginError ? (
              <div className="auth-error-banner" role="alert">
                {loginError}
              </div>
            ) : null}

            <div className="auth-field">
              <label className="auth-label" htmlFor="login-email">
                Work email
              </label>
              <input
                id="login-email"
                ref={emailInputRef}
                name="email"
                required
                type="email"
                autoComplete="username"
                inputMode="email"
                placeholder="you@hospital.org"
                className={loginInvalid ? "auth-input-invalid" : undefined}
                aria-invalid={loginInvalid}
                value={loginForm.email}
                disabled={loginSubmitting}
                onChange={(e) => setLoginForm({ ...loginForm, email: e.target.value })}
              />
            </div>

            <div className="auth-field">
              <label className="auth-label" htmlFor="login-password">
                Password
              </label>
              <div className="auth-password-row">
                <input
                  id="login-password"
                  name="password"
                  required
                  type={showPassword ? "text" : "password"}
                  autoComplete="current-password"
                  placeholder="Enter password"
                  className={loginInvalid ? "auth-input-invalid" : undefined}
                  aria-invalid={loginInvalid}
                  value={loginForm.password}
                  disabled={loginSubmitting}
                  onChange={(e) => setLoginForm({ ...loginForm, password: e.target.value })}
                />
                <button
                  type="button"
                  className="auth-password-toggle"
                  aria-pressed={showPassword}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  disabled={loginSubmitting}
                  onClick={() => setShowPassword((v) => !v)}
                >
                  {showPassword ? "Hide" : "Show"}
                </button>
              </div>
            </div>

            <button type="submit" className="auth-submit" disabled={loginSubmitting}>
              {loginSubmitting ? (
                <span className="auth-submit-inner">
                  <span className="auth-btn-spinner" aria-hidden />
                  Signing in…
                </span>
              ) : (
                "Continue to workspace"
              )}
            </button>
          </form>
          <p className="auth-footnote">Having trouble? Confirm caps lock is off and use the email linked to your account.</p>
        </section>
      </main>
    );
  }

  if (error) {
    return <main className="shell"><p>{error}</p></main>;
  }

  const doctorSections = [
    ["overview", "Overview"],
    ["patients", "Patients"],
    ["incidents", "Incidents"],
    ["messaging", "Messaging"],
    ["reports", "Reports"],
    ["profile", "Profile"],
  ];
  const patientSections = [
    ["overview", "Overview"],
    ["monitoring", "Monitoring"],
    ["alerts", "Alerts"],
    ["messaging", "Messaging"],
    ["profile", "Profile"],
  ];
  const sections = me.role === "doctor" ? doctorSections : patientSections;
  const sectionLabel = sections.find(([id]) => id === activeSection)?.[1] || "Overview";
  const isOverview = activeSection === "overview";
  const iconMap = {
    overview: "⌂",
    patients: "☻",
    incidents: "⚠",
    messaging: "✉",
    reports: "▦",
    monitoring: "◍",
    alerts: "✦",
    profile: "◉",
  };
  const sectionMeta = {
    overview: { title: "Clinical Overview", subtitle: "Cross-module snapshot for fast orientation." },
    patients: { title: "Patients Module", subtitle: "Patient roster, onboarding, and assignment management." },
    incidents: { title: "Incidents Module", subtitle: "Incident timeline, escalation visibility, and response context." },
    messaging: { title: "Messaging Module", subtitle: "Asynchronous doctor-patient communication center." },
    reports: { title: "Reports Module", subtitle: "Medical report drafting and export workspace." },
    monitoring: { title: "Monitoring Module", subtitle: "Vitals and trigger signal timeline for the patient." },
    alerts: { title: "Alerts Module", subtitle: "Alert history, status tracking, and action controls." },
    profile: { title: "Profile Module", subtitle: "Account details, credentials context, and workspace controls." },
  };
  const currentMeta = sectionMeta[activeSection] || sectionMeta.overview;

  return (
    <div className="app-shell app-shell--session">
      <aside className="sidebar" aria-label="Desktop navigation">
        <div className="brand-block">
          <p className="brand-eyebrow">InterSense</p>
          <h2>Clinical</h2>
        </div>
        <nav className="side-nav">
          {sections.map(([id, label]) => (
            <button
              key={id}
              className={`side-nav-item ${activeSection === id ? "active" : ""}`}
              onClick={() => setActiveSection(id)}
              title={label}
            >
              <span className="side-icon">{iconMap[id] || "•"}</span>
            </button>
          ))}
        </nav>
      </aside>

      <main className={`main-shell ${isOverview ? "" : "module-focus"}`.trim()}>
        <section className="shell">
          <header className="topbar">
            <div>
              <h1>Dashboard</h1>
              <p className="muted breadcrumb">Dashboard / {sectionLabel}</p>
            </div>
          </header>
          <div className="top-strip-nav panel account-strip">
            <div className="top-strip-links">
              {sections.map(([id, label]) => (
                <button
                  type="button"
                  key={`top-${id}`}
                  className={`top-strip-btn ${activeSection === id ? "active" : ""}`}
                  onClick={() => setActiveSection(id)}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="account-strip-actions">
              <button type="button" className="user-chip-btn" onClick={() => setActiveSection("profile")}>
                {me.username || me.email}
              </button>
              <button type="button" className="ghost-button" onClick={handleLogout}>Logout</button>
            </div>
          </div>
          {!isOverview && (
            <section className="module-stage-header panel">
              <div>
                <p className="eyebrow">{sectionLabel}</p>
                <h2>{currentMeta.title}</h2>
                <p className="muted">{currentMeta.subtitle}</p>
              </div>
              <button type="button" className="ghost-button" onClick={() => setActiveSection("overview")}>
                Back to Overview
              </button>
            </section>
          )}
          <div className={`workspace-surface role-${me.role} section-${activeSection}`}>
            {activeSection === "profile" ? (
              <section className="panel profile-page">
                <div className="panel-head">
                  <h3>Profile</h3>
                  <span className="panel-count">{me.role}</span>
                </div>
                <div className="profile-page-grid">
                  <article className="profile-page-card">
                    <p className="mini-label">Account Name</p>
                    <p className="mini-value">{me.username || "n/a"}</p>
                  </article>
                  <article className="profile-page-card">
                    <p className="mini-label">Email</p>
                    <p className="mini-value">{me.email}</p>
                  </article>
                  <article className="profile-page-card">
                    <p className="mini-label">Role</p>
                    <p className="mini-value">{me.role}</p>
                  </article>
                  <article className="profile-page-card">
                    <p className="mini-label">Linked Patients</p>
                    <p className="mini-value">{me.role === "doctor" ? assignments.length : 1}</p>
                  </article>
                </div>
                <div className="profile-actions">
                  <button type="button" className="ghost-button" onClick={handleRefresh} disabled={refreshing}>
                    {refreshing ? "Refreshing…" : "Refresh Data"}
                  </button>
                  <button type="button" className="ghost-button" onClick={handleLogout} disabled={refreshing}>
                    Logout
                  </button>
                </div>
              </section>
            ) : me.role === "doctor" ? (
              <DoctorDashboard
                me={me}
                activeSection={activeSection}
                assignments={assignments}
                vitals={vitals}
                triggers={triggers}
                voiceChecks={voiceChecks}
                incidents={incidents}
                alerts={alerts}
                onNavigateSection={setActiveSection}
              />
            ) : (
              <PatientDashboard
                me={me}
                assignments={assignments}
                activeSection={activeSection}
                vitals={vitals}
                triggers={triggers}
                voiceChecks={voiceChecks}
                incidents={incidents}
                alerts={alerts}
                onNavigateSection={setActiveSection}
              />
            )}
          </div>
        </section>
      </main>

      <nav className="mobile-bottom-nav" aria-label="Section navigation">
        {sections.map(([id, label]) => (
          <button
            key={`mbn-${id}`}
            type="button"
            className={`mobile-bottom-nav-item ${activeSection === id ? "active" : ""}`}
            onClick={() => setActiveSection(id)}
          >
            <span className="mobile-bottom-nav-icon" aria-hidden>
              {iconMap[id] || "•"}
            </span>
            <span className="mobile-bottom-nav-label">{label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}
