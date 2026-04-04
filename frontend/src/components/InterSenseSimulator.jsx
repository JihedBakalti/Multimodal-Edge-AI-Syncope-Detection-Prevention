import { useState, useEffect } from 'react';
import styles from './InterSenseSimulator.module.css';

const STORAGE_KEY = 'intersense.simulator.panelOpen';

function ToggleField({ id, label, value, onChange }) {
  return (
    <label htmlFor={id} className={styles.switchRow}>
      <span className={styles.switchLabel}>{label}</span>
      <span className={styles.switch}>
        <input
          id={id}
          type="checkbox"
          role="switch"
          aria-checked={value}
          checked={value}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span className={styles.track} aria-hidden>
          <span className={styles.knob} />
        </span>
      </span>
    </label>
  );
}

export default function InterSenseSimulator() {
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
  const [expanded, setExpanded] = useState(() => {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (raw === null) return true;
      return raw === '1';
    } catch {
      return true;
    }
  });

  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY, expanded ? '1' : '0');
    } catch {
      /* ignore */
    }
  }, [expanded]);

  const [heartRate, setHeartRate] = useState(78);
  const [anomalyValue, setAnomalyValue] = useState(0.45);
  const [dlRiskScore, setDlRiskScore] = useState(0.2);
  const [wearableAnomaly, setWearableAnomaly] = useState(false);
  const [humanDetected, setHumanDetected] = useState(false);
  const [faintingDetected, setFaintingDetected] = useState(false);
  const [status, setStatus] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const triggerOrchestrator = async () => {
    setIsLoading(true);
    setStatus('Sending simulation...');

    try {
      const payload = {
        heart_rate: heartRate,
        anomaly_value: Number(anomalyValue),
        dl_risk_score: Number(dlRiskScore),
        wearable_anomaly: wearableAnomaly,
        human_detected: humanDetected,
        fainting_detected: faintingDetected
      };

      const res = await fetch(`${apiBaseUrl}/api/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail || 'Failed to trigger orchestrator');
      }

      setStatus(data.message || 'Orchestrator triggered.');
    } catch (error) {
      const message = error?.message || 'Network error';
      if (message.toLowerCase().includes('failed to fetch')) {
        setStatus('Error: cannot reach orchestrator API. Start: uvicorn orchestrator_api:app --host 127.0.0.1 --port 8000');
      } else {
        setStatus(`Error: ${message}`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const statusClass =
    status.startsWith('Error') || status.startsWith('error')
      ? styles.statusError
      : status && !status.startsWith('Sending')
        ? styles.statusOk
        : '';

  return (
    <div className={styles.dock}>
      {!expanded && (
        <button
          type="button"
          className={styles.revealTab}
          onClick={() => setExpanded(true)}
          aria-expanded={false}
          aria-controls="intersense-simulator-panel"
          title="Show sensor simulation"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="M15 18l-6-6 6-6" />
          </svg>
          <span>Simulation</span>
        </button>
      )}

      {expanded && (
        <aside
          id="intersense-simulator-panel"
          className={styles.panel}
          aria-label="InterSense sensor simulation"
        >
          <header className={styles.header}>
            <div className={styles.headerText}>
              <h2 className={styles.title}>InterSense</h2>
              <p className={styles.subtitle}>Captors &amp; model inputs</p>
            </div>
            <button
              type="button"
              className={styles.iconBtn}
              onClick={() => setExpanded(false)}
              aria-expanded
              aria-controls="intersense-simulator-panel"
              title="Hide panel"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M9 18l6-6-6-6" />
              </svg>
            </button>
          </header>

          <div className={styles.body}>
            <section className={styles.section} aria-labelledby="sec-vitals">
              <h3 id="sec-vitals" className={styles.sectionLabel}>
                Vitals &amp; scores
              </h3>

              <div className={styles.field}>
                <div className={styles.fieldTop}>
                  <span className={styles.fieldLabel}>Heart rate</span>
                  <span className={styles.fieldValue}>{heartRate} bpm</span>
                </div>
                <input
                  className={styles.range}
                  type="range"
                  min={40}
                  max={180}
                  value={heartRate}
                  onChange={(e) => setHeartRate(Number(e.target.value))}
                  aria-valuetext={`${heartRate} beats per minute`}
                />
              </div>

              <div className={styles.field}>
                <div className={styles.fieldTop}>
                  <span className={styles.fieldLabel}>Anomaly signal</span>
                  <span className={styles.fieldValue}>{anomalyValue.toFixed(2)}</span>
                </div>
                <input
                  className={styles.range}
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={anomalyValue}
                  onChange={(e) => setAnomalyValue(Number(e.target.value))}
                />
              </div>

              <div className={styles.field}>
                <div className={styles.fieldTop}>
                  <span className={styles.fieldLabel}>DL fainting risk</span>
                  <span className={styles.fieldValue}>{dlRiskScore.toFixed(2)}</span>
                </div>
                <input
                  className={styles.range}
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={dlRiskScore}
                  onChange={(e) => setDlRiskScore(Number(e.target.value))}
                />
              </div>
            </section>

            <section className={styles.section} aria-labelledby="sec-sensors">
              <h3 id="sec-sensors" className={styles.sectionLabel}>
                Sensor flags
              </h3>
              <ToggleField
                id="sim-wearable"
                label="Wearable anomaly"
                value={wearableAnomaly}
                onChange={setWearableAnomaly}
              />
              <ToggleField
                id="sim-human"
                label="Camera: human detected"
                value={humanDetected}
                onChange={setHumanDetected}
              />
              <ToggleField
                id="sim-faint"
                label="Camera: fainting detected"
                value={faintingDetected}
                onChange={setFaintingDetected}
              />
            </section>

            <section className={styles.section} aria-labelledby="sec-action">
              <h3 id="sec-action" className={styles.sectionLabel}>
                Orchestrator
              </h3>
              <button
                type="button"
                className={styles.primaryBtn}
                onClick={triggerOrchestrator}
                disabled={isLoading}
              >
                {isLoading ? 'Sending…' : 'Send to orchestrator'}
              </button>
              <p className={`${styles.status} ${statusClass}`.trim()} role="status">
                {status}
              </p>
            </section>
          </div>
        </aside>
      )}
    </div>
  );
}
