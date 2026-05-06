import { useState, useEffect, useCallback, useRef } from 'react';
import styles from './InterSenseSimulator.module.css';

const STORAGE_KEY = 'intersense.simulator.panelOpen';
const STORAGE_SESSION = 'intersense.simulator.sessionId';

const DEFAULT_SIM_HR = 78;
const DEFAULT_SIM_SPO2 = 98;

function getOrCreateSessionId() {
  try {
    let sid = sessionStorage.getItem(STORAGE_SESSION);
    if (!sid && typeof crypto !== 'undefined' && crypto.randomUUID) {
      sid = crypto.randomUUID();
      sessionStorage.setItem(STORAGE_SESSION, sid);
    }
    return sid || 'default-session';
  } catch {
    return 'default-session';
  }
}

/** Human-readable lines: which channel (SpO₂ vs HR) drove the clinical alert. */
function buildVitalTriggerNarrative(clinical) {
  if (!clinical || typeof clinical !== 'object') {
    return {
      title: 'Orchestration triggered',
      lines: ['Vital rules crossed the server alert threshold.']
    };
  }
  const ss = Number(clinical.spo2_score ?? 0);
  const bs = Number(clinical.bpm_score ?? 0);
  const score = Number(clinical.score ?? Math.max(ss, bs));
  const spo2State = String(clinical.spo2_state ?? '—');
  const bpmState = String(clinical.bpm_state ?? '—');
  const tie = Math.abs(ss - bs) < 0.03;
  const lines = [];
  if (ss < 0.02 && bs < 0.02) {
    lines.push(`Combined clinical score ${score} (rule engine).`);
  } else if (tie && ss > 0.02) {
    lines.push(`Blood oxygen (SpO₂): ${spo2State} (channel score ${ss}).`);
    lines.push(`Heart rate: ${bpmState} (channel score ${bs}).`);
    lines.push('Both channels contributed about equally to the alert.');
  } else if (ss > bs) {
    lines.push(`Primary driver: blood oxygen (SpO₂) — ${spo2State} (channel score ${ss}).`);
    lines.push(`Heart rate pattern: ${bpmState} (channel score ${bs}).`);
  } else {
    lines.push(`Primary driver: heart rate — ${bpmState} (channel score ${bs}).`);
    lines.push(`Blood oxygen (SpO₂) pattern: ${spo2State} (channel score ${ss}).`);
  }
  lines.push(`Alert score = max(SpO₂, HR) = ${score} (above server threshold).`);
  return { title: 'Orchestration triggered', lines };
}

const TIER_CLASS = {
  critical: styles.tierCritical,
  warning: styles.tierWarning,
  caution: styles.tierCaution,
  info: styles.tierInfo,
  normal: styles.tierNormal,
  error: styles.tierError
};

/** Backend orchestrator `state` → display tier + labels. */
function buildPipelineStatus(stateRaw) {
  const key = String(stateRaw ?? 'unknown');
  const map = {
    critical_emergency: { code: 'CRITICAL', label: 'Critical emergency — escalation path', tier: 'critical' },
    warning: { code: 'WARNING', label: 'Warning — proactive voice / checks', tier: 'warning' },
    no_action: { code: 'NO ACTION', label: 'No human confirmed — voice safety protocol', tier: 'info' },
    normal: { code: 'NORMAL', label: 'System normal', tier: 'normal' },
    camera_scan_failed: { code: 'ERROR', label: 'Camera scan failed', tier: 'error' },
    syncope_runtime_failed: { code: 'ERROR', label: 'Syncope runtime failed', tier: 'error' },
    body_fall_runtime_failed: { code: 'ERROR', label: 'Body fall detector failed', tier: 'error' },
    timeout: { code: 'TIMEOUT', label: 'Orchestrator timed out', tier: 'error' },
    unknown: { code: 'UNKNOWN', label: 'Unknown orchestrator state', tier: 'info' }
  };
  return map[key] || { code: key.toUpperCase().slice(0, 12), label: key.replace(/_/g, ' '), tier: 'info' };
}

/** Clinical spo2_state / bpm_state → worst-case tier for the banner. */
function buildVitalsRulesStatus(clinical) {
  if (!clinical || typeof clinical !== 'object') {
    return {
      code: '—',
      label: 'SpO₂ / HR rules not evaluated on this path (manual orchestrator only).',
      tier: 'info'
    };
  }
  const spo = String(clinical.spo2_state ?? '—');
  const bpm = String(clinical.bpm_state ?? '—');
  const blob = `${spo} ${bpm}`.toUpperCase();
  let tier = 'normal';
  if (blob.includes('CRITICAL')) tier = 'critical';
  else if (blob.includes('HIGH RISK') || blob.includes('FAINTING')) tier = 'warning';
  else if (blob.includes('EARLY') || blob.includes('DECLINING') || blob.includes('RISING')) tier = 'caution';
  const tierUpper =
    tier === 'critical'
      ? 'CRITICAL'
      : tier === 'warning'
        ? 'WARNING'
        : tier === 'caution'
          ? 'CAUTION'
          : tier === 'normal'
            ? 'NORMAL'
            : 'INFO';
  return {
    code: tierUpper,
    label: `SpO₂: ${spo} · HR: ${bpm}`,
    tier
  };
}

const CHANNEL_TIER_CLASS = {
  normal: styles.vitalsChannelNormal,
  caution: styles.vitalsChannelCaution,
  warning: styles.vitalsChannelWarning,
  critical: styles.vitalsChannelCritical
};

/** Per-channel color tier from one rule label (SpO₂ or HR). */
function tierFromRuleLabel(label) {
  const s = String(label || '').toUpperCase();
  if (s.includes('CRITICAL')) return 'critical';
  if (s.includes('HIGH RISK') || s.includes('FAINTING')) return 'warning';
  if (s.includes('EARLY') || s.includes('DECLINING') || s.includes('RISING')) return 'caution';
  return 'normal';
}

function VitalsRuleStatusCard({ clinical, heartRate, bloodOxygen }) {
  if (!clinical || typeof clinical !== 'object') return null;
  const spoTier = tierFromRuleLabel(clinical.spo2_state);
  const bpmTier = tierFromRuleLabel(clinical.bpm_state);
  const overall = buildVitalsRulesStatus(clinical);
  return (
    <div className={styles.vitalsStatusCard}>
      <div className={styles.vitalsStatusTitle}>Clinical rule engine (last sample)</div>
      <div className={styles.vitalsSliderEcho}>
        Values sent with that sample: <strong>{heartRate} bpm</strong> · <strong>{bloodOxygen}% SpO₂</strong>
        <span className={styles.vitalsSliderHint}>
          Rule labels can differ from the sliders: the engine uses baselines and trends over recent samples
          (e.g. HR can show CRITICAL FAINTING after a sharp drop vs baseline even if SpO₂ looks fine).
        </span>
      </div>
      <div className={styles.vitalsChannelsRow}>
        <div className={`${styles.vitalsChannel} ${CHANNEL_TIER_CLASS[spoTier] ?? styles.vitalsChannelNormal}`}>
          <div className={styles.vitalsChannelLbl}>SpO₂ rule</div>
          <div className={styles.vitalsChannelState}>{String(clinical.spo2_state ?? '—')}</div>
          <div className={styles.vitalsChannelScore}>Channel score {clinical.spo2_score ?? '—'}</div>
        </div>
        <div className={`${styles.vitalsChannel} ${CHANNEL_TIER_CLASS[bpmTier] ?? styles.vitalsChannelNormal}`}>
          <div className={styles.vitalsChannelLbl}>Heart rate rule</div>
          <div className={styles.vitalsChannelState}>{String(clinical.bpm_state ?? '—')}</div>
          <div className={styles.vitalsChannelScore}>Channel score {clinical.bpm_score ?? '—'}</div>
        </div>
      </div>
      <div className={`${styles.vitalsOverall} ${TIER_CLASS[overall.tier] ?? styles.tierInfo}`}>
        <span className={styles.vitalsOverallLbl}>Combined alert</span>
        <span className={styles.vitalsOverallCode}>{overall.code}</span>
        <span className={styles.vitalsOverallMeta}>
          max(SpO₂ score, HR score) = {clinical.score ?? '—'}
        </span>
      </div>
    </div>
  );
}

function OrchestrationStatusChips({ pipelineState, clinical }) {
  const pipe = buildPipelineStatus(pipelineState);
  const vit = buildVitalsRulesStatus(clinical);
  return (
    <div className={styles.statusStrip}>
      <div className={`${styles.statusChip} ${TIER_CLASS[pipe.tier] ?? styles.tierInfo}`}>
        <span className={styles.statusChipLbl}>Orchestrator status</span>
        <span className={styles.statusChipVal}>{pipe.code}</span>
        <span className={styles.statusChipSub}>{pipe.label}</span>
      </div>
      <div className={`${styles.statusChip} ${TIER_CLASS[vit.tier] ?? styles.tierInfo}`}>
        <span className={styles.statusChipLbl}>Vitals rule status</span>
        <span className={styles.statusChipVal}>{vit.code}</span>
        <span className={styles.statusChipSub}>{vit.label}</span>
      </div>
    </div>
  );
}

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

  const [sessionId] = useState(() => getOrCreateSessionId());
  const [heartRate, setHeartRate] = useState(78);
  const [bloodOxygen, setBloodOxygen] = useState(98);
  const [streamVitals, setStreamVitals] = useState(false);
  const [clinicalSnapshot, setClinicalSnapshot] = useState(null);
  const [anomalyValue, setAnomalyValue] = useState(0.45);
  const [dlRiskScore, setDlRiskScore] = useState(0.2);
  const [wearableAnomaly, setWearableAnomaly] = useState(false);
  const [humanDetected, setHumanDetected] = useState(false);
  const [faintingDetected, setFaintingDetected] = useState(false);
  const [status, setStatus] = useState('');
  const [verboseSteps, setVerboseSteps] = useState([]);
  const [liveHeartRate, setLiveHeartRate] = useState(null);
  const [liveBloodOxygen, setLiveBloodOxygen] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [orchNotice, setOrchNotice] = useState(null);
  const [lastOrchestratorSnapshot, setLastOrchestratorSnapshot] = useState(null);
  const orchNoticeIdRef = useRef(0);
  const streamVitalsRef = useRef(false);
  const heartRateRef = useRef(78);
  const bloodOxygenRef = useRef(98);
  const humanDetectedRef = useRef(false);
  const faintingDetectedRef = useRef(false);

  /** After a critical escalation from the vital stream (or server session reset), stop auto-send and restore benign simulator inputs. */
  const applySimulatorSafeDefaultsAfterCritical = useCallback(() => {
    setStreamVitals(false);
    setHeartRate(DEFAULT_SIM_HR);
    setBloodOxygen(DEFAULT_SIM_SPO2);
    heartRateRef.current = DEFAULT_SIM_HR;
    bloodOxygenRef.current = DEFAULT_SIM_SPO2;
    setAnomalyValue(0.45);
    setDlRiskScore(0.2);
    setWearableAnomaly(false);
    setHumanDetected(false);
    setFaintingDetected(false);
    humanDetectedRef.current = false;
    faintingDetectedRef.current = false;
    setClinicalSnapshot(null);
  }, []);

  useEffect(() => {
    streamVitalsRef.current = streamVitals;
  }, [streamVitals]);

  useEffect(() => {
    heartRateRef.current = heartRate;
  }, [heartRate]);
  useEffect(() => {
    bloodOxygenRef.current = bloodOxygen;
  }, [bloodOxygen]);
  useEffect(() => {
    humanDetectedRef.current = humanDetected;
  }, [humanDetected]);
  useEffect(() => {
    faintingDetectedRef.current = faintingDetected;
  }, [faintingDetected]);

  const showOrchestrationAlert = useCallback((detail) => {
    const id = ++orchNoticeIdRef.current;
    setOrchNotice({ id, ...detail });
    setLastOrchestratorSnapshot({
      pipelineState: detail.pipelineState,
      clinical: detail.clinical ?? null
    });
  }, []);

  useEffect(() => {
    if (!orchNotice) return undefined;
    const { id } = orchNotice;
    const t = setTimeout(() => {
      setOrchNotice((cur) => (cur && cur.id === id ? null : cur));
    }, 16000);
    return () => clearTimeout(t);
  }, [orchNotice]);

  useEffect(() => {
    let mounted = true;
    const fetchLiveVitals = async () => {
      try {
        const q = new URLSearchParams({ user_id: '1', session_id: sessionId });
        const res = await fetch(`${apiBaseUrl}/api/live-vitals?${q}`);
        const data = await res.json();
        if (!mounted) return;
        if (res.ok && data?.ok) {
          setLiveHeartRate(
            Number.isFinite(Number(data?.effective_heart_rate)) ? Number(data.effective_heart_rate) : null
          );
          setLiveBloodOxygen(
            Number.isFinite(Number(data?.effective_blood_oxygen)) ? Number(data.effective_blood_oxygen) : null
          );
        }
      } catch {
        // keep last shown vitals
      }
    };

    fetchLiveVitals();
    const timer = setInterval(fetchLiveVitals, 3000);
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, [apiBaseUrl, sessionId]);

  const postVitalSample = useCallback(
    async (timestampSec) => {
      const ts = typeof timestampSec === 'number' ? timestampSec : Date.now() / 1000;
      const res = await fetch(`${apiBaseUrl}/api/vital-sample`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          heart_rate: heartRateRef.current,
          blood_oxygen: bloodOxygenRef.current,
          timestamp: ts,
          user_id: '1',
          session_id: sessionId,
          auto_orchestrate: true,
          human_detected: humanDetectedRef.current,
          fainting_detected: faintingDetectedRef.current
        })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail || 'Vital sample failed');
      }
      const orchSt = String(data?.orchestrator_result?.state || '').toLowerCase();
      if (data?.clinical_session_reset || orchSt === 'critical_emergency') {
        applySimulatorSafeDefaultsAfterCritical();
      } else if (data?.clinical && !data?.deduplicated) {
        setClinicalSnapshot(data.clinical);
      }
      return data;
    },
    [apiBaseUrl, sessionId, applySimulatorSafeDefaultsAfterCritical]
  );

  useEffect(() => {
    if (!streamVitals) return undefined;
    let cancelled = false;
    let timeoutId = null;

    const scheduleNext = () => {
      if (timeoutId != null) window.clearTimeout(timeoutId);
      timeoutId = window.setTimeout(() => {
        void runTick();
      }, 2000);
    };

    async function runTick() {
      if (cancelled || !streamVitalsRef.current) return;
      try {
        const data = await postVitalSample(Date.now() / 1000);
        if (cancelled || !streamVitalsRef.current) return;
        if (data?.orchestrator_triggered && data?.orchestrator_result) {
          const orch = data.orchestrator_result;
          const steps = Array.isArray(orch?.verbose_steps) ? orch.verbose_steps : [];
          let msg = orch?.message || 'Orchestrator ran from vital anomaly.';
          if (data?.clinical_session_reset || String(data?.orchestrator_result?.state || '').toLowerCase() === 'critical_emergency') {
            msg +=
              ' Simulator reset to safe defaults and vital stream stopped. Re-enable streaming only when you want a new run.';
          }
          setStatus(msg);
          setVerboseSteps(
            steps.length
              ? [`[vital-stream] Clinical score ${(data.clinical || {}).score}`, ...steps]
              : [`[vital-stream] Orchestrator state=${orch?.state || 'unknown'}`]
          );
          const narrative = buildVitalTriggerNarrative(data.clinical);
          showOrchestrationAlert({
            title: narrative.title,
            pipelineState: orch?.state,
            clinical: data.clinical ?? null,
            orchMessage: orch?.message || null,
            lines: narrative.lines,
            source: 'vital-stream'
          });
        }
      } catch (e) {
        if (!cancelled) {
          setStatus(`Vital stream error: ${e?.message || e}`);
        }
      } finally {
        if (!cancelled && streamVitalsRef.current) {
          scheduleNext();
        }
      }
    }

    runTick();
    return () => {
      cancelled = true;
      if (timeoutId != null) {
        window.clearTimeout(timeoutId);
        timeoutId = null;
      }
    };
  }, [streamVitals, postVitalSample, showOrchestrationAlert]);

  const sendOneVitalSample = async () => {
    setIsLoading(true);
    setStatus('Sending vital sample…');
    try {
      const data = await postVitalSample();
      if (data?.deduplicated) {
        setStatus('Sample ignored (timestamp must increase). Move sliders or wait.');
        return;
      }
      const c = data.clinical || {};
      const minCh = Number.isFinite(Number(data?.clinical_escalation_min_channel_score))
        ? Number(data.clinical_escalation_min_channel_score)
        : 1.0;
      let line = data?.orchestrator_triggered
        ? 'Clinical anomaly → orchestrator triggered.'
        : `Clinical score ${c.score ?? '—'} (${c.spo2_state || '—'} / ${c.bpm_state || '—'})`;
      if (!data?.orchestrator_triggered && data?.post_critical_auto_orch_paused) {
        const rem = Math.ceil(Number(data?.post_critical_auto_orch_remaining_sec) || 0);
        line = `Recent critical escalation: auto-orchestrate paused (~${rem}s left). Tune ORCH_VITAL_POST_CRITICAL_SILENCE_SEC on the server if needed.`;
      } else if (!data?.orchestrator_triggered && data?.orchestration_suppressed_repeat) {
        line = `At full escalation (max channel score ≥ ${minCh}) — repeat orchestration suppressed until scores drop below ${minCh} (same simulator values won’t re-alert).`;
      } else if (!data?.orchestrator_triggered && data?.clinical_score_above_threshold && !data?.clinical_channel_gate) {
        line = `Combined score ${c.score ?? '—'} may be elevated, but auto-orchestration requires max(SpO₂ score, HR score) ≥ ${minCh} (CRITICAL = 1.0). Current: SpO₂ ${c.spo2_score ?? '—'}, HR ${c.bpm_score ?? '—'}.`;
      }
      if (
        data?.clinical_session_reset ||
        String(data?.orchestrator_result?.state || '').toLowerCase() === 'critical_emergency'
      ) {
        line +=
          ' Session reset / critical path — simulator back to safe defaults and vital stream stopped; send fresh samples or re-enable streaming when ready.';
      }
      setStatus(line);
      if (data?.orchestrator_triggered && data?.orchestrator_result) {
        const orch = data.orchestrator_result;
        const steps = Array.isArray(orch?.verbose_steps) ? orch.verbose_steps : [];
        setVerboseSteps(steps.length ? steps : [`state=${orch?.state || 'unknown'}`]);
        const narrative = buildVitalTriggerNarrative(data.clinical);
        showOrchestrationAlert({
          title: narrative.title,
          pipelineState: orch?.state,
          clinical: data.clinical ?? null,
          orchMessage: orch?.message || null,
          lines: narrative.lines,
          source: 'vital-sample'
        });
      } else {
        setVerboseSteps([
            `[vital-sample] score=${c.score} spo2_sc=${c.spo2_score} bpm_sc=${c.bpm_score} thr=${data.clinical_threshold ?? '—'} ` +
            `chGate=${Boolean(data?.clinical_channel_gate)} minCh=${Number(data?.clinical_escalation_min_channel_score) || 1} ` +
            `latch=${Boolean(data?.orchestration_latch_active)} supRep=${Boolean(data?.orchestration_suppressed_repeat)}`
        ]);
      }
    } catch (error) {
      setStatus(`Error: ${error?.message || 'Network error'}`);
      setVerboseSteps([]);
    } finally {
      setIsLoading(false);
    }
  };

  const triggerOrchestrator = async () => {
    setIsLoading(true);
    setStatus('Sending simulation...');
    setVerboseSteps([]);

    try {
      const payload = {
        heart_rate: heartRate,
        blood_oxygen: bloodOxygen,
        prefer_simulator_vitals: true,
        anomaly_value: Number(anomalyValue),
        dl_risk_score: Number(dlRiskScore),
        wearable_anomaly: wearableAnomaly,
        human_detected: humanDetected,
        fainting_detected: faintingDetected,
        user_id: '1',
        session_id: sessionId
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
      const steps = Array.isArray(data?.verbose_steps) ? data.verbose_steps : [];
      setVerboseSteps(steps.length ? steps : [`[client] No backend verbose steps returned. state=${data?.state || 'unknown'}`]);
      setLiveHeartRate(
        Number.isFinite(Number(data?.effective_heart_rate)) ? Number(data.effective_heart_rate) : null
      );
      setLiveBloodOxygen(
        Number.isFinite(Number(data?.effective_blood_oxygen)) ? Number(data.effective_blood_oxygen) : null
      );
      const st = data?.state;
      setLastOrchestratorSnapshot({
        pipelineState: st ?? 'unknown',
        clinical: null
      });
      if (st && st !== 'normal') {
        const lines = [];
        if (wearableAnomaly) {
          lines.push('Reason: "Wearable anomaly" was on — orchestrator ran the camera / safety workflow.');
        } else {
          lines.push('Orchestrator returned a non-normal state (see verbose trace for details).');
        }
        if (data.message) lines.push(String(data.message));
        showOrchestrationAlert({
          title: 'Orchestration started',
          pipelineState: st,
          clinical: null,
          orchMessage: data.message || null,
          lines,
          source: 'manual-simulate'
        });
      }
      if (String(st || '').toLowerCase() === 'critical_emergency') {
        applySimulatorSafeDefaultsAfterCritical();
      }
    } catch (error) {
      const message = error?.message || 'Network error';
      if (message.toLowerCase().includes('failed to fetch')) {
        setStatus('Error: cannot reach orchestrator API. Start: uvicorn orchestrator_api:app --host 127.0.0.1 --port 8000');
      } else {
        setStatus(`Error: ${message}`);
      }
      setVerboseSteps([`[client] ${message}`]);
      setLiveHeartRate(null);
      setLiveBloodOxygen(null);
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
          className={`${styles.panel} ${orchNotice ? styles.panelOrchestrationHot : ''}`.trim()}
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

          {orchNotice && (
            <div className={styles.orchestrationBanner} role="alert" aria-live="assertive">
              <div className={styles.orchBanIcon} aria-hidden>
                !
              </div>
              <div className={styles.orchBanBody}>
                <div className={styles.orchBanTitle}>{orchNotice.title}</div>
                <OrchestrationStatusChips
                  pipelineState={orchNotice.pipelineState}
                  clinical={orchNotice.clinical}
                />
                {orchNotice.orchMessage ? (
                  <div className={styles.orchBanLine} style={{ fontWeight: 600, marginBottom: '6px' }}>
                    {orchNotice.orchMessage}
                  </div>
                ) : null}
                {Array.isArray(orchNotice.lines) &&
                  orchNotice.lines.map((line, idx) => (
                    <div key={`${idx}-${line}`} className={styles.orchBanLine}>
                      {line}
                    </div>
                  ))}
                {orchNotice.source && (
                  <div className={styles.orchBanMeta}>Source: {orchNotice.source}</div>
                )}
              </div>
              <button
                type="button"
                className={styles.orchBanDismiss}
                onClick={() => setOrchNotice(null)}
                aria-label="Dismiss alert"
              >
                ×
              </button>
            </div>
          )}

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
                  <span className={styles.fieldLabel}>Blood oxygen (SpO₂)</span>
                  <span className={styles.fieldValue}>{bloodOxygen}%</span>
                </div>
                <input
                  className={styles.range}
                  type="range"
                  min={75}
                  max={100}
                  value={bloodOxygen}
                  onChange={(e) => setBloodOxygen(Number(e.target.value))}
                  aria-valuetext={`${bloodOxygen} percent oxygen saturation`}
                />
              </div>

              {clinicalSnapshot ? (
                <VitalsRuleStatusCard
                  clinical={clinicalSnapshot}
                  heartRate={heartRate}
                  bloodOxygen={bloodOxygen}
                />
              ) : null}

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

            <section className={styles.section} aria-labelledby="sec-vital-stream">
              <h3 id="sec-vital-stream" className={styles.sectionLabel}>
                Vital stream (clinical model)
              </h3>
              <p style={{ fontSize: '11px', opacity: 0.75, margin: '0 0 8px' }}>
                Each sample uses a monotonic timestamp. Auto-orchestration runs only when{' '}
                <strong>max(SpO₂ score, HR score) ≥ 1.0</strong> (CRITICAL on a channel). After a{' '}
                <strong>critical_emergency</strong> run, the server pauses further auto-orchestration for{' '}
                <strong>ORCH_VITAL_POST_CRITICAL_SILENCE_SEC</strong> (default 90s) so tracker resets cannot immediately
                re-trigger on the same extreme sliders; the UI also stops streaming and resets vitals to safe defaults.
                Tune with ORCH_VITAL_ORCHESTRATE_MIN_CHANNEL_SCORE (e.g. 0.8 to allow HIGH RISK).
              </p>
              <ToggleField
                id="sim-stream-vitals"
                label="Stream samples every 2s"
                value={streamVitals}
                onChange={setStreamVitals}
              />
              <button
                type="button"
                className={styles.secondaryBtn}
                onClick={sendOneVitalSample}
                disabled={isLoading || streamVitals}
              >
                Send one vital sample now
              </button>
            </section>

            <section className={styles.section} aria-labelledby="sec-live-vitals">
              <h3 id="sec-live-vitals" className={styles.sectionLabel}>
                Live Patient Vitals
              </h3>
              <div
                style={{
                  border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: '10px',
                  padding: '10px',
                  background: 'rgba(0,0,0,0.22)'
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    marginBottom: '8px',
                    fontSize: '12px'
                  }}
                >
                  <span style={{ color: 'rgba(226,232,240,0.88)' }}>Heart Rate</span>
                  <strong style={{ color: '#7dd3fc' }}>
                    {liveHeartRate === null ? '--' : `${liveHeartRate} bpm`}
                  </strong>
                </div>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: '12px'
                  }}
                >
                  <span style={{ color: 'rgba(226,232,240,0.88)' }}>Blood Oxygen</span>
                  <strong style={{ color: '#86efac' }}>
                    {liveBloodOxygen === null ? '--' : `${liveBloodOxygen}%`}
                  </strong>
                </div>
              </div>
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
              <div
                style={{
                  marginTop: '10px',
                  border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: '10px',
                  padding: '10px',
                  maxHeight: '180px',
                  overflowY: 'auto',
                  background: 'rgba(0,0,0,0.28)',
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
                  fontSize: '0.72rem',
                  lineHeight: 1.4
                }}
              >
                <div style={{ marginBottom: '6px', opacity: 0.8 }}>Verbose Trace</div>
                {verboseSteps.length === 0 ? (
                  <div style={{ opacity: 0.55 }}>No trace yet.</div>
                ) : (
                  verboseSteps.map((line, idx) => (
                    <div key={`${idx}-${line}`}>{line}</div>
                  ))
                )}
              </div>
              {lastOrchestratorSnapshot && (
                <div className={styles.persistedStatus}>
                  <div className={styles.persistedStatusLabel}>Last orchestrator run — status</div>
                  <OrchestrationStatusChips
                    pipelineState={lastOrchestratorSnapshot.pipelineState}
                    clinical={lastOrchestratorSnapshot.clinical}
                  />
                </div>
              )}
            </section>
          </div>
        </aside>
      )}
    </div>
  );
}
