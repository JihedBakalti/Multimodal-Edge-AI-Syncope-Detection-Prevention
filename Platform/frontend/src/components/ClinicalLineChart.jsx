/**
 * Lightweight SVG time-series for vitals (HR, SpO2) and optional second metric.
 * Points sorted ascending by `logged_at` string (ISO).
 */
function parseTs(row) {
  const raw = row.logged_at;
  if (!raw) return NaN;
  const t = Date.parse(raw);
  return Number.isFinite(t) ? t : NaN;
}

export default function ClinicalLineChart({ vitals = [], voiceSeries = [], height = 220 }) {
  const w = 640;
  const pad = { l: 48, r: 52, t: 16, b: 36 };
  const iw = w - pad.l - pad.r;
  const ih = height - pad.t - pad.b;

  const sorted = [...vitals].filter((r) => r && r.logged_at).sort((a, b) => parseTs(a) - parseTs(b));
  const voiceSorted = [...voiceSeries].filter((r) => r && r.logged_at).sort((a, b) => parseTs(a) - parseTs(b));
  const xsV = sorted.map((r) => parseTs(r)).filter((t) => Number.isFinite(t));
  const xsVoice = voiceSorted.map((r) => parseTs(r)).filter((t) => Number.isFinite(t));
  const allT = [...xsV, ...xsVoice].filter((t) => Number.isFinite(t));
  if (!allT.length) {
    return (
      <div className="clinical-chart clinical-chart--empty">
        <p className="muted">No vitals or voice samples in this window.</p>
      </div>
    );
  }
  const x0 = Math.min(...allT);
  const x1 = Math.max(...allT) || x0 + 1;
  const sx = (t) => pad.l + ((t - x0) / (x1 - x0 || 1)) * iw;

  const hrs = sorted.map((r) => Number(r.heart_rate)).filter((n) => Number.isFinite(n));
  const spo = sorted.map((r) => (r.blood_oxygen != null ? Number(r.blood_oxygen) : null)).filter((n) => n != null && Number.isFinite(n));
  const hrMin = hrs.length ? Math.min(50, ...hrs) : 50;
  const hrMax = hrs.length ? Math.max(120, ...hrs) : 120;
  const syHr = (v) => pad.t + ih - ((v - hrMin) / (hrMax - hrMin || 1)) * ih;
  const spoMin = spo.length ? Math.min(90, ...spo) : 90;
  const spoMax = spo.length ? Math.max(100, ...spo) : 100;
  const sySpo = (v) => pad.t + ih - ((v - spoMin) / (spoMax - spoMin || 1)) * ih;

  const hrPoints = sorted
    .map((r) => {
      const t = parseTs(r);
      if (!Number.isFinite(t) || !Number.isFinite(Number(r.heart_rate))) return null;
      return `${sx(t)},${syHr(Number(r.heart_rate))}`;
    })
    .filter(Boolean)
    .join(" ");

  const spoPoints = sorted
    .map((r) => {
      const t = parseTs(r);
      const v = r.blood_oxygen != null ? Number(r.blood_oxygen) : null;
      if (!Number.isFinite(t) || v == null || !Number.isFinite(v)) return null;
      return `${sx(t)},${sySpo(v)}`;
    })
    .filter(Boolean)
    .join(" ");

  let voicePoints = "";
  if (xsVoice.length) {
    voicePoints = voiceSorted
      .map((r) => {
        const t = parseTs(r);
        const p = Number(r.prob_dysarthric);
        if (!Number.isFinite(t) || !Number.isFinite(p)) return null;
        return `${sx(t)},${pad.t + ih - p * ih}`;
      })
      .filter(Boolean)
      .join(" ");
  }

  return (
    <div className="clinical-chart">
      <svg viewBox={`0 0 ${w} ${height}`} className="clinical-chart-svg" preserveAspectRatio="xMidYMid meet">
        <rect x={0} y={0} width={w} height={height} className="clinical-chart-bg" />
        <text x={pad.l} y={14} className="clinical-chart-legend clinical-chart-legend-hr">
          Heart rate (BPM)
        </text>
        <text x={pad.l + 160} y={14} className="clinical-chart-legend clinical-chart-legend-spo">
          SpO2 (%)
        </text>
        {voicePoints ? (
          <text x={pad.l + 300} y={14} className="clinical-chart-legend clinical-chart-legend-voice">
            Voice dysarthria risk
          </text>
        ) : null}

        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const y = pad.t + ih * f;
          return <line key={f} x1={pad.l} x2={w - pad.r} y1={y} y2={y} className="clinical-chart-grid" />;
        })}

        {hrPoints ? <polyline fill="none" className="clinical-chart-line clinical-chart-line-hr" points={hrPoints} /> : null}
        {spoPoints ? <polyline fill="none" className="clinical-chart-line clinical-chart-line-spo" points={spoPoints} /> : null}
        {voicePoints ? <polyline fill="none" className="clinical-chart-line clinical-chart-line-voice" points={voicePoints} /> : null}

        <text x={8} y={pad.t + ih / 2} className="clinical-chart-axis-label" transform={`rotate(-90 8 ${pad.t + ih / 2})`}>
          HR / risk
        </text>
        <text x={w / 2} y={height - 6} className="clinical-chart-axis-label">
          Time
        </text>
      </svg>
    </div>
  );
}
