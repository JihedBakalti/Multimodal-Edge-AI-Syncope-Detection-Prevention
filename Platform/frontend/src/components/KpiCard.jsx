export default function KpiCard({ label, value, tone = "neutral", hint = "" }) {
  return (
    <article className={`kpi-card ${tone}`}>
      <p className="kpi-label">{label}</p>
      <p className="kpi-value">{value}</p>
      {hint ? <p className="kpi-hint">{hint}</p> : null}
    </article>
  );
}
