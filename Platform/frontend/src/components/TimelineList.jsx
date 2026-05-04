export default function TimelineList({ title, items, renderItem }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h3>{title}</h3>
        <span className="panel-count">{items.length}</span>
      </div>
      <div className="timeline">
        {items.length === 0 && <p className="muted timeline-empty">No data available.</p>}
        {items.map((item) => (
          <div key={item.id} className="timeline-row">
            {renderItem(item)}
          </div>
        ))}
      </div>
    </section>
  );
}
