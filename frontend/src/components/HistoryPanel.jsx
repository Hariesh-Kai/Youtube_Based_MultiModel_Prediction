export default function HistoryPanel({ history, onUse }) {
  return (
    <section className="card">
      <h2>Recent Sessions</h2>
      {history.length === 0 && <p className="muted">No saved sessions yet.</p>}
      <div className="history-list">
        {history.map((item) => (
          <button key={item.id} className="history-item" onClick={() => onUse(item)} type="button">
            <strong>{item.emotion} • {item.environment}</strong>
            <span>{item.age} / {item.gender} / {item.languages.join(', ')}</span>
          </button>
        ))}
      </div>
    </section>
  )
}
