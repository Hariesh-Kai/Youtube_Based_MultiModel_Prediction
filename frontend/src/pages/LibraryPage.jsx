export default function LibraryPage() {
  const history = JSON.parse(localStorage.getItem('emotion_music_history_v1') || '[]')
  return (
    <section className="card">
      <h3>Library</h3>
      {history.length === 0 ? <p className="muted">No saved sessions yet.</p> : history.map((item)=> (
        <div key={item.id} className="library-item">
          <strong>{item.emotion} • {item.environment}</strong>
          <p>{item.age} / {item.gender} / {item.languages?.join(', ')}</p>
        </div>
      ))}
    </section>
  )
}
