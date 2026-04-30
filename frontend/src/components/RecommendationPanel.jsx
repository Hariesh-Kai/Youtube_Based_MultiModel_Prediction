export default function RecommendationPanel({
  songs,
  suggestion,
  suggestionProvider,
  onSuggest,
  onRecommend,
  canRecommend,
  loading,
  suggesting,
  error
}) {
  return (
    <section className="card">
      <h2>3) Recommendations</h2>
      <button onClick={onSuggest} className="secondary" disabled={suggesting}>
        {suggesting ? 'Generating suggestion...' : 'Generate LLM Suggestion'}
      </button>
      {suggestion && <p className="suggestion"><strong>Suggestion ({suggestionProvider}):</strong> {suggestion}</p>}
      <button disabled={!canRecommend} onClick={onRecommend} className="primary">
        {loading ? 'Loading...' : 'Recommend Songs'}
      </button>
      {!canRecommend && <p className="muted">Upload an image and choose at least one language.</p>}
      {error && <p className="error">{error}</p>}
      <ul className="songs">
        {songs.map((song, idx) => (
          <li key={`${song.title}-${idx}`}><strong>{song.title}</strong><span>{song.reason}</span></li>
        ))}
      </ul>
    </section>
  )
}
