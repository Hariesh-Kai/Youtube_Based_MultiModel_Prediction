import { Link } from 'react-router-dom'

export default function HomePage() {
  return (
    <section className="card hero">
      <h2>Your mood. Your moment. Your music.</h2>
      <p>Get AI suggestions using emotion, profile, and environment context.</p>
      <div className="actions">
        <Link className="primary" to="/discover">Start Discovering</Link>
        <Link className="secondary" to="/library">View Library</Link>
      </div>
    </section>
  )
}
