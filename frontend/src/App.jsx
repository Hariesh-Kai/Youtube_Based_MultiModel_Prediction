import { useEffect, useMemo, useState } from 'react'
import HistoryPanel from './components/HistoryPanel'
import MobileNav from './components/MobileNav'
import RecommendationPanel from './components/RecommendationPanel'
import { fetchRecommendations, fetchSuggestion } from './services/api'

const LANGUAGES = ['English', 'Hindi', 'Tamil', 'Spanish', 'French', 'German', 'Chinese', 'Japanese']
const HISTORY_KEY = 'emotion_music_history_v1'

export default function App() {
  const [step, setStep] = useState(1)
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState('')
  const [languages, setLanguages] = useState(['English'])
  const [age, setAge] = useState('Adult')
  const [gender, setGender] = useState('Female')
  const [emotion, setEmotion] = useState('neutrality')
  const [environment, setEnvironment] = useState('library')
  const [songs, setSongs] = useState([])
  const [loading, setLoading] = useState(false)
  const [suggesting, setSuggesting] = useState(false)
  const [error, setError] = useState('')
  const [suggestion, setSuggestion] = useState('')
  const [suggestionProvider, setSuggestionProvider] = useState('')
  const [history, setHistory] = useState([])

  useEffect(() => {
    const raw = localStorage.getItem(HISTORY_KEY)
    if (raw) setHistory(JSON.parse(raw))
  }, [])

  useEffect(() => {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 8)))
  }, [history])

  const canRecommend = useMemo(() => image && languages.length > 0 && !loading, [image, languages, loading])

  function onImageSelect(event) {
    const file = event.target.files?.[0]
    if (!file) return
    setImage(file)
    setImagePreview(URL.createObjectURL(file))
  }

  function toggleLanguage(lang) {
    setLanguages((prev) => (prev.includes(lang) ? prev.filter((l) => l !== lang) : [...prev, lang]))
  }

  function saveSession() {
    const item = {
      id: `${Date.now()}`,
      age,
      gender,
      emotion,
      environment,
      languages,
      songs,
      suggestion
    }
    setHistory((prev) => [item, ...prev])
  }

  function loadSession(item) {
    setAge(item.age)
    setGender(item.gender)
    setEmotion(item.emotion)
    setEnvironment(item.environment)
    setLanguages(item.languages)
    setSongs(item.songs || [])
    setSuggestion(item.suggestion || '')
    setStep(3)
  }

  async function generateSuggestion() {
    setSuggesting(true)
    setError('')
    try {
      const data = await fetchSuggestion({ age, gender, emotion, environment, languages })
      setSuggestion(data.suggestion || '')
      setSuggestionProvider(data.provider || '')
      setStep(3)
    } catch (err) {
      setError(err.message || 'Failed to generate LLM suggestion')
    } finally {
      setSuggesting(false)
    }
  }

  async function recommendSongs() {
    if (!image) return
    setLoading(true)
    setError('')
    try {
      const data = await fetchRecommendations({ image, age, gender, emotion, environment, languages })
      setSongs(data.songs || [])
      setStep(3)
      saveSession()
    } catch (err) {
      setError(err.message || 'Failed to fetch recommendations')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="container">
      <header className="header-row">
        <div>
          <h1>Emotion Music Recommender</h1>
          <p className="muted">Mobile-ready music discovery by emotion, profile, and context.</p>
        </div>
        <button className="secondary" onClick={() => { setSongs([]); setSuggestion('') }} type="button">Reset</button>
      </header>

      <MobileNav step={step} setStep={setStep} />

      {step === 1 && (
        <section className="card">
          <h2>1) Upload Image</h2>
          <input type="file" accept="image/png,image/jpeg,image/jpg" onChange={onImageSelect} />
          {imagePreview && <img src={imagePreview} alt="Uploaded preview" className="preview" />}
        </section>
      )}

      {step === 2 && (
        <section className="card">
          <h2>2) Profile & Preferences</h2>
          <div className="grid">
            <label>Age Category<select value={age} onChange={(e) => setAge(e.target.value)}><option>Child</option><option>Teenager</option><option>Adult</option><option>Older Adult</option></select></label>
            <label>Gender<select value={gender} onChange={(e) => setGender(e.target.value)}><option>Female</option><option>Male</option></select></label>
            <label>Emotion<select value={emotion} onChange={(e) => setEmotion(e.target.value)}><option>angry</option><option>disgust</option><option>fear</option><option>happiness</option><option>neutrality</option><option>sadness</option><option>surprise</option></select></label>
            <label>Environment<input value={environment} onChange={(e) => setEnvironment(e.target.value)} /></label>
          </div>
          <p className="label">Languages</p>
          <div className="language-list">
            {LANGUAGES.map((lang) => (
              <button type="button" key={lang} className={languages.includes(lang) ? 'chip active' : 'chip'} onClick={() => toggleLanguage(lang)}>{lang}</button>
            ))}
          </div>
        </section>
      )}

      {step === 3 && (
        <RecommendationPanel
          songs={songs}
          suggestion={suggestion}
          suggestionProvider={suggestionProvider}
          onSuggest={generateSuggestion}
          onRecommend={recommendSongs}
          canRecommend={canRecommend}
          loading={loading}
          suggesting={suggesting}
          error={error}
        />
      )}

      <HistoryPanel history={history} onUse={loadSession} />
    </main>
  )
}
