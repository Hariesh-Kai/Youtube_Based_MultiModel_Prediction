import { useMemo, useState } from 'react'
import RecommendationPanel from '../components/RecommendationPanel'
import { fetchRecommendations, fetchSuggestion } from '../services/api'

const LANGUAGES = ['English', 'Hindi', 'Tamil', 'Spanish', 'French', 'German', 'Chinese', 'Japanese']

export default function DiscoverPage() {
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

  const canRecommend = useMemo(() => image && languages.length > 0 && !loading, [image, languages, loading])

  const toggleLanguage = (lang) => setLanguages((prev) => prev.includes(lang) ? prev.filter((l) => l !== lang) : [...prev, lang])

  async function generateSuggestion() {
    setSuggesting(true); setError('')
    try {
      const data = await fetchSuggestion({ age, gender, emotion, environment, languages })
      setSuggestion(data.suggestion || ''); setSuggestionProvider(data.provider || '')
    } catch (err) { setError(err.message || 'Failed to generate suggestion') }
    finally { setSuggesting(false) }
  }

  async function recommendSongs() {
    if (!image) return
    setLoading(true); setError('')
    try {
      const data = await fetchRecommendations({ image, age, gender, emotion, environment, languages })
      setSongs(data.songs || [])
    } catch (err) { setError(err.message || 'Failed to fetch recommendations') }
    finally { setLoading(false) }
  }

  return (
    <>
      <section className="card">
        <h3>Discover</h3>
        <input type="file" accept="image/png,image/jpeg,image/jpg" onChange={(e)=>{const f=e.target.files?.[0]; if(f){setImage(f); setImagePreview(URL.createObjectURL(f))}}} />
        {imagePreview && <img src={imagePreview} alt="preview" className="preview" />}
        <div className="grid">
          <label>Age<select value={age} onChange={(e)=>setAge(e.target.value)}><option>Child</option><option>Teenager</option><option>Adult</option><option>Older Adult</option></select></label>
          <label>Gender<select value={gender} onChange={(e)=>setGender(e.target.value)}><option>Female</option><option>Male</option></select></label>
          <label>Emotion<select value={emotion} onChange={(e)=>setEmotion(e.target.value)}><option>angry</option><option>disgust</option><option>fear</option><option>happiness</option><option>neutrality</option><option>sadness</option><option>surprise</option></select></label>
          <label>Environment<input value={environment} onChange={(e)=>setEnvironment(e.target.value)} /></label>
        </div>
        <div className="language-list">{LANGUAGES.map((lang)=><button key={lang} className={languages.includes(lang)?'chip active':'chip'} onClick={()=>toggleLanguage(lang)}>{lang}</button>)}</div>
      </section>
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
    </>
  )
}
