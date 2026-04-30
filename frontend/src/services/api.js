const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export async function fetchSuggestion(payload) {
  const response = await fetch(`${API_BASE}/suggest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  if (!response.ok) throw new Error(`Suggestion API error: ${response.status}`)
  return response.json()
}

export async function fetchRecommendations(payload) {
  const formData = new FormData()
  formData.append('image', payload.image)
  formData.append('age', payload.age)
  formData.append('gender', payload.gender)
  formData.append('emotion', payload.emotion)
  formData.append('environment', payload.environment)
  formData.append('languages', payload.languages.join(','))

  const response = await fetch(`${API_BASE}/recommend`, { method: 'POST', body: formData })
  if (!response.ok) throw new Error(`Recommendation API error: ${response.status}`)
  return response.json()
}
