import Header from '@/components/Header'

const sessions = [
  { emotion: 'Happy', insight: 'Upbeat energy detected. Try bright acoustic rhythm.' },
  { emotion: 'Calm', insight: 'You appear focused. Soft ambient textures suit this mood.' }
]

export default function HistoryPage() {
  return (
    <div className="min-h-screen bg-white">
      <Header title="History" showBack />
      <main className="mx-auto max-w-3xl space-y-3 px-4 py-6">
        {sessions.map((s, i) => (
          <div key={i} className="rounded-2xl border border-slate-200 bg-surface p-4">
            <p className="text-sm font-medium text-slate-800">{s.emotion}</p>
            <p className="mt-1 text-sm text-slate-600">{s.insight}</p>
            <button className="mt-3 rounded-lg border border-slate-300 px-3 py-1 text-sm text-slate-700">Play sound</button>
          </div>
        ))}
      </main>
    </div>
  )
}
