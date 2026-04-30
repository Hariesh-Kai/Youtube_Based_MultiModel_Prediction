'use client'
import { useState } from 'react'

export default function AudioPlayer() {
  const [playing, setPlaying] = useState(false)

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-6">
      <h3 className="mb-4 text-sm font-semibold text-slate-700">Audio Atmosphere</h3>
      <div className="flex flex-wrap gap-3">
        <button onClick={() => setPlaying((p) => !p)} className="rounded-xl bg-blue-600 px-4 py-2 text-sm text-white">
          {playing ? 'Pause' : 'Play'}
        </button>
        <button className="rounded-xl border border-slate-300 px-4 py-2 text-sm text-slate-700">Generate New Sound</button>
      </div>
    </section>
  )
}
