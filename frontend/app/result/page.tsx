'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import Header from '@/components/Header'

export default function ResultPage() {
  const [stage, setStage] = useState<'analyzing' | 'generating' | 'done'>('analyzing')
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    const t1 = setTimeout(() => setStage('generating'), 1200)
    const t2 = setTimeout(() => setStage('done'), 2400)
    return () => {
      clearTimeout(t1)
      clearTimeout(t2)
    }
  }, [])

  return (
    <div className="min-h-screen bg-slate-50">
      <Header showBack />
      <main className="mx-auto flex w-full max-w-2xl flex-col items-center px-6 py-10 text-center">
        <div className="mt-12 w-full">
        {stage !== 'done' ? (
          <p className="text-sm text-slate-600 fade-in">
            {stage === 'analyzing' ? 'Analyzing image...' : 'Generating atmosphere...'}
          </p>
        ) : (
          <div className="fade-in space-y-12">
            <section className="space-y-1">
              <p className="text-lg font-medium text-slate-800">Calm · Library · 92%</p>
            </section>

            <section className="space-y-3">
              <p className="text-2xl leading-snug text-slate-900">
                You seem reflective and focused — a soft ambient layer may help you stay grounded.
              </p>
            </section>

            <section className="mx-auto w-full max-w-xl space-y-5 rounded-2xl border border-slate-200 bg-white p-8 sm:p-10">
              <div className="flex flex-col items-center gap-4 sm:flex-row sm:justify-center">
                <button className="btn-polish w-full rounded-lg bg-blue-600 px-10 py-3 text-base font-semibold text-white hover:bg-blue-700 sm:w-auto">Generate New Sound</button>
                <button onClick={() => setPlaying((p) => !p)} className="btn-polish w-full rounded-lg border border-slate-300 bg-white px-6 py-2.5 text-sm text-slate-700 hover:bg-slate-100 sm:w-auto">
                  {playing ? 'Pause' : 'Play'}
                </button>
              </div>
            </section>

            <section className="flex justify-center">
              <Link href="/" className="btn-polish inline-flex rounded-lg border border-slate-300 bg-white px-5 py-2.5 text-sm text-slate-700 hover:bg-slate-100">
                Try Another Image
              </Link>
            </section>
          </div>
        )}
        </div>
      </main>
    </div>
  )
}
