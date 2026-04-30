'use client'

import { useEffect, useState } from 'react'
import ActionButtons from '@/components/ActionButtons'
import AnalysisCard from '@/components/AnalysisCard'
import AudioPlayer from '@/components/AudioPlayer'
import Header from '@/components/Header'
import InsightCard from '@/components/InsightCard'

export default function ResultPage() {
  const [stage, setStage] = useState<'analyzing' | 'generating' | 'done'>('analyzing')

  useEffect(() => {
    const t1 = setTimeout(() => setStage('generating'), 1200)
    const t2 = setTimeout(() => setStage('done'), 2400)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [])

  return (
    <div className="min-h-screen bg-white">
      <Header showBack />
      <main className="mx-auto max-w-3xl space-y-4 px-4 py-6 fade-in">
        {stage !== 'done' && (
          <div className="rounded-2xl border border-slate-200 bg-surface p-5 text-sm text-slate-600">
            {stage === 'analyzing' ? 'Analyzing image...' : 'Generating atmosphere...'}
          </div>
        )}

        {stage === 'done' && (
          <>
            <AnalysisCard emotion="Calm" environment="Library" confidence={92} />
            <InsightCard text="You seem reflective and focused — a soft ambient layer may help you stay grounded." />
            <AudioPlayer />
            <ActionButtons />
          </>
        )}
      </main>
    </div>
  )
}
