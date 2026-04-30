type Props = { emotion: string; environment: string; confidence: number }

export default function AnalysisCard({ emotion, environment, confidence }: Props) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-surface p-5">
      <h3 className="mb-3 text-sm font-semibold text-slate-700">Emotion Analysis</h3>
      <div className="space-y-2 text-sm">
        <p><span className="text-slate-500">Emotion:</span> <span className="font-medium">{emotion}</span></p>
        <p><span className="text-slate-500">Environment:</span> <span className="font-medium">{environment}</span></p>
        <p><span className="text-slate-500">Confidence:</span> <span className="font-medium">{confidence}%</span></p>
      </div>
    </section>
  )
}
