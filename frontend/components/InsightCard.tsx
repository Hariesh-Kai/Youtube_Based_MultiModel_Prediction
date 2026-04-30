export default function InsightCard({ text }: { text: string }) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-6">
      <h3 className="mb-3 text-sm font-semibold text-slate-700">AI Insight</h3>
      <p className="text-sm leading-relaxed text-slate-600">{text}</p>
    </section>
  )
}
