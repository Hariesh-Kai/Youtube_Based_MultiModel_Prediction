import Link from 'next/link'

export default function ActionButtons() {
  return (
    <div>
      <Link href="/" className="inline-flex rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm text-slate-700">
        Try Another Image
      </Link>
    </div>
  )
}
