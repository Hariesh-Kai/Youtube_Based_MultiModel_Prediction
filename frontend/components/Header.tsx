import Link from 'next/link'

type Props = {
  title?: string
  showBack?: boolean
}

export default function Header({ title = 'AI Mood Engine', showBack = false }: Props) {
  return (
    <header className="w-full border-b border-slate-200 bg-white">
      <div className="mx-auto flex h-14 max-w-3xl items-center justify-between px-4">
        {showBack ? <Link href="/" className="text-sm text-slate-600">← Back</Link> : <span className="text-sm font-medium">{title}</span>}
        <span className="text-sm font-medium">{showBack ? title : ''}</span>
        <button className="text-slate-500" aria-label="settings">⚙︎</button>
      </div>
    </header>
  )
}
