import Link from 'next/link'

type Props = {
  title?: string
  showBack?: boolean
}

export default function Header({ title = 'AI Mood Engine', showBack = false }: Props) {
  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex h-16 w-full max-w-3xl items-center justify-between px-4">
        <div className="w-20 text-left text-sm text-slate-600">
          {showBack ? <Link href="/">Back</Link> : <span className="font-medium text-slate-900">{title}</span>}
        </div>
        <h1 className="text-sm font-semibold text-slate-900">{showBack ? title : ''}</h1>
        <button aria-label="settings" className="w-20 text-right text-sm text-slate-500">Settings</button>
      </div>
    </header>
  )
}
