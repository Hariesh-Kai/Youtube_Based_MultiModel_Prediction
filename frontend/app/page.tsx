'use client'

import { useRouter } from 'next/navigation'
import { useState } from 'react'
import Header from '@/components/Header'
import UploadBox from '@/components/UploadBox'

export default function HomePage() {
  const router = useRouter()
  const [file, setFile] = useState<File | null>(null)

  return (
    <div className="min-h-screen bg-white">
      <Header />
      <main className="mx-auto flex max-w-3xl flex-col items-center px-4 py-16">
        <h1 className="text-3xl font-semibold text-slate-900">AI Mood Engine</h1>
        <p className="mt-2 text-sm text-slate-500">Upload an image to generate atmosphere</p>
        <div className="mt-8 w-full max-w-xl">
          <UploadBox onFileSelect={setFile} fileName={file?.name} />
        </div>
        <button
          disabled={!file}
          onClick={() => router.push('/result')}
          className="mt-6 rounded-xl bg-accent px-6 py-3 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          Analyze Mood
        </button>
      </main>
    </div>
  )
}
