'use client'

import { useRouter } from 'next/navigation'
import { useState } from 'react'
import Header from '@/components/Header'
import UploadBox from '@/components/UploadBox'

export default function HomePage() {
  const router = useRouter()
  const [file, setFile] = useState<File | null>(null)

  return (
    <div className="min-h-screen bg-slate-50">
      <Header />
      <main className="mx-auto w-full max-w-3xl px-4 py-16 sm:py-20">
        <section className="mx-auto max-w-xl text-center">
          <h2 className="text-3xl font-semibold tracking-tight text-slate-900">AI Mood Engine</h2>
          <p className="mt-3 text-sm text-slate-600">Upload an image to generate atmosphere</p>
        </section>

        <section className="mx-auto mt-10 w-full max-w-xl">
          <UploadBox onFileSelect={setFile} fileName={file?.name} />
        </section>

        <section className="mx-auto mt-8 flex w-full max-w-xl justify-center">
          <button
            disabled={!file}
            onClick={() => router.push('/result')}
            className="rounded-xl bg-blue-600 px-6 py-3 text-sm font-medium text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
          >
            Analyze Mood
          </button>
        </section>
      </main>
    </div>
  )
}
