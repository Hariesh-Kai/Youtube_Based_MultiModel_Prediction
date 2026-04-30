'use client'

type Props = {
  onFileSelect: (file: File) => void
  fileName?: string
}

export default function UploadBox({ onFileSelect, fileName }: Props) {
  return (
    <label className="block cursor-pointer rounded-2xl border border-dashed border-slate-300 bg-surface p-8 text-center">
      <input
        type="file"
        accept="image/png,image/jpeg,image/jpg"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFileSelect(file)
        }}
      />
      <p className="text-sm text-slate-500">Drag & drop an image, or click to upload</p>
      {fileName && <p className="mt-2 text-sm font-medium text-slate-700">{fileName}</p>}
    </label>
  )
}
