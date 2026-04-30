import type { Config } from 'tailwindcss'

export default {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        accent: '#2563eb',
        surface: '#f5f7fb'
      }
    }
  },
  plugins: []
} satisfies Config
