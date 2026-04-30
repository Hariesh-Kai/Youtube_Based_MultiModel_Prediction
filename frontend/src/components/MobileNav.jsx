export default function MobileNav({ step, setStep }) {
  const items = [
    { id: 1, label: 'Upload' },
    { id: 2, label: 'Profile' },
    { id: 3, label: 'Results' }
  ]

  return (
    <nav className="mobile-nav">
      {items.map((item) => (
        <button
          key={item.id}
          className={step === item.id ? 'nav-btn active' : 'nav-btn'}
          onClick={() => setStep(item.id)}
          type="button"
        >
          {item.label}
        </button>
      ))}
    </nav>
  )
}
