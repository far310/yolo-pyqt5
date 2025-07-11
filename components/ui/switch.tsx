"use client"

interface SwitchProps {
  id?: string
  checked: boolean
  onCheckedChange: (checked: boolean) => void
  disabled?: boolean
}

export function Switch({ id, checked, onCheckedChange, disabled = false }: SwitchProps) {
  const handleClick = () => {
    if (!disabled) {
      onCheckedChange(!checked)
    }
  }

  return (
    <div
      className={`switch ${checked ? "switch-checked" : ""} ${disabled ? "switch-disabled" : ""}`}
      onClick={handleClick}
      id={id}
    >
      <div className="switch-thumb"></div>
    </div>
  )
}
