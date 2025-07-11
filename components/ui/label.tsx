"use client"

import type React from "react"

interface LabelProps extends React.LabelHTMLAttributes<HTMLLabelElement> {
  children: React.ReactNode
}

export function Label({ children, className = "", ...props }: LabelProps) {
  return (
    <label className={`label ${className}`} {...props}>
      {children}
    </label>
  )
}
