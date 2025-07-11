"use client"

import type React from "react"

interface BadgeProps {
  variant?: "default" | "destructive" | "outline"
  children: React.ReactNode
  className?: string
}

export function Badge({ variant = "default", children, className = "" }: BadgeProps) {
  return <span className={`badge badge-${variant} ${className}`}>{children}</span>
}
