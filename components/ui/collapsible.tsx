"use client"

import React, { useState } from "react"

interface CollapsibleProps {
  open?: boolean
  onOpenChange?: (open: boolean) => void
  children: React.ReactNode
}

interface CollapsibleTriggerProps {
  children: React.ReactNode
  className?: string
  onClick?: () => void
}

interface CollapsibleContentProps {
  children: React.ReactNode
  className?: string
}

export function Collapsible({ open: controlledOpen, onOpenChange, children }: CollapsibleProps) {
  const [internalOpen, setInternalOpen] = useState(false)
  const isOpen = controlledOpen !== undefined ? controlledOpen : internalOpen

  const handleToggle = () => {
    const newOpen = !isOpen
    if (onOpenChange) {
      onOpenChange(newOpen)
    } else {
      setInternalOpen(newOpen)
    }
  }

  return (
    <div className="collapsible">
      {React.Children.map(children, (child) => {
        if (React.isValidElement(child)) {
          if (child.type === CollapsibleTrigger) {
            return React.cloneElement(child as React.ReactElement<any>, {
              onClick: handleToggle,
            })
          }
          if (child.type === CollapsibleContent) {
            return isOpen ? child : null
          }
        }
        return child
      })}
    </div>
  )
}

export function CollapsibleTrigger({ children, className = "", onClick }: CollapsibleTriggerProps) {
  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (onClick) {
      onClick()
    }
  }

  return (
    <div
      className={`collapsible-trigger ${className}`}
      onClick={handleClick}
      style={{ cursor: "pointer", userSelect: "none" }}
    >
      {children}
    </div>
  )
}

export function CollapsibleContent({ children, className = "" }: CollapsibleContentProps) {
  return <div className={`collapsible-content ${className}`}>{children}</div>
}
