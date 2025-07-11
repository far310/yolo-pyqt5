"use client"

import React, { useState, useRef, useEffect } from "react"

interface SelectProps {
  value: string
  onValueChange: (value: string) => void
  disabled?: boolean
  children: React.ReactNode
}

interface SelectTriggerProps {
  children: React.ReactNode
  className?: string
}

interface SelectContentProps {
  children: React.ReactNode
}

interface SelectItemProps {
  value: string
  children: React.ReactNode
}

interface SelectValueProps {
  placeholder?: string
}

export function Select({ value, onValueChange, disabled = false, children }: SelectProps) {
  const [isOpen, setIsOpen] = useState(false)
  const selectRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (selectRef.current && !selectRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  const handleItemClick = (itemValue: string) => {
    onValueChange(itemValue)
    setIsOpen(false)
  }

  return (
    <div className="select-container" ref={selectRef}>
      {React.Children.map(children, (child) => {
        if (React.isValidElement(child)) {
          if (child.type === SelectTrigger) {
            return React.cloneElement(child as React.ReactElement<any>, {
              onClick: () => !disabled && setIsOpen(!isOpen),
              disabled,
              isOpen,
            })
          }
          if (child.type === SelectContent) {
            return isOpen
              ? React.cloneElement(child as React.ReactElement<any>, {
                  onItemClick: handleItemClick,
                  currentValue: value,
                })
              : null
          }
        }
        return child
      })}
    </div>
  )
}

export function SelectTrigger({ children, className = "", onClick, disabled, isOpen }: SelectTriggerProps & any) {
  return (
    <div
      className={`select-trigger ${disabled ? "select-disabled" : ""} ${isOpen ? "select-open" : ""} ${className}`}
      onClick={onClick}
    >
      {children}
      <span className="select-arrow">▼</span>
    </div>
  )
}

export function SelectContent({ children, onItemClick, currentValue }: SelectContentProps & any) {
  return (
    <div className="select-content">
      {React.Children.map(children, (child) => {
        if (React.isValidElement(child) && child.type === SelectItem) {
          return React.cloneElement(child as React.ReactElement<any>, {
            onClick: () => onItemClick(child.props.value),
            isSelected: child.props.value === currentValue,
          })
        }
        return child
      })}
    </div>
  )
}

export function SelectItem({ value, children, onClick, isSelected }: SelectItemProps & any) {
  return (
    <div className={`select-item ${isSelected ? "select-item-selected" : ""}`} onClick={onClick}>
      {children}
    </div>
  )
}

export function SelectValue({ placeholder }: SelectValueProps) {
  return <span className="select-value">{placeholder}</span>
}
