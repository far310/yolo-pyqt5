"use client"

import type React from "react"

import { useState } from "react"
import { ChevronDown, ChevronRight } from "@/components/ui/icons"

interface TestCollapsibleProps {
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
}

export function TestCollapsible({ title, children, defaultOpen = false }: TestCollapsibleProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  const toggle = () => {
    console.log(`Toggling ${title}: ${isOpen} -> ${!isOpen}`)
    setIsOpen(!isOpen)
  }

  return (
    <div className="w-full">
      <div
        className="flex items-center justify-between w-full p-2 hover:bg-gray-50 rounded border-t pt-3 cursor-pointer"
        onClick={toggle}
        style={{ userSelect: "none" }}
      >
        <h4 className="font-medium text-sm text-gray-700">{title}</h4>
        {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
      </div>
      {isOpen && <div className="pt-2">{children}</div>}
    </div>
  )
}
