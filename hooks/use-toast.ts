"use client"

import { useState, useCallback } from "react"

interface Toast {
  id: string
  message: string
  type?: "success" | "error" | "info"
  duration?: number
}

export function useToast() {
  const [toasts, setToasts] = useState<Toast[]>([])

  const addToast = useCallback((message: string, type: "success" | "error" | "info" = "info", duration = 3000) => {
    const id = Math.random().toString(36).substr(2, 9)
    const newToast: Toast = { id, message, type, duration }

    setToasts((prev) => [...prev, newToast])

    // 自动移除
    setTimeout(() => {
      setToasts((prev) => prev.filter((toast) => toast.id !== id))
    }, duration + 300)
  }, [])

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id))
  }, [])

  const showSuccess = useCallback(
    (message: string, duration?: number) => {
      addToast(message, "success", duration)
    },
    [addToast],
  )

  const showError = useCallback(
    (message: string, duration?: number) => {
      addToast(message, "error", duration)
    },
    [addToast],
  )

  const showInfo = useCallback(
    (message: string, duration?: number) => {
      addToast(message, "info", duration)
    },
    [addToast],
  )

  return {
    toasts,
    addToast,
    removeToast,
    showSuccess,
    showError,
    showInfo,
  }
}
