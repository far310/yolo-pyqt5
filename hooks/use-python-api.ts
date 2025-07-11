"use client"

import { useState, useEffect, useCallback } from "react"
import { pythonAPI } from "@/services/python-api"
import type { DetectedObject, ImageParams, RecognitionSettings } from "@/types"

export function usePythonAPI() {
  const [isConnected, setIsConnected] = useState(false)
  const [systemStatus, setSystemStatus] = useState({
    fps: 0,
    cpuUsage: 0,
    memoryUsage: 0,
    gpuUsage: 0,
  })
  const [currentFrame, setCurrentFrame] = useState<string>("")
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([])

  // 轮询获取数据
  useEffect(() => {
    let frameInterval: NodeJS.Timeout
    let detectionInterval: NodeJS.Timeout
    let statusInterval: NodeJS.Timeout

    const startPolling = () => {
      // 获取图像帧 - 每100ms
      frameInterval = setInterval(async () => {
        try {
          const result = await pythonAPI.getCurrentFrame()
          if (result.success && result.image) {
            setCurrentFrame(result.image)
          }
        } catch (error) {
          console.error("Failed to get frame:", error)
        }
      }, 100)

      // 获取检测结果 - 每200ms
      detectionInterval = setInterval(async () => {
        try {
          const result = await pythonAPI.getDetectionResults()
          if (result.objects) {
            setDetectedObjects(result.objects)
          }
        } catch (error) {
          console.error("Failed to get detection results:", error)
        }
      }, 200)

      // 获取系统状态 - 每1000ms
      statusInterval = setInterval(async () => {
        try {
          const status = await pythonAPI.getSystemStatus()
          setSystemStatus({
            fps: status.fps || 0,
            cpuUsage: status.cpuUsage || 0,
            memoryUsage: status.memoryUsage || 0,
            gpuUsage: status.gpuUsage || 0,
          })
        } catch (error) {
          console.error("Failed to get system status:", error)
        }
      }, 1000)
    }

    startPolling()

    return () => {
      if (frameInterval) clearInterval(frameInterval)
      if (detectionInterval) clearInterval(detectionInterval)
      if (statusInterval) clearInterval(statusInterval)
    }
  }, [])

  // 检查Python连接状态
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await pythonAPI.getSystemStatus()
        setIsConnected(true)
      } catch (error) {
        setIsConnected(false)
        console.warn("Python API not connected, using mock mode")
      }
    }

    checkConnection()
    const interval = setInterval(checkConnection, 5000) // 每5秒检查一次连接

    return () => clearInterval(interval)
  }, [])

  const startCamera = useCallback(async (deviceId: string) => {
    try {
      const result = await pythonAPI.startCamera(deviceId)
      return result
    } catch (error) {
      console.error("Failed to start camera:", error)
      throw error
    }
  }, [])

  const stopCamera = useCallback(async () => {
    try {
      const result = await pythonAPI.stopCamera()
      // 清空当前帧
      setCurrentFrame("")
      setDetectedObjects([])
      return result
    } catch (error) {
      console.error("Failed to stop camera:", error)
      throw error
    }
  }, [])

  const loadModel = useCallback(async (modelUrl: string) => {
    try {
      const result = await pythonAPI.loadModel(modelUrl)
      return result
    } catch (error) {
      console.error("Failed to load model:", error)
      throw error
    }
  }, [])

  const updateImageParams = useCallback(async (params: ImageParams) => {
    try {
      const result = await pythonAPI.setImageParams(params)
      return result
    } catch (error) {
      console.error("Failed to update image params:", error)
      throw error
    }
  }, [])

  const updateRecognitionSettings = useCallback(async (settings: RecognitionSettings) => {
    try {
      const result = await pythonAPI.setRecognitionSettings(settings)
      return result
    } catch (error) {
      console.error("Failed to update recognition settings:", error)
      throw error
    }
  }, [])

  const getDetectionResults = useCallback(async (): Promise<DetectedObject[]> => {
    try {
      const result = await pythonAPI.getDetectionResults()
      return result.objects || []
    } catch (error) {
      console.error("Failed to get detection results:", error)
      return []
    }
  }, [])

  const saveImage = useCallback(async (filename?: string) => {
    try {
      const result = await pythonAPI.saveImage(filename)
      return result
    } catch (error) {
      console.error("Failed to save image:", error)
      throw error
    }
  }, [])

  const exportReport = useCallback(async (format: "json" | "csv" | "pdf" = "json") => {
    try {
      const result = await pythonAPI.exportReport(format)
      return result
    } catch (error) {
      console.error("Failed to export report:", error)
      throw error
    }
  }, [])

  return {
    isConnected,
    systemStatus,
    currentFrame, // 从轮询获取的当前帧
    detectedObjects, // 从轮询获取的检测结果
    startCamera,
    stopCamera,
    loadModel,
    updateImageParams,
    updateRecognitionSettings,
    getDetectionResults,
    saveImage,
    exportReport,
  }
}
