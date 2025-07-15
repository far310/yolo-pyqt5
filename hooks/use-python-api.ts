"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { pythonAPI } from "@/services/python-api"
import type { DetectedObject, ImageParams, RecognitionSettings } from "@/types"

export function usePythonAPI() {
  const [isConnected, setIsConnected] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false) // 添加流状态跟踪
  const [systemStatus, setSystemStatus] = useState({
    fps: 0,
    cpuUsage: 0,
    memoryUsage: 0,
    gpuUsage: 0,
  })
  const [currentFrame, setCurrentFrame] = useState<string>("")
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([])

  // 使用 useRef 来存储定时器引用
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const statusIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // 添加背景图像状态
  const [backgroundImageStatus, setBackgroundImageStatus] = useState({
    isSet: false,
    path: "",
    isValid: false,
  })

  // 启动轮询
  const startPolling = useCallback(() => {
    console.log("Starting polling...")

    // 清除现有的定时器
    stopPolling()

    // 获取图像帧 - 每100ms
    frameIntervalRef.current = setInterval(async () => {
      if (!isStreaming) return // 只有在流状态下才获取帧

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
    detectionIntervalRef.current = setInterval(async () => {
      if (!isStreaming) return // 只有在流状态下才获取检测结果

      try {
        const result = await pythonAPI.getDetectionResults()
        if (result.objects) {
          setDetectedObjects(result.objects)
        }
      } catch (error) {
        console.error("Failed to get detection results:", error)
      }
    }, 200)

    // 获取系统状态 - 每1000ms（这个可以一直运行）
    statusIntervalRef.current = setInterval(async () => {
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
  }, [isStreaming])

  // 停止轮询
  const stopPolling = useCallback(() => {
    console.log("Stopping polling...")

    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current)
      frameIntervalRef.current = null
    }
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current)
      detectionIntervalRef.current = null
    }
    if (statusIntervalRef.current) {
      clearInterval(statusIntervalRef.current)
      statusIntervalRef.current = null
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

  // 当流状态改变时，启动或停止轮询
  useEffect(() => {
    if (isStreaming) {
      startPolling()
    } else {
      // 停止帧和检测结果轮询，但保持系统状态轮询
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current)
        frameIntervalRef.current = null
      }
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current)
        detectionIntervalRef.current = null
      }

      // 清空当前帧和检测结果
      setCurrentFrame("")
      setDetectedObjects([])
    }

    return () => {
      if (!isStreaming) {
        stopPolling()
      }
    }
  }, [isStreaming, startPolling, stopPolling])

  // 组件卸载时清理
  useEffect(() => {
    return () => {
      stopPolling()
    }
  }, [stopPolling])

  const startCamera = useCallback(async (deviceId: string) => {
    try {
      console.log("Starting camera...")
      const result = await pythonAPI.startCamera(deviceId)
      if (result.success) {
        setIsStreaming(true) // 设置流状态为true，这会触发轮询开始
      }
      return result
    } catch (error) {
      console.error("Failed to start camera:", error)
      throw error
    }
  }, [])

  const stopCamera = useCallback(async () => {
    try {
      console.log("Stopping camera...")
      const result = await pythonAPI.stopCamera()
      setIsStreaming(false) // 设置流状态为false，这会停止轮询
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

  // 设置背景图像路径
  const setBackgroundImagePath = useCallback(async (imagePath: string) => {
    try {
      const result = await pythonAPI.setBackgroundImagePath(imagePath)
      if (result.success) {
        // 更新背景图像状态
        setBackgroundImageStatus({
          isSet: true,
          path: imagePath,
          isValid: true,
        })
      }
      return result
    } catch (error) {
      console.error("Failed to set background image path:", error)
      throw error
    }
  }, [])

  // 清除背景图像
  const clearBackgroundImage = useCallback(async () => {
    try {
      const result = await pythonAPI.clearBackgroundImage()
      if (result.success) {
        // 清除背景图像状态
        setBackgroundImageStatus({
          isSet: false,
          path: "",
          isValid: false,
        })
      }
      return result
    } catch (error) {
      console.error("Failed to clear background image:", error)
      throw error
    }
  }, [])

  // 获取背景图像状态
  const getBackgroundImageStatus = useCallback(async () => {
    try {
      const status = await pythonAPI.getBackgroundImageStatus()
      if (status.success) {
        setBackgroundImageStatus({
          isSet: status.isSet,
          path: status.path || "",
          isValid: status.isValid || false,
        })
      }
      return status
    } catch (error) {
      console.error("Failed to get background image status:", error)
      throw error
    }
  }, [])

  return {
    isConnected,
    isStreaming, // 暴露流状态
    systemStatus,
    currentFrame, // 从轮询获取的当前帧
    detectedObjects, // 从轮询获取的检测结果
    backgroundImageStatus, // 背景图像状态
    startCamera,
    stopCamera,
    loadModel,
    updateImageParams,
    updateRecognitionSettings,
    setBackgroundImagePath, // 设置背景图像路径
    clearBackgroundImage, // 清除背景图像
    getBackgroundImageStatus, // 获取背景图像状态
    getDetectionResults,
    saveImage,
    exportReport,
  }
}
