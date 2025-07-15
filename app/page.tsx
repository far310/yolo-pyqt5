"use client"

import { useState, useRef, useCallback } from "react"
import { ModelSelector } from "@/components/model-selector"
import { CameraControl } from "@/components/camera-control"
import { ImageParamsControl } from "@/components/image-params-control"
import { RecognitionSettingsComponent } from "@/components/recognition-settings"
import { VideoDisplay } from "@/components/video-display"
import { DetectionResults } from "@/components/detection-results"
import { SystemStats } from "@/components/system-stats"
import { usePythonAPI } from "@/hooks/use-python-api"
import type { ImageParams, RecognitionSettings } from "@/types"
import { useToast } from "@/hooks/use-toast"
import { ToastContainer } from "@/components/ui/toast"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight } from "@/components/ui/icons"

export default function ImageRecognitionDashboard() {
  const videoRefs = useRef<{ video: HTMLVideoElement | null; canvas: HTMLCanvasElement | null }>({
    video: null,
    canvas: null,
  })

  // Python API Hook - 现在包含推送的数据
  const {
    isConnected,
    systemStatus,
    currentFrame, // 从推送获取的当前帧
    detectedObjects, // 从推送获取的检测结果
    startCamera,
    stopCamera,
    loadModel,
    updateImageParams,
    updateRecognitionSettings,
    saveImage,
    exportReport,
  } = usePythonAPI()

  // 状态管理
  const [isStreaming, setIsStreaming] = useState(false)
  const [selectedCamera, setSelectedCamera] = useState<string>("0")
  const [currentModelPath, setCurrentModelPath] = useState<string>("./model/best_float32-2.tflite")
  const [modelLoading, setModelLoading] = useState(false)

  // 新增：左侧面板收起状态
  const [isLeftPanelCollapsed, setIsLeftPanelCollapsed] = useState(false)

  // 折叠状态管理
  const [collapsedSections, setCollapsedSections] = useState({
    basicAdjustment: false,
    detectionThreshold: true,
    perspectiveTransform: true,
    distortionCorrection: true,
    cameraParams: true,
    sizeClassification: true,
    realSize: true,
    backgroundImage: true, // 添加背景图像折叠状态
  })

  // 图像参数
  const [imageParams, setImageParams] = useState<ImageParams>({
    contrast: 100,
    brightness: 100,
    saturation: 100,
    blur: 0,
    delaySeconds: 1, // 延迟设置
    objThresh: 10,
    nmsThresh: 10,
    scoreThresh: 0,
    perspectiveEnabled: true,
    srcPoints: {
      topLeft: { x: 650, y: 330 },
      topRight: { x: 1425, y: 330 },
      bottomRight: { x: 1830, y: 889 },
      bottomLeft: { x: 230, y: 889 },
    },
    distortionEnabled: true,
    distortionK1: -44.9,
    distortionK2: 25.1,
    distortionP1: 0.04,
    distortionP2: 0.02,
    distortionK3: -7.9,
    focalLength: 1298.55,
    cameraHeight: 13.0,
    targetHeight: 2.7,
    hamburgerSizeMin: 10,
    hamburgerSizeMax: 13,
    realWidthCm: 29,
    realHeightCm: 18.5,
    backgroundImagePath: "", // 背景图像路径
  })

  // 识别设置
  const [recognition, setRecognition] = useState<RecognitionSettings>({
    foreignObjectDetection: false,
    sizeClassification: false,
    edgeDetection: false,
    colorAnalysis: false,
    backgroundChangeDetection: false,
    contourDetection: false,
    heightCorrection: false,
  })

  // 新增状态
  const [isCameraStarting, setIsCameraStarting] = useState(false)
  const [isSavingImage, setIsSavingImage] = useState(false)
  const [isExportingReport, setIsExportingReport] = useState(false)

  // Toast 功能
  const { toasts, removeToast, showSuccess, showError, showInfo } = useToast()

  // 处理模型切换
  const handleModelChange = async (modelPath: string) => {
    setModelLoading(true)
    setCurrentModelPath(modelPath)

    try {
      await loadModel(modelPath)
      console.log(`模型加载成功: ${modelPath}`)
    } catch (error) {
      console.error("Failed to load model:", error)
    } finally {
      setModelLoading(false)
    }
  }

  // 修改处理摄像头启动的函数
  const handleStartCamera = async () => {
    setIsCameraStarting(true)
    showInfo(`正在启动摄像头 ${selectedCamera}...`)

    try {
      const result = await startCamera(selectedCamera)
      if (result.success) {
        setIsStreaming(true)
        showSuccess(`摄像头 ${selectedCamera} 启动成功！`)
      } else {
        showError(result.error || "摄像头启动失败")
      }
    } catch (error) {
      console.error("Failed to start camera:", error)
      showError("摄像头启动失败，请检查设备连接")
    } finally {
      setIsCameraStarting(false)
    }
  }

  // 修改处理摄像头停止的函数
  const handleStopCamera = async () => {
    try {
      const result = await stopCamera()
      setIsStreaming(false)
      if (result.success) {
        showInfo("摄像头已停止")
      }
    } catch (error) {
      console.error("Failed to stop camera:", error)
      showError("停止摄像头失败")
    }
  }

  // 修改图像参数变化
  const handleImageParamsChange = useCallback(
    async (params: ImageParams) => {
      setImageParams(params)
      try {
        await updateImageParams(params)
      } catch (error) {
        console.error("Failed to update image params:", error)
      }
    },
    [updateImageParams],
  )

  // 处理识别设置变化
  const handleRecognitionChange = useCallback(
    async (settings: RecognitionSettings) => {
      setRecognition(settings)
      try {
        await updateRecognitionSettings(settings)
      } catch (error) {
        console.error("Failed to update recognition settings:", error)
      }
    },
    [updateRecognitionSettings],
  )

  // 切换折叠状态
  const toggleSection = useCallback(
    (section: string) => {
      console.log(`Toggling section: ${section}, current state:`, collapsedSections[section])
      setCollapsedSections((prev) => {
        const newState = {
          ...prev,
          [section]: !prev[section],
        }
        console.log(`New collapsed state for ${section}:`, newState[section])
        return newState
      })
    },
    [collapsedSections],
  )

  // 重置参数
  const resetParams = () => {
    const defaultParams: ImageParams = {
      contrast: 100,
      brightness: 100,
      saturation: 100,
      blur: 0,
      delaySeconds: 1,
      objThresh: 10,
      nmsThresh: 10,
      scoreThresh: 0,
      perspectiveEnabled: true,
      srcPoints: {
        topLeft: { x: 650, y: 330 },
        topRight: { x: 1425, y: 330 },
        bottomRight: { x: 1830, y: 889 },
        bottomLeft: { x: 230, y: 889 },
      },
      distortionEnabled: true,
      distortionK1: -44.9,
      distortionK2: 25.1,
      distortionP1: 0.04,
      distortionP2: 0.02,
      distortionK3: -7.9,
      focalLength: 1298.55,
      cameraHeight: 13.0,
      targetHeight: 2.7,
      hamburgerSizeMin: 10,
      hamburgerSizeMax: 13,
      realWidthCm: 29,
      realHeightCm: 18.5,
      backgroundImagePath: "", // 重置背景图像路径
    }
    handleImageParamsChange(defaultParams)
  }

  // 修改保存图像的函数
  const handleSaveImage = async () => {
    setIsSavingImage(true)

    try {
      const result = await saveImage()
      if (result.success) {
        // 从消息中提取文件路径
        const pathMatch = result.message.match(/图像已保存: (.+)/)
        const filePath = pathMatch ? pathMatch[1] : "未知路径"
        showSuccess(`图像保存成功！\n路径: ${filePath}`, 5000)
      } else {
        showError(result.error || "保存图像失败")
      }
    } catch (error) {
      console.error("Failed to save image:", error)
      showError("保存图像失败")
    } finally {
      setIsSavingImage(false)
    }
  }

  // 修改导出报告的函数
  const handleExportReport = async () => {
    setIsExportingReport(true)

    try {
      const result = await exportReport("json")
      if (result.success) {
        // 从消息中提取文件路径
        const pathMatch = result.message.match(/报告已导出: (.+)/)
        const filePath = pathMatch ? pathMatch[1] : "未知路径"
        showSuccess(`检测报告导出功！\n路径: ${filePath}`, 5000)
      } else {
        showError(result.error || "导出报告失败")
      }
    } catch (error) {
      console.error("Failed to export report:", error)
      showError("导出报告失败")
    } finally {
      setIsExportingReport(false)
    }
  }

  // 计算统计数据
  const objectCount = detectedObjects.length
  const anomalyCount = detectedObjects.filter((obj) => obj.type === "异物" || obj.type === "缺陷").length
  const averageConfidence =
    objectCount > 0 ? (detectedObjects.reduce((sum, obj) => sum + obj.confidence, 0) / objectCount) * 100 : 0

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">图像识别系统</h1>
              <p className="text-gray-600">
                实时图像处理与智能识别平台 - 主动推送版
                {isConnected ? (
                  <span className="ml-2 text-green-600">● Python已连接</span>
                ) : (
                  <span className="ml-2 text-red-600">● Python未连接</span>
                )}
              </p>
            </div>

            {/* 收起/展开按钮 */}
            <Button
              onClick={() => setIsLeftPanelCollapsed(!isLeftPanelCollapsed)}
              variant="outline"
              size="sm"
              className="flex items-center gap-2"
            >
              {isLeftPanelCollapsed ? (
                <>
                  <ChevronRight className="w-4 h-4" />
                  展开控制面板
                </>
              ) : (
                <>
                  <ChevronLeft className="w-4 h-4" />
                  收起控制面板
                </>
              )}
            </Button>
          </div>
        </div>

        <div
          className={`grid gap-6 transition-all duration-300 ${
            isLeftPanelCollapsed ? "grid-cols-1" : "grid-cols-1 lg:grid-cols-4"
          }`}
        >
          {/* 控制面板 - 可收起 */}
          <div
            className={`space-y-4 transition-all duration-300 overflow-hidden ${
              isLeftPanelCollapsed
                ? "lg:w-0 lg:opacity-0 lg:pointer-events-none lg:absolute lg:-translate-x-full"
                : "lg:col-span-1 lg:w-auto lg:opacity-100 lg:pointer-events-auto lg:relative lg:translate-x-0"
            }`}
          >
            <ModelSelector
              currentModelPath={currentModelPath}
              modelLoading={modelLoading}
              onModelChange={handleModelChange}
            />

            <CameraControl
              selectedCamera={selectedCamera}
              isStreaming={isStreaming}
              modelLoading={modelLoading}
              isStarting={isCameraStarting}
              onCameraChange={setSelectedCamera}
              onStartCamera={handleStartCamera}
              onStopCamera={handleStopCamera}
            />

            <ImageParamsControl
              imageParams={imageParams}
              collapsedSections={collapsedSections}
              onParamsChange={handleImageParamsChange}
              onToggleSection={toggleSection}
              onResetParams={resetParams}
            />

            <RecognitionSettingsComponent settings={recognition} onSettingsChange={handleRecognitionChange} />
          </div>

          {/* 主显示区域 - 动态调整宽度 */}
          <div
            className={`space-y-4 transition-all duration-300 ${isLeftPanelCollapsed ? "col-span-1" : "lg:col-span-3"}`}
          >
            <VideoDisplay
              ref={videoRefs}
              isStreaming={isStreaming}
              modelLoading={modelLoading}
              currentFrame={currentFrame}
              isSaving={isSavingImage}
              onSaveImage={handleSaveImage}
            />

            <DetectionResults
              objects={detectedObjects}
              isExporting={isExportingReport}
              onExportReport={handleExportReport}
            />

            <SystemStats
              objectCount={objectCount}
              anomalyCount={anomalyCount}
              averageConfidence={averageConfidence}
              fps={systemStatus.fps}
              isStreaming={isStreaming}
            />
          </div>
        </div>
      </div>

      {/* Toast 容器 */}
      <ToastContainer toasts={toasts} onRemoveToast={removeToast} />
    </div>
  )
}
