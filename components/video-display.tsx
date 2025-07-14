"use client"

import { useRef, useEffect, forwardRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Camera, Download } from "@/components/ui/icons"

interface VideoDisplayProps {
  isStreaming: boolean
  modelLoading: boolean
  currentFrame?: string
  isSaving?: boolean // 新增：保存状态
  onSaveImage?: () => void
}

export const VideoDisplay = forwardRef<
  { video: HTMLVideoElement | null; canvas: HTMLCanvasElement | null },
  VideoDisplayProps
>(({ isStreaming, modelLoading, currentFrame, isSaving = false, onSaveImage }, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)

  // 暴露refs给父组件
  useEffect(() => {
    if (typeof ref === "function") {
      ref({ video: videoRef.current, canvas: canvasRef.current })
    } else if (ref) {
      ref.current = { video: videoRef.current, canvas: canvasRef.current }
    }
  }, [ref])

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between">
          <span>实时图像</span>
          <div className="flex items-center gap-2">
            {onSaveImage && (
              <Button onClick={onSaveImage} disabled={!isStreaming || isSaving} variant="outline" size="sm">
                <Download className="w-4 h-4 mr-1" />
                {isSaving ? "保存中..." : "保存"}
              </Button>
            )}
            <div className={`w-2 h-2 rounded-full ${isStreaming ? "bg-green-500" : "bg-red-500"}`} />
            <span className="text-sm text-gray-600">{isStreaming ? "运行中" : "已停止"}</span>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative bg-black rounded-lg overflow-hidden w-full h-[520px]">
          {/* 显示从后端推送的图像 */}
          {isStreaming && currentFrame ? (
            <img
              ref={imgRef}
              src={currentFrame || "/placeholder.svg"}
              alt="Camera feed"
              className="absolute inset-0 w-full h-full object-cover"
              onError={(e) => {
                console.error("Image load error:", e)
              }}
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center text-white">
              <div className="text-center">
                <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg">请启动摄像头开始识别</p>
                {modelLoading && <p className="text-sm text-blue-400 mt-2">模型加载中，请稍候...</p>}
              </div>
            </div>
          )}

          {/* 保留原有的video和canvas元素，但隐藏它们 */}
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="absolute inset-0 w-full h-full object-cover"
            style={{ display: "none" }}
          />
          <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover" style={{ display: "none" }} />
        </div>
      </CardContent>
    </Card>
  )
})

VideoDisplay.displayName = "VideoDisplay"
