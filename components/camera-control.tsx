"use client"

import type React from "react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Camera, Play, Square } from "@/components/ui/icons"
import { useState } from "react"

interface CameraControlProps {
  selectedCamera: string
  isStreaming: boolean
  modelLoading: boolean
  isStarting?: boolean // 新增：摄像头启动状态
  onCameraChange: (cameraId: string) => void
  onStartCamera: () => void
  onStopCamera: () => void
}

export function CameraControl({
  selectedCamera,
  isStreaming,
  modelLoading,
  isStarting = false, // 新增参数
  onCameraChange,
  onStartCamera,
  onStopCamera,
}: CameraControlProps) {
  const [cameraIndex, setCameraIndex] = useState(selectedCamera || "0")

  const handleCameraIndexChange = (value: string) => {
    // 只允许输入0-10的数字
    const numValue = Number.parseInt(value)
    if (value === "" || (!isNaN(numValue) && numValue >= 0 && numValue <= 10)) {
      setCameraIndex(value)
      onCameraChange(value)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    // 只允许数字键
    if (!/[0-9]/.test(e.key) && !["Backspace", "Delete", "ArrowLeft", "ArrowRight", "Tab"].includes(e.key)) {
      e.preventDefault()
    }
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Camera className="w-5 h-5" />
          摄像头控制
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <Label htmlFor="camera-index">摄像头索引 (0-10)</Label>
          <Input
            id="camera-index"
            type="text"
            value={cameraIndex}
            onChange={(e) => handleCameraIndexChange(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="输入摄像头索引，如: 0"
            className="mt-2"
            maxLength={2}
            disabled={isStreaming}
          />
          <div className="text-xs text-gray-500 mt-1">通常 0 为默认摄像头，1-10 为外接摄像头</div>
        </div>

        {/* 确保按钮容器不嵌套 */}
        <div className="flex gap-2">
          <Button
            onClick={onStartCamera}
            disabled={isStreaming || modelLoading || !cameraIndex.trim() || isStarting}
            className="flex-1"
            size="sm"
          >
            <Play className="w-4 h-4 mr-1" />
            {isStarting ? "启动中..." : "启动"}
          </Button>
          <Button
            onClick={onStopCamera}
            disabled={!isStreaming || isStarting}
            variant="outline"
            className="flex-1 bg-transparent"
            size="sm"
          >
            <Square className="w-4 h-4 mr-1" />
            停止
          </Button>
        </div>

        {/* 摄像头状态显示 */}
        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="flex items-center justify-between">
            <span className="font-medium text-sm">摄像头状态</span>
            <div
              className={`w-2 h-2 rounded-full ${
                isStarting ? "bg-yellow-500" : isStreaming ? "bg-green-500" : "bg-red-500"
              }`}
            />
          </div>
          <div className="text-xs text-gray-600 mt-1">
            {isStarting
              ? `正在启动摄像头 ${cameraIndex}...`
              : isStreaming
                ? `正在使用摄像头 ${cameraIndex}`
                : "摄像头未启动"}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
