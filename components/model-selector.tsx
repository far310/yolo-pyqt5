"use client"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type React from "react"

import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Brain, Cpu } from "@/components/ui/icons"
import { useState } from "react"

interface ModelSelectorProps {
  currentModelPath: string
  modelLoading: boolean
  onModelChange: (modelPath: string) => void
}

export function ModelSelector({ currentModelPath, modelLoading, onModelChange }: ModelSelectorProps) {
  const [inputPath, setInputPath] = useState(currentModelPath)

  const handleLoadModel = () => {
    if (inputPath.trim()) {
      onModelChange(inputPath.trim())
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleLoadModel()
    }
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Brain className="w-5 h-5" />
          AI模型设置
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <Label htmlFor="model-path">模型文件路径</Label>
          <Input
            id="model-path"
            type="text"
            value={inputPath}
            onChange={(e) => setInputPath(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="例如: ./model/best_float32-2.tflite"
            className="mt-2"
            disabled={modelLoading}
          />
        </div>

        <Button onClick={handleLoadModel} disabled={modelLoading || !inputPath.trim()} className="w-full" size="sm">
          {modelLoading ? (
            <>
              <Cpu className="w-4 h-4 mr-2 animate-spin" />
              加载中...
            </>
          ) : (
            "加载模型"
          )}
        </Button>

        {/* 当前模型信息 */}
        <div className="bg-gray-50 p-3 rounded-lg space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-medium text-sm">当前模型</span>
            {modelLoading && (
              <div className="flex items-center gap-1 text-xs text-blue-600">
                <Cpu className="w-3 h-3 animate-spin" />
                加载中...
              </div>
            )}
          </div>

          <div className="text-xs">
            <div className="text-gray-500 break-all">{currentModelPath || "未加载模型"}</div>
          </div>
        </div>

        {/* 常用模型路径示例 */}
        <div className="text-xs text-gray-500">
          <div className="font-medium mb-1">常用路径示例：</div>
          <div className="space-y-1">
            <div>• ./model/yolov5_food.tflite</div>
            <div>• ./model/yolov8_general.tflite</div>
            <div>• ./model/best_float32-2.tflite</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
