"use client"

import { Card, CardContent } from "@/components/ui/card"

interface SystemStatsProps {
  objectCount: number
  anomalyCount: number
  averageConfidence: number
  fps: number
  isStreaming: boolean
}

export function SystemStats({ objectCount, anomalyCount, averageConfidence, fps, isStreaming }: SystemStatsProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-blue-600">{objectCount}</div>
          <div className="text-sm text-gray-600">检测对象</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-red-600">{anomalyCount}</div>
          <div className="text-sm text-gray-600">异常数量</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-green-600">
            {objectCount > 0 ? `${averageConfidence.toFixed(1)}%` : "0%"}
          </div>
          <div className="text-sm text-gray-600">平均置信度</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-purple-600">{isStreaming ? fps : 0} FPS</div>
          <div className="text-sm text-gray-600">处理帧率</div>
        </CardContent>
      </Card>
    </div>
  )
}
