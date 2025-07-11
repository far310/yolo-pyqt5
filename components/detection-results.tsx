"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Zap, Eye, FileText } from "@/components/ui/icons"
import type { DetectedObject } from "@/types"

interface DetectionResultsProps {
  objects: DetectedObject[]
  isExporting?: boolean // 新增：导出状态
  onExportReport?: () => void
}

export function DetectionResults({ objects, isExporting = false, onExportReport }: DetectionResultsProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5" />
            检测结果
          </div>
          {onExportReport && objects.length > 0 && (
            <Button onClick={onExportReport} disabled={isExporting} variant="outline" size="sm">
              <FileText className="w-4 h-4 mr-1" />
              {isExporting ? "导出中..." : "导出报告"}
            </Button>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {objects.length > 0 ? (
          <div className="space-y-3">
            {objects.map((obj) => (
              <div key={obj.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-3">
                  <Badge variant={obj.type === "异物" || obj.type === "缺陷" ? "destructive" : "default"}>
                    {obj.type}
                  </Badge>
                  <span className="text-sm text-gray-600">置信度: {(obj.confidence * 100).toFixed(1)}%</span>
                  <Badge variant="outline">{obj.size === "small" ? "小" : obj.size === "medium" ? "中" : "大"}</Badge>
                </div>
                <div className="text-sm text-gray-500">
                  位置: ({obj.x.toFixed(0)}, {obj.y.toFixed(0)})
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <Eye className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>暂无检测结果</p>
            <p className="text-sm">启用识别功能并开始检测</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
