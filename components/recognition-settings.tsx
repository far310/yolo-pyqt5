"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Eye } from "@/components/ui/icons"
import type { RecognitionSettings } from "@/types"

interface RecognitionSettingsProps {
  settings: RecognitionSettings
  onSettingsChange: (settings: RecognitionSettings) => void
}

export function RecognitionSettingsComponent({ settings, onSettingsChange }: RecognitionSettingsProps) {
  const updateSettings = (updates: Partial<RecognitionSettings>) => {
    onSettingsChange({ ...settings, ...updates })
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Eye className="w-5 h-5" />
          识别功能
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* 确保每个 Switch 都有独立的容器，不在按钮内部 */}
        <div className="flex items-center justify-between">
          <Label htmlFor="foreign-object">异物识别</Label>
          <Switch
            id="foreign-object"
            checked={settings.foreignObjectDetection}
            onCheckedChange={(checked) => updateSettings({ foreignObjectDetection: checked })}
          />
        </div>

        <div className="flex items-center justify-between">
          <Label htmlFor="size-classification">大中小识别</Label>
          <Switch
            id="size-classification"
            checked={settings.sizeClassification}
            onCheckedChange={(checked) => updateSettings({ sizeClassification: checked })}
          />
        </div>

        {/* <div className="flex items-center justify-between">
          <Label htmlFor="background-change">背景变化检测</Label>
          <Switch
            id="background-change"
            checked={settings.backgroundChangeDetection}
            onCheckedChange={(checked) => updateSettings({ backgroundChangeDetection: checked })}
          />
        </div> */}

        {/* <div className="flex items-center justify-between">
          <Label htmlFor="contour-detection">轮廓检测</Label>
          <Switch
            id="contour-detection"
            checked={settings.contourDetection}
            onCheckedChange={(checked) => updateSettings({ contourDetection: checked })}
          />
        </div> */}

        <div className="flex items-center justify-between">
          <Label htmlFor="height-correction">高度补偿</Label>
          <Switch
            id="height-correction"
            checked={settings.heightCorrection}
            onCheckedChange={(checked) => updateSettings({ heightCorrection: checked })}
          />
        </div>

        {/* <div className="flex items-center justify-between">
          <Label htmlFor="edge-detection">边缘检测</Label>
          <Switch
            id="edge-detection"
            checked={settings.edgeDetection}
            onCheckedChange={(checked) => updateSettings({ edgeDetection: checked })}
          />
        </div>

        <div className="flex items-center justify-between">
          <Label htmlFor="color-analysis">颜色分析</Label>
          <Switch
            id="color-analysis"
            checked={settings.colorAnalysis}
            onCheckedChange={(checked) => updateSettings({ colorAnalysis: checked })}
          />
        </div> */}
      </CardContent>
    </Card>
  )
}
