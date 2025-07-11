"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Settings, RotateCcw, ChevronDown, ChevronRight, Clock } from "@/components/ui/icons"
import type { ImageParams } from "@/types"

interface ImageParamsControlProps {
  imageParams: ImageParams
  collapsedSections: Record<string, boolean>
  onParamsChange: (params: ImageParams) => void
  onToggleSection: (section: string) => void
  onResetParams: () => void
}

export function ImageParamsControl({
  imageParams,
  collapsedSections,
  onParamsChange,
  onToggleSection,
  onResetParams,
}: ImageParamsControlProps) {
  const updateParams = (updates: Partial<ImageParams>) => {
    onParamsChange({ ...imageParams, ...updates })
  }

  const SectionHeader = ({
    title,
    sectionKey,
    className = "",
  }: {
    title: string
    sectionKey: string
    className?: string
  }) => (
    <div
      className={`flex items-center justify-between w-full p-2 hover:bg-gray-50 rounded cursor-pointer ${className}`}
      onClick={() => {
        console.log(`Toggling section: ${sectionKey}`)
        onToggleSection(sectionKey)
      }}
      style={{ userSelect: "none" }}
    >
      <h4 className="font-medium text-sm text-gray-700">{title}</h4>
      {collapsedSections[sectionKey] ? <ChevronRight className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
    </div>
  )

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Settings className="w-5 h-5" />
          图像处理参数
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* 延迟设置 */}
        <div className="bg-blue-50 p-3 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-4 h-4 text-blue-600" />
            <Label htmlFor="delaySeconds" className="font-medium text-blue-800">
              延迟设置: {imageParams.delaySeconds}秒
            </Label>
          </div>
          <Input
            id="delaySeconds"
            type="number"
            value={imageParams.delaySeconds}
            onChange={(e) => updateParams({ delaySeconds: Number(e.target.value) || 0 })}
            min={0}
            max={60}
            step={0.1}
            className="bg-white"
          />
        </div>

        {/* 基础图像调整 */}
        <div>
          <SectionHeader title="基础调整" sectionKey="basicAdjustment" />
          {!collapsedSections.basicAdjustment && (
            <div className="space-y-3 pt-2">
              <div>
                <Label htmlFor="contrast">对比度: {imageParams.contrast}%</Label>
                <Input
                  id="contrast"
                  type="number"
                  value={imageParams.contrast}
                  onChange={(e) => updateParams({ contrast: Number(e.target.value) || 0 })}
                  min={0}
                  max={200}
                  step={1}
                  className="mt-2"
                />
              </div>

              <div>
                <Label htmlFor="brightness">亮度: {imageParams.brightness}%</Label>
                <Input
                  id="brightness"
                  type="number"
                  value={imageParams.brightness}
                  onChange={(e) => updateParams({ brightness: Number(e.target.value) || 0 })}
                  min={0}
                  max={200}
                  step={1}
                  className="mt-2"
                />
              </div>

              <div>
                <Label htmlFor="saturation">饱和度: {imageParams.saturation}%</Label>
                <Input
                  id="saturation"
                  type="number"
                  value={imageParams.saturation}
                  onChange={(e) => updateParams({ saturation: Number(e.target.value) || 0 })}
                  min={0}
                  max={200}
                  step={1}
                  className="mt-2"
                />
              </div>

              <div>
                <Label htmlFor="blur">模糊度: {imageParams.blur}px</Label>
                <Input
                  id="blur"
                  type="number"
                  value={imageParams.blur}
                  onChange={(e) => updateParams({ blur: Number(e.target.value) || 0 })}
                  min={0}
                  max={10}
                  step={0.1}
                  className="mt-2"
                />
              </div>
            </div>
          )}
        </div>

        {/* 检测阈值 */}
        <div>
          <SectionHeader title="检测阈值" sectionKey="detectionThreshold" className="border-t pt-3" />
          {!collapsedSections.detectionThreshold && (
            <div className="space-y-3 pt-2">
              <div>
                <Label htmlFor="objThresh">目标阈值: {imageParams.objThresh}%</Label>
                <Input
                  id="objThresh"
                  type="number"
                  value={imageParams.objThresh}
                  onChange={(e) => updateParams({ objThresh: Number(e.target.value) || 0 })}
                  min={0}
                  max={100}
                  step={1}
                  className="mt-2"
                />
              </div>

              <div>
                <Label htmlFor="nmsThresh">NMS阈值: {imageParams.nmsThresh}%</Label>
                <Input
                  id="nmsThresh"
                  type="number"
                  value={imageParams.nmsThresh}
                  onChange={(e) => updateParams({ nmsThresh: Number(e.target.value) || 0 })}
                  min={0}
                  max={100}
                  step={1}
                  className="mt-2"
                />
              </div>

              <div>
                <Label htmlFor="scoreThresh">分数阈值: {imageParams.scoreThresh}</Label>
                <Input
                  id="scoreThresh"
                  type="number"
                  value={imageParams.scoreThresh}
                  onChange={(e) => updateParams({ scoreThresh: Number(e.target.value) || 0 })}
                  min={0}
                  max={100}
                  step={1}
                  className="mt-2"
                />
              </div>
            </div>
          )}
        </div>

        {/* 透视变换 */}
        <div>
          <SectionHeader title="透视变换" sectionKey="perspectiveTransform" className="border-t pt-3" />
          {!collapsedSections.perspectiveTransform && (
            <div className="pt-2">
              <div className="mb-3">
                <div className="flex items-center gap-2">
                  <Label htmlFor="perspectiveEnabled">启用透视变换</Label>
                  <Switch
                    id="perspectiveEnabled"
                    checked={imageParams.perspectiveEnabled}
                    onCheckedChange={(checked) => updateParams({ perspectiveEnabled: checked })}
                  />
                </div>
              </div>
              {imageParams.perspectiveEnabled && (
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <Label htmlFor="topLeftX">左上 X</Label>
                    <Input
                      id="topLeftX"
                      type="number"
                      value={imageParams.srcPoints.topLeft.x}
                      onChange={(e) =>
                        updateParams({
                          srcPoints: {
                            ...imageParams.srcPoints,
                            topLeft: { ...imageParams.srcPoints.topLeft, x: Number(e.target.value) || 0 },
                          },
                        })
                      }
                      min={0}
                      max={1920}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="topLeftY">左上 Y</Label>
                    <Input
                      id="topLeftY"
                      type="number"
                      value={imageParams.srcPoints.topLeft.y}
                      onChange={(e) =>
                        updateParams({
                          srcPoints: {
                            ...imageParams.srcPoints,
                            topLeft: { ...imageParams.srcPoints.topLeft, y: Number(e.target.value) || 0 },
                          },
                        })
                      }
                      min={0}
                      max={1080}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="topRightX">右上 X</Label>
                    <Input
                      id="topRightX"
                      type="number"
                      value={imageParams.srcPoints.topRight.x}
                      onChange={(e) =>
                        updateParams({
                          srcPoints: {
                            ...imageParams.srcPoints,
                            topRight: { ...imageParams.srcPoints.topRight, x: Number(e.target.value) || 0 },
                          },
                        })
                      }
                      min={0}
                      max={1920}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="topRightY">右上 Y</Label>
                    <Input
                      id="topRightY"
                      type="number"
                      value={imageParams.srcPoints.topRight.y}
                      onChange={(e) =>
                        updateParams({
                          srcPoints: {
                            ...imageParams.srcPoints,
                            topRight: { ...imageParams.srcPoints.topRight, y: Number(e.target.value) || 0 },
                          },
                        })
                      }
                      min={0}
                      max={1080}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="bottomRightX">右下 X</Label>
                    <Input
                      id="bottomRightX"
                      type="number"
                      value={imageParams.srcPoints.bottomRight.x}
                      onChange={(e) =>
                        updateParams({
                          srcPoints: {
                            ...imageParams.srcPoints,
                            bottomRight: { ...imageParams.srcPoints.bottomRight, x: Number(e.target.value) || 0 },
                          },
                        })
                      }
                      min={0}
                      max={1920}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="bottomRightY">右下 Y</Label>
                    <Input
                      id="bottomRightY"
                      type="number"
                      value={imageParams.srcPoints.bottomRight.y}
                      onChange={(e) =>
                        updateParams({
                          srcPoints: {
                            ...imageParams.srcPoints,
                            bottomRight: { ...imageParams.srcPoints.bottomRight, y: Number(e.target.value) || 0 },
                          },
                        })
                      }
                      min={0}
                      max={1080}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="bottomLeftX">左下 X</Label>
                    <Input
                      id="bottomLeftX"
                      type="number"
                      value={imageParams.srcPoints.bottomLeft.x}
                      onChange={(e) =>
                        updateParams({
                          srcPoints: {
                            ...imageParams.srcPoints,
                            bottomLeft: { ...imageParams.srcPoints.bottomLeft, x: Number(e.target.value) || 0 },
                          },
                        })
                      }
                      min={0}
                      max={1920}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="bottomLeftY">左下 Y</Label>
                    <Input
                      id="bottomLeftY"
                      type="number"
                      value={imageParams.srcPoints.bottomLeft.y}
                      onChange={(e) =>
                        updateParams({
                          srcPoints: {
                            ...imageParams.srcPoints,
                            bottomLeft: { ...imageParams.srcPoints.bottomLeft, y: Number(e.target.value) || 0 },
                          },
                        })
                      }
                      min={0}
                      max={1080}
                      className="mt-1"
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* 畸变矫正 */}
        <div>
          <SectionHeader title="畸变矫正" sectionKey="distortionCorrection" className="border-t pt-3" />
          {!collapsedSections.distortionCorrection && (
            <div className="pt-2">
              <div className="mb-3">
                <div className="flex items-center gap-2">
                  <Label htmlFor="distortionEnabled">启用畸变矫正</Label>
                  <Switch
                    id="distortionEnabled"
                    checked={imageParams.distortionEnabled}
                    onCheckedChange={(checked) => updateParams({ distortionEnabled: checked })}
                  />
                </div>
              </div>
              {imageParams.distortionEnabled && (
                <div className="space-y-2">
                  <div>
                    <Label htmlFor="distortionK1">K1: {imageParams.distortionK1}</Label>
                    <Input
                      id="distortionK1"
                      type="number"
                      value={imageParams.distortionK1}
                      onChange={(e) => updateParams({ distortionK1: Number(e.target.value) || 0 })}
                      step={0.1}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="distortionK2">K2: {imageParams.distortionK2}</Label>
                    <Input
                      id="distortionK2"
                      type="number"
                      value={imageParams.distortionK2}
                      onChange={(e) => updateParams({ distortionK2: Number(e.target.value) || 0 })}
                      step={0.1}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="distortionP1">P1: {imageParams.distortionP1}</Label>
                    <Input
                      id="distortionP1"
                      type="number"
                      value={imageParams.distortionP1}
                      onChange={(e) => updateParams({ distortionP1: Number(e.target.value) || 0 })}
                      step={0.01}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="distortionP2">P2: {imageParams.distortionP2}</Label>
                    <Input
                      id="distortionP2"
                      type="number"
                      value={imageParams.distortionP2}
                      onChange={(e) => updateParams({ distortionP2: Number(e.target.value) || 0 })}
                      step={0.01}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="distortionK3">K3: {imageParams.distortionK3}</Label>
                    <Input
                      id="distortionK3"
                      type="number"
                      value={imageParams.distortionK3}
                      onChange={(e) => updateParams({ distortionK3: Number(e.target.value) || 0 })}
                      step={0.1}
                      className="mt-1"
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* 相机参数 */}
        <div>
          <SectionHeader title="相机参数" sectionKey="cameraParams" className="border-t pt-3" />
          {!collapsedSections.cameraParams && (
            <div className="space-y-3 pt-2">
              <div>
                <Label htmlFor="focalLength">焦距: {imageParams.focalLength}</Label>
                <Input
                  id="focalLength"
                  type="number"
                  value={imageParams.focalLength}
                  onChange={(e) => updateParams({ focalLength: Number(e.target.value) || 0 })}
                  step={0.1}
                  className="mt-2"
                />
              </div>

              <div>
                <Label htmlFor="cameraHeight">相机高度: {imageParams.cameraHeight}cm</Label>
                <Input
                  id="cameraHeight"
                  type="number"
                  value={imageParams.cameraHeight}
                  onChange={(e) => updateParams({ cameraHeight: Number(e.target.value) || 0 })}
                  step={0.1}
                  className="mt-2"
                />
              </div>

              <div>
                <Label htmlFor="targetHeight">目标高度: {imageParams.targetHeight}cm</Label>
                <Input
                  id="targetHeight"
                  type="number"
                  value={imageParams.targetHeight}
                  onChange={(e) => updateParams({ targetHeight: Number(e.target.value) || 0 })}
                  step={0.1}
                  className="mt-2"
                />
              </div>
            </div>
          )}
        </div>

        {/* 尺寸分类 */}
        <div>
          <SectionHeader title="尺寸分类" sectionKey="sizeClassification" className="border-t pt-3" />
          {!collapsedSections.sizeClassification && (
            <div className="space-y-3 pt-2">
              <div>
                <Label htmlFor="hamburgerSizeMin">小号上限: {imageParams.hamburgerSizeMin}cm</Label>
                <Input
                  id="hamburgerSizeMin"
                  type="number"
                  value={imageParams.hamburgerSizeMin}
                  onChange={(e) => updateParams({ hamburgerSizeMin: Number(e.target.value) || 0 })}
                  step={0.1}
                  className="mt-2"
                />
              </div>

              <div>
                <Label htmlFor="hamburgerSizeMax">中号上限: {imageParams.hamburgerSizeMax}cm</Label>
                <Input
                  id="hamburgerSizeMax"
                  type="number"
                  value={imageParams.hamburgerSizeMax}
                  onChange={(e) => updateParams({ hamburgerSizeMax: Number(e.target.value) || 0 })}
                  step={0.1}
                  className="mt-2"
                />
              </div>
            </div>
          )}
        </div>

        {/* 实际尺寸 */}
        <div>
          <SectionHeader title="实际尺寸" sectionKey="realSize" className="border-t pt-3" />
          {!collapsedSections.realSize && (
            <div className="space-y-3 pt-2">
              <div>
                <Label htmlFor="realWidthCm">实际宽度: {imageParams.realWidthCm}cm</Label>
                <Input
                  id="realWidthCm"
                  type="number"
                  value={imageParams.realWidthCm}
                  onChange={(e) => updateParams({ realWidthCm: Number(e.target.value) || 0 })}
                  step={0.1}
                  className="mt-2"
                />
              </div>

              <div>
                <Label htmlFor="realHeightCm">实际高度: {imageParams.realHeightCm}cm</Label>
                <Input
                  id="realHeightCm"
                  type="number"
                  value={imageParams.realHeightCm}
                  onChange={(e) => updateParams({ realHeightCm: Number(e.target.value) || 0 })}
                  step={0.1}
                  className="mt-2"
                />
              </div>
            </div>
          )}
        </div>

        <Button onClick={onResetParams} variant="outline" size="sm" className="w-full bg-transparent">
          <RotateCcw className="w-4 h-4 mr-1" />
          重置所有参数
        </Button>
      </CardContent>
    </Card>
  )
}
