"use client"

import type React from "react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ChevronDown, ChevronUp, RotateCcw } from "@/components/ui/icons"
import type { ImageParams } from "@/types"
import { useCallback } from "react"

interface ImageParamsControlProps {
  imageParams: ImageParams
  collapsedSections: Record<string, boolean>
  onParamsChange: (params: ImageParams) => void
  onToggleSection: (section: string) => void
  onResetParams: () => void
}

interface SectionHeaderProps {
  title: string
  isCollapsed: boolean
  onToggle: () => void
}

const SectionHeader = ({ title, isCollapsed, onToggle }: SectionHeaderProps) => {
  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      e.stopPropagation()
      console.log(`Section header clicked: ${title}`)
      onToggle()
    },
    [onToggle, title],
  )

  return (
    <div
      className="flex items-center justify-between cursor-pointer hover:bg-gray-50 p-2 rounded transition-colors"
      onClick={handleClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          onToggle()
        }
      }}
    >
      <h4 className="text-sm font-medium pointer-events-none">{title}</h4>
      <div className="pointer-events-none">
        {isCollapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
      </div>
    </div>
  )
}

export function ImageParamsControl({
  imageParams,
  collapsedSections,
  onParamsChange,
  onToggleSection,
  onResetParams,
}: ImageParamsControlProps) {
  const updateParam = useCallback(
    (key: keyof ImageParams, value: any) => {
      const newParams = { ...imageParams, [key]: value }
      onParamsChange(newParams)
    },
    [imageParams, onParamsChange],
  )

  const updateSrcPoint = useCallback(
    (point: keyof ImageParams["srcPoints"], axis: "x" | "y", value: number) => {
      const newSrcPoints = {
        ...imageParams.srcPoints,
        [point]: {
          ...imageParams.srcPoints[point],
          [axis]: value,
        },
      }
      updateParam("srcPoints", newSrcPoints)
    },
    [imageParams.srcPoints, updateParam],
  )

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between">
          <span>图像处理参数</span>
          <Button onClick={onResetParams} variant="outline" size="sm">
            <RotateCcw className="w-4 h-4 mr-1" />
            重置
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* 基础调整 */}
        <div className="space-y-2">
          <SectionHeader
            title="基础调整"
            isCollapsed={collapsedSections.basicAdjustment}
            onToggle={() => onToggleSection("basicAdjustment")}
          />
          {!collapsedSections.basicAdjustment && (
            <div className="space-y-4 pl-4">
              <div className="space-y-2">
                <Label>对比度: {imageParams.contrast}%</Label>
                <Slider
                  value={[imageParams.contrast]}
                  onValueChange={([value]) => updateParam("contrast", value)}
                  min={0}
                  max={200}
                  step={1}
                />
              </div>
              <div className="space-y-2">
                <Label>亮度: {imageParams.brightness}%</Label>
                <Slider
                  value={[imageParams.brightness]}
                  onValueChange={([value]) => updateParam("brightness", value)}
                  min={0}
                  max={200}
                  step={1}
                />
              </div>
              <div className="space-y-2">
                <Label>饱和度: {imageParams.saturation}%</Label>
                <Slider
                  value={[imageParams.saturation]}
                  onValueChange={([value]) => updateParam("saturation", value)}
                  min={0}
                  max={200}
                  step={1}
                />
              </div>
              <div className="space-y-2">
                <Label>模糊: {imageParams.blur}</Label>
                <Slider
                  value={[imageParams.blur]}
                  onValueChange={([value]) => updateParam("blur", value)}
                  min={0}
                  max={10}
                  step={0.1}
                />
              </div>
              <div className="space-y-2">
                <Label>延迟 (秒): {imageParams.delaySeconds}</Label>
                <Slider
                  value={[imageParams.delaySeconds]}
                  onValueChange={([value]) => updateParam("delaySeconds", value)}
                  min={0.1}
                  max={5}
                  step={0.1}
                />
              </div>
            </div>
          )}
        </div>

        {/* 检测阈值 */}
        <div className="space-y-2">
          <SectionHeader
            title="检测阈值"
            isCollapsed={collapsedSections.detectionThreshold}
            onToggle={() => onToggleSection("detectionThreshold")}
          />
          {!collapsedSections.detectionThreshold && (
            <div className="space-y-4 pl-4">
              <div className="space-y-2">
                <Label>目标阈值: {imageParams.objThresh}%</Label>
                <Slider
                  value={[imageParams.objThresh]}
                  onValueChange={([value]) => updateParam("objThresh", value)}
                  min={0}
                  max={100}
                  step={1}
                />
              </div>
              <div className="space-y-2">
                <Label>NMS阈值: {imageParams.nmsThresh}%</Label>
                <Slider
                  value={[imageParams.nmsThresh]}
                  onValueChange={([value]) => updateParam("nmsThresh", value)}
                  min={0}
                  max={100}
                  step={1}
                />
              </div>
              <div className="space-y-2">
                <Label>分数阈值: {imageParams.scoreThresh}%</Label>
                <Slider
                  value={[imageParams.scoreThresh]}
                  onValueChange={([value]) => updateParam("scoreThresh", value)}
                  min={0}
                  max={100}
                  step={1}
                />
              </div>
            </div>
          )}
        </div>

        {/* 透视变换 */}
        <div className="space-y-2">
          <SectionHeader
            title="透视变换"
            isCollapsed={collapsedSections.perspectiveTransform}
            onToggle={() => onToggleSection("perspectiveTransform")}
          />
          {!collapsedSections.perspectiveTransform && (
            <div className="space-y-4 pl-4">
              <div className="flex items-center space-x-2">
                <Switch
                  checked={imageParams.perspectiveEnabled}
                  onCheckedChange={(checked) => updateParam("perspectiveEnabled", checked)}
                />
                <Label>启用透视变换</Label>
              </div>
              {imageParams.perspectiveEnabled && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>左上角</Label>
                    <div className="flex space-x-2">
                      <Input
                        type="number"
                        placeholder="X"
                        value={imageParams.srcPoints.topLeft.x}
                        onChange={(e) => updateSrcPoint("topLeft", "x", Number(e.target.value))}
                      />
                      <Input
                        type="number"
                        placeholder="Y"
                        value={imageParams.srcPoints.topLeft.y}
                        onChange={(e) => updateSrcPoint("topLeft", "y", Number(e.target.value))}
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>右上角</Label>
                    <div className="flex space-x-2">
                      <Input
                        type="number"
                        placeholder="X"
                        value={imageParams.srcPoints.topRight.x}
                        onChange={(e) => updateSrcPoint("topRight", "x", Number(e.target.value))}
                      />
                      <Input
                        type="number"
                        placeholder="Y"
                        value={imageParams.srcPoints.topRight.y}
                        onChange={(e) => updateSrcPoint("topRight", "y", Number(e.target.value))}
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>右下角</Label>
                    <div className="flex space-x-2">
                      <Input
                        type="number"
                        placeholder="X"
                        value={imageParams.srcPoints.bottomRight.x}
                        onChange={(e) => updateSrcPoint("bottomRight", "x", Number(e.target.value))}
                      />
                      <Input
                        type="number"
                        placeholder="Y"
                        value={imageParams.srcPoints.bottomRight.y}
                        onChange={(e) => updateSrcPoint("bottomRight", "y", Number(e.target.value))}
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>左下角</Label>
                    <div className="flex space-x-2">
                      <Input
                        type="number"
                        placeholder="X"
                        value={imageParams.srcPoints.bottomLeft.x}
                        onChange={(e) => updateSrcPoint("bottomLeft", "x", Number(e.target.value))}
                      />
                      <Input
                        type="number"
                        placeholder="Y"
                        value={imageParams.srcPoints.bottomLeft.y}
                        onChange={(e) => updateSrcPoint("bottomLeft", "y", Number(e.target.value))}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* 畸变矫正 */}
        <div className="space-y-2">
          <SectionHeader
            title="畸变矫正"
            isCollapsed={collapsedSections.distortionCorrection}
            onToggle={() => onToggleSection("distortionCorrection")}
          />
          {!collapsedSections.distortionCorrection && (
            <div className="space-y-4 pl-4">
              <div className="flex items-center space-x-2">
                <Switch
                  checked={imageParams.distortionEnabled}
                  onCheckedChange={(checked) => updateParam("distortionEnabled", checked)}
                />
                <Label>启用畸变矫正</Label>
              </div>
              {imageParams.distortionEnabled && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>K1: {imageParams.distortionK1}</Label>
                    <Slider
                      value={[imageParams.distortionK1]}
                      onValueChange={([value]) => updateParam("distortionK1", value)}
                      min={-100}
                      max={100}
                      step={0.1}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>K2: {imageParams.distortionK2}</Label>
                    <Slider
                      value={[imageParams.distortionK2]}
                      onValueChange={([value]) => updateParam("distortionK2", value)}
                      min={-100}
                      max={100}
                      step={0.1}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>P1: {imageParams.distortionP1}</Label>
                    <Slider
                      value={[imageParams.distortionP1]}
                      onValueChange={([value]) => updateParam("distortionP1", value)}
                      min={-1}
                      max={1}
                      step={0.01}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>P2: {imageParams.distortionP2}</Label>
                    <Slider
                      value={[imageParams.distortionP2]}
                      onValueChange={([value]) => updateParam("distortionP2", value)}
                      min={-1}
                      max={1}
                      step={0.01}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>K3: {imageParams.distortionK3}</Label>
                    <Slider
                      value={[imageParams.distortionK3]}
                      onValueChange={([value]) => updateParam("distortionK3", value)}
                      min={-100}
                      max={100}
                      step={0.1}
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* 相机内参 */}
        <div className="space-y-2">
          <SectionHeader
            title="相机内参"
            isCollapsed={collapsedSections.cameraIntrinsics}
            onToggle={() => onToggleSection("cameraIntrinsics")}
          />
          {!collapsedSections.cameraIntrinsics && (
            <div className="space-y-4 pl-4">
              <div className="space-y-2">
                <Label>焦距 fx: {imageParams.focalLengthX}</Label>
                <Slider
                  value={[imageParams.focalLengthX]}
                  onValueChange={([value]) => updateParam("focalLengthX", value)}
                  min={500}
                  max={2000}
                  step={0.01}
                />
              </div>
              <div className="space-y-2">
                <Label>焦距 fy: {imageParams.focalLengthY}</Label>
                <Slider
                  value={[imageParams.focalLengthY]}
                  onValueChange={([value]) => updateParam("focalLengthY", value)}
                  min={500}
                  max={2000}
                  step={0.01}
                />
              </div>
              <div className="space-y-2">
                <Label>主点 cx: {imageParams.principalPointX}</Label>
                <Slider
                  value={[imageParams.principalPointX]}
                  onValueChange={([value]) => updateParam("principalPointX", value)}
                  min={0}
                  max={2000}
                  step={0.01}
                />
              </div>
              <div className="space-y-2">
                <Label>主点 cy: {imageParams.principalPointY}</Label>
                <Slider
                  value={[imageParams.principalPointY]}
                  onValueChange={([value]) => updateParam("principalPointY", value)}
                  min={0}
                  max={1500}
                  step={0.01}
                />
              </div>
            </div>
          )}
        </div>

        {/* 相机参数 */}
        <div className="space-y-2">
          <SectionHeader
            title="相机参数"
            isCollapsed={collapsedSections.cameraParams}
            onToggle={() => onToggleSection("cameraParams")}
          />
          {!collapsedSections.cameraParams && (
            <div className="space-y-4 pl-4">
              <div className="space-y-2">
                <Label>相机高度 (cm): {imageParams.cameraHeight}</Label>
                <Slider
                  value={[imageParams.cameraHeight]}
                  onValueChange={([value]) => updateParam("cameraHeight", value)}
                  min={5}
                  max={50}
                  step={0.1}
                />
              </div>
              <div className="space-y-2">
                <Label>目标高度 (cm): {imageParams.targetHeight}</Label>
                <Slider
                  value={[imageParams.targetHeight]}
                  onValueChange={([value]) => updateParam("targetHeight", value)}
                  min={1}
                  max={10}
                  step={0.1}
                />
              </div>
            </div>
          )}
        </div>

        {/* 背景图像设置 */}
        <div className="space-y-2">
          <SectionHeader
            title="背景图像设置"
            isCollapsed={collapsedSections.backgroundImage}
            onToggle={() => onToggleSection("backgroundImage")}
          />
          {!collapsedSections.backgroundImage && (
            <div className="space-y-4 pl-4">
              <div className="space-y-2">
                <Label>背景图像路径</Label>
                <Input
                  type="text"
                  placeholder="例如: ./img/bg.jpg"
                  value={imageParams.backgroundImagePath}
                  onChange={(e) => updateParam("backgroundImagePath", e.target.value)}
                  className="w-full"
                />
                <p className="text-xs text-gray-500">用于异物检测的背景参考图像路径，支持相对路径和绝对路径</p>
              </div>
            </div>
          )}
        </div>

        {/* 尺寸分类 */}
        <div className="space-y-2">
          <SectionHeader
            title="尺寸分类"
            isCollapsed={collapsedSections.sizeClassification}
            onToggle={() => onToggleSection("sizeClassification")}
          />
          {!collapsedSections.sizeClassification && (
            <div className="space-y-4 pl-4">
              <div className="space-y-2">
                <Label>汉堡最小尺寸 (cm): {imageParams.hamburgerSizeMin}</Label>
                <Slider
                  value={[imageParams.hamburgerSizeMin]}
                  onValueChange={([value]) => updateParam("hamburgerSizeMin", value)}
                  min={5}
                  max={20}
                  step={0.1}
                />
              </div>
              <div className="space-y-2">
                <Label>汉堡最大尺寸 (cm): {imageParams.hamburgerSizeMax}</Label>
                <Slider
                  value={[imageParams.hamburgerSizeMax]}
                  onValueChange={([value]) => updateParam("hamburgerSizeMax", value)}
                  min={10}
                  max={25}
                  step={0.1}
                />
              </div>
            </div>
          )}
        </div>

        {/* 实际尺寸 */}
        <div className="space-y-2">
          <SectionHeader
            title="实际尺寸"
            isCollapsed={collapsedSections.realSize}
            onToggle={() => onToggleSection("realSize")}
          />
          {!collapsedSections.realSize && (
            <div className="space-y-4 pl-4">
              <div className="space-y-2">
                <Label>实际宽度 (cm): {imageParams.realWidthCm}</Label>
                <Slider
                  value={[imageParams.realWidthCm]}
                  onValueChange={([value]) => updateParam("realWidthCm", value)}
                  min={10}
                  max={50}
                  step={0.1}
                />
              </div>
              <div className="space-y-2">
                <Label>实际高度 (cm): {imageParams.realHeightCm}</Label>
                <Slider
                  value={[imageParams.realHeightCm]}
                  onValueChange={([value]) => updateParam("realHeightCm", value)}
                  min={10}
                  max={30}
                  step={0.1}
                />
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
