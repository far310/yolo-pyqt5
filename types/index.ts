export interface ModelConfig {
  path: string
  name?: string
}

export interface ImageParams {
  contrast: number
  brightness: number
  saturation: number
  blur: number
  // 延迟设置
  delaySeconds: number
  // 检测阈值参数
  objThresh: number
  nmsThresh: number
  scoreThresh: number
  // 透视变换参数
  perspectiveEnabled: boolean
  srcPoints: {
    topLeft: { x: number; y: number }
    topRight: { x: number; y: number }
    bottomRight: { x: number; y: number }
    bottomLeft: { x: number; y: number }
  }
  // 畸变矫正参数
  distortionEnabled: boolean
  distortionK1: number
  distortionK2: number
  distortionP1: number
  distortionP2: number
  distortionK3: number
  // 相机内参
  focalLengthX: number
  focalLengthY: number
  principalPointX: number
  principalPointY: number
  // 相机参数
  cameraHeight: number
  targetHeight: number
  // 尺寸分类参数
  hamburgerSizeMin: number
  hamburgerSizeMax: number
  // 实际尺寸参数
  realWidthCm: number
  realHeightCm: number
}

export interface RecognitionSettings {
  foreignObjectDetection: boolean
  sizeClassification: boolean
  edgeDetection: boolean
  colorAnalysis: boolean
  backgroundChangeDetection: boolean
  contourDetection: boolean
  heightCorrection: boolean
}

export interface DetectedObject {
  id: string
  type: string
  confidence: number
  size: "small" | "medium" | "large"
  x: number
  y: number
  width: number
  height: number
}

export interface PythonResponse<T = any> {
  success: boolean
  data?: T
  error?: string
}
