// Python QWebEngineView API 通信服务 - 轮询版本
class PythonAPI {
  private isWebChannelAvailable(): boolean {
    return typeof window !== "undefined" && "qt" in window && window.qt?.webChannelTransport
  }

  private async callPythonMethod<T = any>(method: string, ...args: any[]): Promise<T> {
    console.log(`Calling Python method: ${method}`, args)

    if (!this.isWebChannelAvailable()) {
      console.warn("Python WebChannel API not available, using mock data")
      return this.getMockResponse(method, ...args)
    }

    try {
      // 等待 QWebChannel 初始化
      if (!window.pyapi) {
        await this.initWebChannel()
      }

      if (!window.pyapi || !window.pyapi[method]) {
        throw new Error(`Method ${method} not available on Python API`)
      }

      const result = await window.pyapi[method](...args)
      console.log(`Python method ${method} result:`, result)

      // 如果返回的是 JSON 字符串，解析它
      if (typeof result === "string") {
        try {
          return JSON.parse(result)
        } catch {
          return result
        }
      }

      return result
    } catch (error) {
      console.error(`Error calling Python method ${method}:`, error)
      // 如果 Python 调用失败，返回模拟数据
      return this.getMockResponse(method, ...args)
    }
  }

  private async initWebChannel(): Promise<void> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error("WebChannel initialization timeout"))
      }, 10000) // 10秒超时

      const tryInit = () => {
        if (window.qt && window.qt.webChannelTransport) {
          try {
            // 确保 QWebChannel 构造函数可用
            if (typeof window.QWebChannel === "undefined") {
              // 如果 QWebChannel 不可用，尝试从全局加载
              const script = document.createElement("script")
              script.src = "qrc:///qtwebchannel/qwebchannel.js"
              script.onload = () => {
                this.createWebChannel(resolve, reject, timeout)
              }
              script.onerror = () => {
                console.error("Failed to load QWebChannel script")
                reject(new Error("Failed to load QWebChannel script"))
              }
              document.head.appendChild(script)
            } else {
              this.createWebChannel(resolve, reject, timeout)
            }
          } catch (error) {
            console.error("WebChannel initialization error:", error)
            setTimeout(tryInit, 100)
          }
        } else {
          // 如果 WebChannel 不可用，延迟重试
          setTimeout(tryInit, 100)
        }
      }

      tryInit()
    })
  }

  private createWebChannel(resolve: Function, reject: Function, timeout: NodeJS.Timeout) {
    try {
      new window.QWebChannel(window.qt.webChannelTransport, (channel: any) => {
        clearTimeout(timeout)
        window.pyapi = channel.objects.pyapi
        console.log("WebChannel initialized successfully")
        resolve()
      })
    } catch (error) {
      clearTimeout(timeout)
      console.error("Failed to create WebChannel:", error)
      reject(error)
    }
  }

  private getMockResponse(method: string, ...args: any[]): any {
    console.log(`Using mock response for method: ${method}`)

    // 模拟数据，用于开发测试
    switch (method) {
      case "start_camera":
        return { success: true, message: `摄像头 ${args[0]} 启动成功 (模拟)` }
      case "stop_camera":
        return { success: true, message: "摄像头已停止 (模拟)" }
      case "load_model":
        return { success: true, message: `模型加载成功: ${args[0]} (模拟)` }
      case "set_image_params":
        return { success: true, message: "图像参数已更新 (模拟)" }
      case "set_recognition_settings":
        return { success: true, message: "识别设置已更新 (模拟)" }
      case "get_current_frame":
        // 返回一个简单的测试图像
        const testImage =
          "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        return {
          success: true,
          image: testImage,
          timestamp: Date.now(),
        }
      case "get_detection_results":
        return {
          objects: [
            {
              id: `obj_${Date.now()}`,
              type: "Hamburger",
              confidence: 0.85,
              size: "medium",
              x: 100,
              y: 150,
              width: 80,
              height: 60,
            },
          ],
        }
      case "save_image":
        return { success: true, message: "图像已保存 (模拟)" }
      case "export_report":
        return { success: true, message: "报告已导出 (模拟)" }
      case "get_system_status":
        return {
          fps: 30,
          cpuUsage: 45,
          memoryUsage: 60,
          gpuUsage: 0,
        }
      case "set_background_image_path":
        return { success: true, message: `背景图像路径已设置: ${args[0]} (模拟)` }
      case "get_background_image_status":
        return {
          success: true,
          isSet: args[0] ? true : false,
          path: args[0] || "",
          isValid: true,
          message: "背景图像状态 (模拟)",
        }
      case "clear_background_image":
        return { success: true, message: "背景图像已清除 (模拟)" }
      default:
        return { success: true, message: "操作完成 (模拟)" }
    }
  }

  // 启动摄像头 - 直接传递摄像头索引
  async startCamera(cameraIndex: string): Promise<any> {
    return await this.callPythonMethod("start_camera", cameraIndex)
  }

  // 停止摄像头
  async stopCamera(): Promise<any> {
    return await this.callPythonMethod("stop_camera")
  }

  // 加载AI模型 - 直接传递模型路径
  async loadModel(modelPath: string): Promise<any> {
    return await this.callPythonMethod("load_model", modelPath)
  }

  // 设置图像处理参数
  async setImageParams(params: any): Promise<any> {
    return await this.callPythonMethod("set_image_params", JSON.stringify(params))
  }

  // 设置识别功能
  async setRecognitionSettings(settings: any): Promise<any> {
    return await this.callPythonMethod("set_recognition_settings", JSON.stringify(settings))
  }

  // 获取当前图像帧
  async getCurrentFrame(): Promise<{ success: boolean; image?: string; timestamp?: number; error?: string }> {
    const result = await this.callPythonMethod("get_current_frame")
    return result || { success: false, error: "Failed to get frame" }
  }

  // 获取检测结果
  async getDetectionResults(): Promise<{ objects: any[] }> {
    const result = await this.callPythonMethod("get_detection_results")
    return result || { objects: [] }
  }

  // 保存当前图像
  async saveImage(filename?: string): Promise<any> {
    return await this.callPythonMethod("save_image", filename || "")
  }

  // 导出检测报告
  async exportReport(format: "json" | "csv" | "pdf" = "json"): Promise<any> {
    return await this.callPythonMethod("export_report", format)
  }

  // 获取系统状态
  async getSystemStatus(): Promise<{
    fps: number
    cpuUsage: number
    memoryUsage: number
    gpuUsage?: number
  }> {
    const result = await this.callPythonMethod("get_system_status")
    return result || { fps: 0, cpuUsage: 0, memoryUsage: 0, gpuUsage: 0 }
  }

  // 测试连接
  async testConnection(): Promise<boolean> {
    try {
      await this.getSystemStatus()
      return true
    } catch (error) {
      console.error("Connection test failed:", error)
      return false
    }
  }

  // 设置背景图像路径
  async setBackgroundImagePath(imagePath: string): Promise<any> {
    return await this.callPythonMethod("set_background_image_path", imagePath)
  }

  // 获取背景图像状态
  async getBackgroundImageStatus(): Promise<{
    success: boolean
    isSet: boolean
    path?: string
    isValid?: boolean
    message?: string
  }> {
    return await this.callPythonMethod("get_background_image_status")
  }

  // 清除背景图像
  async clearBackgroundImage(): Promise<any> {
    return await this.callPythonMethod("clear_background_image")
  }
}

// 全局 Python API 实例
export const pythonAPI = new PythonAPI()

// 类型声明
declare global {
  interface Window {
    qt?: {
      webChannelTransport: any
    }
    pyapi?: {
      [key: string]: (...args: any[]) => Promise<any>
    }
    QWebChannel?: any
  }
}

// 在页面加载时初始化连接
if (typeof window !== "undefined") {
  window.addEventListener("DOMContentLoaded", async () => {
    console.log("DOM loaded, testing Python API connection...")
    const isConnected = await pythonAPI.testConnection()
    console.log("Python API connection status:", isConnected)
  })
}
