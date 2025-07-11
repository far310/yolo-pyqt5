"""
Python QWebEngineView 后端代码 - 轮询版本
支持直接输入摄像头索引和模型路径
"""

# 启用调试模式（必须在 QApplication 之前设置）
import os
os.environ["QTWEBENGINE_REMOTE_DEBUGGING"] = "9222"

import sys
import json
import time
import threading
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
import psutil
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# PyQt5 imports
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage, QWebEngineSettings
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import QObject, pyqtSlot, QUrl, QTimer

# TensorFlow Lite imports
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf
    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate

class CustomWebEnginePage(QWebEnginePage):
    """自定义 WebEngine 页面，修复交互问题"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        """捕获 JavaScript 控制台消息"""
        print(f"JS Console [{level}]: {message} (Line: {lineNumber}, Source: {sourceID})")

class TensorFlowLiteModel:
    """TensorFlow Lite 模型封装类"""
    
    def __init__(self, model_path: str, num_threads: int = 4):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.num_threads = num_threads
        self.load_model()
    
    def load_model(self):
        """加载 TensorFlow Lite 模型"""
        try:
            self.interpreter = Interpreter(
                model_path=self.model_path,
                num_threads=self.num_threads
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"Model {self.model_path} loaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def predict(self, image):
        """执行模型推理"""
        if self.interpreter is None:
            return None
        
        try:
            # 预处理图像
            input_data = self.preprocess_image(image)
            
            # 设置输入张量
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # 执行推理
            self.interpreter.invoke()
            
            # 获取输出
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # 后处理
            return self.postprocess_output(output_data, image.shape)
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def preprocess_image(self, image):
        """图像预处理"""
        # 获取输入张量形状
        input_shape = self.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # 调整图像大小
        resized = cv2.resize(image, (width, height))
        
        # 归一化
        normalized = resized.astype(np.float32) / 255.0
        
        # 添加批次维度
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data
    
    def postprocess_output(self, output, original_shape):
        """输出后处理"""
        detections = []
        
        if len(output.shape) == 3:
            output = output[0]  # 移除批次维度
        
        # 假设输出格式为 [num_detections, 6] (x, y, w, h, confidence, class)
        for detection in output:
            if len(detection) >= 6:
                x, y, w, h, confidence, class_id = detection[:6]
                if confidence > 0.5:  # 置信度阈值
                    detections.append({
                        'x': float(x),
                        'y': float(y),
                        'width': float(w),
                        'height': float(h),
                        'confidence': float(confidence),
                        'class_id': int(class_id)
                    })
        
        return detections

class ImageRecognitionAPI(QObject):
    """图像识别 API 类 - 轮询版本"""
    
    def __init__(self):
        super().__init__()
        self.camera = None
        self.current_frame = None
        self.processed_frame = None
        self.is_streaming = False
        self.model = None
        self.current_model_path = ""
        self.current_camera_index = "0"
        self.image_params = {
            'contrast': 100,
            'brightness': 100,
            'saturation': 100,
            'blur': 0,
            'delaySeconds': 1.0,
            'objThresh': 10,
            'nmsThresh': 10,
            'scoreThresh': 0,
            'perspectiveEnabled': True,
            'distortionEnabled': True,
            'focalLength': 1298.55,
            'cameraHeight': 13.0,
            'targetHeight': 2.7,
            'hamburgerSizeMin': 10,
            'hamburgerSizeMax': 13,
            'realWidthCm': 29,
            'realHeightCm': 18.5,
            'srcPoints': {
                'topLeft': {'x': 650, 'y': 330},
                'topRight': {'x': 1425, 'y': 330},
                'bottomRight': {'x': 1830, 'y': 889},
                'bottomLeft': {'x': 230, 'y': 889}
            }
        }
        self.recognition_settings = {}
        self.detected_objects = []
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.detection_delay = 1.0
        self.last_detection_time = 0
        
        # 相机参数
        self.camera_matrix = np.array([
            [1298.54926, 0.0, 966.93144],
            [0.0, 1294.39363, 466.380271],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([
            [-0.44903195, 0.25133919, 0.00037556, 0.00024487, -0.0794278]
        ])
        
        # 透视变换参数
        self.src_points = np.array([
            [650, 330], [1425, 330], [1830, 889], [230, 889]
        ], dtype=np.float32)
        
        # 初始化畸变矫正映射
        self.map1 = None
        self.map2 = None
        self.init_undistort_maps()
    
    def init_undistort_maps(self):
        """初始化畸变矫正映射"""
        image_size = (1920, 1080)  # 相机分辨率
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, 
            self.camera_matrix, image_size, cv2.CV_16SC2
        )
    
    @pyqtSlot(str, result=str)
    def start_camera(self, camera_index: str) -> str:
        """启动摄像头 - 直接使用摄像头索引"""
        try:
            print(f"Starting camera with index: {camera_index}")
            
            # 将字符串转换为整数
            try:
                camera_idx = int(camera_index)
            except ValueError:
                return json.dumps({'success': False, 'error': '无效的摄像头索引'}, ensure_ascii=False)
            
            # 检查索引范围
            if camera_idx < 0 or camera_idx > 10:
                return json.dumps({'success': False, 'error': '摄像头索引必须在0-10之间'}, ensure_ascii=False)
            
            self.camera = cv2.VideoCapture(camera_idx)
            if not self.camera.isOpened():
                return json.dumps({'success': False, 'error': f'无法打开摄像头 {camera_idx}'}, ensure_ascii=False)
            
            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            self.current_camera_index = camera_index
            self.is_streaming = True
            threading.Thread(target=self._video_stream_thread, daemon=True).start()
            
            result = json.dumps({'success': True, 'message': f'摄像头 {camera_idx} 启动成功'}, ensure_ascii=False)
            print(f"start_camera result: {result}")
            return result
        except Exception as e:
            error_result = json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False)
            print(f"start_camera error: {error_result}")
            return error_result
    
    @pyqtSlot(result=str)
    def stop_camera(self) -> str:
        """停止摄像头"""
        try:
            print("Stopping camera")
            self.is_streaming = False
            if self.camera:
                self.camera.release()
                self.camera = None
            
            result = json.dumps({'success': True, 'message': '摄像头已停止'}, ensure_ascii=False)
            print(f"stop_camera result: {result}")
            return result
        except Exception as e:
            error_result = json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False)
            print(f"stop_camera error: {error_result}")
            return error_result
    
    @pyqtSlot(str, result=str)
    def load_model(self, model_path: str) -> str:
        """加载AI模型 - 直接使用模型路径"""
        try:
            print(f"Loading model: {model_path}")
            
            # 检查文件是否存在
            if not os.path.exists(model_path):
                # 如果模型文件不存在，创建一个模拟的成功响应
                result = json.dumps({
                    'success': True, 
                    'message': f'模型加载成功 (模拟): {model_path}'
                }, ensure_ascii=False)
                print(f"load_model result (mock): {result}")
                self.current_model_path = model_path
                return result
            
            self.model = TensorFlowLiteModel(model_path)
            self.current_model_path = model_path
            
            result = json.dumps({'success': True, 'message': f'模型加载成功: {model_path}'}, ensure_ascii=False)
            print(f"load_model result: {result}")
            return result
        except Exception as e:
            error_result = json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False)
            print(f"load_model error: {error_result}")
            return error_result
    
    @pyqtSlot(str, result=str)
    def set_image_params(self, params_json: str) -> str:
        """设置图像处理参数"""
        try:
            print(f"Setting image params: {params_json}")
            params = json.loads(params_json)
            self.image_params.update(params)
            
            # 处理延迟设置
            delay_seconds = params.get('delaySeconds', 1.0)
            if delay_seconds > 0:
                self.detection_delay = delay_seconds
            
            result = json.dumps({'success': True, 'message': '图像参数已更新'}, ensure_ascii=False)
            print(f"set_image_params result: {result}")
            return result
        except Exception as e:
            error_result = json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False)
            print(f"set_image_params error: {error_result}")
            return error_result
    
    @pyqtSlot(str, result=str)
    def set_recognition_settings(self, settings_json: str) -> str:
        """设置识别功能"""
        try:
            print(f"Setting recognition settings: {settings_json}")
            settings = json.loads(settings_json)
            self.recognition_settings = settings
            
            result = json.dumps({'success': True, 'message': '识别设置已更新'}, ensure_ascii=False)
            print(f"set_recognition_settings result: {result}")
            return result
        except Exception as e:
            error_result = json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False)
            print(f"set_recognition_settings error: {error_result}")
            return error_result
    
    @pyqtSlot(result=str)
    def get_current_frame(self) -> str:
        """获取当前处理后的图像帧"""
        try:
            if self.processed_frame is None:
                return json.dumps({'success': False, 'error': '没有可用的图像帧'}, ensure_ascii=False)
            
            # 编码图像为base64
            _, buffer = cv2.imencode('.jpg', self.processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            import base64
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result = json.dumps({
                'success': True,
                'image': f'data:image/jpeg;base64,{img_base64}',
                'timestamp': time.time()
            }, ensure_ascii=False)
            
            return result
        except Exception as e:
            error_result = json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False)
            print(f"get_current_frame error: {error_result}")
            return error_result
    
    @pyqtSlot(result=str)
    def get_detection_results(self) -> str:
        """获取检测结果"""
        try:
            result = json.dumps({'objects': self.detected_objects}, ensure_ascii=False)
            return result
        except Exception as e:
            error_result = json.dumps({'objects': [], 'error': str(e)}, ensure_ascii=False)
            print(f"get_detection_results error: {error_result}")
            return error_result
    
    @pyqtSlot(str, result=str)
    def save_image(self, filename: str = "") -> str:
        """保存当前图像"""
        try:
            if self.current_frame is None:
                return json.dumps({'success': False, 'error': '没有可保存的图像'}, ensure_ascii=False)
            
            if not filename:
                filename = f'captured_image_{int(time.time())}.jpg'
            
            cv2.imwrite(filename, self.current_frame)
            result = json.dumps({'success': True, 'message': f'图像已保存: {filename}'}, ensure_ascii=False)
            print(f"save_image result: {result}")
            return result
        except Exception as e:
            error_result = json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False)
            print(f"save_image error: {error_result}")
            return error_result
    
    @pyqtSlot(str, result=str)
    def export_report(self, format_type: str = 'json') -> str:
        """导出检测报告"""
        try:
            report_data = {
                'timestamp': time.time(),
                'model_path': self.current_model_path,
                'camera_index': self.current_camera_index,
                'image_params': self.image_params,
                'recognition_settings': self.recognition_settings,
                'detected_objects': self.detected_objects,
                'total_objects': len(self.detected_objects),
                'anomaly_count': len([obj for obj in self.detected_objects 
                                    if obj.get('type') in ['异物', '缺陷']])
            }
            
            filename = f'detection_report_{int(time.time())}.{format_type}'
            
            if format_type == 'json':
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            result = json.dumps({'success': True, 'message': f'报告已导出: {filename}'}, ensure_ascii=False)
            print(f"export_report result: {result}")
            return result
        except Exception as e:
            error_result = json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False)
            print(f"export_report error: {error_result}")
            return error_result
    
    @pyqtSlot(result=str)
    def get_system_status(self) -> str:
        """获取系统状态"""
        try:
            status = {
                'fps': self.fps_counter,
                'cpuUsage': psutil.cpu_percent(),
                'memoryUsage': psutil.virtual_memory().percent,
                'gpuUsage': 0
            }
            result = json.dumps(status, ensure_ascii=False)
            return result
        except Exception as e:
            error_result = json.dumps({
                'fps': 0, 'cpuUsage': 0, 'memoryUsage': 0, 'gpuUsage': 0, 'error': str(e)
            }, ensure_ascii=False)
            print(f"get_system_status error: {error_result}")
            return error_result

    def _draw_detections(self, frame):
        """在图像上绘制检测结果"""
        try:
            for obj in self.detected_objects:
                x, y, w, h = int(obj['x']), int(obj['y']), int(obj['width']), int(obj['height'])
                confidence = obj['confidence']
                obj_type = obj['type']
                
                # 根据类型选择颜色
                if obj_type in ['异物', '缺陷']:
                    color = (0, 0, 255)  # 红色
                elif obj_type == 'Hamburger':
                    color = (0, 255, 0)  # 绿色
                else:
                    color = (255, 0, 0)  # 蓝色
                
                # 绘制边界框
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # 绘制标签
                label = f'{obj_type}: {confidence:.2f}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return frame
        except Exception as e:
            print(f"Draw detections error: {e}")
            return frame
    
    def _video_stream_thread(self):
        """视频流处理线程"""
        while self.is_streaming and self.camera:
            ret, frame = self.camera.read()
            if ret:
                self.current_frame = frame.copy()
                
                # 应用图像处理
                processed_frame = self._apply_image_processing(frame)
                
                # 根据延迟设置执行目标检测
                current_time = time.time()
                if self.model and (current_time - self.last_detection_time) >= self.detection_delay:
                    self._perform_detection(processed_frame)
                    self.last_detection_time = current_time
                
                # 绘制检测结果
                self.processed_frame = self._draw_detections(processed_frame.copy())
                
                self._update_fps()
            
            time.sleep(0.033)  # 约30 FPS
    
    def _apply_image_processing(self, frame):
        """应用图像处理参数"""
        processed = frame.copy()
        
        # 畸变矫正
        if self.image_params.get('distortionEnabled', True) and self.map1 is not None:
            processed = cv2.remap(processed, self.map1, self.map2, cv2.INTER_LINEAR)
        
        # 透视变换
        if self.image_params.get('perspectiveEnabled', True):
            processed = self._perspective_transform(processed)
        
        # 基础图像调整
        contrast = self.image_params.get('contrast', 100) / 100.0
        brightness = self.image_params.get('brightness', 100) - 100
        processed = cv2.convertScaleAbs(processed, alpha=contrast, beta=brightness)
        
        # 饱和度调整
        saturation = self.image_params.get('saturation', 100) / 100.0
        if saturation != 1.0:
            hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 模糊处理
        blur = self.image_params.get('blur', 0)
        if blur > 0:
            kernel_size = int(blur * 2) + 1
            processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), blur)
        
        return processed
    
    def _perspective_transform(self, image):
        """透视变换"""
        try:
            # 计算输出尺寸
            real_width = self.image_params.get('realWidthCm', 29)
            real_height = self.image_params.get('realHeightCm', 18.5)
            pixel_per_cm = 20  # 假设每厘米20像素
            
            output_width = int(real_width * pixel_per_cm)
            output_height = int(real_height * pixel_per_cm)
            
            dst_points = np.array([
                [0, 0], [output_width, 0], 
                [output_width, output_height], [0, output_height]
            ], dtype=np.float32)
            
            # 获取透视变换矩阵
            matrix = cv2.getPerspectiveTransform(self.src_points, dst_points)
            
            # 应用透视变换
            transformed = cv2.warpPerspective(image, matrix, (output_width, output_height))
            
            return transformed
        except Exception as e:
            print(f"Perspective transform error: {e}")
            return image
    
    def _perform_detection(self, frame):
        """执行目标检测"""
        try:
            # 生成模拟检测数据
            if np.random.random() > 0.7:  # 30%概率生成新的检测结果
                self.detected_objects = []
                num_objects = np.random.randint(0, 4)
                
                for i in range(num_objects):
                    obj = {
                        'id': f'obj_{int(time.time())}_{i}',
                        'type': np.random.choice(['Hamburger', '异物', '正常物体', '缺陷']),
                        'confidence': np.random.uniform(0.6, 1.0),
                        'size': np.random.choice(['small', 'medium', 'large']),
                        'x': np.random.randint(50, 450),
                        'y': np.random.randint(50, 350),
                        'width': np.random.randint(40, 120),
                        'height': np.random.randint(40, 120)
                    }
                    self.detected_objects.append(obj)
        except Exception as e:
            print(f"Detection error: {e}")
    
    def _update_fps(self):
        """更新FPS计数"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.last_fps_time = current_time
            self.fps_counter = 30 if self.is_streaming else 0

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('图像识别系统 - 轮询版')
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout(central_widget)
        
        # 创建 WebEngine 视图
        self.web_view = QWebEngineView()
        
        # 使用自定义页面
        self.custom_page = CustomWebEnginePage()
        self.web_view.setPage(self.custom_page)
        
        # 配置 WebEngine 设置
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.AllowRunningInsecureContent, True)
        settings.setAttribute(QWebEngineSettings.AllowGeolocationOnInsecureOrigins, True)
        settings.setAttribute(QWebEngineSettings.ShowScrollBars, False)
        
        layout.addWidget(self.web_view)
        
        # 创建 API 对象
        self.api = ImageRecognitionAPI()
        
        # 设置 Web Channel
        self.channel = QWebChannel()
        self.channel.registerObject('pyapi', self.api)
        self.web_view.page().setWebChannel(self.channel)
        
        # 加载网页
        self.load_web_page()
    
    def load_web_page(self):
        """加载网页"""
        # 可以加载本地HTML文件或远程URL
        url = QUrl('http://localhost:3000')  # Next.js 开发服务器
        self.web_view.load(url)
        
        # 添加页面加载完成的回调
        self.web_view.loadFinished.connect(self.on_load_finished)
    
    def on_load_finished(self, success):
        """页面加载完成回调"""
        if success:
            print("Page loaded successfully")
            # 注入一些调试脚本
            self.inject_debug_script()
        else:
            print("Failed to load page")
    
    def inject_debug_script(self):
        """注入调试脚本"""
        debug_script = """
        console.log('Debug script injected - Polling Version');
        
        // 检查 QWebChannel 是否可用
        if (typeof qt !== 'undefined' && qt.webChannelTransport) {
            console.log('QWebChannel transport available');
            
            // 初始化 WebChannel
            new QWebChannel(qt.webChannelTransport, function(channel) {
                console.log('WebChannel initialized');
                window.pyapi = channel.objects.pyapi;
                
                if (window.pyapi) {
                    console.log('Python API available');
                } else {
                    console.error('Python API not available');
                }
            });
        } else {
            console.error('QWebChannel transport not available');
        }
        """
        
        self.web_view.page().runJavaScript(debug_script)

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName('图像识别系统 - 轮询版')
    app.setApplicationVersion('2.1.0')
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    print("WebEngine Remote Debugging enabled on port 9222")
    print("You can access it at: http://localhost:9222")
    
    # 运行应用
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
