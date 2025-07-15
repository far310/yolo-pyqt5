"""
Python QWebEngineView 后端代码 - 轮询版本 + YOLO检测
支持直接输入摄像头索引和模型路径，集成TensorFlow Lite YOLO检测
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

# YOLO检测相关导入
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

class CustomWebEnginePage(QWebEnginePage):
    """自定义 WebEngine 页面，修复交互问题"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        """捕获 JavaScript 控制台消息"""
        print(f"JS Console [{level}]: {message} (Line: {lineNumber}, Source: {sourceID})")

class YOLODetector:
    """YOLO检测器类"""
    
    def __init__(self, model_path: str = None):
        self.interpreter = None
        self.model_path = model_path
        self.input_details = None
        self.output_details = None
        
        # YOLO类别
        self.classes = [
            "Hamburger", "Shreddedchicken", "Burrito", "Slicedroastbeef", "Shreddedpork",
            "Fajitasteak", "Fajitachicken", "Cheeseburger", "Doublehamburger", 
            "DoubleCheeseburger", "Chickenbreast bites", "Shrimpscampi", "Broccoliflorets",
            "ChoppedFajitaChicken", "ChoppedFajitaSteak"
        ]
        
        # 检测参数
        self.obj_thresh = 0.1
        self.nms_thresh = 0.1
        self.score_thresh = 0.0
        
        # 汉堡尺寸分类
        self.hamburger_size_range = (10, 13)
        
        # 透视变换参数
        self.src_points = np.array([[650, 330], [1425, 330], [1830, 889], [230, 889]], dtype=np.float32)
        self.real_width_cm = 29.0
        self.real_height_cm = 18.5
        self.pixel_per_cm = self.compute_pixel_per_cm()
        
        # 计算输出尺寸
        self.output_width = int(self.real_width_cm * self.pixel_per_cm)
        self.output_height = int(self.real_height_cm * self.pixel_per_cm)
        self.dst_points = np.array([
            [0, 0], [self.output_width, 0], 
            [self.output_width, self.output_height], [0, self.output_height]
        ], dtype=np.float32)
        
        # 相机参数
        self.camera_matrix = np.array([
            [1298.54926, 0.0, 966.93144],
            [0.0, 1294.39363, 466.380271],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([
            [-0.44903195, 0.25133919, 0.00037556, 0.00024487, -0.0794278]
        ])
        
        # 初始化畸变矫正映射
        self.map1 = None
        self.map2 = None
        self.init_undistort_maps()
        
        # 背景图像
        self.background_image = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def compute_pixel_per_cm(self):
        """计算每厘米像素数"""
        ref_width_px = np.linalg.norm(np.array(self.src_points[1]) - np.array(self.src_points[0]))
        ref_height_px = np.linalg.norm(np.array(self.src_points[3]) - np.array(self.src_points[0]))
        
        pixel_per_cm_w = ref_width_px / self.real_width_cm
        pixel_per_cm_h = ref_height_px / self.real_height_cm
        
        return (pixel_per_cm_w + pixel_per_cm_h) / 2.0
    
    def init_undistort_maps(self):
        """初始化畸变矫正映射"""
        image_size = (1920, 1080)
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, image_size, 1.0
        )
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, 
            newcameramtx, image_size, cv2.CV_16SC2
        )
    
    def load_model(self, model_path: str):
        """加载TensorFlow Lite模型"""
        try:
            self.interpreter = Interpreter(model_path=model_path, num_threads=4)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model_path = model_path
            print(f"YOLO模型加载成功: {model_path}")
            return True
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")
            return False
    
    def preprocess_image(self, image):
        """图像预处理"""
        if self.interpreter is None:
            return None
            
        # 畸变矫正
        if self.map1 is not None:
            image = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)
        
        # 透视变换
        image = cv2.warpPerspective(
            image, 
            cv2.getPerspectiveTransform(self.src_points, self.dst_points),
            (self.output_width, self.output_height)
        )
        
        # 获取输入尺寸
        input_shape = self.input_details[0]['shape']
        input_height, input_width = input_shape[1], input_shape[2]
        
        # LetterBox缩放
        processed_image = self.letterbox_resize(image, (input_width, input_height))
        
        # 归一化
        processed_image = processed_image.astype(np.float32) / 255.0
        processed_image = np.expand_dims(processed_image, axis=0)
        
        return processed_image, image
    
    def letterbox_resize(self, image, target_size):
        """LetterBox缩放"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建目标尺寸的图像并居中放置
        result = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return result
    
    def postprocess_detections(self, output, original_image_shape):
        """后处理检测结果"""
        if len(output.shape) == 3:
            output = output[0]
        
        output = output.T
        boxes = output[..., :4]
        scores = np.max(output[..., 4:], axis=1)
        class_ids = np.argmax(output[..., 4:], axis=1)
        
        # 应用NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), 
            self.obj_thresh, self.nms_thresh
        )
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                score = scores[i]
                class_id = class_ids[i]
                
                if score >= self.score_thresh:
                    # 转换坐标格式
                    x, y, w, h = box
                    detection = {
                        'id': f'obj_{int(time.time())}_{i}',
                        'type': self.classes[class_id] if class_id < len(self.classes) else 'Unknown',
                        'confidence': float(score),
                        'x': float(x),
                        'y': float(y),
                        'width': float(w),
                        'height': float(h),
                        'class_id': int(class_id)
                    }
                    
                    # 如果是汉堡，计算实际尺寸
                    if self.classes[class_id] == 'Hamburger':
                        actual_w = w / self.pixel_per_cm
                        actual_h = h / self.pixel_per_cm
                        
                        # 高度补偿
                        target_height = 2.7  # 汉堡高度
                        camera_height = 13.0
                        actual_w = actual_w / (1 + (target_height / camera_height))
                        
                        # 尺寸分类
                        min_size, max_size = self.hamburger_size_range
                        if actual_w <= min_size:
                            size_category = 'small'
                        elif actual_w < max_size:
                            size_category = 'medium'
                        else:
                            size_category = 'large'
                        
                        detection['size'] = size_category
                        detection['actual_width_cm'] = actual_w
                        detection['actual_height_cm'] = actual_h
                    
                    detections.append(detection)
        
        return detections
    
    def detect_background_change(self, current_image, detected_boxes):
        """检测背景变化（异物检测）"""
        if self.background_image is None:
            return False, 0.0
        
        try:
            # 创建遮罩排除检测框区域
            mask = np.ones(current_image.shape[:2], dtype=np.uint8) * 255
            
            for detection in detected_boxes:
                x, y, w, h = int(detection['x']), int(detection['y']), int(detection['width']), int(detection['height'])
                x1 = max(0, x - 10)
                y1 = max(0, y - 10)
                x2 = min(current_image.shape[1], x + w + 10)
                y2 = min(current_image.shape[0], y + h + 10)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
            
            # 灰度转换
            gray1 = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(self.background_image, cv2.COLOR_BGR2GRAY)
            
            # 应用遮罩
            gray1 = cv2.bitwise_and(gray1, gray1, mask=mask)
            gray2 = cv2.bitwise_and(gray2, gray2, mask=mask)
            
            # 缩放
            gray1 = cv2.resize(gray1, (120, 120))
            gray2 = cv2.resize(gray2, (120, 120))
            
            # 计算余弦相似度
            vec1 = gray1.flatten().reshape(1, -1)
            vec2 = gray2.flatten().reshape(1, -1)
            similarity = cosine_similarity(vec1, vec2)[0][0]
            
            is_different = similarity < 0.98
            return is_different, similarity
            
        except Exception as e:
            print(f"背景变化检测错误: {e}")
            return False, 0.0
    
    def detect_objects(self, image):
        """执行目标检测"""
        if self.interpreter is None:
            return []
        
        try:
            # 预处理
            processed_image, perspective_image = self.preprocess_image(image)
            if processed_image is None:
                return []
            
            # 量化处理（如果需要）
            input_scale, input_zero_point = self.input_details[0]["quantization"]
            if input_scale != 0:  # 量化模型
                processed_image = (processed_image / input_scale + input_zero_point).astype(np.int8)
            
            # 推理
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
            self.interpreter.invoke()
            
            # 获取输出
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # 反量化（如果需要）
            output_scale, output_zero_point = self.output_details[0]["quantization"]
            if output_scale != 0:  # 量化模型
                output = (output.astype(np.float32) - output_zero_point) * output_scale
            
            # 后处理
            detections = self.postprocess_detections(output, perspective_image.shape)
            
            # 检测背景变化
            if detections:
                is_different, similarity = self.detect_background_change(perspective_image, detections)
                if is_different:
                    # 添加异物检测结果
                    detections.append({
                        'id': f'foreign_matter_{int(time.time())}',
                        'type': '异物',
                        'confidence': 1.0 - similarity,
                        'x': 0,
                        'y': 0,
                        'width': perspective_image.shape[1],
                        'height': perspective_image.shape[0],
                        'similarity': similarity
                    })
            
            return detections
            
        except Exception as e:
            print(f"目标检测错误: {e}")
            return []
    
    def set_background_image(self, image):
        """设置背景图像"""
        if self.map1 is not None:
            image = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)
        
        self.background_image = cv2.warpPerspective(
            image,
            cv2.getPerspectiveTransform(self.src_points, self.dst_points),
            (self.output_width, self.output_height)
        )

class ImageRecognitionAPI(QObject):
    """图像识别 API 类 - 轮询版本 + YOLO检测"""
    
    def __init__(self):
        super().__init__()
        self.camera = None
        self.current_frame = None
        self.processed_frame = None
        self.is_streaming = False  # 摄像头流状态
        self.current_model_path = ""
        self.current_camera_index = "0"
        
        # YOLO检测器
        self.yolo_detector = YOLODetector()
        
        # 图像参数
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
        
        # 视频流线程控制
        self.video_thread = None
        self.video_thread_running = False
    
    @pyqtSlot(str, result=str)
    def start_camera(self, camera_index: str) -> str:
        """启动摄像头"""
        try:
            print(f"Starting camera with index: {camera_index}")
            
            try:
                camera_idx = int(camera_index)
            except ValueError:
                return json.dumps({'success': False, 'error': '无效的摄像头索引'}, ensure_ascii=False)
            
            if camera_idx < 0 or camera_idx > 10:
                return json.dumps({'success': False, 'error': '摄像头索引必须在0-10之间'}, ensure_ascii=False)
            
            # 停止现有的摄像头
            if self.camera:
                self.stop_camera()
            
            self.camera = cv2.VideoCapture(camera_idx)
            if not self.camera.isOpened():
                return json.dumps({'success': False, 'error': f'无法打开摄像头 {camera_idx}'}, ensure_ascii=False)
            
            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            self.current_camera_index = camera_index
            self.is_streaming = True
            
            # 设置背景图像（第一帧）
            ret, frame = self.camera.read()
            if ret:
                self.yolo_detector.set_background_image(frame)
            
            # 启动视频流线程
            self.video_thread_running = True
            self.video_thread = threading.Thread(target=self._video_stream_thread, daemon=True)
            self.video_thread.start()
            
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
            
            # 停止视频流线程
            self.video_thread_running = False
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=2.0)  # 等待线程结束，最多2秒
            
            self.is_streaming = False
            if self.camera:
                self.camera.release()
                self.camera = None
            
            # 清空帧数据
            self.current_frame = None
            self.processed_frame = None
            self.detected_objects = []
            
            result = json.dumps({'success': True, 'message': '摄像头已停止'}, ensure_ascii=False)
            print(f"stop_camera result: {result}")
            return result
        except Exception as e:
            error_result = json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False)
            print(f"stop_camera error: {error_result}")
            return error_result
    
    @pyqtSlot(str, result=str)
    def load_model(self, model_path: str) -> str:
        """加载AI模型"""
        try:
            print(f"Loading model: {model_path}")
            
            if not os.path.exists(model_path):
                result = json.dumps({
                    'success': True, 
                    'message': f'模型加载成功 (模拟): {model_path}'
                }, ensure_ascii=False)
                print(f"load_model result (mock): {result}")
                self.current_model_path = model_path
                return result
            
            # 加载YOLO模型
            success = self.yolo_detector.load_model(model_path)
            if success:
                self.current_model_path = model_path
                result = json.dumps({'success': True, 'message': f'YOLO模型加载成功: {model_path}'}, ensure_ascii=False)
            else:
                result = json.dumps({'success': False, 'error': f'YOLO模型加载失败: {model_path}'}, ensure_ascii=False)
            
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
            
            # 更新YOLO检测器参数
            if 'objThresh' in params:
                self.yolo_detector.obj_thresh = params['objThresh'] / 100.0
            if 'nmsThresh' in params:
                self.yolo_detector.nms_thresh = params['nmsThresh'] / 100.0
            if 'scoreThresh' in params:
                self.yolo_detector.score_thresh = params['scoreThresh'] / 100.0
            
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
        """获取当前处理后的图像帧 - 只有在摄像头启动时才返回帧"""
        try:
            # 检查摄像头是否正在运行
            if not self.is_streaming or self.processed_frame is None:
                return json.dumps({'success': False, 'error': '摄像头未启动或没有可用的图像帧'}, ensure_ascii=False)
            
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
        """获取检测结果 - 只有在摄像头启动时才返回结果"""
        try:
            if not self.is_streaming:
                return json.dumps({'objects': []}, ensure_ascii=False)
            
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
            if not self.is_streaming or self.current_frame is None:
                return json.dumps({'success': False, 'error': '摄像头未启动或没有可保存的图像'}, ensure_ascii=False)
            
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
                
                # 创建标签
                label = f'{obj_type}: {confidence:.2f}'
                if 'size' in obj:
                    label += f' ({obj["size"]})'
                if 'actual_width_cm' in obj:
                    label += f' {obj["actual_width_cm"]:.1f}cm'
                
                # 绘制标签背景和文字
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return frame
        except Exception as e:
            print(f"Draw detections error: {e}")
            return frame
    
    def _video_stream_thread(self):
        """视频流处理线程 - 只有在摄像头启动时才运行"""
        print("Video stream thread started")
        
        while self.video_thread_running and self.is_streaming and self.camera:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                self.current_frame = frame.copy()
                
                # 根据延迟设置执行目标检测
                current_time = time.time()
                if (current_time - self.last_detection_time) >= self.detection_delay:
                    self.detected_objects = self.yolo_detector.detect_objects(frame)
                    self.last_detection_time = current_time
                
                # 应用图像处理并绘制检测结果
                processed_frame = self._apply_image_processing(frame)
                self.processed_frame = self._draw_detections(processed_frame.copy())
                
                self._update_fps()
                
                time.sleep(0.033)  # 约30 FPS
                
            except Exception as e:
                print(f"Video stream thread error: {e}")
                break
        
        print("Video stream thread stopped")
    
    def _apply_image_processing(self, frame):
        """应用图像处理参数"""
        processed = frame.copy()
        
        # 畸变矫正
        if self.image_params.get('distortionEnabled', True) and self.yolo_detector.map1 is not None:
            processed = cv2.remap(processed, self.yolo_detector.map1, self.yolo_detector.map2, cv2.INTER_LINEAR)
        
        # 透视变换
        if self.image_params.get('perspectiveEnabled', True):
            processed = cv2.warpPerspective(
                processed,
                cv2.getPerspectiveTransform(self.yolo_detector.src_points, self.yolo_detector.dst_points),
                (self.yolo_detector.output_width, self.yolo_detector.output_height)
            )
        
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
        self.setWindowTitle('图像识别系统 - YOLO检测版')
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
        url = QUrl('http://localhost:3000')  # Next.js 开发服务器
        self.web_view.load(url)
        
        # 添加页面加载完成的回调
        self.web_view.loadFinished.connect(self.on_load_finished)
    
    def on_load_finished(self, success):
        """页面加载完成回调"""
        if success:
            print("Page loaded successfully")
            self.inject_debug_script()
        else:
            print("Failed to load page")
    
    def inject_debug_script(self):
        """注入调试脚本"""
        debug_script = """
        console.log('Debug script injected - YOLO Detection Version');
        
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
    app.setApplicationName('图像识别系统 - YOLO检测版')
    app.setApplicationVersion('3.0.0')
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    print("WebEngine Remote Debugging enabled on port 9222")
    print("You can access it at: http://localhost:9222")
    print("YOLO Detection System Ready!")
    
    # 运行应用
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
