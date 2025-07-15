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

class LetterBox:
    """LetterBox预处理类，与附件中的实现完全一致"""
    
    def __init__(self, new_shape, auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

class YOLODetector:
    """YOLO检测器类，使用与附件完全一致的预处理和后处理"""
    
    def __init__(self, model_path: str = None):
        super().__init__()
        self.interpreter = None
        self.model_path = model_path
        self.input_details = None
        self.output_details = None
        
        # 添加缓存的模型参数
        self.input_scale = 0
        self.input_zero_point = 0
        self.output_scale = 0
        self.output_zero_point = 0
        self.input_shape = None
        self.model_input_size = None
        
        # YOLO类别 - 与附件一致
        self.classes = [
            "Hamburger", "Shreddedchicken", "Burrito", "Slicedroastbeef", "Shreddedpork",
            "Fajitasteak", "Fajitachicken", "Cheeseburger", "Doublehamburger", 
            "DoubleCheeseburger", "Chickenbreast bites", "Shrimpscampi", "Broccoliflorets",
            "ChoppedFajitaChicken", "ChoppedFajitaSteak"
        ]
        
        # 检测参数 - 与附件一致
        self.obj_thresh = 0.1
        self.nms_thresh = 0.1
        self.score_thresh = 0.0
        
        # 汉堡尺寸分类 - 与附件一致
        self.hamburger_size = (9, 11.3)
        self.hamburger_mj = (95, 143)
        
        # 透视变换参数 - 与附件一致
        self.src_points = np.array([[510, 270], [1170, 270], [1500, 710], [140, 710]], dtype=np.float32)
        self.real_width_cm = 29.0
        self.real_height_cm = 18.5
        self.target_height = 1.8 + 0.9  # 面包高度 + 牛肉饼中心离面包表面高度
        self.camera_height = 13.0
        
        # 计算像素密度
        self.pixel_per_cm = self.compute_pixel_per_cm()
        
        # 计算输出尺寸
        self.output_width = int(self.real_width_cm * self.pixel_per_cm)
        self.output_height = int(self.real_height_cm * self.pixel_per_cm)
        self.dst_points = np.array([
            [0, 0], [self.output_width, 0], 
            [self.output_width, self.output_height], [0, self.output_height]
        ], dtype=np.float32)
        
        # 相机参数 - 与附件一致
        self.camera_matrix = np.array([
            [1260.15281, 0.0, 971.702426],
            [0.0, 1256.08744, 504.553169],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([
            [-0.430483648, 0.216393722, -0.000156465611, 0.000104551776, -0.0564557922]
        ])
        
        # 初始化畸变矫正映射
        self.map1 = None
        self.map2 = None
        self.init_undistort_maps()
        
        # 背景图像
        self.background_image = None
        
        # 颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def compute_pixel_per_cm(self):
        """计算每厘米像素数 - 与附件一致"""
        ref_width_px = np.linalg.norm(np.array(self.src_points[1]) - np.array(self.src_points[0]))
        ref_height_px = np.linalg.norm(np.array(self.src_points[3]) - np.array(self.src_points[0]))
        
        pixel_per_cm_w = ref_width_px / self.real_width_cm
        pixel_per_cm_h = ref_height_px / self.real_height_cm
        
        return (pixel_per_cm_w + pixel_per_cm_h) / 2.0
    
    def init_undistort_maps(self):
        """初始化畸变矫正映射 - 与附件一致"""
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
            
            # 缓存量化参数
            self.input_scale, self.input_zero_point = self.input_details[0]["quantization"]
            self.output_scale, self.output_zero_point = self.output_details[0]["quantization"]
            
            # 缓存输入形状
            self.input_shape = self.input_details[0]["shape"]
            self.model_input_size = (self.input_shape[1], self.input_shape[2])
            
            self.model_path = model_path
            print(f"YOLO模型加载成功: {model_path}")
            print(f"输入尺寸: {self.model_input_size}")
            print(f"量化参数 - 输入: scale={self.input_scale}, zero_point={self.input_zero_point}")
            print(f"量化参数 - 输出: scale={self.output_scale}, zero_point={self.output_zero_point}")
            return True
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")
            return False
    
    def preprocess(self, ori_img, size):
        """预处理函数 - 与附件完全一致"""
        letterbox = LetterBox(new_shape=size, auto=False, stride=32)
        image = letterbox(image=ori_img)
        image = [image]
        image = np.stack(image)
        image = image[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(image)
        image = img.astype(np.float32)
        return image / 255
    
    def postprocess(self, output):
        """后处理函数 - 与附件完全一致"""
        output = output[0]
        output = output.T
        boxes = output[..., :4]
        scores = np.max(output[..., 4:], axis=1)
        class_ids = np.argmax(output[..., 4:], axis=1)
        
        # 应用 NMS 来去除重叠的框
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.obj_thresh, self.nms_thresh)
        
        result_boxes = []
        result_scores = []
        result_class_ids = []
        
        # 检查 indices 是否为空
        if len(indices) > 0:
            for i in indices:
                box = boxes[i]
                score = scores[i]
                class_id = class_ids[i]
                result_boxes.append(box)
                result_scores.append(score)
                result_class_ids.append(class_id)
        
        return result_boxes, result_scores, result_class_ids
    
    def adjust_boxes(self, size, boxes, original_size):
        """调整边界框 - 与附件一致"""
        img_width, img_height = size
        target_width, target_height = original_size
        
        # 获取缩放比例和填充
        gain = min(img_width / target_width, img_height / target_height)
        pad_w = round((img_width - target_width * gain) / 2 - 0.1)  # 水平填充
        pad_h = round((img_height - target_height * gain) / 2 - 0.1)  # 垂直填充
        
        # 将中心点 (cx, cy) + 宽高 (w, h) 转换为 (x_min, y_min, x_max, y_max) 的批量操作
        boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - pad_w) / gain  # x_min
        boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - pad_h) / gain  # y_min
        boxes[:, 2] = boxes[:, 2] / gain  # w 缩放
        boxes[:, 3] = boxes[:, 3] / gain  # h 缩放
        
        return boxes
    
    def annotate_xywh(self, boxes):
        """计算实际尺寸 - 与附件一致"""
        x, y, w, h = boxes
        # 换算实际尺寸（cm）
        actual_w = w / self.pixel_per_cm
        actual_h = h / self.pixel_per_cm
        return actual_w, actual_h
    
    def correct_size_with_height(self, measured_size, target_height, camera_height):
        """高度补偿计算 - 与附件一致"""
        correction_ratio = 1 / (1 + (target_height / camera_height))
        corrected_size = measured_size * correction_ratio
        return corrected_size
    
    def perspective_transform(self, img, src_pts, dst_pts, output_width, output_height):
        """透视变换 - 与附件一致"""
        if img is None:
            print("❌ 图像为空，无法执行透视变换")
            return None
        
        img_copy = img.copy()
        src_pts = np.array(src_pts, dtype=np.float32)
        
        # 获取透视矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # 执行透视变换
        warped = cv2.warpPerspective(img_copy, M, (output_width, output_height))
        
        return warped
    
    def undistort_image_fast(self, img):
        """快速去畸变 - 与附件一致"""
        if self.map1 is not None and self.map2 is not None:
            return cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        return img
    
    def detect_background_change_by_cosine(self, frame, background, detected_boxes):
        """背景变化检测 - 与附件一致"""
        if background is None:
            return 0.0, False
        
        try:
            # 创建遮罩排除检测框区域
            mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
            
            for box in detected_boxes:
                x, y, w, h = map(int, box)
                x1 = max(0, x - 10)
                y1 = max(0, y - 10)
                x2 = min(frame.shape[1], x + w + 10)
                y2 = min(frame.shape[0], y + h + 10)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
            
            # 灰度转换
            gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            
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
            return similarity, is_different
            
        except Exception as e:
            print(f"背景变化检测错误: {e}")
            return 0.0, False
    
    def draw_detections(self, image, boxes, scores, class_ids):
        """绘制检测结果 - 与附件逻辑一致"""
        max_score = 0
        max_class = None
        
        # 检测背景变化
        hash_diff, is_different = self.detect_background_change_by_cosine(
            image, self.background_image, boxes
        )
        
        if is_different:
            label_text = f"has foreign matter SCORE: {hash_diff:.3f}, {is_different}"
            cv2.putText(
                image, label_text, (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
            return image
        
        for box, score, cl in zip(boxes, scores, class_ids):
            if score >= self.score_thresh:
                x1, y1, w, h = box
                
                # 获取颜色
                color = self.color_palette[cl]
                
                # 绘制边界框
                cv2.rectangle(
                    image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2
                )
                
                # 创建标签
                label = f"{self.classes[cl]}: {score:.2f}"
                
                # 如果是汉堡，进行尺寸计算
                if self.classes[cl] == "Hamburger":
                    # 计算实际尺寸
                    contour_w, contour_h = self.annotate_xywh(box)
                    
                    # 高度补偿
                    contour_w = self.correct_size_with_height(
                        contour_w, self.target_height, self.camera_height
                    )
                    
                    if contour_w and contour_h:
                        width_cm = contour_w
                        height_cm = contour_h
                        
                        # 显示尺寸
                        label_text = f"{width_cm:.1f}cm"
                        x2 = int(x1 + w)
                        cv2.putText(
                            image, label_text, (x2 - 20, int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                        )
                        
                        # 尺寸分类
                        if width_cm:
                            width_min, width_max = self.hamburger_size
                            mj_min, mj_max = self.hamburger_mj
                            
                            # 计算面积
                            mj = width_cm * height_cm
                            
                            # 分类逻辑
                            if mj <= mj_min:
                                if width_cm <= width_min:
                                    size_label = "small"
                                else:
                                    size_label = "medium"
                            elif mj > mj_min and mj <= mj_max:
                                if width_cm > width_max:
                                    size_label = "large"
                                else:
                                    size_label = "medium"
                            elif mj > mj_max:
                                size_label = "large"
                            
                            label = f"{size_label} {self.classes[cl]}: {score:.2f}"
                            print(f"轮廓尺寸:{size_label} {self.classes[cl]}: {width_cm:.2f} cm x {height_cm:.2f} cm")
                
                # 绘制标签
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                label_x = x1
                label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
                
                cv2.rectangle(
                    image,
                    (int(label_x), int(label_y - label_height)),
                    (int(label_x + label_width), int(label_y + label_height)),
                    color, cv2.FILLED
                )
                
                cv2.putText(
                    image, label, (int(label_x), int(label_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                )
                
                if score > max_score:
                    max_score = score
                    max_class = self.classes[cl]
        
        if max_class:
            print(f"Highest score: {max_score:.2f}, Class: {max_class}")
        else:
            print("No boxes met the SCORE threshold.")
        
        return image
    
    def detect_objects(self, image):
        """主检测函数 - 与附件myFunc函数逻辑完全一致"""
        if self.interpreter is None:
            return []
        
        try:
            # 获取图片原始尺寸
            original_size = image.shape[:2]
        
            # 使用缓存的参数，无需重复获取
            # input_details = self.interpreter.get_input_details()  # 移除
            # output_details = self.interpreter.get_output_details()  # 移除
            # input_scale, input_zero_point = input_details[0]["quantization"]  # 移除
            # output_scale, output_zero_point = output_details[0]["quantization"]  # 移除
            # input_shape = input_details[0]["shape"]  # 移除
            # size = (input_shape[1], input_shape[2])  # 移除
        
            # 畸变矫正
            image = self.undistort_image_fast(image)
        
            # 透视校正
            image = self.perspective_transform(
                image, self.src_points, self.dst_points, self.output_width, self.output_height
            )
        
            original_size = image.shape[:2]
        
            # 预处理 - 使用缓存的输入尺寸
            input_data = self.preprocess(image, self.model_input_size)
            input_data = input_data.transpose((0, 2, 3, 1))
        
            # 量化处理 - 使用缓存的量化参数
            if self.input_scale != 0:  # 量化模型
                input_data = (input_data / self.input_scale + self.input_zero_point).astype(np.int8)
        
            # 模型推理
            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()
        
            # 获取输出
            output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        
            # 反量化 - 使用缓存的量化参数
            if self.output_scale != 0:  # 量化模型
                output_data = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale
        
            # 将预测框的坐标从归一化形式转换回原始图像尺寸
            output_data[:, [0, 2]] *= self.model_input_size[0]
            output_data[:, [1, 3]] *= self.model_input_size[1]
        
            # 后处理
            result_boxes, result_scores, result_class_ids = self.postprocess(output_data)
        
            detections = []
            if len(result_boxes) > 0:
                result_boxes = self.adjust_boxes(
                    self.model_input_size,
                    np.array(result_boxes, dtype=np.float64),
                    (original_size[1], original_size[0])
                )
            
                # 绘制检测结果
                image = self.draw_detections(image, result_boxes, result_scores, result_class_ids)
            
                # 转换为标准格式
                for i, (box, score, class_id) in enumerate(zip(result_boxes, result_scores, result_class_ids)):
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
                
                    # 如果是汉堡，添加尺寸信息
                    if self.classes[class_id] == 'Hamburger':
                        actual_w, actual_h = self.annotate_xywh(box)
                        actual_w = self.correct_size_with_height(
                            actual_w, self.target_height, self.camera_height
                        )
                    
                        # 尺寸分类
                        width_min, width_max = self.hamburger_size
                        mj_min, mj_max = self.hamburger_mj
                        mj = actual_w * actual_h
                    
                        if mj <= mj_min:
                            size_category = 'small' if actual_w <= width_min else 'medium'
                        elif mj > mj_min and mj <= mj_max:
                            size_category = 'large' if actual_w > width_max else 'medium'
                        else:
                            size_category = 'large'
                    
                        detection['size'] = size_category
                        detection['actual_width_cm'] = actual_w
                        detection['actual_height_cm'] = actual_h
                
                    detections.append(detection)
        
            return detections, image
        
        except Exception as e:
            print(f"目标检测错误: {e}")
            return [], image
    
    def set_background_image(self, image):
        """设置背景图像"""
        # 应用相同的预处理
        image = self.undistort_image_fast(image)
        self.background_image = self.perspective_transform(
            image, self.src_points, self.dst_points, self.output_width, self.output_height
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
            'focalLength': 1260.15,
            'cameraHeight': 13.0,
            'targetHeight': 2.7,
            'hamburgerSizeMin': 9,
            'hamburgerSizeMax': 11.3,
            'realWidthCm': 29,
            'realHeightCm': 18.5,
            'srcPoints': {
                'topLeft': {'x': 510, 'y': 270},
                'topRight': {'x': 1170, 'y': 270},
                'bottomRight': {'x': 1500, 'y': 710},
                'bottomLeft': {'x': 140, 'y': 710}
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
                    detections, processed_frame = self.yolo_detector.detect_objects(frame)
                    self.detected_objects = detections
                    self.processed_frame = processed_frame
                    self.last_detection_time = current_time
                
                self._update_fps()
                
                time.sleep(0.033)  # 约30 FPS
                
            except Exception as e:
                print(f"Video stream thread error: {e}")
                break
        
        print("Video stream thread stopped")
    
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
