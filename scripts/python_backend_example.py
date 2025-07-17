"""
Python QWebEngineView 后端代码 - 轮询版本 + YOLO检测
支持直接输入摄像头索引和模型路径，集成TensorFlow Lite YOLO检测
"""

# 启用调试模式（必须在 QApplication 之前设置）
import os

from utils import (
    compute_pixel_per_cm,
    init_undistort_maps,
    adjust_boxes,
    annotate_xywh,
    correct_size_with_height,
    undistort_image_fast,
    perspective_transform,
    detects_background_change_by_cosine,
)

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
        print(
            f"JS Console [{level}]: {message} (Line: {lineNumber}, Source: {sourceID})"
        )


class LetterBox:
    """LetterBox预处理类，与附件中的实现完全一致"""

    def __init__(
        self,
        new_shape,
        auto=False,
        scaleFill=False,
        scaleup=True,
        center=True,
        stride=32,
    ):
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
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

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
        self.image_size = (1080, 1920)
        # YOLO类别 - 与附件一致
        self.classes = [
            "Hamburger",
            "Shreddedchicken",
            "Burrito",
            "Slicedroastbeef",
            "Shreddedpork",
            "Fajitasteak",
            "Fajitachicken",
            "Cheeseburger",
            "Doublehamburger",
            "DoubleCheeseburger",
            "Chickenbreast bites",
            "Shrimpscampi",
            "Broccoliflorets",
            "ChoppedFajitaChicken",
            "ChoppedFajitaSteak",
        ]

        # 检测参数 - 与附件一致
        self.obj_thresh = 0.1
        self.nms_thresh = 0.1
        self.score_thresh = 0.0

        # 汉堡尺寸分类 - 与附件一致
        self.hamburger_size = (9, 11.3)
        self.hamburger_mj = (95, 143)

        # 透视变换参数 - 与附件一致
        self.src_points = np.array(
            [[510, 270], [1170, 270], [1500, 710], [140, 710]], dtype=np.float32
        )
        self.real_width_cm = 29.0
        self.real_height_cm = 18.5
        self.target_height = 1.8 + 0.9  # 面包高度 + 牛肉饼中心离面包表面高度
        self.camera_height = 13.0

        # 计算像素密度
        self.pixel_per_cm = compute_pixel_per_cm(
            self.src_points, self.real_width_cm, self.real_height_cm
        )

        # 计算输出尺寸
        self.output_width = int(self.real_width_cm * self.pixel_per_cm)
        self.output_height = int(self.real_height_cm * self.pixel_per_cm)
        self.dst_points = np.array(
            [
                [0, 0],
                [self.output_width, 0],
                [self.output_width, self.output_height],
                [0, self.output_height],
            ],
            dtype=np.float32,
        )

        # 相机内参矩阵 - 与前端一致
        self.camera_matrix = np.array(
            [
                [1260.15281, 0.0, 971.702426],
                [0.0, 1256.08744, 504.553169],
                [0.0, 0.0, 1.0],
            ]
        )
        self.dist_coeffs = np.array(
            [
                [
                    -0.430483648,
                    0.216393722,
                    -0.000156465611,
                    0.000104551776,
                    -0.0564557922,
                ]
            ]
        )

        # 初始化畸变矫正映射
        self.map1 = None
        self.map2 = None
        self.map1, self.map2 = init_undistort_maps(
            self.camera_matrix, self.dist_coeffs, self.image_size
        )
        self.background_image_path = "./img/bg.jpg"  # 用于背景变化检测
        self.set_background_image()  # 设置默认背景图像

        # 颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # 识别参数
        self.recognition_settings = {
            # 异物识别
            "foreignObjectDetection": True,
            # 尺寸分类
            "sizeClassification": True,
            # 透视变换
            "perspectiveEnabled": True,
            # 畸变矫正
            "distortionEnabled": True,
            # 高度补偿
            "heightCorrection": False,
        }

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载TensorFlow Lite模型"""
        try:
            self.interpreter = Interpreter(model_path=model_path, num_threads=4)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # 缓存量化参数
            self.input_scale, self.input_zero_point = self.input_details[0][
                "quantization"
            ]
            self.output_scale, self.output_zero_point = self.output_details[0][
                "quantization"
            ]

            # 缓存输入形状
            self.input_shape = self.input_details[0]["shape"]
            self.model_input_size = (self.input_shape[1], self.input_shape[2])

            self.model_path = model_path
            print(f"YOLO模型加载成功: {model_path}")
            print(f"输入尺寸: {self.model_input_size}")
            print(
                f"量化参数 - 输入: scale={self.input_scale}, zero_point={self.input_zero_point}"
            )
            print(
                f"量化参数 - 输出: scale={self.output_scale}, zero_point={self.output_zero_point}"
            )
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

    def draw_detections(self, image, boxes, scores, class_ids):
        """绘制检测结果 - 与附件逻辑一致"""
        max_score = 0
        max_class = None
        if self.recognition_settings.get("foreignObjectDetection", True):
            # 检测背景变化
            if self.background_image is not None:
                hash_diff, is_different = detects_background_change_by_cosine(
                    image, self.background_image, boxes
                )
                if is_different:
                    label_text = (
                        f"has foreign matter SCORE: {hash_diff:.3f}, {is_different}"
                    )
                    cv2.putText(
                        image,
                        label_text,
                        (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
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
                if self.classes[cl] == "Hamburger" and self.recognition_settings.get(
                    "sizeClassification", False
                ):
                    # 计算实际尺寸（只计算一次）
                    contour_w, contour_h = annotate_xywh(box, self.pixel_per_cm)

                    # 高度补偿（只计算一次）
                    contour_w = correct_size_with_height(
                        contour_w, self.target_height, self.camera_height
                    )

                    if contour_w and contour_h:
                        width_cm = contour_w
                        height_cm = contour_h

                        # 显示尺寸
                        label_text = f"{width_cm:.1f}cm"
                        x2 = int(x1 + w)
                        cv2.putText(
                            image,
                            label_text,
                            (x2 - 20, int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )

                        # 尺寸分类（使用已计算的值）
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
                        print(
                            f"轮廓尺寸:{size_label} {self.classes[cl]}: {width_cm:.2f} cm x {height_cm:.2f} cm"
                        )

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
                    color,
                    cv2.FILLED,
                )

                cv2.putText(
                    image,
                    label,
                    (int(label_x), int(label_y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
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
            if self.recognition_settings.get("distortionEnabled", True):
                # 畸变矫正
                image = undistort_image_fast(image, self.map1, self.map2)
            if self.recognition_settings.get("perspectiveEnabled", True):
                # 透视校正
                image = perspective_transform(
                    image,
                    self.src_points,
                    self.dst_points,
                    self.output_width,
                    self.output_height,
                )
                original_size = image.shape[:2]

            # 预处理 - 使用缓存的输入尺寸
            input_data = self.preprocess(image, self.model_input_size)
            input_data = input_data.transpose((0, 2, 3, 1))

            # 量化处理 - 使用缓存的量化参数
            if self.input_scale != 0:  # 量化模型
                input_data = (
                    input_data / self.input_scale + self.input_zero_point
                ).astype(np.int8)

            # 模型推理
            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()

            # 获取输出
            output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
            print(f"获取输出{output_data}")
            # 反量化 - 使用缓存的量化参数
            if self.output_scale != 0:  # 量化模型
                output_data = (
                    output_data.astype(np.float32) - self.output_zero_point
                ) * self.output_scale

            # 将预测框的坐标从归一化形式转换回原始图像尺寸
            output_data[:, [0, 2]] *= self.model_input_size[0]
            output_data[:, [1, 3]] *= self.model_input_size[1]

            # 后处理
            result_boxes, result_scores, result_class_ids = self.postprocess(
                output_data
            )
            print(f"后处理{result_boxes}")
            detections = []
            if len(result_boxes) > 0:
                result_boxes = adjust_boxes(
                    self.model_input_size,
                    np.array(result_boxes, dtype=np.float64),
                    (original_size[1], original_size[0]),
                )

                # 绘制检测结果
                image = self.draw_detections(
                    image, result_boxes, result_scores, result_class_ids
                )

                # 转换为标准格式
                for i, (box, score, class_id) in enumerate(
                    zip(result_boxes, result_scores, result_class_ids)
                ):
                    x, y, w, h = box
                    detection = {
                        "id": f"obj_{int(time.time())}_{i}",
                        "type": (
                            self.classes[class_id]
                            if class_id < len(self.classes)
                            else "Unknown"
                        ),
                        "confidence": float(score),
                        "x": float(x),
                        "y": float(y),
                        "width": float(w),
                        "height": float(h),
                        "class_id": int(class_id),
                    }

                    # 如果是汉堡，添加尺寸信息（使用已有的计算结果）
                    if self.classes[
                        class_id
                    ] == "Hamburger" and self.recognition_settings.get(
                        "sizeClassification", False
                    ):
                        # 重用 draw_detections 中的计算逻辑，避免重复计算
                        actual_w, actual_h = annotate_xywh(box, self.pixel_per_cm)
                        # 高度补偿
                        if self.recognition_settings.get("heightCorrection", True):
                            actual_w = correct_size_with_height(
                                actual_w, self.target_height, self.camera_height
                            )
                        if actual_w and actual_h:
                            # 尺寸分类
                            width_min, width_max = self.hamburger_size
                            mj_min, mj_max = self.hamburger_mj
                            mj = actual_w * actual_h

                            if mj <= mj_min:
                                size_category = (
                                    "small" if actual_w <= width_min else "medium"
                                )
                            elif mj > mj_min and mj <= mj_max:
                                size_category = (
                                    "large" if actual_w > width_max else "medium"
                                )
                            else:
                                size_category = "large"

                            detection["size"] = size_category
                            detection["actual_width_cm"] = actual_w
                            detection["actual_height_cm"] = actual_h

                    detections.append(detection)

            return detections, image

        except Exception as e:
            print(f"目标检测错误: {e}")
            return [], image

    def set_background_image(self):
        bg_image = cv2.imread(self.background_image_path)
        """设置背景图像"""
        # 应用相同的预处理
        image = undistort_image_fast(bg_image, self.map1, self.map2)
        self.background_image = perspective_transform(
            image,
            self.src_points,
            self.dst_points,
            self.output_width,
            self.output_height,
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
            "contrast": 100,
            "brightness": 100,
            "saturation": 100,
            "blur": 0,
            "delaySeconds": 0,
            "objThresh": 10,
            "nmsThresh": 10,
            "scoreThresh": 0,
            "perspectiveEnabled": True,
            "distortionEnabled": True,
            "focalLength": 1260.15,
            "cameraHeight": 13.0,
            "targetHeight": 2.7,
            "hamburgerSizeMin": 9,
            "hamburgerSizeMax": 11.3,
            "hamburgerSizeMjMin": 95,
            "hamburgerSizeMjMax": 143,
            "realWidthCm": 29,
            "realHeightCm": 18.5,
            "backgroundImagePath": "./img/bg.jpg",
            "srcPoints": {
                "topLeft": {"x": 510, "y": 270},
                "topRight": {"x": 1170, "y": 270},
                "bottomRight": {"x": 1500, "y": 710},
                "bottomLeft": {"x": 140, "y": 710},
            },
        }

        self.recognition_settings = {}
        self.detected_objects = []
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.detection_delay = 0
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
                return json.dumps(
                    {"success": False, "error": "无效的摄像头索引"}, ensure_ascii=False
                )

            if camera_idx < 0 or camera_idx > 10:
                return json.dumps(
                    {"success": False, "error": "摄像头索引必须在0-10之间"},
                    ensure_ascii=False,
                )

            # 停止现有的摄像头
            if self.camera:
                self.stop_camera()

            self.camera = cv2.VideoCapture(camera_idx)
            if not self.camera.isOpened():
                return json.dumps(
                    {"success": False, "error": f"无法打开摄像头 {camera_idx}"},
                    ensure_ascii=False,
                )

            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)

            self.current_camera_index = camera_index
            self.is_streaming = True

            # 设置背景图像（第一帧）
            # ret, frame = self.camera.read()
            # if ret:
            #     self.yolo_detector.set_background_image(frame)

            # 启动视频流线程
            self.video_thread_running = True
            self.video_thread = threading.Thread(
                target=self._video_stream_thread, daemon=True
            )
            self.video_thread.start()

            result = json.dumps(
                {"success": True, "message": f"摄像头 {camera_idx} 启动成功"},
                ensure_ascii=False,
            )
            print(f"start_camera result: {result}")
            return result
        except Exception as e:
            error_result = json.dumps(
                {"success": False, "error": str(e)}, ensure_ascii=False
            )
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

            result = json.dumps(
                {"success": True, "message": "摄像头已停止"}, ensure_ascii=False
            )
            print(f"stop_camera result: {result}")
            return result
        except Exception as e:
            error_result = json.dumps(
                {"success": False, "error": str(e)}, ensure_ascii=False
            )
            print(f"stop_camera error: {error_result}")
            return error_result

    @pyqtSlot(str, result=str)
    def load_model(self, model_path: str) -> str:
        """加载AI模型"""
        try:
            print(f"Loading model: {model_path}")

            if not os.path.exists(model_path):
                result = json.dumps(
                    {"success": True, "message": f"模型加载成功 (模拟): {model_path}"},
                    ensure_ascii=False,
                )
                print(f"load_model result (mock): {result}")
                self.current_model_path = model_path
                return result

            # 加载YOLO模型
            success = self.yolo_detector.load_model(model_path)
            if success:
                self.current_model_path = model_path
                result = json.dumps(
                    {"success": True, "message": f"YOLO模型加载成功: {model_path}"},
                    ensure_ascii=False,
                )
            else:
                result = json.dumps(
                    {"success": False, "error": f"YOLO模型加载失败: {model_path}"},
                    ensure_ascii=False,
                )

            print(f"load_model result: {result}")
            return result
        except Exception as e:
            error_result = json.dumps(
                {"success": False, "error": str(e)}, ensure_ascii=False
            )
            print(f"load_model error: {error_result}")
            return error_result

    @pyqtSlot(str, result=str)
    def set_image_params(self, params_json: str) -> str:
        """设置图像处理参数"""
        try:
            print(f"Setting image params: {params_json}")
            params = json.loads(params_json)

            # 功能依赖检查
            distortion_enabled = params.get(
                "distortionEnabled", self.image_params.get("distortionEnabled", True)
            )
            perspective_enabled = params.get(
                "perspectiveEnabled", self.image_params.get("perspectiveEnabled", True)
            )

            # 检查：如果畸变矫正没开启就不能开启透视变换
            if perspective_enabled and not distortion_enabled:
                return json.dumps(
                    {"success": False, "error": "透视变换功能需要先启用畸变矫正功能"},
                    ensure_ascii=False,
                )

            self.image_params.update(params)

            # 更新YOLO检测器参数
            if "objThresh" in params:
                self.yolo_detector.obj_thresh = params["objThresh"] / 100.0
            if "nmsThresh" in params:
                self.yolo_detector.nms_thresh = params["nmsThresh"] / 100.0
            if "scoreThresh" in params:
                self.yolo_detector.score_thresh = params["scoreThresh"] / 100.0

            # 更新透视变换参数
            if "srcPoints" in params:
                src_points = params["srcPoints"]
                self.yolo_detector.src_points = np.array(
                    [
                        [src_points["topLeft"]["x"], src_points["topLeft"]["y"]],
                        [src_points["topRight"]["x"], src_points["topRight"]["y"]],
                        [
                            src_points["bottomRight"]["x"],
                            src_points["bottomRight"]["y"],
                        ],
                        [src_points["bottomLeft"]["x"], src_points["bottomLeft"]["y"]],
                    ],
                    dtype=np.float32,
                )

                # 重新计算像素密度和输出尺寸
                self.yolo_detector.pixel_per_cm = compute_pixel_per_cm(
                    self.yolo_detector.src_points,
                    self.yolo_detector.real_width_cm,
                    self.yolo_detector.real_height_cm,
                )
                self.yolo_detector.output_width = int(
                    self.yolo_detector.real_width_cm * self.yolo_detector.pixel_per_cm
                )
                self.yolo_detector.output_height = int(
                    self.yolo_detector.real_height_cm * self.yolo_detector.pixel_per_cm
                )
                self.yolo_detector.dst_points = np.array(
                    [
                        [0, 0],
                        [self.yolo_detector.output_width, 0],
                        [
                            self.yolo_detector.output_width,
                            self.yolo_detector.output_height,
                        ],
                        [0, self.yolo_detector.output_height],
                    ],
                    dtype=np.float32,
                )

            # 更新畸变矫正参数
            if any(
                key in params
                for key in [
                    "distortionK1",
                    "distortionK2",
                    "distortionP1",
                    "distortionP2",
                    "distortionK3",
                ]
            ):
                # 更新畸变系数
                self.yolo_detector.dist_coeffs = np.array(
                    [
                        [
                            params.get(
                                "distortionK1", self.yolo_detector.dist_coeffs[0][0]
                            ),
                            params.get(
                                "distortionK2", self.yolo_detector.dist_coeffs[0][1]
                            ),
                            params.get(
                                "distortionP1", self.yolo_detector.dist_coeffs[0][2]
                            ),
                            params.get(
                                "distortionP2", self.yolo_detector.dist_coeffs[0][3]
                            ),
                            params.get(
                                "distortionK3", self.yolo_detector.dist_coeffs[0][4]
                            ),
                        ]
                    ]
                )

                # 重新初始化畸变矫正映射
                self.yolo_detector.map1, self.yolo_detector.map2 = init_undistort_maps(
                    self.yolo_detector.camera_matrix,
                    self.yolo_detector.dist_coeffs,
                    self.yolo_detector.image_size,
                )

            # 更新相机内参
            if any(
                key in params
                for key in [
                    "focalLengthX",
                    "focalLengthY",
                    "principalPointX",
                    "principalPointY",
                ]
            ):
                self.yolo_detector.camera_matrix = np.array(
                    [
                        [
                            params.get("focalLengthX", self.camera_matrix[0][0]),
                            0.0,
                            params.get("principalPointX", self.camera_matrix[0][2]),
                        ],
                        [
                            0.0,
                            params.get("focalLengthY", self.camera_matrix[1][1]),
                            params.get("principalPointY", self.camera_matrix[1][2]),
                        ],
                        [0.0, 0.0, 1.0],
                    ]
                )
                # 重新初始化畸变矫正映射
                self.yolo_detector.map1, self.yolo_detector.map2 = init_undistort_maps(
                    self.yolo_detector.camera_matrix,
                    self.yolo_detector.dist_coeffs,
                    self.yolo_detector.image_size,
                )

            # 更新相机高度和目标高度
            if "cameraHeight" in params:
                self.yolo_detector.camera_height = params["cameraHeight"]
            if "targetHeight" in params:
                self.yolo_detector.target_height = params["targetHeight"]

            # 更新汉堡尺寸分类参数
            if "hamburgerSizeMin" in params or "hamburgerSizeMax" in params:
                min_size = params.get(
                    "hamburgerSizeMin", self.yolo_detector.hamburger_size[0]
                )
                max_size = params.get(
                    "hamburgerSizeMax", self.yolo_detector.hamburger_size[1]
                )
                self.yolo_detector.hamburger_size = (min_size, max_size)
            # 更新汉堡尺寸面积分类参数
            if "hamburgerSizeMjMin" in params or "hamburgerSizeMjMax" in params:
                min_size = params.get(
                    "hamburgerSizeMjMin", self.yolo_detector.hamburger_mj[0]
                )
                max_size = params.get(
                    "hamburgerSizeMjMax", self.yolo_detector.hamburger_mj[1]
                )
                self.yolo_detector.hamburger_mj = (min_size, max_size)

            # 更新实际尺寸参数
            if "realWidthCm" in params or "realHeightCm" in params:
                self.yolo_detector.real_width_cm = params.get(
                    "realWidthCm", self.yolo_detector.real_width_cm
                )
                self.yolo_detector.real_height_cm = params.get(
                    "realHeightCm", self.yolo_detector.real_height_cm
                )

                # 重新计算像素密度和输出尺寸
                self.yolo_detector.pixel_per_cm = compute_pixel_per_cm(
                    self.yolo_detector.src_points,
                    self.yolo_detector.real_width_cm,
                    self.yolo_detector.real_height_cm,
                )
                self.yolo_detector.output_width = int(
                    self.yolo_detector.real_width_cm * self.yolo_detector.pixel_per_cm
                )
                self.yolo_detector.output_height = int(
                    self.yolo_detector.real_height_cm * self.yolo_detector.pixel_per_cm
                )
                self.yolo_detector.dst_points = np.array(
                    [
                        [0, 0],
                        [self.yolo_detector.output_width, 0],
                        [
                            self.yolo_detector.output_width,
                            self.yolo_detector.output_height,
                        ],
                        [0, self.yolo_detector.output_height],
                    ],
                    dtype=np.float32,
                )

            # 更新背景图像路径
            if "backgroundImagePath" in params:
                self.yolo_detector.background_image_path = params["backgroundImagePath"]
                self.yolo_detector.set_background_image()

            # 更新识别功能开关
            if "perspectiveEnabled" in params:
                self.yolo_detector.recognition_settings["perspectiveEnabled"] = params[
                    "perspectiveEnabled"
                ]
            if "distortionEnabled" in params:
                self.yolo_detector.recognition_settings["distortionEnabled"] = params[
                    "distortionEnabled"
                ]

            # 更新延迟参数
            delay_seconds = params.get("delaySeconds", 0)
            if delay_seconds >= 0:
                self.detection_delay = delay_seconds

            result = json.dumps(
                {"success": True, "message": "图像参数已更新"}, ensure_ascii=False
            )
            print(f"set_image_params result: {result}")
            return result
        except Exception as e:
            error_result = json.dumps(
                {"success": False, "error": str(e)}, ensure_ascii=False
            )
            print(f"set_image_params error: {error_result}")
            return error_result

    @pyqtSlot(str, result=str)
    def set_recognition_settings(self, settings_json: str) -> str:
        """设置识别功能"""
        try:
            print(f"Setting recognition settings: {settings_json}")
            settings = json.loads(settings_json)

            # 功能依赖检查
            current_distortion = self.yolo_detector.recognition_settings.get(
                "distortionEnabled", True
            )
            current_perspective = self.yolo_detector.recognition_settings.get(
                "perspectiveEnabled", True
            )

            size_classification = settings.get("sizeClassification", False)
            height_correction = settings.get("heightCorrection", False)
            foreign_object_detection = settings.get("foreignObjectDetection", False)

            # 检查：如果透视变换没开启就不能开启大中小、高度矫正、异物识别
            if (
                size_classification or height_correction or foreign_object_detection
            ) and not current_perspective:
                return json.dumps(
                    {
                        "success": False,
                        "error": "尺寸分类、高度矫正和异物识别功能需要先启用透视变换功能",
                    },
                    ensure_ascii=False,
                )

            # 检查：如果畸变矫正没开启就不能开启透视变换相关功能
            if (
                size_classification or height_correction or foreign_object_detection
            ) and not current_distortion:
                return json.dumps(
                    {
                        "success": False,
                        "error": "尺寸分类、高度矫正和异物识别功能需要先启用畸变矫正功能",
                    },
                    ensure_ascii=False,
                )

            # 更新YOLO检测器的识别设置
            self.yolo_detector.recognition_settings.update(self.recognition_settings)
            result = json.dumps(
                {"success": True, "message": "识别设置已更新"}, ensure_ascii=False
            )
            print(f"set_recognition_settings result: {result}")
            return result
        except Exception as e:
            error_result = json.dumps(
                {"success": False, "error": str(e)}, ensure_ascii=False
            )
            print(f"set_recognition_settings error: {error_result}")
            return error_result

    @pyqtSlot(result=str)
    def get_current_frame(self) -> str:
        """获取当前处理后的图像帧 - 只有在摄像头启动时才返回帧"""
        try:
            # 检查摄像头是否正在运行
            if not self.is_streaming or self.processed_frame is None:
                return json.dumps(
                    {"success": False, "error": "摄像头未启动或没有可用的图像帧"},
                    ensure_ascii=False,
                )

            # 编码图像为base64
            _, buffer = cv2.imencode(
                ".jpg", self.processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            import base64

            img_base64 = base64.b64encode(buffer).decode("utf-8")

            result = json.dumps(
                {
                    "success": True,
                    "image": f"data:image/jpeg;base64,{img_base64}",
                    "timestamp": time.time(),
                },
                ensure_ascii=False,
            )

            return result
        except Exception as e:
            error_result = json.dumps(
                {"success": False, "error": str(e)}, ensure_ascii=False
            )
            print(f"get_current_frame error: {error_result}")
            return error_result

    @pyqtSlot(result=str)
    def get_detection_results(self) -> str:
        """获取检测结果 - 只有在摄像头启动时才返回结果"""
        try:
            if not self.is_streaming:
                return json.dumps({"objects": []}, ensure_ascii=False)

            result = json.dumps({"objects": self.detected_objects}, ensure_ascii=False)
            return result
        except Exception as e:
            error_result = json.dumps(
                {"objects": [], "error": str(e)}, ensure_ascii=False
            )
            print(f"get_detection_results error: {error_result}")
            return error_result

    @pyqtSlot(str, result=str)
    def save_image(self, filename: str = "") -> str:
        """保存当前图像"""
        try:
            if not self.is_streaming or self.current_frame is None:
                return json.dumps(
                    {"success": False, "error": "摄像头未启动或没有可保存的图像"},
                    ensure_ascii=False,
                )
            # 获取当前用户桌面路径（支持中文或英文系统）
            desktop_path = os.path.join(os.path.expanduser("~"), "桌面")
            if not os.path.exists(desktop_path):
                desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            # 创建保存文件夹路径
            save_folder = os.path.join(desktop_path, "SavedFrames")
            os.makedirs(save_folder, exist_ok=True)  # 如果文件夹不存在就创建
            # 构造保存文件名
            if not filename:
                filename = os.path.join(
                    save_folder, f"captured_image_{int(time.time())}.jpg"
                )
            cv2.imwrite(filename, self.current_frame)
            result = json.dumps(
                {"success": True, "message": f"图像已保存: {filename}"},
                ensure_ascii=False,
            )
            print(f"save_image result: {result}")
            return result
        except Exception as e:
            error_result = json.dumps(
                {"success": False, "error": str(e)}, ensure_ascii=False
            )
            print(f"save_image error: {error_result}")
            return error_result

    @pyqtSlot(str, result=str)
    def export_report(self, format_type: str = "json") -> str:
        """导出检测报告"""
        try:
            report_data = {
                "timestamp": time.time(),
                "model_path": self.current_model_path,
                "camera_index": self.current_camera_index,
                "image_params": self.image_params,
                "recognition_settings": self.recognition_settings,
                "detected_objects": self.detected_objects,
                "total_objects": len(self.detected_objects),
                "anomaly_count": len(
                    [
                        obj
                        for obj in self.detected_objects
                        if obj.get("type") in ["异物", "缺陷"]
                    ]
                ),
            }
            # 获取当前用户桌面路径（支持中文或英文系统）
            desktop_path = os.path.join(os.path.expanduser("~"), "桌面")
            if not os.path.exists(desktop_path):
                desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            # 创建保存文件夹路径
            save_folder = os.path.join(desktop_path, "SavedFrames")
            os.makedirs(save_folder, exist_ok=True)  # 如果文件夹不存在就创建

            filename = os.path.join(
                save_folder, f"detection_report_{int(time.time())}.{format_type}"
            )

            if format_type == "json":
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(report_data, f, ensure_ascii=False, indent=2)

            result = json.dumps(
                {"success": True, "message": f"报告已导出: {filename}"},
                ensure_ascii=False,
            )
            print(f"export_report result: {result}")
            return result
        except Exception as e:
            error_result = json.dumps(
                {"success": False, "error": str(e)}, ensure_ascii=False
            )
            print(f"export_report error: {error_result}")
            return error_result

    @pyqtSlot(result=str)
    def get_system_status(self) -> str:
        """获取系统状态"""
        try:
            status = {
                "fps": self.fps_counter,
                "cpuUsage": psutil.cpu_percent(),
                "memoryUsage": psutil.virtual_memory().percent,
                "gpuUsage": 0,
            }
            result = json.dumps(status, ensure_ascii=False)
            return result
        except Exception as e:
            error_result = json.dumps(
                {
                    "fps": 0,
                    "cpuUsage": 0,
                    "memoryUsage": 0,
                    "gpuUsage": 0,
                    "error": str(e),
                },
                ensure_ascii=False,
            )
            print(f"get_system_status error: {error_result}")
            return error_result

    def _video_stream_thread(self):
        """视频流处理线程 - 只有在摄像头启动时才运行"""
        print("Video stream thread started")
        print(f"只有在摄像头启动时才运行")
        while self.video_thread_running and self.is_streaming and self.camera:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                self.current_frame = frame.copy()
                print(f"只有在摄像头启动时才运行1")
                # 根据延迟设置执行目标检测
                current_time = time.time()
                if (current_time - self.last_detection_time) >= self.detection_delay:
                    print(f"只有在摄像头启动时才运行2")
                    frame = self._apply_image_processing(frame)
                    detections, processed_frame = self.yolo_detector.detect_objects(
                        frame
                    )
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

    # === 基础的图像处理函数 ===
    def _apply_image_processing(self, frame):
        """应用图像处理参数"""
        processed = frame.copy()
        # 基础图像调整
        contrast = self.image_params.get("contrast", 100) / 100.0
        brightness = self.image_params.get("brightness", 100) - 100
        processed = cv2.convertScaleAbs(processed, alpha=contrast, beta=brightness)

        # 饱和度调整
        saturation = self.image_params.get("saturation", 100) / 100.0
        if saturation != 1.0:
            hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 模糊处理
        blur = self.image_params.get("blur", 0)
        if blur > 0:
            kernel_size = int(blur * 2) + 1
            processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), blur)

        return processed


class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像识别系统 - YOLO检测版")
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
        settings.setAttribute(
            QWebEngineSettings.AllowGeolocationOnInsecureOrigins, True
        )
        settings.setAttribute(QWebEngineSettings.ShowScrollBars, False)

        layout.addWidget(self.web_view)

        # 创建 API 对象
        self.api = ImageRecognitionAPI()

        # 设置 Web Channel
        self.channel = QWebChannel()
        self.channel.registerObject("pyapi", self.api)
        self.web_view.page().setWebChannel(self.channel)

        # 加载网页
        self.load_web_page()

    def load_web_page(self):
        """加载网页"""
        url = QUrl("http://localhost:3000")  # Next.js 开发服务器
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
    app.setApplicationName("图像识别系统 - YOLO检测版")
    app.setApplicationVersion("3.0.0")

    # 创建主窗口
    window = MainWindow()
    window.show()

    print("WebEngine Remote Debugging enabled on port 9222")
    print("You can access it at: http://localhost:9222")
    print("YOLO Detection System Ready!")

    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
