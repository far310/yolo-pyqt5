"""
Python QWebEngineView 后端代码 - 轮询版本 + YOLO检测
支持直接输入摄像头索引和模型路径，集成TensorFlow Lite YOLO检测
"""

# 启用调试模式（必须在 QApplication 之前设置）
import os

from scripts.utils import (
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
        
        # 背景图像路径和背景图像
        self.background_image_path = "./img/bg.jpg"  # 默认背景图像路径
        self.background_image = None
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
                if self.classes[cl] == "Hamburger":
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
                [output_data]
            )

            # 调整边界框坐标
            result_boxes = adjust_boxes(
                result_boxes, self.model_input_size, original_size
            )

            # 绘制检测结果
            annotated_image = self.draw_detections(
                image, result_boxes, result_scores, result_class_ids
            )

            # 转换为检测对象列表
            detected_objects = []
            for i, (box, score, class_id) in enumerate(
                zip(result_boxes, result_scores, result_class_ids)
            ):
                if score >= self.score_thresh:
                    x, y, w, h = box
                    detected_objects.append(
                        {
                            "id": f"obj_{i}",
                            "type": self.classes[class_id],
                            "confidence": float(score),
                            "size": "medium",  # 默认尺寸
                            "x": float(x),
                            "y": float(y),
                            "width": float(w),
                            "height": float(h),
                        }
                    )

            return detected_objects, annotated_image

        except Exception as e:
            print(f"检测过程中出错: {e}")
            return [], image

    def update_detection_params(self, params: Dict[str, Any]):
        """更新检测参数"""
        if "objThresh" in params:
            self.obj_thresh = params["objThresh"] / 100.0
        if "nmsThresh" in params:
            self.nms_thresh = params["nmsThresh"] / 100.0
        if "scoreThresh" in params:
            self.score_thresh = params["scoreThresh"] / 100.0

        # 更新透视变换参数
        if "srcPoints" in params:
            src_points = params["srcPoints"]
            self.src_points = np.array(
                [
                    [src_points["topLeft"]["x"], src_points["topLeft"]["y"]],
                    [src_points["topRight"]["x"], src_points["topRight"]["y"]],
                    [src_points["bottomRight"]["x"], src_points["bottomRight"]["y"]],
                    [src_points["bottomLeft"]["x"], src_points["bottomLeft"]["y"]],
                ],
                dtype=np.float32,
            )

        # 更新畸变矫正参数
        if any(
            key in params
            for key in ["distortionK1", "distortionK2", "distortionP1", "distortionP2", "distortionK3"]
        ):
            self.dist_coeffs = np.array(
                [
                    [
                        params.get("distortionK1", self.dist_coeffs[0][0]) / 100.0,
                        params.get("distortionK2", self.dist_coeffs[0][1]) / 100.0,
                        params.get("distortionP1", self.dist_coeffs[0][2]),
                        params.get("distortionP2", self.dist_coeffs[0][3]),
                        params.get("distortionK3", self.dist_coeffs[0][4]) / 100.0,
                    ]
                ]
            )
            # 重新计算畸变矫正映射
            self.map1, self.map2 = init_undistort_maps(
                self.camera_matrix, self.dist_coeffs, self.image_size
            )

        # 更新相机内参
        if any(
            key in params
            for key in ["focalLengthX", "focalLengthY", "principalPointX", "principalPointY"]
        ):
            self.camera_matrix = np.array(
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
            # 重新计算畸变矫正映射
            self.map1, self.map2 = init_undistort_maps(
                self.camera_matrix, self.dist_coeffs, self.image_size
            )

        # 更新相机参数
        if "cameraHeight" in params:
            self.camera_height = params["cameraHeight"]
        if "targetHeight" in params:
            self.target_height = params["targetHeight"]

        # 更新汉堡尺寸参数
        if "hamburgerSizeMin" in params or "hamburgerSizeMax" in params:
            self.hamburger_size = (
                params.get("hamburgerSizeMin", self.hamburger_size[0]),
                params.get("hamburgerSizeMax", self.hamburger_size[1]),
            )

        # 更新实际尺寸参数
        if "realWidthCm" in params or "realHeightCm" in params:
            self.real_width_cm = params.get("realWidthCm", self.real_width_cm)
            self.real_height_cm = params.get("realHeightCm", self.real_height_cm)
            # 重新计算像素密度和输出尺寸
            self.pixel_per_cm = compute_pixel_per_cm(
                self.src_points, self.real_width_cm, self.real_height_cm
            )
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

        # 更新背景图像路径
        if "backgroundImagePath" in params:
            self.background_image_path = params["backgroundImagePath"]
            self.set_background_image()

        print(f"检测参数已更新: {params}")

    def set_background_image(self):
        """设置背景图像"""
        try:
            if os.path.exists(self.background_image_path):
                self.background_image = cv2.imread(self.background_image_path)
                if self.background_image is not None:
                    print(f"背景图像加载成功: {self.background_image_path}")
                else:
                    print(f"背景图像加载失败: {self.background_image_path}")
                    self.background_image = None
            else:
                print(f"背景图像文件不存在: {self.background_image_path}")
                self.background_image = None
        except Exception as e:
            print(f"设置背景图像时出错: {e}")
            self.background_image = None

    def update_recognition_settings(self, settings: Dict[str, Any]):
        """更新识别设置"""
        # 参数依赖关系检查
        if settings.get("perspectiveEnabled", False) and not settings.get("distortionEnabled", False):
            print("警告: 透视变换需要先启用畸变矫正")
            settings["perspectiveEnabled"] = False
            
        if (settings.get("sizeClassification", False) or 
            settings.get("heightCorrection", False) or 
            settings.get("foreignObjectDetection", False)) and not settings.get("perspectiveEnabled", False):
            print("警告: 尺寸分类、高度补偿、异物检测需要先启用透视变换")
            settings["sizeClassification"] = False
            settings["heightCorrection"] = False
            settings["foreignObjectDetection"] = False

        self.recognition_settings.update(settings)
        print(f"识别设置已更新: {self.recognition_settings}")


class PythonBackend(QObject):
    """Python后端类，处理与前端的通信"""

    def __init__(self):
        super().__init__()
        self.detector = YOLODetector()
        self.camera = None
        self.is_streaming = False
        self.current_frame = None
        self.detected_objects = []
        self.frame_queue = Queue(maxsize=10)
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 性能监控
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # 定时器用于推送数据
        self.push_timer = QTimer()
        self.push_timer.timeout.connect(self.push_data_to_frontend)
        self.push_timer.start(100)  # 每100ms推送一次数据

        # 图像处理参数
        self.image_params = {
            "contrast": 100,
            "brightness": 100,
            "saturation": 100,
            "blur": 0,
            "delaySeconds": 1,
        }

        # 保存的图像和报告
        self.saved_images = []
        self.detection_reports = []

    def push_data_to_frontend(self):
        """定时推送数据到前端"""
        if hasattr(self, "web_page") and self.web_page:
            # 构建推送数据
            push_data = {
                "type": "data_push",
                "data": {
                    "currentFrame": self.get_current_frame_base64(),
                    "detectedObjects": self.detected_objects,
                    "systemStatus": self.get_system_status(),
                    "isStreaming": self.is_streaming,
                },
            }

            # 推送到前端
            js_code = f"window.handlePythonPush && window.handlePythonPush({json.dumps(push_data)});"
            self.web_page.runJavaScript(js_code)

    def get_current_frame_base64(self):
        """获取当前帧的base64编码"""
        if self.current_frame is not None:
            try:
                # 编码为JPEG
                _, buffer = cv2.imencode(".jpg", self.current_frame)
                # 转换为base64
                import base64

                frame_base64 = base64.b64encode(buffer).decode("utf-8")
                return f"data:image/jpeg;base64,{frame_base64}"
            except Exception as e:
                print(f"编码帧失败: {e}")
        return None

    def get_system_status(self):
        """获取系统状态"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            return {
                "fps": self.current_fps,
                "cpu": cpu_percent,
                "memory": memory.percent,
                "isConnected": True,
            }
        except:
            return {"fps": 0, "cpu": 0, "memory": 0, "isConnected": False}

    def apply_image_adjustments(self, frame):
        """应用图像调整"""
        try:
            # 对比度和亮度调整
            contrast = self.image_params["contrast"] / 100.0
            brightness = self.image_params["brightness"] - 100

            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

            # 饱和度调整
            saturation = self.image_params["saturation"] / 100.0
            if saturation != 1.0:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = hsv[:, :, 1] * saturation
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # 模糊处理
            blur_value = self.image_params["blur"]
            if blur_value > 0:
                kernel_size = int(blur_value * 2) + 1
                frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

            return frame
        except Exception as e:
            print(f"图像调整失败: {e}")
            return frame

    def camera_thread(self):
        """摄像头线程"""
        delay_seconds = self.image_params.get("delaySeconds", 1)

        while self.is_streaming and self.camera is not None:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break

                # 应用图像调整
                frame = self.apply_image_adjustments(frame)

                # YOLO检测
                detected_objects, annotated_frame = self.detector.detect_objects(frame)

                # 更新当前帧和检测结果
                self.current_frame = annotated_frame
                self.detected_objects = detected_objects

                # 更新FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = current_time

                # 延迟
                time.sleep(delay_seconds)

            except Exception as e:
                print(f"摄像头线程错误: {e}")
                break

        print("摄像头线程结束")

    @pyqtSlot(str, result=str)
    def handle_request(self, request_json: str) -> str:
        """处理前端请求"""
        try:
            request = json.loads(request_json)
            action = request.get("action")
            params = request.get("params", {})

            print(f"收到请求: {action}")

            if action == "start_camera":
                return self.start_camera(params.get("camera_id", "0"))
            elif action == "stop_camera":
                return self.stop_camera()
            elif action == "load_model":
                return self.load_model(params.get("model_path"))
            elif action == "update_image_params":
                return self.update_image_params(params)
            elif action == "update_recognition_settings":
                return self.update_recognition_settings(params)
            elif action == "save_image":
                return self.save_image()
            elif action == "export_report":
                return self.export_report(params.get("format", "json"))
            elif action == "get_system_status":
                return json.dumps(
                    {"success": True, "data": self.get_system_status()}
                )
            else:
                return json.dumps({"success": False, "error": f"未知操作: {action}"})

        except Exception as e:
            print(f"处理请求时出错: {e}")
            return json.dumps({"success": False, "error": str(e)})

    def start_camera(self, camera_id: str) -> str:
        """启动摄像头"""
        try:
            if self.is_streaming:
                self.stop_camera()

            # 尝试将camera_id转换为整数，如果失败则作为字符串处理
            try:
                cam_id = int(camera_id)
            except ValueError:
                cam_id = camera_id

            self.camera = cv2.VideoCapture(cam_id)
            if not self.camera.isOpened():
                return json.dumps({"success": False, "error": "无法打开摄像头"})

            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

            self.is_streaming = True

            # 启动摄像头线程
            camera_thread = threading.Thread(target=self.camera_thread, daemon=True)
            camera_thread.start()

            return json.dumps({"success": True, "message": f"摄像头 {camera_id} 启动成功"})

        except Exception as e:
            return json.dumps({"success": False, "error": f"启动摄像头失败: {str(e)}"})

    def stop_camera(self) -> str:
        """停止摄像头"""
        try:
            self.is_streaming = False
            if self.camera:
                self.camera.release()
                self.camera = None

            self.current_frame = None
            self.detected_objects = []

            return json.dumps({"success": True, "message": "摄像头已停止"})

        except Exception as e:
            return json.dumps({"success": False, "error": f"停止摄像头失败: {str(e)}"})

    def load_model(self, model_path: str) -> str:
        """加载模型"""
        try:
            if not model_path:
                return json.dumps({"success": False, "error": "模型路径不能为空"})

            success = self.detector.load_model(model_path)
            if success:
                return json.dumps({"success": True, "message": f"模型加载成功: {model_path}"})
            else:
                return json.dumps({"success": False, "error": "模型加载失败"})

        except Exception as e:
            return json.dumps({"success": False, "error": f"加载模型失败: {str(e)}"})

    def update_image_params(self, params: Dict[str, Any]) -> str:
        """更新图像参数"""
        try:
            self.image_params.update(params)
            # 同时更新检测器的参数
            self.detector.update_detection_params(params)
            return json.dumps({"success": True, "message": "图像参数更新成功"})

        except Exception as e:
            return json.dumps({"success": False, "error": f"更新图像参数失败: {str(e)}"})

    def update_recognition_settings(self, settings: Dict[str, Any]) -> str:
        """更新识别设置"""
        try:
            self.detector.update_recognition_settings(settings)
            return json.dumps({"success": True, "message": "识别设置更新成功"})

        except Exception as e:
            return json.dumps(
                {"success": False, "error": f"更新识别设置失败: {str(e)}"}
            )

    def save_image(self) -> str:
        """保存当前图像"""
        try:
            if self.current_frame is None:
                return json.dumps({"success": False, "error": "没有可保存的图像"})

            # 创建保存目录
            save_dir = "saved_images"
            os.makedirs(save_dir, exist_ok=True)

            # 生成文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)

            # 保存图像
            cv2.imwrite(filepath, self.current_frame)
            self.saved_images.append(filepath)

            return json.dumps(
                {"success": True, "message": f"图像已保存: {filepath}", "path": filepath}
            )

        except Exception as e:
            return json.dumps({"success": False, "error": f"保存图像失败: {str(e)}"})

    def export_report(self, format_type: str = "json") -> str:
        """导出检测报告"""
        try:
            # 创建报告目录
            report_dir = "reports"
            os.makedirs(report_dir, exist_ok=True)

            # 生成报告数据
            report_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_objects": len(self.detected_objects),
                "objects": self.detected_objects,
                "system_status": self.get_system_status(),
            }

            # 生成文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.{format_type}"
            filepath = os.path.join(report_dir, filename)

            # 保存报告
            if format_type == "json":
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(report_data, f, ensure_ascii=False, indent=2)
            else:
                return json.dumps({"success": False, "error": f"不支持的格式: {format_type}"})

            self.detection_reports.append(filepath)

            return json.dumps(
                {"success": True, "message": f"报告已导出: {filepath}", "path": filepath}
            )

        except Exception as e:
            return json.dumps({"success": False, "error": f"导出报告失败: {str(e)}"})


class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像识别系统 - Python后端")
        self.setGeometry(100, 100, 1200, 800)

        # 创建Python后端
        self.backend = PythonBackend()

        # 创建Web视图
        self.web_view = QWebEngineView()

        # 使用自定义页面
        custom_page = CustomWebEnginePage()
        self.web_view.setPage(custom_page)

        # 设置Web引擎设置
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)

        # 设置Web通道
        self.channel = QWebChannel()
        self.channel.registerObject("pythonBackend", self.backend)
        custom_page.setWebChannel(self.channel)

        # 将页面引用传递给后端
        self.backend.web_page = custom_page

        # 设置中央部件
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.web_view)
        self.setCentralWidget(central_widget)

        # 加载前端页面
        self.load_frontend()

    def load_frontend(self):
        """加载前端页面"""
        try:
            # 尝试加载本地开发服务器
            self.web_view.load(QUrl("http://localhost:3000"))
            print("已加载本地开发服务器: http://localhost:3000")
        except Exception as e:
            print(f"加载前端页面失败: {e}")


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序属性
    app.setApplicationName("图像识别系统")
    app.setApplicationVersion("1.0")

    # 创建主窗口
    window = MainWindow()
    window.show()

    print("Python后端已启动")
    print("等待前端连接...")

    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
