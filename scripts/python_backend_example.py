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
                            width_cm = actual_w
                            height_cm = actual_h
                            mj = width_cm * height_cm

                            # 尺寸分类
                            width_min, width_max = self.hamburger_size
                            mj_min, mj_max = self.hamburger_mj

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

                            detection.update(
                                {
                                    "size": size_label,
                                    "width_cm": width_cm,
                                    "height_cm": height_cm,
                                    "area_cm2": mj,
                                }
                            )

                    detections.append(detection)

            return detections, image

        except Exception as e:
            print(f"检测过程出错: {e}")
            return [], image

    def set_background_image(self, image_path=None):
        """设置背景图像用于背景变化检测"""
        if image_path is None:
            image_path = self.background_image_path

        if os.path.exists(image_path):
            self.background_image = cv2.imread(image_path)
            print(f"背景图像已设置: {image_path}")
        else:
            # 如果没有背景图像，创建一个空的
            self.background_image = np.zeros((480, 640, 3), dtype=np.uint8)
            print("使用默认空背景图像")

    def update_detection_params(self, params: dict):
        """更新检测参数"""
        if "objThresh" in params:
            self.obj_thresh = params["objThresh"] / 100.0
        if "nmsThresh" in params:
            self.nms_thresh = params["nmsThresh"] / 100.0
        if "scoreThresh" in params:
            self.score_thresh = params["scoreThresh"] / 100.0

        # 透视变换参数
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

        # 畸变矫正参数
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
            self.dist_coeffs = np.array(
                [
                    [
                        params.get("distortionK1", -0.430483648) / 100.0,
                        params.get("distortionK2", 0.216393722) / 100.0,
                        params.get("distortionP1", -0.000156465611),
                        params.get("distortionP2", 0.000104551776),
                        params.get("distortionK3", -0.0564557922) / 100.0,
                    ]
                ]
            )
            # 重新初始化畸变矫正映射
            self.map1, self.map2 = init_undistort_maps(
                self.camera_matrix, self.dist_coeffs, self.image_size
            )

        # 相机内参更新
        if any(
            key in params
            for key in [
                "focalLengthX",
                "focalLengthY",
                "principalPointX",
                "principalPointY",
            ]
        ):
            self.camera_matrix = np.array(
                [
                    [
                        params.get("focalLengthX", 1260.15281),
                        0.0,
                        params.get("principalPointX", 971.702426),
                    ],
                    [
                        0.0,
                        params.get("focalLengthY", 1256.08744),
                        params.get("principalPointY", 504.553169),
                    ],
                    [0.0, 0.0, 1.0],
                ]
            )
            # 重新初始化畸变矫正映射
            self.map1, self.map2 = init_undistort_maps(
                self.camera_matrix, self.dist_coeffs, self.image_size
            )

        # 相机参数
        if "cameraHeight" in params:
            self.camera_height = params["cameraHeight"]
        if "targetHeight" in params:
            self.target_height = params["targetHeight"]

        # 尺寸分类参数
        if "hamburgerSizeMin" in params or "hamburgerSizeMax" in params:
            self.hamburger_size = (
                params.get("hamburgerSizeMin", 10),
                params.get("hamburgerSizeMax", 13),
            )

        # 实际尺寸参数
        if "realWidthCm" in params or "realHeightCm" in params:
            self.real_width_cm = params.get("realWidthCm", 29.0)
            self.real_height_cm = params.get("realHeightCm", 18.5)

            # 重新计算像素密度和目标点
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

        print(f"检测参数已更新: obj_thresh={self.obj_thresh}, nms_thresh={self.nms_thresh}")

    def update_recognition_settings(self, settings: dict):
        """更新识别设置"""
        # 检查依赖关系
        if not settings.get("distortionEnabled", True) and settings.get(
            "perspectiveEnabled", False
        ):
            print("警告: 透视变换需要先启用畸变矫正")
            settings["perspectiveEnabled"] = False

        if not settings.get("perspectiveEnabled", False):
            # 如果透视变换未启用，禁用依赖功能
            if settings.get("sizeClassification", False):
                print("警告: 尺寸分类需要先启用透视变换")
                settings["sizeClassification"] = False
            if settings.get("heightCorrection", False):
                print("警告: 高度补偿需要先启用透视变换")
                settings["heightCorrection"] = False
            if settings.get("foreignObjectDetection", False):
                print("警告: 异物检测需要先启用透视变换")
                settings["foreignObjectDetection"] = False

        self.recognition_settings.update(settings)
        print(f"识别设置已更新: {self.recognition_settings}")


class PythonBackend(QObject):
    """Python后端类，处理与前端的通信"""

    def __init__(self):
        super().__init__()
        self.camera = None
        self.is_streaming = False
        self.current_frame = None
        self.detector = YOLODetector()
        self.frame_queue = Queue(maxsize=2)
        self.detection_results = []
        self.executor = ThreadPoolExecutor(max_threads=2)

        # 性能监控
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # 图像处理参数
        self.image_params = {
            "contrast": 100,
            "brightness": 100,
            "saturation": 100,
            "blur": 0,
            "delaySeconds": 1.0,
        }

        # 定时器用于推送数据到前端
        self.push_timer = QTimer()
        self.push_timer.timeout.connect(self.push_data_to_frontend)
        self.push_timer.start(100)  # 每100ms推送一次

    def apply_image_adjustments(self, frame):
        """应用图像调整参数"""
        if frame is None:
            return None

        # 对比度和亮度调整
        contrast = self.image_params["contrast"] / 100.0
        brightness = self.image_params["brightness"] - 100

        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

        # 饱和度调整
        saturation = self.image_params["saturation"] / 100.0
        if saturation != 1.0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 模糊处理
        blur_value = self.image_params["blur"]
        if blur_value > 0:
            kernel_size = int(blur_value * 2) + 1
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

        return frame

    def camera_thread(self):
        """摄像头线程函数"""
        delay_seconds = self.image_params.get("delaySeconds", 1.0)

        while self.is_streaming and self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                # 应用图像调整
                processed_frame = self.apply_image_adjustments(frame)

                # 将帧放入队列（非阻塞）
                if not self.frame_queue.full():
                    self.frame_queue.put(processed_frame)

                # FPS计算
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = current_time

                # 延迟控制
                time.sleep(delay_seconds)
            else:
                print("无法读取摄像头帧")
                break

    def detection_thread(self):
        """检测线程函数"""
        while self.is_streaming:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    if frame is not None:
                        # 执行检测
                        detections, annotated_frame = self.detector.detect_objects(frame)
                        self.detection_results = detections
                        self.current_frame = annotated_frame
                else:
                    time.sleep(0.01)  # 短暂休眠避免CPU占用过高
            except Exception as e:
                print(f"检测线程错误: {e}")
                time.sleep(0.1)

    def push_data_to_frontend(self):
        """推送数据到前端"""
        if self.current_frame is not None:
            # 将OpenCV图像转换为base64
            _, buffer = cv2.imencode(".jpg", self.current_frame)
            frame_base64 = buffer.tobytes().hex()

            # 构建推送数据
            push_data = {
                "type": "frame_update",
                "data": {
                    "frame": frame_base64,
                    "detections": self.detection_results,
                    "fps": self.current_fps,
                    "timestamp": time.time(),
                },
            }

            # 推送到前端
            self.push_to_frontend(json.dumps(push_data))

    @pyqtSlot(str, result=str)
    def push_to_frontend(self, data):
        """推送数据到前端（由QTimer调用）"""
        # 这个方法会被前端的JavaScript轮询调用
        return data

    @pyqtSlot(result=str)
    def get_system_status(self):
        """获取系统状态"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            status = {
                "success": True,
                "data": {
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory_percent,
                    "fps": self.current_fps,
                    "is_streaming": self.is_streaming,
                    "model_loaded": self.detector.interpreter is not None,
                    "detection_count": len(self.detection_results),
                },
            }
            return json.dumps(status)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    @pyqtSlot(str, result=str)
    def start_camera(self, camera_index):
        """启动摄像头"""
        try:
            if self.is_streaming:
                self.stop_camera()

            # 尝试打开摄像头
            camera_idx = int(camera_index)
            self.camera = cv2.VideoCapture(camera_idx)

            if not self.camera.isOpened():
                return json.dumps({"success": False, "error": f"无法打开摄像头 {camera_idx}"})

            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

            self.is_streaming = True

            # 启动摄像头线程
            self.camera_thread_obj = threading.Thread(target=self.camera_thread)
            self.camera_thread_obj.daemon = True
            self.camera_thread_obj.start()

            # 启动检测线程
            self.detection_thread_obj = threading.Thread(target=self.detection_thread)
            self.detection_thread_obj.daemon = True
            self.detection_thread_obj.start()

            return json.dumps({"success": True, "message": f"摄像头 {camera_idx} 启动成功"})

        except Exception as e:
            return json.dumps({"success": False, "error": f"启动摄像头失败: {str(e)}"})

    @pyqtSlot(result=str)
    def stop_camera(self):
        """停止摄像头"""
        try:
            self.is_streaming = False

            if self.camera is not None:
                self.camera.release()
                self.camera = None

            # 清空队列
            while not self.frame_queue.empty():
                self.frame_queue.get()

            self.current_frame = None
            self.detection_results = []

            return json.dumps({"success": True, "message": "摄像头已停止"})

        except Exception as e:
            return json.dumps({"success": False, "error": f"停止摄像头失败: {str(e)}"})

    @pyqtSlot(str, result=str)
    def load_model(self, model_path):
        """加载模型"""
        try:
            success = self.detector.load_model(model_path)
            if success:
                return json.dumps({"success": True, "message": f"模型加载成功: {model_path}"})
            else:
                return json.dumps({"success": False, "error": "模型加载失败"})
        except Exception as e:
            return json.dumps({"success": False, "error": f"加载模型失败: {str(e)}"})

    @pyqtSlot(str, result=str)
    def update_image_params(self, params_json):
        """更新图像参数"""
        try:
            params = json.loads(params_json)
            self.image_params.update(params)

            # 更新检测器参数
            self.detector.update_detection_params(params)

            return json.dumps({"success": True, "message": "图像参数已更新"})
        except Exception as e:
            return json.dumps({"success": False, "error": f"更新图像参数失败: {str(e)}"})

    @pyqtSlot(str, result=str)
    def update_recognition_settings(self, settings_json):
        """更新识别设置"""
        try:
            settings = json.loads(settings_json)
            self.detector.update_recognition_settings(settings)
            return json.dumps({"success": True, "message": "识别设置已更新"})
        except Exception as e:
            return json.dumps(
                {"success": False, "error": f"更新识别设置失败: {str(e)}"}
            )

    @pyqtSlot(result=str)
    def save_image(self):
        """保存当前图像"""
        try:
            if self.current_frame is None:
                return json.dumps({"success": False, "error": "没有可保存的图像"})

            # 创建保存目录
            save_dir = "./saved_images"
            os.makedirs(save_dir, exist_ok=True)

            # 生成文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)

            # 保存图像
            cv2.imwrite(filepath, self.current_frame)

            return json.dumps(
                {"success": True, "message": f"图像已保存: {filepath}"}
            )

        except Exception as e:
            return json.dumps({"success": False, "error": f"保存图像失败: {str(e)}"})

    @pyqtSlot(str, result=str)
    def export_report(self, format_type):
        """导出检测报告"""
        try:
            if not self.detection_results:
                return json.dumps({"success": False, "error": "没有检测结果可导出"})

            # 创建报告目录
            report_dir = "./reports"
            os.makedirs(report_dir, exist_ok=True)

            # 生成文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"detection_report_{timestamp}.{format_type}"
            filepath = os.path.join(report_dir, filename)

            # 准备报告数据
            report_data = {
                "timestamp": timestamp,
                "total_detections": len(self.detection_results),
                "detections": self.detection_results,
                "system_info": {
                    "fps": self.current_fps,
                    "model_path": self.detector.model_path,
                },
            }

            # 根据格式保存
            if format_type == "json":
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(report_data, f, ensure_ascii=False, indent=2)
            else:
                return json.dumps({"success": False, "error": f"不支持的格式: {format_type}"})

            return json.dumps(
                {"success": True, "message": f"报告已导出: {filepath}"}
            )

        except Exception as e:
            return json.dumps({"success": False, "error": f"导出报告失败: {str(e)}"})


class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像识别系统 - Python后端")
        self.setGeometry(100, 100, 1200, 800)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 创建WebEngine视图
        self.web_view = QWebEngineView()

        # 使用自定义页面
        custom_page = CustomWebEnginePage()
        self.web_view.setPage(custom_page)

        # 启用开发者工具
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.ErrorPageEnabled, True)
        settings.setAttribute(QWebEngineSettings.PluginsEnabled, True)

        # 创建Python后端
        self.backend = PythonBackend()

        # 设置WebChannel
        self.channel = QWebChannel()
        self.channel.registerObject("backend", self.backend)
        custom_page.setWebChannel(self.channel)

        layout.addWidget(self.web_view)

        # 加载前端页面
        self.load_frontend()

    def load_frontend(self):
        """加载前端页面"""
        # 这里需要根据实际情况修改URL
        # 如果是开发环境，使用 http://localhost:3000
        # 如果是生产环境，使用本地文件路径
        frontend_url = "http://localhost:3000"  # Next.js开发服务器
        self.web_view.load(QUrl(frontend_url))

    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止摄像头
        self.backend.stop_camera()
        event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序信息
    app.setApplicationName("图像识别系统")
    app.setApplicationVersion("1.0")

    # 创建主窗口
    window = MainWindow()
    window.show()

    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
