import time
import cv2
import numpy as np
from utils import (
    adjust_boxes,
    annotate_xywh,
    compute_pixel_per_cm,
    correct_size_with_height,
    detects_background_change_by_cosine,
    init_undistort_maps,
    perspective_transform,
    undistort_image_fast,
)

# 设置阈值和常量
OBJ_THRESH, NMS_THRESH, SCORE = 0.1, 0.1, 0  # 将阈值调低以查看更多框
# 小中大，小的小于中的就是小的
HAMBURGER_SIZE = (10, 13)
CLASSES = (
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
)
COLOR_PALETTE = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# 参考区域四角像素坐标（左上、右上、右下、左下）
# SRC_POINTS = [[659, 331], [1417, 331], [1788, 860], [313, 860]]
# 参考区域四角像素坐标（左上、右上、右下、左下）
# SRC_POINTS = [[670, 330], [1417, 330], [1788, 885], [300, 885]]
# SRC_POINTS = [[650, 330], [1425, 330], [1830, 889], [230, 889]]
# SRC_POINTS = [[650, 230], [1425, 230], [1920, 889], [160, 889]]
SRC_POINTS = [510,270], [1170, 270], [1500, 710], [140, 710]
# 微波炉腔内大小 单位：厘米）
REAL_WIDTH_CM = 29

REAL_HEIGHT_CM = 18.5
# 目标高度
TARGET_HEIGHT = (
    1.8 + 0.9
)  # 面包高度 + 牛肉饼中心离面包表面高度（假设牛肉饼厚度约1cm，取一半）

CAMERA_HEIGHT = 13.0  # 相机到微波炉底部的垂直距离

# 焦距
FX = 1260.15 
FY = 1256.09 
# 相机内参矩阵（mtx）
CAMERA_MATRIX = np.array([
    [1260.15281, 0.0, 971.702426],
    [0.0, 1256.08744, 504.553169],
    [0.0, 0.0, 1.0]
])

# 畸变系数（dist） (k1, k2, p1, p2, k3)
DIST_COEFFS = np.array([[
    -0.430483648, 0.216393722,
    -0.000156465611, 0.000104551776,
    -0.0564557922
]])

# 旋转向量（rvec）- 使用第一组
RVEC = np.array([
    [-0.74272717],
    [-0.02205108],
    [-0.00466471]
])

# 平移向量（tvec）- 使用第一组
TVEC = np.array([
    [-8.83518051],
    [-4.21560336],
    [15.07577132]
])
# 1、计算像素比
PIXEL_PER_CM = compute_pixel_per_cm(SRC_POINTS, REAL_WIDTH_CM, REAL_HEIGHT_CM)
# 2、 计算输出图像大小（像素）
OUTPUT_WIDTH = int(REAL_WIDTH_CM * PIXEL_PER_CM)

OUTPUT_HEIGHT = int(REAL_HEIGHT_CM * PIXEL_PER_CM)
# 3、定义目标四角点
DST_PTS = np.array(
    [[0, 0], [OUTPUT_WIDTH, 0], [OUTPUT_WIDTH, OUTPUT_HEIGHT], [0, OUTPUT_HEIGHT]],
    dtype=np.float32,
)
# 替换为你相机分辨率（W, H）
h, w = 1080, 1920
image_size = (w, h)
# 先初始化畸变参数
map1, map2 = init_undistort_maps(CAMERA_MATRIX, DIST_COEFFS, image_size)

IMAGE_PATH = "./img/bg.jpg"  # 替换为你的图片路径
BG_IMAGE = cv2.imread(IMAGE_PATH)
BG_IMAGE = undistort_image_fast(BG_IMAGE, map1, map2)
BG_IMAGE = perspective_transform(
    BG_IMAGE, SRC_POINTS, DST_PTS, OUTPUT_WIDTH, OUTPUT_HEIGHT
)


class LetterBox:
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


def draw_detections(img, box, score, class_id):
    """
    Draws bounding boxes and labels on the input image based on the detected objects.

    Args:
        img: The input image to draw detections on.
        box: Detected bounding box.
        score: Corresponding detection score.
        class_id: Class ID for the detected object.

    Returns:
        None
    """

    # Extract the coordinates of the bounding box
    x1, y1, w, h = box

    # Retrieve the color for the class ID
    color = COLOR_PALETTE[class_id]

    # Draw the bounding box on the image
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

    # Create the label text with class name and score
    label = f"{CLASSES[class_id]}: {score:.2f}"

    # Calculate the dimensions of the label text
    (label_width, label_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    # Calculate the position of the label text
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

    # Draw a filled rectangle as the background for the label text
    cv2.rectangle(
        img,
        (int(label_x), int(label_y - label_height)),
        (int(label_x + label_width), int(label_y + label_height)),
        color,
        cv2.FILLED,
    )

    # Draw the label text on the image
    cv2.putText(
        img,
        label,
        (int(label_x), int(label_y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def draw(image, boxes, scores, classes):
    max_score = 0
    max_class = None
    t1 = time.time()
    # 检测背景变化 计算异物
    hash_diff, is_different = detects_background_change_by_cosine(
        image, BG_IMAGE, boxes
    )
    # print(f"⏱️ 检测背景变化耗时: {time.time() - t1:.3f}s")
    # print(f"是否一样: {hash_diff} ==== {is_different} ")
    if is_different:
        label_text = f"has foreign matter SCORE: {hash_diff} , {is_different} "
        cv2.putText(
            image,
            label_text,
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        return

    for box, score, cl in zip(boxes, scores, classes):
        if score >= SCORE:
            x1, y1, w, h = box
            # measure_and_annotate_xywh(image, box, PIXEL_PER_CM)
            # 你也可以将这个新尺寸添加到标签文本中
            # Retrieve the color for the class ID
            color = COLOR_PALETTE[cl]
            # Draw the bounding box on the image
            cv2.rectangle(
                image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2
            )

            # Create the label text with class name and score
            label = f"{CLASSES[cl]}: {score:.2f},x {int(x1)},y {int(y1)},w {int(w)},h {int(h)} "
            t1 = time.time()
            if CLASSES[cl] == "Hamburger":
                # 1. 使用初始检测框的坐标来裁剪出感兴趣区域（Region of Interest, ROI）。
                #    如果检测框太紧，可以给坐标添加一些缓冲（padding）。
                # 使用 map(int, ...) 将 box 中的所有值转换为整数
                # x2, y2, w2, h2 = map(int, box)
                # roi = image[y2 : y2 + h2, x2 : x2 + w2]
                # 2. 从物体的轮廓中获取精确的尺寸。
                # contour_w, contour_h = get_ellipse_diameter_and_draw(roi, PIXEL_PER_CM)
                w_pixel = w - x1
                image_points = [
                    (x1, y1),  # 左上
                    (w, y1),  # 右上
                    (w, h),  # 右下
                    (x1, h),  # 左下
                ]
                # 实际物体宽度
                contour_w, contour_h = annotate_xywh(box, PIXEL_PER_CM)
                # X, Y, Z, jl = compute_3d_distance(
                #     CAMERA_MATRIX, DIST_COEFFS, image_points, contour_w, contour_h
                # )
                jl = None

                if jl is None:
                    jl = 0.0
                contour_w = correct_size_with_height(
                    contour_w, TARGET_HEIGHT, CAMERA_HEIGHT
                )
                # 3. 使用这些更精确的尺寸进行测量。
                if contour_w and contour_h:
                    # 用轮廓尺寸替换你原来的尺寸计算
                    width_cm = contour_w  # / PIXEL_PER_CM
                    height_cm = contour_h  # / PIXEL_PER_CM
                    label_text = f"{width_cm:.1f}cm x {height_cm:.1f}cm,jl: {jl:.1f}cm "
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
                    if width_cm:
                        min_size, max_size = HAMBURGER_SIZE
                        if width_cm <= min_size:
                            size_label = "small"
                        elif width_cm < max_size:
                            size_label = "medium"
                        else:
                            size_label = "large"
                        label = f"{size_label} {CLASSES[cl]}: {score:.2f},x {int(x1)},y {int(y1)},w {int(w)},h {int(h)} "
                        print(
                            f"轮廓尺寸:{size_label} {CLASSES[cl]}: {width_cm:.2f} cm x {height_cm:.2f} cm，,x {int(x1)},y {int(y1)},w {int(w)},h {int(h)}"
                        )
            # print(f"⏱️ hamburger 耗时: {time.time() - t1:.3f}s")

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(
                image,
                (int(label_x), int(label_y - label_height)),
                (int(label_x + label_width), int(label_y + label_height)),
                color,
                cv2.FILLED,
            )
            # Draw the label text on the image
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
                max_class = CLASSES[cl]

    if max_class:
        print(f"Highest score: {max_score:.2f}, Class: {max_class}")
        # throttle_send(max_class)
    else:
        print("No boxes met the SCORE threshold.")


def postprocess(output):
    """
    对 YOLOv5 模型输出进行后处理，提取边界框、分数和类别ID，并应用NMS。

    Args:
        input_image (numpy.ndarray): 输入的图像。
        output (numpy.ndarray): YOLOv5 模型的输出。

    Returns:
        numpy.ndarray: 带有检测结果的输入图像。
    """
    output = output[0]
    output = output.T
    boxes = output[..., :4]
    scores = np.max(output[..., 4:], axis=1)
    class_ids = np.argmax(output[..., 4:], axis=1)
    # 应用 NMS 来去除重叠的框

    indices = cv2.dnn.NMSBoxes(boxes, scores, OBJ_THRESH, NMS_THRESH)
    # 打印 NMS 返回的 indices
    # print(f"NMS 后的 indices:\n{indices}")
    result_boxes = []
    result_scores = []
    result_class_ids = []
    # 检查 indices 是否为空
    if len(indices) > 0:
        for i in indices:
            # 不再需要 i[0]，因为 i 本身就是索引
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            result_boxes.append(box)
            result_scores.append(score)
            result_class_ids.append(class_id)
    return result_boxes, result_scores, result_class_ids


def preprocess(ori_img, size):
    """
    Preprocesses the input image before performing inference.

    Returns:
        image_data: Preprocessed image data ready for inference.
    """

    # Read the input image using OpenCV
    # self.img = cv2.imread(self.input_image)

    # print("image before", self.img)
    # Get the height and width of the input image
    letterbox = LetterBox(new_shape=size, auto=False, stride=32)
    image = letterbox(image=ori_img)
    image = [image]
    image = np.stack(image)
    image = image[..., ::-1].transpose((0, 3, 1, 2))
    img = np.ascontiguousarray(image)
    # n, h, w, c
    image = img.astype(np.float32)
    # 将处理后的图像转换回 HWC 格式，以便用 OpenCV 显示
    # processed_image_for_display = image[0].transpose(1, 2, 0)
    # processed_image_for_display = processed_image_for_display.astype(np.uint8)
    # 显示预处理后的图像
    # cv2.imshow("Processed Image", processed_image_for_display)
    # filename = time.strftime("%Y%m%d_%H%M%S") + "_warped.jpg"
    # cv2.imwrite(filename, processed_image_for_display)
    # print(f"✅ 已保存透视图：{filename}")
    return image / 255


def myFunc(interpreter, image):
    # 获取图片原始尺寸
    # ocl_frame = cv2.UMat(image)
    # ocl_frame = cv2.cvtColor(ocl_frame, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    # 获取输入和输出张量的详细信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # 获取量化参数（例如输入） 如果是float 需要注释掉
    input_scale, input_zero_point = input_details[0]["quantization"]
    output_scale, output_zero_point = output_details[0]["quantization"]
    # 获取输入张量的形状
    input_shape = input_details[0]["shape"]
    size = (input_shape[1], input_shape[2])
    t1 = time.time()

    # 畸变矫正
    # image = undistort_image(image, CAMERA_MATRIX, DIST_COEFFS, 1.0)
    # === 实时阶段：每帧图像处理 ===
    # 假设 frame 是当前帧图像
    image = undistort_image_fast(image, map1, map2)
    print(f"⏱️ 畸变矫正耗时: {time.time() - t1:.3f}s")
    # 调整图片大小以匹配输入张量的形状
    t1 = time.time()
    # 透视校正
    image = perspective_transform(
        image, SRC_POINTS, DST_PTS, OUTPUT_WIDTH, OUTPUT_HEIGHT
    )
    # print(f"⏱️ 透视校正耗时: {time.time() - t1:.3f}s")
    t1 = time.time()
    original_size = image.shape[:2]
    input_data = preprocess(image, size)
    # print(f"⏱️ 尺寸调整耗时: {time.time() - t1:.3f}s")
    input_data = input_data.transpose((0, 2, 3, 1))  # .astype(np.int8)
    t1 = time.time()
    # input_data = input_data.transpose((0, 2, 3, 1))
    # input_data = (input_data / input_scale + input_zero_point).astype(
    #     np.int8
    # )  # 如果是float 需要注释掉
    # 模型输入
    interpreter.set_tensor(input_details[0]["index"], input_data)
    # 执行推理
    interpreter.invoke()
    # 获取输出张量
    output_data = interpreter.get_tensor(output_details[0]["index"])
    # print(f"⏱️ 推理耗时: {time.time() - t1:.3f}s")
    # print(f"size===={size}")
    # print(f"output_data===={output_data}")
    # print(f"output_data[:, :4] ===={output_data[:, :4]}")
    # 反量化：将量化输出转换回浮点数
    # output_data = (
    #     output_data.astype(np.float32) - output_zero_point
    # ) * output_scale  # 如果是float 需要注释掉
    # print(f"pred将量化输出转换回浮点数[:, :4] ===={pred[:, :4]}")
    # print(f"original_size[:, :4] ===={ original_size[1]},{original_size[0]}")
    # 将预测框的坐标从归一化形式转换回原始图像尺寸
    output_data[:, [0, 2]] *= size[0]
    output_data[:, [1, 3]] *= size[1]
    # print(f"output_data[0, :, :10] ===={output_data[0, :, :10]}")
    result_boxes, result_scores, result_class_names = postprocess(output_data)
    output_data = None
    if len(result_boxes) > 0:
        result_boxes = adjust_boxes(
            size,
            np.array(result_boxes, dtype=np.float64),
            (original_size[1], original_size[0]),
        )
        draw(image, result_boxes, result_scores, result_class_names)
    return image
