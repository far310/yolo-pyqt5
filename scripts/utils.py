import time
from PIL import Image
import os
import cv2
import numpy as np
import imagehash
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity


# ==== 反投影函数 ====
def project_to_world(u, v, mtx, dist, rvec, tvec):
    """
    利用完整的相机内外参反投影重建真实尺寸（推荐）
    将像素坐标 (u, v) 反投影到世界坐标 Z=0 平面
    """
    uv = np.array([[[u, v]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(uv, mtx, dist, P=mtx)
    uv_norm = undistorted[0][0]
    uv_dir = np.linalg.inv(mtx) @ np.array([uv_norm[0], uv_norm[1], 1.0])

    R, _ = cv2.Rodrigues(rvec)
    R_inv = np.linalg.inv(R)
    tvec = tvec.reshape(3, 1)

    s = -tvec[2][0] / (R_inv @ uv_dir)[2]
    world_point = R_inv @ (s * uv_dir - tvec)
    return world_point


# ==== 人工标定拟合法 ====
def try_alpha(alpha):
    """
    基于假设的修正（简单，但精度有限）
    人工标定拟合法
        1.	拿 5~10 个你认识宽度的目标物（比如汉堡宽度是 10cm）；
        2.	每个目标你都记录下这三样数据：
        •	真实宽度（cm）（你用尺子量的）
        •	鸟瞰图中测量宽度（你算出来的 measured_width）
        •	bbox 高度（像素）（来自图像）
        3.	然后用这些数据去拟合 alpha 使得
    """
    total_error = 0
    data = [
        (100, 18.0, 10.0),
        (80, 16.0, 11.0),
        (120, 20.0, 9.5),
    ]
    for h, mw, rw in data:
        corrected = mw / (1 + alpha * h)
        error = abs(corrected - rw)
        total_error += error
    #  # 从0.001到0.02之间试一圈
    # best_alpha = min([round(a, 4) for a in np.linspace(0.001, 0.02, 100)],
    #                 key=try_alpha)
    # print(f"最合适的 alpha 是: {best_alpha}")
    return total_error


# ==== 高度补偿尺寸识别计算 ====
def correct_size_with_height(measured_size, target_height, camera_height):
    """
    高度补偿计算公式
    # measured_size：透视变换后测量得到的尺寸（单位：cm）
    # target_height：目标物体离参考平面的高度（单位：cm），如面包高度 + 牛肉饼厚度/2
    # camera_height：相机到参考平面的垂直距离（单位：cm）
    """
    # 计算补偿比例
    correction_ratio = 1 / (1 + (target_height / camera_height))

    # 实际尺寸补偿
    corrected_size = measured_size * correction_ratio

    return corrected_size


# ==== 标定相机焦距（单位：像素） ====
def calibrate_focal_length(w_pixel, real_distance_cm, real_width_cm):
    """
    标定相机焦距（单位：像素）
    参数:
        w_pixel: 标定图片中目标的像素宽度
        real_distance_cm: 拍照时实际相机到目标的距离（单位：cm）
        real_width_cm: 目标实际宽度（单位：cm）
    返回:
        相机等效焦距 f（单位：像素）
    """
    f = (w_pixel * real_distance_cm) / real_width_cm
    return f


# ==== 单目测距 ====
def estimate_distance(focal_length, real_width_cm, w_pixel):
    """
    单目测距
    参数:
        focal_length: 相机焦距（像素）
        real_width_cm: 目标实际宽度（cm）
        w_pixel: 检测框像素宽度（px）
    返回:
        估算距离（单位：cm）
    """
    if w_pixel <= 0:
        return None  # 避免除0错误
    distance = (focal_length * real_width_cm) / w_pixel
    return distance


# ==== 修正鸟瞰图中因高度引起的宽度测量误差 ====
def correct_width(measured_width_cm, bbox_height_px, alpha):
    """
    基于假设的修正（简单，但精度有限）
    修正鸟瞰图中因高度引起的宽度测量误差。
    :param measured_width_cm: 通过鸟瞰图计算出的宽度（单位：cm）
    :param bbox_height_px: 在原始图像中，该目标的边界框高度（单位：像素）
    :param alpha: 经验系数，用于控制修正强度
    :return: 修正后的物理宽度（单位：cm）
    """
    corrected = measured_width_cm / (1 + alpha * bbox_height_px)
    return corrected


# ==== 图片背景对比算法 多框 ====
def detects_background_change_by_ssim(
    frame: np.ndarray,
    background: np.ndarray,
    detected_boxes: list,
    ssim_threshold: float = 0.94,
    resize_to: int = 120,
):
    """
    使用 SSIM 检测背景是否发生变化（如异物）

    参数:
    - frame: 当前帧图像（BGR）
    - background: 背景图像（BGR）
    - detected_boxes: list of (x, y, w, h)，多个目标框（排除检测区域）
    - ssim_threshold: SSIM 相似性阈值（越低越敏感）
    - resize_to: 图像缩放尺寸（默认 480）

    返回:
    - ssim_score: SSIM 分数（1 表示完全相同）
    - is_different: True 表示检测到差异，False 表示无异常
    - diff_mask: 差异热力图（可视化用）
    """

    if frame.shape != background.shape:
        raise ValueError("frame 和 background 图像尺寸不一致")

    # 创建全白 mask，排除所有检测框
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255

    for box in detected_boxes:
        x, y, w, h = map(int, box)
        x1 = max(0, x - 10)
        y1 = max(0, y - 10)
        x2 = min(frame.shape[1], x + w + 10)
        y2 = min(frame.shape[0], y + h + 10)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

    # 灰度图 + 掩码
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.bitwise_and(gray1, gray1, mask=mask)
    gray2 = cv2.bitwise_and(gray2, gray2, mask=mask)

    # 二值化
    _, gray1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, gray2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 缩放
    gray1 = cv2.resize(gray1, (resize_to, resize_to))
    gray2 = cv2.resize(gray2, (resize_to, resize_to))

    # 计算 SSIM 差异图
    score = ssim(
        gray1,
        gray2,
        full=False,
        channel_axis=-1,  # 多通道图像
        win_size=11,  # 控制感知尺度
        gaussian_weights=True,  # 使用高斯加权
        sigma=1.5,  # 平滑程度
    )
    # diff_mask = (1 - diff_map) * 255
    # diff_mask = diff_mask.astype(np.uint8)

    is_different = score < ssim_threshold

    return score, is_different


# ==== 图片背景对比算法 ====
def detect_background_change_by_ssim(
    frame: np.ndarray,
    background: np.ndarray,
    detected_boxes: list,
    ssim_threshold: float = 0.94,
    resize_to: int = 480,
):
    """
    使用 SSIM 检测背景是否发生变化（如异物）

    参数:
    - frame: 当前帧图像（BGR）
    - background: 背景图像（BGR）
    - detected_boxes: list of (x, y, w, h)，目标区域（排除检测区域）
    - ssim_threshold: SSIM 相似性阈值（越低越敏感）
    - resize_to: 图像缩放尺寸（默认 256）

    返回:
    - ssim_score: SSIM 分数（1 表示完全相同）
    - is_different: True 表示检测到差异，False 表示无异常
    - diff_mask: 差异热力图（可视化用）
    """

    if frame.shape != background.shape:
        raise ValueError("frame 和 background 图像尺寸不一致")

    # 创建排除目标区域的 mask
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    x, y, w, h = detected_boxes
    x, y, w, h = int(x), int(y), int(w), int(h)
    # 上下左右各扩展20像素
    x_left_expanded = max(0, x - 10)
    x_right_expanded = min(frame.shape[1], x + w + 10)
    y_top_expanded = max(0, y - 10)
    y_bottom_expanded = min(frame.shape[0], y + h + 10)
    # 绘制矩形
    cv2.rectangle(
        mask,
        (x_left_expanded, y_top_expanded),
        (x_right_expanded, y_bottom_expanded),
        0,
        -1,
    )

    # 灰度图 + 掩码
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.bitwise_and(gray1, gray1, mask=mask)
    gray2 = cv2.bitwise_and(gray2, gray2, mask=mask)
    _, gray1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, gray2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 缩放
    gray1 = cv2.resize(gray1, (resize_to, resize_to))
    gray2 = cv2.resize(gray2, (resize_to, resize_to))

    # 计算 SSIM 差异图
    score, diff_map = ssim(
        gray1,
        gray2,
        full=True,
        channel_axis=-1,  # 多通道图像
        win_size=11,  # 控制感知尺度
        gaussian_weights=True,  # 使用高斯加权
        sigma=1.5,  # 平滑程度
    )
    diff_mask = (1 - diff_map) * 255  # 反向处理成"差异强度"
    diff_mask = diff_mask.astype(np.uint8)

    is_different = score < ssim_threshold

    return score, is_different


# ==== 图片背景对比算法 使用灰度余弦相似度检测背景是否发生变化，速度快，精度高====
def detects_background_change_by_cosine(
    frame: np.ndarray,
    background: np.ndarray,
    detected_boxes: list,
    similarity_threshold: float = 0.98,
    resize_to: int = 120,
):
    """
    使用灰度余弦相似度检测背景是否发生变化（如异物）

    参数：
    - frame, background: 当前帧和背景图（BGR）
    - detected_boxes: [(x,y,w,h), ...] 目标框（排除区域）
    - similarity_threshold: 相似度阈值（越低越敏感）
    - resize_to: 比较前统一缩放大小

    返回：
    - similarity: 0~1，相似度得分
    - is_different: True 表示检测到差异
    """
    if frame.shape != background.shape:
        raise ValueError("图像尺寸不一致")

    # Step 1: 构建遮罩
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255

    for box in detected_boxes:
        x, y, w, h = map(int, box)
        x1 = max(0, x - 10)
        y1 = max(0, y - 10)
        x2 = min(frame.shape[1], x + w + 10)
        y2 = min(frame.shape[0], y + h + 10)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

    # Step 2: 灰度 + 掩码
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.bitwise_and(gray1, gray1, mask=mask)
    gray2 = cv2.bitwise_and(gray2, gray2, mask=mask)
    gray1 = remove_specular_reflection(gray1)
    gray2 = remove_specular_reflection(gray2)

    # Step 3: 缩放
    gray1 = cv2.resize(gray1, (resize_to, resize_to))
    gray2 = cv2.resize(gray2, (resize_to, resize_to))

    # Step 4: 扁平化 + 计算余弦相似度
    vec1 = gray1.flatten().reshape(1, -1)
    vec2 = gray2.flatten().reshape(1, -1)
    similarity = cosine_similarity(vec1, vec2)[0][0]

    is_different = similarity < similarity_threshold
    return similarity, is_different


def remove_specular_reflection(
    gray: np.ndarray, threshold: int = 210, kernel_size: int = 5
):
    """
    去除灰度图中的反光区域（高亮区）

    参数:
    - gray: 输入灰度图 (np.ndarray)
    - threshold: 提取反光区域的亮度阈值（默认240）
    - kernel_size: 替代反光区域的中值滤波核大小

    返回:
    - gray_clean: 去除反光后的图像
    """
    # 1. 提取反光区域（亮度 > threshold）
    mask = cv2.inRange(gray, threshold, 255)

    # 2. 扩展反光区域（可选，提高鲁棒性）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)

    # 3. 中值滤波整个图像
    median_filtered = cv2.medianBlur(gray, kernel_size)

    # 4. 用非反光区域的像素替换
    gray_clean = gray.copy()
    gray_clean[mask_dilated > 0] = median_filtered[mask_dilated > 0]

    return gray_clean


# ==== 计算实际像素比 ====
def compute_pixel_per_cm(src_pts, real_width_cm, real_height_cm):
    """
    计算每厘米像素数（像素密度）。

    参数:
        src_pts: 四个角点像素坐标，顺序为 [左上, 右上, 右下, 左下]
        real_width_cm: 实际宽度 (单位: cm)
        real_height_cm: 实际高度 (单位: cm)

    返回:
        平均每厘米的像素数量 (float)
    """
    # 计算像素距离
    ref_width_px = np.linalg.norm(
        np.array(src_pts[1]) - np.array(src_pts[0])
    )  # 宽度（上边）
    ref_height_px = np.linalg.norm(
        np.array(src_pts[3]) - np.array(src_pts[0])
    )  # 高度（左边）

    # 计算水平方向和垂直方向的像素密度
    pixel_per_cm_w = ref_width_px / real_width_cm
    pixel_per_cm_h = ref_height_px / real_height_cm

    # 取平均更稳定
    pixel_per_cm = (pixel_per_cm_w + pixel_per_cm_h) / 2.0
    return pixel_per_cm


# ==== 透视变换 ====
def compute_perspective_transform(
    img, src_pts, real_width_cm, real_height_cm, save_result=False
):
    """
    执行透视变换，并返回变换结果与像素密度。

    参数:
        img: 输入图像 (numpy.ndarray)
        src_pts: 四个参考点坐标 (顺序：左上、右上、右下、左下)
        real_width_cm: 参考区域的实际宽度（cm）
        real_height_cm: 参考区域的实际高度（cm）
        save_result: 是否保存转换后的图像为文件（默认保存）

    返回:
        M: 透视变换矩阵
        output_width: 变换后图像宽度（px）
        output_height: 变换后图像高度（px）
        pixel_per_cm: 每厘米对应的像素数（float）
        warped: 透视变换后的图像（numpy.ndarray）
    """
    if img is None:
        print("❌ 图像为空，无法执行透视变换")
        return None, 0, 0, 0, None
    h, w = img.shape[:2]
    img_copy = img.copy()
    src_pts = np.array(src_pts, dtype=np.float32)

    # 1. 计算像素密度
    pixel_per_cm = compute_pixel_per_cm(src_pts, real_width_cm, real_height_cm)

    # 2. 计算输出图像大小（像素）
    output_width = int(real_width_cm * pixel_per_cm)
    output_height = int(real_height_cm * pixel_per_cm)

    # 3. 定义目标四角点
    dst_pts = np.array(
        [[0, 0], [output_width, 0], [output_width, output_height], [0, output_height]],
        dtype=np.float32,
    )

    # 4. 获取透视矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 5. 执行透视变换
    warped = cv2.warpPerspective(img_copy, M, (output_width,output_height))

    # 6. 保存结果
    if save_result:
        filename = time.strftime("%Y%m%d_%H%M%S") + "_warped.jpg"
        cv2.imwrite(filename, warped)
        print(f"✅ 已保存透视图：{filename}")

    return M, output_width, output_height, pixel_per_cm, warped


# ==== 图像去畸变，初始化先获取参数 ====
def init_undistort_maps(mtx, dist, image_size, alpha=1.0):
    """
    初始化去畸变映射表（只调用一次）

    参数：
    - mtx: 相机内参
    - dist: 畸变系数
    - image_size: (w, h)，图像分辨率
    - alpha: 保留图像比例（0 = 裁剪，1 = 保留黑边）

    返回：
    - map1, map2: 映射矩阵，用于快速 remap
    """
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, image_size, alpha)
    map1, map2 = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, image_size, cv2.CV_16SC2
    )
    return map1, map2


# ==== 图像去畸变，再矫正 ====
def undistort_image_fast(img, map1, map2):
    """
    使用预计算的映射表进行快速图像去畸变
    """
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)


# ==== 在图像上绘制半透明棋盘 ====
def draw_checkerboard(image, cols=7, rows=8, alpha=0.3, color=(255, 255, 255)):
    """
    在图像上绘制半透明棋盘。
    :param image: 输入图像
    :param cols: 列数
    :param rows: 行数
    :param alpha: 棋盘半透明度
    :param color: 棋盘颜色 (BGR)
    """
    h, w = image.shape[:2]
    cell_w = w // cols
    cell_h = h // rows
    grid = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                top_left = (j * cell_w, i * cell_h)
                bottom_right = ((j + 1) * cell_w, (i + 1) * cell_h)
                cv2.rectangle(grid, top_left, bottom_right, color, -1)

    overlay = cv2.addWeighted(image, 1.0, grid, alpha, 0)
    return overlay


# ==== 透视变换 少参数====
def perspective_transform(img, src_pts, dst_pts, output_width, output_height):
    """
    执行透视变换，并返回变换结果与像素密度。

    参数:
        img: 输入图像 (numpy.ndarray)
        src_pts: 四个参考点坐标 (顺序：左上、右上、右下、左下)
        dst_pts: 目标四角点
        output_width: 输出宽度
        output_height: 输出高度

    返回:
        warped: 透视变换后的图像（numpy.ndarray）
    """
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


# ==== 根据后处理得到的坐标，来识别尺寸，返回尺寸 ====
def annotate_xywh(boxes, pixel_per_cm):
    x, y, w, h = boxes
    # 换算实际尺寸（cm）
    actual_w = w / pixel_per_cm
    actual_h = h / pixel_per_cm
    return actual_w, actual_h


# ==== 根据输入图像的大小和原始图像的尺寸，调整边界框的位置和大小。 ====
def adjust_boxes(size, boxes, original_size):
    """
    根据输入图像的大小和原始图像的尺寸，调整边界框的位置和大小。

    Args:
        size (tuple): 当前输入图像的尺寸 (宽度, 高度)。
        result_boxes: 检测出的边界框，格式为 [x_center, y_center, width, height]。
        original_size (tuple): 原始图像的尺寸 (宽度, 高度)。

    Returns:
        numpy.ndarray: 调整后的边界框，格式为 [x_min, y_min, x_max, y_max]。
    """
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
