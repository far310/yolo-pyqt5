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
    diff_mask = (1 - diff_map) * 255  # 反向处理成“差异强度”
    diff_mask = diff_mask.astype(np.uint8)

    is_different = score < ssim_threshold

    return score, is_different


# ==== 图片背景对比算法，效果差 ====
def detect_background_change_from_image(
    frame: np.ndarray,
    background: np.ndarray,
    detected_boxes: list,
    phash_size: int = 480,
    threshold: int = 10,
):
    """
    检测背景图像是否发生变化（如有异物）

    参数:
    - frame: 当前帧图像（BGR 格式）
    - background: 背景图像（BGR 格式）
    - detected_boxes: list of (x, y, w, h)，目标区域（排除区域）
    - phash_size: 缩放尺寸（默认 256 x 256）
    - threshold: pHash 差异阈值（越大越宽松）

    返回:
    - hash_diff: pHash 差异值
    - is_different: True 表示背景发生变化，False 表示无异常
    """

    if frame is None or background is None:
        raise ValueError("图像不能为空")

    if frame.shape != background.shape:
        raise ValueError("frame 和 background 图像尺寸不一致")

    # 创建遮罩：目标区域为黑，其他区域为白
    # 创建遮罩并绘制扩展后的矩形
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
    # 应用遮罩
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    masked_bg = cv2.bitwise_and(background, background, mask=mask)

    # 转灰度并缩放为 PIL 图像
    def prepare(img, type):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (5, 5),0)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        resized = cv2.resize(gray, (phash_size, phash_size))
        # if type==1:
        # cv2.imwrite("masked_frame.jpg", resized)
        # else:
        # cv2.imwrite("masked_bg.jpg", resized)
        return Image.fromarray(resized)

    img1 = prepare(masked_frame, 1)
    img2 = prepare(masked_bg, 2)

    # 计算 pHash
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)
    hash_diff = hash1 - hash2

    is_different = hash_diff > threshold

    return hash_diff, is_different


# ==== 图片背景对比算法 加了裁剪范围，效果差 ====
def compare_background_phash(
    frame,
    background,
    detected_boxes=[],
    region_points=None,
    phash_size=256,
    threshold=6,
):
    """
    对比当前图像与背景图像在指定区域（可多边形裁剪）、排除目标框后的差异（pHash）

    参数：
    - frame: 当前图像（np.ndarray）
    - background: 背景图像（np.ndarray）
    - detected_boxes: [(x, y, w, h), ...] 需要排除的目标框（黑色遮罩）
    - region_points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] 区域四角点（左上、右上、右下、左下）
    - phash_size: pHash 缩放尺寸（默认 256）
    - threshold: 差异判断阈值

    返回：
    - hash_diff: 差异值
    - is_different: True 表示检测到变化
    """
    if frame.shape != background.shape:
        raise ValueError("图像尺寸不一致")

    h_img, w_img = frame.shape[:2]

    # 1. 如果给定 region_points，裁剪出该区域（不做透视）
    if region_points is not None:
        mask_region = np.zeros((h_img, w_img), dtype=np.uint8)
        region_np = np.array([region_points], dtype=np.int32)
        cv2.fillPoly(mask_region, region_np, 255)
        frame = cv2.bitwise_and(frame, frame, mask=mask_region)
        background = cv2.bitwise_and(background, background, mask=mask_region)

    # 2. 构建遮罩：先全白，然后在目标区域内画黑色矩形
    mask = np.ones((h_img, w_img), dtype=np.uint8) * 255
    for x, y, w, h in detected_boxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)

    if region_points is not None:
        mask = cv2.bitwise_and(mask, mask_region)

    # 3. 应用遮罩到两张图
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    masked_bg = cv2.bitwise_and(background, background, mask=mask)

    # 4. 准备感知哈希输入
    def prepare(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (phash_size, phash_size))
        return Image.fromarray(resized)

    img1 = prepare(masked_frame)
    img2 = prepare(masked_bg)

    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)

    hash_diff = hash1 - hash2
    is_different = hash_diff > threshold

    return hash_diff, is_different


# ==== 图片背景对比算法 phash 优化后 ====
def compares_background_phash(
    frame: np.ndarray,
    background: np.ndarray,
    detected_boxes: list,
    hash_size: int = 64,
    similarity_threshold: float = 0.65,
    resize_to: int = 256,
):
    """
    使用 pHash 对比图像与背景图的差异（支持多目标遮挡排除）

    参数：
    - frame: 当前帧图像（np.ndarray, BGR）
    - background: 背景图像（np.ndarray, BGR）
    - detected_boxes: [(x, y, w, h), ...] 要排除的目标框
    - hash_size: pHash 哈希大小（越大越精细，默认8，实际图像将被缩放至 hash_size*4）
    - similarity_threshold: 相似度阈值（0~1，越低越敏感）
    - resize_to: 中间处理时灰度图缩放尺寸

    返回：
    - similarity: 0~1 相似度得分，1 表示完全相同
    - is_different: 是否不同（即相似度低于阈值）
    """
    print(f"我就来了0")
    if frame.shape != background.shape:
        raise ValueError("frame 和 background 图像尺寸不一致")
    print(f"我就来了-1")
    # Step 1: 创建遮罩，排除目标区域
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    for box in detected_boxes:
        x, y, w, h = map(int, box)
        x1 = max(0, x - 10)
        y1 = max(0, y - 10)
        x2 = min(frame.shape[1], x + w + 10)
        y2 = min(frame.shape[0], y + h + 10)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
    print(f"我就来了1")
    # Step 2: 灰度 + 掩码
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.bitwise_and(gray1, gray1, mask=mask)
    gray2 = cv2.bitwise_and(gray2, gray2, mask=mask)
    print(f"我就来了2")
    # Step 3: 缩放
    # _, gray1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # _, gray2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    gray1 = cv2.resize(gray1, (resize_to, resize_to))
    gray2 = cv2.resize(gray2, (resize_to, resize_to))
    print(f"我就来了3")
    # Step 4: 转为 PIL.Image 后计算哈希
    img1 = Image.fromarray(gray1)
    img2 = Image.fromarray(gray2)
    print(f"我就来了4")
    hash1 = imagehash.phash(img1, hash_size=hash_size)
    hash2 = imagehash.phash(img2, hash_size=hash_size)
    print(f"我就来了5")
    # Step 5: 计算相似度（0~1）
    hamming_dist = hash1 - hash2
    max_bits = hash_size * hash_size
    similarity = 1 - (hamming_dist / max_bits)
    is_different = similarity < similarity_threshold

    return similarity, is_different


# ==== 图片背景对比算法 absdiff ====
def detects_background_change_by_absdiff_umat(
    frame: np.ndarray,
    background: np.ndarray,
    detected_boxes: list,
    diff_threshold: float = 25.0,
    pixel_change_ratio: float = 0.01,
    resize_to: int = 120,
):
    """
    使用绝对差异检测背景变化（支持 cv2.UMat 以加速）

    参数：
    - frame: 当前帧图像（BGR）
    - background: 背景图像（BGR）
    - detected_boxes: list of (x, y, w, h)
    - diff_threshold: 差异强度阈值（像素级）
    - pixel_change_ratio: 差异像素比例阈值
    - resize_to: 缩放后边长

    返回：
    - diff_ratio: 差异像素比例（0~1）
    - is_different: 是否检测到变化
    """

    if frame.shape != background.shape:
        raise ValueError("frame 和 background 图像尺寸不一致")

    # 判断是否支持 OpenCL
    use_umat = cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL()

    # 创建全白 mask，排除所有目标框（使用普通 NumPy）
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    for x, y, w, h in detected_boxes:
        x1, y1 = max(0, x - 10), max(0, y - 10)
        x2, y2 = min(frame.shape[1], x + w + 10), min(frame.shape[0], y + h + 10)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

    # 灰度图（UMat 或普通）
    gray1 = cv2.cvtColor(cv2.UMat(frame) if use_umat else frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(
        cv2.UMat(background) if use_umat else background, cv2.COLOR_BGR2GRAY
    )

    # bitwise_and 掩码（mask 是 numpy 类型，兼容）
    gray1 = cv2.bitwise_and(gray1, gray1, mask=mask)
    gray2 = cv2.bitwise_and(gray2, gray2, mask=mask)

    # 转换回 NumPy 数据，进行裁剪操作
    if isinstance(gray1, cv2.UMat):
        gray1 = gray1.get()
    if isinstance(gray2, cv2.UMat):
        gray2 = gray2.get()

    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    gray1 = gray1[y : y + h, x : x + w]
    gray2 = gray2[y : y + h, x : x + w]

    # 缩放后再转为 UMat（用于加速后续操作）
    gray1 = cv2.resize(gray1, (resize_to, resize_to))
    gray2 = cv2.resize(gray2, (resize_to, resize_to))

    if use_umat:
        gray1 = cv2.UMat(gray1)
        gray2 = cv2.UMat(gray2)

    # 差异图
    diff = cv2.absdiff(gray1, gray2)

    # 二值化
    _, diff_bin = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    # 统计差异像素数（需要从 UMat 转为 NumPy）
    if isinstance(diff_bin, cv2.UMat):
        diff_bin = diff_bin.get()

    changed_pixels = np.count_nonzero(diff_bin)
    total_pixels = diff_bin.size
    ratio = changed_pixels / total_pixels

    return ratio, ratio > pixel_change_ratio


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
    # 找高亮区域（阈值可调）
    # _, mask1 = cv2.threshold(gray1, 210, 255, cv2.THRESH_BINARY)
    # # 使用 inpaint 修复
    # gray1 = cv2.inpaint(gray1, mask1, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # _, mask2 = cv2.threshold(gray2, 210, 255, cv2.THRESH_BINARY)
    # # 使用 inpaint 修复
    # gray2 = cv2.inpaint(gray2, mask2, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    gray1 = cv2.bitwise_and(gray1, gray1, mask=mask)
    gray2 = cv2.bitwise_and(gray2, gray2, mask=mask)
    gray1 = remove_specular_reflection(gray1)
    gray2 = remove_specular_reflection(gray2)
    # gray1 = cv2.GaussianBlur(gray1, (15, 15), 3)
    # gray2 = cv2.GaussianBlur(gray2, (15, 15), 3)
    # _, gray1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # _, gray2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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


# ==== 线性 找轮廓 ====
def get_contour_dimensions(roi, yclFlag=True):
    """
    在感兴趣区域（ROI）中找到最大的轮廓，并返回其精确的宽度和高度。
    同时在ROI上绘制两条边长线（水平线和垂直线）。

    Args:
        roi: 包含物体的裁剪后图像。

    Returns:
        一个包含 (轮廓宽度, 轮廓高度) 的元组，如果未找到轮廓则返回 (None, None)。
    """
    print(f"get_contour_dimensions: 我进来了 ==== {yclFlag} ")
    binary_image = None
    if yclFlag:
        # 1. 将ROI转换为灰度图并应用阈值来创建二值图像。
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        _, binary_image = cv2.threshold(
            gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        binary_image = roi
    # 2. 在二值图像中查找所有轮廓。
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # 3. 通过面积找到最大的轮廓。
        largest_contour = max(contours, key=cv2.contourArea)

        # 4. 获取最大轮廓的紧密边界框。
        x, y, contour_w, contour_h = cv2.boundingRect(largest_contour)

        # 5. 在ROI上绘制两条边长线（水平线和垂直线）。
        #    - 水平线：从 (x, y + contour_h//2) 到 (x + contour_w, y + contour_h//2)
        #    - 垂直线：从 (x + contour_w//2, y) 到 (x + contour_w//2, y + contour_h)
        cv2.line(
            roi,
            (x, y + contour_h // 2),
            (x + contour_w, y + contour_h // 2),
            (0, 255, 0),
            2,
        )  # 绿色水平线
        # cv2.line(roi, (x + contour_w // 2, y), (x + contour_w // 2, y + contour_h), (0, 0, 255), 2)    # 红色垂直线

        # 可选：绘制整个轮廓（调试用）
        # cv2.drawContours(roi, [largest_contour], -1, (255, 0, 0), 1)

        return contour_w, contour_h

    return None, None


# ==== 线性 找轮廓 优化版，但是有时候识别不到线长 ====
def get_accurate_dimensions(roi, yclFlag=True):
    """
    【改造版】一个更精确的物体尺寸计算函数。
    学习了参考代码中的 Canny、形状筛选 和 精确测量思想。

    Args:
        roi: 包含物体的裁剪图像。

    Returns:
        如果找到符合条件的物体，则返回一个字典，包含：
        {'width_px': 宽度, 'height_px': 高度, 'box_points': 旋转框的四个角点}
        否则返回 None。
    """
    print(f"get_accurate_dimensions: 我进来了 ==== {yclFlag} ")
    if roi.size == 0:
        return None
    closed_canny = None
    if yclFlag:
        # 1. 预处理与边缘检测 (学习自参考代码中的 Canny 方法)
        #    对于轮廓分明的物体，Canny通常比简单的灰度阈值更有效。
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, blurred = cv2.threshold(
            blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # cv2.imwrite("blurred.jpg", blurred)
        # Canny边缘检测，参数(50, 150)是常用的起始值，可根据实际情况调整
        canny = cv2.Canny(blurred, 20, 150)

        # 使用形态学闭运算连接断开的边缘，使轮廓更完整
        kernel = np.ones((5, 5), np.uint8)
        closed_canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        closed_canny = roi
    # 先膨胀后腐蚀（开运算的逆操作，可以增强边缘）
    # closed_canny = cv2.dilate(closed_canny, kernel, iterations=1)
    # closed_canny = cv2.erode(closed_canny, kernel, iterations=1)
    # 2. 查找并筛选轮廓 (学习自参考代码中的 find_biggest_contour)
    contours, _ = cv2.findContours(
        closed_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # cv2.drawContours(roi, contours, -1, (255, 0, 0), 1)
    if contours:
        # 3. 通过面积找到最大的轮廓。
        largest_contour = max(contours, key=cv2.contourArea)

        # 4. 获取最大轮廓的紧密边界框。
        x, y, contour_w, contour_h = cv2.boundingRect(largest_contour)

        # 5. 在ROI上绘制两条边长线（水平线和垂直线）。
        #    - 水平线：从 (x, y + contour_h//2) 到 (x + contour_w, y + contour_h//2)
        #    - 垂直线：从 (x + contour_w//2, y) 到 (x + contour_w//2, y + contour_h)
        # cv2.line(roi, (x, y + contour_h // 2), (x + contour_w, y + contour_h // 2), (0, 255, 0), 2)  # 绿色水平线
        # cv2.line(roi, (x + contour_w // 2, y), (x + contour_w // 2, y + contour_h), (0, 0, 255), 2)    # 红色垂直线

        # 可选：绘制整个轮廓（调试用）
        # cv2.drawContours(roi, [largest_contour], -1, (255, 0, 0), 1)
        return contour_w, contour_h

    return None, None


# ==== 圆形找轮廓 ====
def get_circle_diameter_and_draw(roi, yclFlag=True):
    """
    【最终圆形版】在ROI中找到最显著的圆形，直接在ROI上把它画出来，并返回其直径。
    函数接口和返回值严格按照您的要求设计。

    Args:
        roi: 包含物体的裁剪图像。该图像将被直接修改（在上面绘图）。

    Returns:
        (直径, 直径) 元组。如果未找到圆形则返回 (None, None)。
    """
    if roi.size == 0:
        return None, None
    blurred = None
    # 1. 预处理：灰度化和高斯模糊
    #    霍夫圆变换对噪声敏感，模糊处理是必要步骤。
    if yclFlag:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 3)
        _, blurred = cv2.threshold(
            blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        blurred = roi
    # cv2.imwrite("blurred.jpg", blurred)
    # 使用形态学闭运算连接断开的边缘，使轮廓更完整
    # cv2.imwrite("blurred.jpg", blurred)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,  # 累加器分辨率与图像相同(1)或更高(>1)
        minDist=30,  # 圆之间的最小距离(根据实际物体间距调整)
        param1=100,  # Canny边缘检测高阈值(降低以检测弱边缘)
        param2=10,  # 累加器阈值(降低以检测更多圆)
        minRadius=50,  # 最小圆半径(根据实际物体大小调整)
        maxRadius=180,  # 最大圆半径(0表示不限制)
    )
    height, width = roi.shape[:2]
    if circles is not None and len(circles[0]) > 0:
        # 3. 遍历所有圆，找出最大直径且不越界的圆
        valid_circles = []
        for c in circles[0]:
            x, y, r = map(int, np.round(c))
            if (
                x - r >= 0
                and y - r >= 0
                and x + r <= (width - 10)
                and y + r <= (height - 10)
            ):
                valid_circles.append((x, y, r))
        if valid_circles:
            # 选最大半径的圆
            # x, y, r = max(valid_circles, key=lambda c: c[2])
            x, y, r = valid_circles[0]
            diameter = float(2 * r)
            # 绘制圆和圆心
            # cv2.circle(roi, (x, y), r, (0, 255, 0), 2)
            # cv2.circle(roi, (x, y), 3, (0, 0, 255), -1)
            return diameter, diameter
    # if circles is not None:
    # 找到半径最大的圆（即直径最大的圆）
    # largest_circle = max(circles[0, :], key=lambda c: c[2])
    # radius = largest_circle[2]
    # diameter_px = 2 * radius
    # radius = int(radius)
    # center_x, center_y = largest_circle[0], largest_circle[1]
    # center_x, center_y = int(round(largest_circle[0])), int(round(largest_circle[1]))
    # circle_data = np.uint16(np.around(circles[0, 0]))
    # # 在ROI上绘制圆和圆心（可视化调试）
    # cv2.circle(roi, (center_x, center_y), radius, (0, 255, 0), 2)  # 绿色圆边界
    # cv2.circle(roi, (center_x, center_y), 2, (0, 0, 255), 3)        # 红色圆心
    # center = (circle_data[0], circle_data[1])
    # radius = circle_data[2]
    # diameter_px = float(radius * 2)

    # 4. 【直接在ROI上绘图】
    #    因为 roi 是原图的一个切片，在这里绘图会直接反映到最终的主图像上。
    # a) 绘制检测到的绿色圆环
    # cv2.circle(roi, center, radius, (0, 255, 0), 2)
    # b) 绘制红色的圆心
    # cv2.circle(roi, center, 3, (0, 0, 255), -1)
    # print(f"{diameter_px}")
    # 5. 【保持返回值不变】
    #    严格按照您的要求，返回 (直径, 直径)
    # return diameter_px, diameter_px
    # 将检测到的圆转换为整数坐标

    # if len(circles) > 0:
    #     # 找到半径最小的圆（即直径最小的圆）
    #     smallest_circle = max(circles, key=lambda c: c[2])  # c[2]是半径
    #     radius = smallest_circle[2]
    #     diameter_px = float(radius * 2)
    #     center_x, center_y = smallest_circle[0], smallest_circle[1]

    #     # 在ROI上绘制圆和圆心（可视化调试）
    #     cv2.circle(roi, (center_x, center_y), radius, (0, 255, 0), 2)  # 绿色圆边界
    #     cv2.circle(roi, (center_x, center_y), 3, (0, 0, 255), -1)      # 红色圆心

    #     print(f"检测到的最小圆直径: {diameter_px}px")
    #     return diameter_px, diameter_px

    # 如果没有找到圆形，返回值也保持不变
    return None, None


# ==== 椭圆找轮廓 ====
def get_ellipse_diameter_and_draw(roi, pixel_per_cm):
    """
    在ROI中检测第一个有效椭圆，绘制并返回其长轴长度（作为近似直径）。
    如果椭圆太小，则尝试更精确的检测方法。
    """
    if roi.size == 0:
        return None, None

    # 1. 预处理：灰度化 + 高斯模糊 + 二值化
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = cv2.GaussianBlur(gray, (15, 15), 3)
    _, binary = cv2.threshold(binary, 0, 127, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. 查找轮廓并尝试拟合椭圆
    diameter = try_detect_ellipse(roi, binary)

    print(f"get_ellipse_diameter_and_draw: 我进来了 ==== {diameter} ")
    # 3. 如果椭圆检测失败或尺寸太小，尝试更精确的方法
    if diameter is None or (diameter / pixel_per_cm <= 6):
        diameter = try_precise_detection(binary, pixel_per_cm)

    # 4. 如果所有方法都失败，返回ROI的尺寸作为默认值
    if diameter is None:
        diameter = roi.shape[1]  # 返回宽度作为默认值(更合理)
    cv2.imwrite("closed_canny.jpg", binary)
    return diameter, diameter


# ==== 椭圆找轮廓，判断方法 ====
def try_detect_ellipse(roi, binary):
    """尝试检测椭圆并返回直径"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if len(cnt) >= 5:  # 拟合椭圆至少需要5个点
            ellipse = cv2.fitEllipse(cnt)
            (center_x, center_y), (major_axis, minor_axis), angle = ellipse
            if not all(
                isinstance(val, (int, float)) and not np.isnan(val)
                for val in [center_x, center_y, major_axis, minor_axis]
            ):
                continue  # 跳过无效的椭圆
            # 检查椭圆是否在ROI内
            if is_ellipse_within_bounds(
                roi, center_x, center_y, major_axis, minor_axis
            ):
                # 绘制椭圆和中心点
                cv2.ellipse(roi, ellipse, (0, 255, 0), 2)
                cv2.circle(roi, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)
                return float(major_axis)  # 返回长轴长度作为直径

    return None


# ==== 椭圆找轮廓，判断方法 ====
def is_ellipse_within_bounds(roi, center_x, center_y, major_axis, minor_axis):
    """检查椭圆是否完全在ROI边界内"""
    center_x = float(center_x)
    center_y = float(center_y)
    major_axis = float(major_axis)
    minor_axis = float(minor_axis)

    x_min = max(0, int(center_x - major_axis / 2))
    y_min = max(0, int(center_y - minor_axis / 2))
    x_max = min(roi.shape[1], int(center_x + major_axis / 2))
    y_max = min(roi.shape[0], int(center_y + minor_axis / 2))

    return x_min >= 0 and y_min >= 0 and x_max <= roi.shape[1] and y_max <= roi.shape[0]


# ==== 椭圆找轮廓，不满足时重新 ====
def try_precise_detection(binary, pixel_per_cm):
    """尝试更精确的尺寸检测方法"""
    # 先尝试检测圆形
    # contour_w, contour_h = get_circle_diameter_and_draw(binary, False)
    # if contour_w and (contour_w/pixel_per_cm > 6):
    # return contour_w
    contour_w, contour_h = get_contour_dimensions(binary, False)
    # 如果圆形太小或未检测到，尝试更精确的尺寸测量
    return contour_w


# ==== 图像去畸变 ====
def undistort_image(img, mtx, dist, alpha=1.0):
    """
    对输入图像进行去畸变处理。

    参数：
    - img: 输入图像（BGR格式）
    - mtx: 相机内参矩阵
    - dist: 畸变系数
    - alpha: 去畸变后图像的缩放参数，范围[0,1]
             0表示裁剪去黑边，1表示保留全部区域可能有黑边

    返回：
    - 去畸变且裁剪后的图像
    """
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))

    # 去畸变
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # 裁剪ROI区域
    # x, y, w_roi, h_roi = roi
    # dst_cropped = dst[y:y+h_roi, x:x+w_roi]
    # filename = time.strftime("%Y%m%d_%H%M%S") + "dst.jpg"
    # cv2.imwrite(filename, dst_cropped)
    return dst  # , newcameramtx, roi


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
    warped = cv2.warpPerspective(img_copy, M, (output_width, output_height))

    # 6. 保存结果
    if save_result:
        filename = time.strftime("%Y%m%d_%H%M%S") + "_warped.jpg"
        cv2.imwrite(filename, warped)
        print(f"✅ 已保存透视图：{filename}")

    return M, output_width, output_height, pixel_per_cm, warped


# ==== 透视变换 少参数====
def perspective_transform(img, src_pts, dst_pts, output_width, output_height):
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

    # 4. 获取透视矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 5. 执行透视变换
    warped = cv2.warpPerspective(img_copy, M, (output_width, output_height))

    return warped


# ==== 透视变换 少参数====
def perspective_homography(img, src_pts, dst_pts, output_width, output_height):
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

    # 4. 获取透视矩阵
    M, status = cv2.findHomography(src_pts, dst_pts)

    # 5. 执行透视变换
    warped = cv2.warpPerspective(img_copy, M, (output_width, output_height))

    return warped


def standardize_image_for_display(img):
    """确保图像可以安全用于 cv2.imshow"""
    if img is None:
        print("❌ 输入图像为 None")
        return None

    if not isinstance(img, np.ndarray):
        print(f"❌ 输入不是 np.ndarray，而是 {type(img)}")
        return None

    if img.dtype in [np.float32, np.float64]:
        img = np.nan_to_num(img)
        img = np.clip(img, 0, 1) if img.max() <= 1 else np.clip(img, 0, 255)
        img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


# ==== 根据后处理得到的坐标，来识别尺寸，返回尺寸 ====
def annotate_xywh(boxes, pixel_per_cm):
    x, y, w, h = boxes
    # 换算实际尺寸（cm）
    actual_w = w / pixel_per_cm
    actual_h = h / pixel_per_cm
    return actual_w, actual_h


# ==== 根据后处理得到的坐标，来识别尺寸，返回的图片 ====
def measure_and_annotate_xywh(img, boxes, pixel_per_cm):
    if img is None:
        print("❌ 图像为空，无法标注尺寸")
        return img
    x, y, w, h = boxes
    # 换算实际尺寸（cm）
    actual_w = w / pixel_per_cm
    actual_h = h / pixel_per_cm

    # 还原为左上角和右下角坐标
    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)
    print(
        f"[debug] type: {type(img)} shape: {getattr(img, 'shape', 'None')} dtype: {getattr(img, 'dtype', 'None')}"
    )
    # 控制台输出实际尺寸
    print(f"📏 目标尺寸: 宽 = {actual_w:.2f} cm, 高 = {actual_h:.2f} cm")
    # 绘制检测框和尺寸标注
    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label_text = f"{actual_w:.1f}cm x {actual_h:.1f}cm"
    cv2.putText(
        img,
        label_text,
        (x2 - 20, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )
    # cv2.imwrite(time.strftime("%Y%m%d_%H%M%S") + "output.jpg", img)
    return img


# ==== 绘制黑白格 ====
def draw_real_size_grid(
    image, grid_size_cm=1.0, color=(0, 255, 0), thickness=1, alpha=0.3, pixels_per_cm=40
):
    """
    绘制实际尺寸网格。
    :param image: 输入图像
    :param grid_size_cm: 每格实际尺寸（cm）
    :param color: 绘制颜色 (BGR 格式)
    :param thickness: 线条粗细
    :param alpha: 网格叠加的透明度
    """
    h, w = image.shape[:2]
    grid_size_px = int(grid_size_cm * pixels_per_cm)

    # 创建网格层
    grid_layer = np.zeros_like(image)

    # 绘制纵向网格线
    for x in range(0, w, grid_size_px):
        cv2.line(grid_layer, (x, 0), (x, h), color, thickness)

    # 绘制横向网格线
    for y in range(0, h, grid_size_px):
        cv2.line(grid_layer, (0, y), (w, y), color, thickness)

    # 合成
    overlay = cv2.addWeighted(image, 1.0, grid_layer, alpha, 0)
    return overlay


# ==== 图片大小转换 ====
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    # cv_image = np.array(image)
    # cv2.imshow("output2", cv_image)
    return new_image


# ==== 裁剪坐标 ====
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
    boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
    boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
    boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2


# ==== 显示窗口 ====
def show_in_moved_window(winname, img, x, y):
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, x, y)  # Move it to (x,y)
    cv2.imshow(winname, img)


def compute_3d_distance(
    camera_matrix, dist_coeffs, image_points, object_width_cm, object_height_cm
):
    # 物体在世界坐标中的4个点（单位cm），Z=0表示在同一个平面
    object_points = np.array(
        [
            [0, 0, 0],
            [object_width_cm, 0, 0],
            [object_width_cm, object_height_cm, 0],
            [0, object_height_cm, 0],
        ],
        dtype=np.float32,
    )

    # 图像中对应角点的像素坐标
    image_points_np = np.array(image_points, dtype=np.float32)

    # 求解 rvec, tvec
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points_np, camera_matrix, dist_coeffs
    )

    if not success:
        # raise ValueError("solvePnP 解算失败")
        return None, None, None, None

    # 平移向量 tvec 表示目标坐标系原点在相机坐标系中的位置
    X, Y, Z = tvec.flatten()
    distance = np.linalg.norm(tvec)

    print("📌 solvePnP 解算成功：")
    print(f"X = {X:.2f} cm, Y = {Y:.2f} cm, Z = {Z:.2f} cm")
    print(f"✅ 相机与物体中心的距离约为：{distance:.2f} cm")
    return X, Y, Z, distance


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
    # print(f"size======{size}")
    # print(f"original_size======{original_size}")
    # 获取缩放比例和填充
    gain = min(img_width / target_width, img_height / target_height)
    pad_w = round((img_width - target_width * gain) / 2 - 0.1)  # 水平填充
    pad_h = round((img_height - target_height * gain) / 2 - 0.1)  # 垂直填充
    # print(f"size======{gain}")
    # print(f"pad======{pad_w, pad_h}")
    # 将中心点 (cx, cy) + 宽高 (w, h) 转换为 (x_min, y_min, x_max, y_max) 的批量操作
    boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - pad_w) / gain  # x_min
    boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - pad_h) / gain  # y_min
    boxes[:, 2] = boxes[:, 2] / gain  # w 缩放
    boxes[:, 3] = boxes[:, 3] / gain  # h 缩放
    # print(f"boxes=========={boxes}")
    return boxes


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    # print(f"coords==========", coords)
    # gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    # pad = (
    #     round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
    #     round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
    # )
    # # x, y, w, h = box[:4]
    # coords[:, 0] = (coords[:, 0] - coords[:, 2] / 2 - pad[0]) / gain  # x_center
    # coords[:, 1] = (coords[:, 1] - coords[:, 3] / 2 - pad[1]) / gain  # y_center
    # coords[:, 2] = coords[:, 2] / gain  # width
    # coords[:, 3] = coords[:, 3] / gain  # height
    return coords


def get_3d_point(camera_matrix, rvec, tvec, image_point, world_z=0.0):
    """
    将图像上的一个2D点反投影到指定Z高度的世界坐标平面上。

    :param image_point: 图像上的2D像素坐标 (u, v)
    :param world_z: 该点所在平面的世界Z坐标，默认为0（微波炉底盘）
    :return: 3D世界坐标 (X, Y, Z)
    """
    # 1. 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # 2. 计算相机外参矩阵 [R|t]
    extrinsic_matrix = np.hstack((rotation_matrix, tvec))

    # 3. 计算完整的投影矩阵 P = K * [R|t]
    projection_matrix = camera_matrix @ extrinsic_matrix

    # 4. 为了求解，我们需要将 P 分解
    # P = [p1, p2, p3, p4] where p_i are column vectors
    # u = (p1.T * X) / (p3.T * X)
    # v = (p2.T * X) / (p3.T * X)
    # where X = [Xw, Yw, Zw, 1].T is the world point in homogeneous coords.

    # 我们有 u, v, Zw，要求 Xw, Yw
    # 整理方程为 A * [Xw, Yw] = B 的形式
    u, v = image_point

    p1 = projection_matrix[:, 0]
    p2 = projection_matrix[:, 1]
    p3 = projection_matrix[:, 2]
    p4 = projection_matrix[:, 3]

    # 构建矩阵 A 和 B
    A = np.zeros((2, 2))
    A[0, 0] = p1[0] - u * p3[0]
    A[0, 1] = p2[0] - u * p3[0]
    A[1, 0] = p1[1] - v * p3[1]
    A[1, 1] = p2[1] - v * p3[1]

    B = np.zeros((2, 1))
    B[0, 0] = u * (p3[2] * world_z + p4[2]) - (p1[2] * world_z + p4[0])
    B[1, 0] = v * (p3[2] * world_z + p4[2]) - (p2[2] * world_z + p4[1])

    # 求解 [Xw, Yw]
    try:
        world_xy = np.linalg.solve(A, B)
        return np.array([world_xy[0, 0], world_xy[1, 0], world_z])
    except np.linalg.LinAlgError:
        print("无法求解线性方程组，请检查相机参数或输入点。")
        return None
