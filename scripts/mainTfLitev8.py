import cv2
import time
from tfLitePool import tfLitePoolExecutor
from mqtt_tool import MQTTClient

# 图像处理函数，实际应用过程中需要自行修改
from tfliteFuncv8 import myFunc
from utils import draw_checkerboard  # ,uartComm

# cap = cv2.VideoCapture('./video/islandBenchmark.mp4')
cap = cv2.VideoCapture(0)
# 设置分辨率为 1920x1080（Full HD）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cv2.ocl.haveOpenCL():
    print("OpenCL is not available.")
else:
    print("OpenCL is available.")
# 启用 OpenCL
# cv2.ocl.setUseOpenCL(True)
# another_script.py
# MQTT 配置
# mqtt_username = "ebop"
# mqtt_password = "ebop-123456"
# mqtt_url = "192.168.0.186"
# mqtt_port = 1883
# mqtt_client_id = "mqttProducer"

# # 创建 MQTT 客户端实例
# mqtt_client = MQTTClient(
#     mqtt_client_id, mqtt_username, mqtt_password, mqtt_url, mqtt_port
# )
# # 连接到 MQTT 服务器
# if mqtt_client.connect():
#     # 发布消息
#     mqtt_client.publish("receive/123123", "Hello from another script!")
# cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
# # 设置分辨率
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# 获取默认分辨率
# default_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# default_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Default resolution: {default_width} x {default_height}")

# # 获取默认帧率
# default_fps = cap.get(cv2.CAP_PROP_FPS)
# print(f"Default FPS: {default_fps}")

# # 获取默认视频格式
# fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
# fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
# print(f"Default video format: {fourcc_str}")

# # 获取默认自动对焦
# autofocus = cap.get(cv2.CAP_PROP_AUTOFOCUS)
# print(f"Default autofocus: {autofocus}")
# 模型陆军
modelPath = "./model/best_float32-2.tflite"
# 线程数, 增大可提高帧率
TPEs = 2
# 初始化rknn池
pool = tfLitePoolExecutor(modelPath=modelPath, TPEs=TPEs, func=myFunc)
# 初始化异步所需要的帧
if cap.isOpened():
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool
            exit(-1)
        pool.put(frame)
else:
    print("Error: Unable to open the camera.")
    exit(-1)

frames, loopTime, initTime = 0, time.time(), time.time()
while cap.isOpened():
    frames += 1
    ret, frame = cap.read()
    if not ret:
        break
    pool.put(frame)
    frame, flag = pool.get()
    if flag == False:
        break
    # gridImg = draw_checkerboard(frame, cols=7, rows=8, alpha=0.3)
    cv2.imshow("test", frame)
    gridImg = None  # 释放内存
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if frames % 30 == 0:
        print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
        loopTime = time.time()

print("总平均帧率\t", frames / (time.time() - initTime))
# 释放cap和rknn线程池
cap.release()
cv2.destroyAllWindows()
pool.release()
# 断开连接
# mqtt_client.disconnect()
# uartComm.close()
