from queue import Queue
import platform
from concurrent.futures import ThreadPoolExecutor

# 使用 tflite_runtime 或 tensorflow 作为解释器
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf
    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate

# 使用委托文件
delegate = {
    "Linux": "/usr/lib/libvx_delegate.so",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]

# 传递给外部委托的选项 (作为字典)
ext_delegate_options = {
    "allowed_cache_mode": 1,  # 启用缓存模式
    "allowed_builtin_code": 1,  # 启用内置算子
    "cache_file_path": "/pwd",  # 设置缓存文件路径
}


# 加载并缓存 tflite 模型
def initTfLite(model="./tfLiteModel/yolov5s.tflite"):
    # 检查模型文件是否存在并加载
    interpreter = Interpreter(
        model_path=model,
        # experimental_delegates=[load_delegate(delegate, ext_delegate_options)],
        num_threads=4,
    )
    interpreter.allocate_tensors()
    interpreter.invoke()
    print(f"Model {model} loaded successfully.")
    return interpreter


# 初始化多个 tflite Runtime 实例并缓存
def initTfLites(model="./tfLiteModel/yolov5s.onnx", TPEs=1):
    return [initTfLite(model) for _ in range(TPEs)]


# 定义一个执行池，用于管理并行推理任务
class tfLitePoolExecutor:
    def __init__(self, modelPath, TPEs, func):
        self.TPEs = TPEs
        self.queue = Queue()
        self.tflitePool = initTfLites(modelPath, TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0

    def put(self, frame):
        # 将任务提交到线程池，并将未来对象存储到队列中
        self.queue.put(
            self.pool.submit(self.func, self.tflitePool[self.num % self.TPEs], frame)
        )
        self.num += 1

    def get(self):
        if self.queue.empty():
            return None, False
        fut = self.queue.get()
        return fut.result(), True

    def release(self):
        self.pool.shutdown()
