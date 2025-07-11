# 图像识别系统 - 最新版本

基于 Next.js 15 + PyQt5 QWebEngineView + TensorFlow Lite 的实时图像识别系统

## 功能特性

- 🎥 实时摄像头视频流
- 🧠 TensorFlow Lite 模型支持
- 🔧 丰富的图像处理参数调节
- 👁️ 多种识别功能 (异物检测、尺寸分类等)
- 📊 实时检测结果展示
- 💾 图像保存和报告导出
- 📈 系统性能监控
- ⏰ 可配置检测延迟

## 技术栈

### 前端
- Next.js 15.1.3
- React 19
- TypeScript 5.7
- Tailwind CSS 3.4
- Radix UI

### 后端
- Python 3.9+
- PyQt5 5.15.11
- OpenCV 4.10
- TensorFlow Lite 2.18
- NumPy 2.2

## 安装和运行

### 1. 环境要求
\`\`\`bash
# Node.js 版本要求
node --version  # >= 18.17.0

# Python 版本要求
python --version  # >= 3.9.0
\`\`\`

### 2. 前端安装
\`\`\`bash
# 安装依赖
npm install

# 或使用 yarn
yarn install

# 或使用 pnpm (推荐)
pnpm install
\`\`\`

### 3. Python 环境安装
\`\`\`bash
# 创建虚拟环境 (推荐)
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 安装 Python 依赖
pip install -r requirements.txt

# 或使用 conda
conda create -n image-recognition python=3.11
conda activate image-recognition
pip install -r requirements.txt
\`\`\`

### 4. 启动应用

#### 方法一：开发模式
\`\`\`bash
# 启动Next.js开发服务器
npm run dev
# 或
pnpm dev

# 在另一个终端启动Python应用
python scripts/python_backend_example.py
\`\`\`

#### 方法二：生产模式
\`\`\`bash
# 构建前端
npm run build
npm run start

# 或
pnpm build
pnpm start
\`\`\`

## 项目结构

\`\`\`
├── app/                    # Next.js应用目录
├── components/            # React组件
├── hooks/                 # React Hooks
├── services/              # 服务层
├── types/                 # TypeScript类型定义
├── scripts/               # Python脚本
│   └── python_backend_example.py
├── model/                 # TensorFlow Lite模型文件
├── requirements.txt       # Python依赖
└── README.md
\`\`\`

## QWebEngineView 通信

### WebChannel 设置
Python端通过 QWebChannel 暴露API：
\`\`\`python
self.channel = QWebChannel()
self.channel.registerObject('pyapi', self.api)
self.web_view.page().setWebChannel(self.channel)
\`\`\`

### 前端调用
JavaScript端通过 WebChannel 调用Python方法：
\`\`\`javascript
// 等待WebChannel初始化
new QWebChannel(qt.webChannelTransport, function(channel) {
    window.pyapi = channel.objects.pyapi;
    // 现在可以调用Python方法
    pyapi.get_cameras().then(result => console.log(result));
});
\`\`\`

## TensorFlow Lite 集成

### 模型加载
\`\`\`python
self.interpreter = Interpreter(model_path=model_path, num_threads=4)
self.interpreter.allocate_tensors()
\`\`\`

### 推理执行
\`\`\`python
# 预处理
input_data = self.preprocess_image(image)

# 推理
self.interpreter.set_tensor(input_details[0]['index'], input_data)
self.interpreter.invoke()

# 后处理
output_data = self.interpreter.get_tensor(output_details[0]['index'])
\`\`\`

## 图像处理流程

1. **摄像头采集** - 1920x1080分辨率
2. **畸变矫正** - 使用相机内参矩阵
3. **透视变换** - 将图像转换为俯视图
4. **图像增强** - 对比度、亮度、饱和度调整
5. **模型推理** - TensorFlow Lite目标检测
6. **结果后处理** - NMS、尺寸分类等

## 配置说明

### 相机参数
- 内参矩阵：用于畸变矫正
- 畸变系数：k1, k2, k3, p1, p2
- 透视变换点：四个角点坐标

### 检测参数
- 目标阈值：检测置信度阈值
- NMS阈值：非极大值抑制阈值
- 延迟设置：检测间隔时间

### 尺寸分类
- 小号上限：10cm
- 中号上限：13cm
- 大号：超过13cm

## 🚀 最新版本特性

### Next.js 15 新特性
- **Turbo 模式** - 更快的构建速度
- **改进的图像优化** - 支持 WebP 和 AVIF
- **更好的 TypeScript 支持** - 类型检查优化
- **实验性功能** - Turbo 打包器支持

### React 19 新特性
- **并发特性** - 更好的性能
- **自动批处理** - 优化状态更新
- **Suspense 改进** - 更好的加载状态处理

### 依赖包更新
- **Radix UI** - 最新的无障碍组件
- **Lucide React** - 最新的图标库
- **Tailwind CSS** - 最新的样式功能

## 🐛 常见问题解决

### 1. Node.js 版本问题
\`\`\`bash
# 使用 nvm 管理 Node.js 版本
nvm install 20
nvm use 20
\`\`\`

### 2. Python 依赖冲突
\`\`\`bash
# 清理 pip 缓存
pip cache purge

# 重新安装依赖
pip install --no-cache-dir -r requirements.txt
\`\`\`

### 3. PyQt5 安装问题
\`\`\`bash
# Ubuntu/Debian
sudo apt-get install python3-pyqt5

# macOS
brew install pyqt5

# Windows
pip install PyQt5 --only-binary=all
\`\`\`

### 4. OpenCV 安装问题
\`\`\`bash
# 如果 opencv-python 安装失败，尝试
pip install opencv-python-headless
\`\`\`

## 🔄 版本迁移指南

### 从旧版本升级
1. **备份项目**
2. **更新 package.json**
3. **重新安装依赖**
4. **检查代码兼容性**
5. **测试功能**

### 兼容性检查
\`\`\`bash
# 检查 Next.js 兼容性
npx @next/codemod@latest

# 检查 React 兼容性
npm audit
\`\`\`

## 📊 性能优化

### 前端优化
- **代码分割** - 自动路由分割
- **图像优化** - WebP/AVIF 支持
- **CSS 优化** - Tailwind CSS 树摇
- **Bundle 分析** - 使用 @next/bundle-analyzer

### 后端优化
- **模型优化** - TensorFlow Lite 量化
- **内存管理** - OpenCV 内存池
- **并发处理** - 多线程推理
- **缓存策略** - 结果缓存

## 🚀 部署建议

### Docker 部署
\`\`\`dockerfile
# Dockerfile 示例
FROM node:20-alpine AS frontend
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM python:3.11-slim AS backend
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY scripts/ ./scripts/
COPY model/ ./model/

# 多阶段构建...
\`\`\`

### 云部署
- **Vercel** - 前端部署
- **Railway** - 后端部署
- **AWS/GCP** - 完整部署

## 📝 更新日志

### v2.0.0 (2024-01-10)
- 升级到 Next.js 15
- 升级到 React 19
- 更新所有依赖包到最新版本
- 修复按钮嵌套问题
- 优化性能和用户体验

### v1.0.0 (2024-01-01)
- 初始版本发布
- 基础图像识别功能
- QWebEngineView 集成

## 📄 许可证

MIT License - 详见 LICENSE 文件
