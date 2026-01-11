 DeepSeek-OCR 智能文档识别系统
基于深度学习的下一代OCR解决方案，重新定义文档识别体验。支持图像和PDF文档的智能识别，具备多种任务模式和实时性能监控功能。

https://img.shields.io/badge/%E7%95%8C%E9%9D%A2-%E7%8E%B0%E4%BB%A3%E5%8C%96-blue
https://img.shields.io/badge/Python-3.8%252B-green
https://img.shields.io/badge/PyTorch-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-orange
https://img.shields.io/badge/Gradio-Web%E7%95%8C%E9%9D%A2-yellow

核心功能特性
智能PDF处理
支持多页PDF文档自动分页处理

保持原始布局结构

智能识别复杂文档格式

多任务识别
提供多种OCR任务模式

满足文档转换、图表解析、对象定位等不同场景需求

灵活的提示词系统

实时性能监控
实时监控系统资源使用情况

优化处理效率

提供详细的性能分析报告

可视化标注
边界框智能标注

识别结果直观展示

便于验证和后续处理

安装步骤
环境要求
Python 3.8+
CUDA 11.0+ (GPU版本)
至少8GB RAM

1. 克隆项目
bash
git clone https://github.com/wuxinwuwen/deepseek-ocr.git
cd deepseek-ocr
2. 安装依赖
bash
pip install -r requirements.txt
如果没有requirements.txt，手动安装：

bash
pip install gradio torch modelscope Pillow PyMuPDF numpy psutil gputil
3. 启动应用
bash
python web5.py
应用将在 http://localhost:7860 启动，并自动在浏览器中打开。

使用方法
基本使用流程
上传文件：拖放或选择图像/PDF文件到上传区域

选择模式：根据需求选择合适的识别模式

设置任务：选择OCR任务类型

开始识别：点击"开始智能识别"按钮

查看结果：在右侧面板查看文本结果和可视化标注

支持的格式
图像格式：PNG, JPG, JPEG, GIF, BMP

文档格式：PDF (多页支持)

处理模式
模式	适用场景	特点
极速模式	简单文档	处理速度快，适合文字清晰的文档
平衡模式	一般文档	速度与精度的平衡
精准模式	复杂文档	高精度识别，适合复杂布局
超清模式	高质量需求	最高精度，适合高分辨率图像
Gundam模式	推荐使用	智能优化，综合性能最佳
任务类型
自由OCR：从图像中提取原始文本

转换为Markdown：将文档转换为Markdown格式

解析图表：从图表和图形中提取结构化数据

通过参考定位对象：查找特定对象/文本

配置说明
模型配置
python
MODEL_SIZE_CONFIGS = {
    "极速模式": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "平衡模式": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "精准模式": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "超清模式": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "Gundam模式": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}
性能监控
系统自动监控：

处理时间和速度

内存使用情况

GPU利用率和显存使用

系统资源状态

API使用
通过Gradio Client调用
python
from gradio_client import Client

# 连接到服务
client = Client("http://localhost:7860/")

# 处理图像
result = client.predict(
    file_path="path/to/your/image.jpg",
    model_size=" Gundam模式",
    task_type="📄 转换为Markdown",
    ref_text="",
    api_name="/process_ocr_task"
)

# 解包结果
performance_report, text_result, result_images = result
print("文本结果:", text_result)
批量处理示例
python
import glob
from gradio_client import Client

def batch_process(folder_path):
    client = Client("http://localhost:7860/")
    image_files = glob.glob(f"{folder_path}/*.jpg") + glob.glob(f"{folder_path}/*.png")
    
    for image_file in image_files:
        result = client.predict(
            file_path=image_file,
            model_size=" Gundam模式", 
            task_type=" 转换为Markdown",
            ref_text="",
            api_name="/process_ocr_task"
        )
        # 处理结果...
 技术架构
核心技术栈
深度学习框架: PyTorch

OCR模型: DeepSeek-OCR (来自ModelScope)

Web框架: Gradio

图像处理: PIL/Pillow, OpenCV

PDF处理: PyMuPDF

性能监控: psutil, GPUtil

系统架构
text
用户界面 (Gradio)
    ↓
业务逻辑层 (OCR处理)
    ↓
模型推理层 (DeepSeek-OCR)
    ↓
硬件加速层 (GPU/CUDA)
注意事项
系统要求
推荐使用NVIDIA GPU以获得最佳性能

确保有足够的磁盘空间存储模型文件(约几个GB)

网络连接用于首次下载模型

性能优化建议
对于大PDF文件，建议使用"极速模式"

高分辨率图像可选用"超清模式"

批量处理时注意内存使用

定位任务需要提供准确的参考文本

常见问题
Q: 模型加载失败怎么办？
A: 检查网络连接，确保能访问ModelScope仓库

Q: GPU内存不足？
A: 尝试使用较小的模型模式或减少批量大小

Q: PDF转换失败？
A: 确保PDF文件没有加密或损坏

开发团队
核心开发团队：

梁展豪

周孝祖

潘祥瑜

沈洺弘

项目地址： GitHub Repository

许可证
本项目基于 MIT License 开源。

未来规划
支持更多文档格式 (Word, Excel等)

添加多语言识别支持

实现批量处理队列

增加用户认证系统

提供RESTful API接口

贡献指南
我们欢迎各种形式的贡献！包括但不限于：

代码改进

文档完善

功能建议

Bug报告

请查看 CONTRIBUTING.md 了解详细指南。

开始使用：运行 python web5.py 即可体验强大的DeepSeek-OCR智能文档识别系统！
