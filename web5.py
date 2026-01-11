"""
DeepSeek-OCR Gradio 应用 - 完整版
支持原始模型和微调模型，集成现代化UI和完整功能
"""

import gradio as gr
import torch
from modelscope import AutoModel, AutoTokenizer
from peft import PeftModel
import os
import tempfile
from PIL import Image, ImageDraw
import re
from typing import Tuple, Optional, Dict, Any, List
import fitz  # PyMuPDF for PDF processing
import numpy as np
import io
import time
import psutil
import GPUtil
import socket

# --- 常量和配置 ---
MODEL_CONFIGS = {
    "🤖 原始DeepSeek-OCR模型": {
        "model_name": "deepseek-ai/DeepSeek-OCR",
        "is_custom": False
    },
    "🎯 微调模型 (LoRA)": {
        "model_name": "deepseek-ai/DeepSeek-OCR",
        "is_custom": True,
        "adapter_path": ".finetuned_model/final_model"  # 默认微调模型路径
    }
}

MODEL_SIZE_CONFIGS = {
    "🚀 极速模式": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "⚖️ 平衡模式": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "🎯 精准模式": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "🔍 超清模式": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "🤖 Gundam模式": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}

TASK_PROMPTS = {
    "📝 自由OCR": "<image>\n自由OCR.",
    "📄 转换为Markdown": "<image>\n<|grounding|>将文档转换为markdown.",
    "📈 解析图表": "<image>\n解析图表.",
}

DEFAULT_MODEL_TYPE = "🤖 原始DeepSeek-OCR模型"
DEFAULT_MODEL_SIZE = "🤖 Gundam模式"
DEFAULT_TASK_TYPE = "📄 转换为Markdown"
BOUNDING_BOX_PATTERN = re.compile(r"<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>")
BOUNDING_BOX_COLOR = "#FF6B6B"
BOUNDING_BOX_WIDTH = 3
NORMALIZATION_FACTOR = 1000

# --- 全局变量 ---
model = None
tokenizer = None
model_gpu = None
current_model_type = None
current_adapter_path = None


def get_available_port(start_port=7860):
    """获取可用的端口号"""
    port = start_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            port += 1
            if port > start_port + 100:
                raise Exception("找不到可用端口")


class PerformanceMonitor:
    """性能监控类"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_before = None
        self.memory_after = None
        self.gpu_before = None
        self.gpu_after = None

    def start(self):
        """开始性能监控"""
        self.start_time = time.time()
        self.memory_before = self.get_memory_usage()
        self.gpu_before = self.get_gpu_usage()

    def stop(self):
        """停止性能监控"""
        self.end_time = time.time()
        self.memory_after = self.get_memory_usage()
        self.gpu_after = self.get_gpu_usage()

    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': psutil.virtual_memory().percent
        }

    def get_gpu_usage(self) -> Optional[Dict[str, Any]]:
        """获取GPU使用情况"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 使用第一个GPU
                return {
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100
                }
        except Exception:
            return None
        return None

    def get_performance_report(self, image_count: int = 1) -> str:
        """生成性能报告"""
        if self.start_time is None or self.end_time is None:
            return "性能数据不可用"

        total_time = self.end_time - self.start_time
        avg_time_per_image = total_time / image_count if image_count > 0 else total_time

        report_lines = [
            "## 📊 性能分析报告",
            f"**⏱️ 处理时间**: {total_time:.2f}秒",
            f"**🖼️ 处理数量**: {image_count}张图像",
            f"**📈 平均速度**: {avg_time_per_image:.2f}秒/张",
            "",
            "## 💾 系统资源",
            f"**内存使用**: {self.memory_after['rss_mb']:.1f}MB",
            f"**系统内存**: {self.memory_after['percent']:.1f}%",
        ]

        if self.gpu_after:
            report_lines.extend([
                f"**GPU利用率**: {self.gpu_after['load']:.1f}%",
                f"**GPU显存**: {self.gpu_after['memory_used']}/{self.gpu_after['memory_total']}MB ({self.gpu_after['memory_percent']:.1f}%)",
            ])

        if self.gpu_before and self.gpu_after:
            gpu_memory_increase = self.gpu_after['memory_used'] - self.gpu_before['memory_used']
            report_lines.append(f"**显存增量**: {gpu_memory_increase:.1f}MB")

        # 添加模型信息
        if current_model_type:
            report_lines.append("")
            report_lines.append("## 🤖 模型信息")
            report_lines.append(f"**当前模型**: {current_model_type}")
            if current_adapter_path:
                report_lines.append(f"**适配器路径**: {current_adapter_path}")

        return "\n".join(report_lines)


def check_finetuned_model_exists(adapter_path: str) -> bool:
    """检查微调模型是否存在"""
    if os.path.exists(adapter_path):
        # 检查必要的文件是否存在
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        model_files = os.listdir(adapter_path)
        has_required = any(file in model_files for file in required_files)

        if has_required:
            print(f"✅ 微调模型存在且完整: {adapter_path}")
            return True
        else:
            print(f"⚠️ 微调模型目录存在但缺少必要文件: {adapter_path}")
            return False
    else:
        print(f"❌ 微调模型目录不存在: {adapter_path}")
        return False


def load_model_and_tokenizer(model_type: str, adapter_path: str = None) -> None:
    """
    加载指定的模型和分词器

    Args:
        model_type: 模型类型
        adapter_path: LoRA适配器路径（对于微调模型）
    """
    global model, tokenizer, current_model_type, current_adapter_path

    # 如果模型类型和适配器路径没有变化，则不需要重新加载
    if (model_type == current_model_type and
            (not MODEL_CONFIGS[model_type]["is_custom"] or adapter_path == current_adapter_path)):
        print("✅ 模型已加载，无需重新加载")
        return

    print(f"正在加载模型: {model_type}")

    try:
        # 清除之前的模型以释放内存
        if model is not None:
            del model
            torch.cuda.empty_cache()

        model_config = MODEL_CONFIGS[model_type]
        model_name = model_config["model_name"]

        if model_config["is_custom"]:
            # 使用提供的适配器路径或默认路径
            actual_adapter_path = adapter_path if adapter_path else model_config.get("adapter_path", "./final_model")

            # 检查微调模型是否存在
            if not check_finetuned_model_exists(actual_adapter_path):
                raise gr.Error(f"微调模型不存在或文件不完整！路径: {actual_adapter_path}")

            print(f"📁 加载微调模型，适配器路径: {actual_adapter_path}")
        else:
            print(f"🌐 加载原始模型: {model_name}")

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # 加载基础模型
        base_model = AutoModel.from_pretrained(
            model_name,
            attn_implementation="eager",
            trust_remote_code=True,
            use_safetensors=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # 如果是微调模型，加载LoRA适配器
        if model_config["is_custom"]:
            model = PeftModel.from_pretrained(
                base_model,
                actual_adapter_path,
                torch_dtype=torch.float16
            )
            current_adapter_path = actual_adapter_path
        else:
            model = base_model
            current_adapter_path = None

        model = model.eval()

        # 更新当前模型信息
        current_model_type = model_type

        print(f"✅ {model_type} 加载成功")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("尝试备选加载方式...")

        try:
            base_model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True,
                device_map="auto"
            )

            if model_config["is_custom"] and actual_adapter_path:
                model = PeftModel.from_pretrained(
                    base_model,
                    actual_adapter_path
                )
                current_adapter_path = actual_adapter_path
            else:
                model = base_model
                current_adapter_path = None

            model = model.eval()

            current_model_type = model_type

            print(f"✅ {model_type} 通过备选方式加载成功")
        except Exception as e2:
            print(f"❌ 备选加载方式也失败: {e2}")
            raise gr.Error(f"模型加载失败: {e2}")


def move_model_to_gpu() -> None:
    """如果模型尚未在GPU上，则将其移动到GPU"""
    global model_gpu
    if model_gpu is None and model is not None:
        print("🚀 正在将模型移动到GPU...")
        model_gpu = model.cuda().to(torch.bfloat16, non_blocking=True)
        print("✅ 模型已在GPU上")


def find_result_image(path: str) -> Optional[Image.Image]:
    """在指定路径中查找预生成的结果图像"""
    for filename in os.listdir(path):
        if "grounding" in filename or "result" in filename:
            try:
                image_path = os.path.join(path, filename)
                return Image.open(image_path)
            except Exception as e:
                print(f"打开结果图像 {filename} 时出错: {e}")
    return None


def build_prompt(task_type: str, ref_text: str) -> str:
    """根据任务类型和参考文本构建适当的提示"""
    if task_type == "🎯 通过参考定位对象":
        if not ref_text or ref_text.strip() == "":
            raise gr.Error("对于'定位'任务，您必须提供要查找的参考文本！")
        return f"<image>\n在图像中定位 <|ref|>{ref_text.strip()}<|/ref|>."

    return TASK_PROMPTS.get(task_type, TASK_PROMPTS["📝 自由OCR"])


def extract_and_draw_bounding_boxes(text_result: str, original_image: Image.Image) -> Optional[Image.Image]:
    """从文本结果中提取边界框坐标并在图像上绘制它们"""
    matches = list(BOUNDING_BOX_PATTERN.finditer(text_result))

    if not matches:
        return None

    print(f"✅ 找到 {len(matches)} 个边界框。正在原始图像上绘制。")

    image_with_bboxes = original_image.copy()
    draw = ImageDraw.Draw(image_with_bboxes)
    w, h = original_image.size

    w_scale = w / NORMALIZATION_FACTOR
    h_scale = h / NORMALIZATION_FACTOR

    for match in matches:
        coords = tuple(int(c) for c in match.groups())
        x1_norm, y1_norm, x2_norm, y2_norm = coords

        x1 = int(x1_norm * w_scale)
        y1 = int(y1_norm * h_scale)
        x2 = int(x2_norm * w_scale)
        y2 = int(y2_norm * h_scale)

        draw.rectangle([x1, y1, x2, y2], outline=BOUNDING_BOX_COLOR, width=BOUNDING_BOX_WIDTH)

    return image_with_bboxes


def run_inference(prompt: str, image_path: str, output_path: str, config: Dict[str, Any]) -> Tuple[str, float]:
    """使用给定参数运行模型推理"""
    print(f"🏃 使用提示运行推理: {prompt}")

    inference_start = time.time()

    text_result = model_gpu.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=output_path,
        base_size=config["base_size"],
        image_size=config["image_size"],
        crop_mode=config["crop_mode"],
        save_results=True,
        test_compress=True,
        eval_mode=True,
    )

    inference_time = time.time() - inference_start

    print(f"====\n📄 文本结果: {text_result}\n⏱️ 推理时间: {inference_time:.2f}秒\n====")
    return text_result, inference_time


def pdf_to_images(pdf_file: str, dpi: int = 200) -> list:
    """将PDF文件转换为图像列表"""
    images = []
    try:
        pdf_document = fitz.open(pdf_file)

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))

            if img.mode != 'RGB':
                img = img.convert('RGB')

            images.append(img)

        pdf_document.close()
        print(f"✅ PDF转换成功，共 {len(images)} 页")

    except Exception as e:
        print(f"❌ PDF转换失败: {e}")
        raise gr.Error(f"PDF转换失败: {str(e)}")

    return images


def process_pdf_ocr(pdf_file: str, model_size: str, task_type: str, ref_text: str,
                    model_type: str, adapter_path: str) -> Tuple[str, str, List[Tuple[Image.Image, str]]]:
    """处理PDF文件的OCR任务"""
    # 先加载模型
    load_model_and_tokenizer(model_type, adapter_path)
    move_model_to_gpu()

    prompt = build_prompt(task_type, ref_text)
    config = MODEL_SIZE_CONFIGS.get(model_size, MODEL_SIZE_CONFIGS[DEFAULT_MODEL_SIZE])

    pdf_images = pdf_to_images(pdf_file)

    if not pdf_images:
        return "PDF转换失败或为空PDF文件。", "", []

    performance_monitor = PerformanceMonitor()
    performance_monitor.start()

    all_results = []
    all_result_images = []
    total_inference_time = 0

    with tempfile.TemporaryDirectory() as output_path:
        for i, page_image in enumerate(pdf_images):
            temp_image_path = os.path.join(output_path, f"temp_page_{i + 1}.png")
            page_image.save(temp_image_path, optimize=True)

            text_result, inference_time = run_inference(prompt, temp_image_path, output_path, config)
            total_inference_time += inference_time

            page_result = f"--- 第 {i + 1} 页 ---\n{text_result}\n"
            all_results.append(page_result)

            result_image = extract_and_draw_bounding_boxes(text_result, page_image)

            if result_image is None:
                print(f"⚠️ 在第 {i + 1} 页文本结果中未找到边界框坐标。回退到搜索结果图像文件。")
                found_image = find_result_image(output_path)
                if found_image:
                    result_image = found_image
                else:
                    result_image = page_image.copy()

            label = f"第 {i + 1} 页"
            all_result_images.append((result_image, label))

    performance_monitor.stop()
    final_text = "\n".join(all_results)
    performance_report = performance_monitor.get_performance_report(len(pdf_images))

    return performance_report, final_text, all_result_images


def process_image_ocr(image: Image.Image, model_size: str, task_type: str, ref_text: str,
                      model_type: str, adapter_path: str) -> Tuple[str, str, List[Tuple[Image.Image, str]]]:
    """处理单张图像的OCR任务"""
    # 先加载模型
    load_model_and_tokenizer(model_type, adapter_path)
    move_model_to_gpu()

    prompt = build_prompt(task_type, ref_text)
    config = MODEL_SIZE_CONFIGS.get(model_size, MODEL_SIZE_CONFIGS[DEFAULT_MODEL_SIZE])

    performance_monitor = PerformanceMonitor()
    performance_monitor.start()

    with tempfile.TemporaryDirectory() as output_path:
        temp_image_path = os.path.join(output_path, "temp_image.png")
        image.save(temp_image_path, optimize=True)

        text_result, inference_time = run_inference(prompt, temp_image_path, output_path, config)
        performance_monitor.stop()

        result_image = extract_and_draw_bounding_boxes(text_result, image)

        if result_image is None:
            print("⚠️ 在文本结果中未找到边界框坐标。回退到搜索结果图像文件。")
            found_image = find_result_image(output_path)
            if found_image:
                result_image = found_image
            else:
                result_image = image.copy()

        result_images = [(result_image, "处理结果")]
        performance_report = performance_monitor.get_performance_report(1)

        return performance_report, text_result, result_images


def process_ocr_task(file_input: Any, model_size: str, task_type: str, ref_text: str,
                     model_type: str, adapter_path: str) -> Tuple[str, str, List[Tuple[Image.Image, str]]]:
    """使用DeepSeek-OCR处理图像或PDF以支持所有任务"""
    if file_input is None:
        return "请先上传图像或PDF文件。", "", []

    try:
        if isinstance(file_input, str) and file_input.lower().endswith('.pdf'):
            return process_pdf_ocr(file_input, model_size, task_type, ref_text, model_type, adapter_path)
        else:
            image = file_input if not isinstance(file_input, str) else Image.open(file_input)
            return process_image_ocr(image, model_size, task_type, ref_text, model_type, adapter_path)
    except Exception as e:
        error_msg = f"处理过程中发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg, "", []


def toggle_ref_text_visibility(task: str) -> gr.Textbox:
    """根据任务类型切换参考文本输入的可见性"""
    return gr.Textbox(visible=True) if task == "🎯 通过参考定位对象" else gr.Textbox(visible=False)


def toggle_adapter_path_visibility(model_type: str) -> gr.Textbox:
    """根据模型类型切换适配器路径输入的可见性"""
    return gr.Textbox(visible=True) if model_type == "🎯 微调模型 (LoRA)" else gr.Textbox(visible=False)


def get_model_status(model_type: str, adapter_path: str) -> str:
    """获取模型状态信息"""
    if model_type == "🎯 微调模型 (LoRA)":
        actual_path = adapter_path if adapter_path else MODEL_CONFIGS[model_type]["adapter_path"]
        if check_finetuned_model_exists(actual_path):
            return f"✅ 微调模型已就绪: {actual_path}"
        else:
            return f"❌ 微调模型不存在或文件不完整: {actual_path}"
    else:
        return "✅ 原始模型已就绪"


# 高级CSS样式 - 现代化设计
custom_css = """
/* 基础重置和变量 */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --accent-gradient: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
    --success-gradient: linear-gradient(45deg, #43e97b 0%, #38f9d7 100%);
    --warning-gradient: linear-gradient(45deg, #fa709a 0%, #fee140 100%);
    --dark-bg: #1a1a2e;
    --darker-bg: #16213e;
    --card-bg: rgba(255, 255, 255, 0.1);
    --text-light: #ffffff;
    --text-muted: rgba(255, 255, 255, 0.8);
    --shadow-soft: 0 8px 32px rgba(0, 0, 0, 0.1);
    --shadow-hard: 0 20px 40px rgba(0, 0, 0, 0.2);
}

/* 主容器样式 */
.gradio-container {
    background: var(--primary-gradient) !important;
    min-height: 100vh;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

/* 标题区域 */
.title-section {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%) !important;
    backdrop-filter: blur(20px);
    border-radius: 0 0 40px 40px !important;
    padding: 40px 20px !important;
    margin-bottom: 30px !important;
    box-shadow: var(--shadow-hard);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.main-title {
    text-align: center;
    font-weight: 800;
    background: linear-gradient(45deg, #FFD93D, #6BCF7F, #4D96FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em !important;
    margin-bottom: 15px !important;
    text-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.subtitle {
    font-size: 1.4em !important;
    color: var(--text-light) !important;
    opacity: 0.9;
    margin-bottom: 25px !important;
    font-weight: 300;
}

/* 卡片和容器 */
.gr-box, .gradio-group {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(20px);
    border-radius: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: var(--shadow-soft) !important;
    margin: 12px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.gr-box:hover, .gradio-group:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-hard) !important;
}

/* 按钮样式 */
button {
    border-radius: 16px !important;
    background: var(--accent-gradient) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 16px 32px !important;
    margin: 8px !important;
    transition: all 0.3s ease !important;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 25px rgba(77, 150, 255, 0.3);
}

button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 12px 35px rgba(77, 150, 255, 0.4);
}

button:active {
    transform: translateY(0) scale(0.98);
}

button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}

button:hover::before {
    left: 100%;
}

/* 输入框和下拉菜单 */
input, textarea, select, .gr-dropdown {
    border-radius: 16px !important;
    border: 2px solid rgba(77, 150, 255, 0.2) !important;
    background: rgba(248, 250, 252, 0.8) !important;
    padding: 16px 20px !important;
    transition: all 0.3s ease !important;
    font-size: 14px !important;
}

input:focus, textarea:focus, select:focus, .gr-dropdown:focus {
    border-color: #4D96FF !important;
    box-shadow: 0 0 0 4px rgba(77, 150, 255, 0.1) !important;
    background: white !important;
    transform: translateY(-2px);
}

/* 文件上传区域 */
.upload-area {
    border: 3px dashed #4D96FF !important;
    border-radius: 24px !important;
    background: rgba(77, 150, 255, 0.05) !important;
    transition: all 0.3s ease !important;
    padding: 50px 30px !important;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.upload-area::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: var(--accent-gradient);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.upload-area:hover {
    border-color: #FF6B6B !important;
    background: rgba(255, 107, 107, 0.05) !important;
    transform: scale(1.02);
}

.upload-area:hover::before {
    opacity: 0.1;
}

/* 选项卡样式 */
.gr-tabs {
    border-radius: 24px !important;
    background: white !important;
    box-shadow: var(--shadow-soft) !important;
}

.tab-nav {
    background: var(--accent-gradient) !important;
    border-radius: 20px 20px 0 0 !important;
    padding: 10px !important;
}

.tab-nav .tab-button {
    border-radius: 12px !important;
    margin: 0 5px !important;
    transition: all 0.3s ease !important;
}

.tab-nav .tab-button.selected {
    background: rgba(255, 255, 255, 0.2) !important;
    backdrop-filter: blur(10px);
}

/* 画廊样式 */
.gallery {
    border-radius: 24px !important;
    background: white !important;
    padding: 25px !important;
    box-shadow: var(--shadow-soft);
}

.gallery .thumbnail {
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
}

.gallery .thumbnail:hover {
    transform: scale(1.05);
    box-shadow: var(--shadow-hard);
}

/* 性能卡片 */
.performance-card {
    background: var(--success-gradient) !important;
    border-radius: 20px !important;
    padding: 25px !important;
    color: white !important;
    box-shadow: var(--shadow-soft);
}

/* 标签和文本 */
.label {
    font-weight: 700 !important;
    color: #2d3748 !important;
    margin-bottom: 12px !important;
    font-size: 1.1em !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* 徽章样式 */
.badge-container {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
    margin: 20px 0;
}

.badge {
    background: rgba(255, 255, 255, 0.2);
    padding: 10px 20px;
    border-radius: 20px;
    color: white;
    font-size: 0.9em;
    font-weight: 500;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    transition: all 0.3s ease;
}

.badge:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: translateY(-2px);
}

/* 加载动画 */
@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.loading-shimmer {
    position: relative;
    overflow: hidden;
}

.loading-shimmer::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    animation: shimmer 1.5s infinite;
}

/* 状态指示器 */
.status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
    box-shadow: 0 0 10px currentColor;
}

.status-ready {
    background: #6BCF7F;
    animation: pulse-green 2s infinite;
}

.status-processing {
    background: #FFD93D;
    animation: pulse-yellow 1.5s infinite;
}

@keyframes pulse-green {
    0%, 100% { 
        box-shadow: 0 0 0 0 rgba(107, 207, 127, 0.7);
    }
    70% { 
        box-shadow: 0 0 0 10px rgba(107, 207, 127, 0);
    }
}

@keyframes pulse-yellow {
    0%, 100% { 
        box-shadow: 0 0 0 0 rgba(255, 217, 61, 0.7);
    }
    70% { 
        box-shadow: 0 0 0 10px rgba(255, 217, 61, 0);
    }
}

/* 响应式设计 */
@media (max-width: 768px) {
    .main-title {
        font-size: 2.2em !important;
    }

    .subtitle {
        font-size: 1.1em !important;
    }

    .gr-box {
        margin: 8px !important;
        border-radius: 20px !important;
    }

    .badge-container {
        gap: 8px;
    }

    .badge {
        padding: 8px 16px;
        font-size: 0.8em;
    }
}

/* 滚动条美化 */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: var(--accent-gradient);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--secondary-gradient);
}

/* 图标动画 */
.icon-animation {
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { 
        transform: translateY(0px) rotate(0deg); 
    }
    50% { 
        transform: translateY(-10px) rotate(5deg); 
    }
}

/* 网格背景 */
.grid-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: -1;
}
"""


def create_ui() -> gr.Blocks:
    """创建和配置Gradio用户界面"""
    with gr.Blocks(
            title="🚀 DeepSeek-OCR 智能文档识别系统",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="pink",
                neutral_hue="slate",
                font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "system-ui", "sans-serif"]
            ),
            css=custom_css
    ) as demo:
        # 网格背景
        gr.HTML("""
        <div class="grid-bg"></div>
        """)

        # 主标题和描述
        gr.HTML("""
        <div class="title-section">
            <h1 class="main-title">🚀 DeepSeek-OCR 智能文档识别系统</h1>
            <p class="subtitle">支持原始模型和微调模型的双重OCR解决方案</p>
            <div class="badge-container">
                <span class="badge">📄 PDF多页处理</span>
                <span class="badge">🎯 双模型支持</span>
                <span class="badge">📊 实时性能监控</span>
                <span class="badge">🖼️ 可视化标注</span>
                <span class="badge">⚡ 极速推理</span>
                <span class="badge">🔧 LoRA微调</span>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False, variant="panel"):
            # 左侧配置面板
            with gr.Column(scale=1, min_width=420, variant="compact"):
                with gr.Group():
                    gr.Markdown("### 📁 文件上传区域")
                    file_input = gr.File(
                        label="🖼️ 拖放或选择图像/PDF文件",
                        file_types=[".png", ".jpg", ".jpeg", ".gif", ".bmp", ".pdf"],
                        type="filepath",
                        elem_classes="upload-area",
                        height=220,
                        scale=1
                    )

                with gr.Group():
                    gr.Markdown("### 🤖 模型选择配置")

                    with gr.Row():
                        model_type = gr.Dropdown(
                            choices=list(MODEL_CONFIGS.keys()),
                            value=DEFAULT_MODEL_TYPE,
                            label="🔧 选择模型类型",
                            info="选择使用原始模型或微调模型",
                            scale=2
                        )

                    adapter_path_input = gr.Textbox(
                        label="📁 微调模型路径 (LoRA适配器)",
                        placeholder="请输入微调模型的路径，例如: ./final_model",
                        value="./final_model",
                        visible=False,
                        lines=2,
                        info="当选择'微调模型'时需要提供LoRA适配器路径"
                    )

                    # 模型状态显示
                    model_status = gr.Markdown(get_model_status(DEFAULT_MODEL_TYPE, "./final_model"))

                    # 检查模型状态按钮
                    check_model_btn = gr.Button("🔄 检查模型状态", variant="secondary", size="sm")

                with gr.Group():
                    gr.Markdown("### ⚙️ 处理参数设置")

                    with gr.Row():
                        model_size = gr.Dropdown(
                            choices=list(MODEL_SIZE_CONFIGS.keys()),
                            value=DEFAULT_MODEL_SIZE,
                            label="🔧 识别模式选择",
                            info="根据文档类型和需求选择合适的处理模式",
                            scale=2
                        )

                    with gr.Row():
                        task_type = gr.Dropdown(
                            choices=list(TASK_PROMPTS.keys()) + ["🎯 通过参考定位对象"],
                            value=DEFAULT_TASK_TYPE,
                            label="🎯 任务类型选择",
                            info="选择适合您需求的OCR任务类型",
                            scale=2
                        )

                    ref_text_input = gr.Textbox(
                        label="🔍 参考文本输入（定位任务专用）",
                        placeholder="请输入您要定位的文本内容，例如：标题、关键词、特定对象...",
                        visible=False,
                        lines=3,
                        max_lines=4
                    )

                with gr.Group():
                    gr.Markdown("### 🎯 操作控制面板")
                    with gr.Row():
                        submit_btn = gr.Button("🚀 开始智能识别", variant="primary", size="lg", scale=2)
                        clear_btn = gr.Button("🧹 一键清空", variant="secondary", scale=1)

                    gr.Markdown("""
                    **💡 使用提示：**
                    - 支持常见图像格式和PDF文档
                    - 大文件处理可能需要较长时间
                    - 定位任务需要提供准确的参考文本
                    - 微调模型需要LoRA适配器文件
                    - 切换模型类型后首次加载需要时间
                    """)

            # 右侧结果面板
            with gr.Column(scale=2, min_width=800, variant="panel"):
                with gr.Tabs(selected=0) as tabs:
                    with gr.TabItem("📊 性能分析面板", id=0):
                        performance_output = gr.Markdown(
                            value="""
                            **系统状态**: <span class='status-indicator status-ready'></span> 就绪等待中

                            ### 🎯 就绪状态
                            - 系统初始化完成
                            - 等待模型选择和文件上传
                            - 支持原始模型和微调模型

                            ### 💡 下一步操作
                            请选择模型类型并上传图像或PDF文件，然后点击"开始智能识别"按钮
                            """,
                            elem_classes="performance-card"
                        )

                    with gr.TabItem("📄 文本识别结果", id=1):
                        with gr.Group():
                            output_text = gr.Textbox(
                                label="📝 识别文本输出",
                                lines=20,
                                show_copy_button=True,
                                placeholder="识别结果将显示在这里...\n\n✨ 功能特色：\n• 支持Markdown格式输出\n• 一键复制结果\n• 智能文本格式化\n• 多语言识别支持",
                                elem_id="result-text",
                                max_lines=25
                            )

                    with gr.TabItem("🖼️ 可视化结果", id=2):
                        with gr.Group():
                            output_gallery = gr.Gallery(
                                label="🖼️ 标注结果预览",
                                show_label=True,
                                elem_id="result-gallery",
                                columns=3,
                                rows=2,
                                height="auto",
                                object_fit="contain",
                                preview=True,
                                show_download_button=True
                            )

        # 底部信息栏
        gr.HTML("""
        <div style="
            text-align: center; 
            margin-top: 50px; 
            padding: 40px 30px; 
            background: linear-gradient(135deg, rgba(44, 62, 80, 0.9) 0%, rgba(52, 152, 219, 0.9) 100%);
            border-radius: 30px; 
            color: white;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        ">
            <h3 style="margin-bottom: 30px; font-size: 2em; background: linear-gradient(45deg, #FFD93D, #6BCF7F); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">✨ 双模型支持特性</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 30px; text-align: left; margin-bottom: 40px;">
                <div style="background: rgba(255,255,255,0.1); padding: 25px; border-radius: 20px; backdrop-filter: blur(10px);">
                    <h4 style="color: #4D96FF; margin-bottom: 15px; font-size: 1.3em;">🤖 原始模型</h4>
                    <p style="line-height: 1.6; opacity: 0.9;">使用官方DeepSeek-OCR模型，提供稳定可靠的OCR识别能力，适合通用场景</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 25px; border-radius: 20px; backdrop-filter: blur(10px);">
                    <h4 style="color: #6BCF7F; margin-bottom: 15px; font-size: 1.3em;">🎯 微调模型</h4>
                    <p style="line-height: 1.6; opacity: 0.9;">基于LoRA技术的微调模型，针对特定领域优化，提供更精准的识别效果</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 25px; border-radius: 20px; backdrop-filter: blur(10px);">
                    <h4 style="color: #FFD93D; margin-bottom: 15px; font-size: 1.3em;">🔄 灵活切换</h4>
                    <p style="line-height: 1.6; opacity: 0.9;">支持在原始模型和微调模型之间无缝切换，无需重启应用</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 25px; border-radius: 20px; backdrop-filter: blur(10px);">
                    <h4 style="color: #FF6B6B; margin-bottom: 15px; font-size: 1.3em;">📊 性能监控</h4>
                    <p style="line-height: 1.6; opacity: 0.9;">实时监控两种模型的性能表现，提供详细的分析报告和资源使用情况</p>
                </div>
            </div>
            <div style="
                margin-top: 30px; 
                border-top: 1px solid rgba(255,255,255,0.3); 
                padding-top: 25px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 20px;
            ">
                <div style="text-align: left;">
                    <p style="margin-bottom: 8px; font-size: 1.1em;"><strong>👥 核心开发团队</strong></p>
                    <p style="opacity: 0.9;">梁展豪 · 周孝祖 · 潘祥瑜 · 沈洺弘</p>
                </div>
                <div style="text-align: right;">
                    <p style="margin-bottom: 8px; font-size: 1.1em;"><strong>🔗 项目资源</strong></p>
                    <p style="opacity: 0.9;">
                        <a href="https://github.com/wuxinwuwen" style="color: #4D96FF; text-decoration: none; font-weight: 600; display: inline-flex; align-items: center; gap: 8px;">
                            <span>GitHub Repository</span>
                            <span style="font-size: 1.2em;">↗</span>
                        </a>
                    </p>
                </div>
            </div>
        </div>
        """)

        # UI交互逻辑
        task_type.change(
            fn=toggle_ref_text_visibility,
            inputs=task_type,
            outputs=ref_text_input
        )

        model_type.change(
            fn=toggle_adapter_path_visibility,
            inputs=model_type,
            outputs=adapter_path_input
        )

        model_type.change(
            fn=get_model_status,
            inputs=[model_type, adapter_path_input],
            outputs=model_status
        )

        adapter_path_input.change(
            fn=get_model_status,
            inputs=[model_type, adapter_path_input],
            outputs=model_status
        )

        check_model_btn.click(
            fn=get_model_status,
            inputs=[model_type, adapter_path_input],
            outputs=model_status
        )

        submit_btn.click(
            fn=process_ocr_task,
            inputs=[file_input, model_size, task_type, ref_text_input, model_type, adapter_path_input],
            outputs=[performance_output, output_text, output_gallery]
        )

        clear_btn.click(
            fn=lambda: [None, """
            **系统状态**: <span class='status-indicator status-ready'></span> 已重置就绪

            ### 🎯 系统已重置
            - 所有输入已清空
            - 结果区域已重置
            - 等待新的文件上传

            ### 💡 准备就绪
            请选择模型类型并上传新的图像或PDF文件开始处理
            """, "", []],
            inputs=[],
            outputs=[file_input, performance_output, output_text, output_gallery]
        )

    return demo


def main() -> None:
    """初始化和启动应用程序的主函数"""
    # 检查微调模型状态
    finetuned_model_exists = check_finetuned_model_exists("./final_model")

    if finetuned_model_exists:
        print("✅ 微调模型已就绪，可以直接使用")
    else:
        print("⚠️ 微调模型不存在或文件不完整，请使用原始模型")

    # 如果示例目录不存在则创建
    if not os.path.exists("examples"):
        os.makedirs("examples")
        print("✅ 创建了examples目录，您可以在此放置示例文件")

    # 获取可用端口
    available_port = get_available_port()
    print(f"🔍 检测到可用端口: {available_port}")

    # 创建并启动UI
    demo = create_ui()

    print("🚀 正在启动DeepSeek-OCR服务...")
    print("🎨 现代化UI界面已加载")
    print("🤖 支持功能:")
    print("   - 原始DeepSeek-OCR模型")
    print("   - 微调模型 (LoRA适配器)")
    print("   - 智能模型切换")
    print("   - 实时性能监控")
    print("   - PDF多页处理")
    print("📝 如果无法访问，请尝试以下解决方案：")
    print("   1. 检查防火墙设置")
    print("   2. 尝试使用 http://localhost:7860")
    print("   3. 确保没有其他程序占用7860端口")
    print("   4. 尝试重启应用程序")

    try:
        demo.queue(max_size=20).launch(
            server_name="0.0.0.0",
            server_port=available_port,
            share=False,
            show_error=True,
            inbrowser=True,
            quiet=False,
            debug=True
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("🔄 尝试使用备用配置...")
        demo.queue(max_size=20).launch(
            server_name="127.0.0.1",
            server_port=available_port,
            share=False,
            show_error=True,
            inbrowser=True
        )


if __name__ == "__main__":
    main()