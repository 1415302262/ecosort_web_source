import os
from pathlib import Path
from typing import Any, Dict

import gradio as gr
from PIL import Image
import requests

from backend import app as backend_app


DEFAULT_MODEL_PATH = "checkpoints/best_model.pth"


def _to_int(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def ensure_checkpoint(model_path: str, model_url: str) -> None:
    path = Path(model_path)
    if path.exists():
        return

    if not model_url:
        raise FileNotFoundError(
            f"checkpoint not found at {model_path}. Please set HF_MODEL_URL environment variable."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(model_url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def init_model() -> str:
    backend_app.load_runtime_mapping_config()

    model_path = os.getenv("HF_MODEL_PATH", DEFAULT_MODEL_PATH)
    model_url = os.getenv("HF_MODEL_URL", "").strip()
    model_type = os.getenv("HF_MODEL_TYPE", "efficientnet")
    num_classes = _to_int(os.getenv("HF_NUM_CLASSES", "14"), 14)

    ensure_checkpoint(model_path, model_url)
    msg = backend_app.load_model(model_path, model_type, num_classes)
    return msg


def predict_image(image: Image.Image) -> Dict[str, Any]:
    if image is None:
        return {"error": "请先上传图片"}

    result = backend_app._infer_image(image.convert("RGB"))
    if result.get("uncertain", False):
        result["advice"] = "模型不确定，建议人工复核（HF 免费版默认不调用云端VLM）"
    return result


STARTUP_MSG = ""
try:
    STARTUP_MSG = init_model()
except Exception as exc:
    STARTUP_MSG = f"模型初始化失败: {exc}"


with gr.Blocks(title="EcoSort 14-Class Demo") as demo:
    gr.Markdown("# EcoSort 14 类垃圾分类（Hugging Face 免费部署版）")
    gr.Markdown(
        "- 本 Space 使用本地模型推理（不依赖 Render/Cloudflare 随机链接）\n"
        "- 若首次启动慢，通常是下载 checkpoint 中"
    )
    gr.Markdown(f"**启动状态：** {STARTUP_MSG}")

    with gr.Row():
        image_input = gr.Image(type="pil", label="上传垃圾图片")
        json_output = gr.JSON(label="预测结果")

    run_btn = gr.Button("开始识别")
    run_btn.click(fn=predict_image, inputs=image_input, outputs=json_output)


if __name__ == "__main__":
    demo.launch()
