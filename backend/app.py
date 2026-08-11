"""
EcoSort Backend API
Flask REST API 服务器
"""

import base64
import hashlib
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
import yaml

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

from src.models.resnet_classifier import create_resnet_model
from src.models.efficientnet_classifier import create_efficientnet_model
from src.train.trainer import load_checkpoint
from src.data.letterbox import ResizeLongestSideWithPadding
from src.inference.cascade import (
    AUDITED_CLASS_NAMES,
    AUDITED_CONFIDENCE_THRESHOLD,
    AUDITED_IMAGE_SIZE,
    CascadePolicy,
    RuntimeTelemetry,
    coarse_category,
    map_vlm_label,
    runtime_parity,
)
from src.inference.seed_vlm import SeedVLMClient
from src.inference.seed_vlm import (
    PROMPT_SHA256 as SEED_PROMPT_SHA256,
    PROMPT_VERSION as SEED_PROMPT_VERSION,
)


app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = int(
    os.getenv('ECOSORT_MAX_REQUEST_BYTES', str(12 * 1024 * 1024))
)


def _parse_cors_origins() -> List[str]:
    raw = os.getenv('ECOSORT_CORS_ORIGINS', '*').strip()
    if not raw or raw == '*':
        return ['*']
    return [origin.strip() for origin in raw.split(',') if origin.strip()]


cors_origins = _parse_cors_origins()
CORS(
    app,
    resources={r"/*": {"origins": cors_origins}},
    methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['Content-Type', 'Authorization'],
)

# 全局变量
model = None
device = None
class_names = list(AUDITED_CLASS_NAMES)
transform = None
confidence_threshold = AUDITED_CONFIDENCE_THRESHOLD
fallback_class = 'trash'
margin_threshold = 0.15
entropy_threshold = 0.78
hazardous_confidence_threshold = 0.82
hazardous_margin_threshold = 0.22
hazardous_classes = {'e_waste', 'haz_battery', 'haz_device', 'haz_medicine'}
model_meta = {}
inference_image_size = int(os.getenv('ECOSORT_INFER_IMAGE_SIZE', str(AUDITED_IMAGE_SIZE)))
use_cuda_autocast = os.getenv('ECOSORT_USE_CUDA_AUTOCAST', '1').strip() != '0'
mapping_config_path = Path(
    os.getenv(
        'ECOSORT_MAPPING_CONFIG',
        str(Path(__file__).resolve().parent.parent / 'ecosort_mapping.yaml')
    )
)
runtime_raw_to_target: Dict[str, str] = {}
cascade_policy = CascadePolicy(confidence_threshold)
runtime_telemetry = RuntimeTelemetry()
seed_vlm_client = SeedVLMClient.from_env()
max_batch_size = max(1, int(os.getenv('ECOSORT_MAX_BATCH_SIZE', '16')))
allow_model_reload = os.getenv('ECOSORT_ALLOW_MODEL_RELOAD', '0').strip().lower() in {
    '1', 'true', 'yes', 'on'
}


def _call_vlm_fallback(image: Image.Image) -> Dict[str, Any]:
    """Call the bounded Seed adapter and map its 42-class output to V6."""
    result = seed_vlm_client.review(image)
    mapped = map_vlm_label(result.get('raw_label'), runtime_raw_to_target)
    result['mapped_class_name'] = mapped
    result['mapping_success'] = mapped is not None
    if mapped is None and result.get('success'):
        result['failure_reason'] = 'unmapped_label'
    return result


def load_runtime_mapping_config() -> None:
    """从 YAML 读取后端推理阈值与兜底类别配置。"""
    global confidence_threshold, fallback_class, class_names
    global cascade_policy, runtime_raw_to_target

    if not mapping_config_path.exists():
        print(f"[runtime-config] mapping file not found: {mapping_config_path}")
        return

    try:
        config = yaml.safe_load(mapping_config_path.read_text(encoding='utf-8')) or {}
        ecosort_cfg = config.get('ecosort_config', {})

        if 'confidence_threshold' in ecosort_cfg:
            confidence_threshold = float(ecosort_cfg['confidence_threshold'])

        if 'fallback_class' in ecosort_cfg:
            fallback_class = str(ecosort_cfg['fallback_class'])

        train_classes = ecosort_cfg.get('train_classes')
        if isinstance(train_classes, list) and len(train_classes) > 0:
            class_names = [str(name) for name in train_classes]

        raw_mapping = ecosort_cfg.get('raw_to_target')
        if isinstance(raw_mapping, dict):
            runtime_raw_to_target = {
                str(source): str(target) for source, target in raw_mapping.items()
            }
        cascade_policy = CascadePolicy(confidence_threshold)

        print(
            f"[runtime-config] loaded from {mapping_config_path} | "
            f"threshold={confidence_threshold:.2f}, fallback={fallback_class}, "
            f"train_classes={len(class_names)}, raw_mapping={len(runtime_raw_to_target)}"
        )
    except Exception as exc:
        print(f"[runtime-config] failed to load {mapping_config_path}: {exc}")


def _configure_runtime_limits() -> None:
    torch_threads = max(1, int(os.getenv('ECOSORT_TORCH_THREADS', '1')))
    interop_threads = max(1, int(os.getenv('ECOSORT_TORCH_INTEROP_THREADS', '1')))
    cv_threads = max(1, int(os.getenv('ECOSORT_CV2_THREADS', '1')))

    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass

    cv2.setNumThreads(cv_threads)
    cv2.ocl.setUseOpenCL(False)

COARSE_LABELS = ['recyclable', 'hazardous', 'kitchen', 'other']

DAILY_LABEL_ORDER = list(AUDITED_CLASS_NAMES)
MISSING_DAILY_LABELS: List[str] = []


def _normalized_entropy(probabilities: torch.Tensor) -> float:
    """计算归一化熵，范围约为 [0, 1]，越高表示越不确定。"""
    p = probabilities.clamp_min(1e-12)
    entropy = float(-(p * p.log()).sum().item())
    max_entropy = float(torch.log(torch.tensor(float(p.numel()))).item()) if p.numel() > 1 else 1.0
    if max_entropy <= 0:
        return 0.0
    return entropy / max_entropy


def _normalize_label(name: str) -> str:
    return name.strip().lower().replace('-', '_').replace(' ', '_')


def _coarse_category_for_label(name: str) -> str:
    return coarse_category(name)


def _daily_label_for_model_class(name: str) -> str:
    normalized = _normalize_label(name)
    mapping = {
        'organic': 'Food waste',
        'printing_industry': 'Magazine / Flyer',
        'trash': 'Residual waste',
        'plastic_bottle': 'Hard plastic container',
        'plastic_cup': 'Hard plastic container',
        'plastic_shaker': 'Hard plastic container',
        'plastic_bag': 'Soft plastic film/bag',
        'zip_plastic_bag': 'Soft plastic film/bag',
    }
    return mapping.get(normalized, name)


def _aggregate_daily_probabilities(class_names_local: List[str], probabilities: torch.Tensor) -> Dict[str, float]:
    daily_probs: Dict[str, float] = {}
    for idx, name in enumerate(class_names_local):
        daily = _daily_label_for_model_class(name)
        daily_probs[daily] = daily_probs.get(daily, 0.0) + float(probabilities[idx].item())
    return daily_probs


def _aggregate_coarse_probabilities(class_names_local: List[str], probabilities: torch.Tensor) -> Dict[str, float]:
    coarse_probs = {label: 0.0 for label in COARSE_LABELS}
    for idx, name in enumerate(class_names_local):
        coarse = _coarse_category_for_label(name)
        coarse_probs[coarse] += float(probabilities[idx].item())
    return coarse_probs


def _apply_unknown_guardrails(
    class_names_local: List[str],
    probabilities: torch.Tensor,
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
) -> Dict[str, Any]:
    """统一的不确定性拦截规则，减少离谱误判。"""
    top1_conf = float(top_values[0].item())
    top1_idx = int(top_indices[0].item())
    top1_name = class_names_local[top1_idx]
    top2_conf = float(top_values[1].item()) if len(top_values) > 1 else 0.0
    margin = top1_conf - top2_conf
    entropy = _normalized_entropy(probabilities)
    coarse_probabilities = _aggregate_coarse_probabilities(class_names_local, probabilities)
    coarse_category = max(coarse_probabilities.items(), key=lambda kv: kv[1])[0]
    coarse_confidence = float(coarse_probabilities[coarse_category])
    daily_probabilities = _aggregate_daily_probabilities(class_names_local, probabilities)

    top1_daily_label = _daily_label_for_model_class(top1_name)

    low_confidence = cascade_policy.should_route(top1_conf)
    low_margin = margin < margin_threshold
    high_entropy = entropy > entropy_threshold

    hazardous_hit = top1_name in hazardous_classes
    hazardous_needs_review = hazardous_hit and (
        top1_conf < hazardous_confidence_threshold or margin < hazardous_margin_threshold
    )

    liquid_reminder = _normalize_label(top1_name) == 'liquid' and top1_conf >= 0.45

    uncertain = low_confidence
    # Rejection is a routing decision, not a replacement prediction. Keep the
    # local class unless a mapped VLM response succeeds.
    final_name = top1_name

    top_k = min(3, len(class_names_local))
    topk_values, topk_indices = torch.topk(probabilities, k=top_k)
    top_predictions = [
        {
            'class_name': class_names_local[int(topk_indices[i].item())],
            'daily_label': _daily_label_for_model_class(class_names_local[int(topk_indices[i].item())]),
            'confidence': float(topk_values[i].item()),
            'coarse_category': _coarse_category_for_label(class_names_local[int(topk_indices[i].item())]),
        }
        for i in range(top_k)
    ]

    return {
        'class_name': final_name,
        'raw_top1_class': top1_name,
        'daily_label': top1_daily_label,
        'daily_probabilities': daily_probabilities,
        'coarse_category': coarse_category,
        'coarse_confidence': coarse_confidence,
        'coarse_probabilities': coarse_probabilities,
        'top_predictions': top_predictions,
        'class_id': top1_idx,
        'confidence': top1_conf,
        'top2_confidence': top2_conf,
        'margin': margin,
        'entropy': entropy,
        'uncertain': uncertain,
        'routing_strategy': 'confidence_only',
        'routing_threshold': confidence_threshold,
        'review_reasons': {
            'low_confidence': low_confidence,
            'low_margin': low_margin,
            'high_entropy': high_entropy,
            'hazardous_guard': hazardous_needs_review,
            'liquid_reminder': liquid_reminder,
        },
        'advice': '检测到液体状态，请先倒空液体再投放容器。' if liquid_reminder else '',
    }


def _decode_base64_image(image_data: str) -> Image.Image:
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_lfs_pointer(path: Path) -> None:
    with path.open('rb') as handle:
        prefix = handle.read(128)
    if prefix.startswith(b'version https://git-lfs.github.com/spec/v1'):
        raise RuntimeError(
            f'{path} is a Git LFS pointer, not model weights. Pull LFS objects '
            'or configure ECOSORT_MODEL_URL before starting inference.'
        )


def _infer_image(image: Image.Image) -> Dict[str, Any]:
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        if device is not None and str(device).startswith('cuda') and use_cuda_autocast:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_tensor)
        else:
            outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        top_values, top_indices = torch.topk(probabilities, k=min(2, len(class_names)))

    response = _apply_unknown_guardrails(class_names, probabilities, top_values, top_indices)
    response['probabilities'] = {
        class_names[i]: float(probabilities[i].item())
        for i in range(len(class_names))
    }
    return response


def _cascade_infer_image(image: Image.Image) -> Dict[str, Any]:
    """Run local inference and the bounded fallback using one audited policy."""
    started = time.perf_counter()
    local_started = time.perf_counter()
    response = _infer_image(image)
    local_latency_ms = (time.perf_counter() - local_started) * 1000.0
    local_class = str(response['raw_top1_class'])
    routed = bool(response.get('uncertain'))
    vlm_result: Dict[str, Any] = {}
    vlm_success = False

    if routed:
        vlm_result = _call_vlm_fallback(image)
        mapped = vlm_result.get('mapped_class_name')
        vlm_success = bool(vlm_result.get('success') and mapped)
        if vlm_success:
            response['class_name'] = mapped
            response['daily_label'] = mapped
            response['coarse_category'] = coarse_category(mapped)

    total_latency_ms = (time.perf_counter() - started) * 1000.0
    attempted = bool(vlm_result.get('attempted'))
    usage = vlm_result.get('usage') if attempted else None
    runtime_telemetry.record(
        routed=routed,
        vlm_attempted=attempted,
        vlm_success=vlm_success,
        fallback_reason=vlm_result.get('failure_reason'),
        latency_ms=total_latency_ms,
        vlm_latency_ms=vlm_result.get('latency_ms') if attempted else None,
        usage=usage,
    )
    response.update({
        'local_prediction': local_class,
        'final_prediction': response['class_name'],
        'local_latency_ms': local_latency_ms,
        'cascade_latency_ms': total_latency_ms,
        'vlm_fallback_used': vlm_success,
        'routing': {
            'strategy': 'confidence_only',
            'threshold': confidence_threshold,
            'routed': routed,
            'provider_attempted': attempted,
            'provider_success': vlm_success,
            'provider_cached': bool(vlm_result.get('cached')),
            'failure_reason': vlm_result.get('failure_reason'),
            'raw_vlm_label': vlm_result.get('raw_label'),
            'mapped_vlm_label': vlm_result.get('mapped_class_name'),
            'provider_model': vlm_result.get('provider_model'),
            'provider_latency_ms': vlm_result.get('latency_ms'),
            'usage': usage or {},
        },
    })
    if routed and not vlm_success:
        response['advice'] = '本地模型不确定；VLM不可用或失败，已安全保留本地预测。'
    elif vlm_success:
        response['advice'] = '本地模型不确定，已采用VLM复核后的映射结果。'
    return response


def _extract_candidate_crops(image: Image.Image, max_objects: int = 5) -> List[Dict[str, Any]]:
    rgb = np.array(image)
    h, w = rgb.shape[:2]
    if h < 40 or w < 40:
        return []

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edge_mask = cv2.Canny(blur, 35, 120)
    edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_otsu = cv2.bitwise_not(otsu)
    blob_mask = cv2.morphologyEx(inv_otsu, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    blob_mask = cv2.morphologyEx(blob_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    min_area = max(650, int(0.004 * w * h))
    max_area = int(0.92 * w * h)

    def collect_boxes(mask):
        found = []
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            aspect = bw / max(1.0, float(bh))
            if area < min_area or area > max_area:
                continue
            if aspect < 0.15 or aspect > 6.0:
                continue
            found.append((x, y, bw, bh, area))
        return found

    boxes = collect_boxes(edge_mask) + collect_boxes(blob_mask)

    boxes.sort(key=lambda item: item[4], reverse=True)

    candidates: List[Dict[str, Any]] = []
    for x, y, bw, bh, _ in boxes:
        if len(candidates) >= max_objects:
            break

        pad_w = int(bw * 0.12)
        pad_h = int(bh * 0.12)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + bw + pad_w)
        y2 = min(h, y + bh + pad_h)

        too_much_overlap = False
        for item in candidates:
            bx1, by1, bx2, by2 = item['bbox_px']
            ix1 = max(x1, bx1)
            iy1 = max(y1, by1)
            ix2 = min(x2, bx2)
            iy2 = min(y2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (bx2 - bx1) * (by2 - by1)
            if inter > 0.6 * min(area1, area2):
                too_much_overlap = True
                break
        if too_much_overlap:
            continue

        crop = image.crop((x1, y1, x2, y2))
        candidates.append({
            'crop': crop,
            'bbox_px': [x1, y1, x2, y2],
            'bbox_norm': [x1 / w, y1 / h, x2 / w, y2 / h],
        })

    # 低成本兜底：若只检测到 0~1 个候选，补充左右分区候选，提升双物体场景召回
    if len(candidates) < 2:
        fallback_regions = [
            [0, 0, int(0.55 * w), h],
            [int(0.45 * w), 0, w, h],
        ]
        for x1, y1, x2, y2 in fallback_regions:
            if len(candidates) >= max_objects:
                break
            overlap = False
            for item in candidates:
                bx1, by1, bx2, by2 = item['bbox_px']
                ix1 = max(x1, bx1)
                iy1 = max(y1, by1)
                ix2 = min(x2, bx2)
                iy2 = min(y2, by2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter > 0.75 * ((x2 - x1) * (y2 - y1)):
                    overlap = True
                    break
            if overlap:
                continue
            crop = image.crop((x1, y1, x2, y2))
            candidates.append({
                'crop': crop,
                'bbox_px': [x1, y1, x2, y2],
                'bbox_norm': [x1 / w, y1 / h, x2 / w, y2 / h],
            })

    return candidates


def _detect_model_type_from_state_dict(state_dict: dict) -> Tuple[str, str]:
    """从 state_dict keys 自动检测模型类型"""
    keys = list(state_dict.keys())
    if any(k.startswith("backbone._conv_stem") or k.startswith("backbone._blocks") for k in keys):
        return "efficientnet", "efficientnet-b3"
    if any(k.startswith("features.") or k.startswith("layer") for k in keys):
        return "resnet", "resnet50"
    return "efficientnet", "efficientnet-b3"


def load_model(model_path: str, model_type: str = None, num_classes: int = None):
    """加载模型 - 自动检测类型和类别数"""
    global model, device, transform, class_names, model_meta

    _configure_runtime_limits()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = Path(model_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Model checkpoint not found: {checkpoint_path}')
    _reject_lfs_pointer(checkpoint_path)
    checkpoint_sha256 = _sha256_file(checkpoint_path)

    # 加载 checkpoint 获取配置
    checkpoint_raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 自动检测模型类型
    if model_type is None:
        state_dict = checkpoint_raw.get('model_state_dict', {})
        model_type, backbone = _detect_model_type_from_state_dict(state_dict)
    else:
        backbone = 'resnet50' if model_type == 'resnet' else 'efficientnet-b3'

    # 从 config 获取类别名称
    config = checkpoint_raw.get('config', {})
    data_cfg = config.get('data', {})
    config_names = data_cfg.get('class_names')
    if isinstance(config_names, list) and len(config_names) > 0:
        class_names = config_names
        num_classes = len(class_names)
    elif num_classes is None:
        class_counts = data_cfg.get('class_counts', [])
        num_classes = len(class_counts) if class_counts else 4

    # 从 state_dict 中检测类别数（作为最后的备选）
    if num_classes is None or num_classes == 4:
        state_dict = checkpoint_raw.get('model_state_dict', {})
        # EfficientNet 分类层权重
        for key in ['backbone._fc.3.weight', 'classifier.3.weight']:
            if key in state_dict:
                num_classes = state_dict[key].shape[0]
                break
        # ResNet 分类层权重
        if num_classes == 4:
            for key in list(state_dict.keys()):
                if 'classifier' in key and state_dict[key].dim() == 1:
                    num_classes = state_dict[key].shape[0]
                    break
                elif key.endswith('.weight') and state_dict[key].shape[0] < 1000:
                    num_classes = state_dict[key].shape[0]
                    break

    print(f"Creating model: {model_type} / {backbone} / num_classes={num_classes}")

    # 创建模型
    if model_type == 'resnet':
        model = create_resnet_model(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=False,
            dropout=0.3,
            use_attention=False
        )
    elif model_type == 'efficientnet':
        model = create_efficientnet_model(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=False
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 加载权重
    model.load_state_dict(checkpoint_raw['model_state_dict'])
    model = model.to(device)
    model.eval()
    if str(device).startswith('cuda'):
        torch.backends.cudnn.benchmark = True

    # 保存元数据
    model_meta = {
        'model_type': model_type,
        'backbone': backbone,
        'num_classes': num_classes,
        'class_names': class_names,
        'checkpoint_sha256': checkpoint_sha256,
    }
    model_meta['runtime_parity'] = runtime_parity(
        checkpoint_sha256=checkpoint_sha256,
        class_names=class_names,
        image_size=inference_image_size,
        confidence_threshold=confidence_threshold,
        provider_model=seed_vlm_client.config.model,
        prompt_version=SEED_PROMPT_VERSION,
        prompt_sha256=SEED_PROMPT_SHA256,
        raw_to_target=runtime_raw_to_target,
    )

    # 创建预处理变换（默认 256，可通过环境变量 ECOSORT_INFER_IMAGE_SIZE 覆盖）
    transform = transforms.Compose([
        ResizeLongestSideWithPadding(inference_image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print(f"Model loaded: {model_type} / {backbone} / classes={num_classes}")
    return f"Loaded {model_type} with {num_classes} classes"


@app.route('/')
def index():
    """主页 - 返回静态HTML文件"""
    return send_from_directory('static', 'index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy' if model is not None else 'degraded',
        'model_loaded': model is not None,
        'device': str(device) if device else None,
        'model_meta': model_meta,
        'class_count': len(class_names),
        'cors_origins': cors_origins,
        'api_base_hint': request.host_url.rstrip('/'),
        'runtime': {
            'infer_image_size': inference_image_size,
            'use_cuda_autocast': use_cuda_autocast,
            'torch_threads': int(torch.get_num_threads()),
            'confidence_threshold': confidence_threshold,
            'fallback_class': fallback_class,
            'mapping_config': mapping_config_path.name,
            'routing_strategy': 'confidence_only',
            'provider': seed_vlm_client.status(),
            'audited_parity': model_meta.get('runtime_parity', {
                'audited_parity': False,
                'reason': 'model_not_loaded',
            }),
        }
    })


@app.route('/metrics', methods=['GET'])
def runtime_metrics():
    """Operational counters only; never includes images, keys, or cache IDs."""
    return jsonify({
        'telemetry': runtime_telemetry.snapshot(),
        'provider': seed_vlm_client.status(),
        'runtime_parity': model_meta.get('runtime_parity', {
            'audited_parity': False,
            'reason': 'model_not_loaded',
        }),
    })


@app.route('/predict', methods=['POST'])
def predict():
    """单张图像分类

    Request:
        - image: base64 编码的图像字符串
        - format: 'base64' (默认)

    Returns:
        - class_name: 类别名称
        - class_id: 类别 ID (0-13)
        - confidence: 置信度
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        # 获取请求数据
        data = request.get_json(silent=True)

        if not isinstance(data, dict) or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data = data['image']
        if data.get('format') != 'base64':
            return jsonify({'error': 'Unsupported format'}), 400

        image = _decode_base64_image(image_data)
        response = _cascade_infer_image(image)

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量图像分类

    Request:
        - images: base64 编码的图像列表

    Returns:
        - predictions: 预测结果列表
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict) or 'images' not in data:
            return jsonify({'error': 'No images provided'}), 400

        images = data['images']
        if not isinstance(images, list):
            return jsonify({'error': 'images must be a list'}), 400
        if len(images) > max_batch_size:
            return jsonify({
                'error': f'batch exceeds ECOSORT_MAX_BATCH_SIZE={max_batch_size}'
            }), 413
        predictions = []

        for img_data in images:
            try:
                # 解码图像
                if ',' in img_data:
                    img_data = img_data.split(',')[1]

                image_bytes = base64.b64decode(img_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                predictions.append(_cascade_infer_image(image))

            except Exception as e:
                predictions.append({'error': str(e)})

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    """单张图片多目标候选裁剪并分别分类（低成本方案）"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        if data.get('format') != 'base64':
            return jsonify({'error': 'Unsupported format'}), 400

        max_objects = int(data.get('max_objects', 5))
        max_objects = max(1, min(max_objects, 8))

        image = _decode_base64_image(data['image'])
        main_result = _infer_image(image)

        candidates = _extract_candidate_crops(image, max_objects=max_objects)
        objects = []
        for idx, cand in enumerate(candidates):
            pred = _infer_image(cand['crop'])
            objects.append({
                'id': idx,
                'bbox_px': cand['bbox_px'],
                'bbox_norm': cand['bbox_norm'],
                'label': pred.get('daily_label') or pred.get('raw_top1_class') or pred.get('class_name'),
                'confidence': pred.get('confidence', 0.0),
                'prediction': pred,
            })

        return jsonify({
            'main': main_result,
            'objects': objects,
            'object_count': len(objects),
            'mode': 'lightweight_multi_object'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        'model_type': model_meta.get('model_type', 'unknown'),
        'backbone': model_meta.get('backbone', 'unknown'),
        'num_classes': model_meta.get('num_classes', len(class_names)),
        'class_names': model_meta.get('class_names', class_names),
        'daily_labels': DAILY_LABEL_ORDER,
        'missing_daily_labels': MISSING_DAILY_LABELS,
        'coarse_labels': COARSE_LABELS,
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'device': str(device)
    }

    return jsonify(info)


@app.route('/reload_model', methods=['POST'])
def reload_model():
    """重新加载模型

    Request:
        - model_path: 模型路径
        - model_type: 模型类型 (resnet/efficientnet)
    """
    if not allow_model_reload:
        return jsonify({'error': 'Model reload is disabled'}), 403

    try:
        data = request.get_json()
        model_path = data.get('model_path')
        model_type = data.get('model_type', 'resnet')

        if not model_path:
            return jsonify({'error': 'model_path is required'}), 400

        load_model(model_path, model_type)

        return jsonify({'status': 'success', 'message': 'Model reloaded'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """启动服务器"""
    import argparse

    parser = argparse.ArgumentParser(description='EcoSort Backend API')
    parser.add_argument('--model-path', type=str,
                        default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default=None,
                        choices=['resnet', 'efficientnet', None],
                        help='Model type (auto-detect if not specified)')
    parser.add_argument('--num-classes', type=int, default=14,
                        help='Number of classes')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port number')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode')

    args = parser.parse_args()

    load_runtime_mapping_config()

    # 加载模型
    print(f"Loading model from {args.model_path}...")
    msg = load_model(args.model_path, args.model_type, args.num_classes)
    print(msg)
    print("Model loaded successfully!")

    # 启动服务器
    print(f"\nStarting server on {args.host}:{args.port}")
    print("Available endpoints:")
    print("  GET  /")
    print("  GET  /health")
    print("  GET  /metrics")
    print("  POST /predict")
    print("  POST /batch_predict")
    print("  GET  /model_info")
    print("  POST /reload_model")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
