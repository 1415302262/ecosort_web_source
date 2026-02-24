"""
EcoSort 模型优化工具
支持量化、剪枝等优化技术
"""

import torch
import torch.nn as nn
from typing import Callable
import copy


def post_training_quantization(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader = None
) -> nn.Module:
    """训练后量化 (动态量化)

    将模型从 FP32 转换为 INT8，减少模型大小和推理时间

    Args:
        model: 原始模型
        calibration_loader: 校准数据加载器 (可选)

    Returns:
        quantized_model: 量化后的模型
    """
    print("Starting post-training quantization...")

    # 复制模型
    model = copy.deepcopy(model)
    model.eval()

    # 动态量化 (适用于 LSTM, Linear 等)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},  # 要量化的层类型
        dtype=torch.qint8
    )

    print("Quantization completed!")
    return quantized_model


def static_quantization(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """静态量化

    需要校准数据集来确定量化参数

    Args:
        model: 原始模型
        calibration_loader: 校准数据加载器
        device: 设备

    Returns:
        quantized_model: 量化后的模型
    """
    print("Starting static quantization...")

    # 复制模型
    model = copy.deepcopy(model).to(device)
    model.eval()

    # 配置量化
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # 准备量化
    prepared_model = torch.quantization.prepare(model, inplace=True)

    # 校准
    print("Calibrating with data...")
    with torch.no_grad():
        for images, _ in calibration_loader:
            images = images.to(device)
            _ = prepared_model(images)
            break  # 只需要一个 batch

    # 转换为量化模型
    quantized_model = torch.quantization.convert(prepared_model, inplace=True)

    print("Static quantization completed!")
    return quantized_model


def compare_model_size(original_model: nn.Module, quantized_model: nn.Module):
    """比较原始模型和量化模型的大小"""
    import tempfile
    import os

    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.save(original_model.state_dict(), tmp.name)
        original_size = os.path.getsize(tmp.name) / (1024 * 1024)  # MB
        os.unlink(tmp.name)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.save(quantized_model.state_dict(), tmp.name)
        quantized_size = os.path.getsize(tmp.name) / (1024 * 1024)  # MB
        os.unlink(tmp.name)

    print(f"\n{'='*60}")
    print("Model Size Comparison:")
    print(f"  Original:  {original_size:.2f} MB")
    print(f"  Quantized: {quantized_size:.2f} MB")
    print(f"  Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    print(f"{'='*60}\n")


def benchmark_inference_time(
    model: nn.Module,
    input_size: tuple = (1, 3, 256, 256),
    device: torch.device = torch.device('cpu'),
    num_runs: int = 100
):
    """基准测试推理时间"""
    import time

    model = model.to(device)
    model.eval()

    # 创建随机输入
    x = torch.randn(input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    # 测试
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # ms

    print(f"\n{'='*60}")
    print(f"Inference Time ({num_runs} runs):")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  FPS: {1000/avg_time:.1f}")
    print(f"{'='*60}\n")


def prune_model(
    model: nn.Module,
    pruning_ratio: float = 0.3
) -> nn.Module:
    """模型剪枝

    移除不重要的权重，减少模型参数量

    Args:
        model: 原始模型
        pruning_ratio: 剪枝比例

    Returns:
        pruned_model: 剪枝后的模型
    """
    import torch.nn.utils.prune as prune

    print(f"Starting model pruning (ratio={pruning_ratio})...")

    model = copy.deepcopy(model)
    model.eval()

    # 对所有卷积层和线性层进行剪枝
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            prune.remove(module, 'weight')

    print("Pruning completed!")
    return model


if __name__ == '__main__':
    from src.models.resnet_classifier import create_resnet_model
    from torch.utils.data import TensorDataset, DataLoader

    # 创建模型
    model = create_resnet_model(
        backbone='resnet50',
        num_classes=4,
        pretrained=False
    )

    # 量化
    quantized_model = post_training_quantization(model)

    # 比较大小
    compare_model_size(model, quantized_model)

    # 基准测试
    benchmark_inference_time(model, device=torch.device('cpu'))
    benchmark_inference_time(quantized_model, device=torch.device('cpu'))
