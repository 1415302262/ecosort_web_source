"""
EcoSort Training Framework
完整的训练和评估框架
"""

import os
import json
import time
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import copy

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import numpy as np

from src.data.dataset import TrashDataset, get_data_transforms
from src.models.resnet_classifier import create_resnet_model
from src.models.efficientnet_classifier import create_efficientnet_model


class Trainer:
    """EcoSort 训练器

    功能:
    - 混合精度训练 (AMP)
    - 梯度累积
    - 早停
    - 学习率调度
    - WandB 日志记录
    - 完整状态保存
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict,
        checkpoint_dir: str = 'checkpoints',
        experiment_name: str = 'baseline',
        use_wandb: bool = True
    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 配置字典
            checkpoint_dir: 检查点保存目录
            experiment_name: 实验名称
            use_wandb: 是否使用 WandB
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 损失函数
        self.criterion = self._create_criterion()

        # 混合精度训练
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # 训练状态
        self.current_epoch = 0
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.monitor_metric = str(config.get('monitor_metric', 'val_f1'))
        self.monitor_mode = str(config.get('monitor_mode', 'max'))
        if self.monitor_metric not in {'val_f1', 'val_acc', 'val_loss'}:
            raise ValueError(
                f"Unsupported monitor_metric: {self.monitor_metric}. "
                "Choose from ['val_f1', 'val_acc', 'val_loss']"
            )
        if self.monitor_mode not in {'max', 'min'}:
            raise ValueError("monitor_mode must be 'max' or 'min'")
        self.best_metric = -math.inf if self.monitor_mode == 'max' else math.inf
        self.patience_counter = 0
        self.history: List[Dict[str, float]] = []

        # WandB 初始化
        if use_wandb:
            import wandb
            wandb_mode = os.getenv('WANDB_MODE', 'offline')
            wandb.init(
                project='ecosort-classification',
                name=experiment_name,
                config=config,
                mode=wandb_mode
            )
            self.wandb = wandb
            print(f"WandB mode: {wandb_mode}")
        else:
            self.wandb = None

        print(f"Trainer initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Monitoring best checkpoint by {self.monitor_metric} ({self.monitor_mode})")

    def _is_better(self, current_value: float) -> bool:
        if self.monitor_mode == 'max':
            return current_value > self.best_metric
        return current_value < self.best_metric

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        opt_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)

        if opt_type == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_type == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

    def _create_scheduler(self) -> Optional[object]:
        """创建学习率调度器"""
        scheduler_type = self.config.get('scheduler', 'cosine')

        if scheduler_type == 'cosine':
            epochs = self.config.get('epochs', 50)
            return CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            step_size = self.config.get('step_size', 10)
            gamma = self.config.get('gamma', 0.1)
            return StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        else:
            return None

    def _create_criterion(self) -> nn.Module:
        """创建损失函数（支持类别加权）"""
        loss_type = self.config.get('loss', {}).get('type', 'cross_entropy')
        data_config = self.config.get('data', {})
        use_weighted_sampler = data_config.get('use_weighted_sampler', False)

        class_weights = None
        if data_config.get('use_class_weights', False):
            class_counts = data_config.get('class_counts', [])
            if class_counts:
                total = sum(class_counts)
                num_classes = len(class_counts)
                class_weights = torch.tensor([
                    total / (num_classes * count) for count in class_counts
                ], dtype=torch.float32)
                class_weights = class_weights / class_weights.sum() * num_classes

                print(f"\n使用类别加权损失:")
                class_names = data_config.get(
                    'class_names',
                    [f'class_{i}' for i in range(len(class_counts))]
                )
                for i, (name, count, weight) in enumerate(zip(
                    class_names,
                    class_counts,
                    class_weights
                )):
                    print(f"  {name:12s}: count={count:4d}, weight={weight:.3f}")

                class_weights = class_weights.to(self.device)

        if use_weighted_sampler and class_weights is not None:
            print("\n检测到 weighted sampler + class weights 同时开启，"
                  "为避免过度重加权，loss 中将禁用 class weights。")
            class_weights = None

        if loss_type == 'cross_entropy':
            label_smoothing = self.config.get('loss', {}).get('label_smoothing', 0.0)
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        elif loss_type == 'focal':
            gamma = self.config.get('loss', {}).get('gamma', 2.0)

            class FocalLoss(nn.Module):
                def __init__(self, alpha=None, gamma=2.0):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma

                def forward(self, logits, targets):
                    log_probs = nn.functional.log_softmax(logits, dim=1)
                    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                    pt = log_pt.exp()

                    ce = -log_pt
                    if self.alpha is not None:
                        alpha_t = self.alpha.gather(0, targets)
                        ce = alpha_t * ce

                    loss = ((1 - pt) ** self.gamma) * ce
                    return loss.mean()

            return FocalLoss(alpha=class_weights, gamma=gamma)
        else:
            raise ValueError(f"Unsupported loss: {loss_type}")

    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch

        Returns:
            metrics: 包含 loss, acc 等指标的字典
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # 统计
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        # 计算 epoch 指标
        avg_loss = total_loss / total
        accuracy = correct / total

        metrics = {
            'train_loss': avg_loss,
            'train_acc': accuracy
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证模型

        Returns:
            metrics: 验证指标
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # 用于计算 F1-score
        all_preds = []
        all_labels = []

        for images, labels in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # 统计
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 收集预测结果
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # 计算指标
        avg_loss = total_loss / total
        accuracy = correct / total

        # 计算 F1-score
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro')

        metrics = {
            'val_loss': avg_loss,
            'val_acc': accuracy,
            'val_f1': f1
        }

        return metrics

    def train(self):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"Starting training: {self.experiment_name}")
        print(f"{'='*60}\n")

        epochs = self.config.get('epochs', 50)
        patience = self.config.get('early_stopping_patience', 10)

        for epoch in range(self.start_epoch, epochs):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()

            # 打印指标
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_acc']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.4f}, "
                  f"Val F1: {val_metrics['val_f1']:.4f}")

            # WandB 日志
            if self.wandb is not None:
                self.wandb.log({
                    **train_metrics,
                    **val_metrics,
                    'epoch': epoch,
                    'lr': self.optimizer.param_groups[0]['lr']
                })

            current_metric = float(val_metrics[self.monitor_metric])
            self.history.append({
                'epoch': int(epoch),
                **{k: float(v) for k, v in train_metrics.items()},
                **{k: float(v) for k, v in val_metrics.items()},
                'lr': float(self.optimizer.param_groups[0]['lr'])
            })

            # 保存最佳模型
            if self._is_better(current_metric):
                self.best_metric = current_metric
                self.best_val_acc = float(val_metrics['val_acc'])
                self.best_val_f1 = float(val_metrics['val_f1'])
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
                print(
                    f"✓ Saved best model "
                    f"({self.monitor_metric}: {self.best_metric:.4f}, "
                    f"Val Acc: {self.best_val_acc:.4f}, Val F1: {self.best_val_f1:.4f})"
                )
            else:
                self.patience_counter += 1

            # 早停
            if self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best {self.monitor_metric}: {self.best_metric:.4f}")
        print(f"Best Val Acc: {self.best_val_acc:.4f}")
        print(f"Best Val F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}\n")

        # 保存训练历史
        self.save_training_summary()

    def resume_from_checkpoint(self, checkpoint_path: str):
        """从完整状态检查点恢复训练"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if checkpoint.get('optimizer_state_dict') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', -1)
        self.start_epoch = self.current_epoch + 1
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        ckpt_monitor_metric = checkpoint.get('monitor_metric')
        ckpt_monitor_mode = checkpoint.get('monitor_mode')
        if ckpt_monitor_metric is not None:
            self.monitor_metric = ckpt_monitor_metric
        if ckpt_monitor_mode is not None:
            self.monitor_mode = ckpt_monitor_mode
        default_best = self.best_val_f1 if self.monitor_metric == 'val_f1' else self.best_val_acc
        self.best_metric = checkpoint.get('best_metric', default_best)
        self.history = checkpoint.get('history', [])
        self.patience_counter = 0

        print(f"Resumed from checkpoint: {checkpoint_path}")
        print(f"Resume start epoch: {self.start_epoch}")
        print(f"Best {self.monitor_metric} so far: {self.best_metric:.4f}")
        print(f"Best Val Acc so far: {self.best_val_acc:.4f}")
        print(f"Best Val F1 so far: {self.best_val_f1:.4f}")

    def save_checkpoint(self, filename: str):
        """保存完整状态检查点

        检查点包含:
        - model_state_dict
        - optimizer_state_dict
        - scheduler_state_dict
        - epoch
        - best_val_acc
        - best_val_f1
        - config
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'best_metric': self.best_metric,
            'monitor_metric': self.monitor_metric,
            'monitor_mode': self.monitor_mode,
            'history': self.history,
            'config': self.config
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)

    def save_training_summary(self):
        """保存训练总结到 JSON"""
        summary = {
            'experiment_name': self.experiment_name,
            'monitor_metric': self.monitor_metric,
            'monitor_mode': self.monitor_mode,
            'best_metric': float(self.best_metric),
            'best_val_acc': float(self.best_val_acc),
            'best_val_f1': float(self.best_val_f1),
            'total_epochs': self.current_epoch + 1,
            'history': self.history,
            'config': self.config
        }

        summary_path = self.checkpoint_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved training summary to {summary_path}")


def load_checkpoint(checkpoint_path: str, model: nn.Module) -> Dict:
    """加载检查点

    Args:
        checkpoint_path: 检查点文件路径
        model: 模型实例

    Returns:
        checkpoint: 检查点字典
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Best Val Acc: {checkpoint['best_val_acc']:.4f}")
    if 'best_val_f1' in checkpoint:
        print(f"Best Val F1: {checkpoint['best_val_f1']:.4f}")
    if 'monitor_metric' in checkpoint and 'best_metric' in checkpoint:
        print(
            f"Best {checkpoint['monitor_metric']}: "
            f"{checkpoint['best_metric']:.4f} ({checkpoint.get('monitor_mode', 'max')})"
        )

    return checkpoint


if __name__ == '__main__':
    # 测试训练器
    from src.data.dataset import create_dataloaders

    # 创建数据加载器 (假设数据在 data/raw)
    try:
        train_loader, val_loader = create_dataloaders(
            data_root='data/raw',
            batch_size=8,
            num_workers=2,
            img_size=256
        )
    except Exception as e:
        print(f"Warning: Could not create dataloaders: {e}")
        print("Creating dummy dataloaders for testing...")

        # 创建虚拟数据
        from torch.utils.data import TensorDataset, DataLoader

        dummy_train = TensorDataset(
            torch.randn(100, 3, 256, 256),
            torch.randint(0, 4, (100,))
        )
        dummy_val = TensorDataset(
            torch.randn(20, 3, 256, 256),
            torch.randint(0, 4, (20,))
        )

        train_loader = DataLoader(dummy_train, batch_size=8, shuffle=True)
        val_loader = DataLoader(dummy_val, batch_size=8)

    # 创建模型
    model = create_resnet_model(
        backbone='resnet50',
        num_classes=4,
        pretrained=False
    )

    # 配置
    config = {
        'epochs': 2,
        'learning_rate': 1e-3,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'use_amp': True,
        'early_stopping_patience': 5,
    }

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        experiment_name='test',
        use_wandb=False
    )

    # 开始训练
    trainer.train()
