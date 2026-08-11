"""
EcoSort Dataset Module
支持动态类别与预切分目录的分类数据集实现
"""

import os
from typing import Callable, Dict, List, Optional, Set, Tuple
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from src.data.letterbox import ResizeLongestSideWithPadding


class TrashDataset(Dataset):
    """垃圾分类数据集

    支持两种目录结构:
    1) 未切分结构: root/class_name/*.jpg (通过 val_split 随机切分 train/val)
    2) 已切分结构: root/train|val|test/class_name/*.jpg (直接按 split 读取)
    """

    # 默认类别（向后兼容）
    DEFAULT_CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    CLASS_NAMES = DEFAULT_CLASS_NAMES

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        weak_class_transform: Optional[Callable] = None,
        weak_class_names: Optional[List[str]] = None,
        split: str = 'train',
        val_split: float = 0.2,
        seed: int = 42,
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            root_dir: 数据集根目录，应包含 class_name/xxx.jpg 结构
            transform: 图像变换
            split: 'train', 'val', 或 'test'
            val_split: 验证集比例
            seed: 随机种子
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.weak_class_transform = weak_class_transform
        self.split = split
        self.class_names = class_names
        self.class_to_idx = {}
        self.weak_class_indices: Set[int] = set()
        self.samples = []
        self.targets = []

        self._split_root = self.root_dir / split
        self.using_pre_split = self._split_root.exists() and self._split_root.is_dir()

        self._initialize_classes()
        if weak_class_names:
            self.weak_class_indices = {
                self.class_to_idx[name]
                for name in weak_class_names
                if name in self.class_to_idx
            }

        # 加载所有样本
        self._load_samples()

        # 划分数据集
        if not self.using_pre_split:
            self._split_data(val_split, seed)

        print(f"[{split.upper()}] Loaded {len(self.samples)} samples "
              f"from {len(self.class_names)} classes")

    def _initialize_classes(self):
        """初始化类别列表"""
        if self.class_names is not None:
            self.class_names = list(self.class_names)
        else:
            search_root = self._split_root if self.using_pre_split else self.root_dir
            discovered = sorted([
                p.name for p in search_root.iterdir()
                if p.is_dir() and not p.name.startswith('.')
            ]) if search_root.exists() else []

            if discovered:
                self.class_names = discovered
            else:
                self.class_names = list(self.DEFAULT_CLASS_NAMES)

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def _load_samples(self):
        """加载所有图像样本路径和标签"""
        base_dir = self._split_root if self.using_pre_split else self.root_dir

        for class_name in self.class_names:
            class_dir = base_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist, skipping...")
                continue

            # 支持多种图像格式
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in class_dir.glob(ext):
                    self.samples.append(str(img_path))
                    self.targets.append(self.class_to_idx[class_name])

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.root_dir}")

    def _split_data(self, val_split: float, seed: int):
        """划分训练/验证/测试集"""
        np.random.seed(seed)
        indices = np.arange(len(self.samples))
        np.random.shuffle(indices)

        # 划分训练集和验证集
        val_size = int(len(indices) * val_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        if self.split == 'train':
            selected_indices = train_indices
        elif self.split == 'val':
            selected_indices = val_indices
        else:  # test - 使用全部数据
            selected_indices = indices

        # 更新样本列表
        self.samples = [self.samples[i] for i in selected_indices]
        self.targets = [self.targets[i] for i in selected_indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: (C, H, W) tensor
            label: 整数标签 [0-3]
        """
        img_path = self.samples[idx]
        label = self.targets[idx]

        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回一个空白图像
            image = Image.new('RGB', (256, 256), color='white')

        selected_transform = self.transform
        if (
            self.split == 'train'
            and self.weak_class_transform is not None
            and label in self.weak_class_indices
        ):
            selected_transform = self.weak_class_transform

        if selected_transform:
            image = selected_transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        distribution = {name: 0 for name in self.class_names}
        for label in self.targets:
            class_name = self.class_names[label]
            distribution[class_name] += 1
        return distribution


def get_data_transforms(
    mode: str = 'train',
    img_size: int = 256,
    strong_aug: bool = False,
    random_erasing_prob: float = 0.0
) -> transforms.Compose:
    """获取数据变换管道

    Args:
        mode: 'train', 'val', 或 'test'
        img_size: 目标图像尺寸
        strong_aug: 是否使用强数据增强（用于42类细粒度分类）

    Returns:
        transforms.Compose
    """
    # ImageNet 标准化参数
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if mode == 'train':
        if strong_aug:
            train_steps = [
                ResizeLongestSideWithPadding(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                normalize,
            ]
        else:
            train_steps = [
                ResizeLongestSideWithPadding(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                normalize,
            ]

        if random_erasing_prob > 0:
            train_steps.append(transforms.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.15)))

        return transforms.Compose(train_steps)
    else:
        # 验证/测试时不做增强
        return transforms.Compose([
            ResizeLongestSideWithPadding(img_size),
            transforms.ToTensor(),
            normalize,
        ])


def get_weak_class_transform(
    img_size: int = 256,
    random_erasing_prob: float = 0.35
) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform_steps = [
        ResizeLongestSideWithPadding(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.12),
        transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.25, p=0.4),
        transforms.ToTensor(),
        normalize,
    ]

    if random_erasing_prob > 0:
        transform_steps.append(transforms.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.18)))

    return transforms.Compose(transform_steps)


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 256,
    val_split: float = 0.2,
    class_names: Optional[List[str]] = None,
    strong_aug: bool = False,
    use_weighted_sampler: bool = False,
    sampler_class_boosts: Optional[Dict[str, float]] = None,
    random_erasing_prob: float = 0.0,
    weak_class_names: Optional[List[str]] = None,
    weak_class_extra_aug: bool = False,
    weak_class_random_erasing_prob: float = 0.35
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """创建训练和验证数据加载器

    Args:
        data_root: 数据集根目录
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        img_size: 图像尺寸
        val_split: 验证集比例
        class_names: 类别列表
        strong_aug: 是否使用强数据增强（用于42类细粒度分类）
        use_weighted_sampler: 是否使用加权采样（提升长尾类别召回）
        sampler_class_boosts: 对指定类别额外提升采样权重

    Returns:
        train_loader, val_loader
    """
    train_transform = get_data_transforms(
        'train',
        img_size,
        strong_aug=strong_aug,
        random_erasing_prob=random_erasing_prob
    )
    weak_transform = None
    if weak_class_extra_aug and weak_class_names:
        weak_transform = get_weak_class_transform(
            img_size=img_size,
            random_erasing_prob=weak_class_random_erasing_prob
        )

    train_dataset = TrashDataset(
        root_dir=data_root,
        transform=train_transform,
        weak_class_transform=weak_transform,
        weak_class_names=weak_class_names,
        split='train',
        val_split=val_split,
        class_names=class_names
    )

    val_dataset = TrashDataset(
        root_dir=data_root,
        transform=get_data_transforms('val', img_size),
        split='val',
        val_split=val_split,
        class_names=train_dataset.class_names
    )

    train_sampler = None
    train_shuffle = True
    if use_weighted_sampler:
        class_counts = train_dataset.get_class_distribution()
        sampler_class_boosts = sampler_class_boosts or {}
        sample_weights = [
            (
                1.0 / class_counts[train_dataset.class_names[label]]
            ) * float(sampler_class_boosts.get(train_dataset.class_names[label], 1.0))
            for label in train_dataset.targets
        ]

        if sampler_class_boosts:
            print("Weighted sampler class boosts:")
            for class_name, multiplier in sampler_class_boosts.items():
                if class_name in train_dataset.class_to_idx:
                    print(f"  {class_name:20s}: x{float(multiplier):.2f}")
                else:
                    print(f"  {class_name:20s}: skipped (not found)")

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_shuffle = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # 测试数据集
    dataset = TrashDataset(
        root_dir='data/raw',
        transform=get_data_transforms('train'),
        split='train'
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")

    # 测试数据加载
    img, label = dataset[0]
    print(f"Image shape: {img.shape}, Label: {label}, "
          f"Class: {dataset.CLASS_NAMES[label]}")
