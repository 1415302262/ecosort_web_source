# CLAUDE.md - EcoSort 垃圾分类项目协议

## 项目概述
**EcoSort** - 基于深度学习的智能垃圾分类系统
- **任务类型**: 多类别图像分类
- **技术栈**: PyTorch + ResNet/EfficientNet + WandB
- **目标**: 实现高精度的可回收/有害/厨余/其他垃圾分类

## 核心哲学
- **代码 (Git)**: 增量开发，修改优于新增
- **产出 (DVC)**: 完整实验状态 (model + optimizer + metrics)
- **实验 (WandB)**: 指标追踪与可视化
- **日志 (Logs)**: 详细运行记录

## 目录结构
```
ecosort/
├── data/
│   ├── raw/          # 原始垃圾图像数据集
│   └── proc/         # 预处理后的数据 (分片/增强)
├── checkpoints/      # DVC 追踪：完整实验状态
├── src/              # 核心模块 (data, models, train, utils)
├── experiments/      # 实验入口脚本
├── configs/          # YAML 配置文件
├── logs/             # 训练日志
├── .dvc/             # DVC 配置
└── environment.yml   # Conda 环境
```

## 代码开发规范
- **DRY 原则**: 通用函数抽取到 `src/`
- **原子提交**: `git commit -m "feat: [描述]"`
- **全状态保存**: checkpoint 必须包含 optimizer + scheduler + epoch

## 垃圾分类类别
```yaml
categories:
  - recyclable    # 可回收物 (塑料/纸张/金属/玻璃)
  - hazardous     # 有害垃圾 (电池/药品/灯管)
  - kitchen       # 厨余垃圾 (食物残渣/果皮)
  - other         # 其他垃圾 (污染纸/陶瓷)
```

## 实验追踪
- **WandB Project**: `ecosort-classification`
- **Metrics**: Accuracy, F1-Score, Confusion Matrix
- **Artifacts**: 仅 Champion 模型

## 快捷命令
- `@setup-dvc-remote`: 配置 DagsHub 远程存储
- `@snapshot`: 提交实验结果 (Git + DVC)
- `@sync`: 同步远程仓库
- `@clean-server`: 清理缓存
