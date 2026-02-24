# EcoSort 模型优化操作报告

**日期**: 2026-02-16
**目标**: 提升垃圾识别准确率从78%至85%+

---

## 一、问题诊断

### 当前模型性能 (crawled42_resnet50)
- **验证准确率**: 78.38%
- **验证F1分数**: 75.86%
- **类别数量**: 42类细粒度分类

### 主要问题分析

1. **类别过多导致混淆**
   - 42类细粒度分类（bag, battery, bone, brick, ...）
   - 相似类别难以区分（plastic vs plastic_bag vs plastic_toy）

2. **数据不均衡**
   - 最少类别: trash (95张)
   - 最多类别: paper (415张)
   - 数据量差异4.4倍

3. **数据增强不足**
   - 未使用RandomErasing、RandomPerspective等现代增强技术
   - 对细粒度分类任务增强不够

4. **训练策略可优化**
   - 学习率策略可能未达最优
   - Dropout率偏小(0.3)

---

## 二、优化方案

### 2.1 配置优化 (configs/optimized_resnet50.yaml)

| 参数 | 原值 | 新值 | 原因 |
|------|------|------|------|
| dropout | 0.3 | 0.4 | 增加正则化防止过拟合 |
| img_size | 256 | 256 | 保持高分辨率 |
| batch_size | 32 | 24 | 适应更大图片+强增强 |
| learning_rate | 0.0005 | 0.0003 | 降低初始lr提高稳定性 |
| epochs | 30 | 60 | 增加训练轮数充分收敛 |
| early_stopping_patience | 8 | 12 | 增加耐心值避免早停 |
| val_split | 0.2 | 0.15 | 减少验证集增加训练数据 |

### 2.2 数据增强强化

**新增增强技术**:
- `RandomPerspective(distortion_scale=0.2, p=0.3)` - 随机透视变换
- `RandomErasing(p=0.2, scale=(0.02, 0.15))` - 随机擦除
- `hue: 0.1` - 颜色调色
- 更大的旋转角度 (20° vs 15°)
- 更强的翻转概率 (vertical: 0.3 vs 0.2)

### 2.3 代码修改

#### 修改文件: `src/data/dataset.py`

1. **新增参数** `strong_aug: bool = False` 到 `get_data_transforms()`
2. **新增强增强分支** 用于42类细粒度分类
3. **传递参数** `create_dataloaders()` 支持 `strong_aug` 参数

#### 修改文件: `experiments/train_baseline.py`

- 自动检测配置中的 `random_erasing_prob` 判断是否使用强增强
- 传递 `strong_aug` 参数到数据加载器

---

## 三、执行记录

### 3.1 文件变更

```bash
# 新增文件
configs/optimized_resnet50.yaml           # 优化配置文件
scripts/train_optimized.sh                # 训练启动脚本
scripts/watch_training.sh                 # 训练监控脚本

# 修改文件
src/data/dataset.py                       # 添加强增强支持
experiments/train_baseline.py             # 支持强增强参数
```

### 3.2 训练启动

```bash
# 启动命令
nohup bash scripts/train_optimized.sh > /dev/null 2>&1 &

# 监控命令
bash scripts/watch_training.sh
```

**训练状态**: ✅ 运行中 (PID: 2014598)
**预计完成时间**: ~2小时 (60 epochs, 42类)

---

## 四、预期效果

### 目标指标

| 指标 | 当前 | 目标 |
|------|------|------|
| 验证准确率 | 78.38% | **85%+** |
| 验证F1分数 | 75.86% | **82%+** |
| 训练稳定性 | 中等 | **高** |

### 改进原理

1. **强数据增强** → 提升泛化能力
2. **更高Dropout** → 减少过拟合
3. **更低学习率** → 更稳定的收敛
4. **更多训练轮数** → 充分学习特征
5. **类别权重** → 平衡数据不均衡

---

## 五、后续计划

### 5.1 监控训练

```bash
# 实时监控
watch -n 10 'bash scripts/watch_training.sh'

# 查看完整日志
tail -f logs/train_optimized_*.log
```

### 5.2 验证结果

训练完成后运行测试:

```bash
conda activate ecosort
python scripts/evaluate_model.py \
    --checkpoint checkpoints/crawled42_optimized/best_model.pth \
    --data-root data/proc/crawled_v1
```

### 5.3 备选方案

如果优化后仍未达到85%，可尝试：

1. **使用EfficientNet-B4** 更强backbone
2. **添加注意力机制** SE模块
3. **知识蒸馏** 从更大模型学习
4. **集成学习** 多模型投票

---

## 六、代码检查清单

- [x] 配置文件已创建 (`configs/optimized_resnet50.yaml`)
- [x] 数据增强已增强 (`src/data/dataset.py`)
- [x] 训练脚本已更新 (`experiments/train_baseline.py`)
- [x] 训练已启动 (PID: 2014598)
- [x] 监控脚本已创建 (`scripts/watch_training.sh`)
- [ ] 等待训练完成 (~2小时)
- [ ] 验证模型性能
- [ ] 提交结果到Git+DVC (@snapshot)

---

## 七、关键命令速查

```bash
# 监控训练进度
bash scripts/watch_training.sh

# 查看GPU使用
nvidia-smi -l 1

# 检查训练进程
ps aux | grep train_baseline

# 停止训练（如需要）
pkill -f train_baseline.py

# 查看完整日志
tail -100 logs/train_optimized_*.log
```

---

**操作人员**: Claude
**状态**: 训练进行中
**下一步**: 等待训练完成后验证性能
