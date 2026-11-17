# 实验复现与数据版本

## 既有固定实验

2026 年 8 月实验复用已有 champion，没有重训。推理元数据记录 checkpoint SHA256、224 letterbox、14 类顺序、L20、FP16 与 warmup；validation 选阈值后应用 test。历史私有原图、完整 manifest、逐样本 CSV、provider cache 与永久调用账本保存在研究工作区，公开仓库只发布脱敏聚合成果。

旧数据规模：train=7210、validation=896、test=912。分别排除确认与 train 精确重复的一个 validation、一个 test 后，阈值分析 N=895，主测试 N=911；训练期验证集重复对 epoch 选择的潜在影响仍存在，不能声称彻底消除。其他近重复及标签风险见原报告。

在仓库根目录运行模块，避免脚本启动路径导致 `src` 无法导入：

```bash
python -m scripts.audit_ecosort_splits --data-root /path/to/frozen/splits \
  --output-dir runtime/reproduction/data_audit
python -m scripts.run_ecosort_local_inference --checkpoint /path/to/champion/best_model.pth \
  --data-root /path/to/frozen/splits --split test --device cpu \
  --output-dir runtime/reproduction/raw
```

CPU 可验证分类，不复现原 GPU 延迟数值。若要复核原延迟，应使用报告记录的环境与设备。以上输出采用新目录，不覆写冻结预测。

已有逐样本 CSV 与确认重复列表可使用 `scripts/analyze_ecosort_routing.py` 重新计算本地与路由指标。使用已有 Seed cache 的 `scripts/analyze_ecosort_vlm_cascade.py --provider seed` 可复核级联；必须显式指定冻结 prediction、overlap、VLM 目录并使用新输出目录/副本。不要对原永久 ledger/cache 执行清空、复制为新预算后重新调用等操作。Gemini pilot runner 是 Linux 历史工具，需要通过 `--backend-path` 提供保留了原 `_call_vlm_primary` 的旧源码或测试 stub；正式 Web 使用 Seed 适配器。

## 新建数据版本

```bash
python -m scripts.build_dataset_from_mapping \
  --raw-root /path/to/raw \
  --output-root /new/path/to/ecosort-next \
  --mapping-key daily_life_mapping_v6_15class --seed 42
```

构建器逐文件验证可读性，先按 SHA256 全局排除精确重复，再按类别切分 80%/10%/10%；重复文件跨类别冲突会报错，已有输出目录会被拒绝。默认复制文件，可显式选择 `--mode symlink`。输出 manifest 包含来源路径与 hash，属于私有数据，不能提交公开仓库。精确文件排重不能排除重编码、裁剪或来源相关的近重复，仍需来源分组与人工审计。

新构建器与收紧后的加载器会影响新训练的数据处理；新结果必须独立命名和报告，不可冒充逐比特复现历史 champion。

```bash
pip install -r requirements-training.txt
python -m experiments.train_baseline \
  --config configs/train_ecosort_v6_14class.yaml \
  --data-root /new/path/to/ecosort-next --exp-name thesis-next --no-wandb
python -m experiments.evaluate \
  --checkpoint checkpoints/thesis-next/best_model.pth \
  --data-root /new/path/to/ecosort-next --model-type efficientnet
```

评估从 checkpoint 读取类别顺序和输入尺寸，要求独立 `test/`；图像读取失败直接报错。以上为可选后续流程，本轮版本统一不启动训练、不生成新的精度数字。
