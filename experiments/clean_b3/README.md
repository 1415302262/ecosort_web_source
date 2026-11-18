# 复现实验

本实验需要一张 CUDA 卡，训练和评测对源图像目录只读；候选权重与正式服务分开保存。Python 依赖见仓库的训练环境。若需离线训练，把 ImageNet 初始化文件 efficientnet-b3-5fb5a3c3.pth 预先放进 `$TORCH_HOME/hub/checkpoints/`。本次该文件完整 SHA-256 为 `5fb5a3c3f1d45062d3d3dbe770f1a7daf78eed140f1c092cceef9ab79677d08e`。

从仓库根目录执行。设置 MANIFEST 为已核对的上轮 split 清单（build_clean_manifest.py 输出的 manifest.csv），SOURCE 为原始 14 类 train/val/test 目录，OLD 为现役 B3 权重，OUT 为新的、非公开的空实验目录。运行前确认 GPU 可用；如共享服务器有作业调度器，按其规则申请单卡。

```bash
set -eu
umask 077
export PYTHONPATH=.
mkdir -p "$OUT"
python experiments/clean_b3/prepare.py "$MANIFEST" "$SOURCE" "$OUT/data"
date -u +%FT%TZ > "$OUT/start.utc"
python experiments/train_baseline.py --config configs/train_ecosort_v6_14class.yaml --data-root "$OUT/data" --exp-name clean_b3 --checkpoint-dir "$OUT/checkpoints" --no-wandb
date -u +%FT%TZ > "$OUT/end.utc"
python experiments/clean_b3/paired.py "$OUT/data" "$OUT/checkpoints/clean_b3/best_model.pth" "$OLD" "$OUT"
cp "$OUT/checkpoints/clean_b3/training_summary.json" "$OUT/training_summary.json"
python experiments/clean_b3/verify.py "$OUT" "$MANIFEST"
python experiments/clean_b3/report.py "$OUT"
```

prepare.py 固定本轮清单哈希，并逐图核验，拒绝再次使用已有目标样本目录。verify.py 从逐图预测独立重算汇总指标。正式服务的权重、阈值和任何云端配置不由这些脚本改动。
生成训练曲线和逐类 F1 对照图： `python experiments/clean_b3/plot_results.py "$OUT"` （需要 matplotlib）。
