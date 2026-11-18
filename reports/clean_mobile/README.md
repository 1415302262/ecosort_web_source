# 14 类样本审计与轻量模型对照

本次保留现役 B3。轻量模型适合继续做低延迟实验，但在同一批 911 张图上的分类效果还不足以替换它。没有发起云端模型调用。

| 模型 | Accuracy | Macro-F1 | 参数量 | CPU 前向 P50 / P95 | L20 前向 P50 / P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 现役 EfficientNet-B3，冻结权重 | 91.44% | 91.39% | 11,490,358 | 48.95 / 51.57 ms | 12.28 / 12.37 ms |
| MobileNetV3-Small，独立训练 | 82.44% | 80.98% | 1,532,206 | 6.56 / 7.41 ms | 3.41 / 3.45 ms |

样本与训练：按文件内容 SHA-256 统计。两个内容相同但分别标为玻璃和塑料的组涉及 4 条训练记录，全部隔离；验证集、测试集各排除 1 条精确重复。原始文件未改写。保留训练 7,206、验证 895、测试 911。此规则不识别裁剪、转码等近重复。B3 是已有权重，没有按新清单重训，其历史选模仍可能受重复样本影响。这次对比回答的是本次训练的候选能否替换现役模型，不能据此断定两种架构的上限。

来源：对照 [TrashNet 作者发布的图像包](https://github.com/garythung/trashnet/blob/master/data/dataset-resized.zip)，先核对压缩包 Git blob，再逐图比对 SHA-256。9,018 条原始记录中有 1,717 条完全一致：玻璃 435、金属 410、纸类 390、塑料 482。其余 7,301 条未匹配这个版本，不能据此判为自采或自行标注。文件夹只表示服务器保存位置。逐图片名与清单留在实验目录，不发布图片。

方法：使用 torchvision 的 ImageNet 初始化，224 像素等比填充、batch 64、AdamW、学习率 0.00035、标签平滑 0.03，以验证集 Macro-F1 选择六个 epoch 中的最佳权重，测试集不参与选模。两模型在同一清单上使用 FP32；历史主报告用过 FP16，数值口径不同。CPU 固定两线程，GPU 使用一张 L20；前向耗时为单张输入、预热 10 次后连续测量 100 次，不含上传、图片解码、数据库及云端请求。CPU 使用固定形状的合成输入。

## 复现

在仓库根目录运行，DATA 指向现有 14 类 train/val/test 目录，B3 指向已审计权重，TRASHNET 指向服务器已有的 trashnet-master.zip，OUT 取新的实验目录。仅训练脚本需要一张可用 GPU。训练依赖为 torchvision、torch、Pillow、numpy、scikit-learn 和项目已有模型依赖；本次使用 torch 2.5.1+cu121、torchvision 0.20.1+cu121。

```sh
export PYTHONPATH=.
python scripts/build_clean_manifest.py "$DATA" "$OUT"
python scripts/verify_trashnet_origin.py "$OUT/manifest.csv" "$TRASHNET" "$OUT"
curl -fL https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth -o "$OUT/init.pth"
python scripts/compare_compact_model.py --root "$DATA" --manifest "$OUT/manifest.csv" --weights "$OUT/init.pth" --baseline "$B3" --out "$OUT"
python scripts/benchmark_compact_cpu.py --baseline "$B3" --compact "$OUT/compact.pth" --out "$OUT/cpu.json"
```

本目录包含逐类指标、验证历史、基线权重哈希、清单哈希和计时结果。预测来自现有数据划分；还需独立实拍样本检验真实投放场景。
