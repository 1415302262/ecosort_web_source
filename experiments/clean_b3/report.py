import csv
import json
import sys
from datetime import datetime
from pathlib import Path

r = Path(sys.argv[1])
p = json.loads((r / "paired.json").read_text())
s = json.loads((r / "training_summary.json").read_text())
new, old = p["clean_b3"], p["deployed_b3"]
best = max(s["history"], key=lambda e: e["val_f1"])
history = [
    {
        k: v
        for k, v in item.items()
        if k in ("epoch", "val_f1", "val_acc", "lr", "train_loss", "val_loss")
    }
    for item in s["history"]
]
(r / "validation-history.json").write_text(json.dumps(history, indent=2))
elapsed = (
    datetime.strptime((r / "end.utc").read_text().strip(), "%Y-%m-%dT%H:%M:%SZ")
    - datetime.strptime((r / "start.utc").read_text().strip(), "%Y-%m-%dT%H:%M:%SZ")
).total_seconds()
lines = [
    "# EfficientNet-B3 清理数据重训验证",
    "",
    f"使用一张 {p['gpu']}，训练进程用时 {elapsed / 60:.1f} 分钟（包括训练、验证和保存，不含输入准备与最终评测）。共运行 {len(history)} 轮，按验证集 Macro-F1 选择第 {best['epoch'] + 1} 轮，选中值 {best['val_f1']:.2%}。",
    "",
    "| 模型 | Accuracy | Macro-F1 | 低于原复核阈值 0.85 |",
    "|---|---:|---:|---:|",
]
for title, m in (("新 B3", new), ("现役 B3", old)):
    lines.append(
        f"| {title} | {m['accuracy']:.2%} | {m['macro_f1']:.2%} | {m['route_count_at_085']}/911 |"
    )
lo, hi = p["paired_accuracy_delta_95pct_bootstrap"]
lines += [
    "",
    f"相对现役 B3，新模型改对 {p['corrected']} 张、改错 {p['regressed']} 张。准确率差的样本级配对 bootstrap 95% 区间：[{lo * 100:.2f}, {hi * 100:.2f}] 个百分点。",
    "",
    "数据与边界：固定清单（7,206/895/911），逐文件复算 SHA-256，与上次审计的清单一致。保留生产模型作为冻结对照，不以旧权重初始化新模型。训练沿用项目 V6 配置：ImageNet 预训练、14 类、224 像素、batch 24、AdamW、混合精度、加权采样、30 轮上限、连续 7 轮验证未改善则停止。测试只在验证选模结束后比较，两模型对相同输入都采用 GPU FP32，batch 32。原阈值仅用于描述请求比例，未针对新模型做标定，未实际调用云端复核。测试图与此前实验相同，不代替新场景盲测。单个种子的差异不能解释为清理数据的因果效应。",
    "",
    "| 类别 | 样本 | 新 F1 | 现役 F1 |",
    "|---|---:|---:|---:|",
]
for label, row in new["per_class"].items():
    if label not in ("accuracy", "macro avg", "weighted avg"):
        lines.append(
            f"| {label} | {int(row['support'])} | {row['f1-score']:.2%} | {old['per_class'][label]['f1-score']:.2%} |"
        )
lines += [
    "",
    "在原阈值 0.85 下，分别用每个模型自己的分类错误作为分母：",
    "",
    "| 模型 | 分类错误 | 其中送复核 | 未送复核的错误 |",
    "|---|---:|---:|---:|",
]
with (r / "predictions.csv").open(encoding="utf-8") as f:
    records = list(csv.DictReader(f))
for key, label in (("clean", "新 B3"), ("old", "现役 B3")):
    wrong = [x for x in records if x[key] != x["truth"]]
    caught = sum(float(x[key + "_conf"]) < 0.85 for x in wrong)
    lines.append(f"| {label} | {len(wrong)} | {caught} | {len(wrong) - caught} |")
lines += ["", "此表只检查路由是否覆盖错误，不假设云端模型能纠正所有送审样本。"]
(r / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
