# EcoSort 实验复现命令

工作目录：仓库根目录。

```bash
# 数据审计
conda run -n cptac \
  python scripts/audit_ecosort_splits.py

# 本地单图推理（本轮已执行 val / test）
PYTHONPATH=. conda run -n ecosort \
  python scripts/run_ecosort_local_inference.py \
  --split val --device cuda:2
PYTHONPATH=. conda run -n ecosort \
  python scripts/run_ecosort_local_inference.py \
  --split test --device cuda:3

# Baseline、路由、敏感性、坏例和图表
conda run -n cptac \
  python scripts/analyze_ecosort_routing.py

# Seed2.0-Lite 无费用调用计划（本轮已执行，固定为 142）
conda run -n ecosort \
  python scripts/run_ecosort_seed_vlm_fallback.py --max-calls 142

# 无网络安全回归
conda run -n cptac pytest -q tests/test_vlm_safety.py

# 真实调用（本轮已执行）：默认从权限受控文件
# ~/.config/ecosort/ark_api_key 读取 Ark key；也可通过
# ECOSORT_ARK_API_KEY_FILE 指向其他权限受控文件。
conda run -n ecosort \
  python scripts/run_ecosort_seed_vlm_fallback.py --execute --max-calls 1

# 只有 smoke 的 successful_this_run=1、failed_this_run=0，且 cache 中
# mapping/usage/timeout 字段通过审计后，才执行剩余最多 141 个样本：
conda run -n ecosort \
  python scripts/run_ecosort_seed_vlm_fallback.py --execute --max-calls 142
# 必须保持默认 output-dir，且不得删除 vlm_attempt_ledger.jsonl；
# 永久账本才会在崩溃、并发和配置变化下继续限制总量为 142。

# 真实调用后分析；无 cache 时只输出 not_evaluated
conda run -n cptac \
  python scripts/analyze_ecosort_vlm_cascade.py \
  --provider seed \
  --vlm-dir reports/ecosort_experiment_results/vlm_seed2_lite_260428
```

报告 PDF 的独立构建命令见 [REPORT_BUILD.md](REPORT_BUILD.md)。
