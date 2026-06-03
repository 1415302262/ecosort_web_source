---
title: EcoSort 14-Class Demo
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: hf_space/app.py
pinned: false
---

# EcoSort 14-Class Demo (Free HF Space)

这个 Space 适合免费托管演示：

- 固定域名：`https://<username>-<space-name>.hf.space`
- 不依赖随机 Cloudflare URL
- 首次启动可通过 `HF_MODEL_URL` 自动下载模型

## 必填环境变量（Space Settings -> Variables）

- `HF_MODEL_URL`：你的 `best_model.pth` 直链地址
- 可选：
  - `HF_MODEL_PATH`（默认已内置）
  - `HF_MODEL_TYPE=efficientnet`
  - `HF_NUM_CLASSES=14`

## 注意

- 免费 CPU Space 冷启动较慢，首次请求可能需要等待。
- 若不提供 `HF_MODEL_URL` 且仓库无模型文件，会显示模型初始化失败提示。
