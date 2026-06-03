# Quick Start

## 1) 安装环境

```bash
conda env create -f environment.yml
conda activate ecosort
pip install -r requirements.txt
```

## 2) 准备数据

将你的训练数据按类别放入：

```text
data/raw/<class_name>/*.jpg
```

可先参考 `data/sample/` 的目录结构。

## 3) 训练

```bash
python experiments/train_baseline.py \
  --config configs/ecosort_v5_14class_effb3_letterbox224.yaml \
  --data-root data/raw
```

## 4) 启动后端推理服务

```bash
python backend/app.py \
  --model-path checkpoints/best_model.pth \
  --host 0.0.0.0 --port 5000
```

## 5) 前端页面

打开 `backend/static/index.html`，按页面提示配置 API 地址并上传图片测试。
