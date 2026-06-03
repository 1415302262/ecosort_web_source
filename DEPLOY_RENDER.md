# Render 固定域名部署（最终版）

## 结果
部署后可获得固定域名：`https://<your-service>.onrender.com`

## 步骤（你只需在 Render 控制台点一次）
1. 在 Render 选择 **Blueprint**，连接仓库 `1415302262/ecosort_web_source`。
2. Render 会自动识别 `render.yaml` 创建服务。
3. 在环境变量里设置：
   - `ECOSORT_MODEL_URL` = 你的模型直链下载地址（必须可直接下载 `.pth`）
4. 点击 Deploy。

## 验收
- 打开：`https://<your-service>.onrender.com/health`
- 看到：`status=healthy` 且 `model_loaded=true`

## 说明
- 代码仓库不放大模型文件；启动时通过 `ECOSORT_MODEL_URL` 自动下载。
- 这样可避免 GitHub 100MB 限制，同时保证服务稳定可重建。
