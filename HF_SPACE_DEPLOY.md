# Hugging Face 免费部署（无 Render 付费）

你现在可以直接部署到 Hugging Face Spaces（免费 CPU）。

## 你只需要做这 4 步

1. 打开 https://huggingface.co/new-space
2. Space SDK 选择 **Gradio**，仓库可设为 `ecosort-14class-demo`
3. 上传本仓库全部文件（必须包含 `backend/`、`src/`、`ecosort_mapping.yaml`、`hf_space/`）
4. 在 Space 的 `Settings -> Variables` 添加：
   - `HF_MODEL_URL=<你的模型直链>`

部署成功后会得到固定域名：

- `https://<your-username>-ecosort-14class-demo.hf.space`

## 为什么免费可行

- Hugging Face 免费层支持公开 CPU Space。
- 你不需要付 Render 的固定资源费用。
- 模型不放仓库，启动时下载，规避 GitHub 大文件限制。
