# Checkpoints

Large model weights are handled separately from normal Git files.

## GitHub Package

This clean package prepares one runtime checkpoint path:

- `checkpoints/best_model.pth`

The file is configured for Git LFS because the actual checkpoint is larger than GitHub's normal file-size limit.

## Local Serving

```bash
python backend/app.py \
  --model-path checkpoints/best_model.pth \
  --host 0.0.0.0 \
  --port 5000
```

## Deployment Alternative

For platforms where Git LFS is inconvenient, keep the model in private object storage or a Hugging Face file URL and set one of these environment variables:

- `ECOSORT_MODEL_URL` for the Flask/Render flow
- `HF_MODEL_URL` for the Hugging Face Space flow
