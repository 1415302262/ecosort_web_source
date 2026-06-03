import os

from backend.app import app, load_model, load_runtime_mapping_config


def _to_int(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


load_runtime_mapping_config()

MODEL_PATH = os.getenv(
    "ECOSORT_MODEL_PATH",
    "checkpoints/best_model.pth",
)
MODEL_TYPE = os.getenv("ECOSORT_MODEL_TYPE", "efficientnet")
NUM_CLASSES = _to_int(os.getenv("ECOSORT_NUM_CLASSES", "14"), 14)
AUTOLOAD = os.getenv("ECOSORT_AUTOLOAD_MODEL", "1").strip().lower() in {"1", "true", "yes", "on"}

if AUTOLOAD:
    try:
        msg = load_model(MODEL_PATH, MODEL_TYPE, NUM_CLASSES)
        print(f"[WSGI] {msg}")
    except Exception as exc:
        print(f"[WSGI] model autoload failed: {exc}")


if __name__ == "__main__":
    port = _to_int(os.getenv("PORT", "5000"), 5000)
    app.run(host="0.0.0.0", port=port)
