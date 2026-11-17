"""Install the current checkout as a loopback-only, no-paid-cloud user service."""

import argparse
import hashlib
import os
import secrets
import shlex
import socket
import subprocess
from pathlib import Path

from src.inference.cascade import AUDITED_CHECKPOINT_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=5187)
    parser.add_argument(
        "--state-dir", default=str(Path.home() / ".local/state/ecosort")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint).resolve()
    with checkpoint.open("rb") as handle:
        if (
            hashlib.file_digest(handle, "sha256").hexdigest()
            != AUDITED_CHECKPOINT_SHA256
        ):
            raise SystemExit("Checkpoint SHA256 does not match the audited model")
    config = Path.home() / ".config/ecosort"
    config.mkdir(mode=0o700, parents=True, exist_ok=True)
    env_path = config / "app.env"
    if env_path.exists():
        raise SystemExit(
            "app.env already exists; preserve it and use the documented update procedure"
        )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    admin_path = config / "admin.token"
    if not admin_path.exists():
        fd = os.open(admin_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(secrets.token_urlsafe(36))
    release = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    state = Path(args.state_dir).resolve()
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    settings = {
        "ECOSORT_MODEL_PATH": str(checkpoint),
        "ECOSORT_STATE_DIR": str(state),
        "ECOSORT_ADMIN_TOKEN_FILE": str(admin_path),
        "ECOSORT_RELEASE": release,
        "ECOSORT_VLM_EXECUTE": "0",
        "ECOSORT_VLM_MAX_CALLS": "0",
        "ECOSORT_VLM_STATE_DIR": str(state / "vlm"),
        "ECOSORT_INFER_IMAGE_SIZE": "224",
        "ECOSORT_TORCH_THREADS": "2",
        "ECOSORT_TORCH_INTEROP_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    # EnvironmentFile quoting is compatible with these quoted path values.
    with env_path.open("x", encoding="utf-8") as handle:
        for key, value in settings.items():
            if any(c in value for c in ("\n", "\r", '"', "\\")):
                raise SystemExit("Unsupported character in deployment configuration")
            handle.write(f'{key}="{value}"\n')
    env_path.chmod(0o600)
    unit_dir = Path.home() / ".config/systemd/user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit = unit_dir / "ecosort.service"
    if unit.exists():
        raise SystemExit("ecosort.service already exists; refusing to overwrite")
    # This installer targets Linux absolute paths without whitespace.
    if any(c.isspace() for c in str(root) + args.python + str(env_path)):
        raise SystemExit("Deployment paths must not contain whitespace")
    unit.write_text(
        f"""[Unit]
Description=EcoSort undergraduate thesis application
After=network.target

[Service]
Type=simple
WorkingDirectory={root}
EnvironmentFile={env_path}
ExecStart={args.python} -m gunicorn backend.wsgi:app --bind 127.0.0.1:{args.port} --workers 1 --threads 4 --timeout 120
Restart=on-failure
RestartSec=5
UMask=0077
MemoryMax=1536M

[Install]
WantedBy=default.target
""",
        encoding="utf-8",
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", "ecosort.service"], check=True
    )
    print(
        f"Service installed; release={release}; port={args.port}; state={state}; admin_token_file={admin_path}"
    )


if __name__ == "__main__":
    main()
