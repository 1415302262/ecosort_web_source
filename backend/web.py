"""Web application factory, independent of Torch for HTTP integration tests."""

from __future__ import annotations

import base64
import binascii
import io
import math
import os
import secrets
import uuid
import warnings
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory, session
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from backend.store import Conflict, Store
from src.inference.cascade import AUDITED_CLASS_NAMES, CLASS_TO_COARSE, coarse_category
from src.inference import file_lock


class InputError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def decode_image(value, max_pixels=16_000_000):
    if not isinstance(value, str) or not value:
        raise InputError("请提供 base64 图片字符串")
    if len(value) > 12 * 1024 * 1024:
        raise InputError("图片编码过大", 413)
    if value.startswith("data:"):
        prefix, sep, value = value.partition(",")
        if not sep or prefix not in {
            "data:image/jpeg;base64",
            "data:image/png;base64",
            "data:image/webp;base64",
        }:
            raise InputError("仅支持 JPG、PNG、WEBP 图片", 415)
    try:
        raw = base64.b64decode(value, validate=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InputError("仅支持 JPG、PNG、WEBP 图片", 415)
                if image.width * image.height > max_pixels:
                    raise InputError("图片像素超过 1600 万上限", 413)
                if getattr(image, "n_frames", 1) != 1:
                    raise InputError("请上传静态图片", 415)
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except InputError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise InputError("图片像素过大", 413) from None
    except (ValueError, TypeError, binascii.Error, UnidentifiedImageError, OSError):
        raise InputError("图片编码无效或文件损坏") from None


def persistent_secret(directory):
    path = Path(directory) / "session.secret"
    with (Path(directory) / "session.lock").open("a+") as lock:
        file_lock.flock(lock.fileno(), file_lock.LOCK_EX)
        if path.exists():
            value = path.read_text(encoding="ascii").strip()
            if len(value) < 32:
                raise ValueError("Invalid persistent session secret")
            return value
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            value = secrets.token_hex(32)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        return value


def create_app(
    predictor=None,
    health_provider=None,
    info_provider=None,
    metrics_provider=None,
    object_detector=None,
    config=None,
):
    app = Flask(__name__, static_folder="static")
    app.config.from_mapping(
        STATE_DIR=os.getenv(
            "ECOSORT_STATE_DIR",
            str(Path(__file__).resolve().parents[1] / "runtime" / "app"),
        ),
        MAX_CONTENT_LENGTH=int(
            os.getenv("ECOSORT_MAX_REQUEST_BYTES", str(12 * 1024 * 1024))
        ),
        MAX_BATCH_SIZE=int(os.getenv("ECOSORT_MAX_BATCH_SIZE", "16")),
        ADMIN_TOKEN=os.getenv("ECOSORT_ADMIN_TOKEN", ""),
        SESSION_COOKIE_NAME="ecosort_session",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("ECOSORT_COOKIE_SECURE", "0") == "1",
    )
    if config:
        app.config.update(config)
    store = Store(app.config["STATE_DIR"])
    app.extensions["store"] = store
    app.secret_key = app.config.get("SECRET_KEY") or persistent_secret(store.directory)
    token_file = os.getenv("ECOSORT_ADMIN_TOKEN_FILE", "")
    if token_file and not app.config["ADMIN_TOKEN"]:
        app.config["ADMIN_TOKEN"] = Path(token_file).read_text(encoding="utf-8").strip()
    if app.config["ADMIN_TOKEN"] and len(app.config["ADMIN_TOKEN"]) < 24:
        raise ValueError("ECOSORT_ADMIN_TOKEN must contain at least 24 characters")

    def owner():
        if "owner" not in session:
            session["owner"] = uuid.uuid4().hex
        return session["owner"]

    def csrf():
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(32)
        return session["csrf"]

    def admin():
        if not session.get("admin"):
            raise InputError("需要管理员登录", 403)

    def scope():
        if request.args.get("scope", "mine") == "all":
            admin()
            return None
        if request.args.get("scope", "mine") != "mine":
            raise InputError("scope 必须为 mine 或 all")
        return owner()

    def payload():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise InputError("请求内容必须为 JSON 对象")
        return data

    def text(data, key, max_length, required=False):
        value = data.get(key, "")
        if (
            not isinstance(value, str)
            or len(value) > max_length
            or (required and not value.strip())
        ):
            raise InputError(f"{key} 内容无效或长度超过 {max_length}")
        return value.strip()

    def pagination():
        try:
            page, limit = (
                int(request.args.get("page", "1")),
                int(request.args.get("limit", "12")),
            )
        except ValueError:
            raise InputError("分页参数必须是整数") from None
        if page < 1 or page > 1_000_000 or limit < 1 or limit > 100:
            raise InputError("page 必须为正整数，limit 范围为 1–100")
        return page, limit

    def record(record_id):
        if len(record_id) != 32 or any(c not in "0123456789abcdef" for c in record_id):
            raise InputError("记录不存在", 404)
        found = store.get_record(record_id, None if session.get("admin") else owner())
        if not found:
            raise InputError("记录不存在", 404)
        return found

    def run_prediction(image_data, filename="image"):
        # Validate before model readiness so bad input receives a stable 4xx.
        image = decode_image(image_data)
        if predictor is None or (
            health_provider and not health_provider().get("model_loaded")
        ):
            raise InputError("识别模型尚未就绪，请稍后重试", 503)
        result = dict(predictor(image))
        label = result.get("class_name")
        if label not in AUDITED_CLASS_NAMES:
            raise RuntimeError("classifier returned an invalid label")
        confidence = float(result.get("confidence", 0))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise RuntimeError("classifier returned an invalid confidence")
        result["coarse_category"] = coarse_category(label)
        result["confidence"] = confidence
        result["record_id"] = store.add_record(owner(), image, result, filename)
        return result

    @app.before_request
    def protect_mutations():
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            expected = session.get("csrf", "")
            supplied = request.headers.get("X-CSRF-Token", "")
            if not expected or not secrets.compare_digest(expected, supplied):
                raise InputError("会话校验失败，请刷新页面后重试", 403)
            origin = request.headers.get("Origin")
            if origin and origin != request.host_url.rstrip("/"):
                raise InputError("不接受跨站写入请求", 403)

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        if request.path.startswith("/results/"):
            # The audited, read-only portfolio contains an embedded stylesheet.
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                "script-src 'none'; frame-ancestors 'none'; base-uri 'self'"
            )
        if request.path.startswith(("/api/", "/predict", "/metrics", "/health")):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(InputError)
    def input_error(exc):
        return jsonify(error=str(exc)), exc.status

    @app.errorhandler(Conflict)
    def conflict_error(exc):
        return jsonify(error=str(exc)), 409

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(
            error="请求过大" if exc.code == 413 else exc.description
        ), exc.code

    @app.errorhandler(Exception)
    def internal_error(exc):
        app.logger.exception("Request failed")
        return jsonify(error="服务处理失败，请稍后重试"), 500

    @app.get("/")
    def index():
        return app.send_static_file("console.html")

    @app.get("/results/")
    @app.get("/results/<path:filename>")
    def experiment_results(filename="index.html"):
        return send_from_directory(
            Path(__file__).resolve().parents[1] / "docs" / "portfolio", filename
        )

    @app.get("/api/session")
    def session_info():
        owner()
        return jsonify(
            csrf_token=csrf(),
            is_admin=bool(session.get("admin")),
            admin_enabled=bool(app.config["ADMIN_TOKEN"]),
            class_names=AUDITED_CLASS_NAMES,
            class_to_coarse=CLASS_TO_COARSE,
        )

    @app.post("/api/admin/login")
    def login():
        token = text(payload(), "token", 256, required=True)
        expected = app.config["ADMIN_TOKEN"]
        if not expected:
            raise InputError("管理员审核尚未配置", 503)
        if not secrets.compare_digest(token, expected):
            raise InputError("管理员口令不正确", 401)
        session["admin"] = True
        return jsonify(is_admin=True)

    @app.post("/api/admin/logout")
    def logout():
        session.pop("admin", None)
        return jsonify(is_admin=False)

    @app.get("/health")
    def health():
        status = (
            health_provider()
            if health_provider
            else {"status": "degraded", "model_loaded": False}
        )
        status = {
            **status,
            "release": os.getenv("ECOSORT_RELEASE", "development"),
            "storage_ready": True,
            "class_count": len(AUDITED_CLASS_NAMES),
        }
        return jsonify(status)

    @app.get("/model_info")
    def model_info():
        data = info_provider() if info_provider else {}
        return jsonify(
            {
                **data,
                "class_names": AUDITED_CLASS_NAMES,
                "class_to_coarse": CLASS_TO_COARSE,
                "coarse_labels": ["recyclable", "hazardous", "kitchen", "other"],
            }
        )

    @app.get("/metrics")
    def metrics():
        admin()
        return jsonify(metrics_provider() if metrics_provider else {})

    @app.post("/predict")
    def predict():
        data = payload()
        if data.get("format", "base64") != "base64":
            raise InputError("仅支持 base64 编码")
        return jsonify(
            run_prediction(data.get("image"), text(data, "filename", 160) or "image")
        )

    @app.post("/batch_predict")
    def batch_predict():
        images = payload().get("images")
        if not isinstance(images, list) or not images:
            raise InputError("images 必须是非空数组")
        if len(images) > app.config["MAX_BATCH_SIZE"]:
            raise InputError(f"每批最多 {app.config['MAX_BATCH_SIZE']} 张图片", 413)
        predictions = []
        for image_data in images:
            try:
                predictions.append(run_prediction(image_data))
            except InputError as exc:
                predictions.append({"error": str(exc), "status": exc.status})
        return jsonify(predictions=predictions)

    @app.post("/detect_objects")
    def detect_objects():
        if object_detector is None:
            raise InputError("候选区域分析不可用", 503)
        data = payload()
        image = decode_image(data.get("image"))
        try:
            count = int(data.get("max_objects", 5))
        except (ValueError, TypeError):
            raise InputError("max_objects 必须是 1–8 的整数") from None
        if not 1 <= count <= 8:
            raise InputError("max_objects 必须是 1–8 的整数")
        return jsonify(object_detector(image, count))

    @app.post("/reload_model")
    def reload_model():
        raise InputError("运行时不允许更换模型，请通过受控部署更新", 403)

    @app.get("/api/records")
    def records():
        page, limit = pagination()
        category = request.args.get("category") or None
        if category and category not in {"recyclable", "hazardous", "kitchen", "other"}:
            raise InputError("分类筛选参数无效")
        return jsonify(store.list_records(scope(), page, limit, category))

    @app.get("/api/records/<record_id>")
    def record_detail(record_id):
        return jsonify(record(record_id))

    @app.get("/api/records/<record_id>/image")
    def record_image(record_id):
        record(record_id)
        return send_file(
            store.images / (record_id + ".jpg"), mimetype="image/jpeg", max_age=0
        )

    @app.post("/api/records/<record_id>/feedback")
    def feedback(record_id):
        original = record(record_id)
        data = payload()
        label = text(data, "expected_label", 80, required=True)
        if label not in AUDITED_CLASS_NAMES:
            raise InputError("请选择有效的 14 类目标标签")
        if label == original["predicted_label"]:
            raise InputError("纠错类别与原预测一致，请选择你认为正确的其他类别")
        feedback_id = store.submit_feedback(
            record_id, label, text(data, "comment", 500)
        )
        return jsonify(id=feedback_id, status="pending"), 201

    @app.get("/api/reviews")
    def reviews():
        admin()
        status = request.args.get("status", "pending")
        if status not in {"pending", "approved", "rejected"}:
            raise InputError("审核状态无效")
        page, limit = pagination()
        return jsonify(store.list_reviews(status, page, limit))

    @app.post("/api/reviews/<feedback_id>/decision")
    def decide(feedback_id):
        admin()
        data = payload()
        if data.get("status") not in {"approved", "rejected"}:
            raise InputError("审核结论必须为 approved 或 rejected")
        version = data.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise InputError("请提供有效的审核版本号")
        result = store.decide(
            feedback_id, data["status"], text(data, "note", 500), version
        )
        if result is None:
            raise InputError("纠错不存在", 404)
        return jsonify(result)

    @app.get("/api/stats")
    def stats():
        return jsonify(store.stats(scope()))

    @app.get("/api/records/export.csv")
    def export_records():
        return app.response_class(
            store.export(scope()),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=ecosort-records.csv"},
        )

    @app.get("/api/reviews/export.csv")
    def export_reviews():
        admin()
        return app.response_class(
            store.export(None, approved_only=True),
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=ecosort-approved-feedback.csv"
            },
        )

    return app
