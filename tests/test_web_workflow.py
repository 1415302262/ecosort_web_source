"""HTTP contracts, ownership and the persistent prediction → review workflow."""

import base64
import csv
import io
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from backend.store import Conflict
from backend.web import create_app, decode_image

ADMIN_TOKEN = "test-admin-token-with-at-least-32-characters"


def image_data(format="PNG", size=(32, 24)):
    output = io.BytesIO()
    Image.new("RGB", size, "green").save(output, format)
    return base64.b64encode(output.getvalue()).decode("ascii")


def fake_predictor(image):
    assert image.mode == "RGB"
    return {
        "class_name": "plastic",
        "coarse_category": "recyclable",
        "confidence": 0.76,
        "uncertain": True,
        "cascade_latency_ms": 12.5,
        "routing": {
            "routed": True,
            "provider_attempted": False,
            "provider_success": False,
            "failure_reason": "execution_disabled",
        },
    }


@pytest.fixture
def app(tmp_path):
    return create_app(
        predictor=fake_predictor,
        health_provider=lambda: {"model_loaded": True, "status": "healthy"},
        config={"TESTING": True, "STATE_DIR": tmp_path, "ADMIN_TOKEN": ADMIN_TOKEN},
    )


def post(client, path, data, **headers):
    token = client.get("/api/session").json["csrf_token"]
    return client.post(path, json=data, headers={"X-CSRF-Token": token, **headers})


def predict(client, filename="sample.png"):
    response = post(client, "/predict", {"image": image_data(), "filename": filename})
    assert response.status_code == 200, response.json
    return response.json["record_id"]


def login(client):
    response = post(client, "/api/admin/login", {"token": ADMIN_TOKEN})
    assert response.status_code == 200, response.json


def feedback(client, record_id, label="haz_battery"):
    return post(
        client,
        f"/api/records/{record_id}/feedback",
        {"expected_label": label, "comment": "电池不是塑料"},
    )


def test_prediction_feedback_review_and_export_preserve_original(app):
    user = app.test_client()
    record_id = predict(user)
    response = feedback(user, record_id)
    assert response.status_code == 201
    feedback_id = response.json["id"]
    reviewer = app.test_client()
    login(reviewer)
    queue = reviewer.get("/api/reviews").json
    assert queue["total"] == 1
    assert queue["items"][0]["expected_label"] == "haz_battery"
    result = post(
        reviewer,
        f"/api/reviews/{feedback_id}/decision",
        {"status": "approved", "note": "已核对原图", "version": 1},
    )
    assert result.status_code == 200
    record = user.get(f"/api/records/{record_id}").json
    assert record["predicted_label"] == record["prediction"]["class_name"] == "plastic"
    assert record["feedback_status"] == "approved"
    assert record["expected_label"] == "haz_battery"
    assert record["review_note"] == "已核对原图"
    stats = reviewer.get("/api/stats?scope=all").json
    assert stats["total"] == 1 and stats["feedback"] == {"approved": 1}
    exported = reviewer.get("/api/reviews/export.csv")
    assert exported.status_code == 200
    rows = list(csv.DictReader(io.StringIO(exported.data.decode("utf-8-sig"))))
    assert rows[0]["predicted_label"] == "plastic"
    assert rows[0]["expected_label"] == "haz_battery"
    with app.extensions["store"].connect() as db:
        assert [
            r[0] for r in db.execute("SELECT action FROM audit_events ORDER BY id")
        ] == ["prediction_saved", "feedback_submitted", "feedback_approved"]


def test_guest_cannot_read_other_guests_record_or_image(app):
    first, second = app.test_client(), app.test_client()
    record_id = predict(first)
    assert second.get("/api/records").json["total"] == 0
    assert second.get(f"/api/records/{record_id}").status_code == 404
    assert second.get(f"/api/records/{record_id}/image").status_code == 404
    assert feedback(second, record_id).status_code == 404
    assert "sample.png" not in second.get("/api/records/export.csv").text


@pytest.mark.parametrize(
    "path",
    [
        "/api/reviews",
        "/api/reviews/export.csv",
        "/api/stats?scope=all",
        "/api/records?scope=all",
        "/metrics",
    ],
)
def test_admin_resources_reject_guest(app, path):
    assert app.test_client().get(path).status_code == 403


def test_admin_login_logout_and_default_scope(app):
    predict(app.test_client())
    admin = app.test_client()
    assert post(admin, "/api/admin/login", {"token": "wrong"}).status_code == 401
    login(admin)
    assert admin.get("/api/session").json["is_admin"]
    assert admin.get("/api/records").json["total"] == 0
    assert admin.get("/api/records?scope=all").json["total"] == 1
    assert post(admin, "/api/admin/logout", {}).status_code == 200
    assert admin.get("/api/reviews").status_code == 403


def test_csrf_and_cross_origin_writes_are_rejected(app):
    client = app.test_client()
    assert client.post("/predict", json={"image": image_data()}).status_code == 403
    assert (
        post(
            client,
            "/predict",
            {"image": image_data()},
            Origin="https://attacker.invalid",
        ).status_code
        == 403
    )
    assert client.get("/api/records").json["total"] == 0


@pytest.mark.parametrize(
    "data,status",
    [
        ({}, 400),
        ({"image": None}, 400),
        ({"image": 7}, 400),
        ({"image": "%not-base64"}, 400),
        ({"image": base64.b64encode(b"not an image").decode()}, 400),
        ({"image": image_data("GIF")}, 415),
        ({"image": image_data(), "format": "file"}, 400),
        ({"image": image_data(), "filename": 42}, 400),
    ],
)
def test_invalid_images_do_not_create_records(app, data, status):
    client = app.test_client()
    assert post(client, "/predict", data).status_code == status
    assert client.get("/api/stats").json["total"] == 0
    assert not list(app.extensions["store"].images.iterdir())


def test_pixel_limit_and_exif_sanitization(app):
    with pytest.raises(Exception, match="像素"):
        decode_image(image_data(size=(32, 24)), max_pixels=20)
    client = app.test_client()
    record_id = predict(client)
    image = client.get(f"/api/records/{record_id}/image")
    assert image.status_code == 200 and image.mimetype == "image/jpeg"
    with Image.open(io.BytesIO(image.data)) as saved:
        assert saved.size == (32, 24) and not saved.getexif()


def test_missing_model_returns_503_without_fake_prediction(tmp_path):
    app = create_app(config={"TESTING": True, "STATE_DIR": tmp_path})
    client = app.test_client()
    assert client.get("/health").json["model_loaded"] is False
    assert post(client, "/predict", {"image": image_data()}).status_code == 503
    assert client.get("/api/records").json["total"] == 0


def test_batch_persists_successes_and_reports_invalid_items(app):
    client = app.test_client()
    result = post(
        client, "/batch_predict", {"images": [image_data(), "invalid", image_data()]}
    ).json
    assert len(result["predictions"]) == 3
    assert result["predictions"][1]["status"] == 400
    assert client.get("/api/records").json["total"] == 2
    assert post(client, "/batch_predict", {"images": []}).status_code == 400
    assert (
        post(client, "/batch_predict", {"images": [image_data()] * 17}).status_code
        == 413
    )


def test_feedback_validation_duplicate_and_decision_conflict(app):
    client = app.test_client()
    record_id = predict(client)
    assert feedback(client, record_id, "unknown").status_code == 400
    submitted = feedback(client, record_id)
    assert feedback(client, record_id).status_code == 409
    assert (
        post(
            client,
            f"/api/reviews/{submitted.json['id']}/decision",
            {"status": "approved", "version": 1},
        ).status_code
        == 403
    )
    login(client)
    path = f"/api/reviews/{submitted.json['id']}/decision"
    assert post(client, path, {"status": "pending", "version": 1}).status_code == 400
    assert (
        post(client, path, {"status": "rejected", "version": True}).status_code == 400
    )
    assert post(client, path, {"status": "rejected", "version": 1}).status_code == 200
    assert post(client, path, {"status": "approved", "version": 1}).status_code == 409


def test_two_reviewers_cannot_overwrite_each_other(app):
    client = app.test_client()
    record_id = predict(client)
    feedback_id = feedback(client, record_id).json["id"]
    store = app.extensions["store"]

    def decide(status):
        try:
            store.decide(feedback_id, status, "parallel review", 1)
            return "saved"
        except Conflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(decide, ["approved", "rejected"]))
    assert sorted(results) == ["conflict", "saved"]


def test_records_and_session_survive_application_restart(app, tmp_path):
    client = app.test_client()
    record_id = predict(client)
    cookie = client.get_cookie("session")
    restarted = create_app(config={"TESTING": True, "STATE_DIR": tmp_path})
    new_client = restarted.test_client()
    new_client.set_cookie("session", cookie.value)
    assert new_client.get(f"/api/records/{record_id}").status_code == 200
    assert new_client.get(f"/api/records/{record_id}/image").status_code == 200


def test_export_escapes_spreadsheet_formulas(app):
    client = app.test_client()
    predict(client, filename='=HYPERLINK("https://evil.invalid")')
    exported = client.get("/api/records/export.csv").data.decode("utf-8-sig")
    row = next(csv.DictReader(io.StringIO(exported)))
    assert row["filename"].startswith("'=")


@pytest.mark.parametrize(
    "query",
    ["page=0", "page=x", "limit=101", "limit=-1", "category=evil", "scope=evil"],
)
def test_record_filters_reject_invalid_values(app, query):
    assert app.test_client().get("/api/records?" + query).status_code == 400


def test_pagination_and_filtering(app):
    client = app.test_client()
    predict(client)
    predict(client)
    page = client.get("/api/records?limit=1&page=2").json
    assert page["total"] == 2 and len(page["items"]) == 1
    assert client.get("/api/records?category=hazardous").json["total"] == 0


def test_model_exception_is_sanitized_and_not_saved(tmp_path):
    def fail(image):
        raise RuntimeError("private /server/path and api-key")

    app = create_app(predictor=fail, config={"TESTING": True, "STATE_DIR": tmp_path})
    client = app.test_client()
    response = post(client, "/predict", {"image": image_data()})
    assert response.status_code == 500
    assert "private" not in response.text
    assert client.get("/api/records").json["total"] == 0


def test_response_security_headers_and_uploaded_html_not_executed(app):
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    predict(client, "<script>alert(1)</script>")
    assert "no-store" in client.get("/api/records").headers["Cache-Control"]
    assert client.get("/reload_model").status_code == 405
    assert post(client, "/reload_model", {"model_path": "/tmp/evil"}).status_code == 403
