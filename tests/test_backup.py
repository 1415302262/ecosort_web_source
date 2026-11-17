import pytest

from backend.store import Store
from PIL import Image
from scripts.backup_state import backup


def test_backup_restores_database_images_and_session_secret(tmp_path):
    source = tmp_path / "state"
    store = Store(source)
    record_id = store.add_record(
        "visitor",
        Image.new("RGB", (20, 20), "green"),
        {"class_name": "plastic", "coarse_category": "recyclable", "confidence": 0.9},
        "sample",
    )
    (source / "session.secret").write_text("a" * 64)
    target = backup(source, tmp_path / "backup")
    restored = Store(target)
    assert restored.get_record(record_id, "visitor")["predicted_label"] == "plastic"
    assert (restored.images / (record_id + ".jpg")).is_file()
    assert (target / "session.secret").read_text() == "a" * 64


def test_backup_refuses_to_overwrite_existing_destination(tmp_path):
    source = tmp_path / "state"
    Store(source)
    destination = tmp_path / "keep"
    destination.mkdir()
    (destination / "marker").write_text("keep")
    with pytest.raises(ValueError, match="must be new"):
        backup(source, destination)
    assert (destination / "marker").read_text() == "keep"
