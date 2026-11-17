import csv
import shutil

import pytest
from PIL import Image

from scripts.build_dataset_from_mapping import build_dataset


def raw_images(root):
    for label in ["bottle", "battery"]:
        (root / label).mkdir(parents=True)
        for i in range(10):
            Image.new("RGB", (20, 20), (i, 40 if label == "bottle" else 80, 20)).save(
                root / label / f"{i}.png"
            )


def test_builder_deduplicates_before_split_and_is_repeatable(tmp_path):
    raw = tmp_path / "raw"
    raw_images(raw)
    shutil.copyfile(raw / "battery/0.png", raw / "battery/duplicate.png")
    mapping = {"bottle": "plastic", "battery": "haz_battery"}
    a, b = tmp_path / "a", tmp_path / "b"
    first = build_dataset(raw, a, mapping, list(mapping.values()), seed=7)
    second = build_dataset(raw, b, mapping, list(mapping.values()), seed=7)
    assert first["unique_images"] == 20 and first["duplicates_removed"] == 1
    assert first["splits"] == {"train": 16, "val": 2, "test": 2}
    rows = list(csv.DictReader((a / "manifest.csv").open(encoding="utf-8")))
    assert len({row["sha256"] for row in rows}) == 20
    assert (a / "manifest.csv").read_text() == (b / "manifest.csv").read_text()


def test_builder_never_deletes_existing_output(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "important"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        build_dataset(tmp_path / "raw", out, {}, [])
    assert marker.read_text() == "keep"


def test_builder_rejects_conflicting_labels_before_creating_output(tmp_path):
    raw = tmp_path / "raw"
    raw_images(raw)
    shutil.copyfile(raw / "battery/0.png", raw / "bottle/conflict.png")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Conflicting labels"):
        build_dataset(
            raw,
            out,
            {"battery": "haz_battery", "bottle": "plastic"},
            ["haz_battery", "plastic"],
        )
    assert not out.exists()


def test_builder_rejects_unreadable_inputs(tmp_path):
    raw = tmp_path / "raw"
    raw_images(raw)
    (raw / "battery/bad.png").write_text("not an image")
    with pytest.raises(OSError):
        build_dataset(
            raw,
            tmp_path / "out",
            {"battery": "haz_battery", "bottle": "plastic"},
            ["haz_battery", "plastic"],
        )
