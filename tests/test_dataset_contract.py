"""Data leakage guards. Run with the ML environment; no model download or GPU."""

import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("torchvision")
from src.data.dataset import TrashDataset


def make_class(root, name, n=10):
    directory = root / name
    directory.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (20, 20), (i, 20, 30)).save(directory / f"{i}.jpg")


def test_test_split_never_reuses_unsplit_training_images(tmp_path):
    make_class(tmp_path, "plastic")
    with pytest.raises(ValueError, match="Explicit test/"):
        TrashDataset(str(tmp_path), split="test")


def test_missing_presplit_val_never_randomly_splits_other_directories(tmp_path):
    make_class(tmp_path / "train", "plastic")
    with pytest.raises(ValueError, match="Explicit val/"):
        TrashDataset(str(tmp_path), split="val")


def test_train_and_validation_are_disjoint_repeatable_and_not_double_counted(tmp_path):
    make_class(tmp_path, "plastic", 20)
    train = TrashDataset(str(tmp_path), split="train", seed=9)
    val = TrashDataset(str(tmp_path), split="val", seed=9)
    repeat = TrashDataset(str(tmp_path), split="train", seed=9)
    assert not set(train.samples) & set(val.samples)
    assert len(train) + len(val) == 20
    assert train.samples == repeat.samples


def test_presplit_test_preserves_checkpoint_class_order(tmp_path):
    make_class(tmp_path / "test", "plastic", 1)
    make_class(tmp_path / "test", "glass", 1)
    dataset = TrashDataset(
        str(tmp_path), split="test", class_names=["plastic", "glass"]
    )
    assert dataset.class_names == ["plastic", "glass"]
    assert dataset.targets == [0, 1]


def test_unreadable_sample_is_not_replaced_with_a_white_image(tmp_path):
    directory = tmp_path / "test" / "plastic"
    directory.mkdir(parents=True)
    (directory / "bad.jpg").write_bytes(b"corrupt")
    dataset = TrashDataset(str(tmp_path), split="test")
    with pytest.raises(RuntimeError, match="Unreadable training image"):
        dataset[0]


def test_explicit_class_names_must_exist(tmp_path):
    make_class(tmp_path / "test", "plastic", 1)
    with pytest.raises(ValueError, match="Missing class directory"):
        TrashDataset(str(tmp_path), split="test", class_names=["plastic", "glass"])


def test_unexpected_test_classes_cannot_be_silently_excluded(tmp_path):
    make_class(tmp_path / "test", "plastic", 1)
    make_class(tmp_path / "test", "unknown", 1)
    with pytest.raises(ValueError, match="Unexpected class directories"):
        TrashDataset(str(tmp_path), split="test", class_names=["plastic"])
