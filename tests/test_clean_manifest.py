from scripts.build_clean_manifest import inventory


def test_content_conflicts_are_quarantined_and_exact_copies_do_not_cross_splits(tmp_path):
    for split in ("train", "val", "test"):
        for label in ("a", "b"):
            d = tmp_path / split / label
            d.mkdir(parents=True)
            (d / "base.png").write_bytes((split + label).encode())
    (tmp_path / "test/a/repeat.png").write_bytes(b"traina")
    for loc in ("train/a", "val/b"):
        (tmp_path / loc / "ambiguous.jpg").write_bytes(b"same-content")
    rows = inventory(tmp_path)
    assert [r["sample_id"] for r in rows if r["decision"] == "duplicate"] == ["test/a/repeat.png"]
    assert {r["sample_id"] for r in rows if r["decision"] == "label_conflict"} == {
        "train/a/ambiguous.jpg", "val/b/ambiguous.jpg"
    }
    assert len({r["sha256"] for r in rows if r["decision"] == "keep"}) == 6
