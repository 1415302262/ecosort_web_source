"""Create a consistent application database backup without rewriting the source."""

import argparse
import shutil
import sqlite3
from pathlib import Path


def backup(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not (source / "ecosort.sqlite3").is_file():
        raise ValueError("Source database does not exist")
    if destination.exists() or source in destination.parents:
        raise ValueError(
            "Backup destination must be new and outside the source directory"
        )
    destination.mkdir(mode=0o700, parents=True)
    original = sqlite3.connect(
        f"file:{(source / 'ecosort.sqlite3').as_posix()}?mode=ro", uri=True
    )
    copied = sqlite3.connect(destination / "ecosort.sqlite3")
    try:
        original.backup(copied)
    finally:
        copied.close()
        original.close()
    # Application images are immutable and created before database insertion.
    shutil.copytree(source / "images", destination / "images")
    if (source / "session.secret").exists():
        shutil.copyfile(source / "session.secret", destination / "session.secret")
        (destination / "session.secret").chmod(0o600)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(backup(args.state_dir, args.output))
