"""SQLite application records. Model output and human review remain separate."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class Conflict(Exception):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_csv(rows, columns):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(columns)
    for row in rows:
        values = []
        for col in columns:
            value = str(row.get(col, "") or "")
            if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(
                ("\t", "\r")
            ):
                value = "'" + value
            values.append(value)
        writer.writerow(values)
    return "\ufeff" + output.getvalue()


class Store:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.images = self.directory / "images"
        self.images.mkdir(mode=0o700, exist_ok=True)
        self.path = self.directory / "ecosort.sqlite3"
        with self.connect() as db:
            # DELETE journaling also supports the single-host deployment on NFS.
            # A local disk is preferred; never use WAL across network clients.
            db.executescript("""
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL,
                    created_at TEXT NOT NULL, filename TEXT NOT NULL,
                    predicted_label TEXT NOT NULL, coarse_category TEXT NOT NULL,
                    confidence REAL NOT NULL, uncertain INTEGER NOT NULL,
                    latency_ms REAL NOT NULL, result_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS records_owner_time ON records(owner, created_at);
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY, record_id TEXT NOT NULL UNIQUE
                        REFERENCES records(id), expected_label TEXT NOT NULL,
                    comment TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending','approved','rejected')),
                    created_at TEXT NOT NULL, reviewed_at TEXT,
                    review_note TEXT NOT NULL DEFAULT '', version INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS feedback_status ON feedback(status, created_at);
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
                    action TEXT NOT NULL, record_id TEXT NOT NULL REFERENCES records(id),
                    details_json TEXT NOT NULL
                );
                PRAGMA user_version=1;
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def event(db, action, record_id, details):
        db.execute(
            "INSERT INTO audit_events(created_at,action,record_id,details_json) VALUES(?,?,?,?)",
            (utc_now(), action, record_id, json.dumps(details, ensure_ascii=False)),
        )

    def add_record(self, owner, image, result, filename):
        record_id = uuid.uuid4().hex
        target = self.images / (record_id + ".jpg")
        # Save sanitized pixels, without the upload's EXIF/GPS metadata.
        image.save(target, "JPEG", quality=92)
        try:
            with self.connect() as db:
                db.execute(
                    "INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        record_id,
                        owner,
                        utc_now(),
                        filename,
                        result["class_name"],
                        result["coarse_category"],
                        float(result["confidence"]),
                        int(bool(result.get("uncertain"))),
                        float(result.get("cascade_latency_ms", 0)),
                        json.dumps(result, ensure_ascii=False, allow_nan=False),
                    ),
                )
                self.event(
                    db, "prediction_saved", record_id, {"label": result["class_name"]}
                )
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return record_id

    @staticmethod
    def serialize(row, detail=False):
        item = dict(row)
        item.pop("owner", None)
        raw = item.pop("result_json", None)
        item["uncertain"] = bool(item["uncertain"])
        item["image_url"] = "/api/records/" + item["id"] + "/image"
        if detail and raw:
            item["prediction"] = json.loads(raw)
        return item

    def get_record(self, record_id, owner=None):
        with self.connect() as db:
            row = db.execute(
                """SELECT r.*, f.id feedback_id, f.expected_label,
                f.status feedback_status, f.comment, f.review_note, f.reviewed_at
                FROM records r LEFT JOIN feedback f ON f.record_id=r.id
                WHERE r.id=?"""
                + (" AND r.owner=?" if owner else ""),
                (record_id, owner) if owner else (record_id,),
            ).fetchone()
            return self.serialize(row, detail=True) if row else None

    def list_records(self, owner, page=1, limit=12, category=None):
        where, args = (" WHERE r.owner=?", [owner]) if owner else (" WHERE 1=1", [])
        if category:
            where += " AND r.coarse_category=?"
            args.append(category)
        with self.connect() as db:
            total = db.execute(
                "SELECT COUNT(*) FROM records r" + where, args
            ).fetchone()[0]
            rows = db.execute(
                """SELECT r.*, f.status feedback_status, f.expected_label,
                f.review_note FROM records r LEFT JOIN feedback f ON f.record_id=r.id"""
                + where
                + " ORDER BY r.created_at DESC,r.id DESC LIMIT ? OFFSET ?",
                [*args, limit, (page - 1) * limit],
            ).fetchall()
        return {
            "items": [self.serialize(r) for r in rows],
            "total": total,
            "page": page,
            "limit": limit,
        }

    def submit_feedback(self, record_id, label, comment):
        feedback_id = uuid.uuid4().hex
        try:
            with self.connect() as db:
                db.execute(
                    "INSERT INTO feedback(id,record_id,expected_label,comment,created_at) VALUES(?,?,?,?,?)",
                    (feedback_id, record_id, label, comment, utc_now()),
                )
                self.event(
                    db,
                    "feedback_submitted",
                    record_id,
                    {"feedback_id": feedback_id, "expected_label": label},
                )
        except sqlite3.IntegrityError as exc:
            raise Conflict("该记录已提交纠错，请勿重复提交") from exc
        return feedback_id

    def list_reviews(self, status, page=1, limit=12):
        with self.connect() as db:
            total = db.execute(
                "SELECT COUNT(*) FROM feedback WHERE status=?", (status,)
            ).fetchone()[0]
            rows = db.execute(
                """SELECT f.*, r.filename,r.predicted_label,r.coarse_category,
                r.confidence FROM feedback f JOIN records r ON r.id=f.record_id
                WHERE f.status=? ORDER BY f.created_at,f.id LIMIT ? OFFSET ?""",
                (status, limit, (page - 1) * limit),
            ).fetchall()
        return {
            "items": [dict(r) for r in rows],
            "total": total,
            "page": page,
            "limit": limit,
        }

    def decide(self, feedback_id, status, note, version):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM feedback WHERE id=?", (feedback_id,)
            ).fetchone()
            if row is None:
                return None
            cursor = db.execute(
                """UPDATE feedback SET status=?,review_note=?,reviewed_at=?,version=version+1
                WHERE id=? AND status='pending' AND version=?""",
                (status, note, utc_now(), feedback_id, version),
            )
            if cursor.rowcount != 1:
                raise Conflict("该纠错已经处理，请刷新审核列表")
            self.event(
                db,
                "feedback_" + status,
                row["record_id"],
                {
                    "feedback_id": feedback_id,
                    "expected_label": row["expected_label"],
                    "note": note,
                },
            )
        return {"id": feedback_id, "status": status, "version": version + 1}

    def stats(self, owner):
        where, args = (" WHERE r.owner=?", (owner,)) if owner else ("", ())
        with self.connect() as db:
            row = db.execute(
                """SELECT COUNT(*) total,COALESCE(SUM(uncertain),0) uncertain,
                COALESCE(AVG(latency_ms),0) mean_latency_ms FROM records r"""
                + where,
                args,
            ).fetchone()
            categories = db.execute(
                "SELECT coarse_category,COUNT(*) n FROM records r"
                + where
                + " GROUP BY coarse_category",
                args,
            ).fetchall()
            feedback = db.execute(
                "SELECT f.status,COUNT(*) n FROM feedback f JOIN records r ON r.id=f.record_id"
                + where
                + " GROUP BY f.status",
                args,
            ).fetchall()
            pairs = db.execute(
                """SELECT r.predicted_label, f.expected_label, COUNT(*) n
                FROM feedback f JOIN records r ON r.id=f.record_id
                WHERE f.status='approved' AND r.predicted_label<>f.expected_label"""
                + (" AND r.owner=?" if owner else "")
                + " GROUP BY r.predicted_label,f.expected_label"
                + " ORDER BY n DESC,r.predicted_label,f.expected_label LIMIT 5",
                args,
            ).fetchall()
        return {
            **dict(row),
            "correction_pairs": [dict(pair) for pair in pairs],
            "categories": {r[0]: r[1] for r in categories},
            "feedback": {r[0]: r[1] for r in feedback},
            "note": "业务使用统计，不是模型准确率；纠错经人工审核后仍不自动进入训练集。",
        }

    def export(self, owner, approved_only=False):
        where, args = (" WHERE r.owner=?", [owner]) if owner else (" WHERE 1=1", [])
        if approved_only:
            where += " AND f.status='approved'"
        with self.connect() as db:
            rows = db.execute(
                """SELECT r.id,r.created_at,r.filename,r.predicted_label,
                r.coarse_category,r.confidence,f.expected_label,f.status feedback_status,
                f.review_note FROM records r LEFT JOIN feedback f ON f.record_id=r.id"""
                + where
                + " ORDER BY r.created_at,r.id LIMIT 10001",
                args,
            ).fetchall()
        if len(rows) > 10000:
            raise Conflict("单次最多导出 10000 条，请先归档数据")
        columns = [
            "id",
            "created_at",
            "filename",
            "predicted_label",
            "coarse_category",
            "confidence",
            "expected_label",
            "feedback_status",
            "review_note",
        ]
        return safe_csv([dict(r) for r in rows], columns)
