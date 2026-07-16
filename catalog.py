"""
Persistent song catalog backed by SQLite.
Stores metadata for all completed karaoke jobs so they survive restarts.
DB file lives in the work directory (already a Docker volume).
"""
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path("work") / "catalog.db"

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the songs table if it doesn't exist."""
    with _lock:
        conn = _connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                job_id       TEXT PRIMARY KEY,
                title        TEXT NOT NULL DEFAULT 'track',
                artist       TEXT DEFAULT '',
                source_url   TEXT DEFAULT '',
                created_at   TEXT DEFAULT (datetime('now')),
                status       TEXT DEFAULT 'pending',
                video_path   TEXT DEFAULT '',
                minus_path   TEXT DEFAULT '',
                ass_path     TEXT DEFAULT '',
                thumbnail_path TEXT DEFAULT '',
                youtube_ready INTEGER DEFAULT 0,
                lyrics       TEXT DEFAULT '',
                cdg_path     TEXT DEFAULT ''
            )
        """)
        # Migrate older databases that predate a column.
        existing = {row[1] for row in conn.execute("PRAGMA table_info(songs)")}
        for col in ("cdg_path",):
            if col not in existing:
                conn.execute(f"ALTER TABLE songs ADD COLUMN {col} TEXT DEFAULT ''")
        conn.commit()
        conn.close()


def _parse_artist_title(raw_title: str) -> tuple[str, str]:
    """Try to split 'Artist - Title' into (artist, title)."""
    for sep in (" - ", " – ", " — "):
        if sep in raw_title:
            parts = raw_title.split(sep, 1)
            return parts[0].strip(), parts[1].strip()
    return "", raw_title.strip()


def upsert_song(job_id: str, **kwargs) -> None:
    """Insert or update a song entry."""
    with _lock:
        conn = _connect()
        existing = conn.execute("SELECT job_id FROM songs WHERE job_id=?", (job_id,)).fetchone()
        if existing:
            if kwargs:
                sets = ", ".join(f"{k}=?" for k in kwargs)
                conn.execute(f"UPDATE songs SET {sets} WHERE job_id=?",
                             [*kwargs.values(), job_id])
        else:
            cols = ["job_id"] + list(kwargs.keys())
            placeholders = ", ".join(["?"] * len(cols))
            conn.execute(f"INSERT INTO songs ({', '.join(cols)}) VALUES ({placeholders})",
                         [job_id, *kwargs.values()])
        conn.commit()
        conn.close()


def get_song(job_id: str) -> Optional[dict]:
    """Get a single song by job_id."""
    with _lock:
        conn = _connect()
        row = conn.execute("SELECT * FROM songs WHERE job_id=?", (job_id,)).fetchone()
        conn.close()
        return dict(row) if row else None


def list_songs(search: str = "", limit: int = 50, offset: int = 0) -> list[dict]:
    """List completed songs, newest first. Optional search by title/artist."""
    with _lock:
        conn = _connect()
        if search:
            query = """SELECT * FROM songs WHERE status='done'
                       AND (title LIKE ? OR artist LIKE ?)
                       ORDER BY created_at DESC LIMIT ? OFFSET ?"""
            pattern = f"%{search}%"
            rows = conn.execute(query, (pattern, pattern, limit, offset)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM songs WHERE status='done' ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]


def count_songs(search: str = "") -> int:
    """Count completed songs."""
    with _lock:
        conn = _connect()
        if search:
            pattern = f"%{search}%"
            row = conn.execute(
                "SELECT COUNT(*) FROM songs WHERE status='done' AND (title LIKE ? OR artist LIKE ?)",
                (pattern, pattern),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM songs WHERE status='done'").fetchone()
        conn.close()
        return row[0] if row else 0


def delete_song(job_id: str) -> bool:
    """Delete a song entry from the catalog."""
    with _lock:
        conn = _connect()
        cur = conn.execute("DELETE FROM songs WHERE job_id=?", (job_id,))
        conn.commit()
        conn.close()
        return cur.rowcount > 0
