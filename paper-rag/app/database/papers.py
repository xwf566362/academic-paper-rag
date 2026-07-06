# -*- coding: utf-8 -*-
"""Paper metadata, summary, keywords, BibTeX."""
import json, sqlite3
from pathlib import Path
from app.config import CHROMA_DIR

def _get_conn():
    p = Path(str(CHROMA_DIR)) / "app.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS papers_meta (
            file_name TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            keywords TEXT NOT NULL DEFAULT '[]',
            bibtex TEXT NOT NULL DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (file_name)
        );
    """)
    conn.commit()
    return conn

def get_paper_meta(file_name):
    conn = _get_conn()
    r = conn.execute("SELECT * FROM papers_meta WHERE file_name=?", (file_name,)).fetchone()
    conn.close()
    if r:
        d = dict(r)
        try:
            d["keywords"] = json.loads(d.get("keywords", "[]"))
        except Exception:
            d["keywords"] = []
        return d
    return None

def upsert_paper_meta(file_name, title="", summary="", keywords=None, bibtex=""):
    conn = _get_conn()
    kw = json.dumps(keywords or [], ensure_ascii=False)
    conn.execute("""
        INSERT INTO papers_meta (file_name, title, summary, keywords, bibtex, updated_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(file_name) DO UPDATE SET
            title=COALESCE(NULLIF(?,''), title),
            summary=COALESCE(NULLIF(?,''), summary),
            keywords=COALESCE(NULLIF(?,'[]'), keywords),
            bibtex=COALESCE(NULLIF(?,''), bibtex),
            updated_at=datetime('now')
    """, (file_name, title, summary, kw, bibtex, title, summary, kw, bibtex))
    conn.commit()
    conn.close()

def list_papers_meta(file_names=None):
    conn = _get_conn()
    if file_names:
        ph = ",".join("?" for _ in file_names)
        rows = conn.execute(f"SELECT * FROM papers_meta WHERE file_name IN ({ph})", file_names).fetchall()
    else:
        rows = conn.execute("SELECT * FROM papers_meta ORDER BY updated_at DESC").fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["keywords"] = json.loads(d.get("keywords", "[]"))
        except Exception:
            d["keywords"] = []
        result.append(d)
    return result

def delete_paper_meta(file_name):
    conn = _get_conn()
    conn.execute("DELETE FROM papers_meta WHERE file_name=?", (file_name,))
    conn.commit()
    conn.close()
