# -*- coding: utf-8 -*-
"""Conversation persistence - local mode, no auth."""
import json, sqlite3, uuid
from pathlib import Path
from app.config import CHROMA_DIR

def _get_conn():
    p = Path(str(CHROMA_DIR)) / "app.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT 'New Chat',
            messages TEXT NOT NULL DEFAULT '[]',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    return conn

def list_conversations():
    conn = _get_conn()
    rows = conn.execute("SELECT id,title,created_at,updated_at FROM conversations ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def create_conversation(title="New Chat"):
    conn = _get_conn()
    cid = str(uuid.uuid4())
    conn.execute("INSERT INTO conversations (id,title,messages) VALUES (?,?,?)", (cid, title, "[]"))
    conn.commit()
    r = conn.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
    conn.close()
    return dict(r)

def get_conversation(conv_id):
    conn = _get_conn()
    r = conn.execute("SELECT * FROM conversations WHERE id=?", (conv_id,)).fetchone()
    conn.close()
    return dict(r) if r else None

def append_message(conv_id, role, content):
    conn = _get_conn()
    r = conn.execute("SELECT messages FROM conversations WHERE id=?", (conv_id,)).fetchone()
    if not r:
        conn.close(); return None
    msgs = json.loads(r["messages"])
    msgs.append({"role": role, "content": content})
    conn.execute("UPDATE conversations SET messages=?, updated_at=datetime('now') WHERE id=?",
                 (json.dumps(msgs, ensure_ascii=False), conv_id))
    conn.commit()
    result = conn.execute("SELECT * FROM conversations WHERE id=?", (conv_id,)).fetchone()
    conn.close()
    return dict(result)

def delete_conversation(conv_id):
    conn = _get_conn()
    conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
    deleted = conn.total_changes > 0
    conn.commit(); conn.close()
    return deleted
