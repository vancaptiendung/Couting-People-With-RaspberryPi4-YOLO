import sqlite3
from const import *
import os
import sys
import logging
import time
import traceback
import json

def log_event(event_type, object_id=None, details=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO system_logs (timestamp, event_type, object_id, went_in, went_out, inRoom, details) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (timestamp, event_type, object_id, TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM, details)
            )
            conn.commit()
    except Exception as e:
        print(f"[DB ERROR] Lỗi ghi DB: {e}")

def save_data():
    with open(DATA_FILE, "w") as f:
        json.dump({"in": TOTAL_IN, "out": TOTAL_OUT, "room": PEOPLE_IN_ROOM}, f)

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                return data.get("in", 0), data.get("out", 0), data.get("room", 0)
        except: pass
    return 0, 0, 0

# database 
def init_db():
    """Khởi tạo cơ sở dữ liệu SQLite siêu nhẹ lưu log nội bộ"""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                object_id INTEGER,
                went_in INTERGER,
                went_out INTEGER,
                inRoom INTEGER,
                details TEXT DEFAULT "Null",
                synced INTEGER DEFAULT 0
            )
        ''')
        conn.commit()

def fetch_log_rows(before_id=None, limit=30):
    limit = max(1, min(int(limit or 30), 100))
    query = "SELECT id, timestamp, event_type, object_id, details, synced FROM system_logs"
    params = []
    if before_id is not None:
        query += " WHERE id < ?"
        params.append(int(before_id))
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit + 1)

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

    has_more = len(rows) > limit
    rows = rows[:limit]
    items = [format_log_entry(row) for row in rows]
    next_before_id = rows[-1][0] if rows else None
    return items, has_more, next_before_id