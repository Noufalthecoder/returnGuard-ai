"""
audit_db.py
-----------
Every flagging decision gets written to SQLite with a human-readable
reason, so any flagged customer/ring can be traced back to *why* it was
flagged. This is the "explainable, bounded, gated" requirement Razorpay
calls out — a model score alone isn't an audit trail.
"""

import sqlite3
import os
import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "audit_trail.db")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ring_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            community_id INTEGER,
            ring_score REAL,
            size INTEGER,
            avg_return_rate REAL,
            density REAL,
            flag_reason TEXT,
            threshold_used REAL,
            flagged_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customer_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            community_id INTEGER,
            flagged_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_flags(flagged_df, threshold_used: float):
    """Write current run's flagged rings + members into the audit DB."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for _, row in flagged_df.iterrows():
        cur.execute("""
            INSERT INTO ring_flags
            (community_id, ring_score, size, avg_return_rate, density, flag_reason, threshold_used, flagged_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(row["community_id"]), float(row["ring_score"]), int(row["size"]),
            float(row["avg_return_rate"]), float(row["density"]), row["flag_reason"],
            threshold_used, now,
        ))
        for member in row["members"]:
            cur.execute("""
                INSERT INTO customer_flags (customer_id, community_id, flagged_at)
                VALUES (?, ?, ?)
            """, (member, int(row["community_id"]), now))

    conn.commit()
    conn.close()


def get_audit_history(limit=50):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM ring_flags ORDER BY flagged_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    conn.close()
    return cols, rows


def get_customer_history(customer_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT cf.customer_id, cf.community_id, rf.flag_reason, cf.flagged_at
        FROM customer_flags cf
        JOIN ring_flags rf ON cf.community_id = rf.community_id
        WHERE cf.customer_id = ?
        ORDER BY cf.flagged_at DESC
    """, (customer_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    from graph_detection import run_pipeline
    result = run_pipeline()
    log_flags(result["flagged"], threshold_used=0.55)
    cols, rows = get_audit_history()
    print(f"Audit trail written. {len(rows)} ring-flag records in DB.")
    for r in rows:
        print(dict(zip(cols, r)))
