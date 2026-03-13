"""
Find and delete DB records where keywords or summary are abnormally long.
These are caused by LLM infinite-loop generation. Deleting lets the indexer re-index properly.
Usage: python fix_bad_records.py [--dry-run]
"""
import sqlite3
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file_guessr.db")
DRY_RUN = "--dry-run" in sys.argv

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT file_path, file_name, length(keywords) as klen, length(summary) as slen "
    "FROM files WHERE length(keywords) > 500 OR length(summary) > 1200 ORDER BY klen DESC"
).fetchall()

print(f"Found {len(rows)} bad record(s):\n")
for r in rows:
    path = r["file_path"]
    name = r["file_name"]
    klen = r["klen"]
    slen = r["slen"]
    print(f"  [{klen} kw chars / {slen} sm chars] {name}")
    print(f"    {path}\n")

if not rows:
    print("No bad records found. DB looks clean!")
    conn.close()
    sys.exit(0)

if DRY_RUN:
    print("[DRY RUN] No records deleted. Remove --dry-run to actually delete.")
    conn.close()
    sys.exit(0)

print("Deleting bad records so indexer can re-index them cleanly...")
for r in rows:
    conn.execute("DELETE FROM files WHERE file_path = ?", (r["file_path"],))
    print(f"  Deleted: {r['file_name']}")

conn.commit()
conn.close()
print(f"\nDone! Deleted {len(rows)} records.")
print("Restart file_guessr to re-index them.")
