import sqlite3
import json

conn = sqlite3.connect('file_guessr.db')
conn.row_factory = sqlite3.Row

with open('db_out.txt', 'w', encoding='utf-8') as f:
    f.write("=== IMAGES IN DB ===\n")
    rows = conn.execute("SELECT file_name, summary, keywords FROM files WHERE file_name LIKE '%.jpg' OR file_name LIKE '%.png'").fetchall()
    for r in rows:
        f.write(json.dumps(dict(r), ensure_ascii=False) + '\n')

    f.write("\n=== HIPPO IN DB ===\n")
    rows2 = conn.execute("SELECT file_name, summary, keywords FROM files WHERE file_name LIKE '%hippo%' OR summary LIKE '%hippo%' OR keywords LIKE '%hippo%'").fetchall()
    for r in rows:
        f.write(json.dumps(dict(r), ensure_ascii=False) + '\n')
