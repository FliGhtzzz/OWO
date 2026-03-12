"""
Database layer - SQLite for metadata + Elasticsearch for full-text search.
"""
import sqlite3
import os
import time
import re
import socket
from typing import Optional
from elasticsearch import Elasticsearch

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file_guessr.db")
ES_INDEX = "file_guessr"
ES_URL = os.environ.get("ES_URL", "http://127.0.0.1:9200")

# ──────────────── Elasticsearch ────────────────

_es: Optional[Elasticsearch] = None
_es_last_failed_check: float = 0.0
_ES_RETRY_INTERVAL = 15  # seconds before retrying after a failed connection

def _get_es() -> Optional[Elasticsearch]:
    """Get a cached Elasticsearch client (lazy init).
    Tries HTTP first, then HTTPS (for ES 8.x which enables security by default).
    Re-tries connection every _ES_RETRY_INTERVAL seconds after failure so that
    apps launched via desktop shortcut (before ES is fully ready) will
    automatically reconnect once the service comes up.
    """
    global _es, _es_last_failed_check

    # Already connected
    if _es is not None:
        return _es

    # Rate-limit reconnection attempts to avoid hammering ES on every request
    now = time.time()
    if _es_last_failed_check > 0 and (now - _es_last_failed_check) < _ES_RETRY_INTERVAL:
        return None

    # Try configured URL first
    urls_to_try = [ES_URL]
    # Also try HTTPS if configured URL is HTTP
    if ES_URL.startswith("http://"):
        urls_to_try.append(ES_URL.replace("http://", "https://"))

    for url in urls_to_try:
        # Fast socket check before trying heavy Elasticsearch client info()
        try:
            # Parse host and port
            from urllib.parse import urlparse
            parsed = urlparse(url)
            host = parsed.hostname
            port = parsed.port or (443 if url.startswith("https") else 80)

            # Check if port is open with a very short timeout
            with socket.create_connection((host, port), timeout=0.5):
                pass
        except Exception:
            # Port is likely closed, skip this URL
            continue

        try:
            kwargs = {}
            if url.startswith("https://"):
                kwargs["verify_certs"] = False
                kwargs["ssl_show_warn"] = False

            # Check for credentials
            es_password = os.environ.get("ES_PASSWORD", "")
            if es_password:
                kwargs["basic_auth"] = ("elastic", es_password)

            # Use short timeout for the check
            client = Elasticsearch(url, request_timeout=2.0, **kwargs)
            info = client.info()
            _es = client
            _es_last_failed_check = 0.0  # Reset failure timer on success
            version = info.get("version", {}).get("number", "unknown")
            print(f"[ES] Connected to Elasticsearch {version} at {url}")
            # Ensure index exists now that we have a connection
            try:
                _ensure_index()
            except Exception:
                pass
            return _es
        except Exception as e:
            print(f"[ES] Cannot connect to {url}: {e}")

    # All URLs failed — record failure time for retry throttling
    _es_last_failed_check = time.time()
    print("[ES] WARNING: Elasticsearch not available, using SQLite fallback")
    return None


def _ensure_index():
    """Create the ES index with custom mappings if it doesn't exist."""
    es = _get_es()
    if es is None:
        return

    if es.indices.exists(index=ES_INDEX):
        return

    es.indices.create(index=ES_INDEX, body={
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "tokenizer": {
                    "comma_tokenizer": {
                        "type": "pattern",
                        "pattern": ",\\s*"
                    }
                },
                "analyzer": {
                    "file_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"]
                    },
                    "keyword_analyzer": {
                        "type": "custom",
                        "tokenizer": "comma_tokenizer",
                        "filter": ["lowercase", "trim", "asciifolding"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "file_name":     {"type": "text", "analyzer": "file_analyzer"},
                "summary":       {"type": "text", "analyzer": "file_analyzer"},
                "keywords":      {
                    "type": "text",
                    "analyzer": "keyword_analyzer",
                    "fields": {
                        "full": {
                            "type": "text",
                            "analyzer": "file_analyzer"
                        }
                    }
                },
                "raw_text":      {"type": "text", "analyzer": "file_analyzer"},
                "file_path":     {"type": "keyword"},
                "file_type":     {"type": "keyword"},
                "file_size":     {"type": "long"},
                "modified_time": {"type": "double"},
            }
        }
    })
    print(f"[ES] Created index '{ES_INDEX}'")


def _index_to_es(file_path: str, file_name: str, file_type: str,
                 file_size: int, modified_time: float,
                 summary: str, keywords: str, raw_text: str):
    """Index a document into Elasticsearch."""
    es = _get_es()
    if es is None:
        return

    doc = {
        "file_name": file_name,
        "summary": summary,
        "keywords": keywords,
        "raw_text": raw_text,
        "file_path": file_path,
        "file_type": file_type,
        "file_size": file_size,
        "modified_time": modified_time,
    }
    # Use file_path as the document ID for easy upsert
    doc_id = _path_to_id(file_path)
    es.index(index=ES_INDEX, id=doc_id, document=doc)


def _delete_from_es(file_path: str):
    """Delete a document from Elasticsearch."""
    es = _get_es()
    if es is None:
        return
    doc_id = _path_to_id(file_path)
    try:
        es.delete(index=ES_INDEX, id=doc_id)
    except Exception:
        pass  # Ignore if not found


def _path_to_id(file_path: str) -> str:
    """Convert a file path to an ES-safe document ID."""
    # Normalize first, then replace special chars
    normalized = _normalize_path(file_path)
    return re.sub(r'[^a-zA-Z0-9_.\-]', '_', normalized)


def _normalize_path(path: str) -> str:
    """Normalize a file path for consistent storage and lookup.
    Ensures forward-slash paths (tkinter) and backslash paths (watchdog)
    are stored in the same format.
    """
    return os.path.normpath(path)


# ──────────────── SQLite ────────────────

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrency (multiple readers + 1 writer)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create SQLite tables and ensure ES index exists."""
    conn = get_connection()
    cursor = conn.cursor()

    # Files metadata table (no FTS5 needed now - ES handles search)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            file_name TEXT NOT NULL,
            file_type TEXT,
            file_size INTEGER,
            modified_time REAL,
            summary TEXT,
            keywords TEXT,
            raw_text TEXT,
            indexed_at REAL
        )
    """)

    # Store monitored folders
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watched_folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_path TEXT UNIQUE NOT NULL,
            added_at REAL
        )
    """)

    # Store generic settings (llm_model, etc.)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    # Insert default model if not exists
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('llm_model', 'gemma3:4b')")

    conn.commit()
    conn.close()

    # Initialize Elasticsearch index (defensively)
    try:
        _ensure_index()
    except Exception as e:
        print(f"[ES] Warning: Error during index ensuring: {e}")


# ──────────────── CRUD ────────────────

def get_setting(key: str, default: str = "") -> str:
    """Get a setting value from SQLite."""
    conn = get_connection()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key: str, value: str):
    """Set a setting value in SQLite."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO settings (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
    """, (key, value))
    conn.commit()
    conn.close()


def upsert_file(file_path: str, file_name: str, file_type: str,
                file_size: int, modified_time: float,
                summary: str, keywords: str, raw_text: str):
    """Insert or update a file record in both SQLite and ES."""
    from llm import ai_logger
    ai_logger.info(f"[DB] Upserting {file_name}: keywords={len(keywords)}, summary={len(summary)}")
    file_path = _normalize_path(file_path)
    conn = get_connection()
    conn.execute("""
        INSERT INTO files (file_path, file_name, file_type, file_size, modified_time,
                           summary, keywords, raw_text, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            file_name=excluded.file_name,
            file_type=excluded.file_type,
            file_size=excluded.file_size,
            modified_time=excluded.modified_time,
            summary=excluded.summary,
            keywords=excluded.keywords,
            raw_text=excluded.raw_text,
            indexed_at=excluded.indexed_at
    """, (file_path, file_name, file_type, file_size, modified_time,
          summary, keywords, raw_text, time.time()))
    conn.commit()
    conn.close()

    # Also index into Elasticsearch
    try:
        _index_to_es(file_path, file_name, file_type, file_size,
                     modified_time, summary, keywords, raw_text)
    except Exception as e:
        print(f"[ES] Warning: Failed to index {file_name}: {e}")


def search(query: str, limit: int = 20) -> list[dict]:
    """
    Search files using Elasticsearch multi_match with fuzziness.
    Falls back to SQLite LIKE if ES is unavailable.
    """
    es = _get_es()
    if es is not None:
        return _search_es(query, limit)
    else:
        print("[Search] ES unavailable, falling back to SQLite LIKE search")
        return _search_sqlite_fallback(query, limit)



def _search_es(query: str, limit: int = 20) -> list[dict]:
    """Search using Elasticsearch multi_match + fuzzy."""
    es = _get_es()
    if es is None:
        return []

    if not query:
        # Empty query: return most recent files using match_all
        body = {
            "size": limit,
            "query": {"match_all": {}},
            "sort": [{"modified_time": {"order": "desc"}}],
            "_source": ["file_path", "file_name", "file_type", "file_size",
                        "summary", "keywords", "modified_time"],
        }
    else:
        # Improved multi-match strategy for long conceptual queries
        body = {
            "size": limit,
            "min_score": 0.5,   # Filter out low-relevance results
            "query": {
                "bool": {
                    "should": [
                        # 1. Exact phrase match across important fields (Priority 1)
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["file_name^10", "keywords^8", "summary^5"],
                                "type": "phrase",
                                "boost": 5.0
                            }
                        },
                        # 2. Keyword full-text match — preserves multi-word terms (Priority 2)
                        {
                            "match": {
                                "keywords.full": {
                                    "query": query,
                                    "operator": "or",
                                    "minimum_should_match": "40%",
                                    "boost": 3.0
                                }
                            }
                        },
                        # 3. Comma-tokenized keyword match — exact multi-word keyword phrases (Priority 2b)
                        {
                            "match": {
                                "keywords": {
                                    "query": query,
                                    "operator": "or",
                                    "minimum_should_match": "40%",
                                    "boost": 2.5
                                }
                            }
                        },
                        # 4. Broad cross-field search (Priority 3)
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["file_name^3", "summary^2", "raw_text"],
                                "type": "cross_fields",
                                "operator": "or",
                                "minimum_should_match": "30%",
                                "boost": 1.0
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": ["file_path", "file_name", "file_type", "file_size",
                         "summary", "keywords", "modified_time"],
        }

    try:
        resp = es.search(index=ES_INDEX, body=body)
        results = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            src["relevance"] = hit["_score"]
            results.append(src)
        return results
    except Exception as e:
        print(f"[ES] Search error: {e}")
        return []


def _search_sqlite_fallback(query: str, limit: int = 20) -> list[dict]:
    """Fallback: simple SQLite LIKE search when ES is unavailable."""
    conn = get_connection()
    if not query:
        # Empty query: return most recent files
        rows = conn.execute(f"""
            SELECT file_path, file_name, file_type, file_size,
                   summary, keywords, modified_time
            FROM files
            ORDER BY indexed_at DESC
            LIMIT ?
        """, [limit]).fetchall()
    else:
        terms = query.split()
        if not terms:
            conn.close()
            return []

        # Build LIKE conditions for each term
        conditions = []
        params = []
        for term in terms:
            like = f"%{term}%"
            conditions.append(
                "(file_name LIKE ? OR summary LIKE ? OR keywords LIKE ?)"
            )
            params.extend([like, like, like])

        where_clause = " OR ".join(conditions)
        rows = conn.execute(f"""
            SELECT file_path, file_name, file_type, file_size,
                   summary, keywords, modified_time
            FROM files
            WHERE {where_clause}
            ORDER BY modified_time DESC
            LIMIT ?
        """, params + [limit]).fetchall()

    conn.close()
    return [dict(row) for row in rows]


def get_file_modified_time(file_path: str) -> Optional[float]:
    """Get the stored modified_time for a file, or None if not indexed."""
    file_path = _normalize_path(file_path)
    conn = get_connection()
    row = conn.execute(
        "SELECT modified_time FROM files WHERE file_path = ?", (file_path,)
    ).fetchone()
    conn.close()
    return row["modified_time"] if row else None


def get_stats() -> dict:
    """Get indexing statistics."""
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]

    type_counts = conn.execute(
        "SELECT file_type, COUNT(*) as c FROM files GROUP BY file_type ORDER BY c DESC"
    ).fetchall()

    conn.close()

    # Also show ES status
    es = _get_es()
    es_status = "connected" if es else "disconnected"

    return {
        "total_files": total,
        "by_type": {row["file_type"]: row["c"] for row in type_counts},
        "search_engine": "elasticsearch" if es else "sqlite_fallback",
        "es_status": es_status,
    }


def clear_db():
    """Clear all indexed data from both SQLite and ES."""
    conn = get_connection()
    conn.execute("DELETE FROM files")
    conn.execute("DELETE FROM watched_folders")
    conn.commit()
    conn.close()

    # Clear ES index
    es = _get_es()
    if es is not None:
        try:
            es.indices.delete(index=ES_INDEX)
            _ensure_index()  # Recreate empty index
        except Exception as e:
            print(f"[ES] Warning: Failed to clear index: {e}")


def add_watched_folder(folder_path: str):
    """Add a folder to watch list."""
    folder_path = _normalize_path(folder_path)
    conn = get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO watched_folders (folder_path, added_at) VALUES (?, ?)",
        (folder_path, time.time())
    )
    conn.commit()
    conn.close()


def get_watched_folders() -> list[str]:
    """Get all watched folders."""
    conn = get_connection()
    rows = conn.execute("SELECT folder_path FROM watched_folders").fetchall()
    conn.close()
    return [row["folder_path"] for row in rows]


def remove_watched_folder(folder_path: str):
    """Remove a folder from watch list and its indexed files."""
    folder_path = _normalize_path(folder_path)
    conn = get_connection()
    # Remove folder from watch list
    conn.execute("DELETE FROM watched_folders WHERE folder_path = ?", (folder_path,))
    # Remove all files under this folder
    conn.execute("DELETE FROM files WHERE file_path LIKE ?", (folder_path + "%",))
    conn.commit()
    conn.close()

    # Also remove from ES
    es = _get_es()
    if es is not None:
        try:
            es.delete_by_query(index=ES_INDEX, body={
                "query": {"prefix": {"file_path": folder_path}}
            }, refresh=True)
        except Exception as e:
            print(f"[ES] Warning: Failed to remove folder files: {e}")


def remove_file(file_path: str):
    """Remove a file from both SQLite and ES index."""
    file_path = _normalize_path(file_path)
    conn = get_connection()
    conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
    conn.commit()
    conn.close()

    _delete_from_es(file_path)
