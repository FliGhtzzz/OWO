"""
File Guessr - Natural Language File Search Tool
FastAPI application with Web UI.
"""
import os
import sys
import asyncio
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, Query, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

import database
from indexer import index_folder, get_index_status
from searcher import search_files
from llm import check_ollama_status, expand_query_with_file
from file_parser import parse_file, get_file_category

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database on startup
    database.init_db()

    # Start file watcher
    from watcher import watcher
    watcher.start()

    # Background thread: keep retrying ES connection until it succeeds.
    # This handles the common case where ES takes > 60s to start (Windows service
    # slow startup) and uvicorn was already launched before ES was ready.
    def _es_reconnect_worker():
        attempt = 0
        while True:
            es = database._get_es()
            if es is not None:
                print(f"[ES] Background reconnect: connected after {attempt} retries.")
                # Also try to ensure the index exists now
                try:
                    database._ensure_index()
                except Exception:
                    pass
                return
            attempt += 1
            time.sleep(20)

    threading.Thread(target=_es_reconnect_worker, daemon=True, name="es-reconnect").start()

    yield

    # Stop watcher
    watcher.stop()


app = FastAPI(title="File Guessr", lifespan=lifespan)

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ──── Pages ────

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ──── API Endpoints ────

@app.get("/api/browse")
async def browse_folder():
    """Open a native folder picker dialog and return the selected path."""
    try:
        selected = await asyncio.to_thread(_open_folder_dialog)
        if selected:
            return {"path": selected}
        return {"path": None, "message": "No folder selected"}
    except Exception as e:
        print(f"[Browse] Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


def _open_folder_dialog() -> Optional[str]:
    """Open folder dialog via subprocess (tkinter can't run in non-main thread)."""
    try:
        script = (
            "import tkinter as tk; "
            "from tkinter import filedialog; "
            "root = tk.Tk(); "
            "root.withdraw(); "
            "root.attributes('-topmost', True); "
            "path = filedialog.askdirectory(title='選擇要索引的資料夾'); "
            "print(path if path else ''); "
            "root.destroy()"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120
        )
        path = result.stdout.strip()
        return path if path else None
    except subprocess.TimeoutExpired:
        print("[Browse] Dialog timed out")
        return None
    except Exception as e:
        print(f"[Browse] Error: {e}")
        return None


@app.post("/api/index")
async def start_indexing(body: dict, background_tasks: BackgroundTasks):
    """Start indexing a folder."""
    folder_path = body.get("folder_path", "").strip()

    if not folder_path:
        return JSONResponse({"error": "folder_path is required"}, status_code=400)

    if not os.path.isdir(folder_path):
        return JSONResponse({"error": f"Folder not found: {folder_path}"}, status_code=400)

    status = get_index_status()
    if status["is_indexing"]:
        return JSONResponse({"error": "Indexing is already in progress"}, status_code=409)

    # Run indexing in background
    background_tasks.add_task(index_folder, folder_path)

    return {"message": "Indexing started", "folder": folder_path}


@app.get("/api/index/status")
async def indexing_status():
    """Get current indexing progress."""
    return get_index_status()


@app.get("/api/search")
async def search(q: str = Query("")):
    """Search files with natural language query. Empty query returns all files."""
    q = q.strip()
    result = await search_files(q)
    return result


@app.post("/api/search/multimodal")
async def search_multimodal(
    file: UploadFile = File(...),
    q: str = Form(default=""),
):
    """Search using both text and an uploaded file (image or document)."""
    temp_path = None
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename or "")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        # Determine how to process the file
        category = get_file_category(temp_path)
        file_content = None
        image_path = None

        if category == "image":
            # Send image directly to LLM vision
            image_path = temp_path
        else:
            # Parse text content from the file
            text, _ = parse_file(temp_path)
            file_content = text

        # LLM: combine text query + file to generate search keywords
        expanded_keywords = await expand_query_with_file(
            user_query=q,
            file_content=file_content,
            image_path=image_path,
        )
        
        # Use ONLY the expanded keywords for the search query.
        # The raw natural language input (e.g. "Help me find...") is noisy.
        final_query = expanded_keywords.strip()
        
        print(f"[MultiSearch] Query: '{q}' + File: '{file.filename}' → Keywords: '{final_query}'")

        # Search with enhanced keywords
        from database import search as db_search
        results = db_search(final_query, limit=20)

        return {
            "original_query": q,
            "uploaded_file": file.filename,
            "expanded_query": final_query,
            "total_results": len(results),
            "results": results,
        }

    except Exception as e:
        print(f"[MultiSearch] Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@app.get("/api/folders")
async def list_folders():
    """Get all watched folders."""
    try:
        folders = database.get_watched_folders()
        return {"folders": folders}
    except Exception as e:
        print(f"[Folders] Error listing folders: {e}")
        return {"folders": []}


@app.post("/api/folders/remove")
async def remove_folder(body: dict):
    """Remove a folder from the watch list."""
    folder_path = body.get("folder_path", "").strip()
    if not folder_path:
        return JSONResponse({"error": "folder_path is required"}, status_code=400)

    database.remove_watched_folder(folder_path)

    # Stop watching this folder
    try:
        from watcher import watcher
        watcher.remove_watch(folder_path)
    except Exception:
        pass

    return {"message": f"Removed: {folder_path}"}


@app.get("/api/stats")
async def stats():
    """Get index statistics."""
    return database.get_stats()


@app.get("/api/health")
async def health():
    """Check Ollama and model status."""
    return await check_ollama_status()


@app.get("/api/llm/models")
async def get_llm_models():
    """Get all available Ollama models and current selection."""
    status = await check_ollama_status()
    return status


@app.post("/api/llm/model")
async def set_llm_model(body: dict):
    """Update the selected LLM model."""
    model_name = body.get("model", "").strip()
    if not model_name:
        return JSONResponse({"error": "model name is required"}, status_code=400)

    database.set_setting("llm_model", model_name)
    print(f"[API] Set LLM model to: {model_name}")

    # Trigger cache clear in llm.py
    from llm import _clear_llm_cache
    _clear_llm_cache()

    return {"message": f"Model updated to: {model_name}"}


@app.get("/api/llm/logs")
async def get_llm_logs():
    """Get the last N lines of the AI engine log."""
    from llm import ai_log_file
    if not os.path.exists(ai_log_file):
        return {"logs": "Log file not found."}
    
    try:
        with open(ai_log_file, "r", encoding="utf-8") as f:
            # Get last 100 lines
            lines = f.readlines()
            return {"logs": "".join(lines[-100:])}
    except Exception as e:
        return {"logs": f"Error reading logs: {e}"}


@app.post("/api/clear")
async def clear_index():
    """Clear all indexed data and forcefully stop any active indexing."""
    status = get_index_status()
    if status["is_indexing"]:
        from indexer import cancel_index
        cancel_index()
        # Give it a small moment to break the loop
        await asyncio.sleep(0.5)

    database.clear_db()
    
    # Trigger cache clear in llm.py
    from llm import _clear_llm_cache
    _clear_llm_cache()
    
    return {"message": "Index cleared"}


@app.get("/api/file/preview")
async def file_preview(path: str = Query(...)):
    """Serve a file for preview (mainly for images)."""
    if not os.path.isfile(path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    return FileResponse(path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
