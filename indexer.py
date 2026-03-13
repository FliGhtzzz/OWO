"""
Indexer - Scan folders, parse files, extract keywords via LLM, and store in database.
"""
import os
import time
import asyncio
from typing import Callable, Optional

# Per-file timeout limits (seconds) — if Ollama hangs, skip the file
LLM_TIMEOUT_IMAGE = 25    # Vision models shouldn't take minutes anymore with devstral
LLM_TIMEOUT_TEXT  = 15    # Text extraction is fast

from file_parser import parse_file, get_file_category, get_document_images, cleanup_temp_images
from llm import extract_keywords, describe_image, ai_logger
from database import upsert_file, get_file_modified_time, add_watched_folder

# Skip files larger than 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024

# Skip these directories
SKIP_DIRS = {
    "__pycache__", ".git", ".svn", "node_modules", ".venv", "venv",
    ".idea", ".vscode", ".vs", "dist", "build", ".next",
}

# Progress state (global for simplicity)
indexing_state = {
    "is_indexing": False,
    "cancel": False,
    "folder": "",
    "total_files": 0,
    "processed_files": 0,
    "current_file": "",
    "errors": [],
    "start_time": 0,
}


def scan_folder(folder_path: str) -> list[str]:
    """Recursively scan a folder and return all file paths."""
    files = []
    for root, dirs, filenames in os.walk(folder_path):
        # Skip hidden/system directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

        for filename in filenames:
            if filename.startswith("."):
                continue
            file_path = os.path.join(root, filename)
            try:
                size = os.path.getsize(file_path)
                if size > MAX_FILE_SIZE or size == 0:
                    continue
                files.append(file_path)
            except OSError:
                continue
    return files


def _merge_results(results: list[dict]) -> dict:
    """Merge multiple LLM results (summary + keywords) into one."""
    all_keywords = []
    summaries = []

    for r in results:
        keywords = r.get("keywords", [])
        all_keywords.extend(keywords)
        summary = r.get("summary", "")
        if summary:
            summaries.append(summary)

    # Deduplicate keywords (case-insensitive) while preserving order
    seen = set()
    unique_keywords = []
    for kw in all_keywords:
        kw_lower = kw.lower().strip()
        if kw_lower and kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw.strip())

    return {
        "summary": " ".join(summaries[:3]),  # Keep first 3 summaries
        "keywords": unique_keywords,
    }


async def index_file(file_path: str) -> bool:
    """
    Index a single file: parse -> LLM -> database.
    Returns True if successful, False otherwise.
    Raises: LLMFailedError if LLM fails (to be caught by index_folder)
    """
    try:
        file_path = os.path.normpath(file_path)
        file_name = os.path.basename(file_path)
        file_type = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        modified_time = os.path.getmtime(file_path)

        # Check if file needs re-indexing
        stored_mtime = get_file_modified_time(file_path)
        if stored_mtime is not None and abs(stored_mtime - modified_time) < 1:
            return True  # Already indexed and not modified

        category = get_file_category(file_path)

        if category == "image":
            # Vision model handles its own internal retry
            result = await describe_image(file_path, file_name)
            raw_text = ""
        elif category == "document":
            # Hybrid approach: extract text + extract images, send both to LLM
            result = await _index_document(file_path, file_name)
            raw_text = ""
            # Also try to get raw text for search
            text_content, _ = parse_file(file_path)
            if text_content and text_content.strip():
                raw_text = text_content
        else:
            # Parse text content first
            text_content, parsed_category = parse_file(file_path)
            if text_content is None and parsed_category == "error":
                return False
            if text_content is None or not text_content.strip():
                # Empty file, store with minimal info
                upsert_file(
                    file_path=file_path,
                    file_name=file_name,
                    file_type=file_type,
                    file_size=file_size,
                    modified_time=modified_time,
                    summary=f"Empty or binary file: {file_name}",
                    keywords=file_name,
                    raw_text="",
                )
                return True

            raw_text = text_content
            # Text model handles its own internal retry
            result = await extract_keywords(text_content, file_name)
            ai_logger.info(f"[Indexer] {file_name}: Extracted {len(result.get('keywords', []))} keywords.")

        # ── Final sanitisation before writing to DB ──────────────────────
        # 1. Deduplicate keywords (case-insensitive), keep order, cap at 50
        raw_kw_list = result.get("keywords", [])
        seen_kw: set[str] = set()
        deduped_kw: list[str] = []
        for kw in raw_kw_list:
            kw_clean = str(kw).strip()
            kl = kw_clean.lower()
            if kl and kl not in seen_kw:
                seen_kw.add(kl)
                deduped_kw.append(kw_clean)
            if len(deduped_kw) >= 50:
                break
        keywords_str = ", ".join(deduped_kw)

        # 2. Truncate summary to 600 chars at a sentence boundary if possible
        raw_summary = result.get("summary", "")
        if len(raw_summary) > 600:
            truncated = raw_summary[:600]
            last_period = truncated.rfind(".")
            raw_summary = (truncated[:last_period + 1] if last_period > 300 else truncated).strip()

        ai_logger.info(f"[Indexer] {file_name}: Saving to DB. summary={len(raw_summary)} chars, keywords={len(deduped_kw)}")
        upsert_file(
            file_path=file_path,
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            modified_time=modified_time,
            summary=raw_summary,
            keywords=keywords_str,
            raw_text=raw_text if category != "image" else "",
        )
        return True


    except Exception as e:
        print(f"[Indexer] Error indexing {file_path}: {e}")
        return False


async def _index_document(file_path: str, file_name: str) -> dict:
    """
    Index a document file (PDF, DOCX, XLSX, PPTX) using a hybrid approach:
    1. Extract text → send to text LLM for keyword extraction
    2. Extract/render images → send to vision LLM for analysis
    3. Merge all results
    """
    results = []

    # Step 1: Text extraction → LLM
    text_content, _ = parse_file(file_path)
    if text_content and text_content.strip():
        try:
            text_result = await extract_keywords(text_content, file_name)
            results.append(text_result)
            print(f"[Indexer] {file_name}: extracted {len(text_result.get('keywords', []))} text keywords")
        except Exception as e:
            print(f"[Indexer] Error extracting text keywords from {file_name}: {e}")
            raise e # Strict: if part of it fails, let index_file handle retry

    # Step 2: Image extraction → Vision LLM
    image_paths = []
    try:
        image_paths = get_document_images(file_path)
        if image_paths:
            print(f"[Indexer] {file_name}: found {len(image_paths)} images, analyzing...")
            # Process up to 5 images to avoid being too slow
            for i, img_path in enumerate(image_paths[:5]):
                try:
                    img_result = await describe_image(img_path, f"{file_name} (image {i+1})")
                    results.append(img_result)
                except Exception as e:
                    print(f"[Indexer] Error describing image {i+1} from {file_name}: {e}")
                    raise e # Strict indexing
    finally:
        # Clean up temp images
        cleanup_temp_images(image_paths)

    # Step 3: Merge results
    if results:
        return _merge_results(results)
    else:
        return {"summary": f"Document file: {file_name}", "keywords": [file_name]}


async def index_folder(folder_path: str):
    """
    Index all files in a folder. Updates global indexing_state for progress tracking.
    """
    global indexing_state

    if indexing_state["is_indexing"]:
        return

    indexing_state.update({
        "is_indexing": True,
        "folder": folder_path,
        "total_files": 0,
        "processed_files": 0,
        "current_file": "Scanning folder...",
        "errors": [],
        "start_time": time.time(),
    })

    try:
        # Scan for files
        files = scan_folder(folder_path)
        indexing_state["total_files"] = len(files)
        
        # Add to watched folders
        add_watched_folder(folder_path)
        
        # Start watching immediately if watcher is running
        try:
            from watcher import watcher
            watcher.add_watch(folder_path)
        except ImportError:
            pass

        # Three-pass indexing strategy
        failed_files = []
        
        # Pass 1: Initial Attempt
        ai_logger.info(f"[Indexer] PASS 1: Indexing {len(files)} files...")
        for i, file_path in enumerate(files):
            if indexing_state.get("cancel"):
                ai_logger.info("[Indexer] Indexing cancelled.")
                break
                
            indexing_state["current_file"] = os.path.basename(file_path)
            indexing_state["processed_files"] = i

            try:
                success = await index_file(file_path)
                if not success:
                    failed_files.append(file_path)
            except Exception as e:
                ai_logger.warning(f"[Indexer] {os.path.basename(file_path)} failed Pass 1: {e}")
                failed_files.append(file_path)

        # Pass 2: Retry Failed Files
        if failed_files and not indexing_state.get("cancel"):
            ai_logger.info(f"[Indexer] PASS 2: Retrying {len(failed_files)} failed files...")
            indexing_state["current_file"] = f"Retry Pass 2 ({len(failed_files)} files)..."
            still_failed = []
            for file_path in failed_files:
                if indexing_state.get("cancel"): break
                try:
                    if await index_file(file_path):
                        continue
                    still_failed.append(file_path)
                except Exception:
                    still_failed.append(file_path)
            failed_files = still_failed

        # Pass 3: Final Retry
        if failed_files and not indexing_state.get("cancel"):
            ai_logger.info(f"[Indexer] PASS 3: Final retry for {len(failed_files)} files...")
            indexing_state["current_file"] = f"Final Retry Pass 3 ({len(failed_files)} files)..."
            still_failed = []
            for file_path in failed_files:
                if indexing_state.get("cancel"): break
                try:
                    if await index_file(file_path):
                        continue
                    still_failed.append(file_path)
                except Exception:
                    still_failed.append(file_path)
            
            if still_failed:
                ai_logger.error(f"[Indexer] Failed to process {len(still_failed)} files after 3 passes.")
                from llm import ai_log_file
                msg = f"注意：有 {len(still_failed)} 個檔案多次重試後仍無法索引，已跳過以避免污然資料庫。"
                indexing_state["errors"].append(msg)
                
                # Write to failed_files.txt
                with open(os.path.join(os.path.dirname(ai_log_file), "failed_files.txt"), "w", encoding="utf-8") as f:
                    f.write("以下檔案多次產生索引失敗（可能 LLM 伺服器超載或超時），已自動跳過：\n")
                    for fn in still_failed:
                        f.write(f"- {fn}\n")

        indexing_state["processed_files"] = len(files)
        indexing_state["current_file"] = "Done!"

    except Exception as e:
        indexing_state["errors"].append(f"Fatal error: {e}")
    finally:
        indexing_state["is_indexing"] = False
        indexing_state["cancel"] = False


def cancel_index():
    """Cancel the current indexing process."""
    if indexing_state["is_indexing"]:
        indexing_state["cancel"] = True


def get_index_status() -> dict:
    """Get current indexing status."""
    elapsed = 0
    if indexing_state["start_time"] > 0:
        if indexing_state["is_indexing"]:
            elapsed = time.time() - indexing_state["start_time"]
        else:
            elapsed = indexing_state.get("_end_time", time.time()) - indexing_state["start_time"]

    return {
        **indexing_state,
        "elapsed_seconds": round(elapsed, 1),
    }
