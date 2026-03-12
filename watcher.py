"""
File Watcher - Monitor folders for changes and update index automatically.
"""
import time
import os
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import database
from indexer import index_file

# Debounce time in seconds
DEBOUNCE_DELAY = 1.0


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop
        self.pending_files = {}
        self.processing = False

    def on_created(self, event):
        if not event.is_directory:
            self._schedule_index(os.path.normpath(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule_index(os.path.normpath(event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            # Remove old file from index
            self.loop.call_soon_threadsafe(
                database.remove_file, os.path.normpath(event.src_path)
            )
            # Index new file
            self._schedule_index(os.path.normpath(event.dest_path))

    def on_deleted(self, event):
        if not event.is_directory:
            normalized = os.path.normpath(event.src_path)
            print(f"[Watcher] File deleted: {normalized}")
            self.loop.call_soon_threadsafe(
                database.remove_file, normalized
            )

    def _schedule_index(self, file_path: str):
        """Schedule file for indexing with debounce."""
        print(f"[Watcher] Change detected: {file_path}")
        # Run indexing in the main loop
        asyncio.run_coroutine_threadsafe(self._process_file(file_path), self.loop)

    async def _process_file(self, file_path: str):
        # Small delay to let file write finish
        await asyncio.sleep(1.0)
        try:
            if os.path.exists(file_path):
                print(f"[Watcher] Indexing: {file_path}")
                await index_file(file_path)
        except Exception as e:
            print(f"[Watcher] Error processing {file_path}: {e}")


class WatcherManager:
    def __init__(self):
        self.observer = Observer()
        self.watched_paths = set()
        self.handler = None
        self._loop = None

    def start(self):
        """Start watching all folders in DB."""
        self._loop = asyncio.get_event_loop()
        self.handler = FileChangeHandler(self._loop)
        
        # Load existing folders
        folders = database.get_watched_folders()
        for folder in folders:
            self.add_watch(folder)
            
        self.observer.start()
        print(f"[Watcher] Started monitoring {len(self.watched_paths)} folders")

    def stop(self):
        self.observer.stop()
        self.observer.join()

    def add_watch(self, folder_path: str):
        """Add a folder to watch if not already watched."""
        if folder_path in self.watched_paths:
            return

        if not os.path.isdir(folder_path):
            return

        try:
            self.observer.schedule(self.handler, folder_path, recursive=True)
            self.watched_paths.add(folder_path)
            database.add_watched_folder(folder_path)
            print(f"[Watcher] Watching: {folder_path}")
        except Exception as e:
            print(f"[Watcher] Failed to watch {folder_path}: {e}")

    def remove_watch(self, folder_path: str):
        """Remove a folder from watching."""
        self.watched_paths.discard(folder_path)
        # Note: watchdog Observer doesn't easily support unscheduling
        # individual watches, so the folder will stop being watched
        # on next restart. We remove it from our tracked set.
        print(f"[Watcher] Removed watch: {folder_path}")


# Global instance
watcher = WatcherManager()
