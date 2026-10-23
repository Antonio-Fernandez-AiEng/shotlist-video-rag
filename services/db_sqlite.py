import os
import sqlite3
import hashlib
from contextlib import contextmanager
from pathlib import Path
from config import SQLITE_DB_PATH

SQLITE_DB_PATH.resolve().parent.mkdir(parents=True, exist_ok=True)
SQLITE_DB_PATH.touch(exist_ok=True)

@contextmanager
def get_db_connection():
    """
    Context manager that ensures an isolated connection per thread,
    handles transactions securely and guarantees the closure of the resource.
    """
  
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row  
    try:
        yield conn
        conn.commit()  # Saves the changes automatically if everything goes well
    except Exception:
        conn.rollback()  # Reverts the changes if an error occurs to avoid corrupting the DB
        raise            # Reraises the error for FastAPI to detect
    finally:
        conn.close()     # Closes the connection regardless of what happens


def init_db():
    """Initialize the table of processed videos if it doesn't exist."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_videos (
                file_hash TEXT PRIMARY KEY,
                task_id TEXT,
                movie_title TEXT,
                filename TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def get_video_by_hash(file_hash: str):
    """Search if a video has already been processed previously using its hash."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT task_id, movie_title FROM processed_videos WHERE file_hash = ?", 
            (file_hash,)
        )
        return cursor.fetchone()  # Returns a row or None if it doesn't exist


def save_processed_video(file_hash: str, task_id: str, movie_title: str, filename: str):
    """Register a new video processed in the database."""
    with get_db_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO processed_videos (file_hash, task_id, movie_title, filename)
            VALUES (?, ?, ?, ?)
        """, (file_hash, task_id, movie_title, filename))


def calculate_file_hash(file_path: Path, chunk_size: int = 65536) -> str:
    """
    Calculates a fast hash by reading only key parts of the file (beginning, middle, and end).
    Ideal for large video files without blocking the server.
    """
    sha256_hash = hashlib.sha256()
    file_size = file_path.stat().st_size

    with open(file_path, "rb") as f:
        #Read the part of the file to hash(this way we dont depend on filename)
        sha256_hash.update(f.read(chunk_size))
        
        
        if file_size > chunk_size * 3:
            f.seek(file_size // 2)
            sha256_hash.update(f.read(chunk_size))
            
        
        if file_size > chunk_size * 2:
            f.seek(chunk_size, os.SEEK_END)
            sha256_hash.update(f.read(chunk_size))

    return sha256_hash.hexdigest()