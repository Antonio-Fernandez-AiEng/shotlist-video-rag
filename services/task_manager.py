import time
import traceback
from pathlib import Path
from services.db_manager import db_manager
from services.video_processor import VideoProcessor

# Global dictionaries to track the status of individual tasks and batch tasks
tasks_db = {}
batch_tasks_db = {}


def process_video_background(
    task_id: str,
    video_path: Path,
    movie_title: str,
    output_dir: Path,
    processor: VideoProcessor,
):
    """Process a single video in the background using FastAPI BackgroundTasks."""
    try:
        print(f"🔥 [BG-TASK] Starting processing for task_id: {task_id} ('{movie_title}')")
        t_start_processing = time.perf_counter()
        tasks_db[task_id] = {
            "status": "processing",
            "movie_title": movie_title,
            "step": "Extracting and vectorizing frames...",
        }

        # 1. Extract frames and embeddings
        shots_data = processor.extract_and_vectorize(
            video_path, output_dir, movie_title=movie_title
        )
        print(f"🔥 [BG-TASK] Extraction completed for {task_id}. Shots obtained: {len(shots_data)}")

        # 2. Save in ChromaDB
        db_manager.add_shots(shots_data)

        t_total_processing = (time.perf_counter() - t_start_processing) * 1000


        # 3. Mark as completed
        tasks_db[task_id] = {
            "status": "completed",
            "movie_title": movie_title,
            "message": "Video processed successfully and shots indexed.",
            "total_shots_indexed": len(shots_data),
        }
        print(f"🔥 [BG-TASK] Task {task_id} finished successfully. Total shots indexed: {len(shots_data)}")
        print(f"🎬 [ML METRIC] Video '{movie_title}' processed successfully: {len(shots_data)} shots extracted & vectorized in {t_total_processing / 1000:.2f}s")

    except Exception as e:
        print(f"❌ [ERROR IN BACKGROUND TASK]: {str(e)}")
        traceback.print_exc()

        tasks_db[task_id] = {
            "status": "failed",
            "movie_title": movie_title,
            "error": str(e),
        }


def process_batch_background(
    batch_id: str,
    items_to_process: list,
    output_dir: Path,
    processor: VideoProcessor,
):
    """Process a whole batch of videos sequentially in the background."""
    completed = 0
    failed = 0

    for item in items_to_process:
        task_id = item["task_id"]
        
        process_video_background(
            task_id=task_id,
            video_path=item["video_path"],
            movie_title=item["movie_title"],
            output_dir=output_dir,
            processor=processor,
        )

        if tasks_db[task_id]["status"] == "completed":
            completed += 1
        else:
            failed += 1

        batch_tasks_db[batch_id]["completed_files"] = completed
        batch_tasks_db[batch_id]["failed_files"] = failed

    batch_tasks_db[batch_id]["overall_status"] = (
        "completed" if failed == 0 else "completed_with_errors"
    )