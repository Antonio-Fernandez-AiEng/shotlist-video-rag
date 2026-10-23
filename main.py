import shutil
import uuid
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
import asyncio
import json

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from schemas import SearchQuery, SearchResponse
from services.db_manager import db_manager
from services.task_manager import (
    batch_tasks_db,
    process_batch_background,
    process_video_background,
    tasks_db,
)
from services.video_processor import VideoProcessor
from services.db_sqlite import (
    calculate_file_hash, 
    get_video_by_hash, 
    init_db,                
    save_processed_video
)
from config import SHOTS_DIR, VIDEOS_DIR

processor: VideoProcessor | None = None

# ------------------------------------------------------------------------------
# 1. LIFESPAN
# ------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor

    print("🚀 [LIFESPAN] Starting backend services")
    
    # 1. Start SQLite for hash storage and duplicate detection
    init_db()
    print("✅ [LIFESPAN] SQLite database started")

    # 2. Connect to ChromaDB and start SigLIP
    db_manager.connect()
    print("✅ [LIFESPAN] ChromaDB connection succesful")

    processor = VideoProcessor()
    print("✅ [LIFESPAN] SigLIP processor loaded")

    yield

    print("🛑 [LIFESPAN] Closing services")
    db_manager.disconnect()
    processor = None
    print("👋 [LIFESPAN] Resources liberated succesfully")

app = FastAPI(
    title="Filmmaker Shotlist RAG API",
    version="1.0",
    description="Multimodal RAG API for video shots search",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.mount("/api/frames", StaticFiles(directory=SHOTS_DIR), name="frames")


# ------------------------------------------------------------------------------
# 2. API ENDPOINTS
# ------------------------------------------------------------------------------

@app.get("/", response_class=FileResponse)
async def serve_index():
    return FileResponse("index.html")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy", "database": "connected"}


@app.get("/api/tasks/{task_id}/stream")
async def task_status_stream(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Tarea no encontrada.")

    async def event_generator():
        last_status = None
        while True:
            task_data = tasks_db.get(task_id, {})
            current_status = task_data.get("status")
            
            if task_data != last_status:
                yield f"data: {json.dumps(task_data)}\n\n"
                last_status = task_data.copy()

            if current_status in ["completed", "failed"]:
                break
                
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_video(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="MP$4 or MOV video files to be processed"),
    movie_title: List[str] | None = Form(default=None, description="Títle(s) optional"),
):
    if processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The video processor has not been started yet"
        )

    for file in files:
        if not file.filename.lower().endswith((".mp4", ".mov")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Format not supported for '{file.filename}'. Only .mp4 or .mov.",
            )

    batch_id = str(uuid.uuid4())
    items_to_process = []
    file_statuses = []

    for index, file in enumerate(files):
        task_id = str(uuid.uuid4())

        final_title = Path(file.filename).stem
        if movie_title is not None and isinstance(movie_title, list):
            if index < len(movie_title) and movie_title[index] and movie_title[index].strip():
                final_title = movie_title[index].strip()

        save_path = VIDEOS_DIR / f"{task_id}_{file.filename}"

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --- DUPLICATE VALIDATION WITH PARTIAL HASH (SHA256) ---
        file_hash = await asyncio.to_thread(calculate_file_hash, save_path)
        existing_video = await asyncio.to_thread(get_video_by_hash, file_hash)
        
        if existing_video:
            existing_task_id, existing_title = existing_video
            save_path.unlink()  
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"This video has already been processed as '{existing_title}' (Task ID: {existing_task_id})."
            )

        # Register succesfully the video in SQLite to avoid duplicates in future uploads
        await asyncio.to_thread(save_processed_video, file_hash, task_id, final_title, file.filename)

        tasks_db[task_id] = {
            "status": "pending",
            "movie_title": final_title,
            "filename": file.filename,
        }

        items_to_process.append({
            "task_id": task_id,
            "filename": file.filename,
            "movie_title": final_title,
            "video_path": save_path,
        })

        file_statuses.append({
            "task_id": task_id,
            "filename": file.filename,
            "movie_title": final_title,
            "status_url": f"/api/tasks/{task_id}",
        })

    if len(items_to_process) == 1:
        single_item = items_to_process[0]
        
        background_tasks.add_task(
            process_video_background,
            task_id=single_item["task_id"],
            video_path=single_item["video_path"],
            movie_title=single_item["movie_title"],
            output_dir=SHOTS_DIR,
            processor=processor,
        )

        return {
            "status": "accepted",
            "task_id": single_item["task_id"],
            "movie_title": single_item["movie_title"],
            "message": f"VVideo received. Processing '{single_item['movie_title']}' in background.",
            "status_url": f"/api/tasks/{single_item['task_id']}",
        }
    else:
        batch_tasks_db[batch_id] = {
            "batch_id": batch_id,
            "total_files": len(items_to_process),
            "completed_files": 0,
            "failed_files": 0,
            "overall_status": "processing",
            "files": file_statuses,
        }

        background_tasks.add_task(
            process_batch_background,
            batch_id=batch_id,
            items_to_process=items_to_process,
            output_dir=SHOTS_DIR,
            processor=processor,
        )

        return {
            "status": "accepted",
            "batch_id": batch_id,
            "total_files": len(items_to_process),
            "message": f"Batch of {len(items_to_process)} videos received. Processing in background.",
            "status_url": f"/api/tasks/batch/{batch_id}",
        }

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return tasks_db[task_id]


@app.get("/api/tasks/batch/{batch_id}")
async def get_batch_task_status(batch_id: str):
    if batch_id not in batch_tasks_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found")
    return batch_tasks_db[batch_id]


@app.post("/api/search", response_model=SearchResponse)
async def search_shots(payload: SearchQuery):
    if not processor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The video processor is not initialized yet.",
        )

    # ⏱️ 1. Measure encoding time of the Encoder (SigLIP)
    t_start_enc = time.perf_counter()
    query_vector = processor.encode_text_query(payload.query)
    t_encoder_ms = (time.perf_counter() - t_start_enc) * 1000

    # ⏱️ 2. Measure search time in ChromaDB
    t_start_db = time.perf_counter()
    results = db_manager.search_similar(
        query_vector=query_vector,
        n_results=payload.n_results,
        movie_title=payload.movie_title,
    )
    t_db_ms = (time.perf_counter() - t_start_db) * 1000

    print(f"📊 [ML METRIC] Query: '{payload.query}' | SigLIP Encoder: {t_encoder_ms:.2f}ms | ChromaDB Search: {t_db_ms:.2f}ms")

    if not results:
        return SearchResponse(query=payload.query, movie_title_filter=payload.movie_title, results=[])

    #Cleaning data and ensuring proper types for the response
    cleaned_results = []
    for r in results:
        meta = r.get("metadata") or {}
        cleaned_meta = {
            "movie_title": meta.get("movie_title") or "Desconocido",
            "shot_index": int(meta.get("shot_index", 0) or 0),
            "timestamp": float(meta.get("timestamp", 0.0) or 0.0),
            "file_path": str(meta.get("file_path", "") or "")
        }

        cleaned_results.append({
            "id": r.get("id"),
            "score": float(r.get("score", 0.0) or 0.0),
            "metadata": cleaned_meta
        })

    return SearchResponse(
        query=payload.query,
        movie_title_filter=payload.movie_title,
        results=cleaned_results,
    )