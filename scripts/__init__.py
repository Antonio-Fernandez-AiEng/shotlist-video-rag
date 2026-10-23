from pydantic import BaseModel, Field
from typing import List

#REQUESTS

class IngestRequest(BaseModel):
    video_path: str = Field(..., description="Ruta del video a procesar")
    movie_title: str = Field(..., description="Lista de shots a buscar en el video")

class SearchQuery(BaseModel):
    query: str = Field(..., description="Frase de búsqueda para encontrar planos similares")
    n_results: int = Field(default=5, ge=1, le=50, description="Número de resultados a devolver (por defecto 3)")
    movie_title: str | None = Field(None, description="Título de la película para filtrar resultados (opcional)")


#RESPONSES

class ShotMetadata(BaseModel):
    movie_title: str
    shot_index: int
    timestamp: float
    file_path: str

class ShotResult(BaseModel):
    id: str
    score: float
    metadata: ShotMetadata

class SearchResponse(BaseModel):
    query: str
    movie_title_filter: str | None = None
    results: List[ShotResult]