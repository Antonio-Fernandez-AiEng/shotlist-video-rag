from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

# Creamos el cliente de pruebas virtual para tu aplicación FastAPI
client = TestClient(app)

def test_health_check():
    """Prueba que el endpoint de salud responde correctamente."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    print("✅ [TEST] Health check pasado con éxito.")


@patch("main.processor")
@patch("main.calculate_file_hash")
@patch("main.get_video_by_hash")
@patch("main.save_processed_video")
def test_ingest_video_endpoint_mocked(
    mock_save_video, mock_get_hash, mock_calc_hash, mock_processor
):
    """Prueba el endpoint de ingesta (/api/ingest) utilizando Mocks para aislar 
    tanto la IA como la lógica de deduplicación de SQLite.
    """
    
    # 1. Simulamos el comportamiento del hash y la validación de duplicados en SQLite
    mock_calc_hash.return_value = "fake_sha256_hash_12345"
    mock_get_hash.return_value = None  # Simulamos que el vídeo NO es un duplicado (no existe en BD)

    # 2. Simulamos el comportamiento del procesador de vídeo para la tarea en background
    mock_processor.extract_and_vectorize.return_value = [
        {
            "id": "test-shot-uuid-1",
            "metadata": {
                "movie_title": "Test Movie",
                "timestamp": 12.5,
                "file_path": "data/extracted_shots/test_shot.jpg",
                "shot_index": 0
            }
        }
    ]

    # 3. Simulamos un archivo de vídeo binario falso (.mp4)
    fake_video_bytes = b"fake video content bytes for testing"
    files = {"files": ("test_trailer.mp4", fake_video_bytes, "video/mp4")}
    data = {"movie_title": ["Test Movie"]}

    # 4. Realizamos la petición POST virtual al endpoint
    response = client.post("/api/ingest", files=files, data=data)

    # 5. Verificaciones (Assertions)
    assert response.status_code == 202
    json_data = response.json()
    
    # Comprobamos que devuelve la estructura esperada para una tarea única
    assert json_data["status"] == "accepted"
    assert "task_id" in json_data
    assert json_data["movie_title"] == "Test Movie"
    assert "status_url" in json_data
    
    print("✅ [TEST] Endpoint de ingesta con deduplicación SQLite verificado con mocks correctamente.")


@patch("main.processor")
@patch("main.calculate_file_hash")
@patch("main.get_video_by_hash")
def test_ingest_duplicate_video_blocked(mock_get_hash, mock_calc_hash, mock_processor):
    """Prueba que el sistema rechaza correctamente un vídeo duplicado (400 Bad Request)
    gracias a la persistencia de hashes en SQLite.
    """
    mock_calc_hash.return_value = "existing_sha256_hash_999"
    # Simulamos que SQLite devuelve un registro existente: (task_id, movie_title)
    mock_get_hash.return_value = ("task-id-previo-abc", "Película Duplicada")

    fake_video_bytes = b"same video content bytes"
    files = {"files": ("duplicate_trailer.mp4", fake_video_bytes, "video/mp4")}

    response = client.post("/api/ingest", files=files)

    # Debe fallar con código 400 por duplicidad
    assert response.status_code == 400
    assert "ya fue procesado" in response.json()["detail"]
    print("✅ [TEST] Bloqueo de vídeos duplicados verificado con éxito.")

def test_get_task_status_not_found():
    """Prueba que consultar el estado de una tarea inexistente devuelve un error 404."""
    fake_task_id = "task-que-no-existe-12345"
    response = client.get(f"/api/tasks/{fake_task_id}")
    
    assert response.status_code == 404
    assert response.json()["detail"] == "Tarea no encontrada."
    print("✅ [TEST] Control de errores de tareas no encontradas verificado.")