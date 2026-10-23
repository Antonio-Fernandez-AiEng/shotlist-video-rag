from pathlib import Path
from services.video_processor import VideoProcessor

def test_video_processing():
    print("🚀 [TEST REAL] Inicializando VideoProcessor...")
    processor = VideoProcessor()
    
    # Rutas absolutas basadas en la raíz del proyecto
    project_root = Path(__file__).resolve().parent
    video_path = project_root / "data" / "videos_ingested" / "custom_test_clip.mp4"
    output_dir = project_root / "data" / "extracted_shots"
    
    if not video_path.exists():
        print(f"❌ No se encontró el vídeo en: {video_path}")
        return

    print(f"🎬 [TEST REAL] Procesando vídeo de prueba: {video_path.name}")
    shots = processor.extract_and_vectorize(
        video_path=video_path,
        output_dir=output_dir,
        movie_title="SyntheticTestMovie"
    )
    
    print("\n" + "="*50)
    print(f"📊 RESULTADOS DEL PROCESAMIENTO:")
    print(f"Total de fotogramas válidos extraídos: {len(shots)}")
    print("="*50)
    
    for shot in shots:
        print(f"• ID: {shot['id']}")
        print(f"  Timestamp: {shot['metadata']['timestamp']}s")
        print(f"  Escena: {shot['metadata'].get('scene_index')}")
        print(f"  Fichero: {shot['metadata']['file_path']}")
        print(f"  Dimensión del Vector: {len(shot['vector'])}\n")

if __name__ == "__main__":
    test_video_processing()