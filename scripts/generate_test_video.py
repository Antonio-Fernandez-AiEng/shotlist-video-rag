import cv2
import numpy as np
from pathlib import Path

def create_custom_synthetic_video():
    # Obtenemos la ruta absoluta de la raíz del proyecto (subiendo un nivel desde 'scripts/')
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "videos_ingested"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "custom_test_clip.mp4"
    
    width, height = 640, 360
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print("🎬 [DEMO] Generando vídeo sintético personalizado...")
    
    # 1. Inicio en Negro (1 segundo)
    for _ in range(fps * 1):
        out.write(np.zeros((height, width, 3), dtype=np.uint8))
        
    # 2. 1 segundo Azul (BGR: Azul puro)
    for _ in range(fps * 1):
        out.write(np.full((height, width, 3), (255, 0, 0), dtype=np.uint8))
        
    # 3. 1 segundo Rojo (BGR: Rojo puro)
    for _ in range(fps * 1):
        out.write(np.full((height, width, 3), (0, 0, 255), dtype=np.uint8))
        
    # 4. Transición en Negro (1 segundo)
    for _ in range(fps * 1):
        out.write(np.zeros((height, width, 3), dtype=np.uint8))
        
    # 5. 7 segundos de Verde (Plano largo -> Activa submuestreo adaptativo)
    for _ in range(fps * 7):
        out.write(np.full((height, width, 3), (0, 255, 0), dtype=np.uint8))
        
    # 6. 1 segundo de Rojo final
    for _ in range(fps * 1):
        out.write(np.full((height, width, 3), (0, 0, 255), dtype=np.uint8))
        
    out.release()
    print(f"✅ [DEMO] Vídeo creado con éxito en: {output_path}")

if __name__ == "__main__":
    create_custom_synthetic_video()