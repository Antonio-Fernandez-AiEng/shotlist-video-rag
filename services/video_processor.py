import re
from pathlib import Path
from scenedetect import detect, ContentDetector
from PIL import Image
import cv2
import numpy as np
import torch
import open_clip
from config import MODEL_NAME, PRETRAINED_SOURCE


class VideoProcessor:
  def __init__(
      self,
      model_name: str = MODEL_NAME,
      pretrained: str = PRETRAINED_SOURCE
  ):
    # Force cpu to avoid CUDA issues
    self.device = "cpu"
    print(
        f"[INFO] Loading {model_name} ({pretrained}) on"
        f" {self.device.upper()}..."
    )

    self.model, _, self.preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=self.device
    )
    self.tokenizer = open_clip.get_tokenizer(model_name)
    self.model.eval()
    print("[INFO] SigLIP model loaded successfully.")

  @staticmethod
  def sanitize_filename(name: str) -> str:
    """Deletes special characters to avoid errors in file paths
    """
    clean_name = re.sub(r'[^\w\s-]', '', name)
    return re.sub(r'[-\s]+', '_', clean_name).strip().lower()

  @staticmethod
  def is_black_or_empty_frame(
      image_rgb: np.ndarray, brightness_threshold: float = 12.0
  ) -> bool:
    """Analyzes a frame searching for black screens using average luminance
    """
    try:
      gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
      mean_brightness = float(np.mean(gray))
      return mean_brightness < brightness_threshold
    except Exception as e:
      print(f"⚠️ Error analyzing frame brightness: {e}")
      return False

  def extract_and_vectorize(
      self, video_path: Path, output_dir: Path, movie_title: str = None
  ):
    """Process a video, detects scenes, saves to disk and encodes the frames with SigLip, the 
    stores the vectorss in ChromaDB
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Obtain title of filename
    raw_title = (
        movie_title.strip()
        if movie_title and movie_title.strip()
        else video_path.stem
    )

    # 3. Sanitize title to make sure is safe for paths  
    final_movie_title = self.sanitize_filename(raw_title)

    print(
        f"[INFO] Analyzing shots in: {video_path.name} ('{final_movie_title}')"
    )
    scene_list = detect(str(video_path), ContentDetector(threshold=30))
    print(f"[INFO] Shots found: {len(scene_list)}")

    cap = cv2.VideoCapture(str(video_path))
    processed_shots = []
    global_shot_counter = 0

    for i, scene in enumerate(scene_list):
      start_time, end_time = scene

      start_sec = start_time.get_seconds()
      end_sec = end_time.get_seconds()
      shot_duration = end_sec - start_sec

      # --- Subsampling based on scene duration ---
      timestamps_to_extract = []

      if shot_duration <= 6.0:
        middle_sec = start_sec + (shot_duration / 2)
        timestamps_to_extract.append(middle_sec)
      else:
        current_sec = start_sec + 2.0
        while current_sec < (end_sec - 1.0):
          timestamps_to_extract.append(current_sec)
          current_sec += 4.0

        if not timestamps_to_extract:
          timestamps_to_extract.append(start_sec + (shot_duration / 2))

      # --- Processing extracted timestamps---
      for sub_idx, sec in enumerate(timestamps_to_extract):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
        ret, frame = cap.read()

        if not ret:
          continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Filtering black screens
        if self.is_black_or_empty_frame(
            image_rgb, brightness_threshold=12.0
        ):
          continue

        pil_image = Image.fromarray(image_rgb)
        timestamp_sec = round(sec, 2)

        filename = (
            f"{final_movie_title}_shot_{i:03d}_sub_{sub_idx}_ts_{timestamp_sec}s.jpg"
        )
        filepath = output_dir / filename
        pil_image.save(filepath, quality=90)

        # Preprocessing and generating vectors with SigLip
        processed_image = (
            self.preprocess(pil_image).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
          image_features = self.model.encode_image(processed_image)
          image_features /= image_features.norm(dim=-1, keepdim=True)
          vector = image_features.numpy()[0].tolist()

        shot_id = f"{final_movie_title}_shot_{global_shot_counter:04d}"

        # Clean and structured 
        metadata = {
            "movie_title": final_movie_title,
            "scene_index": i,
            "sub_index": sub_idx,
            "shot_duration": round(shot_duration, 2),
            "timestamp": timestamp_sec,
            "file_path": str(filepath.as_posix()),
        }

        processed_shots.append(
            {"id": shot_id, "vector": vector, "metadata": metadata}
        )

        global_shot_counter += 1

    cap.release()
    print(
        f"[INFO] Processing completed. Valid frames"
        f" vectorized: {len(processed_shots)}"
    )
    return processed_shots

  def encode_text_query(self, text: str) -> list[float]:
    """Converts a query into a normalized SigLip vector"""
    with torch.no_grad():
      text_tokens = self.tokenizer([text]).to(self.device)
      text_features = self.model.encode_text(text_tokens)
      text_features /= text_features.norm(dim=-1, keepdim=True)
      return text_features.numpy()[0].tolist()