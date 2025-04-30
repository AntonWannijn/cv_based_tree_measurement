import cv2
import os
import logging
from typing import Optional
import subprocess

def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 30,
    max_frames: Optional[int] = None,
    prefix: str = "frame",
    ffmpeg_attempts: int = 16384,  # Increased further
    use_ffmpeg_direct: bool = False  # Alternative method
) -> int:
    """
    Robust frame extraction with multiple fallback methods.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(ffmpeg_attempts)
    
    extracted_count = 0
    
    if use_ffmpeg_direct:
        # Method 1: Use FFmpeg directly via subprocess (most reliable)
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f'select=not(mod(n\,{frame_interval}))',
                '-vsync', 'vfr',
                f'{output_dir}/{prefix}_%04d.jpg'
            ]
            subprocess.run(cmd, check=True)
            extracted_count = len(os.listdir(output_dir))
            logging.info(f"FFmpeg direct extracted {extracted_count} frames")
            return extracted_count
        except Exception as e:
            logging.warning(f"FFmpeg direct failed: {e}. Falling back to OpenCV")

    # Method 2: Try OpenCV with multiple backends
    backends = [
        cv2.CAP_FFMPEG,
        cv2.CAP_MSMF,  # Windows Media Foundation
        cv2.CAP_DSHOW  # DirectShow
    ]
    
    for backend in backends:
        cap = cv2.VideoCapture(video_path, backend)
        if not cap.isOpened():
            continue
            
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    frame_filename = f"{output_dir}/{prefix}_{extracted_count:04d}.jpg"
                    cv2.imwrite(frame_filename, frame)
                    extracted_count += 1
                    logging.info(f"Saved {frame_filename}")

                    if max_frames is not None and extracted_count >= max_frames:
                        break

                frame_idx += 1
            break
        finally:
            cap.release()

    if extracted_count == 0:
        raise RuntimeError(f"All methods failed for {video_path}")
    
    logging.info(f"Extracted {extracted_count} frames from {video_path}")
    return extracted_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage with fallback logic
    video_path = "dataset/videos/eastbound_20240319.MP4"
    output_dir = "dataset/frames/eastbound"
    
    try:
        extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            frame_interval=30,
            use_ffmpeg_direct=True  # Try FFmpeg first
        )
    except Exception as e:
        logging.error(f"Primary method failed: {e}. Retrying with OpenCV...")
        extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            frame_interval=30,
            use_ffmpeg_direct=False
        )