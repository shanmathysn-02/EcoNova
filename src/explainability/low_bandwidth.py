import os
import numpy as np
from PIL import Image


def generate_low_bandwidth_explanation(overlay: np.ndarray, heatmap: np.ndarray, output_dir: str, request_id: str) -> dict:
    """Generates smaller versions of explanations for low-connectivity environments."""
    os.makedirs(output_dir, exist_ok=True)
    
    thumb_path = os.path.join(output_dir, f"thumb_hm_{request_id}.jpg")
    lite_path = os.path.join(output_dir, f"lite_hm_{request_id}.jpg")
    full_path = os.path.join(output_dir, f"hm_{request_id}.jpg")
    
    overlay_pil = Image.fromarray(overlay)
    
    # 1. Thumbnail
    thumb_pil = overlay_pil.resize((150, 150), Image.Resampling.LANCZOS)
    thumb_pil.save(thumb_path, format="JPEG", quality=70)
    
    # 2. Lite
    overlay_pil.save(lite_path, format="JPEG", quality=60)
    
    return {
        "thumbnail_url": f"/static/heatmaps/thumb_hm_{request_id}.jpg",
        "lite_url": f"/static/heatmaps/lite_hm_{request_id}.jpg",
        "full_url": f"/static/heatmaps/hm_{request_id}.jpg",
        "size_bytes": {
            "thumbnail": os.path.getsize(thumb_path),
            "lite": os.path.getsize(lite_path),
            "full": os.path.getsize(full_path) if os.path.exists(full_path) else 0
        }
    }
