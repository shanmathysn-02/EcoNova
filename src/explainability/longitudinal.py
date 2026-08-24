import os
import json
import datetime
from src.explainability.gradcam import Region


def compute_region_iou(region_a: Region, region_b: Region) -> float:
    """Computes the Intersection over Union (IoU) of two Regions."""
    x_left = max(region_a.x, region_b.x)
    y_top = max(region_a.y, region_b.y)
    x_right = min(region_a.x + region_a.w, region_b.x + region_b.w)
    y_bottom = min(region_a.y + region_a.h, region_b.y + region_b.h)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = region_a.w * region_a.h
    area_b = region_b.w * region_b.h
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def compare_with_prior(current_regions: list[Region], prior_regions: list[Region], iou_threshold: float = 0.3) -> dict:
    """Compares current highlighted regions with prior regions to track changes."""
    matches = []
    matched_prior_indices = set()
    
    for curr in current_regions:
        best_iou = 0.0
        best_prior_idx = -1
        
        for idx, prior in enumerate(prior_regions):
            if idx in matched_prior_indices:
                continue
            iou = compute_region_iou(curr, prior)
            if iou > best_iou:
                best_iou = iou
                best_prior_idx = idx
                
        if best_iou >= iou_threshold:
            prior_match = prior_regions[best_prior_idx]
            matches.append({
                "current": {"x": curr.x, "y": curr.y, "confidence": curr.confidence},
                "prior": {"x": prior_match.x, "y": prior_match.y, "confidence": prior_match.confidence},
                "iou": round(best_iou, 4),
                "confidence_delta": round(curr.confidence - prior_match.confidence, 4)
            })
            matched_prior_indices.add(best_prior_idx)
            
    stable_count = len(matches)
    new_count = len(current_regions) - stable_count
    resolved_count = len(prior_regions) - stable_count
    max_len = max(len(current_regions), len(prior_regions))
    
    return {
        "stable_regions": stable_count,
        "new_regions": new_count,
        "resolved_regions": resolved_count,
        "region_matches": matches,
        "stability_score": round(stable_count / max_len, 4) if max_len > 0 else 1.0
    }


def load_prior_explanation(patient_id: str, reports_dir: str = os.path.join("data", "patient_history")) -> list[Region] | None:
    """Loads the last explanation regions for a patient."""
    file_path = os.path.join(reports_dir, patient_id, "last_regions.json")
    if not os.path.exists(file_path):
        return None
        
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return [Region(**r) for r in data]
    except Exception:
        return None


def save_current_explanation(patient_id: str, regions: list[Region], request_id: str, reports_dir: str = os.path.join("data", "patient_history")) -> str:
    """Saves the current explanation regions for a patient to track longitudinally."""
    patient_dir = os.path.join(reports_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)
    
    regions_file = os.path.join(patient_dir, "last_regions.json")
    history_file = os.path.join(patient_dir, "history_log.jsonl")
    
    with open(regions_file, "w") as f:
        json.dump([vars(r) for r in regions], f)
        
    with open(history_file, "a") as f:
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "request_id": request_id,
            "region_count": len(regions)
        }
        f.write(json.dumps(entry) + "\n")
        
    return regions_file
