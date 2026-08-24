import os
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.explainability.gradcam import Region
import cv2


class ReportGenerationError(Exception):
    pass


def generate_clinical_report(original_image: np.ndarray, heatmap: np.ndarray, overlay: np.ndarray, prediction_dict: dict, highlighted_regions: list[Region], output_path: str) -> str:
    """Generates a 1200x800 clinical-grade combined report image for downloading."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        canvas = Image.new("RGB", (1200, 800), "white")
        draw = ImageDraw.Draw(canvas)
        
        try:
            font_title = ImageFont.truetype("arialbd.ttf", 24)
            font_text = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font_title = ImageFont.load_default()
            font_text = ImageFont.load_default()
            
        # Top banner
        draw.rectangle([0, 0, 1200, 60], fill="black")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw.text((20, 15), f"Diabetic Retinopathy Screening Report | {timestamp}", fill="white", font=font_title)
        
        # Format panels
        orig_pil = Image.fromarray(original_image).resize((400, 400), Image.Resampling.LANCZOS)
        
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heat_pil = Image.fromarray(heatmap_colored).resize((400, 400), Image.Resampling.LANCZOS)
        
        overlay_pil = Image.fromarray(overlay).resize((400, 400), Image.Resampling.LANCZOS)
        
        # Paste panels
        canvas.paste(orig_pil, (0, 60))
        canvas.paste(heat_pil, (400, 60))
        canvas.paste(overlay_pil, (800, 60))
        
        # Labels for panels
        draw.text((10, 470), "Original Fundus Image", fill="black", font=font_title)
        draw.text((410, 470), "Activation Heatmap", fill="black", font=font_title)
        draw.text((810, 470), "AI Overlay", fill="black", font=font_title)
        
        # Diagnosis
        conf = prediction_dict.get('confidence', 0.0)
        draw.text((20, 520), f"Diagnosis: {prediction_dict.get('class_label', 'Unknown')} | Confidence: {conf:.1%}", fill="black", font=font_title)
        
        probs = prediction_dict.get('probabilities', {})
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        y_offset = 560
        for label, prob in sorted_probs:
            draw.text((20, y_offset), f"- {label}: {prob:.1%}", fill="black", font=font_text)
            y_offset += 25
            
        # Highlighted Regions
        draw.text((600, 520), "Top Highlighted Regions:", fill="black", font=font_title)
        y_offset = 560
        for i, r in enumerate(highlighted_regions[:3]):
            draw.text((600, y_offset), f"Region {i+1}: x={r.x}, y={r.y}, w={r.w}, h={r.h}, activation={r.confidence:.2f}", fill="black", font=font_text)
            y_offset += 25
            
        # Footer
        draw.text((20, 750), "This AI-assisted screening tool does not replace professional clinical evaluation.", fill="red", font=font_text)
        
        canvas.save(output_path, format="JPEG", quality=95)
        return os.path.abspath(output_path)
    except Exception as e:
        raise ReportGenerationError(f"Failed to generate clinical report: {str(e)}") from e
