import os
from PIL import Image, ImageDraw, ImageFont


def save_explanation(
    original_image,
    overlay,
    prediction,
    output_path
) -> str:
    """Saves the visual explanation as a JPEG image, annotated with predictions.

    Args:
        original_image: NumPy array (H, W, 3) uint8 of the original image.
        overlay: NumPy array (H, W, 3) uint8 containing the Grad-CAM heatmap overlay.
        prediction: Dict with keys 'class_label' (str) and 'confidence' (float).
        output_path: Full file path string where the image should be saved.

    Returns:
        The absolute path to the saved JPEG image.

    Raises:
        IOError: If directory creation or image saving fails.
    """
    # 1. Create directory if not exists
    dir_name = os.path.dirname(output_path)
    if dir_name:
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception as e:
            raise IOError(
                f"Failed to create directory '{dir_name}': {str(e)}"
            ) from e

    # 2. Convert overlay to PIL Image
    try:
        overlay_pil = Image.fromarray(overlay)
        if overlay_pil.mode != "RGB":
            overlay_pil = overlay_pil.convert("RGB")
    except Exception as e:
        raise IOError(
            f"Failed to convert overlay array to PIL Image: {str(e)}"
        ) from e

    # 3. Annotate top-left with text: "Moderate DR | Confidence: 91%"
    try:
        draw = ImageDraw.Draw(overlay_pil)
        
        # Try to load a clean Arial font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", size=14)
        except Exception:
            font = ImageFont.load_default()

        # Construct the label
        class_label = prediction.get("class_label", "Unknown")
        conf_val = prediction.get("confidence", 0.0)
        
        # Check if confidence is in range [0, 1] or pre-scaled to [0, 100]
        if conf_val <= 1.0:
            conf_percentage = int(round(conf_val * 100))
        else:
            conf_percentage = int(round(conf_val))
            
        text = f"{class_label} | Confidence: {conf_percentage}%"
        
        # Draw background box for text readability
        try:
            bbox = draw.textbbox((10, 10), text, font=font)
            padded_bbox = (bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4)
            draw.rectangle(padded_bbox, fill=(30, 30, 30))
        except AttributeError:
            pass  # Fallback if PIL version is old and doesn't support textbbox
            
        # Draw the text in white
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)
        
    except Exception as e:
        raise IOError(
            f"Failed to annotate overlay with text: {str(e)}"
        ) from e

    # 4. Save as JPEG at output_path
    try:
        overlay_pil.save(output_path, format="JPEG", quality=95)
    except Exception as e:
        raise IOError(
            f"Failed to save explanation image to '{output_path}': {str(e)}"
        ) from e

    # 5. Return absolute path
    return os.path.abspath(output_path)
