import os
import logging
import yaml
import numpy as np
import tensorflow as tf

from .gradcam import (
    generate_gradcam,
    InvalidLayerError,
    InvalidInputError,
    ComputationError,
)
from .visualizer import save_explanation

try:
    from .visualizer import VisualizerIOError
except ImportError:
    # Fallback exception definition if not defined in visualizer.py
    class VisualizerIOError(IOError):
        pass

logger = logging.getLogger(__name__)


def run_explanation_pipeline(
    preprocessed_image: np.ndarray,
    original_image_uint8: np.ndarray,
    model: tf.keras.Model,
    prediction_result,
    request_id: str,
) -> dict | None:
    """Executes the explainability pipeline (Grad-CAM and visual overlay generation).

    Args:
        preprocessed_image: Normalised image array (1, 224, 224, 3) float32.
        original_image_uint8: Original image array (224, 224, 3) uint8 RGB.
        model: Pre-trained TensorFlow/Keras model.
        prediction_result: Result object containing predictions (class_index, class_label, confidence).
        request_id: Unique 8-character request identifier.

    Returns:
        A dictionary containing heatmap_url, highlighted_regions, and computation_time_ms.

    Raises:
        InvalidLayerError: If the target conv layer is not found.
        InvalidInputError: If input arguments or shapes are invalid.
        ComputationError: If Grad-CAM calculation fails.
        VisualizerIOError: If explanation overlay save operation fails.
    """
    try:
        # 1. Load config from "config.yaml"
        config_path = "config.yaml"
        last_conv_layer = "top_conv"  # Default fallback
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                if config and isinstance(config, dict):
                    if "model.last_conv_layer" in config:
                        last_conv_layer = config["model.last_conv_layer"]
                    elif "model" in config and isinstance(config["model"], dict):
                        last_conv_layer = config["model"].get("last_conv_layer", "top_conv")
            except Exception as parse_err:
                logger.warning(
                    f"Could not parse '{config_path}', using default fallback: {str(parse_err)}"
                )

        # 2. Call generate_gradcam
        gradcam_result = generate_gradcam(
            image=preprocessed_image,
            model=model,
            last_conv_layer_name=last_conv_layer,
            class_index=prediction_result.class_index,
            alpha=0.4,
        )

        # 3. Construct filename
        filename = f"hm_{request_id}.jpg"

        # 4. Construct output_path using os.path.join for cross-platform compatibility
        output_path = os.path.join("static", "heatmaps", filename)

        # 5. Call save_explanation
        prediction_data = {
            "class_label": prediction_result.class_label,
            "confidence": prediction_result.confidence,
        }
        
        try:
            save_explanation(
                original_image=original_image_uint8,
                overlay=gradcam_result.overlay,
                prediction=prediction_data,
                output_path=output_path,
            )
        except IOError as io_err:
            if not isinstance(io_err, VisualizerIOError):
                raise VisualizerIOError(str(io_err)) from io_err
            raise io_err

        # 6. Return dict exactly as requested
        return {
            "heatmap_url": f"/static/heatmaps/{filename}",
            "highlighted_regions": [
                {
                    "x": r.x,
                    "y": r.y,
                    "w": r.w,
                    "h": r.h,
                    "confidence": round(r.confidence, 4),
                }
                for r in gradcam_result.highlighted_regions
            ],
            "computation_time_ms": round(gradcam_result.computation_time_ms, 2),
        }

    except (InvalidLayerError, InvalidInputError, ComputationError, VisualizerIOError) as e:
        logger.error(f"Grad-CAM Explanation Pipeline failure: {str(e)}", exc_info=True)
        raise e
