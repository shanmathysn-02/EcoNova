import os
import logging
import yaml
import numpy as np
import tensorflow as tf

from src.explainability.gradcam import (
    generate_gradcam,
    InvalidLayerError,
    InvalidInputError,
    ComputationError,
)
from src.explainability.visualizer import save_explanation
from src.explainability.advanced_cam import generate_scorecam, should_use_scorecam, ScoreCAMError
from src.explainability.report_generator import generate_clinical_report, ReportGenerationError
from src.explainability.low_bandwidth import generate_low_bandwidth_explanation
from src.explainability.longitudinal import compare_with_prior, load_prior_explanation, save_current_explanation

try:
    from src.explainability.visualizer import VisualizerIOError
except ImportError:
    class VisualizerIOError(IOError):
        pass

logger = logging.getLogger(__name__)


def run_explanation_pipeline(
    preprocessed_image: np.ndarray,
    original_image_uint8: np.ndarray,
    model: tf.keras.Model,
    prediction_result,
    request_id: str,
    explain_mode: str = "standard",
    low_bandwidth: bool = False,
    generate_report: bool = False,
    patient_id: str | None = None,
) -> dict:
    """Executes the explainability pipeline with advanced features."""
    try:
        config_path = "config.yaml"
        last_conv_layer = "top_conv"
        
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
                logger.warning(f"Could not parse '{config_path}', using default fallback: {str(parse_err)}")

        cam_result = None
        if explain_mode in ["standard", "auto"]:
            cam_result = generate_gradcam(
                image=preprocessed_image,
                model=model,
                last_conv_layer_name=last_conv_layer,
                class_index=prediction_result.class_index,
                alpha=0.4,
            )

        if explain_mode == "advanced" or (explain_mode == "auto" and should_use_scorecam(cam_result)):
            cam_result = generate_scorecam(
                image=preprocessed_image,
                model=model,
                target_size=(224, 224),
                class_index=prediction_result.class_index
            )

        filename = f"hm_{request_id}.jpg"
        heatmaps_dir = os.path.join("static", "heatmaps")
        output_path = os.path.join(heatmaps_dir, filename)

        prediction_data = {
            "class_label": prediction_result.class_label,
            "confidence": prediction_result.confidence,
            "probabilities": getattr(prediction_result, "probabilities", {})
        }
        
        try:
            save_explanation(
                original_image=original_image_uint8,
                overlay=cam_result.overlay,
                prediction=prediction_data,
                output_path=output_path,
            )
        except IOError as io_err:
            if not isinstance(io_err, VisualizerIOError):
                raise VisualizerIOError(str(io_err)) from io_err
            raise io_err

        explanation_dict = {
            "heatmap_url": f"/static/heatmaps/{filename}",
            "highlighted_regions": [
                {"x": r.x, "y": r.y, "w": r.w, "h": r.h, "confidence": round(r.confidence, 4)}
                for r in cam_result.highlighted_regions
            ],
            "computation_time_ms": round(cam_result.computation_time_ms, 2),
        }

        if low_bandwidth:
            explanation_dict["low_bandwidth"] = generate_low_bandwidth_explanation(
                overlay=cam_result.overlay,
                heatmap=cam_result.heatmap,
                output_dir=heatmaps_dir,
                request_id=request_id
            )

        if generate_report:
            report_path = os.path.join("static", "reports", f"report_{request_id}.jpg")
            explanation_dict["report_url"] = f"/static/reports/report_{request_id}.jpg"
            generate_clinical_report(
                original_image=original_image_uint8,
                heatmap=cam_result.heatmap,
                overlay=cam_result.overlay,
                prediction_dict=prediction_data,
                highlighted_regions=cam_result.highlighted_regions,
                output_path=report_path
            )

        if patient_id:
            prior_regions = load_prior_explanation(patient_id)
            if prior_regions is not None:
                explanation_dict["longitudinal_comparison"] = compare_with_prior(cam_result.highlighted_regions, prior_regions)
            save_current_explanation(patient_id, cam_result.highlighted_regions, request_id)

        return explanation_dict

    except (InvalidLayerError, InvalidInputError, ComputationError, VisualizerIOError, ScoreCAMError, ReportGenerationError) as e:
        logger.error(f"Explanation Pipeline failure: {str(e)}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Unexpected Pipeline failure: {str(e)}", exc_info=True)
        raise e
