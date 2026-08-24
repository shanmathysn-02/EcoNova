import uuid
import logging
import numpy as np
from flask import Blueprint, request, jsonify

# Placeholders for Member 1 and Member 3 functions to be mocked/patched in tests
def load_model():
    """Stand-in function to load the Keras model (mocked in tests)."""
    pass

def preprocess_image(image_bytes):
    """Stand-in function to preprocess the uploaded image bytes (mocked in tests)."""
    pass

# Mock PredictionResult structure as described by Member 1
from dataclasses import dataclass

@dataclass
class PredictionResult:
    class_index: int
    class_label: str
    confidence: float
    probabilities: dict


api_bp = Blueprint("api", __name__)
logger = logging.getLogger(__name__)


@api_bp.route("/api/v1/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    file_bytes = file.read()

    # 1. Preprocess the image (Member 3's task)
    preprocessed_image, original_image_uint8 = preprocess_image(file_bytes)

    # 2. Load model & Predict (Member 1's task)
    model = load_model()
    
    # Dummy mock prediction results (overwritten/asserted in tests)
    prediction_result = PredictionResult(
        class_index=1,
        class_label="Moderate DR",
        confidence=0.91,
        probabilities={"No DR": 0.05, "Moderate DR": 0.91, "Severe DR": 0.04}
    )

    # 3. Generate request ID (Member 4's task)
    request_id = str(uuid.uuid4())[:8]

    # === MEMBER 2 INTEGRATION START
    import logging
    from src.explainability.pipeline_adapter import run_explanation_pipeline
    from src.explainability.gradcam import InvalidLayerError, InvalidInputError, ComputationError
    try:
        from src.explainability.visualizer import VisualizerIOError
    except ImportError:
        class VisualizerIOError(IOError):
            pass

    integration_logger = logging.getLogger(__name__)

    # Run the explanation pipeline
    try:
        explanation = run_explanation_pipeline(
            preprocessed_image=preprocessed_image,      # np.ndarray shape (1, 224, 224, 3)
            original_image_uint8=original_image_uint8,  # np.ndarray shape (224, 224, 3)
            model=model,                                # Loaded Keras Model
            prediction_result=prediction_result,        # Member 1's PredictionResult
            request_id=request_id                        # 8-character UUID string
        )
    except InvalidInputError as e:
        integration_logger.warning(f"Grad-CAM explanation skipped due to invalid input: {str(e)}")
        explanation = None
    except (InvalidLayerError, ComputationError, VisualizerIOError) as e:
        integration_logger.error(f"Grad-CAM explanation failed: {str(e)}", exc_info=True)
        explanation = None
    # === MEMBER 2 INTEGRATION END

    return jsonify({
        "success": True,
        "prediction": {
            "class_label": prediction_result.class_label,
            "confidence": prediction_result.confidence
        },
        "explanation": explanation
    }), 200
