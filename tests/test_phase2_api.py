import os
import time
import tempfile
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image
from unittest.mock import patch

from src.api.app import create_app
from src.api.routes import PredictionResult


@pytest.fixture
def client():
    """Flask test client fixture."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def temp_image_file():
    """Creates a temporary dummy image (224x224x3, uint8) and returns its path."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name, "JPEG")
        tmp_path = tmp.name
        
    yield tmp_path
    
    # Cleanup after test runs
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@pytest.fixture
def mock_model():
    """Loads and caches the EfficientNetB0 stand-in model."""
    # Using 'imagenet' weights as requested
    return tf.keras.applications.EfficientNetB0(weights='imagenet')


def test_predict_success(client, temp_image_file, mock_model):
    """Verifies that a successful prediction request generates a Grad-CAM explanation."""
    # 1. Setup mock returns for model loader and preprocessor
    dummy_preprocessed = np.random.rand(1, 224, 224, 3).astype(np.float32)
    dummy_original_uint8 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Measure total request latency
    start_time = time.perf_counter()
    
    with patch("src.api.routes.load_model", return_value=mock_model), \
         patch("src.api.routes.preprocess_image", return_value=(dummy_preprocessed, dummy_original_uint8)):
         
        with open(temp_image_file, "rb") as img_file:
            data = {"file": (img_file, "test.jpg")}
            response = client.post("/api/v1/predict", data=data, content_type="multipart/form-data")
            
    total_latency_ms = (time.perf_counter() - start_time) * 1000

    # 2. Assert response status and root keys
    assert response.status_code == 200
    res_json = response.json
    assert res_json["success"] is True
    assert res_json["explanation"] is not None

    # 3. Assert explanation object structure
    explanation = res_json["explanation"]
    assert explanation["heatmap_url"].startswith("/static/heatmaps/hm_")
    assert isinstance(explanation["highlighted_regions"], list)
    assert explanation["computation_time_ms"] > 0

    # 4. Assert physical existence of the saved heatmap image relative to app root
    # Since we set root_path to project root, the physical location matches app.root_path + heatmap_url
    app = create_app()
    heatmap_rel_path = explanation["heatmap_url"].lstrip("/")
    physical_path = os.path.join(app.root_path, heatmap_rel_path)
    assert os.path.exists(physical_path), f"Heatmap file not found at: {physical_path}"

    # Cleanup generated heatmap file
    if os.path.exists(physical_path):
        try:
            os.remove(physical_path)
        except Exception:
            pass

    # 5. Print a visual summary
    print("\n" + "=" * 50)
    print("PHASE 2 SUCCESS TEST SUMMARY")
    print("=" * 50)
    print(f"Prediction Class: {res_json['prediction']['class_label']}")
    print(f"Confidence:       {res_json['prediction']['confidence']:.2f}")
    print(f"Num Regions:      {len(explanation['highlighted_regions'])}")
    print(f"Grad-CAM Time:    {explanation['computation_time_ms']:.2f} ms")
    print(f"Total Latency:    {total_latency_ms:.2f} ms")
    print("=" * 50)


def test_predict_graceful_degradation(client, temp_image_file, mock_model):
    """Verifies that the API handles failures inside run_explanation_pipeline gracefully."""
    from src.explainability.gradcam import InvalidLayerError
    
    dummy_preprocessed = np.random.rand(1, 224, 224, 3).astype(np.float32)
    dummy_original_uint8 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Patch run_explanation_pipeline to raise an InvalidLayerError
    with patch("src.api.routes.load_model", return_value=mock_model), \
         patch("src.api.routes.preprocess_image", return_value=(dummy_preprocessed, dummy_original_uint8)), \
         patch("src.api.routes.run_explanation_pipeline", side_effect=InvalidLayerError("Target layer not found")):
         
        with open(temp_image_file, "rb") as img_file:
            data = {"file": (img_file, "test.jpg")}
            response = client.post("/api/v1/predict", data=data, content_type="multipart/form-data")

    # Assert response is still 200 OK but the explanation field is None
    assert response.status_code == 200
    res_json = response.json
    assert res_json["success"] is True
    assert res_json["explanation"] is None
    print("\n" + "=" * 50)
    print("PHASE 2 GRACEFUL DEGRADATION TEST SUMMARY: PASSED")
    print("=" * 50)
