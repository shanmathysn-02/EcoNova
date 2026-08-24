import os
import shutil
import pytest
import numpy as np
import tensorflow as tf
from PIL import Image
from unittest.mock import patch

from src.explainability.gradcam import GradCAMResult, Region
from src.explainability.advanced_cam import generate_scorecam, should_use_scorecam
from src.explainability.report_generator import generate_clinical_report
from src.explainability.low_bandwidth import generate_low_bandwidth_explanation
from src.explainability.longitudinal import compute_region_iou, compare_with_prior, save_current_explanation, load_prior_explanation
from src.explainability.pipeline_adapter import run_explanation_pipeline


@pytest.fixture
def mock_model():
    return tf.keras.applications.EfficientNetB0(weights='imagenet')

def test_scorecam_runs(mock_model):
    dummy_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
    res = generate_scorecam(dummy_img, mock_model, class_index=0, num_samples=2)
    assert isinstance(res, GradCAMResult)
    assert res.overlay.shape == (224, 224, 3)
    assert res.computation_time_ms > 0

def test_should_use_scorecam_fallback():
    empty_res = GradCAMResult(heatmap=np.zeros((2,2)), overlay=np.zeros((2,2,3)), highlighted_regions=[], computation_time_ms=10)
    assert should_use_scorecam(empty_res) is True
    
    low_conf_res = GradCAMResult(heatmap=np.zeros((2,2)), overlay=np.zeros((2,2,3)), highlighted_regions=[Region(0,0,10,10,0.4)], computation_time_ms=10)
    assert should_use_scorecam(low_conf_res, confidence_threshold=0.5) is True

def test_clinical_report_generated():
    orig = np.zeros((224, 224, 3), dtype=np.uint8)
    hm = np.zeros((224, 224), dtype=np.float32)
    overlay = np.zeros((224, 224, 3), dtype=np.uint8)
    preds = {"class_label": "Test", "confidence": 0.9, "probabilities": {"Test": 0.9}}
    regs = [Region(0, 0, 10, 10, 0.9)]
    
    out = "test_reports/report_1.jpg"
    res = generate_clinical_report(orig, hm, overlay, preds, regs, out)
    assert os.path.exists(res)
    
    with Image.open(res) as img:
        assert img.width == 1200
    
    shutil.rmtree("test_reports", ignore_errors=True)

def test_low_bandwidth_assets():
    hm = np.zeros((224, 224), dtype=np.float32)
    overlay = np.zeros((224, 224, 3), dtype=np.uint8)
    
    res = generate_low_bandwidth_explanation(overlay, hm, "test_static", "123")
    assert os.path.exists("test_static/thumb_hm_123.jpg")
    assert os.path.exists("test_static/lite_hm_123.jpg")
    assert "thumbnail_url" in res
    assert "lite_url" in res
    
    shutil.rmtree("test_static", ignore_errors=True)

def test_longitudinal_iou():
    r1 = Region(0, 0, 10, 10, 0.9)
    r2 = Region(5, 5, 10, 10, 0.8)
    iou = compute_region_iou(r1, r2)
    assert iou > 0.0
    
    res = compare_with_prior([r1], [r2])
    assert res["stable_regions"] == 1

@patch('src.explainability.pipeline_adapter.should_use_scorecam')
@patch('src.explainability.pipeline_adapter.generate_scorecam')
@patch('src.explainability.pipeline_adapter.generate_gradcam')
def test_pipeline_adapter_auto_mode(mock_grad, mock_score, mock_should):
    mock_should.return_value = True
    dummy_res = GradCAMResult(heatmap=np.zeros((224,224)), overlay=np.zeros((224,224,3)), highlighted_regions=[], computation_time_ms=10)
    mock_grad.return_value = dummy_res
    mock_score.return_value = dummy_res
    
    from collections import namedtuple
    Pred = namedtuple('Pred', ['class_index', 'class_label', 'confidence', 'probabilities'])
    
    res = run_explanation_pipeline(
        np.zeros((1,224,224,3)), np.zeros((224,224,3)), None, Pred(0, "A", 0.9, {}), "123", explain_mode="auto"
    )
    assert mock_score.called
    assert "heatmap_url" in res

def test_patient_history_persistence():
    regs = [Region(0, 0, 10, 10, 0.9)]
    save_current_explanation("P1", regs, "123", "test_history")
    loaded = load_prior_explanation("P1", "test_history")
    assert len(loaded) == 1
    assert loaded[0].confidence == 0.9
    
    shutil.rmtree("test_history", ignore_errors=True)
