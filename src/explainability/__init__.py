"""Explainability module for Diabetic Retinopathy screening system.

This module provides tools for explaining model decisions using Grad-CAM, generating
visual overlays showing regions of interest, and adapting the pipeline for API endpoints.
"""

from .gradcam import (
    Region,
    GradCAMResult,
    generate_gradcam,
    InvalidLayerError,
    InvalidInputError,
    ComputationError,
)
from .visualizer import save_explanation
from .pipeline_adapter import run_explanation_pipeline

try:
    from .visualizer import VisualizerIOError
except ImportError:
    class VisualizerIOError(IOError):
        pass

__all__ = [
    "Region",
    "GradCAMResult",
    "generate_gradcam",
    "save_explanation",
    "run_explanation_pipeline",
    "InvalidLayerError",
    "InvalidInputError",
    "ComputationError",
    "VisualizerIOError",
]
