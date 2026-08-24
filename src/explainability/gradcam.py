import time
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


# Custom Exceptions
class InvalidLayerError(Exception):
    """Raised when the specified target layer cannot be found in the model."""
    pass


class InvalidInputError(Exception):
    """Raised when the input image or parameters do not meet requirements."""
    pass


class ComputationError(Exception):
    """Raised when an error occurs during the Grad-CAM mathematical computation."""
    pass


@dataclass
class Region:
    """Represents a bounding box of a highlighted region of interest."""
    x: int
    y: int
    w: int
    h: int
    confidence: float


@dataclass
class GradCAMResult:
    """Holds the complete result of the Grad-CAM generation process."""
    heatmap: np.ndarray             # Normalized heatmap: (H, W), float32 [0, 1]
    overlay: np.ndarray             # Blended image: (H, W, 3), uint8
    highlighted_regions: List[Region] # Sorted bounding boxes of highlighted areas
    computation_time_ms: float      # Computation time in milliseconds


def generate_gradcam(
    image: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str,
    class_index: int,
    alpha: float = 0.4,
) -> GradCAMResult:
    """Generates a Grad-CAM localization map and overlay for a given class index.

    Args:
        image: Input preprocessed image array of shape (1, H, W, 3), dtype float32 [0, 1].
        model: A loaded Keras model.
        last_conv_layer_name: Name of the convolutional layer to compute gradients w.r.t.
        class_index: Index of the class (0-4) to visualize.
        alpha: Blending transparency factor for overlay.

    Returns:
        A GradCAMResult instance containing heatmap, overlay, regions, and timing.

    Raises:
        InvalidInputError: If the input shape or arguments are invalid.
        InvalidLayerError: If the specified convolutional layer is not found.
        ComputationError: If gradients or computation steps fail.
    """
    # 1. Validate image shape
    if not isinstance(image, np.ndarray):
        raise InvalidInputError("Input image must be a numpy ndarray.")
    
    if len(image.shape) != 4 or image.shape[0] != 1:
        raise InvalidInputError(
            f"Input image must have batch size 1 and 4 dimensions. Got shape: {image.shape}"
        )

    # 2. Find last_conv_layer in the model
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError as e:
        raise InvalidLayerError(
            f"Layer '{last_conv_layer_name}' not found in the model."
        ) from e

    # 3. Build the gradient model
    try:
        # Create a model that outputs target conv layer activations and predictions
        grad_model = Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
    except Exception as e:
        raise ComputationError(f"Failed to build the gradient model: {str(e)}") from e

    # Start timing the operation
    start_time = time.perf_counter()

    try:
        # 4. Use tf.GradientTape to compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            
            # Ensure class_index is within prediction boundaries
            if class_index < 0 or class_index >= predictions.shape[-1]:
                raise InvalidInputError(
                    f"Class index {class_index} is out of bounds for predictions "
                    f"of shape {predictions.shape}."
                )
            
            loss = predictions[:, class_index]

        # Gradients of predicted class score w.r.t. conv outputs
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise ComputationError(
                "Gradients w.r.t. convolutional layer outputs are None. "
                "Ensure that target layer outputs are differentiable w.r.t. model outputs."
            )

        # 5. Global average pool gradients to get weights
        # grads shape: (1, H_conv, W_conv, C_conv) -> weights shape: (C_conv,)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))

        # 6. Compute weighted sum of conv feature maps -> raw heatmap
        # conv_outputs[0] shape: (H_conv, W_conv, C_conv)
        heatmap = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)

        # 7. Apply ReLU
        heatmap = heatmap.numpy()
        heatmap = np.maximum(heatmap, 0)

        # 8. Normalize heatmap to [0, 1]
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        else:
            heatmap = np.zeros_like(heatmap)

        # 9. Resize heatmap to (224, 224) using OpenCV
        heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Re-normalize resized heatmap to ensure precision of bounds [0, 1]
        max_val_resized = np.max(heatmap)
        if max_val_resized > 0:
            heatmap = heatmap / max_val_resized

        # 10. Apply COLORMAP_JET, convert BGR->RGB
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # 11. Create overlay
        # Convert input float32 [0, 1] image back to [0, 255] uint8
        original_uint8 = np.uint8(np.clip(image[0] * 255.0, 0, 255))
        overlay = cv2.addWeighted(
            original_uint8, 1.0 - alpha, heatmap_colored, alpha, 0.0
        )

        # 12. Extract top-5 highlighted regions via contour detection on thresholded heatmap
        binary_mask = np.uint8(heatmap >= 0.6) * 255
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create temporary mask to get mean intensity inside the contour boundary
            mask = np.zeros(heatmap.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            mean_val = np.mean(heatmap[mask == 255]) if np.any(mask == 255) else 0.0
            
            regions.append(
                Region(
                    x=int(x),
                    y=int(y),
                    w=int(w),
                    h=int(h),
                    confidence=float(round(mean_val, 4))
                )
            )

        # Sort regions by confidence descending and select top 5
        regions.sort(key=lambda r: r.confidence, reverse=True)
        highlighted_regions = regions[:5]

    except Exception as e:
        if isinstance(e, (InvalidInputError, InvalidLayerError, ComputationError)):
            raise e
        raise ComputationError(f"Error during Grad-CAM computation: {str(e)}") from e

    # Record total computation time
    end_time = time.perf_counter()
    computation_time_ms = float((end_time - start_time) * 1000)

    return GradCAMResult(
        heatmap=heatmap,
        overlay=overlay,
        highlighted_regions=highlighted_regions,
        computation_time_ms=computation_time_ms
    )
