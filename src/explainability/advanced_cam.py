import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from src.explainability.gradcam import GradCAMResult, Region


class ScoreCAMError(Exception):
    pass


def generate_scorecam(image: np.ndarray, model: tf.keras.Model, target_size=(224, 224), class_index: int = 0, num_samples: int = 10) -> GradCAMResult:
    """Generates Score-CAM explanation via forward-pass masking without gradients."""
    start_time = time.perf_counter()
    try:
        # Find the last convolutional layer (handle models with pooling/dense layers at the end)
        last_conv_layer = None
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer = layer
                break
        
        if not last_conv_layer:
            raise ScoreCAMError("Could not find a valid convolutional layer with 4D output.")
            
        feature_model = Model(inputs=model.inputs, outputs=last_conv_layer.output)
        feature_maps = feature_model(image)[0].numpy()  # Shape: (H, W, C)
        C = feature_maps.shape[-1]
        
        samples_to_use = min(C, num_samples)
        if C > samples_to_use:
            variances = np.var(feature_maps, axis=(0, 1))
            top_indices = np.argsort(variances)[-samples_to_use:]
        else:
            top_indices = range(C)
            
        masked_images = []
        upsampled_maps = []
        
        # Upsample and normalize feature maps to mask the input image
        for i in top_indices:
            fmap = feature_maps[:, :, i]
            upsampled = cv2.resize(fmap, target_size, interpolation=cv2.INTER_LINEAR)
            
            fmap_max, fmap_min = np.max(upsampled), np.min(upsampled)
            if fmap_max - fmap_min > 1e-8:
                upsampled = (upsampled - fmap_min) / (fmap_max - fmap_min)
            else:
                upsampled = np.zeros_like(upsampled)
            
            upsampled_maps.append(upsampled)
            masked_images.append(image[0] * np.expand_dims(upsampled, axis=-1))
            
        masked_images = np.array(masked_images)
        
        # Forward pass on all masked images
        preds = model.predict(masked_images, batch_size=samples_to_use, verbose=0)
        scores = preds[:, class_index]
        
        # Apply softmax to scores for weighting
        scores = np.exp(scores) / np.sum(np.exp(scores))
        
        heatmap = np.zeros(target_size, dtype=np.float32)
        for score, upmap in zip(scores, upsampled_maps):
            heatmap += score * upmap
            
        # ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
            
        # Create overlay
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        original_uint8 = np.uint8(np.clip(image[0] * 255.0, 0, 255))
        overlay = cv2.addWeighted(original_uint8, 0.6, heatmap_colored, 0.4, 0.0)
        
        # Extract regions
        binary_mask = np.uint8(heatmap >= 0.6) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            mask = np.zeros(heatmap.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_val = np.mean(heatmap[mask == 255]) if np.any(mask == 255) else 0.0
            regions.append(Region(x=int(x), y=int(y), w=int(w), h=int(h), confidence=float(round(mean_val, 4))))
            
        regions.sort(key=lambda r: r.confidence, reverse=True)
        
        return GradCAMResult(
            heatmap=heatmap,
            overlay=overlay,
            highlighted_regions=regions[:5],
            computation_time_ms=(time.perf_counter() - start_time) * 1000.0
        )
    except Exception as e:
        raise ScoreCAMError(f"Score-CAM computation failed: {str(e)}") from e


def should_use_scorecam(gradcam_result: GradCAMResult, confidence_threshold: float = 0.5) -> bool:
    """Decides if Score-CAM fallback is necessary based on Grad-CAM output quality."""
    if not gradcam_result.highlighted_regions:
        return True
    return max(r.confidence for r in gradcam_result.highlighted_regions) < confidence_threshold
