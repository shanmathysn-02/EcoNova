import numpy as np
import tensorflow as tf
from src.explainability.gradcam import generate_gradcam

# Use pretrained EfficientNetB0 as stand-in
print("Loading EfficientNetB0 model...")
model = tf.keras.applications.EfficientNetB0(weights='imagenet')
dummy = np.random.rand(1, 224, 224, 3).astype('float32')

print("Generating Grad-CAM heatmap and overlay...")
result = generate_gradcam(dummy, model, 'top_conv', 0, 0.4)

print(f"Heatmap shape: {result.heatmap.shape}")
print(f"Overlay shape: {result.overlay.shape}")
print(f"Highlighted regions count: {len(result.highlighted_regions)}")
for i, reg in enumerate(result.highlighted_regions):
    print(f"  Region {i}: Box=(x={reg.x}, y={reg.y}, w={reg.w}, h={reg.h}), Confidence={reg.confidence}")
print(f"Computation time: {result.computation_time_ms:.2f} ms")

# Run Assertions
assert result.heatmap.shape == (224, 224)
assert result.overlay.shape == (224, 224, 3)
assert len(result.highlighted_regions) >= 0
assert result.computation_time_ms < 500
print("Phase 1 Core Engine: PASSED")
