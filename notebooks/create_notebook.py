import os
import nbformat as nbf

def create_experiment_notebook():
    os.makedirs("notebooks", exist_ok=True)
    os.makedirs(os.path.join("notebooks", "figures"), exist_ok=True)
    
    nb = nbf.v4.new_notebook()
    
    nb.cells.append(nbf.v4.new_markdown_cell("# Grad-CAM vs Score-CAM: DR Explanation Experiments\n\nCompare explanation methods on retinal fundus images."))
    
    code_1 = """import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.applications.EfficientNetB0(weights='imagenet')

# Synthetic fundus image
image = np.zeros((224, 224, 3), dtype=np.uint8)
cv2.circle(image, (112, 112), 100, (0, 60, 180), -1)
for _ in range(5):
    pt1 = (np.random.randint(50, 174), np.random.randint(50, 174))
    pt2 = (np.random.randint(50, 174), np.random.randint(50, 174))
    cv2.line(image, pt1, pt2, (0, 0, 100), 2)
    
image_float = (image.astype(np.float32) / 255.0).reshape(1, 224, 224, 3)
"""
    nb.cells.append(nbf.v4.new_code_cell(code_1))
    
    code_2 = """from src.explainability.gradcam import generate_gradcam

gradcam_res = generate_gradcam(image_float, model, 'top_conv', 0)
plt.imshow(gradcam_res.heatmap, cmap='jet')
plt.title("Grad-CAM Heatmap")
plt.axis('off')
plt.show()
"""
    nb.cells.append(nbf.v4.new_code_cell(code_2))
    
    code_3 = """from src.explainability.advanced_cam import generate_scorecam

scorecam_res = generate_scorecam(image_float, model, class_index=0)
plt.imshow(scorecam_res.heatmap, cmap='jet')
plt.title("Score-CAM Heatmap")
plt.axis('off')
plt.show()
"""
    nb.cells.append(nbf.v4.new_code_cell(code_3))
    
    code_4 = """fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(image)
axes[0].set_title("Original")
axes[1].imshow(gradcam_res.heatmap, cmap='jet')
axes[1].set_title("Grad-CAM Heatmap")
axes[2].imshow(scorecam_res.heatmap, cmap='jet')
axes[2].set_title("Score-CAM Heatmap")
axes[3].imshow(gradcam_res.overlay)
axes[3].set_title("Grad-CAM Overlay")
axes[4].imshow(scorecam_res.overlay)
axes[4].set_title("Score-CAM Overlay")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join('notebooks', 'figures', 'comparison_sample.png'))
plt.show()
"""
    nb.cells.append(nbf.v4.new_code_cell(code_4))
    
    code_5 = """print(f"Grad-CAM Regions: {len(gradcam_res.highlighted_regions)} | Time: {gradcam_res.computation_time_ms:.1f}ms")
print(f"Score-CAM Regions: {len(scorecam_res.highlighted_regions)} | Time: {scorecam_res.computation_time_ms:.1f}ms")

gradcam_conf = max([r.confidence for r in gradcam_res.highlighted_regions]) if gradcam_res.highlighted_regions else 0
scorecam_conf = max([r.confidence for r in scorecam_res.highlighted_regions]) if scorecam_res.highlighted_regions else 0

if scorecam_conf > gradcam_conf + 0.1:
    print("Recommendation: Use Score-CAM")
else:
    print("Recommendation: Use Grad-CAM")
"""
    nb.cells.append(nbf.v4.new_code_cell(code_5))
    
    nbf.write(nb, os.path.join("notebooks", "03_gradcam_experiments.ipynb"))
    print("Notebook generated.")

if __name__ == "__main__":
    create_experiment_notebook()
