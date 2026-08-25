"""
image_preprocessor.py
---------------------
Retinal fundus image preprocessing pipeline for diabetic retinopathy detection.

Applies the following transformations:
  1. CLAHE (Contrast Limited Adaptive Histogram Equalization) on green channel
  2. Ben Graham preprocessing (circle crop + Gaussian blur subtract)
  3. Resize to target size
  4. Normalization (ImageNet stats)
  5. Data augmentation (train only)

Usage:
    from src.preprocessing.image_preprocessor import preprocess_image, get_transforms
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────

IMAGE_SIZE = (224, 224)

# ImageNet normalization stats (used by pretrained torchvision models)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Core Preprocessing ───────────────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from disk as a uint8 RGB numpy array.

    Args:
        image_path: Path to the image file.

    Returns:
        RGB numpy array of shape (H, W, 3), dtype uint8.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the image cannot be decoded.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_and_resize(image: np.ndarray, size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Crops to the retinal circle (removes black borders) and resizes.

    Args:
        image: RGB numpy array.
        size:  Target (width, height).

    Returns:
        Cropped and resized RGB array.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Bounding box of the largest contour (the retinal disc)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        image = image[y:y + h, x:x + w]

    return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE to the green channel (most informative for DR lesions).

    Args:
        image:      RGB numpy array (uint8).
        clip_limit: CLAHE contrast clip limit.
        tile_grid:  Grid size for local histogram equalization.

    Returns:
        CLAHE-enhanced RGB array.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    r, g, b = cv2.split(image)
    g_eq = clahe.apply(g)
    return cv2.merge([r, g_eq, b])


def ben_graham_preprocessing(image: np.ndarray,
                              sigmaX: int = 10) -> np.ndarray:
    """
    Applies Ben Graham's preprocessing: subtracts local Gaussian average
    to enhance local contrast and reduce lighting variation.

    Reference: https://www.kaggle.com/c/diabetic-retinopathy-detection

    Args:
        image:  RGB numpy array (uint8).
        sigmaX: Gaussian blur sigma.

    Returns:
        Contrast-enhanced RGB array.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX)
    enhanced = cv2.addWeighted(image, 4, blurred, -4, 128)

    # Apply circular mask to zero out the border
    h, w = image.shape[:2]
    mask = np.zeros_like(image)
    cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, (1, 1, 1), -1)
    return (enhanced * mask).astype(np.uint8)


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Converts to float32 in [0,1] and applies ImageNet mean/std normalization.

    Args:
        image: RGB numpy array (uint8, 0–255).

    Returns:
        Normalized float32 array of same shape.
    """
    img = image.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def preprocess_image(
    image_path: str,
    size: tuple = IMAGE_SIZE,
    use_clahe: bool = True,
    use_ben_graham: bool = True,
    apply_normalization: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single retinal fundus image.

    Steps:
        1. Load as RGB
        2. Crop retinal circle & resize
        3. CLAHE on green channel (optional)
        4. Ben Graham contrast enhancement (optional)
        5. ImageNet normalization (optional)

    Args:
        image_path:          Path to the image file.
        size:                Target (width, height) — default (224, 224).
        use_clahe:           Apply CLAHE enhancement.
        use_ben_graham:      Apply Ben Graham local contrast.
        apply_normalization: Apply ImageNet mean/std normalization.

    Returns:
        Preprocessed float32 numpy array of shape (H, W, 3).
    """
    img = load_image(image_path)
    img = crop_and_resize(img, size)

    if use_clahe:
        img = apply_clahe(img)

    if use_ben_graham:
        img = ben_graham_preprocessing(img)

    if apply_normalization:
        img = normalize(img)

    return img


# ── PyTorch Transforms ───────────────────────────────────────────────────────

def get_transforms(split: str = "train", size: int = 224):
    """
    Returns torchvision transforms for a given dataset split.

    Args:
        split: One of 'train', 'val', 'test'.
        size:  Target image size (square).

    Returns:
        torchvision.transforms.Compose object.

    Raises:
        ImportError: If torchvision is not installed.
    """
    try:
        from torchvision import transforms
    except ImportError:
        raise ImportError("torchvision is required. Install with: "
                          "pip install torch torchvision --index-url "
                          "https://download.pytorch.org/whl/cpu")

    normalize_transform = transforms.Normalize(
        mean=IMAGENET_MEAN.tolist(),
        std=IMAGENET_STD.tolist(),
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05,
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ToTensor(),
            normalize_transform,
        ])

    # val / test — deterministic
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize_transform,
    ])


# ── CLI Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Preview preprocessing on a single retinal image."
    )
    parser.add_argument("image_path", help="Path to a retinal fundus image")
    parser.add_argument("--size", type=int, default=224, help="Output size (default: 224)")
    parser.add_argument("--no-clahe",      action="store_true")
    parser.add_argument("--no-ben-graham", action="store_true")
    args = parser.parse_args()

    # Load original
    original = load_image(args.image_path)
    original_resized = cv2.resize(original, (args.size, args.size))

    # Preprocess (without final normalization for display)
    processed = preprocess_image(
        args.image_path,
        size=(args.size, args.size),
        use_clahe=not args.no_clahe,
        use_ben_graham=not args.no_ben_graham,
        apply_normalization=False,   # keep uint8 for display
    )

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_resized)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(processed)
    axes[1].set_title("Preprocessed (CLAHE + Ben Graham)")
    axes[1].axis("off")

    plt.suptitle("Retinal Fundus Image Preprocessing", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("preprocessing_preview.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved preview to preprocessing_preview.png")
