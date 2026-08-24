"""
dataset_split.py
----------------
Loads retinal fundus images from a directory structure, splits into
train (70%) / validation (15%) / test (15%) sets with stratification,
computes class weights for imbalanced datasets, and saves CSV manifests.

Expected directory structure:
    data/raw/
        No DR/
        Mild/
        Moderate/
        Severe/
        Proliferative DR/

Usage:
    python -m src.preprocessing.dataset_split
    # or with custom args:
    python src/preprocessing/dataset_split.py --data-dir data/raw --output-dir data/processed
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# DR severity grade mapping (ordinal encoding)
DR_GRADE_MAP = {
    "No DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Proliferative DR": 4,
    "Proliferative": 4,  # alias
}


def load_image_paths(data_dir: str) -> pd.DataFrame:
    """
    Scans data_dir for image files organised in class subfolders.

    Args:
        data_dir: Root directory containing one subfolder per DR class.

    Returns:
        DataFrame with columns: filepath, label, grade
    """
    data = []
    class_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    if not class_dirs:
        raise ValueError(
            f"No subdirectories found in '{data_dir}'. "
            "Expected one folder per DR class."
        )

    print(f"\n📂 Found {len(class_dirs)} class folders: {class_dirs}")

    for class_name in class_dirs:
        class_dir = os.path.join(data_dir, class_name)
        images_found = 0
        for img_name in os.listdir(class_dir):
            ext = os.path.splitext(img_name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                data.append({
                    "filepath": os.path.join(class_dir, img_name),
                    "label": class_name,
                    "grade": DR_GRADE_MAP.get(class_name, -1),
                })
                images_found += 1
        print(f"   {class_name:25s}: {images_found:>5} images")

    if not data:
        raise ValueError(
            f"No images found in '{data_dir}'. "
            "Check your directory structure and image extensions."
        )

    return pd.DataFrame(data)


def compute_class_weights_dict(labels: pd.Series) -> dict:
    """
    Computes balanced class weights to handle label imbalance.

    Args:
        labels: Series of string class labels.

    Returns:
        Dict mapping class name -> weight float.
    """
    classes = np.array(sorted(labels.unique()))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels.values,
    )
    return dict(zip(classes, weights))


def print_split_stats(name: str, df: pd.DataFrame, weights: dict) -> None:
    """Prints a per-class breakdown for a given split."""
    print(f"\n{'─' * 52}")
    print(f"  {name} split  ({len(df)} samples)")
    print(f"{'─' * 52}")
    counts = Counter(df["label"])
    for cls in sorted(counts):
        w = weights.get(cls, 1.0)
        pct = counts[cls] / len(df) * 100
        print(f"   {cls:25s}: {counts[cls]:>5}  ({pct:5.1f}%)  weight={w:.4f}")


def create_dataset_csv(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """
    Splits retinal fundus images into train / val / test CSVs with class weights.

    Args:
        data_dir:      Directory with one subfolder per DR class.
        output_dir:    Directory where CSV files are saved.
        train_ratio:   Fraction for training (default 0.70).
        val_ratio:     Fraction for validation (default 0.15).
        test_ratio:    Fraction for testing (default 0.15).
        random_state:  Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"

    print("=" * 60)
    print("  Diabetic Retinopathy — Dataset Splitter")
    print("=" * 60)

    # 1. Load all image paths
    df = load_image_paths(data_dir)
    print(f"\n✅ Total images: {len(df)}  |  Classes: {df['label'].nunique()}")

    # 2. Compute balanced class weights
    class_weights = compute_class_weights_dict(df["label"])
    print("\n⚖️  Class weights (for loss balancing):")
    for cls, w in sorted(class_weights.items(), key=lambda x: DR_GRADE_MAP.get(x[0], 99)):
        print(f"   {cls:25s}: {w:.4f}")

    # 3. Stratified split — train vs (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["label"],
        random_state=random_state,
    )

    # 4. Stratified split — val vs test
    relative_test = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=random_state,
    )

    # 5. Attach class weights column to each split
    for split_df in (train_df, val_df, test_df):
        split_df["class_weight"] = split_df["label"].map(class_weights)

    # 6. Save split CSVs
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # 7. Save class weights as a separate CSV
    weights_df = pd.DataFrame(
        list(class_weights.items()), columns=["label", "weight"]
    )
    weights_df["grade"] = weights_df["label"].map(DR_GRADE_MAP)
    weights_df = weights_df.sort_values("grade").reset_index(drop=True)
    weights_df.to_csv(os.path.join(output_dir, "class_weights.csv"), index=False)

    # 8. Print per-split statistics
    print_split_stats("Train",      train_df, class_weights)
    print_split_stats("Validation", val_df,   class_weights)
    print_split_stats("Test",       test_df,  class_weights)

    print(f"\n{'=' * 60}")
    print(f"✅ Saved to '{output_dir}':")
    print(f"   train.csv          — {len(train_df):>5} samples")
    print(f"   val.csv            — {len(val_df):>5} samples")
    print(f"   test.csv           — {len(test_df):>5} samples")
    print(f"   class_weights.csv  — {len(weights_df):>5} classes")
    print(f"{'=' * 60}\n")

    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split DR retinal image dataset into train/val/test CSVs."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join("data", "raw"),
        help="Root directory containing DR class subfolders (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "processed"),
        help="Output directory for CSV files (default: data/processed)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70,
                        help="Training split ratio (default: 0.70)")
    parser.add_argument("--val-ratio",   type=float, default=0.15,
                        help="Validation split ratio (default: 0.15)")
    parser.add_argument("--test-ratio",  type=float, default=0.15,
                        help="Test split ratio (default: 0.15)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    create_dataset_csv(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )
