from argparse import ArgumentParser
from pathlib import Path
import os
import random
import cv2
import numpy as np

def is_empty(mask_path: Path) -> bool:
    """Return True if the given mask contains only zeros."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask file: {mask_path}")
    return not np.any(mask)

def filter_split(data_dir: Path, split: str, keep_ratio: float) -> None:
    """Remove patches with empty ground-truth masks for a given split."""
    parent_dir = Path(os.getcwd()).parent
    sar_dir = parent_dir / data_dir / "sar_images" / split
    front_dir = parent_dir / data_dir / "fronts" / split
    near_dir = parent_dir / data_dir / "near_fronts" / split
    if not front_dir.exists() or not near_dir.exists():
        print(f"Missing masks for split '{split}', skipping")
        return

    sar_files = [p.name for p in sar_dir.glob("*.png")]

    empty_names = []
    for name in sar_files:
        front_name = name.replace(".png", "_front.png")
        near_name = name.replace(".png", "_near_front.png")
        front_path = front_dir / front_name
        near_path = near_dir / near_name
        front_empty = not front_path.exists() or is_empty(front_path)
        near_empty = not near_path.exists() or is_empty(near_path)
        if front_empty and near_empty:
            empty_names.append(name)

    # keep a fraction of negative samples
    if keep_ratio < 1.0:
        random.shuffle(empty_names)
        keep_n = int(len(empty_names) * keep_ratio)
        empty_names = empty_names[keep_n:]

    for name in empty_names:
        paths = [
            sar_dir / name,
            front_dir / name.replace(".png", "_front.png"),
            near_dir / name.replace(".png", "_near_front.png"),
        ]
        for f in paths:
            if f.exists():
                os.remove(f)

def main(data_dir: str, keep_ratio: float) -> None:
    data_path = Path(data_dir)
    for split in ("train", "val", "test"):
        filter_split(data_path, split, keep_ratio)

if __name__ == "__main__":
    parser = ArgumentParser(description="Remove patches with empty ground-truth masks")
    parser.add_argument("--data_dir", default="data", help="Directory with processed patches")
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.2,
        help="Fraction of empty patches to keep as negatives",
    )
    args = parser.parse_args()
    print("Filtering empty patches...")
    main(args.data_dir, args.keep_ratio)
    print("Empty patches filtered successfully.")