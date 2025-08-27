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
    parent_dir = os.path.dirname(os.getcwd())
    sar_dir = parent_dir / data_dir / "sar_images" / split
    front_dir = parent_dir / data_dir / "fronts" / split
    near_dir = parent_dir / data_dir / "near_fronts" / split
    if not near_dir.exists():
        print(f"No near_fronts directory for split '{split}', skipping")
        return

    empty_masks = [p for p in near_dir.glob("*.png") if is_empty(p)]
    # keep a fraction of negative samples
    if keep_ratio < 1.0:
        random.shuffle(empty_masks)
        keep_n = int(len(empty_masks) * keep_ratio)
        empty_masks = empty_masks[keep_n:]

    for mask_path in empty_masks:
        name = mask_path.name
        # remove mask itself
        try:
            os.remove(mask_path)
        except FileNotFoundError:
            pass
        # remove corresponding sar and front patches
        for d in (sar_dir, front_dir):
            f = d / name
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
        default=0.1,
        help="Fraction of empty patches to keep as negatives",
    )
    args = parser.parse_args()
    main(args.data_dir, args.keep_ratio)