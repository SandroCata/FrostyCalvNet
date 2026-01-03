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

def find_front_masks(sar_name: str, front_dir: Path) -> list[Path]:
    base = Path(sar_name).stem
    candidates = []
    if "__" in base:
        prefix, suffix = base.split("__", 1)
        candidates.append(front_dir / f"{prefix}_front__{suffix}.png")
    candidates.extend(
        [
            front_dir / f"{base}_front.png",
            front_dir / f"{base}.png",
        ]
    )
    existing = [path for path in candidates if path.exists()]
    if existing:
        return existing
    return sorted(front_dir.glob(f"{base}*.png"))

def filter_split(data_dir: Path, split: str, keep_ratio: float) -> None:
    """Remove patches with empty ground-truth masks for a given split."""
    parent_dir = Path(os.getcwd()).parent
    sar_dir = parent_dir / data_dir / "sar_images" / split
    front_dir = parent_dir / data_dir / "fronts" / split
    if not front_dir.exists():
        print(f"Missing masks for split '{split}', skipping")
        return

    sar_files = [p.name for p in sar_dir.glob("*.png")]

    empty_names = []
    empty_fronts = {}
    for name in sar_files:
        front_paths = find_front_masks(name, front_dir)
        if not front_paths or all(is_empty(front_path) for front_path in front_paths):
            empty_fronts[name] = front_paths
            empty_names.append(name)
            print(f"Empty mask found for {name} in split '{split}'")
    print(f"Found {len(empty_names)} empty masks in split '{split}'")

    # keep a fraction of negative samples
    if keep_ratio < 1.0:
        random.shuffle(empty_names)
        keep_n = int(len(empty_names) * keep_ratio)
        empty_names = empty_names[keep_n:]

    for name in empty_names:
        paths = [sar_dir / name]
        paths.extend(empty_fronts.get(name, []))
        for f in set(paths):
            if f.exists():
                os.remove(f)
                print("Removed:", f)

def main(data_dir: str, keep_ratio: float) -> None:
    data_path = Path(data_dir)
    for split in ("train", "val", "test"):
        filter_split(data_path, split, keep_ratio)

if __name__ == "__main__":
    parser = ArgumentParser(description="Remove patches with empty ground-truth masks")
    parser.add_argument("--data_dir", default="data_baseline", help="Directory with processed patches")
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.1,
        help="Fraction of empty patches to keep as negatives",
    )
    args = parser.parse_args()
    print("Filtering empty patches...")
    main(args.data_dir, args.keep_ratio)
    print("Empty patches filtered successfully.")