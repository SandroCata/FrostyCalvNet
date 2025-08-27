from argparse import ArgumentParser
import os
from pathlib import Path

import cv2
import numpy as np


def thicken_fronts(mask: np.ndarray) -> np.ndarray:
    """Dilate front label with a 3x3 kernel."""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def highlight_near_fronts(mask: np.ndarray) -> np.ndarray:
    """Create mask of near-front pixels."""
    mask_line_pad = thicken_fronts(mask)
    mask_highlight = cv2.subtract(mask_line_pad, mask)
    mask_highlight[mask_highlight > 0] = 255
    return mask_highlight


def generate_near_front_masks(raw_data_dir: str) -> None:
    """Generate near-front masks from front labels located in ``raw_data_dir``.

    The generated masks are stored in ``raw_data_dir/near_fronts`` maintaining
    the original directory structure (train/test).
    """
    parent_dir = os.path.dirname(os.getcwd())
    for split in ["train", "test"]:
        src_dir = Path(parent_dir, raw_data_dir, "fronts", split)
        dst_dir = Path(parent_dir, raw_data_dir, "near_fronts", split)
        dst_dir.mkdir(parents=True, exist_ok=True)

        for file in src_dir.rglob("*.png"):
            mask = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            near_mask = highlight_near_fronts(mask)
            out_name = file.name
            if out_name.endswith("_front.png"):
                out_name = out_name.replace("_front.png", "_near_front.png")
            else:
                out_name = f"{file.stem}_near_front.png"
            cv2.imwrite(str(dst_dir / out_name), near_mask)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--raw_data_dir",
        default="data_raw",
        help="Directory containing the raw data (default: 'data_raw').",
    )
    args = parser.parse_args()
    generate_near_front_masks(args.raw_data_dir)