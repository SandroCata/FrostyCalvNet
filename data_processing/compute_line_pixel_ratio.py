"""Utility to compute pixel ratios between predicted and reference glacier front masks."""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence
import os

import numpy as np

try:  # pragma: no cover - optional dependency handling
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency handling
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


SCRIPT_ROOT = os.path.dirname(os.getcwd())
DEFAULT_VAL_PREDICTIONS_DIR = os.path.join(
    SCRIPT_ROOT, "FrostyCalvNet_onlyfront\\validation_results\\fronts\\run_0\\complete_images"
)
DEFAULT_TEST_PREDICTIONS_DIR = os.path.join(
    SCRIPT_ROOT, "FrostyCalvNet_onlyfront\\test_results\\fronts\\run_0\\complete_images"
)
DEFAULT_VAL_FRONT_LABELS_DIR = os.path.join(SCRIPT_ROOT, "data_raw\\fronts\\train")

DEFAULT_TEST_FRONT_LABELS_DIR = os.path.join(SCRIPT_ROOT, "data_raw\\fronts\\test")

DEFAULT_OUTPUT_PATH = os.path.join(SCRIPT_ROOT, "FrostyCalvNet_onlyfront\\pixel_ratios\\line_pixel_ratios.txt")

DATASET_DEFAULTS: dict[str, dict[str, Path]] = {
    "val": {
        "predictions_dir": Path(DEFAULT_VAL_PREDICTIONS_DIR),
        "front_labels_dir": Path(DEFAULT_VAL_FRONT_LABELS_DIR),
    },
    "test": {
        "predictions_dir": Path(DEFAULT_TEST_PREDICTIONS_DIR),
        "front_labels_dir": Path(DEFAULT_TEST_FRONT_LABELS_DIR),
    },
}
DATASET_CHOICES = ("both", *DATASET_DEFAULTS.keys())
DEFAULT_DATASET_CHOICE = "both"


@dataclass
class RatioResult:
    """Container for per-image line pixel ratios."""

    image_id: str
    predicted_front_pixels: int
    ground_truth_front_pixels: int
    front_ratio: float


def build_arg_parser() -> argparse.ArgumentParser:
    description = (
        "Compute the ratio between the number of glacier-front pixels detected in the predictions "
        "and in the reference masks.\n\n"
        "If you run the script without explicitly providing the directories, the following absolute "
        "paths—computed relative to this file—will be used for each dataset split:\n"
        f"  • Validation predictions: {DEFAULT_VAL_PREDICTIONS_DIR}\n"
        f"  • Validation front labels: {DEFAULT_VAL_FRONT_LABELS_DIR}\n"
        f"  • Test predictions: {DEFAULT_TEST_PREDICTIONS_DIR}\n"
        f"  • Test front labels: {DEFAULT_TEST_FRONT_LABELS_DIR}\n"
        f"  • Output CSV: {DEFAULT_OUTPUT_PATH}"
    )

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default=DEFAULT_DATASET_CHOICE,
        help=(
            "Dataset split whose default directories should be used when explicit paths are not "
            "provided. Use 'both' (the default) to process validation and test data in one run."
        ),
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help=(
            "Directory that stores the reconstructed predictions. By default, the directory "
            "corresponding to the chosen dataset split is used."
        ),
    )
    parser.add_argument(
        "--front-labels-dir",
        type=Path,
        default=None,
        help=(
            "Directory with the reference front masks. By default, the directory corresponding to "
            "the chosen dataset split is used."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(DEFAULT_OUTPUT_PATH),
        help=(
            "Where to store the per-image ratios as a CSV file. Defaults to a file named "
            "'line_pixel_ratios.txt' inside the validation results directory."
        ),
    )
    return parser


def _resolve_dataset_directories(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> list[tuple[str, Path, Path]]:
    overrides = {
        "predictions_dir": args.predictions_dir,
        "front_labels_dir": args.front_labels_dir,
    }

    if args.dataset == "both":
        if any(value is not None for value in overrides.values()):
            parser.error(
                "Manual directory overrides cannot be combined with --dataset set to 'both'."
            )
        return [
            (
                dataset_name,
                paths["predictions_dir"],
                paths["front_labels_dir"],
            )
            for dataset_name, paths in DATASET_DEFAULTS.items()
        ]

    defaults = DATASET_DEFAULTS[args.dataset]
    predictions_dir = overrides["predictions_dir"] or defaults["predictions_dir"]
    front_labels_dir = overrides["front_labels_dir"] or defaults["front_labels_dir"]

    return [(args.dataset, predictions_dir, front_labels_dir)]


def _load_grayscale_image(path: Path) -> "np.ndarray":
    """Load a grayscale image using any available backend."""

    if Image is not None:
        with Image.open(path) as img:
            return np.asarray(img.convert("L"))
    if cv2 is not None:  # pragma: no cover - exercised only when OpenCV is available
        array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if array is None:
            raise ValueError(f"Unable to load image: {path}")
        return array
    raise RuntimeError(
        "Neither Pillow nor OpenCV is available. Please install one of them to load PNG masks."
    )


def _count_active_pixels(path: Path) -> int:
    """Count the number of non-zero pixels in a binary mask."""

    image = _load_grayscale_image(path)
    return int(np.count_nonzero(image))


def _strip_known_suffix(name: str, suffixes: Sequence[str]) -> str:
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _iter_prediction_pairs(predictions_dir: Path) -> Iterator[tuple[str, Path]]:
    suffixes = ("_front.png", "_front.tif", "_front.jpg")
    for front_path in sorted(predictions_dir.glob("*_front.*")):
        base = _strip_known_suffix(front_path.name, suffixes)
        yield base, front_path


def _find_label(base: str, directory: Path, candidates: Sequence[str]) -> Optional[Path]:
    for suffix in candidates:
        path = directory / f"{base}{suffix}"
        if path.exists():
            return path
    matches = list(directory.glob(f"{base}*"))
    return matches[0] if matches else None


def compute_line_pixel_ratios(
    predictions_dir: Path,
    front_labels_dir: Path,
) -> list[RatioResult]:
    """Compute pixel ratios for each image found in ``predictions_dir``."""

    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory does not exist: {predictions_dir}")
    if not front_labels_dir.exists():
        raise FileNotFoundError(f"Front labels directory does not exist: {front_labels_dir}")

    results: list[RatioResult] = []
    for image_id, predicted_front_path in _iter_prediction_pairs(predictions_dir):
        front_label_path = _find_label(
            image_id, front_labels_dir, ("_front.png", ".png", "_front.tif", ".tif")
        )

        if front_label_path is None:
            raise FileNotFoundError(
                f"Could not find a front label for '{image_id}' inside {front_labels_dir}"
            )

        predicted_front_pixels = _count_active_pixels(predicted_front_path)
        ground_truth_front_pixels = _count_active_pixels(front_label_path)

        if ground_truth_front_pixels == 0:
            front_ratio = math.nan
        else:
            front_ratio = predicted_front_pixels / ground_truth_front_pixels

        results.append(
            RatioResult(
                image_id=image_id,
                predicted_front_pixels=predicted_front_pixels,
                ground_truth_front_pixels=ground_truth_front_pixels,
                front_ratio=front_ratio,
            )
        )
    return results


def _mean_without_nan(values: Iterable[Optional[float]]) -> float:
    filtered = [value for value in values if value is not None and not math.isnan(value)]
    return float(sum(filtered) / len(filtered)) if filtered else math.nan


def summarise_results(results: Sequence[RatioResult]) -> dict[str, float]:
    return {
        "mean_front_ratio": _mean_without_nan(result.front_ratio for result in results),
    }


def write_results_to_csv(path: Path, results: Sequence[RatioResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "image_id",
                "predicted_front_pixels",
                "ground_truth_front_pixels",
                "front_ratio",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.image_id,
                    result.predicted_front_pixels,
                    result.ground_truth_front_pixels,
                    result.front_ratio,
                ]
            )


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(cli_args)

    dataset_directories = _resolve_dataset_directories(args, parser)
    multiple_datasets = len(dataset_directories) > 1

    for index, (dataset_name, predictions_dir, front_labels_dir) in enumerate(
        dataset_directories
    ):
        results = compute_line_pixel_ratios(
            predictions_dir=predictions_dir,
            front_labels_dir=front_labels_dir,
        )

        output_path = args.output_csv
        if multiple_datasets:
            output_path = output_path.with_name(f"{dataset_name}_{output_path.name}")

        write_results_to_csv(output_path, results)

        summary = summarise_results(results)
        prefix = f"[{dataset_name}] "
        print(prefix + "Computed ratios for", len(results), "images")
        print(prefix + "Mean front-line pixel ratio:", summary["mean_front_ratio"])
        print(prefix + "Results written to:", output_path)

        if multiple_datasets and index < len(dataset_directories) - 1:
            print()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()