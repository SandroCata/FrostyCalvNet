from argparse import ArgumentParser, BooleanOptionalAction
from models.zones_segmentation_model import ZonesUNet
from models.front_segmentation_model import FrontUNet
from data_processing.glacier_zones_data import GlacierZonesDataModule
from data_processing.glacier_front_data import GlacierFrontDataModule
from data_processing.data_postprocessing import reconstruct_from_patches_and_binarize
from data_processing.data_postprocessing import (
    postprocess_zone_segmenation,
    postprocess_front_segmentation,  # Commented out to disable front thinning
    extract_front_from_zones,
)
import os
import tempfile
import pickle
import torch
import re
import cv2
import numpy as np
from scipy.spatial import distance, cKDTree
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score, confusion_matrix
import skimage
import shutil
import builtins
from functools import partial
from callbacks import OneBasedModelCheckpoint, OneBasedProgressBar, LossHistory


def front_error(prediction, label):
    """
    Calculate the distance errors between prediction and label
    :param prediction: mask of the front prediction (WxH)
    :param label: mask of the front label (WxH)
    :return: boolean whether a front pixel is present in the prediction and the distance errors as np.array
    """
    front_is_present_flag = True
    polyline_pred = np.nonzero(prediction)
    polyline_label = np.nonzero(label)

    # Generate Nx2 matrix of pixels that represent the front
    pred_coords = np.array(list(zip(polyline_pred[0], polyline_pred[1])))
    mask_coords = np.array(list(zip(polyline_label[0], polyline_label[1])))

    # Return NaN if front is not detected in either pred or mask
    if pred_coords.shape[0] == 0 or mask_coords.shape[0] == 0:
        front_is_present_flag = False
        return front_is_present_flag, np.nan

    # Generate the pairwise distances between each point and the closest point in the other array
    # Use KDTree to avoid allocating huge pairwise distance matrices
    tree_pred = cKDTree(pred_coords)
    tree_mask = cKDTree(mask_coords)
    distances1 = tree_mask.query(pred_coords, k=1)[0]
    distances2 = tree_pred.query(mask_coords, k=1)[0]
    distances = np.concatenate((distances1, distances2))

    return front_is_present_flag, distances


def multi_class_metric(metric_function, complete_predicted_mask, complete_target):
    """
    Calculate single-class metrics and the multi-class metric for a multi-class task
    :param metric_function: function of the metric to be calculated
    :param complete_predicted_mask: complete prediction
    :param complete_target: complete target mask
    :return: the metrics as list
    """
    metrics = []
    metric_na, metric_stone, metric_glacier, metric_ocean = metric_function(np.ndarray.flatten(complete_target), np.ndarray.flatten(complete_predicted_mask), average=None)
    metric_macro_average = (metric_na + metric_stone + metric_glacier + metric_ocean) / 4
    metrics.append(metric_macro_average)
    metrics.append(metric_na)
    metrics.append(metric_stone)
    metrics.append(metric_glacier)
    metrics.append(metric_ocean)
    return metrics

def multi_class_accuracy(complete_predicted_mask, complete_target):
    """Calculate per-class and macro average accuracy for a multi-class task."""
    metrics = []
    flattened_pred = np.ndarray.flatten(complete_predicted_mask)
    flattened_target = np.ndarray.flatten(complete_target)
    labels = [0, 1, 2, 3]
    cm = confusion_matrix(flattened_target, flattened_pred, labels=labels)
    per_class_total = cm.sum(axis=1)
    per_class_acc = np.divide(cm.diagonal(), per_class_total, out=np.zeros_like(per_class_total, dtype=float), where=per_class_total!=0)
    metrics.append(per_class_acc.mean())
    metrics.extend(per_class_acc.tolist())
    return metrics


def turn_colors_to_class_labels_zones(mask):
    mask_class_labels = np.copy(mask)
    mask_class_labels[mask == 0] = 0
    mask_class_labels[mask == 64] = 1
    mask_class_labels[mask == 127] = 2
    mask_class_labels[mask == 254] = 3
    return mask_class_labels


def turn_colors_to_class_labels_front(mask):
    mask_class_labels = np.copy(mask)
    mask_class_labels[mask == 0] = 0
    mask_class_labels[mask == 255] = 1
    return mask_class_labels


def print_zone_metrics(metric_name, list_of_metrics):
    metrics = [metric for [metric, _, _, _, _] in list_of_metrics if not np.isnan(metric)]
    metrics_na = [metric_na for [_, metric_na, _, _, _] in list_of_metrics if not np.isnan(metric_na)]
    metrics_stone = [metric_stone for [_, _, metric_stone, _, _] in list_of_metrics if not np.isnan(metric_stone)]
    metrics_glacier = [metric_glacier for [_, _, _, metric_glacier, _] in list_of_metrics if not np.isnan(metric_glacier)]
    metrics_ocean = [metric_ocean for [_, _, _, _, metric_ocean] in list_of_metrics if not np.isnan(metric_ocean)]
    print(f"Average {metric_name}: {sum(metrics) / len(metrics)}")
    print(f"Average {metric_name} NA Area: {sum(metrics_na) / len(metrics_na)}")
    print(f"Average {metric_name} Stone: {sum(metrics_stone) / len(metrics_stone)}")
    print(f"Average {metric_name} Glacier: {sum(metrics_glacier) / len(metrics_glacier)}")
    print(f"Average {metric_name} Ocean and Ice Melange: {sum(metrics_ocean) / len(metrics_ocean)}")


def print_front_metric(name, metric):
    print(f"Average {name}: {sum(metric) / len(metric)}")


def get_matching_out_of_folder(file_name, folder, expected_suffix=None):
    """
    Find the file with the file_name that starts with the parameter file_name (minus .png)
    :param file_name: the searched file_name start
    :param folder: the folder which is to be searched
    :param expected_suffix: optional suffix that the matching file should have
    :return: the file name in the folder that starts with the parameter file_name (minus .png)
    """
    folder = folder.replace("\\", os.sep)
    folder = os.path.normpath(folder)
    files = os.listdir(folder)
    pattern_name = os.path.split(file_name)[1][:-4]
    pattern_name = re.sub(r'_front$', '', pattern_name)
    if expected_suffix is not None:
        expected_name = f"{pattern_name}{expected_suffix}"
        matching_files = [a for a in files if os.path.split(a)[1] == expected_name]
    else:
        matching_files = [a for a in files if os.path.split(a)[1].startswith(pattern_name)]
    if len(matching_files) > 1:
        print("Something went wrong!")
        print(f"targets_matching: {matching_files}")
    if len(matching_files) < 1:
        print("Something went wrong! No matches found")
    return matching_files[0]

def crop_to_same_shape(a, b):
    """Crop two arrays to their smallest shared height and width."""
    min_height = min(a.shape[0], b.shape[0])
    min_width = min(a.shape[1], b.shape[1])
    return a[:min_height, :min_width], b[:min_height, :min_width]

def evaluate_model_on_given_dataset(mode, model, datamodule, patch_directory):
    """
    Evaluate the model on the given dataset
    :param mode: "test" or "validate"
    :param model: the model to be evaluated
    :param datamodule: the datamodule for the model
    :param patch_directory: the directory in which the predicted image patches are going to be stored (this folder is deleted again after all calculations)
    :return: The average loss of the model on this dataset
    """
    print("\n\nEvaluate Model on given dataset ...\n\n")
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        losses = []

        datamodule.setup()
        if mode == "test":
            dataloader = datamodule.test_dataloader()
        else:
            dataloader = datamodule.val_dataloader()

        for i, batch in enumerate(dataloader):
            x, y, x_names, y_names = batch
            batch_with_batch_shape = (x.to(device), y.to(device), x_names, y_names)

            assert x.shape[1] == model.n_channels_of_input, \
                f"Network has been defined with {model.n_channels_of_input} input channels, " \
                f"but loaded images have {x.shape[1]} channels. Please check that " \
                "the images are loaded correctly."
            y_hat = model.give_prediction_for_batch(batch_with_batch_shape)
            loss, metric = model.calc_loss(y_hat, y.to(device))
            losses.append(loss.item())

            # Take sigmoid of prediction to get probabilities from logits and save probabilities in files
            y_hat = torch.sigmoid(y_hat)
            for index_in_batch in range(len(y_hat)):
                front_map = y_hat[index_in_batch, 0].to("cpu").clone()
                base_name = x_names[index_in_batch][:-4]
                with open(os.path.join(patch_directory, base_name + "_front.png"), "wb") as fp:
                    pickle.dump(front_map, fp)

        average_loss = sum(losses) / len(losses)
        return average_loss, losses



def calculate_segmentation_metrics(target_mask_modality, complete_predicted_masks, complete_directory,
                                   directory_of_complete_targets_front):
    """
    Calculate IoU, Precision, Recall and F1-Score for all predicted segmentations
    :param target_mask_modality: either "zones" or "fronts"
    :param complete_predicted_masks: All file names of complete predictions
    :param complete_directory: the directory which is going to hold the complete predicted images
    :param directory_of_complete_targets_front: directory holding the complete ground truth segmentations for the front class
    :return:
    """
    print("\n\nCalculate segmentation metrics ...\n\n")
    if target_mask_modality == "zones":
        list_of_ious = []
        list_of_precisions = []
        list_of_recalls = []
        list_of_f1_scores = []
        list_of_accuracies = []
        precision = partial(precision_score, zero_division=0)
        recall = partial(recall_score, zero_division=0)
        f1 = partial(f1_score, zero_division=0)
        for file_name in complete_predicted_masks:
            print(f"File: {file_name}")
            complete_predicted_mask = cv2.imread(os.path.join(complete_directory, file_name).__str__(), cv2.IMREAD_GRAYSCALE)
            matching_target_file = get_matching_out_of_folder(file_name, directory_of_complete_targets_front)
            complete_target = cv2.imread(os.path.join(directory_of_complete_targets_front, matching_target_file).__str__(), cv2.IMREAD_GRAYSCALE)
            if complete_predicted_mask.shape != complete_target.shape:
                complete_predicted_mask, complete_target = crop_to_same_shape(complete_predicted_mask, complete_target)
            complete_predicted_mask_class_labels = turn_colors_to_class_labels_zones(complete_predicted_mask)
            complete_target_class_labels = turn_colors_to_class_labels_zones(complete_target)
            # Segmentation evaluation metrics
            list_of_ious.append(multi_class_metric(jaccard_score, complete_predicted_mask_class_labels, complete_target_class_labels))
            list_of_precisions.append(multi_class_metric(precision, complete_predicted_mask_class_labels, complete_target_class_labels))
            list_of_recalls.append(multi_class_metric(recall, complete_predicted_mask_class_labels, complete_target_class_labels))
            list_of_f1_scores.append(multi_class_metric(f1, complete_predicted_mask_class_labels, complete_target_class_labels))
            list_of_accuracies.append(multi_class_accuracy(complete_predicted_mask_class_labels, complete_target_class_labels))
        
        print_zone_metrics("Precision", list_of_precisions)
        print_zone_metrics("Recall", list_of_recalls)
        print_zone_metrics("F1 Score", list_of_f1_scores)
        print_zone_metrics("IoU", list_of_ious)
        print_zone_metrics("Accuracy", list_of_accuracies)
    else:
        
        metrics_front = {"iou": [], "precision": [], "recall": [], "f1": [], "accuracy": []}
        for file_name in complete_predicted_masks:
            if not file_name.endswith('_front.png'):
                continue
            print(f"File: {file_name}")
            complete_predicted_mask = cv2.imread(os.path.join(complete_directory, file_name).__str__(), cv2.IMREAD_GRAYSCALE)
            matching_target_file = get_matching_out_of_folder(file_name, directory_of_complete_targets_front, expected_suffix="_front.png")
            complete_target = cv2.imread(os.path.join(directory_of_complete_targets_front, matching_target_file).__str__(), cv2.IMREAD_GRAYSCALE)
            if complete_predicted_mask.shape != complete_target.shape:
                complete_predicted_mask, complete_target = crop_to_same_shape(complete_predicted_mask, complete_target)
            complete_predicted_mask_class_labels = turn_colors_to_class_labels_front(complete_predicted_mask)
            complete_target_class_labels = turn_colors_to_class_labels_front(complete_target)
            flattened_complete_target_class_labels = np.ndarray.flatten(complete_target_class_labels)
            flattened_complete_predicted_mask_class_labels = np.ndarray.flatten(complete_predicted_mask_class_labels)
            metrics_front["iou"].append(jaccard_score(flattened_complete_target_class_labels, flattened_complete_predicted_mask_class_labels))
            metrics_front["precision"].append(
                precision_score(
                    flattened_complete_target_class_labels,
                    flattened_complete_predicted_mask_class_labels,
                    zero_division=0,
                )
            )
            metrics_front["recall"].append(
                recall_score(
                    flattened_complete_target_class_labels,
                    flattened_complete_predicted_mask_class_labels,
                    zero_division=0,
                )
            )
            metrics_front["f1"].append(
                f1_score(
                    flattened_complete_target_class_labels,
                    flattened_complete_predicted_mask_class_labels,
                    zero_division=0,
                )
            )
            metrics_front["accuracy"].append(
                accuracy_score(
                    flattened_complete_target_class_labels,
                    flattened_complete_predicted_mask_class_labels,
                )
            )
        print("\nFront:")
        print_front_metric("Precision", metrics_front["precision"])
        print_front_metric("Recall", metrics_front["recall"])
        print_front_metric("F1 Score", metrics_front["f1"])
        print_front_metric("IoU", metrics_front["iou"])
        print_front_metric("Accuracy", metrics_front["accuracy"])


def mask_prediction_with_bounding_box(post_complete_predicted_mask, file_name, bounding_boxes_directory):
    """
    Set the predicted pixels outside the bounding box to background
    :param post_complete_predicted_mask: post-processed prediction
    :param file_name: File name of the prediction
    :param bounding_boxes_directory: directory holding the bounding boxes
    :return: altered prediction
    """
    matching_bounding_box_file = get_matching_out_of_folder(file_name, bounding_boxes_directory)
    with open(os.path.join(bounding_boxes_directory, matching_bounding_box_file)) as f:
        coord_file_lines = f.readlines()
    left_upper_corner_x, left_upper_corner_y = [round(float(coord)) for coord in coord_file_lines[1].split(",")]
    left_lower_corner_x, left_lower_corner_y = [round(float(coord)) for coord in coord_file_lines[2].split(",")]
    right_lower_corner_x, right_lower_corner_y = [round(float(coord)) for coord in coord_file_lines[3].split(",")]
    right_upper_corner_x, right_upper_corner_y = [round(float(coord)) for coord in coord_file_lines[4].split(",")]

    # Make sure the Bounding Box coordinates are within the image
    if left_upper_corner_x < 0: left_upper_corner_x = 0
    if left_lower_corner_x < 0: left_lower_corner_x = 0
    if right_upper_corner_x > len(post_complete_predicted_mask[0]): right_upper_corner_x = len(post_complete_predicted_mask[0]) - 1
    if right_lower_corner_x > len(post_complete_predicted_mask[0]): right_lower_corner_x = len(post_complete_predicted_mask[0]) - 1
    if left_upper_corner_y > len(post_complete_predicted_mask): left_upper_corner_y = len(post_complete_predicted_mask) - 1
    if left_lower_corner_y < 0: left_lower_corner_y = 0
    if right_upper_corner_y > len(post_complete_predicted_mask): right_upper_corner_y = len(post_complete_predicted_mask) - 1
    if right_lower_corner_y < 0: right_lower_corner_y = 0

    # remember cv2 images have the shape (height, width)
    post_complete_predicted_mask[:right_lower_corner_y, :] = 0.0
    post_complete_predicted_mask[left_upper_corner_y:, :] = 0.0
    post_complete_predicted_mask[:, :left_upper_corner_x] = 0.0
    post_complete_predicted_mask[:, right_lower_corner_x:] = 0.0

    return post_complete_predicted_mask


def post_processing(target_masks, complete_predicted_masks, bounding_boxes_directory):
    """
    Post-process all predictions
    :param target_masks: either "zones" or "fronts"
    :param complete_predicted_masks: All file names of complete predictions
    :param bounding_boxes_directory: directory holding the bounding boxes
    :return:
    """
    meter_threshold = 750     # in meter
    print("\n\nPost-processing ...\n\n")
    for file_name in complete_predicted_masks:
        print(f"File: {file_name}")
        name = os.path.split(file_name)[1][:-4]
        name = re.sub(r'_front$', '', name)
        resolution = int(name.split('_')[-3])
        # pixel_threshold (pixel) * resolution (m/pixel) = meter_threshold (m)
        pixel_threshold = meter_threshold / resolution
        complete_predicted_mask = cv2.imread(os.path.join(complete_directory, file_name).__str__(), cv2.IMREAD_GRAYSCALE)

        if target_masks == "zones":
            post_complete_predicted_mask = postprocess_zone_segmenation(complete_predicted_mask)
            post_complete_predicted_mask = extract_front_from_zones(post_complete_predicted_mask, pixel_threshold)
        else:
            complete_predicted_mask_class_labels = turn_colors_to_class_labels_front(complete_predicted_mask)
            post_complete_predicted_mask = complete_predicted_mask_class_labels
            #restored since thickening still occurs in mk0
            post_complete_predicted_mask = postprocess_front_segmentation(complete_predicted_mask_class_labels, pixel_threshold)
            post_complete_predicted_mask = post_complete_predicted_mask * 255

        post_complete_predicted_mask = mask_prediction_with_bounding_box(post_complete_predicted_mask, file_name,
                                                                         bounding_boxes_directory)
        cv2.imwrite(os.path.join(complete_postprocessed_directory, file_name), post_complete_predicted_mask)


def calculate_front_delineation_metric(
        post_processed_predicted_masks,
        directory_of_target_fronts,
        mask_suffix="_front.png",
        structure_name="front",
):
    """
    Calculate distance errors and corresponding statistics
    :param post_processed_predicted_masks: All file names of post-processed complete predictions
    :param directory_of_target_fronts: directory holding the complete ground truth fronts
    :param mask_suffix: the suffix identifying the prediction files to evaluate
    :param structure_name: name of the structure for printing purposes
    :return: a list of all distance errors, if a prediction does not show a single front pixel, that image is neglected
    """

    """
    1) front error detects pixel coordinates belonging to the front in both prediction and label. If one of those is empty, image is discarded

    2) For each front predicted point compute distance to closest ground truth point distance1.
       For each ground truth point compute distance to closest predicted point distance2.
       The two distance series are concatenated.

    3) The distances are multiplied with the image resolution to get distances in meters.

    4) All distances of all images are concatenated to one big list of distances.

    5) The mean of all distances is calculated and returned.
    
    """

    post_processed_predicted_masks = [
        f for f in post_processed_predicted_masks if f.endswith(mask_suffix)
    ]
    list_of_all_errors = []
    number_of_images_with_no_prediction = 0
    for file_name in post_processed_predicted_masks:
        post_processed_predicted_mask = cv2.imread(
            os.path.join(complete_postprocessed_directory, file_name).__str__(),
            cv2.IMREAD_GRAYSCALE,
        )
        
        matching_target_file = get_matching_out_of_folder(
            file_name, directory_of_target_fronts, expected_suffix=mask_suffix
        )
        target_front = cv2.imread(
            os.path.join(directory_of_target_fronts, matching_target_file).__str__(),
            cv2.IMREAD_GRAYSCALE,
        )
        name = os.path.split(file_name)[1][:-4]
        name = re.sub(r'_front$', '', name)
        resolution = int(name.split('_')[-3])

        # images need to be turned into a Tensor [0, ..., n_classes-1]
        post_processed_predicted_mask_class_labels = turn_colors_to_class_labels_front(
            post_processed_predicted_mask
        )
        target_front_class_labels = turn_colors_to_class_labels_front(target_front)
        front_is_present_flag, errors = front_error(
            post_processed_predicted_mask_class_labels, target_front_class_labels
        )
        if not front_is_present_flag:
            number_of_images_with_no_prediction += 1
        else:
            list_of_all_errors = np.concatenate(
                (list_of_all_errors, resolution * errors)
            )
    print(
        f"Number of images with no predicted {structure_name}: {number_of_images_with_no_prediction}"
    )
    if number_of_images_with_no_prediction >= len(post_processed_predicted_masks):
        print(
            f"Number of images with no predicted {structure_name} is equal to complete set of images. No metrics can be calculated."
        )
        return
    list_of_all_errors_without_nan = [
        err for err in list_of_all_errors if not np.isnan(err)
    ]
    list_of_all_errors_without_nan = np.array(list_of_all_errors_without_nan)
    mean = np.mean(list_of_all_errors_without_nan)
    mean_deviation = np.mean(
        np.abs(list_of_all_errors_without_nan - mean)
    )
    print(f"Mean distance error ({structure_name}): {mean}")
    print(f"Mean deviation ({structure_name}): {mean_deviation}")
    return list_of_all_errors_without_nan


def check_whether_winter_half_year(name):
    """
    Check whether the image with the file name "name" was taken during the winter half year (the hemisphere is also checked)
    :param name: file name of the image (Glacier_Date_Satellite_Resolution_Qualityfactor_Orbit(_Modality).png
    :return: boolean (true if winter half year, false otherwise)
    """
    split_name = name[:-4].split('_')
    if split_name[0] == "COL" or split_name[0] == "JAC":
        nord_halbkugel = True
    else:                                                   # Jorum, Maple, Crane, SI, DBE
        nord_halbkugel = False
    month = int(split_name[1].split('-')[1])
    if nord_halbkugel:
        if month < 4 or month > 8:
            winter = True
        else:
            winter = False
    else:
        if month < 4 or month > 8:
            winter = False
        else:
            winter = True
    return winter


def front_delineation_metric(
        complete_postprocessed_directory,
        directory_of_target_fronts,
):
    """
    Calculate front delineation statistics (MDE and mean deviation) and break them up by glacier,
    season, resolution and satellite
    :param complete_postprocessed_directory: the directory holding the complete predicted and post-processed images
    :param directory_of_target_fronts: directory holding the complete ground truth fronts
    :return:
    """
    print("\n\nCalculating distance errors ...\n\n")
    post_processed_predicted_masks = os.listdir(os.path.join(complete_postprocessed_directory))

    def _print_header(title: str) -> None:
        print("")
        print("####################################################################")
        print(f"# {title}")
        print("####################################################################")

    def _report_subset(title: str, subset: list[str]) -> None:
        if len(subset) == 0:
            return
        _print_header(title)
        print(f"Number of images: {len(subset)}")
        print("\n-- Front --")
        calculate_front_delineation_metric(
            subset,
            directory_of_target_fronts,
            mask_suffix="_front.png",
            structure_name="front",
        )

    _print_header("Results for all images")
    print(f"Number of images: {len(post_processed_predicted_masks)}")
    print("\n-- Front --")
    list_of_all_front_errors_without_nan = calculate_front_delineation_metric(
        post_processed_predicted_masks,
        directory_of_target_fronts,
        mask_suffix="_front.png",
        structure_name="front",
    )
    if list_of_all_front_errors_without_nan is not None:
        np.savetxt(
            os.path.join(complete_postprocessed_directory, os.pardir, "distance_errors_front.txt"),
            list_of_all_front_errors_without_nan,
        )

    # Season subsetting
    for season in ["winter", "summer"]:
        
        subset = []
        for file_name in post_processed_predicted_masks:
            winter = check_whether_winter_half_year(file_name)
            if (winter and season == "summer") or (not winter and season == "winter"):
                continue
            subset.append(file_name)
        _report_subset(f"Results for only images in {season}", subset)
            
    # Glacier subsetting
    for glacier in ["Mapple", "COL", "Crane", "DBE", "JAC", "Jorum", "SI"]:
        subset = [
            file_name
            for file_name in post_processed_predicted_masks
            if file_name[:-4].split('_')[0] == glacier
        ]
        _report_subset(f"Results for only images from {glacier}", subset)

    # Sensor subsetting
    for sensor in ["RSAT", "S1", "ENVISAT", "ERS", "PALSAR", "TSX/TDX"]:
        
        subset= []
        for file_name in post_processed_predicted_masks:
            if sensor == "TSX/TDX":
                if not (file_name[:-4].split('_')[2] == "TSX" or file_name[:-4].split('_')[2] == "TDX"):
                    continue
            elif not file_name[:-4].split('_')[2] == sensor:
                continue
            subset.append(file_name)
        _report_subset(f"Results for only images from {sensor}", subset)

    # Resolution subsetting
    for res in [20, 17, 12, 7, 6]:
        subset = [
            file_name
            for file_name in post_processed_predicted_masks
            if int(file_name[:-4].split('_')[3]) == res
        ]
        _report_subset(f"Results for only images with a resolution of {res}", subset)

    # Season and glacier subsetting
    for glacier in ["Mapple", "COL", "Crane", "DBE", "JAC", "Jorum", "SI"]:
        for season in ["winter", "summer"]:
            subset= []
            for file_name in post_processed_predicted_masks:
                winter = check_whether_winter_half_year(file_name)
                if file_name[:-4].split('_')[0] != glacier:
                    continue
                if (winter and season == "summer") or (not winter and season == "winter"):
                    continue
                subset.append(file_name)
            _report_subset(
                f"Results for only images in {season} and from {glacier}",
                subset,
            )

    # Sensor and glacier subsetting
    for glacier in ["Mapple", "COL", "Crane", "DBE", "JAC", "Jorum", "SI"]:
        for sensor in ["RSAT", "S1", "ENVISAT", "ERS", "PALSAR", "TSX/TDX"]:
            subset = []
            for file_name in post_processed_predicted_masks:
                if file_name[:-4].split('_')[0] != glacier:
                    continue
                if sensor == "TSX/TDX":
                    if not (file_name[:-4].split('_')[2] == "TSX" or file_name[:-4].split('_')[2] == "TDX"):
                        continue
                elif file_name[:-4].split('_')[2] != sensor:
                    continue
                subset.append(file_name)
            _report_subset(
                f"Results for only images of {sensor} and from {glacier}",
                subset,
            )
            

    # Resolution and glacier subsetting
    for glacier in ["Mapple", "COL", "Crane", "DBE", "JAC", "Jorum", "SI"]:
        for res in [20, 17, 12, 7, 6]:
            subset = [
                file_name
                for file_name in post_processed_predicted_masks
                if file_name[:-4].split('_')[0] == glacier
                and int(file_name[:-4].split('_')[3]) == res
            ]
            _report_subset(
                f"Results for only images with resolution {res} and from {glacier}",
                subset,
            )


def visualizations(complete_postprocessed_directory,
                   directory_of_target_fronts,
                   directory_of_sar_images,
                   bounding_boxes_directory,
                   visualizations_dir):
    """
    Produce visualizations showing the predicted front, the ground truth and the bounding box on the SAR image
    :param complete_postprocessed_directory: the directory holding the complete predicted and post-processed images
    :param directory_of_target_fronts: directory holding the complete ground truth fronts
    :param directory_of_sar_images: directory holding the sar images
    :param bounding_boxes_directory: directory holding the bounding boxes
    :param visualizations_dir: directory which is going to hold the visualizations
    :return:
    """
    print("\n\nCreating visualizations ...\n\n")
    post_processed_predicted_masks = [
        f
        for f in os.listdir(complete_postprocessed_directory)
        if f.endswith('_front.png')
    ]
    for file_name in post_processed_predicted_masks:
        target_dir = directory_of_target_fronts
        matching_target_file = get_matching_out_of_folder(
            file_name, target_dir, expected_suffix="_front.png"
        )
        name = os.path.split(file_name)[1][:-4]
        name = re.sub(r'_front$', '', name)
        resolution = int(name.split('_')[-3])
        if resolution < 10:
            dilation = 9
        else:
            dilation = 3

        post_processed_predicted_mask = cv2.imread(
            os.path.join(complete_postprocessed_directory, file_name).__str__(),
            cv2.IMREAD_GRAYSCALE,
        )
        target_mask = cv2.imread(
            os.path.join(target_dir, matching_target_file).__str__(),
            cv2.IMREAD_GRAYSCALE,
        )
        matching_sar_file = get_matching_out_of_folder(file_name, directory_of_sar_images)
        sar_image = cv2.imread(
            os.path.join(directory_of_sar_images, matching_sar_file).__str__(),
            cv2.IMREAD_GRAYSCALE,
        )

        predicted_mask = np.array(post_processed_predicted_mask)
        ground_truth_mask = np.array(target_mask)
        sar_image = np.array(sar_image)

        # Ensure all arrays share the same spatial dimensions
        predicted_mask, ground_truth_mask = crop_to_same_shape(predicted_mask, ground_truth_mask)
        predicted_mask, sar_image = crop_to_same_shape(predicted_mask, sar_image)
        ground_truth_mask, sar_image = crop_to_same_shape(ground_truth_mask, sar_image)

        kernel = np.ones((dilation, dilation), np.uint8)
        predicted_mask = cv2.dilate(predicted_mask, kernel, iterations=1)
        ground_truth_mask = cv2.dilate(ground_truth_mask, kernel, iterations=1)

        sar_image_rgb = skimage.color.gray2rgb(sar_image)
        sar_image_rgb = np.uint8(sar_image_rgb)

        sar_image_rgb[predicted_mask > 0] = [0, 255, 255]  # b, g, r
        sar_image_rgb[ground_truth_mask > 0] = [255, 51, 51]
        correct_prediction = np.logical_and(predicted_mask, ground_truth_mask)
        sar_image_rgb[correct_prediction > 0] = [255, 0, 255]

        # Insert Bounding Box
        matching_bounding_box_file = get_matching_out_of_folder(file_name, bounding_boxes_directory)
        with open(os.path.join(bounding_boxes_directory, matching_bounding_box_file)) as f:
            coord_file_lines = f.readlines()
        left_upper_corner_x, left_upper_corner_y = [round(float(coord)) for coord in coord_file_lines[1].split(",")]
        left_lower_corner_x, left_lower_corner_y = [round(float(coord)) for coord in coord_file_lines[2].split(",")]
        right_lower_corner_x, right_lower_corner_y = [round(float(coord)) for coord in coord_file_lines[3].split(",")]
        right_upper_corner_x, right_upper_corner_y = [round(float(coord)) for coord in coord_file_lines[4].split(",")]

        bounding_box = np.zeros((len(sar_image_rgb), len(sar_image_rgb[0])))
        if left_upper_corner_x < 0: left_upper_corner_x = 0
        if left_lower_corner_x < 0: left_lower_corner_x = 0
        if right_upper_corner_x > len(sar_image_rgb[0]): right_upper_corner_x = len(sar_image_rgb[0]) - 1
        if right_lower_corner_x > len(sar_image_rgb[0]): right_lower_corner_x = len(sar_image_rgb[0]) - 1
        if left_upper_corner_y > len(sar_image_rgb): left_upper_corner_y = len(sar_image_rgb) - 1
        if left_lower_corner_y < 0: left_lower_corner_y = 0
        if right_upper_corner_y > len(sar_image_rgb): right_upper_corner_y = len(sar_image_rgb) - 1
        if right_lower_corner_y < 0: right_lower_corner_y = 0

        bounding_box[left_upper_corner_y, left_upper_corner_x:right_upper_corner_x] = 1
        bounding_box[left_lower_corner_y, left_lower_corner_x:right_lower_corner_x] = 1
        bounding_box[left_lower_corner_y:left_upper_corner_y, left_upper_corner_x] = 1
        bounding_box[right_lower_corner_y:right_upper_corner_y, right_lower_corner_x] = 1
        bounding_box = cv2.dilate(bounding_box, kernel, iterations=1)
        sar_image_rgb[bounding_box > 0] = [255, 255, 0]

        cv2.imwrite(os.path.join(visualizations_dir, file_name), sar_image_rgb)


def main(mode, model, datamodule, patch_directory, complete_directory, complete_postprocessed_directory, visualizations_dir, data_raw_dir, save_visualizations=True):
    """
    Performance evaluation on the either the test or validation dataset.
    :param mode: Either "test" or "validate" -> on which dataset the performance of the model shall be tested
    :param model: the model to be evaluated
    :param datamodule: the datamodule for the model
    :param patch_directory: the directory in which the predicted image patches are going to be stored (this folder is deleted again after all calculations)
    :param complete_directory: the directory which is going to hold the complete predicted images
    :param complete_postprocessed_directory: the directory which is going to hold the complete predicted and post-processed images
    :param visualizations_dir: the directory which is going to hold the visualizations
    :return:
    """
    threshold_front_prob = 0.12
    # #############################################################################################################
    # EVALUATE MODEL ON GIVEN DATASET
    # #############################################################################################################
    average_loss, losses = evaluate_model_on_given_dataset(mode, model, datamodule, patch_directory)
    print(f"\nAverage Loss: {average_loss}")

    # ###############################################################################################
    # CONSTRUCT BINARIZED PREDICTIONS FROM PATCHES
    # ###############################################################################################
    print("\n\nConstructing complete predictions from patches ...\n\n")
    reconstruct_from_patches_and_binarize(
        src_directory=patch_directory,
        dst_directory=complete_directory,
        modality=model.hparams.target_masks,
        threshold_front_prob=threshold_front_prob,
    )
    shutil.rmtree(patch_directory)

    # ###############################################################################################
    # CALCULATE SEGMENTATION METRICS (IoU & Hausdorff Distance)
    # ###############################################################################################
    complete_predicted_masks = os.listdir(complete_directory)
    if model.hparams.target_masks == "fronts":
        if mode == "test":
            directory_of_target_fronts = os.path.join(data_raw_dir, "fronts", 'test')
            
        else:
            directory_of_target_fronts = os.path.join(data_raw_dir, "fronts", 'train')
            
    else:
        if mode == "test":
            directory_of_target_fronts = os.path.join(data_raw_dir, model.hparams.target_masks, 'test')
        else:
            directory_of_target_fronts = os.path.join(data_raw_dir, model.hparams.target_masks, 'train')
        

    calculate_segmentation_metrics(model.hparams.target_masks, complete_predicted_masks, complete_directory,
                                   directory_of_target_fronts)

    # ###############################################################################################
    # POST-PROCESSING
    # ###############################################################################################
    bounding_boxes_directory = os.path.join(data_raw_dir, "bounding_boxes")
    post_processing(model.hparams.target_masks, complete_predicted_masks, bounding_boxes_directory)

    # ###############################################################################################
    # CALCULATE FRONT DELINEATION METRIC (Mean distance error and mean deviation)
    # ###############################################################################################
    if mode == "test":
        directory_of_target_fronts = os.path.join(data_raw_dir, "fronts", 'test')
        
    else:
        directory_of_target_fronts = os.path.join(data_raw_dir, "fronts", 'train')
        
    front_delineation_metric(complete_postprocessed_directory, directory_of_target_fronts)

    # ###############################################################################################
    # MAKE VISUALIZATIONS
    # ###############################################################################################
    if save_visualizations:
        if mode == "test":
            directory_of_sar_images = os.path.join(data_raw_dir, "sar_images", 'test')
        else:
            directory_of_sar_images = os.path.join(data_raw_dir, "sar_images", 'train')
        visualizations(complete_postprocessed_directory,
                       directory_of_target_fronts,
                       directory_of_sar_images,
                       bounding_boxes_directory,
                       visualizations_dir)


if __name__ == "__main__":
    src = os.getcwd()

    parser = ArgumentParser(add_help=False)
    parser.add_argument('--mode', default="validate", help="Either 'validate' or 'test'.")
    parser.add_argument('--target_masks', default="fronts", help="Either 'fronts' or 'zones'.")
    parser.add_argument('--run_number', default=0,
                        help="The model's run number of the checkpoint file you want to load. "
                             "You can check the run number in the folder structure of the checkpoint"
                             "(checkpoints/..._segmentation/run_?).")
    parser.add_argument('--version_number', default=2,
                        help="The version number of the hparams file you want to load. "
                             "You can check the version number in the folder structure of the hparams file "
                             "(tb_logs/..._segmentation/run_.../log/version_?).")
    parser.add_argument('--checkpoint_file', default="epoch=210-avg_train_loss=0.43.ckpt",
                        help="The name of the checkpoint file you want to load.")
    parser.add_argument('--save_complete_predictions', action=BooleanOptionalAction, default=True,
                        help="Whether to keep the reconstructed complete predictions on disk.")
    parser.add_argument('--save_postprocessed_predictions', action=BooleanOptionalAction, default=True,
                        help="Whether to keep the post-processed complete predictions on disk.")
    parser.add_argument('--save_visualizations', action=BooleanOptionalAction, default=True,
                        help="Whether to generate and keep visualizations on disk.")
    parser.add_argument('--data_raw_parent_dir', default="../",
                        help="Where the data_raw directory lies, default: .")
    parser.add_argument('--data_parent_dir', default="../",
                        help="Where the data directory lies, default: .")
    hparams = parser.parse_args()

    assert hparams.target_masks == "fronts" or hparams.target_masks == "zones", \
        "Please set --target_masks correctly. Either 'fronts' or 'zones'."

    if hparams.target_masks == "fronts":
        assert os.path.isfile(os.path.join(src, "checkpoints", "fronts_segmentation", "run_" + str(hparams.run_number), hparams.checkpoint_file)), "Checkpoint file does not exist"
        assert os.path.isfile(os.path.join(src, "tb_logs", "fronts_segmentation", "run_" + str(hparams.run_number), "log", "version_" + str(hparams.version_number), "hparams.yaml")), "hparams file does not exist"

        model = FrontUNet.load_from_checkpoint(
            checkpoint_path=os.path.join(src, "checkpoints", "fronts_segmentation", "run_" + str(hparams.run_number), hparams.checkpoint_file),
            hparams_file=os.path.join(src, "tb_logs", "fronts_segmentation", "run_" + str(hparams.run_number), "log", "version_" + str(hparams.version_number), "hparams.yaml"),
            map_location=None
        )
        datamodule = GlacierFrontDataModule(batch_size=model.hparams.batch_size, augmentation=False, parent_dir=hparams.data_parent_dir, bright=0, wrap=0, noise=0, rotate=0, flip=0)
    else:
        assert os.path.isfile(os.path.join(src, "checkpoints", "zones_segmentation", "run_" + str(hparams.run_number), hparams.checkpoint_file)), "Checkpoint file does not exist"
        assert os.path.isfile(os.path.join(src, "tb_logs", "zones_segmentation", "run_" + str(hparams.run_number), "log", "version_" + str(hparams.version_number), 'hparams.yaml')), "hparams file does not exist"

        model = ZonesUNet.load_from_checkpoint(
            checkpoint_path=os.path.join(src, "checkpoints", "zones_segmentation", "run_" + str(hparams.run_number), hparams.checkpoint_file),
            hparams_file=os.path.join(src, "tb_logs", "zones_segmentation", "run_" + str(hparams.run_number), "log", "version_" + str(hparams.version_number), 'hparams.yaml'),
            map_location=None
        )
        datamodule = GlacierZonesDataModule(batch_size=model.hparams.batch_size, augmentation=False, parent_dir=hparams.data_parent_dir, bright=0, wrap=0, noise=0, rotate=0, flip=0)

    if hparams.mode == "test":
        result_directory_name = "test_results"
    else:
        result_directory_name = "validation_results"

    checkpoint_directory_name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.splitext(os.path.basename(hparams.checkpoint_file))[0])
    run_directory = os.path.join(src, result_directory_name, hparams.target_masks, "run_" + str(hparams.run_number))
    output_directory = os.path.join(run_directory, checkpoint_directory_name)

    patch_directory = os.path.join(output_directory, "patches")
    complete_directory = os.path.join(output_directory, "complete_images")
    complete_postprocessed_directory = os.path.join(output_directory, "complete_postprocessed_images")
    visualizations_dir = os.path.join(output_directory, "visualizations")
    data_raw_dir = os.path.join(hparams.data_raw_parent_dir, "data_raw")

    if not os.path.exists(os.path.join(src, result_directory_name)):
        os.makedirs(os.path.join(src, result_directory_name))
    if not os.path.exists(os.path.join(src, result_directory_name, hparams.target_masks)):
        os.makedirs(os.path.join(src, result_directory_name, hparams.target_masks))
    if not os.path.exists(os.path.join(src, result_directory_name, hparams.target_masks, "run_" + str(hparams.run_number))):
        os.makedirs(os.path.join(src, result_directory_name, hparams.target_masks, "run_" + str(hparams.run_number)))
    if not os.path.exists(patch_directory):
        os.makedirs(patch_directory)
    
    temp_dirs_to_cleanup = []

    if hparams.save_complete_predictions:
        if not os.path.exists(complete_directory):
            os.makedirs(complete_directory)
    else:
        temp_complete_dir = tempfile.TemporaryDirectory()
        temp_dirs_to_cleanup.append(temp_complete_dir)
        complete_directory = temp_complete_dir.name

    if hparams.save_postprocessed_predictions:
        if not os.path.exists(complete_postprocessed_directory):
            os.makedirs(complete_postprocessed_directory)
    else:
        temp_complete_postprocessed_dir = tempfile.TemporaryDirectory()
        temp_dirs_to_cleanup.append(temp_complete_postprocessed_dir)
        complete_postprocessed_directory = temp_complete_postprocessed_dir.name

    if hparams.save_visualizations:
        if not os.path.exists(visualizations_dir):
            os.makedirs(visualizations_dir)


    log_file_path = os.path.join(output_directory, "log.txt")
    log_file = open(log_file_path, "w")
    original_print = builtins.print

    def print_and_log(*args, **kwargs):
        original_print(*args, **kwargs)
        original_print(*args, file=log_file, **kwargs)
        log_file.flush()

    builtins.print = print_and_log

    try:
        main(hparams.mode, model, datamodule, patch_directory, complete_directory,
             complete_postprocessed_directory, visualizations_dir, data_raw_dir, hparams.save_visualizations)
    finally:
        for temp_dir in temp_dirs_to_cleanup:
            temp_dir.cleanup()
        builtins.print = original_print
        log_file.close()
