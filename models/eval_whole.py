import torch
import torch.nn.functional as F
import os
import sys
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.config import cfg
from models.dice_loss import dice_coeff
from models.loss import make_one_hot
import logging
import glob
import tempfile

# ``torchmetrics`` has renamed ``IoU`` to ``JaccardIndex`` in newer
# versions. Import lazily so that the evaluation code can adapt to the
# available API and still run even if ``torchmetrics`` is not installed
# at import time.
try:  # pragma: no cover - defensive import
    import torchmetrics  # type: ignore
except Exception:  # pragma: no cover
    torchmetrics = None

import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
import sklearn.metrics
from torchvision.utils import save_image


def _get_iou_metric(num_classes, reduction=None):
    """Return an IoU/Jaccard metric compatible with different torchmetrics versions."""
    if torchmetrics is None:  # pragma: no cover - runtime safety
        raise RuntimeError("torchmetrics is required for IoU computation")

    if hasattr(torchmetrics, "IoU"):
        kwargs = {"num_classes": num_classes, "absent_score": 1}
        if reduction is not None:
            kwargs["reduction"] = reduction
        return torchmetrics.IoU(**kwargs)

    # ``IoU`` was renamed to ``JaccardIndex`` in newer torchmetrics releases.
    from torchmetrics import JaccardIndex
    task = "binary" if num_classes == 2 else "multiclass"
    average = None if reduction == "none" else reduction
    return JaccardIndex(task=task, num_classes=num_classes, average=average)

"""
def mask_preprocess(pil_img):
    img_nd = np.array(pil_img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    img_trans = img_nd.transpose((2, 0, 1))
    img_trans = img_trans[0, :, :]
    C, _ = img_trans.shape
    mask = np.ones([C, C]) * 15
    na_area = np.where(img_trans == 0)
    stone = np.where(img_trans == 63)
    stones = np.where(img_trans == 64)
    glacier = np.where(img_trans == 127)
    ocean_ice = np.where(img_trans == 254)
    mask[na_area] = 0
    mask[stone] = 1
    mask[stones] = 1
    mask[glacier] = 2
    mask[ocean_ice] = 3

    return mask
"""

def mask_preprocess(pil_img):
    """Convert a PIL mask image to a binary NumPy array."""
    img = np.array(pil_img)
    if img.ndim == 3:
        img = img[:, :, 0]
    mask = (img > 0).astype(np.uint8)
    return mask


"""
def whole_preprocess(pil_img):
    img_nd = np.array(pil_img)
    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    img_trans = img_nd[:, :, 0]
    H, W = img_trans.shape

    mask = np.ones([H, W]) * 15
    na_area = np.where(img_trans == 0)
    stone = np.where(img_trans == 63)
    stones = np.where(img_trans == 64)
    glacier = np.where(img_trans == 127)
    ocean_ice = np.where(img_trans == 254)
    mask[na_area] = 0
    mask[stone] = 1
    mask[stones] = 1
    mask[glacier] = 2
    mask[ocean_ice] = 3

    return mask
"""

def whole_preprocess(pil_img):
    """Convert a PIL image of the whole mask to a binary NumPy array."""
    img = np.array(pil_img)
    if img.ndim == 3:
        img = img[:, :, 0]
    mask = (img > 0).astype(np.uint8)
    return mask

""""
def mask_to_image(pre_mask, save_path, suffix):
    c, h, w = pre_mask.shape
    out = np.zeros((3, h, w), dtype=np.uint8)
    na_area = np.where(pre_mask.to('cpu') == 0)
    stone = np.where(pre_mask.to('cpu') == 1)
    glacier = np.where(pre_mask.to('cpu') == 2)
    ocean_ice = np.where(pre_mask.to('cpu') == 3)
    out[na_area] = 0
    out[stone] = 64
    out[glacier] = 127
    out[ocean_ice] = 254
    out[1] = out[0]
    out[2] = out[0]
    out = out.transpose((1, 2, 0))
    cv2.imwrite(os.path.join(save_path, suffix + '.png'), out, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
"""
def mask_to_image(pre_mask, save_path=None, suffix=None):
    """Convert a prediction mask to a color image.

    The function is tolerant to different tensor shapes coming from the model
    (``(N,1,H,W)``, ``(N,H,W)`` or ``(H,W)``).  It always converts the mask to a
    2-D array before performing the color mapping to avoid indexing errors.
    """
    if pre_mask.ndim == 4:
        pre_mask = pre_mask.squeeze(1)
    
    if pre_mask.ndim == 3:
        pre_mask = pre_mask.squeeze(0)

    mask_cpu = pre_mask.to("cpu").numpy()
    h, w = mask_cpu.shape

    out = np.zeros((3, h, w), dtype=np.uint8)


    if mask_cpu.max() <= 1:
        out[:, mask_cpu == 1] = 255
    else:
        out[:, mask_cpu == 1] = 64
        out[:, mask_cpu == 2] = 127
        out[:, mask_cpu == 3] = 254

    out = out.transpose((1, 2, 0))
    if save_path is not None and suffix is not None:
        cv2.imwrite(os.path.join(save_path, suffix + '.png'), out, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    return out



def eval_net(net, loader, device):
    net.eval()
    mask_type = torch.float32 if net.n_classes <= 2 else torch.int64
    n_val = len(loader)
    iou_ratio = 0
    success_ratio = 0
    tmp_dir = os.getenv('TMPDIR', tempfile.gettempdir())
    test_save = os.path.join(tmp_dir, 'patch_save')
    if not os.path.exists(test_save):
        os.makedirs(test_save)
    comparison_dir = os.path.join(cfg.output_path_predictions, 'comparisons')
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    with tqdm(total=n_val, desc='Round', unit='img') as pbar:
        for batch in loader:
            imgs_target = batch['image_target']
            imgs_context = batch['image_context']
            true_masks_target = batch['mask_target']
            suffix = batch['suffix'][0]

            imgs_target = imgs_target.to(device=device, dtype=torch.float32)
            imgs_context = imgs_context.to(device=device, dtype=torch.float32)
            true_masks_target = true_masks_target.to(device=device, dtype=mask_type)

            with torch.no_grad():
                masks_pred = net(imgs_context, imgs_target)

            if net.n_classes > 2:
                IOU = _get_iou_metric(num_classes=4)

                prd_target = F.log_softmax(masks_pred[1], dim=1)
                prd_target = torch.argmax(prd_target, dim=1)

                pred_img = mask_to_image(prd_target, test_save, suffix)
                gt_img = mask_to_image(true_masks_target)
                comparison = np.concatenate([gt_img, pred_img], axis=1)
                cv2.imwrite(os.path.join(comparison_dir, suffix + '.png'), comparison)

                c, h, w = true_masks_target.shape
                iou = IOU(prd_target.cpu().detach(), true_masks_target.cpu().detach())
                iou_ratio += iou
                correct = torch.where(prd_target == true_masks_target)
                new = torch.zeros([c, h, w])
                new[correct] = 1
                success_ratio += torch.sum(new) / (c * h * w)

            else:
                IOU = _get_iou_metric(num_classes=2)
                pred = F.softmax(masks_pred[1], dim=1)
                pred = torch.argmax(pred, dim=1)
                success_ratio += dice_coeff(pred.float().cpu().detach(), true_masks_target.cpu().detach()).item()

                pred = (pred > 0.5).float()
                success_ratio += dice_coeff(pred.cpu().detach(), true_masks_target.cpu().detach()).item()
                iou = IOU(pred.cpu().detach(), true_masks_target.type(torch.int64).cpu().detach())
                iou_ratio += iou
                pred_img = mask_to_image(pred.unsqueeze(1))
                gt_img = mask_to_image(true_masks_target)
                comparison = np.concatenate([gt_img, pred_img], axis=1)
                cv2.imwrite(os.path.join(comparison_dir, suffix + '.png'), comparison)
            pbar.update()

    groundtruth = os.path.join(tmp_dir, 'ori')
    whole_save = os.path.join(tmp_dir, 'whole_save')
    
    masks_ground = sorted(glob.glob(os.path.join(groundtruth, '*.png')))

    if len(masks_ground) == 0:
        mean_success = success_ratio / max(n_val, 1)
        mean_iou = iou_ratio / max(n_val, 1)
        if torch.is_tensor(mean_iou):
            mean_iou = mean_iou.item()
        net.train()
        return mean_success, mean_iou, mean_iou

    totensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    if not os.path.exists(whole_save):
        os.makedirs(whole_save)

    IOU = _get_iou_metric(num_classes=4 if net.n_classes > 2 else 2, reduction='none')
    iou_ratio = 0
    success_ratio = 0
    for i in range(len(masks_ground)):
        ground = masks_ground[i]
        suffix = ground.split('/')[-1].split('.')[0][0:-6]
        img = Image.open(ground).convert("RGB")
        W, H = img.size
        HH = H // 288 + 1
        WW = W // 288 + 1
        length = 288
        test = sorted(glob.glob(os.path.join(test_save, suffix + '*.png')))
        all = []
        for j in range(len(test)):
            ti = Image.open(test[j]).convert("RGB")
            all.append(ti)

        whole = Image.new('RGB', (WW * length, HH * length))

        for k in range(len(all)):
            whole.paste(all[k],
                        (length * (k % WW), length * (k // WW), length * (k % WW + 1), length * (k // WW + 1)))

        whole = transforms.CenterCrop((H, W))(whole)

        wholes = totensor(whole)
        save_image(wholes, os.path.join(whole_save, suffix + '.png'))

        img = whole_preprocess(img)
        whole = whole_preprocess(whole)

        h, w = whole.shape

        iou = IOU(torch.from_numpy(whole).type(torch.int64), torch.from_numpy(img).type(torch.int64))
        iou_ratio += iou
        correct = np.where(whole == img)

        new = np.zeros([h, w])
        new[correct] = 1
        success_ratio += np.sum(new) / (h * w)

    iou_ratio = iou_ratio / len(masks_ground)
    ave_iou = sum(iou_ratio) / len(iou_ratio)
    success_ratio = success_ratio / len(masks_ground)

    net.train()
    return success_ratio, ave_iou, iou_ratio
