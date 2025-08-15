import os
import numpy as np
from glob import glob
import torch
import cv2
from torch.utils.data import Dataset
import logging
import time
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import torchvision
import tormentor
import warnings


class BasDataset(Dataset):
    def __init__(self, dir_img_target, dir_img_context, dir_mask_target, dir_mask_context, scale=1,
                 mask_suffix=''):
        self.imgs_target = dir_img_target
        self.imgs_context = dir_img_context
        self.masks_target = dir_mask_target
        self.masks_context = dir_mask_context
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(dir_img_target)]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def Fliph(self, target_image, context_image, target_mask, context_mask):
        target = torchvision.transforms.functional.hflip(target_image)
        mask_target = torchvision.transforms.functional.hflip(target_mask)
        context = torchvision.transforms.functional.hflip(context_image)
        mask_context = torchvision.transforms.functional.hflip(context_mask)
        return target, context, mask_target, mask_context

    def Flipv(self, target_image, context_image, target_mask, context_mask):
        target = torchvision.transforms.functional.vflip(target_image)
        mask_target = torchvision.transforms.functional.vflip(target_mask)
        context = torchvision.transforms.functional.vflip(context_image)
        mask_context = torchvision.transforms.functional.vflip(context_mask)
        return target, context, mask_target, mask_context

    def Rotate(self, target_image, context_image, target_mask, context_mask):
        random = np.random.randint(0, 3)
        angle = 90
        if random == 1:
            angle = 180
        elif random == 2:
            angle = 270
            
        # Rotate images with bilinear interpolation
        target = torchvision.transforms.functional.rotate(
            target_image,
            angle=angle,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        context = torchvision.transforms.functional.rotate(
            context_image,
            angle=angle,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )

        # Masks require nearest interpolation to maintain discrete values
        mask_target = torchvision.transforms.functional.rotate(
            target_mask.unsqueeze(0),
            angle=angle,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )
        mask_context = torchvision.transforms.functional.rotate(
            context_mask.unsqueeze(0),
            angle=angle,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )

        return target, context, mask_target.squeeze(0), mask_context.squeeze(0)

    def preprocess(self, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        img_trans = cv2.cvtColor(img_nd, cv2.COLOR_BGR2GRAY)
        img_trans = np.expand_dims(img_trans, axis=0)

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans
    
    def mask_preprocess(self, pil_img, scale):
        img = np.array(pil_img)
        if img.ndim == 3:
            img = img[:, :, 0]
        mask = (img > 0).astype(np.uint8) 
        return mask
    """
    def mask_preprocess(self, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

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
    def __getitem__(self, i):
        idx = self.ids[i]
        imgs_target = glob(self.imgs_target + '/' + idx + '.*')
        imgs_context = glob(self.imgs_context + '/' + idx + '.*')
        masks_target = glob(self.masks_target + '/' + idx + self.mask_suffix + '.*')
        masks_context = glob(self.masks_context + '/' + idx + self.mask_suffix + '.*')

        suffix = idx + self.mask_suffix

        mask_target = Image.open(masks_target[0])
        target = Image.open(imgs_target[0])
        mask_context = Image.open(masks_context[0])
        context = Image.open(imgs_context[0])

        assert len(masks_target) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_target}'
        assert len(imgs_target) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {target}'

        assert target.size == mask_target.size, \
            f'Image and mask {idx} should be the same size, but are {target.size} and {mask_target.size}'

        target = self.preprocess(target, self.scale)
        context = self.preprocess(context, self.scale)
        target = torch.from_numpy(target).type(torch.FloatTensor)
        context = torch.from_numpy(context).type(torch.FloatTensor)

        mask_target = self.mask_preprocess(mask_target, self.scale)
        mask_context = self.mask_preprocess(mask_context, self.scale)
        mask_target = torch.from_numpy(mask_target).type(torch.int64)
        mask_context = torch.from_numpy(mask_context).type(torch.int64)

        """
        Computed the target-mask pixel sum in __getitem__, logging the value and warning/asserting if no foreground pixels are present to catch empty datasets early
        """
        """
        mask_sum = int(mask_target.sum())
        logging.debug("mask_target.sum() for %s: %d", idx, mask_sum)
        if mask_sum == 0:
            warnings.warn(
                f"Mask for {idx} has no foreground pixels.",
                RuntimeWarning,
            )
            logging.warning(f"Mask for {idx} has no foreground pixels.")
        assert mask_sum != 0, f"Mask for {idx} has no foreground pixels."
        """
        return {
            'image_target': target,
            'mask_target': mask_target,
            'image_context': context,
            'mask_context': mask_context,
            'suffix': suffix,
        }
