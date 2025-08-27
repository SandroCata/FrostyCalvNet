from data_processing.glacier_data import GlacierDataModule
from data_processing.glacier_data import GlacierDataset
import torch
from torchvision.transforms import functional as F
import numpy as np
import os
import cv2


class ToTensorFront(object):
    def __call__(self, image, target_front, target_near):
        image = F.to_tensor(image)
        front = torch.as_tensor(np.array(target_front))
        near = torch.as_tensor(np.array(target_near))
        # value for front=255, background=0
        front[front == 255] = 1
        front[front == 0] = 0
        near[near == 255] = 1
        near[near == 0] = 0
        target = torch.stack([front, near])
        # class ids for front=1, background=0
        return image, target


class GlacierFrontDataset(GlacierDataset):
    def __init__(self, mode, augmentation, parent_dir, bright, wrap, noise, rotate, flip):
        super(GlacierFrontDataset, self).__init__(target="fronts",
                                                  mode=mode,
                                                  augmentation=augmentation,
                                                  parent_dir=parent_dir,
                                                  bright=bright,
                                                  wrap=wrap,
                                                  noise=noise,
                                                  rotate=rotate,
                                                  flip=flip)
        
        # path to near-front masks
        self.near_targets_path = os.path.join(parent_dir, "data", "near_fronts", mode)

    def custom_to_tensor(self, image, target):
        mask_front, mask_near = target
        to_tensor = ToTensorFront()
        image, mask = to_tensor(image=image, target_front=mask_front, target_near=mask_near)
        return image, mask
    
    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        label_name = self.labels[idx]
        # Labels for fronts are stored with the image name plus the ``_front``
        # suffix. Near-front masks are stored separately and use the
        # ``_near_front`` suffix. Match the front label and derive the expected
        # near-front label from the image name.
        expected_front = img_name.replace(".png", "_front.png")
        expected_near = img_name.replace(".png", "_near_front.png")
        assert (
            label_name == expected_front
        ), f"image and label name don't match. Image name: {img_name}. Label name: {label_name}"

        image = cv2.imread(
            os.path.join(self.images_path, img_name).__str__(), cv2.IMREAD_GRAYSCALE
        )
        mask_front_path = os.path.join(self.targets_path, label_name).__str__()
        mask_near_path = os.path.join(self.near_targets_path, expected_near).__str__()
        mask_front = cv2.imread(mask_front_path, cv2.IMREAD_GRAYSCALE)
        mask_near = cv2.imread(mask_near_path, cv2.IMREAD_GRAYSCALE)
        if mask_front is None:
            raise FileNotFoundError(f"Front mask not found: {mask_front_path}")
        if mask_near is None:
            raise FileNotFoundError(f"Near-front mask not found: {mask_near_path}")
        x, y = self.transform(image, (mask_front, mask_near))
        return x, y, img_name, label_name


class GlacierFrontDataModule(GlacierDataModule):

    def __init__(self, batch_size, augmentation, parent_dir, bright, wrap, noise, rotate, flip):
        super(GlacierFrontDataModule, self).__init__(batch_size=batch_size,
                                                     target="fronts",
                                                     augmentation=augmentation,
                                                     parent_dir=parent_dir,
                                                     bright=bright,
                                                     wrap=wrap,
                                                     noise=noise,
                                                     rotate=rotate,
                                                     flip=flip)

    def setup(self, stage=None):
        if stage == 'test' or stage is None:
            self.glacier_test = GlacierFrontDataset(mode='test', augmentation=self.augmentation,
                                                    parent_dir=self.parent_dir,
                                                    bright=self.bright,
                                                    wrap=self.wrap,
                                                    noise=self.noise,
                                                    rotate=self.rotate,
                                                    flip=self.flip)
        if stage == 'fit' or stage is None:
            self.glacier_train = GlacierFrontDataset(mode='train', augmentation=self.augmentation,
                                                     parent_dir=self.parent_dir,
                                                     bright=self.bright,
                                                     wrap=self.wrap,
                                                     noise=self.noise,
                                                     rotate=self.rotate,
                                                     flip=self.flip)
            self.glacier_val = GlacierFrontDataset(mode='val', augmentation=self.augmentation,
                                                   parent_dir=self.parent_dir,
                                                   bright=self.bright,
                                                   wrap=self.wrap,
                                                   noise=self.noise,
                                                   rotate=self.rotate,
                                                   flip=self.flip)
