import os
import cv2
from torchvision.transforms import functional as F
import torch

# This script computes the mean and standard deviation of the training data images.
# These values can be used for normalizing the images during training.
# x_normalized = (x - mean) / std_dev
# mean = 1/N * sum(x_i)
# std_dev = sqrt(1/N * sum((x_i - mean)^2))
# N = number of pixels in all training images
# i = index of a pixel in all training images

# This step is important for these reasons:
# 1. It standardizes values scale;
# 2. It speeds up convergence; 
# 3. It enhances generalization.

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.getcwd()), "data_raw", "sar_images", "train")
    imgs = os.listdir(path)
    list_imgs = []
    for img_name in imgs:
        image = cv2.imread(os.path.join(path, img_name).__str__(), cv2.IMREAD_GRAYSCALE)
        image = F.to_tensor(image)
        list_imgs.append(torch.flatten(image))
    torch_imgs = torch.cat(list_imgs, dim=0)
    mean = torch.mean(torch_imgs)
    std_dev = torch.std(torch_imgs)
    print(f"Mean of training data: {mean}")
    print(f"Standard deviation of training data: {std_dev}")
