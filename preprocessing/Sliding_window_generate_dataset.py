# PREPROCESSING STEP 2:

import glob
from pathlib import Path
import os
import PIL
import cv2
import numpy as np
from configs.config import cfg
import time
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

print("Starting Sliding Window process...")

start = time.time()

base_dir = Path(__file__).resolve().parents[2] / "CAFFE"

ip = base_dir / "sar_images" / "center_crop"
mp = base_dir / "zones" / "center_crop"
mfp = base_dir / "fronts" / "center_crop"

ipv = base_dir / "sar_images" / "center_crop_val"
mpv = base_dir / "zones" / "center_crop_val"
mfv = base_dir / "fronts" / "center_crop_val"

ipt = base_dir / "sar_images" / "center_crop_test"
mpt = base_dir / "zones" / "center_crop_test"
mft = base_dir / "fronts" / "center_crop_test"

save_path_context = base_dir / "sar_images" / "sliding_window"
save_path_context_mask = base_dir / "zones" / "sliding_window"
save_path_context_mask_f = base_dir / "fronts" / "sliding_window"

images = sorted(glob.glob(os.path.join(ip, '*.png')))
masks = sorted(glob.glob(os.path.join(mp, '*.png')))
masksf = sorted(glob.glob(os.path.join(mfp, '*.png')))

totensor = transforms.Compose([
    transforms.ToTensor(),
])

if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)
if not os.path.exists(save_path_context_mask_f):
    os.makedirs(save_path_context_mask_f)

target_size = 288
interval = target_size * 2

print("Sliding Training sar_images...")

for i in range(len(images)):
    image = Image.open(images[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0]

    toimage = totensor(image)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context, suffix + '_{:03d}.png'.format(id)))
                id += 1

print("Complete!!")

print("Sliding Training zones...")

for i in range(len(masks)):
    mask = Image.open(masks[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0][0:-6]
    toimage = totensor(mask)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context_mask, suffix + '_{:03d}'.format(id) + '_zones_NA.png'))
                id += 1

print("Complete!!")

print("Sliding Training fronts...")

for i in range(len(masksf)):
    image = Image.open(masksf[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0]

    toimage = totensor(image)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context_mask_f, suffix + '_{:03d}.png'.format(id)))
                id += 1

print("Complete!!")

imagesv = sorted(glob.glob(os.path.join(ipv, '*.png')))
masksv = sorted(glob.glob(os.path.join(mpv, '*.png')))
masksfv = sorted(glob.glob(os.path.join(mfv, '*.png')))


save_path_context = base_dir / "sar_images" / "sliding_window_valid"
save_path_context_mask = base_dir / "zones" / "sliding_window_valid"
save_path_context_mask_f = base_dir / "fronts" / "sliding_window_valid"

totensor = transforms.Compose([
    transforms.ToTensor(),
])

if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)
if not os.path.exists(save_path_context_mask_f):
    os.makedirs(save_path_context_mask_f)

print("Sliding Validating sar_images...")

for i in range(len(imagesv)):
    image = Image.open(imagesv[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0]
    toimage = totensor(image)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context, suffix + '_{:03d}.png'.format(id)))
                id += 1

print("Complete!!")

print("Sliding Validating zones...")

for i in range(len(masksv)):
    mask = Image.open(masksv[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0][0:-6]
    toimage = totensor(mask)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context_mask, suffix + '_{:03d}'.format(id) + '_zones_NA.png'))
                id += 1

print("Complete!!")

print("Sliding Validating fronts...")

for i in range(len(masksfv)):
    image = Image.open(masksfv[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0]
    toimage = totensor(image)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context_mask_f, suffix + '_{:03d}.png'.format(id)))
                id += 1

print("Complete!!")

imagest = sorted(glob.glob(os.path.join(ipt, '*.png')))
maskst = sorted(glob.glob(os.path.join(mpt, '*.png')))
masksft = sorted(glob.glob(os.path.join(mft, '*.png')))

save_path_context = base_dir / "sar_images" / "sliding_window_test"
save_path_context_mask = base_dir / "zones" / "sliding_window_test"
save_path_context_mask_f = base_dir / "fronts" / "sliding_window_test"

totensor = transforms.Compose([
    transforms.ToTensor(),
])

if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)
if not os.path.exists(save_path_context_mask_f):
    os.makedirs(save_path_context_mask_f)    

print("Sliding Testing sar_images...")

for i in range(len(imagest)):
    image = Image.open(imagest[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0]
    toimage = totensor(image)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context, suffix + '_{:03d}.png'.format(id)))
                id += 1

print("Complete!!")

print("Sliding Testing zones...")

for i in range(len(maskst)):
    mask = Image.open(maskst[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0][0:-6]
    toimage = totensor(mask)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context_mask, suffix + '_{:03d}'.format(id) + '_zones_NA.png'))
                id += 1

print("Complete!!")

print("Sliding Testing fronts...")

for i in range(len(masksft)):
    image = Image.open(masksft[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0]
    toimage = totensor(image)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context_mask_f, suffix + '_{:03d}.png'.format(id)))
                id += 1

print("Complete!!")

end = time.time()

minutes = (end - start)/60

print(f"\nSliding Window completed in about {minutes:.0f} minutes.")