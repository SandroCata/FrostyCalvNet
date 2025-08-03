# PREPROCESSING STEP 1:

#This script:

# - pads the images and masks so that their dimensions are a multiple of target_size (=288)

# - does not crop, but centralizes and uniformizes the size

# - generates "center-aligned" images and masks ready to be subsequently cut into patches by 'Sliding_window_generate_dataset.py' (PREPROCESSING STEP 2)

from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import glob
import os

im = '../CAFFE/sar_images/train'
ma = '../CAFFE/zones/train'

imv = '../CAFFE/sar_images/valid'
mav = '../CAFFE/zones/valid'

imt = '../CAFFE/sar_images/test'
mat = '../CAFFE/zones/test'

images = '../CAFFE/sar_images/center_crop'
masks = '../CAFFE/zones/center_crop'

imagesv = '../CAFFE/sar_images/center_crop_valid'
masksv = '../CAFFE/zones/center_crop_valid'

imagest = '../CAFFE/sar_images/center_crop_test'
maskst = '../CAFFE/zones/center_crop_test'

if not os.path.exists(images):
    os.makedirs(images)
if not os.path.exists(masks):
    os.makedirs(masks)
if not os.path.exists(imagesv):
    os.makedirs(imagesv)
if not os.path.exists(masksv):
    os.makedirs(masksv)
if not os.path.exists(imagest):
    os.makedirs(imagest)
if not os.path.exists(maskst):
    os.makedirs(maskst)

ims = sorted(glob.glob(os.path.join(im, '*.png')))
mas = sorted(glob.glob(os.path.join(ma, '*.png')))

imsv = sorted(glob.glob(os.path.join(imv, '*.png')))
masv = sorted(glob.glob(os.path.join(mav, '*.png')))

imst = sorted(glob.glob(os.path.join(imt, '*.png')))
mast = sorted(glob.glob(os.path.join(mat, '*.png')))

target_size = 288
context_size = target_size * 2

for i in range(len(ims)):
    img = Image.open(ims[i])
    suffix = ims[i].split('/')[-1]
    W, H = img.size

    WW = (W // target_size) + 2
    HH = (H // target_size) + 2
    # pil_img = transforms.CenterCrop((HH * target_size, WW * target_size))(img)
    crop_height, crop_width = (HH * target_size, WW * target_size)
    image_width, image_height = img.size
    padding_ltrb = [
        int(round((crop_width - image_width) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height) / 2.0)) if crop_height > image_height else 0,
        int(round((crop_width - image_width + 1) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height + 1) / 2.0)) if crop_height > image_height else 0,
    ]
    pil_img = transforms.Pad(padding_ltrb, fill=0)(img)

    pil_img = transforms.ToTensor()(pil_img)
    save_image(pil_img, os.path.join(images, suffix))

for i in range(len(mas)):
    img = Image.open(mas[i])
    suffix = mas[i].split('/')[-1]
    W, H = img.size

    WW = (W // target_size) + 2
    HH = (H // target_size) + 2
    # pil_img = transforms.CenterCrop((HH * target_size, WW * target_size))(img)
    crop_height, crop_width = (HH * target_size, WW * target_size)
    image_width, image_height = img.size
    padding_ltrb = [
        int(round((crop_width - image_width) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height) / 2.0)) if crop_height > image_height else 0,
        int(round((crop_width - image_width + 1) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height + 1) / 2.0)) if crop_height > image_height else 0,
    ]
    pil_img = transforms.Pad(padding_ltrb, fill=0)(img)

    pil_img = transforms.ToTensor()(pil_img)
    save_image(pil_img, os.path.join(masks, suffix))

for i in range(len(imsv)):
    img = Image.open(imsv[i])
    suffix = imsv[i].split('/')[-1]
    W, H = img.size

    WW = (W // target_size) + 2
    HH = (H // target_size) + 2
    # pil_img = transforms.CenterCrop((HH * target_size, WW * target_size))(img)
    crop_height, crop_width = (HH * target_size, WW * target_size)
    image_width, image_height = img.size
    padding_ltrb = [
        int(round((crop_width - image_width) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height) / 2.0)) if crop_height > image_height else 0,
        int(round((crop_width - image_width + 1) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height + 1) / 2.0)) if crop_height > image_height else 0,
    ]
    pil_img = transforms.Pad(padding_ltrb, fill=0)(img)

    pil_img = transforms.ToTensor()(pil_img)
    save_image(pil_img, os.path.join(imagesv, suffix))

for i in range(len(masv)):
    img = Image.open(masv[i])
    suffix = masv[i].split('/')[-1]
    W, H = img.size

    WW = (W // target_size) + 2
    HH = (H // target_size) + 2
    # pil_img = transforms.CenterCrop((HH * target_size, WW * target_size))(img)
    crop_height, crop_width = (HH * target_size, WW * target_size)
    image_width, image_height = img.size
    padding_ltrb = [
        int(round((crop_width - image_width) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height) / 2.0)) if crop_height > image_height else 0,
        int(round((crop_width - image_width + 1) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height + 1) / 2.0)) if crop_height > image_height else 0,
    ]
    pil_img = transforms.Pad(padding_ltrb, fill=0)(img)

    pil_img = transforms.ToTensor()(pil_img)
    save_image(pil_img, os.path.join(masksv, suffix))

for i in range(len(imst)):
    img = Image.open(imst[i])
    suffix = imst[i].split('/')[-1]
    W, H = img.size

    WW = (W // target_size) + 2
    HH = (H // target_size) + 2
    # pil_img = transforms.CenterCrop((HH * target_size, WW * target_size))(img)
    crop_height, crop_width = (HH * target_size, WW * target_size)
    image_width, image_height = img.size
    padding_ltrb = [
        int(round((crop_width - image_width) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height) / 2.0)) if crop_height > image_height else 0,
        int(round((crop_width - image_width + 1) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height + 1) / 2.0)) if crop_height > image_height else 0,
    ]
    pil_img = transforms.Pad(padding_ltrb, fill=0)(img)

    pil_img = transforms.ToTensor()(pil_img)
    save_image(pil_img, os.path.join(imagest, suffix))

for i in range(len(mast)):
    img = Image.open(mast[i])
    suffix = mast[i].split('/')[-1]
    W, H = img.size

    WW = (W // target_size) + 2
    HH = (H // target_size) + 2
    # pil_img = transforms.CenterCrop((HH * target_size, WW * target_size))(img)
    crop_height, crop_width = (HH * target_size, WW * target_size)
    image_width, image_height = img.size
    padding_ltrb = [
        int(round((crop_width - image_width) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height) / 2.0)) if crop_height > image_height else 0,
        int(round((crop_width - image_width + 1) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height + 1) / 2.0)) if crop_height > image_height else 0,
    ]
    pil_img = transforms.Pad(padding_ltrb, fill=0)(img)

    pil_img = transforms.ToTensor()(pil_img)
    save_image(pil_img, os.path.join(maskst, suffix))
