# PREPROCESSING STEP 1:

#This script:

# - pads the images and masks so that their dimensions are a multiple of target_size (=288)

# - does not crop, but centralizes and uniformizes the size

# - generates "center-aligned" images and masks ready to be subsequently cut into patches by 'Sliding_window_generate_dataset.py' (PREPROCESSING STEP 2)

from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import glob
import os

base_dir = Path(__file__).resolve().parents[2] / "CAFFE"

im = base_dir / "sar_images" / "train"
ma = base_dir / "zones" / "train"

imv = base_dir / "sar_images" / "valid"
mav = base_dir / "zones" / "valid"

imt = base_dir / "sar_images" / "test"
mat = base_dir / "zones" / "test"

save_train_sar = base_dir / "sar_images" / "center_crop"
save_train_zones = base_dir / "zones" / "center_crop"

save_valid_sar = base_dir / "sar_images" / "center_crop_val"
save_valid_zones = base_dir / "zones" / "center_crop_val"

save_test_sar = base_dir / "sar_images" / "center_crop_test"
save_test_zones = base_dir / "zones" / "center_crop_test"

for path in [
    save_train_sar,
    save_train_zones,
    save_valid_sar,
    save_valid_zones,
    save_test_sar,
    save_test_zones,
]:
    os.makedirs(path, exist_ok=True)

ims = sorted(glob.glob(os.path.join(im, '*.png')))
mas = sorted(glob.glob(os.path.join(ma, '*.png')))

imsv = sorted(glob.glob(os.path.join(imv, '*.png')))
masv = sorted(glob.glob(os.path.join(mav, '*.png')))

imst = sorted(glob.glob(os.path.join(imt, '*.png')))
mast = sorted(glob.glob(os.path.join(mat, '*.png')))

target_size = 288
context_size = target_size * 2

print("Cropping of Training sar_images...")

for img_path in ims:
    img = Image.open(img_path)
    suffix = Path(img_path).name
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

    #save and move in dest folder
    save_image(pil_img, save_train_sar / suffix)

print("Complete!!")

print("Cropping of Training zones...")

for mask_path in mas:
    img = Image.open(mask_path)
    suffix = Path(mask_path).name

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
    
    save_image(pil_img, save_train_zones / suffix)

print("Complete!!")


print("Cropping of Validation sar_images...")

for img_path in imsv:
    img = Image.open(img_path)
    suffix = Path(img_path).name

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
    
    save_image(pil_img, save_valid_sar / suffix)

print("Complete!!")


print("Cropping of Validation zones...")

for mask_path in masv:
    img = Image.open(mask_path)
    suffix = Path(mask_path).name

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
    
    save_image(pil_img, save_valid_zones / suffix)

print("Complete!!")


print("Cropping of Testing sar_images...")

for img_path in imst:
    img = Image.open(img_path)
    suffix = Path(img_path).name

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

    save_image(pil_img, save_test_sar / suffix)

print("Complete!!")

print("Cropping of Testing zones...")


for mask_path in mast:
    img = Image.open(mask_path)
    suffix = Path(mask_path).name

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

    save_image(pil_img, save_test_zones / suffix)

print("Complete!!")

print("Center Crop completed!!")