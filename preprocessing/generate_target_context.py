# PREPROCESSING STEP 3:

import glob
from pathlib import Path
import os
import PIL
import numpy as np3
from configs.config import cfg
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import time

print("Starting target/context process...")

start = time.time()

base_dir = Path(__file__).resolve().parents[2] / "CAFFE"

ti = base_dir / "sar_images" / "sliding_window"
mi = base_dir / "zones" / "sliding_window"
fi = base_dir / "fronts" / "sliding_window"

images = sorted(glob.glob(os.path.join(ti, '*.png')))
masks = sorted(glob.glob(os.path.join(mi, '*.png')))
fronts = sorted(glob.glob(os.path.join(fi, '*.png')))

save_path_target = base_dir / "sar_images" / "target_imgs"
save_path_context = base_dir / "sar_images" / "context_imgs"

save_path_target_mask = base_dir / "zones" / "target_imgs"
save_path_context_mask = base_dir / "zones" / "context_imgs"

save_path_target_mask_f = base_dir / "fronts" / "target_imgs"
save_path_context_mask_f = base_dir / "fronts" / "context_imgs"

size = 288

target = transforms.Compose([
    transforms.CenterCrop((size, size)),
    transforms.ToTensor(),
])

context_images = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
])

context_masks = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor(),
])

if not os.path.exists(save_path_target):
    os.makedirs(save_path_target)
if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_target_mask):
    os.makedirs(save_path_target_mask)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)
if not os.path.exists(save_path_target_mask_f):
    os.makedirs(save_path_target_mask_f)
if not os.path.exists(save_path_context_mask_f):
    os.makedirs(save_path_context_mask_f)

print("Generating target and context Training sar_images...")

for i in range(len(images)):
    image = Image.open(images[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0]

    target_image = target(image)
    context_image = context_images(image)
    save_image(target_image, os.path.join(save_path_target, suffix + '.png'))
    save_image(context_image, os.path.join(save_path_context, suffix + '.png'))

print("Complete!!")

print("Generating target and context Training zones...")


for i in range(len(masks)):
    mask = Image.open(masks[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0]

    target_mask = target(mask)
    context_mask = context_masks(mask)
    save_image(target_mask, os.path.join(save_path_target_mask, suffix + '.png'))
    save_image(context_mask, os.path.join(save_path_context_mask, suffix + '.png'))

print("Complete!!")

print("Generating target and context Training fronts...")

for i in range(len(fronts)):
    image = Image.open(fronts[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0]

    target_image = target(image)
    context_image = context_images(image)
    save_image(target_image, os.path.join(save_path_target_mask_f, suffix + '.png'))
    save_image(context_image, os.path.join(save_path_context_mask_f, suffix + '.png'))

print("Complete!!")

ti = base_dir / "sar_images" / "sliding_window_valid"
mi = base_dir / "zones" / "sliding_window_valid"
fi = base_dir / "fronts" / "sliding_window_valid"

images = sorted(glob.glob(os.path.join(ti, '*.png')))
masks = sorted(glob.glob(os.path.join(mi, '*.png')))
fronts = sorted(glob.glob(os.path.join(fi, '*.png')))

save_path_target = base_dir / "sar_images" / "target_imgs_valid"
save_path_context = base_dir / "sar_images" / "context_imgs_valid"

save_path_target_mask = base_dir / "zones" / "target_imgs_valid"
save_path_context_mask = base_dir / "zones" / "context_imgs_valid"

save_path_target_mask_f = base_dir / "fronts" / "target_imgs_valid"
save_path_context_mask_f = base_dir / "fronts" / "context_imgs_valid"

target = transforms.Compose([
    transforms.CenterCrop((size, size)),
    transforms.ToTensor(),
])

context_images = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
])

context_masks = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor(),
])

if not os.path.exists(save_path_target):
    os.makedirs(save_path_target)
if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_target_mask):
    os.makedirs(save_path_target_mask)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)
if not os.path.exists(save_path_target_mask_f):
    os.makedirs(save_path_target_mask_f)
if not os.path.exists(save_path_context_mask_f):
    os.makedirs(save_path_context_mask_f)

print("Generating target and context Validating sar_images...")

for i in range(len(images)):
    image = Image.open(images[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0]

    target_image = target(image)
    context_image = context_images(image)
    save_image(target_image, os.path.join(save_path_target, suffix + '.png'))
    save_image(context_image, os.path.join(save_path_context, suffix + '.png'))

print("Complete!!")

print("Generating target and context Validating zones...")

for i in range(len(masks)):
    mask = Image.open(masks[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0]

    target_mask = target(mask)
    context_mask = context_masks(mask)
    save_image(target_mask, os.path.join(save_path_target_mask, suffix + '.png'))
    save_image(context_mask, os.path.join(save_path_context_mask, suffix + '.png'))

print("Complete!!")

print("Generating target and context Validating fronts...")

for i in range(len(fronts)):
    mask = Image.open(fronts[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0]

    target_mask = target(mask)
    context_mask = context_masks(mask)
    save_image(target_mask, os.path.join(save_path_target_mask_f, suffix + '.png'))
    save_image(context_mask, os.path.join(save_path_context_mask_f, suffix + '.png'))

print("Complete!!")

ti = base_dir / "sar_images" / "sliding_window_test"
mi = base_dir / "zones" / "sliding_window_test"
fi = base_dir / "fronts" / "sliding_window_test"

images = sorted(glob.glob(os.path.join(ti, '*.png')))
masks = sorted(glob.glob(os.path.join(mi, '*.png')))
fronts = sorted(glob.glob(os.path.join(fi, '*.png')))

save_path_target = base_dir / "sar_images" / "target_imgs_test"
save_path_context = base_dir / "sar_images" / "context_imgs_test"

save_path_target_mask = base_dir / "zones" / "target_imgs_test"
save_path_context_mask = base_dir / "zones" / "context_imgs_test"

save_path_target_mask_f = base_dir / "fronts" / "target_imgs_test"
save_path_context_mask_f = base_dir / "fronts" / "context_imgs_test"

target = transforms.Compose([
    transforms.CenterCrop((size, size)),
    transforms.ToTensor(),
])

context_images = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
])

context_masks = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor(),
])

if not os.path.exists(save_path_target):
    os.makedirs(save_path_target)
if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_target_mask):
    os.makedirs(save_path_target_mask)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)
if not os.path.exists(save_path_target_mask_f):
    os.makedirs(save_path_target_mask_f)
if not os.path.exists(save_path_context_mask_f):
    os.makedirs(save_path_context_mask_f)    

print("Generating target and context Testing sar_images...")

for i in range(len(images)):
    image = Image.open(images[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0]

    target_image = target(image)
    context_image = context_images(image)
    save_image(target_image, os.path.join(save_path_target, suffix + '.png'))
    save_image(context_image, os.path.join(save_path_context, suffix + '.png'))

print("Complete!!")

print("Generating target and context Testing zones...")

for i in range(len(masks)):
    mask = Image.open(masks[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0]

    target_mask = target(mask)
    context_mask = context_masks(mask)
    save_image(target_mask, os.path.join(save_path_target_mask, suffix + '.png'))
    save_image(context_mask, os.path.join(save_path_context_mask, suffix + '.png'))

print("Complete!!")

print("Generating target and context Testing fronts...")

for i in range(len(fronts)):
    mask = Image.open(fronts[i])
    suffix = os.path.splitext(os.path.basename(images[i]))[0]

    target_mask = target(mask)
    context_mask = context_masks(mask)
    save_image(target_mask, os.path.join(save_path_target_mask_f, suffix + '.png'))
    save_image(context_mask, os.path.join(save_path_context_mask_f, suffix + '.png'))

print("Complete!!")

end = time.time()

minutes = (end - start)/60

print(f"\nGeneration of target/context completed in about {minutes:.0f} minutes.")
