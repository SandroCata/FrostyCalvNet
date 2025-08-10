# IMPORT SECTION:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from pathlib import Path
import random
import torch
import torch.nn.functional as F
import torch.nn as nn

import logging
import sys
import cv2
from datetime import datetime
import numpy as np
from torch import optim
from PIL import Image
from tqdm import tqdm

#Config module loaded with yacs
from configs.config import cfg

#compute metrics on validation                            
from models.eval_whole import eval_net

#model architecture
from models.AMD_HookNet_model import AMD_HookNet

#load patch images + masks 
from data.dataset import BasicDataset
from data.valid_dataset import BasDataset

#Custom loss along with Crossentropy Loss
from models.loss import DiceLoss, make_one_hot

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


#-----------------------------------------------------------------------

# DATA PATHS:

dir_checkpoint = cfg.output_path


base_data_path = Path(__file__).resolve().parents[1] / "Preprocessed_CAFFE_fronts"  #_zones if std_segm

#print(base_data_path)

#load data directly from the preprocessed dataset structure
scratch_train = os.path.join(base_data_path, 'Training')
scratch_valid = os.path.join(base_data_path, 'Validation')

#Define the path to the target (centered) and context (wide) images, for both images and masks in training and validation.
dir_img_target = os.path.join(scratch_train, 'target_images')
dir_img_context = os.path.join(scratch_train, 'context_images')
dir_mask_target = os.path.join(scratch_train, 'target_masks')
dir_mask_context = os.path.join(scratch_train, 'context_masks')

valid_img_target = os.path.join(scratch_valid, 'target_images')
valid_img_context = os.path.join(scratch_valid, 'context_images')
valid_mask_target = os.path.join(scratch_valid, 'target_masks')
valid_mask_context = os.path.join(scratch_valid, 'context_masks')

#-----------------------------------------------------------------------

# GLOBAL SEED FUNCTION:

#Set seed for reproducibility (random, numpy, torch). Also set cudnn.benchmark=True (dynamic GPU optimization).
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

#-----------------------------------------------------------------------

# TRAINING FUNCTION:

#
def train_net(net,
              device,
              optimizer,
              scheduler,
              eepoch,
              epochs,
              batch_size,
              lr,
              save_cp,
              img_scale,
              ):
    
    #Create dataloader with extracted patches (using BasicDataset and BasDataset). Batch_size is configurable, num_workers=16 → multi-threaded. In the test, the batch size is 1 for single patch evaluation.
    dataset = BasicDataset(dir_img_target, dir_img_context, dir_mask_target, dir_mask_context, img_scale)
    valid_dataset = BasDataset(valid_img_target, valid_img_context, valid_mask_target, valid_mask_context, cfg.scale)

    n_dataset = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    logging.info(f'''Starting training:
        Total epochs:          {epochs}
        Batch size:            {batch_size}
        Initial learning rate: {lr}
        Save checkpoints:      {save_cp}
        Device:                {device.type} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})
    ''')
    
    #Use CE + Dice combination for both target and context → more stability and accuracy on edges.
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = DiceLoss()

    #Training loop
    for epoch in range(eepoch, epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_dataset, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                
                #Extract target patches and context (images and masks).
                imgs_target = batch['image_target']
                imgs_context = batch['image_context']
                true_masks_target = batch['mask_target']
                true_masks_context = batch['mask_context']
                assert imgs_target.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs_target.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                #Generate 4 downsampled versions of the target masks to supervise the intermediate outputs of the network.
                masks = [true_masks_target]
                
                for i in range(4):
                    big_mask = masks[-1]
                    small_mask = F.avg_pool2d(big_mask, 2)
                    masks.append(small_mask)

                small = [m.squeeze(1).long() for m in masks[1:]]

                imgs_target = imgs_target.to(device=device, dtype=torch.float32)
                imgs_context = imgs_context.to(device=device, dtype=torch.float32)

                #mask_type = torch.float32 if net.n_classes <= 2 else torch.int64
                true_masks_target = true_masks_target.to(device=device, dtype=torch.int64)
                true_masks_context = true_masks_context.to(device=device, dtype=torch.int64)

                #Pass both patches (context and target) to the AMD-HookNet network.
                masks_pred = net(imgs_context, imgs_target)

                true_masks_context_dice = torch.unsqueeze(true_masks_context, dim=1)
                true_masks_context_dice = make_one_hot(true_masks_context_dice.type(torch.int64), net.n_classes)
                true_masks_target_dice = torch.unsqueeze(true_masks_target, dim=1)
                true_masks_target_dice = make_one_hot(true_masks_target_dice.type(torch.int64), net.n_classes)

                c_loss1 = criterion1(masks_pred[0], true_masks_context)
                t_loss1 = criterion1(masks_pred[1], true_masks_target)

                c_loss2 = criterion2(masks_pred[0], true_masks_context_dice)
                t_loss2 = criterion2(masks_pred[1], true_masks_target_dice)

                #Three contributions:

                # - Loss on the context branch

                # - Loss on the target branch

                # - Deep supervision loss for intermediate features

                c_loss = c_loss1 + c_loss2
                t_loss = t_loss1 + t_loss2

                loss_deep = 0

                for i in range(len(masks_pred[2])):
                    target_small = small[i].to(device, dtype=torch.int64)
                    lossc = criterion1(masks_pred[2][2 - i], target_small)
                    dice = torch.unsqueeze(target_small, dim=1)
                    dice = make_one_hot(dice, net.n_classes)
                    lossd = criterion2(masks_pred[2][2 - i], dice)
                    loss_deep += lossc + lossd

                #Combined
                loss = t_loss + c_loss + 0.5 * loss_deep

                epoch_loss += loss.item() / imgs_context.shape[0]

                pbar.set_postfix({'loss': loss.item() / imgs_context.shape[0]})

                #Classic backward cycle
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs_context.shape[0])

        print('Train average loss:', epoch_loss / n_dataset)
        print('lr:', optimizer.param_groups[0]['lr'])

        
        #Evaluate model at each epoch (print metrics)
        success_ratio, vaild_iou_ratio, iou = eval_net(net, valid_loader, device)
        print('Validation iou ratio: {}, {}'.format(vaild_iou_ratio, iou))

        #Save model, optimizer, learning rate scheduler, current epoch.
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                print('Created checkpoint directory')
            except OSError:
                pass
            checkpoint_path = os.path.join(
                dir_checkpoint, f'AMD_HookNet_epoch{epoch + 1:03d}.pth'
            )
            torch.save({
                        'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'lr_schedule': scheduler.state_dict(),
                        },
                        checkpoint_path)

            print(f'Checkpoint {checkpoint_path} saved!\n')
            
        if epoch < epochs - 1:

            #Learning rate scheduler (gamma decreases at each epoch)
            scheduler.step()


#main section
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        device_name = torch.cuda.get_device_name(device)
        logging.info(f'Using device {device} ({device_name})')
    else:
        device_name = 'CPU'
        logging.info(f'Using device {device}')
    seed_torch(cfg.seed)
    eepoch = 0
    

    # log hyperparameters
    hyperparams = {
        'META_ARC': cfg.META_ARC,
        'CUDA': cfg.CUDA,
        'output_path': cfg.output_path,
        'in_channels': cfg.in_channels,
        'n_classes': cfg.n_classes,
        'n_filters': cfg.n_filters,
        'filter_size': cfg.filter_size,
        'learning_rate': cfg.learning_rate,
        'gamma': cfg.gamma,
        'total_epoch': cfg.total_epoch,
        'steps': cfg.steps,
        'batch_size': cfg.batch_size,
        'seed': cfg.seed,
        'load': cfg.load,
        'scale': cfg.scale,
        'device': device.type,
        'device_name': device_name,
    }

    for k, v in hyperparams.items():
        logging.info(f'{k}: {v}')

    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'hyperparams_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    with open(log_file, 'w') as f:
        for k, v in hyperparams.items():
            f.write(f'{k}: {v}\n')
    logging.info(f'Hyperparameters saved to {log_file}')



    net = AMD_HookNet(cfg.in_channels, cfg.n_classes, cfg.filter_size, cfg.n_filters)
    net.init_weights()
    net.to(device=device)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)

    if cfg.load:
        checkpoints = torch.load(cfg.load, map_location=device)
        net.load_state_dict(checkpoints['net_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        eepoch = checkpoints['epoch']
        scheduler.load_state_dict(checkpoints['lr_schedule'])
        scheduler.last_epoch = eepoch
        logging.info(f'Model loaded from {cfg.load}')

    try:
        train_net(net=net,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  eepoch=eepoch,
                  epochs=cfg.total_epoch,
                  batch_size=cfg.batch_size,
                  lr=cfg.learning_rate,
                  device=device,
                  save_cp=True,
                  img_scale=cfg.scale,
                  )

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
