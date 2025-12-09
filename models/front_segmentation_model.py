import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from einops import rearrange
from models.ParentUNet import UNet
from scipy.ndimage.morphology import distance_transform_edt as edt
#import torchmetrics
import numpy as np
import cv2 
from segmentation_models_pytorch.losses import FocalLoss, DiceLoss


class DistanceMapBCE(nn.Module):
    def __init__(self, device, w, k, relax):
        super(DistanceMapBCE, self).__init__()
        self.device = device
        self.w = w  # parameter w of dmap loss
        self.k = k  # parameter k of dmap loss
        self.relax = relax

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        distance_maps = torch.stack(list(map(self.distance_map, torch.unbind(targets))))
        preds = torch.stack(list(map(torch.mul, torch.unbind(preds), torch.unbind(distance_maps))))
        criterion = DiceLoss('binary', from_logits=False)
        targets = rearrange(targets, 'b h w -> b 1 h w')
        loss = criterion(preds, targets)
        return loss

    def distance_map(self, target):
        if target.max() == 0:
            distance_map_image = 1 - target
        else:
            kernel = np.ones((self.w, self.w), dtype=np.float32)
            fat_gt = cv2.dilate(target.cpu().numpy(), kernel)
            euclidean_distance_transformed = torch.from_numpy(edt(fat_gt)).to(self.device).type(torch.float32)
            soft_gt = torch.sigmoid(torch.div(euclidean_distance_transformed, self.relax))
            
            soft_gt = torch.sub(soft_gt, torch.min(soft_gt))
            soft_gt = torch.div(soft_gt, torch.max(soft_gt))
            distance_map_image = soft_gt + (1 - torch.from_numpy(fat_gt).to(self.device).type(torch.float32)) * self.k

        return distance_map_image


class FrontSegmentationModelBase(UNet):

    def __init__(self, hparams):
        
        super().__init__(hparams=hparams, metric=None, n_classes=1)
        """
        self.w = hparams["w"]
        self.k = hparams["k"]
        self.relax = hparams["relax"]""
        """
        self.model_arch = hparams.get("model_arch", "deeplab")
        self.encoder_name = hparams.get("encoder_name", "resnet50")
        self.encoder_weights = hparams.get("encoder_weights", "imagenet")
        self.encoder_output_stride = hparams.get("encoder_output_stride", 16)

        self.gamma = hparams["gamma"]
        self.alpha = hparams.get("alpha", None)

        self.loss_focal_weight = hparams.get("loss_focal_weight", 0.5)
        self.loss_dice_weight = hparams.get("loss_dice_weight", 0.5)
        self.dice = DiceLoss(mode="binary")
        

    def adapt_mask(self, y):
        return y.type(torch.float32)

    def make_batch_dictionary(self, loss, metric, name_of_loss):
        """ Give batch_dictionary corresponding to the number of metrics for front segmentation """
        batch_dictionary = {
            name_of_loss: loss,
            "IoU": metric,
        }
        return batch_dictionary

    def log_metric(self, outputs, train_or_val_or_test):
        # calculating average metric
        avg_iou = torch.stack([x["IoU"] for x in outputs]).mean()
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("IoUs/" + train_or_val_or_test + "/IoU", avg_iou, self.current_epoch)
        if train_or_val_or_test == "Val":
            self.log('avg_metric_validation', avg_iou)
        metrics = {
            "IoU": avg_iou,
        }
        return metrics
    

    def _single_iou(self, pred, target):
        pred = (pred > 0).float()
        target = target.float()
        intersection = (pred * target).sum(dim=(1, 2))
        union = ((pred + target) > 0).float().sum(dim=(1, 2))
        return (intersection / (union + 1e-6)).mean()

    def class_wise_iou(self, y_hat, y):
        front_iou = self._single_iou(y_hat[:, 0], y[:, 0])
        return front_iou

    def calc_loss(self, y_hat, y):
        y = self.adapt_mask(y)
        """
        dmap_criterion = DistanceMapBCE(self.device, w=self.w, k=self.k, relax=self.relax)
        loss = dmap_criterion(y_hat[:, 0], y[:, 0])
        """
        focal_criterion = FocalLoss('binary', gamma=self.gamma, alpha=self.alpha)
        front_pred = y_hat[:, 0].contiguous()
        front_true = y[:, 0].contiguous()
        loss_front_focal = focal_criterion(front_pred, front_true)
        
        loss_front_dice = self.dice(front_pred, front_true)
        loss = (
            self.loss_focal_weight * loss_front_focal
            + self.loss_dice_weight * loss_front_dice
        )
        
        return loss, self.class_wise_iou(y_hat, y)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("FrontSegmentation")
        parser.add_argument('--model_arch', default='deeplab', choices=['unet', 'deeplab'],
                            help="Architecture to use for front segmentation.")
        parser.add_argument('--encoder_name', default='resnet50',
                            help="Encoder backbone for DeepLabV3+. Ignored for U-Net.")
        parser.add_argument('--encoder_weights', default='imagenet',
                            help="Pre-trained weights for the encoder. Use 'None' to disable.")
        parser.add_argument('--encoder_output_stride', type=int, default=16,
                            help="Encoder output stride for DeepLabV3+. Ignored for U-Net.")
        parser.add_argument('--base_lr', default=1e-4)
        parser.add_argument('--max_lr', default=5e-4)
        parser.add_argument('--batch_size', default=16)
        parser.add_argument('--kernel_size', type=int, default=3)

        # Focal and Dice Loss arguments
        parser.add_argument('--gamma', default=1.0, type=float, help="Gamma parameter for FocalLoss used for near segmentation")
        parser.add_argument('--alpha', default=0.9, type=float, help="Alpha parameter for FocalLoss used for near segmentation")

        parser.add_argument('--loss_focal_weight', default=0.5, type=float, help="Weight for FocalLoss component of near segmentation loss")
        parser.add_argument('--loss_dice_weight', default=0.5, type=float, help="Weight for DiceLoss component of near segmentation loss")

        # Loss arguments
        """
        parser.add_argument('--k', default=0.1, type=float, help="Parameter for the distance map-based loss for front segmentation")
        parser.add_argument('--w', default=5, type=int, help="Parameter for the distance map-based loss for front segmentation")
        parser.add_argument('--relax', default=1, type=int, help="Parameter for the distance map-based loss for front segmentation")
        """
        
        # Hyperparameters for augmentation
        parser.add_argument('--bright', default=0.65, type=float)
        parser.add_argument('--wrap', default=0.65, type=float)
        parser.add_argument('--noise', default=0.65, type=float)
        parser.add_argument('--rotate', default=0.65, type=float)
        parser.add_argument('--flip', default=0.65, type=float)

        # Layer arguments
        parser.add_argument('--aspp', default=True, type=lambda x: (str(x).lower() == 'true'))
        return parent_parser
    
class FrontUNet(FrontSegmentationModelBase):

    def __init__(self, hparams):
        super().__init__(hparams)


class FrontDeepLab(FrontSegmentationModelBase):

    def __init__(self, hparams):
        super().__init__(hparams)
        encoder_weights = self.encoder_weights
        if isinstance(encoder_weights, str) and encoder_weights.lower() == 'none':
            encoder_weights = None
        self.backbone = smp.DeepLabV3Plus(
            encoder_name=self.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=self.n_channels_of_input,
            classes=self.n_classes,
            encoder_output_stride=self.encoder_output_stride,
        )

    def forward(self, x):
        return self.backbone(x)
