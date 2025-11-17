import logging
import os
from copy import deepcopy

import numpy as np
import torch.nn as nn

from models.front_segmentation_model import FrontUNet
from models.transunet.vit_seg_modeling import CONFIGS, VisionTransformer


LOGGER = logging.getLogger(__name__)


class FrontTransUNet(FrontUNet):
    """Lightning module wrapping the TransUNet backbone for front segmentation."""

    def __init__(self, hparams):
        self.transunet_variant = hparams.get("transunet_variant", "R50-ViT-B_16")
        self.transunet_variant = hparams.get("transunet_variant", "mk10-small")
        self.transunet_img_size = hparams.get("transunet_img_size", 256)
        self.transunet_pretrained = hparams.get("transunet_pretrained", "")
        self.transunet_n_skip = hparams.get("transunet_n_skip", None)
        self.transunet_decoder_channels = hparams.get("transunet_decoder_channels", None)

        super().__init__(hparams)

        # Remove the parent UNet layers as we are replacing the backbone entirely.
        self.layers = nn.ModuleList()
        self.bottleneck = None
        self.aspp = False

        config = deepcopy(CONFIGS[self.transunet_variant])
        config.n_classes = 1
        if self.transunet_decoder_channels is not None:
            if len(self.transunet_decoder_channels) != 4:
                raise ValueError("Expected four values for --transunet_decoder_channels.")
            config.decoder_channels = tuple(self.transunet_decoder_channels)
        if self.transunet_n_skip is not None:
            config.n_skip = int(self.transunet_n_skip)
        if not hasattr(config, "n_skip"):
            config.n_skip = 0
        if not hasattr(config, "skip_channels"):
            config.skip_channels = [0, 0, 0, 0]

        self.transunet = VisionTransformer(config, img_size=self.transunet_img_size, zero_head=True, vis=False)

        if self.transunet_pretrained:
            self._load_pretrained_weights(self.transunet_pretrained)

    def _load_pretrained_weights(self, checkpoint_path: str) -> None:
        if not os.path.isfile(checkpoint_path):
            LOGGER.warning("Pretrained checkpoint %s could not be found. Skipping load.", checkpoint_path)
            return

        weights = np.load(checkpoint_path, allow_pickle=True)
        if isinstance(weights, np.lib.npyio.NpzFile):
            weights = {key: weights[key] for key in weights.files}
        elif isinstance(weights, np.ndarray):
            weights = weights.item()
        self.transunet.load_from(weights)

    def forward(self, x):
        return self.transunet(x)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = FrontUNet.add_model_specific_args(parent_parser)
        group = parser.add_argument_group("TransUNet")
        group.add_argument(
            "--transunet_variant",
            default="mk10-small",
            choices=sorted(CONFIGS.keys()),
            help="Backbone variant to use.",
        )
        group.add_argument(
            "--transunet_img_size",
            type=int,
            default=256,
            help="Input resolution expected by TransUNet (must be divisible by the patch size).",
        )
        group.add_argument(
            "--transunet_pretrained",
            default="",
            help="Optional path to a .npz checkpoint with pretrained TransUNet weights.",
        )
        group.add_argument(
            "--transunet_n_skip",
            type=int,
            default=None,
            help="Override the number of skip connections used by the decoder (defaults to config value).",
        )
        group.add_argument(
            "--transunet_decoder_channels",
            nargs="+",
            type=int,
            default=None,
            help="Override decoder channel widths. Provide four integers.",
        )
        return parent_parser