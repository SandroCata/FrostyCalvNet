# FrostyCalvNet only front variants

Alternative version of main code. Experimental.

Models:

Scheme 1 — FrontUNet (U-Net + ASPP bottleneck)

Input (1×H×W)
  │
  ├─ Encoder U-Net (4 downsample con MaxPool)
  │    1→32 → 64 → 128 → 256 → 512   [DoubleConv + BN + LeakyReLU]
  │
  ├─ Bottleneck: ASPP (rates 1,2,4,8 + pooling globale) → proj 512
  │
  ├─ Decoder U-Net (4 upsample con ConvTranspose2d)
  │    + skip simmetrici “ricchi” → DoubleConv
  │
  └─ Head: Conv1×1 (32→1)  → logits
       Loss: DistanceMapBCE(Dice su pred · dmap)


Scheme 2 — FrontDeepLab (⚠️ only DeepLabV3+; U-Net not used)

Input (1×H×W)
  │
  ├─ Encoder ResNet (smp) con dilatazioni (OS=16 di default)
  │    conv7x7/2 → maxpool → layer1 → layer2(stride2)
  │    → layer3(stride2) → layer4(dilation)
  │
  ├─ ASPP (rates tipici 12/24/36 + pooling) → decoder_channels (es. 256)
  │
  ├─ Decoder DeepLabV3+
  │    upsample ×4 → fuse con low-level (proiettati a 48 canali)
  │    → separable conv
  │
  └─ Segmentation head: Conv1×1 (256→1) → upsample ×4 → logits
       Loss: DistanceMapBCE(Dice su pred · dmap)