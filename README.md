# FrostyCalvNet

## 📌 Overview

This repository is part of the research work conducted for a Master's thesis in Engineering in Computer Science at the University of Rome "La Sapienza". 

The study focuses on the application of Deep Learning techniques for the automatic detection of glacier calving fronts from satellite imagery.

The objective of this thesis is to design, implement, and evaluate deep neural network architectures capable of accurately segmenting glacier front boundaries, in order to analyze the melting and retreat processes affecting various glaciers worldwide. The analysis spans a temporal range from 1995 to 2020, using SAR satellite images and geospatial data provided primarily by the CAFFEE dataset.

By leveraging solid segmentation models (such as AMD-HookNet, Bayesian U-Net, and other convolutional and transformer-based networks) this project aims to provide both precise front delineation and an estimation of glacier evolution patterns over time. The ultimate goal is to produce a system that not only offers retrospective insights on glacier behavior, but that may also serve as a foundation for future applications in climate change monitoring, glaciology, and environmental risk assessment.


---

## 📂 Project Structure

```bash
FrostyCalvNet/
│
├── data/                  # CAFFEE Dataset download link
├── preprocessing/         # Scripts to prepare data patches
├── models/                # AMD-HookNet and possible variants
├── notebooks/             # Jupyter experiments and visualizations
├── results/               # Outputs, logs, metrics, plots
├── configs/               # YAML/JSON config files for training
├── utils/                 # References and Documentation
├── main.py                # Entry point for training/inference
└── README.md
```

---

## 🧭 Step 1 — Literature Review & Research

Most coherent and relevant Neural Networks: 

- AMD-HookNet,
- Bayesian U-Net,
- nnU-Net

Related Papers:

- AMD-HookNet for Glacier Front Segmentation (Wu et al., 2023)
- Bayesian U-Net for Glacier Segmentation in SAR Imagery
- Globally Scalable Glacier Mapping by Deep Learning

Selected Dataset: 

- CAFFEE Dataset

## 🚀 Step 2 — Reproducing AMD-HookNet on CAFFEE

Deep study of related paper and code, especially for:

- Getting acquainted with data and neural network structure.
- Understanding its complexity
- Examining its results
- Looking for improvements and original ideas to integrate

### Step 2.1 - Data Preprocessing

 - From raw data, generating target e context images to get inputs for AMD Hooknet network.
 - Sequence of preprocessing process:

    train_valid_split.py – Step 0

      - Splits the dataset into train/validation with an 80/20 ratio, creating validation folders if missing and moving each SAR image along with its fronts and zones masks.

    center_crop.py – Step 1

      - Padding the images and masks so that their dimensions are multiples of 288 pixels, keeping the contents centered and uniform across all sets (train/valid/test).

    Sliding_window_generate_dataset.py – Step 2

      - Applies a sliding window with a step of 288 and a patch of 576×576 pixels to generate tiles from already centered images. The process is repeated for training, validation, and testing.
    
    generate_target_context.py – Step 3

      - From the tiles, it produces target (central 288×288 crop) and context (576×576 window resized to 288×288) pairs for images and masks in all partitions of the dataset.
 
