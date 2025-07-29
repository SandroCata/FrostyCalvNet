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
├── data/                  # Dataset (CAFFEE + optional)
├── preprocessing/         # Scripts to prepare data patches
├── models/                # AMD-HookNet and possible variants
├── notebooks/             # Jupyter experiments and visualizations
├── results/               # Outputs, logs, metrics, plots
├── configs/               # YAML/JSON config files for training
├── utils/                 # Helpers and custom modules
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
