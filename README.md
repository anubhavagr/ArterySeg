# ArterySeg
Coronary artery segmentation using u-net with resnet34 backbone

A high-performance artery segmentation pipeline using a ResNet34-based encoder with attention mechanisms and a custom Sobel edge detection layer. The project is optimized for GPU acceleration with mixed precision training and efficient data handling.

---

## 🔍 Overview

This project focuses on segmenting arterial structures from medical images using a custom deep learning model. The architecture leverages:

- ResNet34 encoder
- Sobel edge enhancement
- Spatial attention
- GPU-accelerated operations (with mixed precision support)

---

## 📁 Project Structure

.
├── config.py                # Configuration settings and hyperparameters
├── dataset.py               # Dataset and data augmentation pipeline using Albumentations
├── model.py                 # Model architecture with Sobel layer, attention, and ResNet34 backbone
├── train.py                 # Training and validation routines
├── utils.py                 # Utility functions (GPU mask creation, RGB palette)
├── main.py                  # Entry point for training
├── results/                 # Folder to store validation output images
└── README.md                # Overview
