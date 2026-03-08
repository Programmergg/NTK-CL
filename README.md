<div align="center">

# Parameter-Efficient Fine-Tuning for Continual Learning: A Neural Tangent Kernel Perspective

[![arXiv](https://img.shields.io/badge/arXiv-2407.17120-b31b1b.svg)](https://arxiv.org/abs/2407.17120)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](#installation)

**Jingren Liu**, Zhong Ji, Yunlong Yu, Jiale Cao, Yanwei Pang, Jungong Han, Xuelong Li
</div>

---

## 🔥 News
- **[2026-03-08]** Code and scripts released.
- **[2026-03-xx]** Reproducibility guide updated.

---

## 🧩 Overview
This repository provides the official implementation for:

**Parameter-Efficient Fine-Tuning for Continual Learning: A Neural Tangent Kernel Perspective**.

We study PEFT-CL via **Neural Tangent Kernel (NTK)** theory and introduce **NTK-CL**, a framework that:
- avoids task-specific parameter storage,
- adaptively generates task-relevant features,
- reduces task-interplay and task-specific generalization gaps,
- achieves strong performance on established PEFT-CL benchmarks.

---

## ✅ Key Features
- [x] Training / evaluation scripts for PEFT-CL benchmarks
- [x] NTK-CL implementation
- [x] Dataset preparation scripts
---

## 🗂️ Repository Structure

```text
.
├── data/                       # datasets / prepared data
├── dataloader/                 # dataset & dataloader implementations
├── models/                     # model definitions / PEFT modules / backbones
├── trainer.py                  # training loop / continual learning pipeline
├── requirements.txt            # python dependencies for reproducing the environment
└── main.py                     # main entry for training/evaluation
```
---

## 🧰 Installation

### 1) Create environment
```bash
conda create -n ntk_cl python=3.11 -y
conda activate ntk_cl
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```
---

## 📦 Datasets

To facilitate reproduction, we provide downloadable dataset packages/links for the benchmarks used in this project, including:
CIFAR-100, ImageNet-R, ImageNet-A, DomainNet, Oxford Pets, EuroSAT, PlantVillage, VTAB ,and Kvasir.

### Download
You can download the datasets from the following links:

- **Baidu Netdisk**: `(TODO: add Baidu link)`
- **Google Drive**: `(TODO: add Google Drive link)`

> Please download the required datasets and place them under the `data/` directory following the structure expected by the code.

### Directory Structure
A recommended dataset layout is:

```text
data/
├── cifar-100-python/
├── imagenet-r/
├── imagenet-a/
├── DomainNet/
├── OxfordPets/
├── EuroSAT/
├── PlantVillage/
├── VTAB/
└── Kvasir/
```

