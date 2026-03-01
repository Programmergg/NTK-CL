<div align="center">

# Parameter-Efficient Fine-Tuning for Continual Learning: A Neural Tangent Kernel Perspective

[![arXiv](https://img.shields.io/badge/arXiv-2407.17120-b31b1b.svg)](https://arxiv.org/abs/2407.17120)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](#installation)

**Jingren Liu**, Zhong Ji, Yunlong Yu, Jiale Cao, Yanwei Pang, Jungong Han, Xuelong Li
</div>

---

## 🔥 News
- **[2026-03-01]** Code and scripts released.
- **[2026-03-01]** Pretrained checkpoints uploaded.
- **[2026-03-01]** Reproducibility guide updated.

---

## 🧩 Overview
This repository provides the official implementation for:

**Parameter-Efficient Fine-Tuning for Continual Learning: A Neural Tangent Kernel Perspective** (arXiv:2407.17120).

We study PEFT-CL via **Neural Tangent Kernel (NTK)** theory and introduce **NTK-CL**, a framework that:
- avoids task-specific parameter storage,
- adaptively generates task-relevant features,
- reduces task-interplay and task-specific generalization gaps,
- achieves strong performance on established PEFT-CL benchmarks.

---

## ✅ Key Features
- [x] Training / evaluation scripts for PEFT-CL benchmarks
- [x] NTK-CL implementation
- [x] Pretrained checkpoints (optional)
- [x] Dataset preparation scripts (optional)
---

## 🗂️ Repository Structure

```text
.
├── data/                       # datasets / prepared data
├── dataloader/                 # dataset & dataloader implementations
├── models/                     # model definitions / PEFT modules / backbones
├── main.py                      # main entry for training/evaluation
└── trainer.py                   # training loop / continual learning pipeline
