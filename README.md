# Single-Source Domain Defect-Aware Adaptation and Style-Modulated Generalization Network (SDSnet)

This repository contains the official implementation of the paper:

> **Single-Source Domain Defect-Aware Adaptation and Style-Modulated Generalization Network for Multispectral Image Segmentation**  
> *Wei Li, Boyu Zhao, Mengmeng Zhang, Yunhao Gao, and Junjie Wang*  
> IEEE Transactions on Cybernetics (TCYB), 2025  
>  
> [![Paper](https://ieeexplore.ieee.org/document/11223031/)]()()

---

## 🧩 Abstract

Multispectral remote sensing image (MSI) semantic segmentation faces challenges of limited labeled data and significant scene variability. Although Domain Adaptation (DA) and Domain Generalization (DG) methods alleviate these issues to some extent, they still have limitations: DA requires target domain data, while DG suffers from limited task adaptability.  

The recently emerged **Segment Anything (SAM)** model demonstrates exceptional zero-shot generalization capabilities, yet its visible-light training data and interactive prompt requirements prevent direct application to MSI segmentation tasks.  

To address these challenges, we propose **SDSnet** (*Single-Source Domain Defect-Aware Adaptation and Style-Modulated Generalization Network*), which integrates two key innovations:
- **Defect-Aware Prompt Learning**: automatically focuses on high-difficulty regions via entropy-based defect detection.  
- **Style Generalization Learning**: enhances cross-domain adaptability through codebook-based style modulation.  

Through **knowledge distillation**, SDSnet enables efficient inference using only the base network without extra computational overhead. Extensive experiments on three target domains demonstrate SDSnet’s superiority over state-of-the-art DA, DG, and SAM-based methods.

---

## 🧠 Framework Overview

<p align="center">
  <img src="./Frame.png" width="80%">
</p>
