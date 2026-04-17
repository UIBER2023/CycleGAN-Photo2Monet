# 🎨 Cycle-GAN: Landscape ↔ Monet Style Transfer

This is a major project for a Machine Learning course. The goal is to implement an image style transfer system using Cycle-GAN, enabling conversion between real landscape photos and oil paintings in the style of Claude Monet.

## 🗒️ Project Introduction

- **Course Project:** Machine Learning (2026 Spring)
- **Topic:** Image Style Transfer with Cycle-GAN
- **Objective:** Transform landscape photographs into Monet-style paintings (and vice versa) using deep learning.

## 📚 Background

Monet’s impressionist artworks are renowned for their dreamy colors and brushwork. Image-to-image translation techniques, especially Cycle-GAN, enable us to learn the mapping between two visual domains without paired training data.

- Cycle-GAN is a type of Generative Adversarial Network (GAN) for unpaired image translation.
- It consists of two generator–discriminator pairs that learn to translate images between two domains (landscapes ↔ Monet paintings).

## 🧠 Method

- **Framework:** PyTorch
- **Model:** Cycle-GAN
- **Training Data:** Landscape photos & Monet paintings (downloaded from Kaggle/other sources)
- **Key Features:**
  - Unpaired training
  - Bidirectional translation: Landscape → Monet, Monet → Landscape

## 🖼️ Results

### Example: Landscape to Monet-style

![Sample: Landscape to Monet](images/1.png)

### Example: Monet-style to Landscape

![Sample: Monet to Landscape](images/2.png)

### More Results

<div align="center">
  <img src="images/3.png" width="300">
  <img src="images/4.png" width="300">
</div>

## 🚀 Usage

### 1. Installation

```bash
pip install torch torchvision
pip install -r requirements.txt
