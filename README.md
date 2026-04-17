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

### Example 1: Landscape to Monet-style

![Sample: Landscape to Monet](output_cyclegan/samples/epoch_200_sample_1.png)

### Example 2: Monet-style to Landscape

![Sample: Monet to Landscape](output_cyclegan/samples/epoch_200_sample_2.png)


## 🚀 Usage

### 1. Installation

```bash
pip install torch torchvision
pip install -r requirements.txt
