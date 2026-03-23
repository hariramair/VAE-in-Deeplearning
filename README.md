# VAE-in-Deeplearning
Convolutional Variational Autoencoder (VAE) in PyTorch for Fashion-MNIST, with experiments on latent-space structure, KL regularisation, reconstructions, and generated samples. Built as a learning and documentation project for understanding VAE architecture and training.

# Fashion-MNIST Variational Autoencoder (VAE)

A PyTorch implementation of a convolutional Variational Autoencoder (VAE) for the Fashion-MNIST dataset. This project explores how encoder-decoder design, latent-space size, KL regularisation, and model capacity affect reconstruction quality and latent-space structure.

## Overview

The goal of this project is to learn a compact latent representation of 28×28 grayscale Fashion-MNIST images, reconstruct the input images, and study how the latent space behaves under different configurations. The notebook implements:

- a convolutional encoder
- a convolutional decoder with transposed convolutions
- the reparameterisation trick
- VAE loss = reconstruction loss + KL divergence
- training and evaluation loops
- latent-space and reconstruction visualisations
- comparison of multiple model configurations

The work compares three configurations, including larger latent size, stronger KL regularisation, and a wider architecture. In the attached report, the best reconstruction result came from the model with `latent_size=16`, `kernels=(8,16,24)`, and `lmbda=1.0`. :contentReference[oaicite:1]{index=1}

## Dataset

This project uses the **Fashion-MNIST** dataset from `torchvision.datasets.FashionMNIST`.

- input shape: `1 × 28 × 28`
- grayscale images
- 10 classes
- pixel values scaled to `[0,1]` using `transforms.ToTensor()`

The notebook uses mini-batch loading with `DataLoader`, shuffling for training, and fixed random seeds for more repeatable experiments. :contentReference[oaicite:2]{index=2}

## Model Architecture

### Encoder
The encoder reduces the image spatial size in three steps:

- `1 × 28 × 28`
- `k[0] × 14 × 14`
- `k[1] × 7 × 7`
- `k[2] × 4 × 4`

This is followed by flattening and two linear heads:

- `fc_mu` for latent mean
- `fc_log_var` for latent log-variance

### Decoder
The decoder mirrors the encoder:

- latent vector
- linear expansion
- reshape to `k[2] × 4 × 4`
- transposed convolutions to recover:
  - `7 × 7`
  - `14 × 14`
  - `28 × 28`

ReLU is used in hidden decoder layers, and Sigmoid is used at the output so that reconstructed pixels stay in `[0,1]`. The report explains the exact shape flow and why `output_padding=1` is used in the later transpose-convolution layers. :contentReference[oaicite:3]{index=3}

## VAE Loss

The training objective is:

`total loss = reconstruction loss + lambda * KL divergence`

Where:

- **reconstruction loss** uses binary cross-entropy
- **KL divergence** regularises the latent space towards a standard normal distribution

This gives the usual VAE trade-off:

- smaller `lambda` → better reconstruction, weaker latent organisation
- larger `lambda` → stronger latent organisation, blurrier reconstruction

This trade-off is explicitly analysed in the report’s configuration comparison. :contentReference[oaicite:4]{index=4}

## Training Setup

Typical training settings used in the notebook:

- optimizer: `Adam`
- learning rate: `1e-3`
- epochs: `30`
- fixed random seed: `42`

A fixed seed was used to make comparisons between configurations more repeatable, since weight initialisation, latent sampling, and shuffled batch order all introduce randomness. :contentReference[oaicite:5]{index=5}

## Visualisations

The notebook includes helper functions to inspect model behaviour:

- **loss curves** for train/test total loss, reconstruction loss, and KL loss
- **latent scatter plot** using encoder `mu`
- **latent grid decoding** for `latent_size=2`
- **input vs reconstruction** comparison

These plots were used to compare how latent size, KL strength, and architecture width affect the VAE. :contentReference[oaicite:6]{index=6}

## Configurations Compared

The report compares three main models:

### Model A — Larger latent space
- `kernels=(8,16,24)`
- `latent_size=16`
- `lmbda=1.0`

### Model B — Higher lambda
- `kernels=(8,16,24)`
- `latent_size=2`
- `lmbda=4.0`

### Model C — Bigger architecture
- `kernels=(16,32,64)`
- `latent_size=2`
- `lmbda=1.0`

### Best model
Based on automated testing and reconstruction loss, the best model in the report was:

- `kernels=(8,16,24)`
- `latent_size=16`
- `lmbda=1.0`

with reconstruction loss around **229.9** on the test set. All three trained models passed the automated checks. :contentReference[oaicite:7]{index=7}

## What I Learned

Through this project, I focused on understanding:

- how convolution layers compress image structure
- why a VAE outputs `mu` and `log_var` instead of one fixed latent vector
- how sampling and KL regularisation work together
- how `latent_size` affects reconstruction quality and visualisability
- how `kernels`, `stride`, `padding`, and `flat_size` determine tensor shapes
- how different `lambda` values affect the trade-off between reconstruction and latent-space structure

## Repository Structure

Example structure:

```text
.
├── A2_VAE.ipynb
├── README.md
├── requirements.txt
└── data/
