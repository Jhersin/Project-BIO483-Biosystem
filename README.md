# Project BIO483 - MR Aceleration - DeepLearning based

Magnetic Resonance Imaging (MRI) is a crucial diagnostic tool that allows us to capture detailed images of the inside of our body. However, acquiring these images often takes a long time. One way to speed up this process is by skipping certain lines in the k-space data beyond the Nyquist limit. While this reduces scanning time, it also introduces undesirable artifacts in the images.  

This project explores a method to skip lines in the k-space domain by treating the problem as an inverse problem. The goal is to reconstruct the best high-resolution image-x from the undersampled k-space data-s. This problem can be expressed mathematically as:  

$$
f(x) = \min \|s - Ex\|_2^2 + \lambda \|x\|_2^2
$$  

Where:  
- \(s\) is the undersampled data,  
- \(x\) is the high-resolution image,  
- \(E\) combines the undersampling mask, coil sensitivity, and Fourier transform,  
- \(\lambda\) is a regularization term.  

The project explores simple 1D and 2D examples to address this problem. It focuses on reconstructing \(x\) (the high-resolution image) using methods such as the pseudoinverse matrix and gradient descent.  

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Contributing](#contributing)

## Features
- List of the main features of the project. This project includes three main codes:

  - **1D_Toy_Remasterd**: 1D Numerical Solution  
  - **Gradient Descent Numpy**: 2D Gradient Descent Example  
  - **Gradient Descent Torch**: 2D Gradient Descent Example  
  - **Sigpy SENSE**: SENSE library  
  - **Data**: Contains Sensitivity Maps and Knee Images  
  - **Utils**: Contains useful functions  

## Installation
Follow these step-by-step instructions to set up the project:

1. Install the required libraries:
   ```bash
   pip install sigpy
   conda install matplotlib
   conda install numpy
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/Jhersin/Project-BIO483-Biosystem.git
   ```

## Contributing
Contributors to this project include:
- Jhersin Garcia
