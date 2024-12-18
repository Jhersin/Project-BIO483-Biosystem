# Project BIO483 - MR Aceleration - DeepLearning based

Magnetic Resonance Imaging (MRI) is a crucial diagnostic tool that allows us to capture detailed images of the inside of our body. However, acquiring these images often takes a long time. One way to speed up this process is by skipping certain lines in the k-space data beyond the Nyquist limit. While this reduces scanning time, it also introduces undesirable artifacts in the images.  

This project explores a machine learning-based approach for improving image quality in MRI reconstruction by selectively skipping lines in the k-space. Specifically, only 25% and 13% of the k-space lines are selected. The goal is to use machine learning techniques to reconstruct the missing data and improve the quality of the resulting image.

In this approach, we have access to both the undersampled and fully sampled data. The objective is to calculate the parameters of a matrix that can convert the undersampled data into fully sampled data. This problem can be mathematically expressed as:

$$
x = B(y)
$$

Where:  
- x represents the undersampled data in the image domain,  
- y represents the fully sampled data in the image domain,  
- B is the matrix that we want to calculate. This matrix transforms the undersampled data into fully sampled data.

### Loss Function

The loss function used to train the model is as follows:

$$
B_{\theta} = \min_{\theta} \left( \frac{1}{2} \sum_{i=0}^{N_{\text{data}}} \left\| B_{\theta}(y) - x \right\|^2 \right)
$$

Where:  
- $$B_{\theta}$$ represents the parameters of the matrix \(B\) that we want to compute.  
- The goal is to minimize the difference between the undersampled data and the reconstruction of the fully sampled data.

### Optimization

The optimization problem is solved using the Adam optimization method.


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
