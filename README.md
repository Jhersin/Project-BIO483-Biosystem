# Project BIO483 - MR Aceleration - DeepLearning based

Magnetic Resonance Imaging (MRI) is a crucial diagnostic tool that allows us to capture detailed images of the inside of our body. However, acquiring these images often takes a long time. One way to speed up this process is by skipping certain lines in the k-space data beyond the Nyquist limit. While this reduces scanning time, it also introduces undesirable artifacts in the images.  

This project explores a machine learning-based approach for improving image quality in MRI reconstruction by selectively skipping lines in the k-space. Specifically, only 25% and 13% of the k-space lines are selected. The goal is to use machine learning techniques to reconstruct the missing data and improve the quality of the resulting image.

In this approach, we have access to both the undersampled and fully sampled data. The objective is to calculate the parameters of a matrix that can convert the undersampled data into fully sampled data. This problem can be mathematically expressed as:

$$
x = B_{\theta}(y)
$$

Where:  
- **x**: Undersampled data in the image domain.
- **y**: Fully sampled data in the image domain.
- **B**: Transformation matrix that maps the undersampled data to the fully sampled data.

The relationship can be expressed as:

$$
\mathbf{y} = \mathbf{B} \cdot \mathbf{x}
$$


### Loss Function

The loss function used to train the model is as follows:

$$
B_{\theta} = \min_{\theta} \left( \frac{1}{2} \sum_{i=0}^{N_{\text{data}}} \left\| B_{\theta}(y) - x \right\|^2 \right)
$$

Where:  
- **$$B_{\theta}$$** represents the parameters of the matrix \(B\) that we want to compute.  

The goal is to minimize the difference between the undersampled data and the reconstruction of the fully sampled data.

The optimization problem is solved using the Adam optimization algortihm.


## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Contributing](#contributing)

## Features
- List of the main features of the project:

  ## Project Structure

| **Program**                        | **Description**                                                                 |
|------------------------------------|---------------------------------------------------------------------------------|
| **Data Exploration**               | Contains initial exploration of the train, validation, and test datasets.       |
| **Dataset**                        | Contains the `kneedataset` class that merges the input and output images.       |
| **FastTakitoMulticoilPreprocessing** | Contains the program that organizes the data into 4x and 8x acceleration factors. |
| **Models**                         | Contains the CNN-based models.                                                  |
| **Transfer Learning**              | Part of the code that freezes the encoder part in the U-Net.                    |
| **Utils**                          | Contains various utility functions such as Fourier transform, inverse Fourier transform, cropping, etc. |
| **Model1_UNET2017_MC4**            | Contains the model to train Single Coil to Multi Coil.                          |
| **Model2_UNET2017_MC4_TL**         | Contains the model to train undersampled images (4x acceleration) to fully sampled images. |
| **Model3_UNET2017_MC8_TL**         | Contains the model to train undersampled images (8x acceleration) to fully sampled images. |


## Installation
Follow these step-by-step instructions to set up the project:

1. Install the required libraries:
   ```bash   
   conda install matplotlib
   conda install numpy
   Install CUDA12.4
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install pytorch-lightning
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/Jhersin/Project-BIO483-Biosystem.git
   ```

## Contributing
Contributors to this project include:
- Jhersin Garcia
