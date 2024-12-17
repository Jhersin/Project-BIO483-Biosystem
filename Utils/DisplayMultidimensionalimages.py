import matplotlib.pyplot as plt
import numpy as np

def display_images(image, a_values, b_values):
    """
    Diplay images from 4 dimensions

    Parameters:
        images (numpy.ndarray): The input images with shape (N, C, H, W), where
                                N is the number of images, C is the number of channels,
                                H is height, and W is width.
        a_values and b_vales:   Are both arrays of the images that we want to display

    Returns:
        subplot a times b images.
    """


    plt.figure(figsize=(15, 15))  # Adjust the figure size as needed

    # Number of rows and columns based on a_values and b_values
    num_rows = len(a_values)
    num_cols = len(b_values)

    for i, a in enumerate(a_values):
        for j, b in enumerate(b_values):
            plt.subplot(num_rows, num_cols, i * num_cols + j + 1)  # Create subplots based on rows and cols
            plt.imshow(np.log(np.abs(image[a][b]+1e-12)), cmap='gray')
            plt.title(f'slice={a}, Coil={b}', fontsize=12)

    plt.tight_layout()
    plt.show()