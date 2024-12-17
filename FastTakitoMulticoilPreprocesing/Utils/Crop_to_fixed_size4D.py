import cv2
import numpy as np


def crop_to_fixed_size(images, target_size=(320, 320)):
    """
    Crop or resize each image in a batch to a fixed size of 320x320 pixels.

    Parameters:
        images (numpy.ndarray): The input images with shape (N, C, H, W), where
                                N is the number of images, C is the number of channels,
                                H is height, and W is width.
        target_size (tuple): The desired crop size (height, width).

    Returns:
        numpy.ndarray: The cropped or resized images with shape (N, C, target_height, target_width).
    """
    target_height, target_width = target_size
    batch_size, channels, height, width = images.shape

    # Initialize an array to hold the resized images
    resized_images = np.zeros((batch_size, channels, target_height, target_width), dtype=images.dtype)

    for i in range(batch_size):
        image = images[i]  # Get the individual image

        # If the image is smaller than the target size, resize it
        if image.shape[1] < target_height or image.shape[2] < target_width:
            resized_images[i] = cv2.resize(image, (target_width, target_height))
        else:
            # Calculate the coordinates for cropping
            start_x = (width - target_width) // 2
            start_y = (height - target_height) // 2

            # Crop the image
            cropped_image = image[:, start_y:start_y + target_height, start_x:start_x + target_width]
            resized_images[i] = cropped_image

    return resized_images
