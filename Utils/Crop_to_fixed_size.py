import cv2


def crop_to_fixed_size(image, target_size=(320, 320)):
    """
    Crop or resize the image to a fixed size of 320x320 pixels.

    Parameters:
        image (numpy.ndarray): The input image.
        target_size (tuple): The desired crop size (width, height).

    Returns:
        numpy.ndarray: The cropped or resized image.
    """
    height, width = image.shape[:2]
    target_width, target_height = target_size

    # If the image is smaller than the target size, resize it
    if height < target_height or width < target_width:
        return cv2.resize(image, target_size)

    # Calculate the coordinates for cropping
    start_x = (width - target_width) // 2
    start_y = (height - target_height) // 2

    # Crop the image
    cropped_image = image[start_y:start_y + target_height, start_x:start_x + target_width]

    return cropped_image
