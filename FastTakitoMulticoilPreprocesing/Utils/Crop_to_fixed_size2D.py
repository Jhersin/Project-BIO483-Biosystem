import torch
import torch.nn.functional as F

def crop_to_fixed_size(images: torch.Tensor, target_size=(320, 320)):
    """
    Crop or resize an image or a batch of images to a fixed size of 320x320 pixels.

    Parameters:
        images (torch.Tensor): The input image(s) with shape (N, H, W) or (H, W), where
                               N is the number of images,
                               H is height, and W is width.
        target_size (tuple): The desired crop size (height, width).

    Returns:
        torch.Tensor: The cropped or resized images with shape (N, target_height, target_width) or (target_height, target_width).
    """
    target_height, target_width = target_size

    # Check if the input is a single image or a batch
    if images.dim() == 2:  # Single image (H, W)
        images = images.unsqueeze(0)  # Add a batch dimension (1, H, W)

    batch_size, height, width = images.shape

    # Initialize an empty tensor to hold the resized images
    resized_images = torch.zeros((batch_size, target_height, target_width), dtype=images.dtype, device=images.device)

    for i in range(batch_size):
        image = images[i]  # Get the individual image

        # If the image is smaller than the target size, resize it
        if image.shape[0] < target_height or image.shape[1] < target_width:
            resized_image = F.interpolate(image.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
            resized_images[i] = resized_image.squeeze(0).squeeze(0)  # Remove the added dimensions
        else:
            # Calculate the coordinates for cropping
            start_x = (width - target_width) // 2
            start_y = (height - target_height) // 2

            # Crop the image
            cropped_image = image[start_y:start_y + target_height, start_x:start_x + target_width]
            resized_images[i] = cropped_image

    # If the input was a single image, return without batch dimension
    return resized_images.squeeze(0) if resized_images.shape[0] == 1 else resized_images
