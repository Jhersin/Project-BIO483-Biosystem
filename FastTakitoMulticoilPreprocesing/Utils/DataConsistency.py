import torch
from Utils.Transform_tensor import transform_image_to_kspace, transform_kspace_to_image



def dataconsitency(fourier_mask, fourier_mask_inv, lr_image, hr_image):
    lr_kspace = transform_image_to_kspace(lr_image, dim=(2, 3))
    hr_kspace = transform_image_to_kspace(hr_image, dim=(2, 3))

    # The mask is received in fastmri format, we make a 1 and 0 mask in the fourier space
    masked_kspace_low_res = fourier_mask[..., None] * lr_kspace
    masked_kspace_high_res = fourier_mask_inv[..., None] * hr_kspace
    kspace_consitencia = masked_kspace_low_res + masked_kspace_high_res

    # Convert the result back to image space
    imagen_consistencia = transform_kspace_to_image(kspace_consitencia, dim=(2, 3))
    imagen_consistencia = torch.abs(imagen_consistencia)
    return imagen_consistencia