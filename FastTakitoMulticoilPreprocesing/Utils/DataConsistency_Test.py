import torch
from Utils.Transform_tensor import transform_image_to_kspace_tensor, transform_kspace_to_image_tensor



def dataconsitency(fourier_mask, fourier_mask_inv, lr_image, hr_image,shape):
    lr_kspace = lr_image
    hr_kspace = transform_image_to_kspace_tensor(hr_image, dim=(2, 3),k_shape=shape)
    print('hr_kspace',hr_kspace.shape)

    # The mask is received in fastmri format, we make a 1 and 0 mask in the fourier space
    masked_kspace_low_res = fourier_mask[..., None] * lr_kspace
    masked_kspace_high_res = fourier_mask_inv[..., None] * hr_kspace
    kspace_consitencia = masked_kspace_low_res + masked_kspace_high_res
    print('kspace_consitencia',kspace_consitencia.shape)
    # Convert the result back to image space
    imagen_consistencia = transform_kspace_to_image_tensor(kspace_consitencia, dim=(2, 3), img_shape=(320,320))
    imagen_consistencia = torch.abs(imagen_consistencia)
    return imagen_consistencia