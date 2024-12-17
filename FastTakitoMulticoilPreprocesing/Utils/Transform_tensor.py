import torch

def transform_kspace_to_image_tensor(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions

    :param k: k-space data (PyTorch tensor)
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = list(range(k.ndim))

    img = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(k, dim=dim), s=img_shape, dim=dim), dim=dim)
    img *= torch.sqrt(torch.prod(torch.tensor([k.shape[d] for d in dim])))
    return img


def transform_image_to_kspace_tensor(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions

    :param img: image space data (PyTorch tensor)
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = list(range(img.ndim))

    k = torch.fft.ifftshift(torch.fft.fftn(torch.fft.fftshift(img, dim=dim), s=k_shape, dim=dim), dim=dim)
    k /= torch.sqrt(torch.prod(torch.tensor([img.shape[d] for d in dim])))
    return k