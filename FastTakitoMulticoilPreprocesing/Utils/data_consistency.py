import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.masking import _mask_torch

class MultiplyScalar(nn.Module):
    def __init__(self):
        super(MultiplyScalar, self).__init__()
        self.sample_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.complex(self.sample_weight.item(), 0) * x

def _replace_values_on_mask(x):
    # TODO: check in multicoil case
    cnn_fft, kspace_input, mask = x
    anti_mask = (1.0 - mask).unsqueeze(-1)
    replace_cnn_fft = anti_mask * cnn_fft + kspace_input
    return replace_cnn_fft

def enforce_kspace_data_consistency(kspace, kspace_input, mask, input_size, multiply_scalar=None, noiseless=True):
    if noiseless:
        data_consistent_kspace = _replace_values_on_mask([kspace, kspace_input, mask])
    else:
        if multiply_scalar is None:
            multiply_scalar = MultiplyScalar()
        kspace_masked = -_mask_torch(kspace, mask)
        data_consistent_kspace = kspace_input + kspace_masked
        data_consistent_kspace = multiply_scalar(data_consistent_kspace)
        data_consistent_kspace = data_consistent_kspace + kspace
    return data_consistent_kspace
