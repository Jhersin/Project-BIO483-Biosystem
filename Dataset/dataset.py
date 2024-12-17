from pathlib import Path
import torch
import numpy as np

class KneeDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.all_files = self.extract_files(root)

    @staticmethod
    def extract_files(root):
        """
        Extract the paths to all slices given the root path (ends with train or val)
        """
        files = []
        for subject in root.glob("*"):   # Iterate over the subjects
            slice_path = subject/"high_resolution"  # Get the slices for current subject
            for slice in slice_path.glob("*.npy"):
                files.append(slice)
        return files

    @staticmethod
    def change_img_to_label_path(path):
        """
        Replace data with mask to get the masks
        """
        parts = list(path.parts)
        parts[parts.index("high_resolution")] = "low_resolution"
        return Path(*parts)


    def standardize(self, normalized_data):
        """
        Standardize the normalized data into the 0-1 range
        """
        standardized_data = (normalized_data - normalized_data.min()) / (normalized_data.max() - normalized_data.min())
        return standardized_data

    def __len__(self):
        """
        Return the length of the dataset (length of all files)
        """
        return len(self.all_files)


    def __getitem__(self, idx):
        """
        Given an index return the (augmented) slice and corresponding mask
        Add another dimension for pytorch
        """
        mri_HR_path = self.all_files[idx]
        mri_LR_path = self.change_img_to_label_path(mri_HR_path)
        mri_HR = np.load(mri_HR_path)  # Convert to float for torch
        mri_LR = np.load(mri_LR_path)
        mri_HR_norm = self.standardize(mri_HR)
        mri_LR_norm = self.standardize(mri_LR)

        # Note that pytorch expects the input of shape BxCxHxW, where B corresponds to the batch size, C to the channels, H to the height and W to Width.
        # As our data is of shape (HxW) we need to manually add the C axis by using expand_dims.
        # The batch dimension is later added by the dataloader

        return np.expand_dims(mri_HR_norm,0), np.expand_dims(mri_LR_norm,0)
