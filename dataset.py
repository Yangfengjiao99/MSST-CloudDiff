import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MultiGranularityDataset(Dataset):
    """
    A dataset class that handles multi-granularity time series data.
    It loads the original data and dynamically generates coarse-grained data for each request.
    """

    def __init__(self, mg_dict, use_index_list=None, dataset_type='train', seed=0, config=None):
        """
        Args:
            mg_dict (dict): A dictionary specifying the coarse granularities.
                            e.g., {'gran_2': 2, 'gran_5': 5} means two levels of coarsening with factors 2 and 5.
            use_index_list (list, optional): Indices of the original data to use for this dataset split.
            dataset_type (str, optional): Type of dataset (e.g., 'train', 'valid', 'test').
            seed (int, optional): Random seed for reproducibility.
            config (dict, optional): Configuration dictionary containing paths and parameters.
        """
        self.config = config
        self.dataset_type = dataset_type
        self.seed = seed
        self.mg_dict = mg_dict

        # Extract configuration parameters
        self.eval_length = config['others']['eval_length']
        self.feature_num = config['others']['feature_num']
        dir_dataset = config['others']['dir_dataset']
        # Set random seed
        np.random.seed(seed)
        path_observed_values = os.path.join(dir_dataset, 'data_20_12_5.pkl')
        path_observed_masks = os.path.join(dir_dataset, 'mask_20_12_5_1.pkl')
        path_gt_masks = os.path.join(dir_dataset, 'mask_20_12_5_10.pkl')
        with open(path_observed_values, "rb") as f:
            self.observed_values = pickle.load(f).astype(float)
        with open(path_observed_masks, "rb") as f:
            self.observed_masks = pickle.load(f).astype(float)
        with open(path_gt_masks, "rb") as f:
            self.gt_masks = pickle.load(f).astype(float)
        num_samples_to_use = 3000
        self.observed_values = self.observed_values[:num_samples_to_use]
        self.observed_masks = self.observed_masks[:num_samples_to_use]
        self.gt_masks = self.gt_masks[:num_samples_to_use]
        # Determine which indices of the original data to use
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def _coarsen_array(self, arr, gran):
        """
        Internal helper function to coarsen a 1D array by averaging over 'gran' elements.
        The output array is aligned in length with the original array by repeating the coarse values.

        Args:
            arr (np.ndarray): Input 1D array (e.g., a time series of one feature).
            gran (int): Coarsening factor (must be > 0).

        Returns:
            np.ndarray: Coarsened array with the same length as input.
        """
        if gran <= 1:
            return arr.copy()

        length = len(arr)
        # Calculate the length after truncation to be divisible by gran
        cut_index = length - (length % gran)

        # Coarsen the first part of the array
        arr_before = arr[:cut_index].reshape(-1, gran)
        arr_before_mean = np.mean(arr_before, axis=1)
        mean_arr_align = np.repeat(arr_before_mean, gran)

        # Handle the remaining part if the length is not divisible by gran
        if length % gran != 0:
            arr_after = arr[cut_index:]
            arr_after_mean = np.mean(arr_after)
            mean_arr_align_after = np.full_like(arr_after, arr_after_mean)
            mean_arr_align = np.concatenate((mean_arr_align, mean_arr_align_after))

        return mean_arr_align
    def __getitem__(self, org_index):
        """
        Returns a dictionary containing the original and all coarse-grained versions of the data.

        The returned dictionary structure is:
        {
            "observed_data": {
                "fine": ...,          # Original fine-grained data
                "gran_2": ...,        # Coarse-grained with factor 2
                "gran_5": ...         # Coarse-grained with factor 5
            },
            "observed_mask": { ... }, # Same structure as observed_data
            "gt_mask": { ... },       # Same structure as observed_data
            "timepoints": np.ndarray  # Original timepoints
        }
        """
        index = self.use_index_list[org_index]
        # Prepare the base data for this sample
        observed_data_base = self.observed_values[index]
        observed_mask_base = self.observed_masks[index]
        gt_mask_base = self.gt_masks[index]
        timepoints = np.arange(self.eval_length)



        sample = {}


        sample['observed_data'] = observed_data_base
        sample['observed_mask'] = observed_mask_base
        sample['gt_mask'] = gt_mask_base
        sample['timepoints'] = timepoints

        for gran_name, gran_value in self.mg_dict.items():

            observed_data_coarse = np.array([self._coarsen_array(feat, gran_value) for feat in observed_data_base.T]).T

            sample[f'observed_data_{gran_name}'] = observed_data_coarse


        return sample


    def __len__(self):
        return len(self.use_index_list)

def get_dataloader(mg_dict, seed=1, batch_size=8, config=None):
    """
    Creates DataLoaders for training, validation, and test sets using the MultiGranularityDataset.

    Args:
        mg_dict (dict): A dictionary specifying the coarse granularities.
        seed (int, optional): Random seed for reproducible train/val/test splits.
        batch_size (int, optional): Batch size for the DataLoaders.
        config (dict, optional): Configuration dictionary.

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    full_dataset = MultiGranularityDataset(mg_dict=mg_dict, dataset_type='full', seed=seed, config=config)
    # Generate a shuffled list of indices
    indlist = np.arange(len(full_dataset))
    np.random.seed(seed)
    np.random.shuffle(indlist)

    # Calculate split indices
    train_end = int(0.8 * len(full_dataset))
    val_end = train_end + int(0.1 * len(full_dataset))

    train_index = indlist[:train_end]
    valid_index = indlist[train_end:val_end]
    test_index = indlist[val_end:]

    # Create dataset splits with the appropriate indices
    train_dataset = MultiGranularityDataset(mg_dict=mg_dict, use_index_list=train_index, dataset_type='train',
                                            seed=seed, config=config)
    valid_dataset = MultiGranularityDataset(mg_dict=mg_dict, use_index_list=valid_index, dataset_type='valid',
                                            seed=seed, config=config)
    test_dataset = MultiGranularityDataset(mg_dict=mg_dict, use_index_list=test_index, dataset_type='test', seed=seed,
                                           config=config)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)

    return train_loader, valid_loader, test_loader




