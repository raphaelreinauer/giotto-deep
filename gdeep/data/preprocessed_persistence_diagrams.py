from typing import NewType, List, Tuple, Callable, Union, Optional

import numpy as np
import torch

from gdeep.data import PersistenceDiagramDataset

Tensor = torch.Tensor
Array = np.ndarray

def _create_preprocessing_transform(dataset: PersistenceDiagramDataset,
                                   test_indices: Optional[Array] = None,
                                   normalize_features: bool = True,
                                   ) -> Callable[[Tensor], Tensor]:
    
    """
    Create a preprocessing transform for the data in the specified dataloader.
    
    Args:
        dataloader(DataLoader): The dataloader to preprocess.
        test_indices(Optional[Array]): The indices of the test set. If None,
            the whole dataset is used.
        normalize_features(bool): Whether to normalize the features.
        num_points_to_keep(Optional[int]): The number of the most persistent
            points to keep.
        
    Returns:
        The preprocessing transform.
    """
    # Find the mean and standard deviation of the features
    if normalize_features:
        features_mean: float = torch.mean(
            torch.stack([x[:2] for x in dataset[test_indices]], dim=0), dim=0
        ).item()
        features_std: float = torch.std(
            torch.stack([x[:2] for x in dataset[test_indices]], dim=0), dim=0
        ).item()
    
    # Create the preprocessing transform
    def transform(x: Tensor) -> Tensor:
        """
        Transform the input data.
        
        Args:
            x: The input data.
        """
        # Normalize the features
        if normalize_features:
            x[:, :2] = (x[:, :2] - features_mean) / features_std
        
        return x
    
    return transform
    
