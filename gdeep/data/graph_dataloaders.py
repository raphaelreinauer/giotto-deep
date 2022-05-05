from typing import Tuple, Optional

import numpy as np
from sklearn.model_selection import train_test_split
import torch

from gdeep.data.graph_datasets import PersistenceDiagramFromGraphDataset
from gdeep.data.preprocessed_persistence_diagrams import \
    _create_preprocessing_transform

def create_dataloaders(dataset_name: str,
                       diffusion_parameter: float = 10.0,
                       batch_size: int = 32,
                       test_size = 0.2,
                       random_state: int = 42,
                       normalize_persistence_diagrams: bool = True,
                       num_points_to_keep: Optional[int] = None,
                       ) -> Tuple[torch.utils.data.DataLoader,
                                  torch.utils.data.DataLoader]:
    """ Create the dataloaders for the dataset.

    Args:
        dataset_name: The name of the graph dataset to load, e.g. "MUTAG".
        diffusion_parameter: The diffusion parameter of the heat kernel
            signature. These are usually chosen to be as {0.1, 1.0, 10.0}.
        batch_size: The batch size of the dataloaders.
        test_size: The test size of the train and test split.

    Returns:
        train_loader: The dataloader for the training set.
        test_loader: The dataloader for the test set.
    """
    # Load the dataset
    dataset = PersistenceDiagramFromGraphDataset(
        dataset_name,
        diffusion_parameter
    )

    # Create train and test splits
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=test_size,
        random_state=random_state
    )
    
    if normalize_persistence_diagrams or num_points_to_keep is not None:
        # Create the transform to normalize the persistence diagrams
        transform = _create_preprocessing_transform(
            dataset,
            train_idx,
            normalize_features=normalize_persistence_diagrams,
            )
    dataset = TransformableDataset(dataset, transform)
    
    min_max_transform: Transform = MinMaxScaler()
    dataset = TransformableDataset(dataset, min_max_transform)
    

    # Create the dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idx),
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(test_idx),
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    return train_loader, test_loader


