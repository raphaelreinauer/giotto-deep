from typing import Callable, List
import torch
import torchvision.transforms as transforms

Module = torch.nn.Module
Tensor = torch.Tensor

class NormalizePersistenceDiagram(Module):
    """
    Normalize persistence diagrams.
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: The input tensor.
        """
        return (x - self.mean) / self.std
    
def composition_transforms(transforms: List[Callable[[Tensor], Tensor]])\
    -> Callable[[Tensor], Tensor]:
    """
    Compose multiple transforms.
     
    Args:
        transforms: The transforms to compose.
        
    Returns:
        The compose transform.
    """
    def transform(x: Tensor) -> Tensor:
        for transform in transforms:
            x = transform(x)
        return x
    return transform
    
         
    
def _sort_diagram_by_lifetime(diagram: Tensor) -> Tensor:
    """
    Sort a single persistence diagram by the lifetime.

    Args:
        diagram (Tensor): The persistence diagram to sort. The shape is
            (n_points, 2 + number_homology_types).

    Returns:
        Tensor: The sorted persistence diagram.
    """
    return diagram[
                (diagram[:, 1] -
                    diagram[:, 0]).argsort()
            ]
    
def keep_k_most_persistent_points(x: Tensor, k: int) -> Tensor:
    """
    Keep the k-most persistent points in the persistence diagrams.
     
    Args:
        Tensor: The persistence diagrams.
        k: The number of the most persistent points to keep.
         
    Returns:
        The persistence diagrams with the k-most persistent points kept.
    """
    # Find the indices of the k-most persistent points
    indices: Tensor = torch.topk(
        x[:, 1] - x[:, 0], k, largest=True, sorted=False
        ).indices
    
    return x[indices]
    