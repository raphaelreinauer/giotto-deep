from typing import Any, Callable, Tuple, Sized

from torch.utils.data import Dataset

class TransformingDataset(Dataset[Any]):
    dataset: Dataset[Any]
    transform: Callable[[Any], Any]
    
    def __init__(self, dataset: Dataset[Any], transform:Callable[[Any], Any]) -> None:
        self.dataset = dataset
        self.transform = transform
        
    def append_transform(self, transform: Callable[[Any], Any]) -> None:
        self.transform = lambda x: transform(self.transform(x))
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        return self.transform(self.dataset[idx][0]), self.dataset[idx][0]
    
    def __len__(self) -> int:
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        else:
            raise NotImplementedError("The dataset does not implement __len__")
    
