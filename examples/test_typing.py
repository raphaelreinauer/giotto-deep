#%%

class C:
    i: int
    j: int = 0
    
    def __init__(self, i: int) -> None:
        self.i = i
        self.j += 1
        print(type(self))
    
x = C(1)
y = C(2)

print(x.i)
print(y.i)
# %%
C.i
# %%
from typing import Union, List

import torch

FloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
IntTensor = Union[torch.IntTensor, torch.cuda.IntTensor]

def get_tensor_by_indices(
    tensor: FloatTensor,
    indices: IntTensor,
    ) -> FloatTensor:
    """
    Get the values of a tensor by indices.
    
    Args:
        tensor: The tensor to get the values from.
        indices: The indices to get the values from.
         
    Returns:
        The values of the tensor at the indices.
    """
    # check if tensor is contiguous
    assert tensor.is_contiguous(), "tensor must be contiguous"
    
    flattened_indices: List[int] = []
    stride: int = tensor.shape[1]
    for i in range(indices.shape[0]):
        flattened_indices.append(
            (indices[i, 0] * stride + indices[i, 1]).item()
            )
    return torch.flatten(tensor)[flattened_indices]

tensor = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [4.0, 5.0, 6.0]])
indices = torch.tensor([[0, 1], [1, 2]])

assert torch.allclose(
    get_tensor_by_indices(tensor[[0, 2]], indices),
    torch.tensor([2.0, 6.0])
)
# %%
