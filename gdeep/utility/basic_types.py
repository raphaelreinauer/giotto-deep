from typing import Union

import torch

FTensor = Union[torch.FloatTensor,
                torch.DoubleTensor,
                torch.HalfTensor,]

ITensor = Union[torch.IntTensor,
                torch.ShortTensor,
                torch.LongTensor,]