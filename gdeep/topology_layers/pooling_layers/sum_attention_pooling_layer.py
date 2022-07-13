from typing import Optional

import torch
from torch.nn import Module

from ..persformer_config import PersformerConfig

# Type aliases
Tensor = torch.Tensor

class SumAttentionPoolingLayer(Module):
        
        config: PersformerConfig
        
        def __init__(self, config: PersformerConfig):
            super().__init__()
            self.config = config
            
        def forward(self,
                    input_batch: Tensor,
                    attention_mask: Optional[Tensor] = None
                    ) -> Tensor:
            raise NotImplementedError