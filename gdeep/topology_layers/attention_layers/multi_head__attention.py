from typing import Union, TypeVar, List, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

from gdeep.utils import FTensor, ITensor
        
        
class MultiHeadAttention(nn.Module):
    """
    A multi-head attention layer.
    
    Args:
        dim_model: The dimension of the model.
        num_heads: The number of heads.
        activation: The activation function to use.
        dropout: The dropout probability.
        layer_norm: Whether to use layer normalization.
    """
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
        ) -> None:
        
        super().__init__()
        
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout
        self.bias = bias
        
        self.multi_head_attention = MultiheadAttention(
            num_heads=num_heads,
            embed_dim=dim_model,
            dropout=dropout,
            bias=bias,
            )
        
    def forward(
        self,
        query: FTensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        ) -> Tensor:
        """
        Forward propagation of the multi-head attention layer.
        
        Args:
            query: The query tensor.
            key: The key tensor.
            value: The value tensor.
            mask: The mask tensor.
        
        Returns:
            The multi-head attention output tensor.
        """
        output = self.multi_head_attention(
            query,
            key,
            value,
            mask=mask,
            )
        
        return output