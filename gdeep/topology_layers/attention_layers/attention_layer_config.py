from dataclasses import dataclass
from enum import Enum
from dataclasses import dataclass

class AttentionType(Enum):
    """
    A class to define attention types.
    """
    SELF_ATTENTION = "self_attention"
    SPARSE_ATTENTION = "sparse_attention"
    FOURIER_MIXXER = "fourier_mixer"
    

@dataclass
class AttentionLayerConfig:
    """
    Configuration class to define a multi-head attention layer.
    
    Args:
        dim_model: The dimension of the model.
        num_heads: The number of heads.
        activation: The activation function to use.
        dropout: The dropout probability.
        layer_norm: Whether to use layer normalization.
    """
    dim_model: int
    num_heads: int
    bias: bool
    attention_type: AttentionType
    
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        bias: bool = True,
        attention_type: AttentionType = AttentionType.SELF_ATTENTION,
        ) -> None:
        
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.bias = bias
        self.attention_type = attention_type