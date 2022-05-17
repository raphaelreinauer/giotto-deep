from enum import Enum, auto
from typing import Dict
from transformers.configuration_utils import PretrainedConfig
from torch.nn import Module, MultiheadAttention

class LayerNormStyle(Enum):
    """
    The style of layer normalization.
    """
    NO_LAYER_NORMALIZATION = auto()
    PRE_LAYER_NORMALIZATION = auto()
    POST_LAYER_NORMALIZATION = auto()
    
class AttentionType(Enum):
    """
    The type of attention.
    """
    NO_ATTENTION = auto()
    DOT_PRODUCT = auto()
    UNNORMALIZED_DOT_PRODUCT = auto()
    INDUCED_ATTENTION = auto()
    FOURIER_MIXER = auto()

    

class PersformerConfig(PretrainedConfig):
    """
    Configuration class to define a persformer model.
    
    Examples:
    ```python
    >>> from gdeep.topological_layers import PersformerConfig, PersformerModel
    
    # Initialize the configuration object
    >>> config = PersformerConfig()
    
    # Initialize the model
    >>> model = PersformerModel(config)
    
    # Access the configuration object
    >>> config = model.config
    
    ```
    """
    
    input_size: int # input size of the model
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    hidden_act: str
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    layer_norm_eps: float
    classifier_dropout_prob: float
    layer_norm_style: LayerNormStyle
    attention_type: AttentionType
    
    def __init__(self,
                 input_size: int = 4,
                 hidden_size: int = 32,
                 num_hidden_layers: int = 2,
                 num_attention_heads: int = 4,
                 intermediate_size: int = 32,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 layer_norm_eps: float = 1e-12,
                 classifier_dropout_prob: float = 0.1,
                 use_layer_norm: LayerNormStyle = \
                     LayerNormStyle.NO_LAYER_NORMALIZATION,
                 attention_type: AttentionType = \
                     AttentionType.DOT_PRODUCT
                 **kwargs  # type: ignore
                 ):
        super().__init__(**kwargs)  # type: ignore
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout_prob = classifier_dropout_prob
        self.layer_norm_style = use_layer_norm
        self.attention_type = attention_type
    
class AttentionFactory():
    """
    Factory for creating attention modules.
    """
    attention_modules: Dict[AttentionType, Module] = {}
    
    def __init__(self,
                 config: PretrainedConfig):
        self.config = config
    
    def register_attention(self,
                           attention_type: AttentionType,
                           attention_module: Module):
        """
        Register an attention module.
        """
        self.attention_modules[attention_type] = attention_module
        
    def create_attention(self,
                         attention_type: AttentionType):
            """
            Create an attention module.
            """
            return self.attention_modules[attention_type](self.config)
        

class DotProductAttention(Module):
    """
    Dot product attention.
    """
    def __init__(self,
                 config: PersformerConfig):
        super().__init__()
        self.config = config
        
        self.dot_product_attention = \
            MultiheadAttention(embed_dim=config.hidden_size,
                               num_heads=config.num_attention_heads,
                               dropout=config.attention_probs_dropout_prob,
                               batch_first=True)
        
     