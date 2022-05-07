import torch
import torch.nn as nn

from gdeep.utility import FTensor, ITensor

class FourierMixer(nn.Module):
    """
    The Fourier Mixer as described in
    
    "FNet: Mixing Tokens with Fourier Transforms"
        Lee-Thorp et al., 2021, https://arxiv.org/pdf/2105.03824.pdf

    Args:
        dropout: The dropout probability.
    """
    
    def __init__(self,
                 dropout: float=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout, inplace=False)
        
    def forward(self,
                x: FTensor,
                mask: ITensor) -> FTensor:
        """
        Forward pass of the Fourier Mixer layer.
        
        Args:
            x: The input tensor.
        """
        x = torch.fft.fft2(x).real
        return self.dropout(x)