# from https://github.com/juho-lee/set_transformer/blob/master/max_regression_demo.ipynb  # noqa: E501
#### Author: Raphael Reinauer

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.dropout import Dropout
from gdeep.topology_layers.modules import ISAB, PMA, SAB, FastAttention  # type: ignore


class SmallDeepSet(nn.Module):
    def __init__(self, pool="max"):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )
        self.pool = pool

    def forward(self, x):
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        x = self.dec(x)
        return x


class SetTransformer(nn.Module):
    """ SetTransformer from
    https://github.com/juho-lee/set_transformer/blob/master/main_pointcloud.py
    with either induced attention or classical self attention.
    """
    def __init__(
        self,
        dim_input=3,
        num_outputs=1,
        dim_output=40,  # number of classes
        dim_hidden=128,
        num_heads=4,
        ln=False,  # use layer norm
        n_layers=1,
        attention_type="self_attention",
        dropout=0.0
    ):
        """Init of SetTransformer.

        Args:
            dim_input (int, optional): Dimension of input data for each
            element in the set. Defaults to 3.
            num_outputs (int, optional): Number of outputs. Defaults to 1.
            dim_output (int, optional): Number of classes. Defaults to 40.
            dim_hidden (int, optional): Number of induced points, see  Set
                Transformer paper. Defaults to 128.
            num_heads (int, optional): Number of attention heads. Defaults
                to 4.
            ln (bool, optional): If `True` layer norm will not be applied.
                Defaults to False.
        """
        super().__init__()

        assert dim_hidden % num_heads == 0, \
            "Number of hidden dimensions must be divisible by number of heads."

        self.emb = SAB(dim_input, dim_hidden, num_heads, ln=ln)

        self.n_layers = n_layers
        if attention_type == "induced_attention":
            self.enc_list = nn.ModuleList([
                nn.Sequential(
                ISAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Dropout(p=dropout)
                )
                for _ in range(n_layers)
            ])
        elif attention_type == "self_attention":
            self.enc_list = nn.ModuleList([
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
                for _ in range(n_layers)
            ])
        elif attention_type == "fast_attention":
            self.enc_list = nn.ModuleList([
                nn.Sequential(
                    FastAttention(dim_hidden, heads=num_heads, dim_head=64),
                    nn.Dropout(p=dropout),
                )
                for _ in range(n_layers)
            ])
        else:
            raise ValueError("Unknown attention type:", attention_type)
        self.dec = nn.Sequential(
            nn.Dropout(p=dropout),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(p=dropout),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): Batch tensor of shape
                [batch, sequence_length, dim_in]

        Returns:
            torch.Tensor: Tensor with predictions of the `dim_output` classes.
        """
        x = self.emb(x)
        for l in self.enc_list:
            x = l(x)
        x = self.dec(x)
        return x.squeeze()

    @property
    def num_params(self) -> int:
        """Returns number of trainable parameters.

        Returns:
            int: Number of trainable parameters.
        """
        total_params = 0
        for parameter in self.parameters():
            total_params += parameter.nelement()
        return total_params

class GraphClassifier(nn.Module):
    """Classifier for Graphs using persistence features and additional
    features. The vectorization is based on a set transformer.
    """
    def __init__(self,
                 num_features,
                 dim_input=6,
                 num_outputs=1,
                 dim_output=50,
                 num_classes=2,
                 ln=True,
                 num_heads=4,
                 use_induced_attention=False,
                 dim_hidden=128,
                 dropout=0.0,
                ):
        super().__init__()
        self.st = SetTransformer(
            dim_input=dim_input,
            num_outputs=num_outputs,
            dim_output=dim_output,
            ln=ln,
            num_heads=num_heads,
            dim_hidden=dim_hidden,
            use_induced_attention=use_induced_attention,
            dropout=dropout,
            )
        self.num_classes = num_classes
        self.ln = nn.LayerNorm(dim_output + num_features)
        self.ff_1 = nn.Linear(dim_output + num_features, 50)
        self.ff_2 = nn.Linear(50, 20)
        self.ff_3 = nn.Linear(20, num_classes)

    def forward(self, x_pd: Tensor, x_feature: Tensor) -> Tensor:
        """Forward pass of the graph classifier.
        The persistence features are encoded with a set transformer
        and concatenated with the feature vector. These concatenated
        features are used for classification using a fully connected
        feed -forward layer.

        Args:
            x_pd (Tensor): persistence diagrams of the graph
            x_feature (Tensor): additional graph features
        """
        pd_vector = self.st(x_pd)
        features_stacked = torch.hstack((pd_vector, x_feature))
        x = self.ln(features_stacked)
        x = nn.ReLU()(self.ff_1(x))
        x = nn.ReLU()(self.ff_2(x))
        x = self.ff_3(x)
        return x