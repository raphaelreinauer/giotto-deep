# %%
from cProfile import label
from dataclasses import dataclass
import os
from typing import List, Sequence, Tuple
from numpy import ndarray

import torch
import torch.nn as nn
from gdeep.data import PreprocessingPipeline
from gdeep.data.datasets import PersistenceDiagramFromFiles
from gdeep.data.datasets.base_dataloaders import (DataLoaderBuilder,
                                                  DataLoaderParamsTuples)
from gdeep.data.datasets.persistence_diagrams_from_graphs_builder import \
    PersistenceDiagramFromGraphBuilder
from gdeep.data.persistence_diagrams.one_hot_persistence_diagram import (
    OneHotEncodedPersistenceDiagram, collate_fn_persistence_diagrams)
from gdeep.data.preprocessors import (
    FilterPersistenceDiagramByHomologyDimension,
    FilterPersistenceDiagramByLifetime, NormalizationPersistenceDiagram)
from gdeep.search.hpo import GiottoSummaryWriter
from gdeep.topology_layers import Persformer, PersformerConfig, PersformerWrapper
from gdeep.topology_layers.persformer_config import PoolerType
from gdeep.trainer.trainer import Trainer
from gdeep.search import HyperParameterOptimization
from gdeep.utility import DEFAULT_GRAPH_DIR, PoolerType
from gdeep.utility.constants import DEFAULT_DATA_DIR
from gdeep.utility.utils import autoreload_if_notebook
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import Subset
from torch.utils.tensorboard.writer import SummaryWriter
from gdeep.data.datasets import OrbitsGenerator, DataLoaderKwargs, Orbit5kConfig

autoreload_if_notebook()

# %%

config_data = Orbit5kConfig(
    num_orbits_per_class=1000,
)
    

og = OrbitsGenerator.from_config(config_data)

dgs: ndarray = og.get_persistence_diagrams()  # type: ignore
labels: ndarray = og.get_labels()

output_dir = os.path.join(DEFAULT_DATA_DIR, "orbits_5k")

label_list: List[Tuple[int, int]] = []
for dg_idx in range(len(dgs)):
    dg_one_hot = OneHotEncodedPersistenceDiagram(dgs[dg_idx])
    
    
    label_list.append((dg_idx, labels[dg_idx]))



# %%

# Define the data loader

dataloaders_dicts = DataLoaderKwargs(
    train_kwargs={"batch_size": config_data.batch_size_train,},
    val_kwargs={"batch_size": 4},
    test_kwargs={"batch_size": 3},
)


dl_train, _, _ = og.get_dataloader_persistence_diagrams(dataloaders_dicts)
    
model_config = PersformerConfig(
    input_size=2 + len(config_data.homology_dimensions), # there are 2 coordinates and 2 homology dimensions
    output_size=len(config_data.parameters),  # there are 5 classes
    hidden_size=64,
    intermediate_size=128,
    num_attention_layers=2,
    num_attention_heads=8,
)
# %%

model = Persformer(model_config)

writer = SummaryWriter()

loss_function =  nn.CrossEntropyLoss()

trainer = Trainer(model, [dl_train], loss_function, writer)

trainer.train(Adam, 10, False, 
              {"lr":0.01}, 
              {"batch_size":16})
    
    
# %%
# Define the model by using a Wrapper for the Persformer model

wrapped_model = PersformerWrapper(
    num_attention_layers=3,
    num_attention_heads=4,
    input_size= 2 + 2,
    ouptut_size=5,
    pooler_type=PoolerType.ATTENTION,
)
writer = GiottoSummaryWriter()

loss_function =  nn.CrossEntropyLoss()

trainer = Trainer(wrapped_model, [dl_train, dl_train], loss_function, writer)  # type: ignore

# initialise hpo object
search = HyperParameterOptimization(trainer, "accuracy", 2, best_not_last=True)

# if you want to store pickle files of the models instead of the state_dicts
search.store_pickle = True

# dictionaries of hyperparameters
optimizers_params = {"lr": [0.001, 0.01]}
dataloaders_params = {"batch_size": [2, 4, 2]}
models_hyperparams = {
    "input_size": [4],
    "output_size": [5],
    "num_attention_layers": [1, 2, 1],
    "num_attention_heads": [8, 16, 8],
    "hidden_size": [16],
    "intermediate_size": [16],
}

# %%
# starting the HPO
search.start(
    [Adam],
    3,
    False,
    optimizers_params,
    dataloaders_params,
    models_hyperparams,
)

