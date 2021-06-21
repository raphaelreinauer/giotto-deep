# %% [markdown]
# ## Benchmarking PersFormer on the graph datasets.
# We will compare the accuracy on the graph datasets of our SetTransformer
# based on PersFormer with the perslayer introduced in the paper:
# https://arxiv.org/abs/1904.09378

# %% [markdown]
# ## Benchmarking MUTAG
# We will compare the test accuracies of PersLay and PersFormer on the MUTAG
# dataset. It consists of 188 graphs categorised into two classes.
# We will train the PersFormer on the same input features as PersFormer to
# get a fair comparison.
# The features PersLay is trained on are the extended persistence diagrams of
# the vertices of the graph filtered by the heat kernel signature (HKS)
# at time t=10.
# The maximum (wrt to the architecture and the hyperparameters) mean test
# accuracy of PersLay is 89.8(±0.9) and the train accuracy with the same
# model and the same hyperparameters is 92.3.
# They performed 10-fold evaluation, i.e. splitting the dataset into
# 10 equally-sized folds and then record the test accuracy of the i-th
# fold and training the model on the 9 other folds.

# %%
# Import libraries:
from typing import Tuple, Dict
import numpy as np  # typing: ignore
import random
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, dataset
from einops import rearrange  # typing: ignore
from os.path import join, isfile
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from gdeep.topology_layers import ISAB, PMA
# for loading extended persistence diagrams that are in hdf5 format
import h5py  # typing: ignore

# %%


def persistence_diagrams_to_sequence(
        tensor_dict: Dict[str, Dict[str, Tensor]]
     ):
    """Convert tensor dictionary to sequence of Tensors
        Output will be a List of tensors of the shape [graphs, absolute
        number of points per graph, 2(for the x and y coordinate)
        + number of types]

    Args:
        tensor_dict (Dict[str, Dict[str, Tensor]]): Dictionary of types and
            Dictionary of graphs and Tensors of points in the persistence
            diagrams.

    Returns:
        Dict[Int, Tensor]: List of tensors of the shape described above
    """
    raise DeprecationWarning("persistence_diagrams_to_sequence is deprecated")
    types = list(tensor_dict.keys())

    sequence_dict = {}

    def encode_points(graph_idx, type_idx, type_, n_pts):
        one_hot = F.one_hot(
                torch.tensor([type_idx] * n_pts),
                num_classes=len(types))
        return torch.cat([
                    tensor_dict[type_][str(graph_idx)],
                    one_hot.expand((n_pts, len(types)))
                ], axis=-1)

    for graph_idx in [int(k) for k in tensor_dict[types[0]].keys()]:
        tensor_list = []
        for type_idx, type_ in enumerate(types):
            n_pts = tensor_dict[type_][str(graph_idx)].shape[0]
            tensor_list.append(encode_points(graph_idx,
                                             type_idx,
                                             type_,
                                             n_pts))
        sequence_dict[graph_idx] = torch.cat(tensor_list, axis=0)
    return sequence_dict


# %%
# Load extended persistence diagrams and additional features
    

def load_data(
        dataset_: str = "MUTAG",
        path_dataset: str = "graph_data",
        verbose: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load dataset from files.

    Args:
        dataset (str, optional): File name of the dataset to load. There should
            be a hdf5 file for the extended persistence diagrams of the dataset
            as well as a csv file for the additional features in the path
            dataset directory. Defaults
            to "MUTAG".
        path_dataset (str, optional): Directory name of the dataset to load.
            Defaults to None.
        verbose (bool, optional): If `True` print size of the loaded dataset.
            Defaults to False.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of the loaded
            dataset consisting of the persistent features of the graphs, the
            additional features.
    """
    filenames = {}
    for file_suffix in [".hdf5", ".csv"]:
        try:
            filenames[file_suffix] = join(path_dataset,
                                          dataset_,
                                          dataset_ + file_suffix)
            assert(isfile(filenames[file_suffix]))
        except AssertionError:
            print(dataset_ + file_suffix +
                  " does not exist in given directory!")
    diagrams_file = h5py.File(filenames[".hdf5"], "r")
    # directory with persistance diagram type as keys
    # every directory corresponding to a key contains
    # subdirectories '0', '1', ... corresponding to the graphs.
    # For example, one can access a diagram by
    # diagrams_file['Ext1_10.0-hks']['1']
    # This is a hdf5 dataset object that contains the points of then
    # corresponding persistence diagram. These may contain different
    # numbers of points.

    persistence_array_dict: Dict[int, np.array] = {}
    # list of tensorised persistence diagrams

    additional_features = pd.read_csv(filenames[".csv"], index_col=0, header=0)
    labels = additional_features[['label']].values  # true labels of graphs
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels.reshape(-1))
    x_features = np.array(additional_features)[:, 1:]
    # additional graph features

    number_of_graphs = additional_features.shape[0]
    # number of graphs in the dataset

    # convert values in diagrams_file from numpy.ndarray to torch.tensor
    tensor_dict = {}

    for type_ in diagrams_file.keys():
        tensor_dict[type_] = {}
        for graph in diagrams_file[type_].keys():
            tensor_dict[type_][graph] = torch.tensor(
                                            diagrams_file[type_][graph]
                                            )

    # for pt_idx, persistence_type in enumerate(diagrams_file.keys()):
    #     diagram_dict = diagrams_file[persistence_type]
    #     temp_arr_dict: Dict[int, np.array] = {}
    #     # dictionary containing the graph index as key and the
    #     # points (as np.array) of the persistence diagram of type
    #     # `persistence_type` as value.

    #     # compute maximal number of points an store persistence points in
    #     # temp_arr_dict
    #     max_number_of_points = 0
    #     for graph_idx in diagrams_file[persistence_type].keys():
    #         pt_arr = np.array(diagram_dict[graph_idx])
    #         max_number_of_points = max(max_number_of_points, pt_arr.shape[0])
    #         temp_arr_dict[int(graph_idx)] = pt_arr
    #     persistence_array_dict[pt_idx] = np.zeros((
    #                                                 number_of_graphs,
    #                                                 max_number_of_points,
    #                                                 2
    #                                              ))
    #     # store all persistence points in temp_arr_dict in a single tensor
    #     # arrays will be filled by zeros to have a uniform tensor of shape
    #     # [number_of_graphs, max_number_of_points, 2]
    #     for graph_idx in temp_arr_dict.keys():
    #         persistence_array_dict[pt_idx] = temp_arr_dict[graph_idx]

    if verbose:
        print(
            "Dataset:", dataset_,
            "\nNumber of graphs:", number_of_graphs,
            "\nNumber of classes", label_encoder.classes_.shape[0]
            )
    return (persistence_diagrams_to_sequence(tensor_dict),
            torch.tensor(x_features),
            torch.tensor(y))

# def convert_to_one_hot(x: torch.Tensor, axis: int = 0) -> torch.Tensor:
#     """Convert tensor to one-hot representation along given axis.

#     Args:
#         x (torch.Tensor): Tensor to be converted.
#         axis (int, optional): Axis of the conversion. Defaults to 0.

#     Returns:
#         torch.Tensor: Tensor converted
#     """

#     return x


x_pd, x_features, y = load_data("MUTAG")
# %%
# Test persistence_diagrams_to_sequence with MUTAG dataset

def random_compatibility_test(n_trials: int = 10) -> None:
    """Randomly check if persistence diagram is correctly converted to
    sequence.

    Raises:
        Exception: Assertion error.

    Returns:
        None
    """
    filename = join("graph_data",
                    "MUTAG",
                    "MUTAG" + ".hdf5")
    diagrams_file = h5py.File(filename, "r")
    
    # load_data with `load_data` methode
    seq_pd, _ , _ = load_data("MUTAG")

    for _ in range(n_trials):

        type_ = random.choice(list(diagrams_file.keys()))

        type_index = list(diagrams_file.keys()).index(type_)


        graph_number = random.choice(list(diagrams_file[type_].keys()))

        # find indices that belong to type_
        idx = seq_pd[int(graph_number)][:, 2 + type_index] == 1.0

        computed_pts = seq_pd[int(graph_number)][idx][:, :2]


        original_pts = torch.tensor(diagrams_file[type_][graph_number])

        try:
            assert torch.allclose(original_pts, computed_pts)
        except AssertionError:
            raise AssertionError("persistence_diagrams_to_sequence does not" +
                                 "return the right sequence tensor")
            
random_compatibility_test()
# %%
# Test persistence_diagrams_to_sequence


tensor_dict = {"type1": {
                    "1": torch.tensor([[0.0066, 0.7961],
                                       [0.6612, 0.0359],
                                       [0.8394, 0.1597]]),
                    "2": torch.tensor([[0.1787, 0.1809],
                                       [0.2645, 0.5766],
                                       [0.5666, 0.1630],
                                       [0.9986, 0.0259]]),
                    "0": torch.tensor([[0.6910, 0.1265],
                                       [0.9085, 0.0230],
                                       [0.4977, 0.6386],
                                       [0.1331, 0.8196],
                                       [0.6929, 0.1859],
                                       [0.4216, 0.2283],
                                       [0.4996, 0.3380]]),
                    },
               "type2": {
                    "1": torch.tensor([[0.0932, 0.7327],
                                       [0.7248, 0.7940],
                                       [0.5550, 0.9960]]),
                    "2": torch.tensor([[0.9541, 0.6892],
                                       [0.7984, 0.8061],
                                       [0.5266, 0.0644],
                                       [0.0630, 0.2176]]),
                    "0": torch.tensor([[0.0896, 0.9181],
                                       [0.8755, 0.4239],
                                       [0.3665, 0.5990],
                                       [0.0960, 0.3615],
                                       [0.7895, 0.0670],
                                       [0.3407, 0.6902],
                                       [0.4052, 0.3058],
                                       [0.4820, 0.6540],
                                       [0.9083, 0.2075],
                                       [0.2015, 0.3533]])
                    }
               }

output = persistence_diagrams_to_sequence(tensor_dict)

expected_output = {1: torch.tensor(
        [[0.0066, 0.7961, 1.0000, 0.0000],
         [0.6612, 0.0359, 1.0000, 0.0000],
         [0.8394, 0.1597, 1.0000, 0.0000],
         [0.0932, 0.7327, 0.0000, 1.0000],
         [0.7248, 0.7940, 0.0000, 1.0000],
         [0.5550, 0.9960, 0.0000, 1.0000]]),
 2: torch.tensor(
        [[0.1787, 0.1809, 1.0000, 0.0000],
         [0.2645, 0.5766, 1.0000, 0.0000],
         [0.5666, 0.1630, 1.0000, 0.0000],
         [0.9986, 0.0259, 1.0000, 0.0000],
         [0.9541, 0.6892, 0.0000, 1.0000],
         [0.7984, 0.8061, 0.0000, 1.0000],
         [0.5266, 0.0644, 0.0000, 1.0000],
         [0.0630, 0.2176, 0.0000, 1.0000]]),
 0: torch.tensor(
        [[0.6910, 0.1265, 1.0000, 0.0000],
         [0.9085, 0.0230, 1.0000, 0.0000],
         [0.4977, 0.6386, 1.0000, 0.0000],
         [0.1331, 0.8196, 1.0000, 0.0000],
         [0.6929, 0.1859, 1.0000, 0.0000],
         [0.4216, 0.2283, 1.0000, 0.0000],
         [0.4996, 0.3380, 1.0000, 0.0000],
         [0.0896, 0.9181, 0.0000, 1.0000],
         [0.8755, 0.4239, 0.0000, 1.0000],
         [0.3665, 0.5990, 0.0000, 1.0000],
         [0.0960, 0.3615, 0.0000, 1.0000],
         [0.7895, 0.0670, 0.0000, 1.0000],
         [0.3407, 0.6902, 0.0000, 1.0000],
         [0.4052, 0.3058, 0.0000, 1.0000],
         [0.4820, 0.6540, 0.0000, 1.0000],
         [0.9083, 0.2075, 0.0000, 1.0000],
         [0.2015, 0.3533, 0.0000, 1.0000]])}

for i in range(3):
    try:
        assert(torch.allclose(output[i], expected_output[i]))
    except AssertionError:
        print("expected:\n", expected_output[i])
        print("actual:\n", output[i])
        raise Exception("persistence_diagrams_to_sequence does not match")


# %%

def diagram_to_tensor(
    tensor_dict_per_type: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
    """Convert dictionary of diagrams for fixed type to tensor representation
    with tailing zeros

    Args:
        tensor_dict (Dict[str, torch.Tensor]): Dictionary of persistence
            diagrams of a fixed type. Keys are strings of graph indices and
            values are tensor representations of persistence diagrams.
            The keys are assumed to be in range(len(tensor_dict_per_type)).

    Returns:
        torch.Tensor: [description]
    """
    try:
        assert all([int(k) in range(len(tensor_dict_per_type))
                    for k in tensor_dict_per_type.keys()])
    except AssertionError:
        print("Tensor dictionary should contain all keys in",
              "range(len(tensor_dict_per_type))")
        raise
    max_number_of_points = max([v.shape[0]
                                for v in tensor_dict_per_type.values()])

    diagram_tensor = torch.zeros((
                            len(tensor_dict_per_type),
                            max_number_of_points,
                            2
                        ))
    for graph_idx, diagram in tensor_dict_per_type.items():
        # number of points in persistence diagram
        npts = tensor_dict_per_type[graph_idx].shape[0]
        diagram_tensor[int(graph_idx)][:npts] = tensor_dict_per_type[graph_idx]

    return diagram_tensor


# check if tensorised diagrams have the correct shape
try:
    assert all((
                diagram_to_tensor(
                    tensor_dict["type1"]).shape == torch.Size([3, 7, 2]),
                diagram_to_tensor(
                    tensor_dict["type2"]).shape == torch.Size([3, 10, 2])
            ))
except AssertionError:
    print("Converted diagrams do not have correct shape.")
    raise

n_types = len(tensor_dict)  # number of diagram types

diagrams_list = []
for type_ in tensor_dict:
    diagrams_list.append(diagram_to_tensor(tensor_dict[type_]))

max_number_of_points_per_type = max([diagram.shape[1] for
                                     diagram in diagrams_list])

data = []

for type_idx, diagram in enumerate(diagrams_list):
    # diagram tensor with one-hot encoding in the zeroth coordinate
    # and a dimension in the second coordinate that fits the number
    # of points in the diagrams for all types
    diagram_tensor = torch.zeros((
                        diagram.shape[0],
                        max_number_of_points_per_type,
                        2
                    ))
    # shape: [graph_idx]
    diagram_cat = torch.tensor([type_idx] * diagram.shape[0], dtype=torch.int32)
    # shape: [graph_idx, point_idx, coordinate]
    diagram_tensor[:, :diagram.shape[1], :] = diagram
    
    # if type_idx == 0:
    #     # shape 
    #     data = 
    print(diagram_tensor.shape)


# %%
# Train Persformer on MUTAG dataset

class GraphClassifier(nn.Module):
    __init__(self, )