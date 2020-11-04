import torch
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram

import utility

def persistence_diagrams_of_activations(model, X_tensor,
                layer_types=[torch.nn.Linear],
                homology_dimensions=[0, 1], layers=utility.Layers_list('All')):
    """Returns list of persistence diagrams of the activations of all
    layers of type layer_types

    Args:
        model (nn.Module): Neural Network
        X_tensor (torch.tensor): [description]
        layer_types (list, optional): [description]. Defaults to [torch.nn.Linear].
        homology_dimensions (list, optional): list of homology dimensions. Defaults to [0, 1].
        layers ([type], optional): list of layer types to consider. Defaults to Layers_list('All').

    Returns:
        list: list of persistence diagrams of activations of the different layers
    """
    
    
    activations_layers = utility.get_activations(model, X_tensor)

    choosen_activations_layers = []

    VR = VietorisRipsPersistence(homology_dimensions=homology_dimensions)

    for i, activations_layer in enumerate(activations_layers.get_outputs()):
        if layers.in_list(i):

            choosen_activations_layer = activations_layer.numpy()
            choosen_activations_layers.append(choosen_activations_layer)

    persistence_diagrams = VR.fit_transform(choosen_activations_layers)

    return persistence_diagrams