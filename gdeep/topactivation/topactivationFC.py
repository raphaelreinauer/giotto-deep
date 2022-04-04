from gdeep.models import ModelExtractor
from gudhi.representations.vector_methods import Entropy
from gudhi import SimplexTree as ST
import networkx as nx
import torch
import numpy as np
from torch.optim import SGD


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using GPU!")
else:
    DEVICE = torch.device("cpu")

class TopactivationFC:

    def __init__(self, pipe, arch):
        self.pipe = pipe
        self.arch = arch
        self.diagrams_training = []



    def get_activation_graph(self, x, index_batch = 1):
        me = ModelExtractor(self.pipe.model, self.pipe.loss_fn)
        activations = me.get_activations(x)
        n_layer = len(self.arch)
        current_node = 0
        activation_graph = ST()
        edge_list = []
        for i in range(n_layer - 1):
            f = lambda x: (x[0] + current_node, x[1] + current_node)
            G = nx.complete_bipartite_graph(self.arch[i], self.arch[i+1])
            l = list(G.edges())
            l = map(f,l)
            edge_list.extend(l)
            current_node += self.arch[i]
        for edge in edge_list:
            activation_graph.insert(list(edge), 0.0)
        activations_flatten = torch.empty(0).to(DEVICE)
        for layer in range(n_layer):
            activations_flatten = torch.cat((activations_flatten, activations[layer][index_batch]))
        for neuron in range(activations_flatten.size()[0]):
            activation_graph.insert([neuron], float(activations_flatten[neuron]))
        return activation_graph




    def get_extended_persistence(self, x):
        diagrams = []
        for i in range(len(x)):
            L = self.get_activation_graph(x, index_batch=i)
            L.extend_filtration()
            pers = L.extended_persistence()
            diagrams.append(pers[3])
        return diagrams



    def diagrams_accross_training(self, x, n_epochs, optimizer = SGD, every_n_epochs = 2, lr = 0.1 ):
        diagrams = []
        for i in range(n_epochs):
            diagrams.append(self.get_extended_persistence(x))
            #self.pipe.train(optimizer, every_n_epochs, lr = lr)
            self.pipe.train(optimizer, 2, True, {"lr": lr}, n_accumulated_grads=5)
        self.diagrams_training = diagrams
        return diagrams

    # FGSM attack code
    def fgsm_attack(self, data, target, epsilon):
        data.requires_grad = True
        output = self.pipe.model(data)
        loss = self.pipe.loss_fn(output, target)
        self.pipe.model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = data + epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image

    def reverse_diagrams(self, diagrams):
        diagrams_reversed = []
        for diagram in diagrams:
            diagram_reversed = []
            for dim, bar in diagram:
                diagram_reversed.append([bar[1], bar[0]])
            diagrams_reversed.append(np.array(diagram_reversed, dtype = np.float32))
        return diagrams_reversed

    def get_entropy_training(self):
        assert self.diagrams_training != [], "diagrams_training are empty!"
        entr = Entropy()
        diagrams_reversed = [self.reverse_diagrams(diagrams) for diagrams in self.diagrams_training]
        entropies = [entr.transform(diagrams) for diagrams in diagrams_reversed]
        n_epochs = len(self.diagrams_training)
        n_batch = len(self.diagrams_training[0])
        E = np.empty([n_epochs, n_batch])
        for epoch in range(n_epochs):
            for batch in range(n_batch):
                E[epoch][batch] = entropies[epoch][batch]
        return E

