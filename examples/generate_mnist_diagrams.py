from gdeep.topactivation import TopactivationFC as TFC
from gdeep.pipeline import Pipeline
from torch.utils.tensorboard import SummaryWriter

from gdeep.models import FFNet
from torch import nn
from gdeep.data import TorchDataLoader
from gdeep.topology_layers.preprocessing import convert_gudhi_extended_persistence_to_persformer_input

writer = SummaryWriter()
dl = TorchDataLoader(name="MNIST")
dl_tr, dl_ts = dl.build_dataloaders(batch_size=32)

arch = [28*28, 50, 50, 10]
model = nn.Sequential(nn.Flatten(), FFNet( arch= arch ))
loss_fn = nn.CrossEntropyLoss()
pipe = Pipeline(model, (dl_tr, dl_ts), loss_fn, writer)

from torch.optim import Adam
n_epochs = 10
topactiv = TFC(pipe,arch)
topactiv.pipe.train(Adam, n_epochs)

import torch
import numpy as np
epsilon = 0.1
diagrams_MNIST= []
labels_MNIST = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

counter = 0
labels = []
for data, target in dl_tr:
    data, target = data.to(device), target.to(device)
    diagrams_batch, labels_batch = topactiv.get_extended_persistence_randomly_perturbed(
        data, target, epsilon)
    for diagram, label in zip(diagrams_batch, labels_batch):
        diagram_one_hot = convert_gudhi_extended_persistence_to_persformer_input([diagram])[0]
        # Save diagram_one_hot to npy file
        np.save(f"data/adversarial_MNIST_10_0.1/diagrams/{counter}.npy", diagram_one_hot)
        labels.append(int(label))
        counter += 1
 

# Save labels to csv file
np.savetxt("data/adversarial_MNIST_10_0.1/labels.csv", labels, delimiter=",")
