# %%
from IPython import get_ipython  # type: ignore
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import networkx as nx
import numpy as np

from time import time
from os.path import join
from copy import deepcopy

from gdeep.extended_persistence import HeatKernelSignature

#%%

def get_local_minima(adj_mat, filtration_vals):
    min_val_nbs = np.min(filtration_vals * (1.0/adj_mat), axis=1)
    return min_val_nbs >= filtration_vals

def get_directed_graph(adj_mat, filtration_vals):
    return nx.from_numpy_matrix(((filtration_vals * (1.0/adj_mat) < filtration_vals.reshape(-1, 1)).T) * 1,
                                create_using=nx.DiGraph)

def get_traversed_nodes(adj_mat, filtration_vals):
    dgraph = get_directed_graph(adj_mat, filtration_vals)
    graph_size = adj_mat.shape[0]
    traversed_nodes = np.zeros((graph_size, graph_size))
    for source in np.argwhere(get_local_minima(adj_mat, filtration_vals)).T[0].tolist():
        traversed = list(nx.dfs_preorder_nodes(dgraph, source=source))[1:]
        traversed_nodes[source][traversed] = 1
    return traversed_nodes

def compute_death_times(adj_mat, filtration_vals):
    traversed_nodes = get_traversed_nodes(adj_mat, filtration_vals + 1) * filtration_vals.reshape(-1, 1)

    min_vals = ((traversed_nodes == 0) * max_filtration + traversed_nodes).min(axis=0)

    death_matrix = traversed_nodes * ((traversed_nodes != 0) & (traversed_nodes > min_vals))

    return (death_matrix != 0).argmax(axis=1)
# %%
path_to_data = join('tests', 'data')


# %%
with open(join(path_to_data, 'xlarge_filtered_graph.npy'), 'rb') as f:
    adj_mat = np.load(f)
    filtration_vals = np.load(f)
del f
np.count_nonzero(get_local_minima(adj_mat, filtration_vals))

# %%
with open(join(path_to_data, 'reddit12k_sample_graph.npy'), 'rb') as f:
    adj_mat = np.load(f)
del f
filtration_vals = HeatKernelSignature(adj_mat, 0.1)()
assert filtration_vals.shape == np.shape(adj_mat[0],), f"filtration_vals has shape {filtration_vals.shape}"
#nx.draw(graph, labels={v: f for (v, f) in enumerate(filtration_vals.tolist())},with_labels=True)

np.count_nonzero(get_local_minima(adj_mat, filtration_vals))

dgraph = nx.from_numpy_matrix(adj_mat)
# %%


# %%
graph_size = 5
adj_mat = np.zeros((graph_size, graph_size))
adj_mat[0, 2] = adj_mat[1, 2] = adj_mat[2, 4] = adj_mat[3, 4] = 1
adj_mat += adj_mat.T
filtration_vals = np.array(list(range(1, graph_size + 1)))
max_filtration = filtration_vals.max()
graph = nx.from_numpy_matrix(adj_mat)
nx.draw(graph, with_labels=True)
assert np.count_nonzero(get_local_minima(adj_mat, filtration_vals)) == graph_size - 2


# %%
from gdeep.extended_persistence.gudhi_implementation import graph_extended_persistence_gudhi
%timeit graph_extended_persistence_gudhi(adj_mat, filtration_vals)
%timeit x = compute_death_times(adj_mat, filtration_vals)
# %%
%timeit x = compute_death_times(adj_mat, filtration_vals)

# %%
