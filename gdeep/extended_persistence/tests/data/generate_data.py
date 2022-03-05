
# %%
import networkx as nx
import numpy as np

# %%
sizes=[('small', 5), ('medium', 10), ('large', 200), ('xlarge', 1000)]

for size in sizes:
    num_vertices = size[1]
    ggg = nx.random_geometric_graph(num_vertices, 0.125)
    adj_mat = nx.adjacency_matrix(ggg).todense()
    filtration_vals = np.random.rand(num_vertices)
    with open(size[0]+'_filtered_graph.npy', 'wb') as f:
        np.save(f, adj_mat)
        np.save(f, filtration_vals)
# %%
