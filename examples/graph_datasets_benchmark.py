# %%
from gdeep.data import PersistenceDiagramFromGraphDataset
from gdeep.utility import autoreload_if_notebook


autoreload_if_notebook()



# %%
dataset_name = "MUTAG"  # REDDIT-BINARY, MUTAG, COLLAB
diffusion_parameter = 0.1

pd_dataset = PersistenceDiagramFromGraphDataset(
    dataset_name,
    diffusion_parameter=diffusion_parameter,
)






# %%
