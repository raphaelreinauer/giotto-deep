# %%
from gdeep.data.persistence_diagrams.one_hot_persistence_diagram import OneHotEncodedPersistenceDiagram


name_graph_dataset: str = 'MUTAG'
diffusion_parameter: float = 10.1

pd_creator = PersistenceDiagramFromGraphDataset(name_graph_dataset, 10.1)
pd_creator.create()

pd_mutag_ds = PersistenceDiagramFromFilesDataset(name_graph_dataset, 10.1)

pd: OneHotEncodedPersistenceDiagram = pd_mutag_ds[0]

pd.plot()

pd_mutag_dl_tr, pd_mutag_dl_te = pd_mutag_ds.get_train_test_split(0.2)