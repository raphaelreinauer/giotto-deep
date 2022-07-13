# %%
import numpy as np
from gtda.homology import VietorisRipsPersistence

from gdeep.data.persistence_diagrams import get_one_hot_encoded_persistence_diagram_from_gtda

VR = VietorisRipsPersistence(homology_dimensions=[0, 1])

# set ranom seed
np.random.seed(0)

points = np.random.rand(100, 2)

diagrams = VR.fit_transform([points])

diagram = diagrams[0]

get_one_hot_encoded_persistence_diagram_from_gtda(diagram)._data.shape == (123, 4)

# %%
