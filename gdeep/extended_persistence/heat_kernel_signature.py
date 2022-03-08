import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import eigh

class HeatKernelSignature():
    def __init__(self,
                 adj_mat,
                 time):
        self.adj_mat = adj_mat
        self.time = time

    @staticmethod
    def _hks(eigenvals, eigenvectors, time):
        hks = np.square(eigenvectors) * np.exp(-time * eigenvals).reshape(1, -1)
        assert (np.square(eigenvectors).dot(np.diag(np.exp(-time * eigenvals))) == hks).all()
        return hks.sum(axis=1)
    
    @staticmethod
    def _eigen_vals_vectors(adj_mat):
            L = csgraph.laplacian(adj_mat, normed=True)
            return eigh(L)
        
    def __call__(self):
        return HeatKernelSignature._hks(*HeatKernelSignature._eigen_vals_vectors(self.adj_mat), self.time)
