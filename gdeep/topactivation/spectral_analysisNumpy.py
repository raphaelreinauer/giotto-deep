
from scipy.sparse.linalg import aslinearoperator, LinearOperator
from scipy.sparse.linalg import eigsh

import numpy as np 
#from xitorch import LinearOperator 
#from xitorch.linalg import symeig
# This code might be possible to implement in torch using xitorch package 



class LaplacianOperator(LinearOperator):
    def __init__(self,layers,normalize='none',positivation='none'):
        
        """
        Normalize : 'root' for Lap = I -D^-1/2 A D^-1/2
                    'sum' for Lap = I - 1/2(D^-1A + A^D^-1)
        Positivation : 'abs' takes the absolute value of the weights 
                        
        """
        
        
        self.normalize=normalize
        if positivation=='abs':
            self.layers=(np.abs(layer) for layer in layers)
        else:
            self.layers = layers
        
        shape = 0 
        for layer in layers:
            if shape==0:
                # First element in the iterator layers
                shape += layer.shape[1]
                dtype = layer.dtype 
                
            shape+= layer.shape[0]
            
        self.D = np.zeros(shape) #Diagonal of the Laplacian operator  
        
        i = 0 
        prevLayer = 0
        
        for layer in layers: 
            
            if i==0:
                # First layer 
                j = i+layer.shape[1]
                self.D[i:j]=layer.sum(axis=0)
                prevLayer = layer 
                i = j
            else:
                # Any Layer 
                j = i+prevLayer.shape[0]
                self.D[i:j]=prevLayer.sum(axis=1)+layer.sum(axis=0) 
                prevLayer = layer 
                i = j 
                
                if i+layer.shape[0]==shape:
                    # Last Layer 
                    j = i+layer.shape[0]
                    self.D[i:j]=layer.sum(axis=1)
                
            
        if self.normalize=='sum':
            self.invD= 1/self.D
        if self.normalize=='root':
            self.isqrtD=(self.D)**(-0.5)
        
        super().__init__(dtype,(shape,shape))
        
        
    
    def adjacency(self,x):
        
        # A revoir ! 
        res = np.zeros(x.shape[0])
        
        i = 0 
        prevLayer = 0
        
        for layer in self.layers: 
            if i==0:
                # First layer 
                j = layer.shape[1]
                res[i:j]=layer.T@x[j:(j+layer.shape[0])]
                prevLayer = layer 
                i = j
            else:
                # Any Layer 
                j = i+prevLayer.shape[0]
                
                res[i:j]=prevLayer@x[(i-prevLayer.shape[1]):i]+layer.T@x[j:(j+layer.shape[0])]
                prevLayer = layer 
                i = j 
                
                if i+layer.shape[0]==self.shape[0]:
                    # Last Layer 
                    j = i+layer.shape[0]
                    res[i:j]=layer@x[(i-layer.shape[1]):i]
        
        return res 
        
        
    def _matvec(self,x):
        
        if self.normalize=='sum':
            return x - 0.5*(self.adjacency(x)*self.invD+self.adjacency(x*self.invD))
        if self.normalize=='root':
            x - self.isqrtD*self.adjacency(self.isqrtD*x)
        else:
            return self.D*x - self.adjacency(x)
             
            
    def _adjoint(self):
        return self 
    def _transpose(self):
        return self
        
        
        
        
        
        
# A tester : l'image de 1 fait 0 
A1 = np.zeros((3,2))
A2 = np.zeros((4,3))
A3 = np.zeros((2,4))
A1[0,0]=1.0
A2[0,0]=2.0
A2[1,2]=5.0
A3[0,0]=3.0
weights = [A1,A2,A3]

op = LaplacianOperator(weights)
#u,v = eigsh(op)