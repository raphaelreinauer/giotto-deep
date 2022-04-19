
#from scipy.sparse.linalg import aslinearoperator, LinearOperator
#from scipy.sparse.linalg import eigsh

import torch 
import numpy as np 
from xitorch import LinearOperator 
from xitorch.linalg import symeig


class LaplacianOperator(LinearOperator):
    def __init__(self,layers,normalize='none',positivation='none'):
        
        """
        Normalize : 'root' for Lap = I -D^-1/2 A D^-1/2
                    'sum' for Lap = I - 1/2(D^-1A + A^D^-1)
        Positivation : 'abs' takes the absolute value of the weights 
                        'relu' takes the positive part 
        The device is defined by that of the first layer 
                        
        """
        layer_device = layers[0].device
         
        self.normalize=normalize
        if positivation=='abs':
            self.layers=[torch.abs(layer) for layer in layers]
        elif positivation=='relu':
            self.layers=[torch.nn.functional.relu(layer) for layer in layers]
        else:
            self.layers = layers
        
        shape = 0 
        for layer in self.layers:
            if shape==0:
                # First element in the iterator layers
                shape += layer.shape[1]
                dtype = layer.dtype 
                
            shape+= layer.shape[0]
            
        self.D = torch.zeros(shape).to(layer_device) #Diagonal of the Laplacian operator  
        
        i = 0 
        prevLayer = 0
        
        for layer in self.layers: 
            
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
        
        super().__init__([shape,shape],dtype=dtype,is_hermitian=True,device=layer_device)
        
    



    def adjacency(self,x):
        """x should be of dimension 1 or 2 
        """  
        
        if len(x.shape)==2:
            x=x.T #We switch from batch size convention to matrix multiplication convention 

        
        res = torch.zeros(x.shape).to(self.device)


        i = 0 
        prevLayer = 0
        
        for layer in self.layers: 
            if i==0:
                # First layer 
                j = layer.shape[1]
                res[i:j]=layer.T@(x[j:(j+layer.shape[0])])
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
        
        if len(x.shape)==2:
            return res.T
            

        return res


    
        
        
    # Note : for better optimization performances, _mm could be implemented in the case of x being of size (shape,shape) using block matrix product 
        
    def _mv(self,x):
        """ x shape can be either 1 or 2, in case of 2 the first dimension is the batch number.
            If you wish to have matrix multiplication  instead, use mm method """
        
        if x.device!=self.device:
            print("WARNING : input is on device : ",x.device, "while the layers are on device : ", self.device,". Changing input device")
            x = x.to(self.device)


        if self.normalize=='sum':
            return x - 0.5*(self.adjacency(x)*self.invD+self.adjacency(x*self.invD))
        if self.normalize=='root':
            x - self.isqrtD*self.adjacency(self.isqrtD*x)
        else:
            return self.D*x - self.adjacency(x)

    

    def _getparamnames(self, prefix=""):
        return [prefix+"layers"]
        
    def diagonalize(self,**kwargs):
        """Internal Wrapper for symeig
            Return u,v where u contains eigenvalues 

        neig: int or None
            The number of eigenpairs to be retrieved. If ``None``, all eigenpairs are
            retrieved
        mode: str
            ``"lowest"`` or ``"uppermost"``/``"uppest"``. If ``"lowest"``,
            it will take the lowest ``neig`` eigenpairs.
            If ``"uppest"``, it will take the uppermost ``neig``.

            """
         #By default we exclude 0 for better numerical stability at first 
        u,v = symeig(self,method='davidson',neig=self.shape[0]-1,mode ="uppest",**kwargs)
        # Adding 0 back
        device = u.device
        zero = torch.zeros(1).to(device)
        ones = torch.ones((v.shape[0],1)).to(device)
        return torch.cat([zero,u]), torch.cat([ones/ones.norm(),v],dim=1)
        
        
        
        
        
# A tester : l'image de 1 fait 0 
A1 =torch.zeros((3,2))
A2 = torch.zeros((4,3))
A3 = torch.zeros((2,4))
A1[0,0]=1.0
A2[0,0]=2.0
A2[1,2]=5.0
A3[0,0]=3.0
weights = [A1,A2,A3]

op = LaplacianOperator(weights)
#u,v = eigsh(op)
#u,v=symeig(op,method='davidson')



class SequentialLaplacianOperator(LinearOperator):
    def __init__(self,layers,normalize='none',positivation='none'):
        
        """
        The layers need to be an iterator containing callable function corresponding to the layers. 

        Normalize : 'root' for Lap = I -D^-1/2 A D^-1/2
                    'sum' for Lap = I - 1/2(D^-1A + A^D^-1)
        Positivation : 'abs' takes the absolute value of the weights 
                        'relu' takes the positive part 
        The device is defined by that of the first layer 
                        
        """
        layer_device = layers[0].device
         
        self.normalize=normalize
        if positivation=='abs':
            self.layers=[torch.abs(layer) for layer in layers]
        elif positivation=='relu':
            self.layers=[torch.nn.functional.relu(layer) for layer in layers]
        else:
            self.layers = layers
        
        shape = 0 
        for layer in self.layers:
            if shape==0:
                # First element in the iterator layers
                shape += layer.shape[1]
                dtype = layer.dtype 
                
            shape+= layer.shape[0]
            
        self.D = torch.zeros(shape).to(layer_device) #Diagonal of the Laplacian operator  
        
        i = 0 
        prevLayer = 0
        
        for layer in self.layers: 
            
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
        
        super().__init__([shape,shape],dtype=dtype,is_hermitian=True,device=layer_device)
        
    



    def adjacency(self,x):
        """x should be of dimension 1 or 2 
        """  
        
        if len(x.shape)==2:
            x=x.T #We switch from batch size convention to matrix multiplication convention 

        
        res = torch.zeros(x.shape).to(self.device)


        i = 0 
        prevLayer = 0
        
        for layer in self.layers: 
            if i==0:
                # First layer 
                j = layer.shape[1]
                res[i:j]=layer.T@(x[j:(j+layer.shape[0])])
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
        
        if len(x.shape)==2:
            return res.T
            

        return res


    
        
        
    # Note : for better optimization performances, _mm could be implemented in the case of x being of size (shape,shape) using block matrix product 
        
    def _mv(self,x):
        """ x shape can be either 1 or 2, in case of 2 the first dimension is the batch number.
            If you wish to have matrix multiplication  instead, use mm method """
        
        if x.device!=self.device:
            print("WARNING : input is on device : ",x.device, "while the layers are on device : ", self.device,". Changing input device")
            x = x.to(self.device)


        if self.normalize=='sum':
            return x - 0.5*(self.adjacency(x)*self.invD+self.adjacency(x*self.invD))
        if self.normalize=='root':
            x - self.isqrtD*self.adjacency(self.isqrtD*x)
        else:
            return self.D*x - self.adjacency(x)

    

    def _getparamnames(self, prefix=""):
        return [prefix+"layers"]
        
    def diagonalize(self,**kwargs):
        """Internal Wrapper for symeig
            Return u,v where u contains eigenvalues 

        neig: int or None
            The number of eigenpairs to be retrieved. If ``None``, all eigenpairs are
            retrieved
        mode: str
            ``"lowest"`` or ``"uppermost"``/``"uppest"``. If ``"lowest"``,
            it will take the lowest ``neig`` eigenpairs.
            If ``"uppest"``, it will take the uppermost ``neig``.

            """
         #By default we exclude 0 for better numerical stability at first 
        u,v = symeig(self,method='davidson',neig=self.shape[0]-1,mode ="uppest",**kwargs)
        # Adding 0 back
        device = u.device
        zero = torch.zeros(1).to(device)
        ones = torch.ones((v.shape[0],1)).to(device)
        return torch.cat([zero,u]), torch.cat([ones/ones.norm(),v],dim=1)