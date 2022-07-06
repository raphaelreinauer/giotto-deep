import torch 
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
import numpy as np 
import os 
from tqdm import tqdm 
import scipy.sparse.linalg
from scipy.stats import chi2,norm
import matplotlib.pyplot as plt 
from matplotlib.colors import to_rgb as color2rgb


class SVR():
    def __init__(self,weights,method='svr',max_modes=128,verbose=True):
        """ weights : a sequence of matrices representing successive linear maps 
            max_modes : the maximum number of modes that are going to be computed during SVD of weights 
            verbose : if True, the code will display progression bars during computations 
        """ 

        #Removing biases 
        weights = [w for w in weights if len(w.shape)>1]

        #Building architecture from maps 
        arch = []
        arch.append(weights[0].shape[1])
        for w in weights:
            arch.append(w.shape[0])
        self.arch = arch 
        self.method = method 
        self.max_modes = max_modes 
        self.verbose = verbose 

        S = []
        V = []
        U = []
        K = []
        convolutional = []
        if self.verbose:
            iterator = tqdm(weights)
        else:
            iterator = weights 
        for layer in iterator:
            if len(layer.shape)==2:
                u,s,v= self.svd(layer)
                
                if len(convolutional)>0 and convolutional[-1]:
                    # we reshape the FC input to match the convolution output 
                    iz,m = v.shape
                    i = U[-1].shape[0]
                    z = iz//i 
                    v = v.reshape(i,z,1,m) 

                convolutional.append(False)
            else: #convolutional layer
                convolutional.append(True)
                o,i,k,k = layer.shape 
                # z = k,k
                if self.method=='svr':
                    #o,i,z -> o,iz 
                    u,s,v= self.svd(layer.flatten(start_dim=1))
                    m = len(s)
                    v = v.reshape(i,k,k,m) # iz,m -> i,z,m
                elif self.method=='cosvr':
                    # o,i,z-> oz,i 
                    w = layer.transpose(0,1) #i,o,z
                    w = w.flatten(start_dim=1) #i,oz
                    w = w.transpose(0,1) #oz,i 
                    u,s,v = self.svd(w)
                    m = len(s)
                    #oz,m -> o,z,m
                    u = u.reshape(o,k,k,m)

                else:
                    raise Exception("Method unknow : "+str(method)) 

            U.append(u.to('cpu'))
            S.append(s.to('cpu'))
            V.append(v.to('cpu'))

        self.S = S
        self.U = U 
        self.V = V 
        
        self.adjacency = []
        self.convolutional = convolutional 
        for i in range(len(V)-1):
            # V[i+1]    : i,n or i,z,n  , transpose : n,z,i 
            # U[i]      : o,m or o,z,m 
            # contraction over o-i    n,m or n,z,m  = n,k,k,m
            # adjacency (V.T@U) : n,m  
            a = torch.tensordot(V[i+1].transpose(0,-1),U[i],1)
            if self.convolutional[i]:
                dimsToReduce = list(range(1,len(a.shape)-1))
                self.adjacency.append((a**2).sum(dim=dimsToReduce))
            else: 
                self.adjacency.append(a**2)


    def svd(self,A):
        if min(A.shape)>self.max_modes:
            u,s,vh = scipy.sparse.linalg.svds(A.cpu().numpy(),k=self.max_modes)
            v = torch.tensor(vh.T)
            u = torch.tensor(u)
            s = torch.tensor(s)
            order = torch.argsort(s,descending=True)
            s =s[order]
            u,v = u[:,order],v[:,order]
        else:
            u,s,vh = torch.linalg.svd(A,full_matrices=False)
            v = vh.T
        
        return u,s,v 



    def _build_fig(self,sigmaThreshold,max_edges,y_scale=lambda x :x,node_color = {'fc' : 'blue', 'conv':'red'},plotValue=False):

        """ 
        sigmaThreshold : confidence threshold under which edges are not shown 
        y_scale : custom lambda function to rescale the plot in the y-dimension (default is identity)
        node_color : what color is used for vertices depending on the type of layer : (fc) is fully connected, (conv) is convolutional
        """ 
        S,arch,adjacency = self.S,self.arch,self.adjacency
        
        df = pd.DataFrame({})
        fig = px.scatter(df)

        layout = go.Layout(
        title="Network SVR - Link minimum significance : "+str(sigmaThreshold)+ " sigma",
        xaxis=dict(
            title="Layer index"
        ),
        yaxis=dict(
            title="Singular value"
        ) ) 
        fig=go.Figure(layout=layout) 

        #Edges  
        if self.verbose:
            iterator = tqdm(range(len(adjacency)))
        else: 
            iterator = (range(len(adjacency)))
        for k in iterator:
            E=adjacency[k]

            mean = 1/arch[k+1]
            
            if self.convolutional[k] and self.method =='svr': 
                kernel_size = (self.V[k+1].shape[1]*self.V[k+1].shape[2])**0.5
            elif self.convolutional[k] and self.method=='cosvr': 
                kernel_size = (self.U[k].shape[1]*self.U[k].shape[2])**0.5
            else: #Fully connected layer
                kernel_size = 1 

            std = np.sqrt(2)*mean/kernel_size 
            z = int(np.round(kernel_size**2)) 
            p = 1-norm.cdf(sigmaThreshold)

            Emin = (chi2.ppf(1-p,z)/z)*mean #Probabilistic threshold 
            F = E.reshape(E.shape[0]*E.shape[1]).argsort(descending=True)[:max_edges] # Practical threshold against overload 
            F = torch.flip(F,dims=[0])
            Fi = F.div(E.shape[1], rounding_mode="floor") 
            Fj = F.remainder(E.shape[1])

            for i,j in zip(Fi,Fj):
                if E[i,j]>=Emin:
                    edge = pd.DataFrame({"x" : [k,k+1],"y": [y_scale(S[k][j]).item(),y_scale(S[k+1][i]).item()]})
                    #figEdge = go.scatter.Line(x=[k,k+1],y=[y_scale(S[k][j]),y_scale(S[k+1][i])],fillcolor='grey')

                    #fig.add_trace(figEdge)
                    coeff = 1-(E[i,j]-Emin)/(E.max()-Emin)
                    color = coeff.item()*np.array([120,120,120])+np.array([105,105,105])
                    r,g,b=int(color[0]),int(color[1]),int(color[2])
                    color = "rgb"+str((r,g,b))
                    fig.add_scatter(x=edge["x"],y=edge["y"],marker={"color":color,"opacity":1},hovertext=str(adjacency[k][i,j])
                                ,hoveron="fills",hoverinfo="text",text=str(adjacency[k][i,j]),showlegend=False ) 

                    if plotValue:
                        middle_node_trace = go.Scatter(
                            x=[np.array(edge["x"]).mean()],
                            y=[np.array(edge["y"]).mean()],
                            text=[str(adjacency[k][i,j])],
                            mode='markers',
                            hoverinfo='text',
                            showlegend=False,
                            marker=go.scatter.Marker(
                                opacity=0,
                                color='lightgrey'
                            )
                        )

                        fig.add_trace(middle_node_trace)



        grey = np.array(color2rgb('grey'))
        base=0
        for i in range(len(S)):
            layer_i = pd.DataFrame({'x' : len(S[i])*[i], 'y': y_scale(torch.flip(S[i],dims=[0]))})
            base+=len(S[i])
            #fig.add_scatter(x=df["x"], y=df["y"])   
            if self.convolutional[i]:
                color = node_color['conv']
            else: 
                color = node_color['fc']

            node_trace = go.Scatter(
                x=layer_i["x"],
                y=layer_i["y"],
                mode='markers',
                showlegend=False,
                marker=go.scatter.Marker(
                    color = torch.flip(self.node_intensity(i),dims=[0]),
                    cmin = 0,
                    cmax=1,
                    colorscale = [[0, 'grey'], [1, color]],
                    opacity = 1
                ))

            fig.add_trace(node_trace)

        return fig 

    def node_intensity(self,layer):
        """ This function returns an array with the same shape as self.S[layer] which contains coefficients between 0 and 1
        Each coefficient corresponds to the importance of each mode measured by the max deviation of its cumulative scalar product 
        distribution with respect to modes of the next layer"""
        if layer==len(self.adjacency):
            return torch.ones(self.S[layer].shape)
        a = self.adjacency[layer]
        n = self.arch[layer+1]
        dim = 0
        b = torch.ones(a.shape)
        res = (n*a-b).cumsum(dim=dim).max(dim=dim)[0]
        res = torch.nn.functional.relu(res)
        return res/res.max()


    def plot(self,sigmaThreshold=3,y_scale=lambda x :x,nodeColor='blue',max_edges=100):
        fig = self._build_fig(sigmaThreshold,max_edges,y_scale)
        fig.show()
    
    def plot_save(self,path,sigmaThreshold=3,y_scale=lambda x :x,nodeColor='blue',max_edges=100):
        """ path : path used for saving the image (including filename) 
        """ 
        fig =  self._build_fig(sigmaThreshold,max_edges,y_scale)
        fig.write_image(path+'.png')
        
    def argmax2d(self,res):
        ymaxs = res.argmax(dim=1)
        xrange = torch.tensor(range(res.shape[0]))
        xmax = res[xrange,ymaxs[xrange]].argmax()
        ymax = ymaxs[xmax]
        return (xmax.item(),ymax.item())


    def internal_dims(self,layer):
        a = self.adjacency[layer]
        wa =(self.adjacency[layer]).cumsum(dim=0).cumsum(dim=1)
        ij = torch.ones(wa.shape).cumsum(dim=0).cumsum(dim=1)
        n = self.arch[layer+1]
        res = (n*wa-ij)
        #plt.imshow(res)
        #plt.show()
        return self.argmax2d(res)


    def pathmetric(self,svdThreshold=0,svdScale=True):
        """ 
            svdThreshold : Discard all paths that cross a mode with a singular value below svdThreshold (default : 0)
            svdScale : Paths receive 
        """ 
        adjacency = [torch.abs(a) for a in self.adjacency]
        device = self.adjacency[0].device
        
        if svdScale:
            S = [s.clone() for s in self.S]
        else: 
            S = [torch.ones(len(s)) for s in self.S]
            
        S = [S[i]*(s>svdThreshold) for i,s in enumerate(self.S) ]
        
        x = torch.ones(len(S[0])).to(device)
        for i in range(len(S)-1):
            x=x*S[i]
            x = adjacency[i]@x
        x = x*S[-1]

        return x.sum().item()
                
    def filters(self,i):
        """ This function returns the effective filters that connects modes from consecutive convolutional layer  
            Input : 
                - i : layer index 
            Returns :
                an array (n,m) where m is the number of mode in layer i and n the number of modes in layer i+1
        """ 
        return torch.tensordot(self.V[i+1].transpose(0,-1),self.U[i],1)

    def plot_filters(self,f):
        """ Input : f with shape nb of channels, K, K :
                        in that case filters will be arranged into a square-like configuration 
                    f with shape h,w,K,K 
                        in that case a (w,h) array of filters will be plotted """
                    
        
        K = f.shape[-1]
        if len(f.shape)==3:
            c = f.shape[0]
            w = int(np.sqrt(c))
            h = c//w
        elif len(f.shape)==4:
            h,w=f.shape[0],f.shape[1]
        else:
            raise Exception("Invalid input shape") 

        res = torch.zeros((K+1)*w,(K+1)*h ) 
        for i in range(w):
            for j in range(h):
                basei = (K+1)*i
                basej = (K+1)*j
                if len(f.shape)==3:
                    res[basei:basei+K,basej:basej+K] = f[i+w*j,:,:]
                else:
                    res[basei:basei+K,basej:basej+K] = f[j,i,:,:]
        plt.imshow(res)
        plt.show() 


    def entropy(self):
        """ Returns an array of layer-wise entropy 
        """ 
        def H(x):
            y = torch.abs(x)
            return -(y*torch.log(y)).sum()
        
        return torch.tensor([H(a) for a in self.adjacency])




    def L1(self):
        return torch.tensor([(a).norm(1) for a in self.adjacency])