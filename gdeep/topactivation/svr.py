import torch 
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
import numpy as np 
import os 
from tqdm import tqdm 
import scipy.sparse.linalg

class SVR():
    def __init__(self,weights,method='svr'):
        """ weights : a sequence of matrices representing successive linear maps 
        """ 

        #Building architecture from maps 
        arch = []
        arch.append(weights[0].shape[1])
        for w in weights:
            arch.append(w.shape[0])
        self.arch = arch 
        self.method = method 

        S = []
        V = []
        U = []
        K = []
        convolutional = []
        for layer in tqdm(weights):
            if len(layer.shape)==2:
                u,s,v= self.svd(layer)
                
                if convolutional[-1]:
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
        self.U=U 
        self.V=V 
        
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
                self.adjacency.append(torch.sqrt((a**2).sum(dim=dimsToReduce) ))
            else: 
                self.adjacency.append(torch.abs(a))

            """
            if not convolutional[i]:
                self.adjacency.append(torch.abs(V[i+1].T@U[i]))
            else: 
                u,v = U[i],V[i+1]
                if convolutional[i+1]:
                    ker_size = weights[i+1].shape[-1]
                    v = v.T.reshape(v.shape[1],int(v.shape[0]/ker_size**2),ker_size**2)
                    filters = torch.einsum('cm,ncx->nmx',u,v)
                else: 
                    v = v.reshape(u.shape[0],v.shape[0]//u.shape[0],v.shape[1])
                    filters = torch.einsum('cm,cxn->nmx',u,v) 
                
                self.adjacency.append(filters.norm(dim=2))
            """

    def svd(self,A):
        if min(A.shape)>128:
            u,s,vh = scipy.sparse.linalg.svds(A.cpu().numpy(),k=128)
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


    def _build_fig(self,sigmaThreshold,y_scale=lambda x :x,nodeColor='blue',plotValue=False):

        """ 
        sigmaThreshold : confidence threshold under which edges are not shown 
        y_scale : lambda functions to rescale the plot in the y-dimension (default is identity)
        nodeColor : what color is used for vertices
        """ 
        S,arch,adjacency = self.S,self.arch,self.adjacency
        
        df = pd.DataFrame({})
        fig = px.scatter(df)


        #Edges  
        for k in range(len(adjacency)):
            E=adjacency[k]
            Emin = (sigmaThreshold) * 1/np.sqrt(arch[k+1])  #sigmaThreshold*1/np.sqrt(arch[k+1]) # 3 sigma away from the random uniform baseline 
            for i in range(E.shape[0]):
                for j in range(E.shape[1]):
                    if E[i,j]>Emin:
                        edge = pd.DataFrame({"x" : [k,k+1],"y": [y_scale(S[k][j]).item(),y_scale(S[k+1][i]).item()]})
                        #figEdge = go.scatter.Line(x=[k,k+1],y=[y_scale(S[k][j]),y_scale(S[k+1][i])],fillcolor='grey')

                        #fig.add_trace(figEdge)
                        coeff = 1-(E[i,j]-Emin)/(E.max()-Emin) # A mieux faire 
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




        base=0
        for i in range(len(S)):
            layer_i = pd.DataFrame({'x' : len(S[i])*[i], 'y': y_scale(S[i])})
            df = layer_i
            base+=len(S[i])
            #fig.add_scatter(x=df["x"], y=df["y"])   
            if self.convolutional[i]:
                color = 'red'
            else: 
                color = nodeColor        
            node_trace = go.Scatter(
                x=df["x"],
                y=df["y"],
                mode='markers',
                showlegend=False,
                marker=go.scatter.Marker(
                    opacity=1,
                    color = color
                ))

            fig.add_trace(node_trace)

        return fig 

    def plot(self,sigmaThreshold=3,y_scale=lambda x :x,nodeColor='blue'):
        fig = self._build_fig(sigmaThreshold,y_scale,nodeColor)
        fig.show()
    
    def plot_save(self,path,sigmaThreshold=2,y_scale=lambda x :x,nodeColor='blue'):
        """ path : path used for saving the image (including filename) 
        """ 
        fig =  self._build_fig(sigmaThreshold,y_scale,nodeColor)
        fig.write_image(path+'.png')
        

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
                


    def entropy(self):
        """ Returns an array of layer-wise entropy 
        """ 
        def H(x):
            y = torch.abs(x)
            return -(y*torch.log(y)).sum()
        
        return torch.tensor([H(a) for a in self.adjacency])




    def L1(self):
        return torch.tensor([(a).norm(1) for a in self.adjacency])