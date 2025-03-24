import numpy as np

class Neuron():

    def __init__(self,nin):
        self.weights = [np.random.randn((1,1)) for _ in range(nin)]
        self.bias = np.random.randn((1,1))

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def __call__(self,x):
        p = np.sum([wi*xi for wi,xi in zip(self.weights,x)])+self.bias
        return self.sigmoid(p)
    
class Layer():

    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out
    
class MLP():

    def __init__(self,nin,nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
