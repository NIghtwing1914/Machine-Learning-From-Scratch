import torch
import torch.nn as nn
from Transformer import ResidualConnection,LayerNormalization,MultiHeadAttentionBlock, FeedForwardBlock

class EncoderBlock(nn.Module):

    def __init__(self,dropout:float=0.4):
        super().__init__()
        self.norm = LayerNormalization()
        self.residualconnections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x,mask):
        x = self.residualconnections[0](x, lambda x:MultiHeadAttentionBlock(x,x,x,mask))
        x= self.residualconnections[1](x, lambda x: FeedForwardBlock())

        return x
    
class Encoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.norm=LayerNormalization()
        self.layers=layers

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)

    
