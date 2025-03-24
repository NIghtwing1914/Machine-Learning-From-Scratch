import torch
import torch.nn as nn
from Transformer import ResidualConnection,MultiHeadAttentionBlock, FeedForwardBlock, LayerNormalization

class DecoderBlock(nn.Module):

    def __init__(self,dropout:float):
        super().__init__()
        self.residualconnections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,x,src_mask,tgt_mask,encoder_output):
        x= self.residualconnections[0](x, lambda x: MultiHeadAttentionBlock(x,x,x,tgt_mask))
        x = self.residualconnections[1](x, lambda x: MultiHeadAttentionBlock(x,encoder_output,encoder_output,src_mask))
        x= self.residualconnections[2](x, lambda x: FeedForwardBlock())

        return x

class Decoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,enc_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,src_mask,tgt_mask,enc_output)
        return self.norm(x)