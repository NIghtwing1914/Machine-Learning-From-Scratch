import torch
import torch.nn as nn
import math

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, num_heads, d_model):
        super().__init__()

        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

        self.w_o = nn.Linear(d_model,d_model)

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

    def forward(self,key,query,value,mask):
        # Initial shapes
        # key, query, value: (batch_size, seq_len, d_model)

        # After linear transformations
        # self.key, self.query, self.value: (batch_size, seq_len, d_model)

        # Reshape and transpose for multi-head attention
        # self.query: (batch_size, num_heads, seq_len, d_k)
        # self.key: (batch_size, num_heads, seq_len, d_k)
        # self.value: (batch_size, num_heads, seq_len, d_k)

        # Attention scores calculation
        # self.attention_scores: (batch_size, num_heads, seq_len, seq_len)

        # Apply mask (if any)
        # self.attention_scores: (batch_size, num_heads, seq_len, seq_len)

        # Attention output
        # x: (batch_size, num_heads, seq_len, d_k)

        # Reshape and transpose back to original dimensions
        # x: (batch_size, seq_len, d_model)
        self.key = self.w_k(key)
        self.query = self.w_q(query) # B , seq_len, d_model
        self.value = self.w_v(value)

        self.query = self.query.view(self.query.shape[0],self.query.shape[1],self.num_heads,self.d_k).transpose(1,2) # B, h, seq_len, d_k
        self.key = self.key.view(self.key.shape[0],self.key.shape[1],self.num_heads,self.d_k).transpose(1,2)
        self.value = self.value.view(self.value.shape[0],self.value.shape[1],self.num_heads,self.d_k).transpose(1,2)

    
        self.attention_scores = self.query @ self.key.transpose(-2,-1) / math.sqrt(self.d_k)

        if mask is not None:
            self.attention_scores = self.attention_scores.masked_fill(mask==0,-1e9)
        
        x = self.attention_scores @ self.value

        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.num_heads*self.d_k)
        return self.attention_scores,self.w_o(x)
    

# Example usage
d_model = 512
num_heads = 8
seq_len = 10
batch_size = 2

# Create a random tensor to represent a batch of sequences
query = torch.rand(batch_size, seq_len, d_model)
key = torch.rand(batch_size, seq_len, d_model)
value = torch.rand(batch_size, seq_len, d_model)
mask = None  # or create a mask tensor with appropriate shape

# Initialize the MultiHeadAttention module
mha = MultiHeadAttentionBlock(num_heads,d_model)

# Forward pass
attention_scores, output = mha(query, key, value, mask)

print("Attention Scores:", attention_scores)
print("Output:", output)