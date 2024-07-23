import torch
import torch.nn as nn

from RoPE import RotaryPositionalEmbedding, apply_rotary_pos_emb
from activation import GeGLU

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(TransformerBlock, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.attention = nn.MultiheadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.fc = nn.Sequential(
            GeGLU(embed_size),
        )
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        self.cache = None
        
    def forward(self, x, use_cache=False):
        b, n, _ = x.shape
        q = k = v = x
        
        # Split into heads and apply RoPE
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        
        pos_emb = self.rotary_emb(q)
        q, k = apply_rotary_pos_emb(q, k, pos_emb)
        
        if use_cache and self.cache is not None:
            k = torch.cat([self.cache[0], k], dim=2)
            v = torch.cat([self.cache[1], v], dim=2)
        
        if use_cache:
            self.cache = (k, v)
        else:
            self.cache = None
        
        # Reshape back to original shape
        q = q.transpose(1, 2).contiguous().view(b, n, self.embed_size)
        k = k.transpose(1, 2).contiguous().view(b, n, self.embed_size)
        v = v.transpose(1, 2).contiguous().view(b, n, self.embed_size)
        
        attn_output, _ = self.attention(q, k, v)
        x = self.norm1(x + attn_output)
        fc_output = self.fc(x)
        x = self.norm2(x + fc_output)
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, context_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, vocab_size)
        self.confidence_fc = nn.Linear(embed_size, 1)  # Confidence prediction layer
        self.context_size = context_size

    def forward(self, x, num_iterations=1, use_cache=False):
        x = self.embedding(x)
        for _ in range(num_iterations):
            for layer in self.layers:
                x = layer(x, use_cache=use_cache)
        output = self.fc(x)
        confidence = torch.sigmoid(self.confidence_fc(x.mean(dim=1)))  # Sigmoid for confidence score
        return output, confidence
