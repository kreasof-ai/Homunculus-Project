import torch
import torch.nn as nn
import torch.nn.functional as F

"""
We use Grouped Query Attention
"""

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_size, num_heads, num_kv_heads):
        super(GroupedQueryAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_size // num_heads
        self.kv_head_dim = embed_size // num_kv_heads
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, num_kv_heads * self.kv_head_dim)
        self.value = nn.Linear(embed_size, num_kv_heads * self.kv_head_dim)
        self.out = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v, mask=None):
        b, n, _ = q.shape
        
        # Linear projections
        q = self.query(q).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(k).view(b, n, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)
        v = self.value(v).view(b, n, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)
        
        # Repeat k and v to match number of query heads
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Attention
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, n, self.embed_size)
        attn_output = self.out(attn_output)
        
        return attn_output, attn_weights
