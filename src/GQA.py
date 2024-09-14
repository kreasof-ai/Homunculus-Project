# GQA.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops

"""
We use Grouped Query Attention with xformers for efficient attention computation
"""

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_size, num_heads, num_groups, attention_type='scaled_dot_product'):
        super(GroupedQueryAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_groups = num_groups

        self.head_dim = embed_size // num_heads
        assert (self.head_dim * num_heads == embed_size), "embed_size must be divisible by num_heads"

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.out = nn.Linear(embed_size, embed_size)

        self.attention_type = attention_type

        if self.attention_type == 'xformers':
            self.attn_op = xops.AttentionOp('softmax')  # Options: 'softmax', 'scaled_dot_product'

    def forward(self, q, k, v, mask=None):
        b, n, _ = q.shape

        # Linear projections
        q = self.query(q).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # (b, num_heads, n, head_dim)
        k = self.key(k).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(v).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # Group queries
        group_size = self.num_heads // self.num_groups
        q_groups = q.chunk(self.num_groups, dim=1)  # list of (b, group_size, n, head_dim)
        k_groups = k.chunk(self.num_groups, dim=1)
        v_groups = v.chunk(self.num_groups, dim=1)

        attn_output = []
        attention_weights = []  # To store attention weights for influential token extraction

        for qg, kg, vg in zip(q_groups, k_groups, v_groups):
            if self.attention_type == 'xformers':
                attn = xops.memory_efficient_attention(qg, kg, vg, attn_bias=None)
                attn_weights = None  # XFormers doesn't expose attention weights directly in this function
            else:
                scores = torch.matmul(qg, kg.transpose(-2, -1)) / (self.head_dim ** 0.5)
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, float('-inf'))
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
                attn = torch.matmul(attn_weights, vg)
            attn_output.append(attn)
            if attn_weights is not None:
                attention_weights.append(attn_weights)

        # Concatenate groups
        attn_output = torch.cat(attn_output, dim=1)  # (b, num_heads, n, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, n, self.embed_size)  # (b, n, embed_size)
        out = self.out(attn_output)

        # Aggregate attention weights if available
        if attention_weights:
            attn_weights = torch.cat(attention_weights, dim=1)  # (b, num_heads, n, n)
        else:
            attn_weights = None

        return out, attn_weights
