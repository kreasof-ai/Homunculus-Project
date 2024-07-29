import torch
import torch.nn as nn
import torch.nn.functional as F

"""
We use Grouped Query Attention
"""

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_size, num_heads, num_groups):
        super(GroupedQueryAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_size // num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.out = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v, mask=None):
        b, n, _ = q.shape

        # Linear projections
        q = self.query(q).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(k).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(v).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # Group queries
        group_size = n // self.num_groups
        q_groups = q.split(group_size, dim=2)
        k_groups = k.split(group_size, dim=2)
        v_groups = v.split(group_size, dim=2)

        attn_output = torch.zeros_like(q)

        for qg, kg, vg in zip(q_groups, k_groups, v_groups):
            scores = torch.einsum('bhqd,bhkd->bhqk', qg, kg) / (self.head_dim ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_output_group = torch.einsum('bhqk,bhkd->bhqd', attn_weights, vg)
            attn_output += attn_output_group

        attn_output = attn_output.transpose(1, 2).contiguous().view(b, n, self.embed_size)
        attn_output = self.out(attn_output)

        return attn_output, attn_weights
