import torch
import torch.nn as nn
import torch.nn.functional as F

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
        q = q.view(b, self.num_heads, n // self.num_groups, self.num_groups, self.head_dim).mean(dim=3)

        # Compute attention scores
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)

        # Compute attention output
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, n, self.embed_size)

        # Output projection
        attn_output = self.out(attn_output)

        return attn_output, attn_weights