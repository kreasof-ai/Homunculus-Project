import torch.nn as nn
from flash_attn import flash_attn_func

"""
This is option for using both Grouped Query Attention and Flash Attention
"""

class FlashAttention(nn.Module):
    def __init__(self, embed_size, num_heads, num_groups):
        super(FlashAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_size // num_heads
        
        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size // (num_heads // num_groups))
        self.v_proj = nn.Linear(embed_size, embed_size // (num_heads // num_groups))
        self.out_proj = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v):
        b, n, _ = q.shape
        
        q = self.q_proj(q).view(b, n, self.num_heads, self.head_dim)
        k = self.k_proj(k).view(b, n, self.num_groups, self.head_dim)
        v = self.v_proj(v).view(b, n, self.num_groups, self.head_dim)
        
        # Repeat k and v to match the number of heads
        k = k.repeat_interleave(self.num_heads // self.num_groups, dim=2)
        v = v.repeat_interleave(self.num_heads // self.num_groups, dim=2)
        
        # Prepare inputs for flash_attn_func
        q = q.transpose(1, 2)  # [b, nh, n, hd]
        k = k.transpose(1, 2)  # [b, nh, n, hd]
        v = v.transpose(1, 2)  # [b, nh, n, hd]

        attn_output = flash_attn_func(q, k, v, softmax_scale=None)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, n, self.embed_size)
        out = self.out_proj(attn_output)
        
        return out, None  # Return None for compatibility with existing implementation
