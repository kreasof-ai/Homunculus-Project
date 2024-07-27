import torch
import torch.nn as nn
import torch.nn.functional as F

from RoPE import RotaryPositionalEmbedding2D, apply_rotary_pos_emb_2d
from activation import GeGLU
from GQA import GroupedQueryAttention
from RMSNorm import RMSNorm  # Import the RMSNorm layer

"""
This is the code for the vision encoder part. Consist of similar block like the main Transformer, but we use 2D RoPE by default. The training objective is fill-in-the-middle objective and integrated seamlessly with the main text generation training pipeline.
"""

class ViTBlock(nn.Module):
    def __init__(self, embed_size, num_heads, num_groups):
        super(ViTBlock, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.attention = GroupedQueryAttention(embed_size, num_heads, num_groups)
        self.norm1 = RMSNorm(embed_size)  # Use RMSNorm instead of LayerNorm
        self.norm2 = RMSNorm(embed_size)  # Use RMSNorm instead of LayerNorm
        self.fc = nn.Sequential(
            GeGLU(embed_size),
        )
        self.rotary_emb = RotaryPositionalEmbedding2D(self.head_dim)
        
    def forward(self, x, cache=None):
        b, n, _ = x.shape
        q = k = v = x
        
        # Split into heads and apply RoPE
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        
        pos_emb = self.rotary_emb(q)
        q, k = apply_rotary_pos_emb_2d(q, k, pos_emb)
        
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        
        # Reshape back to original shape
        q = q.transpose(1, 2).contiguous().view(b, n, self.embed_size)
        k = k.transpose(1, 2).contiguous().view(b, n, self.embed_size)
        v = v.transpose(1, 2).contiguous().view(b, n, self.embed_size)
        
        attn_output, _ = self.attention(q, k, v)
        x = self.norm1(x + attn_output)
        fc_output = self.fc(x)
        x = self.norm2(x + fc_output)
        
        return x, (k, v)

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, embed_size, num_heads, num_layers, num_groups):
        super(VisionTransformer, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, embed_size, kernel_size=patch_size, stride=patch_size)
        self.layers = nn.ModuleList([
            ViTBlock(embed_size, num_heads, num_groups) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_size)  # Use RMSNorm instead of LayerNorm

    def forward(self, x, use_cache=False, middle_training=False, mask_ratio=0.2, seed=None):
        b, c, h, w = x.shape
        x = self.patch_embedding(x)  # (B, embed_size, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_size)

        # If enable fill-in-the-middle training
        if middle_training:
            # Deterministic masking if seed is pre-defined
            if seed is not None:
                torch.manual_seed(seed)
            mask = torch.rand(b, self.num_patches) > mask_ratio
            mask = mask.unsqueeze(-1).expand(x.size()).to(x.device)
            masked_x = x * mask
        else:
            masked_x = x

        # Initialize cache for storing key-value pairs
        cache = [(None, None) for _ in range(len(self.layers))]

        for i, layer in enumerate(self.layers):
            if use_cache:
                masked_x, cache[i] = layer(masked_x, cache=cache[i])
            else:
                masked_x, _ = layer(masked_x)

        # If enable fill-in-the-middle training then return the MSE loss for the masked image patch
        if middle_training:
            loss = F.mse_loss(masked_x[mask == 0], x[mask == 0])
        else:
            loss = 0

        x = self.norm(masked_x)

        return x, loss