import torch
import torch.nn as nn
import torch.nn.functional as F

from RoPE import RotaryPositionalEmbedding2D, apply_rotary_pos_emb_2d
from main import TransformerBlock

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, embed_size, num_heads, num_layers):
        super(VisionTransformer, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, embed_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = RotaryPositionalEmbedding2D(embed_size // num_heads)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, use_cache=False, middle_training=False, mask_ratio=0.2):
        b, c, h, w = x.shape
        x = self.patch_embedding(x)  # (B, embed_size, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_size)

        # Apply 2D rotary positional embedding
        pos_emb = self.pos_embedding(h // self.patch_size, w // self.patch_size)
        pos_emb = pos_emb.flatten(1, 2).unsqueeze(0).expand(b, -1, -1)
        q = k = v = x.view(b, self.num_patches, -1, self.embed_size // self.num_heads).transpose(1, 2)
        q, k = apply_rotary_pos_emb_2d(q, k, pos_emb)
        x = q.view(b, self.num_patches, -1)

        if middle_training:
            mask = torch.randn(b, self.num_patches).bernoulli(p=1 - mask_ratio).unsqueeze(-1).expand(x.size())
            if mask.device != x.device:
                mask = mask.to(x.device)
            masked_x = x * mask
        else:
            masked_x = x

        for layer in self.layers:
            if use_cache:
                masked_x = layer(masked_x, use_cache=use_cache)[0]
            else:
                masked_x = layer(masked_x)

        if middle_training:
            loss = F.mse_loss(masked_x[mask == 0], x[mask == 0])
        else:
            loss = 0

        x = self.norm(masked_x)

        return x, loss