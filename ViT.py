import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from RoPE import RotaryPositionalEmbedding, apply_rotary_pos_emb
from main import TransformerBlock

class ViTBlock(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers):
        super(ViTBlock, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads) for _ in range(num_layers)
        ])
        self.pos_emb = RotaryPositionalEmbedding(embed_size)
        self.caches = None
        
    def forward(self, x, use_cache=False):
        # Apply RoPE
        pos_emb = self.pos_emb(x)
        x = apply_rotary_pos_emb(x, x, pos_emb)[0]
        
        if use_cache and self.caches is None:
            self.caches = [[] for _ in range(len(self.layers))]
        
        for i, layer in enumerate(self.layers):
            if use_cache and self.caches[i]:
                x = layer(x, use_cache=True)
                self.caches[i].append(x)
            else:
                x = layer(x, use_cache=False)
        
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, embed_size, num_heads, num_layers):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_size, kernel_size=patch_size, stride=patch_size)
        self.vit_block = ViTBlock(embed_size, num_heads, num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_size))
        self.caches = None

    def forward(self, x, use_cache=False):
        b, c, h, w = x.shape
        x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        
        if use_cache and self.caches is None:
            self.caches = []
        
        if use_cache:
            if not self.caches:
                x = self.vit_block(x, use_cache=True)
                self.caches.append(x)
            else:
                x = self.caches[-1]
        else:
            x = self.vit_block(x, use_cache=False)
            self.caches = None
        
        return x
