import torch
import torch.nn as nn

from RoPE import RotaryPositionalEmbedding, apply_rotary_pos_emb
from activation import GeGLU
from ViT import VisionTransformer

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
        
    def forward(self, x, cache=None):
        b, n, _ = x.shape
        q = k = v = x
        
        # Split into heads and apply RoPE
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        
        pos_emb = self.rotary_emb(q)
        q, k = apply_rotary_pos_emb(q, k, pos_emb)
        
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

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, context_size, img_size, patch_size, vit_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, vocab_size)
        self.confidence_fc = nn.Linear(embed_size, 1)  # Confidence prediction layer
        self.context_size = context_size
        self.vit = VisionTransformer(img_size, patch_size, embed_size, num_heads, vit_layers)
        self.img_token_id = self.embedding.num_embeddings - 2
        self.end_img_token_id = self.embedding.num_embeddings - 1

    def insert_image_embeddings(self, text_tensor, img_embeddings):
        img_pos = (text_tensor == self.img_token_id).nonzero(as_tuple=True)
        end_img_pos = (text_tensor == self.end_img_token_id).nonzero(as_tuple=True)
        
        assert len(img_pos[0]) == len(end_img_pos[0]) == len(img_embeddings)
        
        for start, end, img_emb in zip(img_pos[0], end_img_pos[0], img_embeddings):
            text_tensor = torch.cat((text_tensor[:start+1], img_emb, text_tensor[end:]), dim=1)
        
        return text_tensor

    def forward(self, x, imgs=None, num_iterations=1, use_cache=False):
        img_seqs = []
        if imgs is not None:
            for img in imgs:
                img_embedding = self.vit(img, use_cache=use_cache)
                img_seqs.append(img_embedding)

        x = self.embedding(x)
        
        if img_seqs:
            x = self.insert_image_embeddings(x, img_seqs)

        caches = [[] for _ in range(len(self.layers))]
        for _ in range(num_iterations):
            for i, layer in enumerate(self.layers):
                if use_cache and caches[i]:
                    x, cache = layer(x, cache=caches[i][-1])
                else:
                    x, cache = layer(x, cache=None)
                if use_cache:
                    caches[i].append(cache)
        output = self.fc(x)
        confidence = torch.sigmoid(self.confidence_fc(x.mean(dim=1)))  # Sigmoid for confidence score
        return output, confidence