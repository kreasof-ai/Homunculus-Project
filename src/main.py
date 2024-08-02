import torch
import torch.nn as nn

from RoPE import RotaryPositionalEmbedding, apply_rotary_pos_emb, apply_rotary_pos_emb_2d, RotaryPositionalEmbedding2D
from activation import GeGLU
from ViT import VisionTransformer
from GQA import GroupedQueryAttention
from RMSNorm import RMSNorm
from MLP import MLP
from flashAttention import FlashAttention

"""
This is the main code containing the main Transformer backbone. Containing few mechanism:
- Independent confidence layer for determine how many internal loop. Implemented as a few layers of MLP.
- Blend the image embedding sequence into the text embedding sequence.
- Selective Rotary Positional Encoding. Given image embedding sequence, the RoPE is applied 2 dimensionally.
- Custom KV-caching based on the number of internal iterations. Making sure every internal iterations have independent KV-cache.
- Flash Attention option.
"""

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, num_groups, use_flash_attention=False):
        super(TransformerBlock, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        if use_flash_attention:
            self.attention = FlashAttention(embed_size, num_heads, num_groups)
        else:
            self.attention = GroupedQueryAttention(embed_size, num_heads, num_groups)
        self.norm1 = RMSNorm(embed_size)  # Use RMSNorm instead of LayerNorm
        self.norm2 = RMSNorm(embed_size)  # Use RMSNorm instead of LayerNorm
        self.fc = nn.Sequential(
            GeGLU(embed_size),
        )
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        self.rotary_emb_2d = RotaryPositionalEmbedding2D(self.head_dim)
        
    def forward(self, x, cache=None, img_pos=[], end_img_pos=[]):
        b, n, _ = x.shape
        q = k = v = x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply 1D RoPE by default
        pos_emb = self.rotary_emb(q)
        q, k = apply_rotary_pos_emb(q, k, pos_emb)
        
        # Apply 2D RoPE for image tokens
        for start, end in zip(img_pos, end_img_pos):
            pos_emb_2d = self.rotary_emb_2d(q[:, :, start:end])
            q[:, :, start:end], k[:, :, start:end] = apply_rotary_pos_emb_2d(q[:, :, start:end], k[:, :, start:end], pos_emb_2d)
        
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        
        # Reshape back to original shape
        q = q.transpose(1, 2).contiguous().view(b, n, self.embed_size)
        k = k.transpose(1, 2).contiguous().view(b, -1, self.embed_size)  # -1 to account for cached tokens
        v = v.transpose(1, 2).contiguous().view(b, -1, self.embed_size)  # -1 to account for cached tokens
        
        attn_output, _ = self.attention(q, k, v)
        x = self.norm1(x + attn_output)
        fc_output = self.fc(x)
        x = self.norm2(x + fc_output)
        
        return x, (k, v)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, context_size, img_size, patch_size, vit_layers, num_groups, use_flash_attention=False):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, num_groups, use_flash_attention) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, vocab_size)
        self.confidence_fc = MLP(embed_size, embed_size // 2, 1, 3)  # Confidence prediction layer
        self.context_size = context_size
        self.softmax = nn.Softmax(dim=-1)
        self.vit = VisionTransformer(img_size, patch_size, embed_size, num_heads, vit_layers, num_groups, use_flash_attention)
        self.img_token_id = self.embedding.num_embeddings - 2
        self.end_img_token_id = self.embedding.num_embeddings - 1

    def insert_image_embeddings(self, text_tensor, img_embeddings):
        img_pos = (text_tensor == self.img_token_id).nonzero(as_tuple=True)
        end_img_pos = (text_tensor == self.end_img_token_id).nonzero(as_tuple=True)
        
        if len(img_pos[0]) != len(end_img_pos[0]) or len(img_pos[0]) != len(img_embeddings):
            raise ValueError("Mismatch in number of image tokens and image embeddings")
        
        new_tensor = text_tensor.clone()
        offset = 0
        for start, end, img_emb in zip(img_pos[0], end_img_pos[0], img_embeddings):
            new_tensor = torch.cat((new_tensor[:start+1+offset], img_emb, new_tensor[end+offset:]), dim=1)
            offset += img_emb.size(1) - (end - start - 1)
        
        return new_tensor, img_pos[0], end_img_pos[0]

    def forward(self, x, imgs=None, num_iterations=1, use_cache=False, middle_training=False, past_outputs=None, past_img_post=None, past_end_img_post=None):
        # Use provided past_outputs if available

        if past_outputs is not None and use_cache:
            x = past_outputs  # Use the output from previous iteration
            img_pos, end_img_pos = past_img_post, past_end_img_post
        else:
            # middle_training: If True, use fill-in-the-middle objective for image training
            # If False, use standard next-token prediction for text

            img_seqs = []
            vit_loss = 0
            if imgs is not None:
                for img in imgs:
                    img_embedding, loss = self.vit(img, use_cache=use_cache, middle_training=middle_training)
                    img_seqs.append(img_embedding)
                    vit_loss += loss

            x = self.embedding(x)
            
            img_pos, end_img_pos = [], []
            if img_seqs:
                x, img_pos, end_img_pos = self.insert_image_embeddings(x, img_seqs)
            
            past_outputs = []  # Initialize if not provided

        caches = [[] for _ in range(len(self.layers))]

        for iteration in range(num_iterations):
            for i, layer in enumerate(self.layers):
                if use_cache and caches[i]:
                    x, caches[i] = layer(x, cache=caches[i][-1], img_pos=img_pos, end_img_pos=end_img_pos)
                else:
                    x, cache = layer(x, cache=None, img_pos=img_pos, end_img_pos=end_img_pos)
            
            # Store the output for last iteration
            if iteration >= num_iterations - 1:  # -1 because iterations start from 0
                past_outputs.append(x)

        output = self.fc(x)
        output = self.softmax(output)  # Apply softmax to the output logits
        confidence = torch.sigmoid(self.confidence_fc(x.mean(dim=1)))  # Sigmoid for confidence score
        if middle_training:
            return output, confidence, past_outputs, img_pos, end_img_pos, vit_loss
        else:
            return output, confidence, past_outputs, img_pos, end_img_pos