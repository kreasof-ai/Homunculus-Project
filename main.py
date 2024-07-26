import torch
import torch.nn as nn

from RoPE import RotaryPositionalEmbedding, apply_rotary_pos_emb
from activation import GeGLU
from ViT import VisionTransformer
from GQA import GroupedQueryAttention
from RMSNorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, num_groups):
        super(TransformerBlock, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.attention = GroupedQueryAttention(embed_size, num_heads, num_groups)
        self.norm1 = RMSNorm(embed_size)  # Use RMSNorm instead of LayerNorm
        self.norm2 = RMSNorm(embed_size)  # Use RMSNorm instead of LayerNorm
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
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, context_size, img_size, patch_size, vit_layers, num_groups):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, num_groups) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, vocab_size)
        self.confidence_fc = nn.Linear(embed_size, 1)  # Confidence prediction layer
        self.context_size = context_size
        self.vit = VisionTransformer(img_size, patch_size, embed_size, num_heads, vit_layers, num_groups)
        self.img_token_id = self.embedding.num_embeddings - 2
        self.end_img_token_id = self.embedding.num_embeddings - 1

    def insert_image_embeddings(self, text_tensor, img_embeddings):
        img_pos = (text_tensor == self.img_token_id).nonzero(as_tuple=True)
        end_img_pos = (text_tensor == self.end_img_token_id).nonzero(as_tuple=True)
        
        assert len(img_pos[0]) == len(end_img_pos[0]) == len(img_embeddings)
        
        for start, end, img_emb in zip(img_pos[0], end_img_pos[0], img_embeddings):
            text_tensor = torch.cat((text_tensor[:start+1], img_emb, text_tensor[end:]), dim=1)
        
        return text_tensor

    def forward(self, x, imgs=None, num_iterations=1, use_cache=False, middle_training=False):
        img_seqs = []
        vit_loss = 0
        if imgs is not None:
            for img in imgs:
                img_embedding, loss = self.vit(img, use_cache=use_cache, middle_training=middle_training)
                img_seqs.append(img_embedding)
                vit_loss += loss

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
        if middle_training:
            return output, confidence, vit_loss
        else:
            return output, confidence

    def generate(self, input_text, tokenizer, max_length=512, imgs=None, num_iterations=1, use_cache=False):
        tokens = tokenizer.encode(input_text).ids
        input_tensor = torch.tensor(tokens).unsqueeze(0)
        img_seqs = []
        if imgs is not None:
            for img in imgs:
                img_embedding, loss = self.vit(img, use_cache=use_cache)
                img_seqs.append(img_embedding)
        
        generated_tokens = input_tensor.clone()
        
        if img_seqs:
            generated_tokens = self.insert_image_embeddings(generated_tokens, img_seqs)
        
        for _ in range(max_length - len(tokens)):
            output, confidence = self.forward(generated_tokens, num_iterations=num_iterations, use_cache=use_cache)
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            if next_token.item() in {self.img_token_id, self.end_img_token_id}:
                # Skip generating tokens for [IMG] and [/IMG]
                continue
            
            generated_tokens = torch.cat((generated_tokens, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == tokenizer.token_to_id("[SEP]"):
                break
        
        return tokenizer.decode(generated_tokens.squeeze().tolist())