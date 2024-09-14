import torch
import torch.nn as nn

from RoPE import RotaryPositionalEmbedding, apply_rotary_pos_emb, apply_rotary_pos_emb_2d, RotaryPositionalEmbedding2D, RotaryPositionalEmbedding3D, apply_rotary_pos_emb_3d
from activation import GeGLU
from ViT import VisionTransformer
from GQA import GroupedQueryAttention
from RMSNorm import RMSNorm
from MLP import MLP
from flashAttention import FlashAttention
from speechEncoder import SpeechEncoder
from DiT import DiT, DiTConfig

"""
This is the main code containing the main Transformer backbone. Containing few mechanism:
- Independent confidence layer for determine how many internal loop. Implemented as a few layers of MLP.
- Blend the image embedding sequence into the text embedding sequence.
- Selective Rotary Positional Encoding. Given image embedding sequence, the RoPE is applied 2 dimensionally.
- Custom KV-caching based on the number of internal iterations. Making sure every internal iterations have independent KV-cache.
- Flash Attention option.
"""

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, num_groups, use_flash_attention=False, modality='text'):
        super(TransformerBlock, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.modality = modality  # 'text', 'image', 'video', 'speech'

        if use_flash_attention:
            self.attention = FlashAttention(embed_size, num_heads, num_groups)
        else:
            self.attention = GroupedQueryAttention(embed_size, num_heads, num_groups, attention_type='xformers')  # Ensure attention_type is set

        self.norm1 = RMSNorm(embed_size)  # Use RMSNorm instead of LayerNorm
        self.norm2 = RMSNorm(embed_size)  # Use RMSNorm instead of LayerNorm
        self.fc = nn.Sequential(
            GeGLU(embed_size),
        )

        if self.modality == 'video':
            self.rotary_emb_3d = RotaryPositionalEmbedding3D(embed_size // num_heads)
        elif self.modality == 'image':
            self.rotary_emb_2d = RotaryPositionalEmbedding2D(embed_size // num_heads)
        else:
            self.rotary_emb = RotaryPositionalEmbedding(embed_size // num_heads)

    def forward(self, x, cache=None, img_pos=[], end_img_pos=[], video_dims=None):
        b, n, _ = x.shape
        q = k = v = x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # (b, num_heads, n, head_dim)

        if self.modality == 'video' and video_dims is not None:
            t, h, w = video_dims  # time, height, width
            pos_emb = self.rotary_emb_3d(q, t, h, w)
            q, k = apply_rotary_pos_emb_3d(q, k, pos_emb)
        elif self.modality == 'image':
            pos_emb = self.rotary_emb_2d(q, len(img_pos), None)  # Provide height and width if needed
            q, k = apply_rotary_pos_emb_2d(q, k, pos_emb)
        else:
            pos_emb = self.rotary_emb(q)
            q, k = apply_rotary_pos_emb(q, k, pos_emb)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        # Reshape back to original shape
        q = q.transpose(1, 2).contiguous().view(b, n, self.embed_size)
        k = k.transpose(1, 2).contiguous().view(b, -1, self.embed_size)  # -1 to account for cached tokens
        v = v.transpose(1, 2).contiguous().view(b, -1, self.embed_size)  # -1 to account for cached tokens

        attn_output, attn_weights = self.attention(q, k, v)
        x = self.norm1(x + attn_output)
        fc_output = self.fc(x)
        x = self.norm2(x + fc_output)

        return x, (k, v), attn_weights


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, context_size, img_size, patch_size, vit_layers, num_groups, use_flash_attention=False, dit_image_size=224):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, num_groups, use_flash_attention, modality='text') for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, vocab_size)
        self.confidence_fc = MLP(embed_size, embed_size // 2, 1, 3)  # Confidence prediction layer
        self.context_size = context_size
        self.softmax = nn.Softmax(dim=-1)
        self.vit = VisionTransformer(img_size, patch_size, embed_size, num_heads, vit_layers, num_groups, use_flash_attention)
        self.dit = DiT(DiTConfig(hidden_size=embed_size, num_layers=12, num_heads=num_heads, image_size=dit_image_size, patch_size=16))  # Add DiT
        self.speech_encoder = SpeechEncoder(pretrained_model='facebook/wav2vec2-base-960h', embed_size=embed_size)  # Add speech encoder
        self.img_token_id = self.embedding.num_embeddings - 2
        self.end_img_token_id = self.embedding.num_embeddings - 1
        self.speech_token_id = self.embedding.num_embeddings - 3
        self.end_speech_token_id = self.embedding.num_embeddings - 4

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
    
    def insert_speech_embeddings(self, text_tensor, speech_embeddings):
        # Define how to insert speech embeddings into text sequence
        # For example, replace a [SPEECH] token with speech embedding
        speech_pos = (text_tensor == self.speech_token_id).nonzero(as_tuple=True)
        end_speech_pos = (text_tensor == self.end_speech_token_id).nonzero(as_tuple=True)

        if len(speech_pos[0]) != len(end_speech_pos[0]) or len(speech_pos[0]) != len(speech_embeddings):
            raise ValueError("Mismatch in number of speech tokens and speech embeddings")
        
        new_tensor = text_tensor.clone()
        offset = 0
        for start, end, speech_emb in zip(speech_pos[0], end_speech_pos[0], speech_embeddings):
            new_tensor = torch.cat((new_tensor[:start+1+offset], speech_emb, new_tensor[end+offset:]), dim=1)
            offset += speech_emb.size(1) - (end - start - 1)
        
        return new_tensor, speech_pos[0], end_speech_pos[0]
    
    def detokenize_images(self, image_tokens):
        """
        Converts image tokens back to images using DiT.
        
        Args:
            image_tokens: Tensor of shape (batch_size, num_patches, embed_size)
        Returns:
            Reconstructed images: Tensor of shape (batch_size, 3, image_size, image_size)
        """
        images = self.dit(image_tokens)
        return images
    
    def get_influential_tokens(self, attention_weights, top_k=1):
        """
        Extract the most influential tokens based on attention_weights.
        
        Args:
            attention_weights: List of tensors, each of shape (batch_size, num_heads, seq_len, seq_len)
            top_k: Number of top tokens to extract per token.

        Returns:
            List of tensors with indices of top_k influential tokens for each layer.
        """
        influential_tokens = []
        for layer_attn in attention_weights:
            # Sum over all heads
            attn_sum = layer_attn.sum(dim=1)  # (batch_size, seq_len, seq_len)
            # For each token, get the indices of the top_k tokens it attends to
            top_k_indices = torch.topk(attn_sum, top_k, dim=-1).indices  # (batch_size, seq_len, top_k)
            influential_tokens.append(top_k_indices)
        return influential_tokens

    def forward(self, x, imgs=None, speech=None, num_iterations=1, use_cache=False, middle_training=False, past_outputs=None, past_img_pos=None, past_end_img_pos=None, past_speech_pos=None, past_end_speech_pos=None, speech_dims=None):
        # Integrate speech inputs
        # speech: list of tensors (batch_size, num_audio_samples)

        if past_outputs is not None and use_cache:
            x = past_outputs  # Use the output from previous iteration
            img_pos, end_img_pos = past_img_pos, past_end_img_pos
            speech_pos, end_speech_pos = past_speech_pos, past_end_speech_pos
        else:
            # middle_training: If True, use fill-in-the-middle objective for image training
            # If False, use standard next-token prediction for text

            img_seqs = []
            speech_seqs = []
            vit_loss = 0
            speech_loss = 0
            if imgs is not None:
                for img in imgs:
                    img_embedding, loss = self.vit(img, use_cache=use_cache, middle_training=middle_training)
                    img_seqs.append(img_embedding)
                    vit_loss += loss

            if speech is not None:
                for s in speech:
                    speech_embedding = self.speech_encoder(s)  # (batch_size, seq_len, embed_size)
                    speech_seqs.append(speech_embedding)

            x = self.embedding(x)
            
            img_pos, end_img_pos = [], []
            speech_pos, end_speech_pos = [], []
            if img_seqs:
                x, img_pos, end_img_pos = self.insert_image_embeddings(x, img_seqs)
            if speech_seqs:
                x, speech_pos, end_speech_pos = self.insert_speech_embeddings(x, speech_seqs)
            
            past_outputs = []  # Initialize if not provided

        caches = [[] for _ in range(len(self.layers))]
        attention_weights_per_layer = []

        for iteration in range(num_iterations):
            for i, layer in enumerate(self.layers):
                if use_cache and caches[i]:
                    x, caches[i], attn_weights = layer(
                        x, 
                        cache=caches[i][-1], 
                        img_pos=img_pos, 
                        end_img_pos=end_img_pos,
                        video_dims=speech_dims if layer.modality == 'video' else None
                    )
                else:
                    x, cache, attn_weights = layer(
                        x, 
                        cache=None, 
                        img_pos=img_pos, 
                        end_img_pos=end_img_pos,
                        video_dims=speech_dims if layer.modality == 'video' else None
                    )
                if attn_weights is not None:
                    attention_weights_per_layer.append(attn_weights)
        
            # Store the output for last iteration
            if iteration >= num_iterations - 1:
                past_outputs.append(x)

        output = self.fc(x)
        output = self.softmax(output)  # Apply softmax to the output logits
        confidence = torch.sigmoid(self.confidence_fc(x.mean(dim=1)))  # Sigmoid for confidence score
        if middle_training:
            return output, confidence, past_outputs, img_pos, end_img_pos, speech_pos, end_speech_pos, vit_loss, speech_loss, attention_weights_per_layer
        else:
            return output, confidence, past_outputs, img_pos, end_img_pos, speech_pos, end_speech_pos, attention_weights_per_layer