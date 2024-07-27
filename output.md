## activation.py

```python
# activation.py

import torch.nn as nn
import torch.nn.functional as F

"""
We use GeGLU (Gated GeLU) activation function
"""

class GeGLU(nn.Module):
    def __init__(self, embed_size):
        super(GeGLU, self).__init__()
        self.fc1 = nn.Linear(embed_size, embed_size)
        self.fc2 = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        return F.gelu(self.fc1(x)) * self.fc2(x)
```

## format.py

```python
# format.py

import os

"""
This is a utility script for extracting the whole code into a single markdown file. Nothing important for the main functionality
"""

def write_python_scripts_to_markdown(directory, output_file):
    with open(output_file, 'w') as md_file:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    md_file.write(f'## {file}\n\n')
                    md_file.write('```python\n')
                    md_file.write(f'# {file}\n\n')
                    with open(file_path, 'r') as py_file:
                        md_file.write(py_file.read())
                    md_file.write('\n```\n\n')

if __name__ == "__main__":
    directory_to_scan = './'  # Replace with your directory path
    output_markdown_file = 'output.md'  # Replace with your desired output file name
    write_python_scripts_to_markdown(directory_to_scan, output_markdown_file)
    print(f'All Python scripts have been written to {output_markdown_file}')

```

## GQA.py

```python
# GQA.py

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

```

## main.py

```python
# main.py

import torch
import torch.nn as nn

from RoPE import RotaryPositionalEmbedding, apply_rotary_pos_emb, apply_rotary_pos_emb_2d, RotaryPositionalEmbedding2D
from activation import GeGLU
from ViT import VisionTransformer
from GQA import GroupedQueryAttention
from RMSNorm import RMSNorm
from MLP import MLP

"""
This is the main code containing the main Transformer backbone. Containing few mechanism:
- Independent confidence layer for determine how many internal loop. Implemented as a few layers of MLP.
- Blend the image embedding sequence into the text embedding sequence.
- Selective Rotary Positional Encoding. Given image embedding sequence, the RoPE is applied 2 dimensionally.
- Custom KV-caching based on the number of internal iterations. Making sure every internal iterations have independent KV-cache.
"""

class TransformerBlock(nn.Module):
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
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, context_size, img_size, patch_size, vit_layers, num_groups):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, num_groups) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, vocab_size)
        self.confidence_fc = MLP(embed_size, embed_size // 2, 1, 3)  # Confidence prediction layer
        self.context_size = context_size
        self.vit = VisionTransformer(img_size, patch_size, embed_size, num_heads, vit_layers, num_groups)
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

    def forward(self, x, imgs=None, num_iterations=1, use_cache=False, middle_training=False):
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

        caches = [[] for _ in range(len(self.layers))]
        for _ in range(num_iterations):
            for i, layer in enumerate(self.layers):
                if use_cache and caches[i]:
                    x, caches[i] = layer(x, cache=caches[i][-1], img_pos=img_pos, end_img_pos=end_img_pos)
                else:
                    x, cache = layer(x, cache=None, img_pos=img_pos, end_img_pos=end_img_pos)
        output = self.fc(x)
        confidence = torch.sigmoid(self.confidence_fc(x.mean(dim=1)))  # Sigmoid for confidence score
        if middle_training:
            return output, confidence, vit_loss
        else:
            return output, confidence

    def generate(self, input_text, tokenizer, max_length=512, imgs=None, num_iterations=1, use_cache=False, beam_size=5):
        tokens = tokenizer.encode(input_text).ids
        input_tensor = torch.tensor(tokens).unsqueeze(0)
        
        # Process images
        img_seqs = []
        if imgs is not None:
            for img in imgs:
                img_embedding, _ = self.vit(img, use_cache=use_cache)
                img_seqs.append(img_embedding)
        
        if img_seqs:
            input_tensor, img_pos, end_img_pos = self.insert_image_embeddings(input_tensor, img_seqs)
        
        # Initialize beam
        beams = [(input_tensor, 0)]
        
        for _ in range(max_length - len(tokens)):
            all_candidates = []
            for beam, score in beams:
                output, _ = self.forward(beam, num_iterations=num_iterations, use_cache=use_cache)
                next_token_logits = output[0, -1, :]
                top_k_logits, top_k_indices = torch.topk(next_token_logits, beam_size)
                
                for logit, index in zip(top_k_logits, top_k_indices):
                    new_beam = torch.cat((beam, index.unsqueeze(0).unsqueeze(0)), dim=1)
                    new_score = score - logit.item()  # Negative log likelihood
                    all_candidates.append((new_beam, new_score))
            
            # Select top beam_size candidates
            beams = sorted(all_candidates, key=lambda x: x[1])[:beam_size]
            
            if beams[0][0][:, -1].item() == tokenizer.token_to_id("[SEP]"):
                break
        
        return tokenizer.decode(beams[0][0].squeeze().tolist())
```

## MLP.py

```python
# MLP.py

import torch.nn as nn

"""
This is simple implementation of MLP for a certain layer that needs more than a single linear layer
"""

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
```

## RMSNorm.py

```python
# RMSNorm.py

import torch
import torch.nn as nn

"""
This is the implementation of Root Mean Square normalization layer for replacing a standard normalization layer
"""

class RMSNorm(nn.Module):
    def __init__(self, embed_size, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.embed_size = embed_size
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_size))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms * self.scale
        return x

```

## RoPE.py

```python
# RoPE.py

import torch
import torch.nn as nn

"""
This is the Rotary Positional Embedding parts. Consist of 1 dimensional for text sequence and 2 dimensional for image sequence.
"""

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=500000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_dim=1):
        # seq_len refers to the length of the sequence in the dimension where RoPE is applied
        seq_len = x.shape[seq_dim]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.outer(t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]

def apply_rotary_pos_emb(q, k, pos_emb):
    sin, cos = pos_emb.chunk(2, dim=-1)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

class RotaryPositionalEmbedding2D(nn.Module):
    def __init__(self, dim, base=500000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, h, w):
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=self.inv_freq.device),
                                        torch.arange(w, device=self.inv_freq.device))
        grid = torch.stack((grid_y, grid_x), dim=-1).float()
        sinusoid_inp = torch.einsum("...d,k->...dk", grid, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        emb = emb.permute(2, 0, 1)
        return emb

def apply_rotary_pos_emb_2d(q, k, pos_emb):
    sin, cos = pos_emb.chunk(2, dim=0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

def rotate_half(x):
    x = x.reshape(x.shape[:-1] + (-1, 2))
    x1, x2 = x.unbind(-1)
    return torch.cat((-x2, x1), dim=-1)

```

## saveModel.py

```python
# saveModel.py

from safetensors.torch import save_file, load_file

"""
This is the code for saving and load the model with safetensors format
"""

def save_model_weights(model, path):
    state_dict = model.state_dict()
    save_file(state_dict, path)

def load_model_weights(model, path):
    state_dict = load_file(path)
    model.load_state_dict(state_dict)
```

## tokenizer.py

```python
# tokenizer.py

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

"""
This is the code to generate the custom tokenizer. Using Byte Pair Encoding
"""

def train_bpe_tokenizer(files, vocab_size=32000):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[IMG]", "[/IMG]"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train(files, trainer)
    return tokenizer

# Train and save the tokenizer
tokenizer = train_bpe_tokenizer(["train.txt"])
tokenizer.save("bpe_tokenizer_autoregressive.json")
```

## train.py

```python
# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from saveModel import save_model_weights, load_model_weights
from main import TransformerModel
from tokenizers import Tokenizer, processors
from torch.utils.data import Dataset, DataLoader

"""
This is the main code for training and define the parameter. Consist of:
- PyTorch Lightning integration.
- Model parameter definition.
- Training loop definition.
- Training based on confidence score and internal looping.
- Training the image with fill-in-the-middle objective combined with the main transformer cross entropy loss.
- Mask the image sequence for next-token text generation objective.
"""

# Define the constants
VOCAB_SIZE = 128000
EMBED_SIZE = 8192
NUM_HEADS =  64
NUM_LAYERS = 80
CONTEXT_SIZE = 128000
LEARNING_RATE = 1.5e-4
NUM_EPOCHS = 10e5
BASE_ITERATIONS = 1
MAX_ITERATIONS = 10
CONFIDENCE_THRESHOLD = 0.8
LOSS_THRESHOLD = 2.0  # Loss value threshold for increasing iterations
IMG_SIZE = 1024
PATCH_SIZE = 16
VIT_LAYERS = 16
NUM_GROUPS = 8  # Number of groups for Grouped Query Attention

# Load tokenizer
tokenizer = Tokenizer.from_file("bpe_tokenizer_autoregressive.json")
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# For pre-processing real dataset
class DatasetLoader(Dataset):
    def __init__(self, text_data, image_data):
        self.text_data = text_data
        self.image_data = image_data

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.tokenizer.encode(self.text_data[idx]).ids
        image = self.image_data[idx]  # Assume this is already a tensor
        return torch.tensor(text), image

# dataset = DatasetLoader(text_data, image_data)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class TransformerLightningModule(pl.LightningModule):
    def __init__(self, model, tokenizer, learning_rate):
        super(TransformerLightningModule, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
        self.confidence_criterion = nn.MSELoss()

    def forward(self, x, imgs=None, num_iterations=1, use_cache=False, middle_training=False):
        return self.model(x, imgs=imgs, num_iterations=num_iterations, use_cache=use_cache, middle_training=middle_training)

    def training_step(self, batch, batch_idx):
        example_input, imgs = batch
        target = example_input.clone().detach()

        # Shift target for autoregressive training while ignoring image token regions
        target = target[:, 1:].contiguous().view(-1)
        mask = (target != self.tokenizer.token_to_id("[IMG]")) & (target != self.tokenizer.token_to_id("[/IMG]"))
        target = target[mask]

        num_iterations = BASE_ITERATIONS
        output, confidence, vit_loss = self(example_input[:, :-1], imgs=imgs, num_iterations=num_iterations, use_cache=True, middle_training=True)
        output = output.view(-1, VOCAB_SIZE)[mask]
        loss = self.criterion(output, target) + vit_loss
        confidence_target = max(0, min(1, 1 - (loss.item() / LOSS_THRESHOLD)))
        confidence_target = torch.tensor([[confidence_target]], dtype=torch.float, device=self.device)
        confidence_loss = self.confidence_criterion(confidence, confidence_target)

        while confidence.mean().item() < CONFIDENCE_THRESHOLD and num_iterations < MAX_ITERATIONS:
            num_iterations += 1
            output, confidence, vit_loss = self(example_input[:, :-1], imgs=imgs, num_iterations=num_iterations, use_cache=True, middle_training=True)
            output = output.view(-1, VOCAB_SIZE)[mask]
            loss = self.criterion(output, target) + vit_loss
            confidence_target = max(0, min(1, 1 - (loss.item() / LOSS_THRESHOLD)))
            confidence_target = torch.tensor([[confidence_target]], dtype=torch.float, device=self.device)
            confidence_loss = self.confidence_criterion(confidence, confidence_target)

        total_loss = loss + confidence_loss
        self.log('train_loss', total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

# Create the model
model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, CONTEXT_SIZE, IMG_SIZE, PATCH_SIZE, VIT_LAYERS, NUM_GROUPS)

# Load model weights before training
load_model_weights(model, "model_weights.safetensors")
print("Model weights loaded.")

# Create the LightningModule
lightning_model = TransformerLightningModule(model, tokenizer, LEARNING_RATE)

# Define the DataLoader
def train_dataloader():
    # Example input (batch size 1, context size 512)
    text = "Your input text here with [IMG][/IMG] and [IMG][/IMG]."
    example_input = torch.tensor(tokenizer.encode(text).ids).unsqueeze(0)[:, :CONTEXT_SIZE]

    # Example image inputs (batch size 1, 3 channels, 224x224)
    imgs = [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)]

    # return dataloader
    return [(example_input, imgs)]

# Define the Trainer
trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    gpus=1,  # Use GPU if available
    callbacks=[ModelCheckpoint(monitor='train_loss')]
)

# Train the model
trainer.fit(lightning_model, train_dataloaders=train_dataloader())

# Save model weights at the end of training
save_model_weights(model, "model_weights.safetensors")
print("Model weights saved.")

print("Training completed.")

```

## ViT.py

```python
# ViT.py

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
```

