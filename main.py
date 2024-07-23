import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
import math

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

def train_bpe_tokenizer(files, vocab_size=32000):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"])
    tokenizer.train(files, trainer)
    return tokenizer

# Train and save the tokenizer
tokenizer = train_bpe_tokenizer(["train.txt"])
tokenizer.save("bpe_tokenizer_autoregressive.json")

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_dim=1):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]

def apply_rotary_pos_emb(q, k, pos_emb):
    sin, cos = pos_emb.chunk(2, dim=-1)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

def rotate_half(x):
    x = x.reshape(x.shape[:-1] + (-1, 2))
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class GeGLU(nn.Module):
    def __init__(self, embed_size):
        super(GeGLU, self).__init__()
        self.fc1 = nn.Linear(embed_size, embed_size)
        self.fc2 = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        return F.gelu(self.fc1(x)) * self.fc2(x)

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
        
    def forward(self, x):
        b, n, _ = x.shape
        q = k = v = x
        
        # Split into heads and apply RoPE
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        
        pos_emb = self.rotary_emb(q)
        q, k = apply_rotary_pos_emb(q, k, pos_emb)
        
        # Reshape back to original shape
        q = q.transpose(1, 2).contiguous().view(b, n, self.embed_size)
        k = k.transpose(1, 2).contiguous().view(b, n, self.embed_size)
        v = v.transpose(1, 2).contiguous().view(b, n, self.embed_size)
        
        attn_output, _ = self.attention(q, k, v)
        x = self.norm1(x + attn_output)
        fc_output = self.fc(x)
        x = self.norm2(x + fc_output)
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, context_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, vocab_size)
        self.confidence_fc = nn.Linear(embed_size, 1)  # Confidence prediction layer
        self.context_size = context_size

    def forward(self, x, num_iterations=1):
        x = self.embedding(x)
        for _ in range(num_iterations):
            for layer in self.layers:
                x = layer(x)
        output = self.fc(x)
        confidence = torch.sigmoid(self.confidence_fc(x.mean(dim=1)))  # Sigmoid for confidence score
        return output, confidence

# Define the constants
VOCAB_SIZE = 32000
EMBED_SIZE = 768
NUM_HEADS = 12
NUM_LAYERS = 6
CONTEXT_SIZE = 512
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
BASE_ITERATIONS = 1
MAX_ITERATIONS = 10
CONFIDENCE_THRESHOLD = 0.8

# Create the model
model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, CONTEXT_SIZE)

# Define the loss function, confidence loss, and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
confidence_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load the tokenizer
tokenizer = Tokenizer.from_file("bpe_tokenizer_autoregressive.json")

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()

    # Example input (batch size 1, context size 512)
    text = "Your input text here."
    example_input = torch.tensor(tokenizer.encode(text).ids).unsqueeze(0)[:, :CONTEXT_SIZE]
    target = example_input.clone().detach()

    # Shift target for autoregressive training
    target = target[:, 1:].contiguous().view(-1)

    num_iterations = BASE_ITERATIONS
    output, confidence = model(example_input[:, :-1], num_iterations=num_iterations)
    loss = criterion(output.view(-1, VOCAB_SIZE), target)
    confidence_target = 1 - (loss.item() / LOSS_THRESHOLD)
    confidence_target = torch.tensor([[confidence_target]], dtype=torch.float)
    confidence_loss = confidence_criterion(confidence, confidence_target)

    while confidence.mean().item() < CONFIDENCE_THRESHOLD and num_iterations < MAX_ITERATIONS:
        num_iterations += 1
        output, confidence = model(example_input[:, :-1], num_iterations=num_iterations)
        loss = criterion(output.view(-1, VOCAB_SIZE), target)
        confidence_target = 1 - (loss.item() / LOSS_THRESHOLD)
        confidence_target = torch.tensor([[confidence_target]], dtype=torch.float)
        confidence_loss = confidence_criterion(confidence, confidence_target)

    total_loss = loss + confidence_loss
    total_loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}, Confidence: {confidence.mean().item()}, Iterations: {num_iterations}')

print("Training completed.")
