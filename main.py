import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string

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
        self.attention = nn.MultiheadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.fc = nn.Sequential(
            GeGLU(embed_size),
        )
        
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        fc_output = self.fc(x)
        x = self.norm2(x + fc_output)
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, context_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, context_size, embed_size))
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, vocab_size)
        self.confidence_fc = nn.Linear(embed_size, 1)  # Confidence prediction layer
        self.context_size = context_size

    def forward(self, x, num_iterations=1):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        
        for _ in range(num_iterations):
            for layer in self.layers:
                x = layer(x)
        
        output = self.fc(x)
        confidence = torch.sigmoid(self.confidence_fc(x.mean(dim=1)))  # Sigmoid for confidence score
        return output, confidence

# Define the constants
VOCAB_SIZE = len(string.ascii_uppercase)  # 26 for A-Z
EMBED_SIZE = 10
NUM_HEADS = 1  # Number of attention heads
NUM_LAYERS = 2  # Number of transformer layers
CONTEXT_SIZE = 10
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
BASE_ITERATIONS = 1
MAX_ITERATIONS = 10  # Maximum number of iterations allowed
CONFIDENCE_THRESHOLD = 0.8  # Confidence threshold for stopping iterations

# Create the model
model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, CONTEXT_SIZE)

# Define the loss function, confidence loss, and optimizer
criterion = nn.CrossEntropyLoss()
confidence_criterion = nn.MSELoss()  # Use MSE loss for confidence prediction
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear the gradients of all optimized tensors

    # Example input (batch size 1, context size 10)
    example_input = torch.randint(0, VOCAB_SIZE, (1, CONTEXT_SIZE))
    target = torch.randint(0, VOCAB_SIZE, (1, CONTEXT_SIZE))

    # Initialize number of iterations
    num_iterations = BASE_ITERATIONS

    # Forward pass with initial number of iterations
    output, confidence = model(example_input, num_iterations=num_iterations)

    # Compute the initial loss
    loss = criterion(output.view(-1, VOCAB_SIZE), target.view(-1))

    # Compute the confidence target (1 - normalized loss)
    confidence_target = 1 - (loss.item() / LOSS_THRESHOLD)
    confidence_target = torch.tensor([[confidence_target]], dtype=torch.float)

    # Compute the confidence loss
    confidence_loss = confidence_criterion(confidence, confidence_target)

    # Adjust the number of iterations if the confidence is low
    while confidence.mean().item() < CONFIDENCE_THRESHOLD and num_iterations < MAX_ITERATIONS:
        num_iterations += 1
        output, confidence = model(example_input, num_iterations=num_iterations)
        loss = criterion(output.view(-1, VOCAB_SIZE), target.view(-1))
        confidence_target = 1 - (loss.item() / LOSS_THRESHOLD)
        confidence_target = torch.tensor([[confidence_target]], dtype=torch.float)
        confidence_loss = confidence_criterion(confidence, confidence_target)

    # Backward pass to compute gradients
    total_loss = loss + confidence_loss  # Combine the main loss and confidence loss
    total_loss.backward()

    # Update the model parameters using the computed gradients
    optimizer.step()

    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}, Confidence: {confidence.mean().item()}, Iterations: {num_iterations}')

print("Training completed.")
