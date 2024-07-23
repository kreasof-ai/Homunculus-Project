import torch
import torch.nn as nn
import torch.optim as optim

from saveModel import save_model_weights, load_model_weights
from main import TransformerModel

from tokenizers import Tokenizer

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
LOSS_THRESHOLD = 2.0  # Loss value threshold for increasing iterations

# Create the model
model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, CONTEXT_SIZE)


# Load the tokenizer
tokenizer = Tokenizer.from_file("bpe_tokenizer_autoregressive.json")

# Define the loss function, confidence loss, and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
confidence_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load model weights before training
load_model_weights(model, "model_weights.safetensors")
print("Model weights loaded.")

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

# Save model weights at the end of training
save_model_weights(model, "model_weights.safetensors")
print("Model weights saved.")

print("Training completed.")
