import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy
from saveModel import save_model_weights, load_model_weights
from main import TransformerModel
from tokenizers import Tokenizer, processors
from torch.utils.data import Dataset, DataLoader

# Define the constants
VOCAB_SIZE = 128000
EMBED_SIZE = 8192
NUM_HEADS = 64
NUM_LAYERS = 80
CONTEXT_SIZE = 128000
LEARNING_RATE = 1.5e-4
NUM_EPOCHS = 10
BASE_ITERATIONS = 1
MAX_ITERATIONS = 10
CONFIDENCE_THRESHOLD = 0.8
LOSS_THRESHOLD = 2.0  # Loss value threshold for increasing iterations
IMG_SIZE = 1024
PATCH_SIZE = 16
VIT_LAYERS = 16
NUM_GROUPS = 8  # Number of groups for Grouped Query Attention
USE_FLASH_ATTENTION = False  # Set this to True to use Flash Attention

"""
This is the main code for training and define the parameter. Consist of:
- PyTorch Lightning integration.
- Model parameter definition.
- Training loop definition.
- Training based on confidence score and internal looping.
- Training the image with fill-in-the-middle objective combined with the main transformer cross entropy loss.
- Mask the image sequence for next-token text generation objective.
- DeepSpeed and ZeRO-3 optimization for parameter offloading to CPU and NVMe
"""

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
model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, CONTEXT_SIZE, IMG_SIZE, PATCH_SIZE, VIT_LAYERS, NUM_GROUPS, USE_FLASH_ATTENTION)

# Load model weights from 4 files
load_model_weights(model, "model_weights", num_files=4)
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

# Define the Trainer with DeepSpeed and ZeRO-3
trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    gpus=1,  # Use GPU if available
    strategy=DeepSpeedStrategy(
        stage=3,  # Use ZeRO-3
        offload_optimizer=True,  # Offload optimizer states to CPU
        offload_parameters=True,  # Offload model parameters to CPU
        nvme_offload_dir="/path/to/nvme",  # Path to NVMe storage for offloading
    ),
    callbacks=[ModelCheckpoint(monitor='train_loss')]
)

# Train the model
trainer.fit(lightning_model, train_dataloaders=train_dataloader())

# Save model weights divided into 4 files
save_model_weights(model, "model_weights", num_files=4)
print("Model weights saved.")

print("Training completed.")
