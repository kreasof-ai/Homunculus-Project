import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from main import TransformerModel
from tokenizer import Tokenizer
from saveModel import load_model_weights, save_model_weights
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from bitnet import replace_linears_in_pytorch_model

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
BATCH_SIZE = 4

"""
This is the scripts for BitNet LoRA finetuning.
"""

# Load tokenizer
tokenizer = Tokenizer.from_file("bpe_tokenizer_autoregressive.json")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a dataset with both text and images
class TextImageDataset(Dataset):
    def __init__(self, data, tokenizer, transform):
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, image_path = self.data[idx]
        encoded = self.tokenizer.encode(text)
        input_ids = torch.tensor(encoded.ids)
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        return input_ids, image

# PyTorch Lightning Module
class BitNetLightningModule(pl.LightningModule):
    def __init__(self, base_model, lora_config):
        super().__init__()
        self.base_model = base_model
        self.model = get_peft_model(self.base_model, lora_config)
        self.criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.MSELoss()
        
        # Replace all linear layers with BitLinear
        replace_linears_in_pytorch_model(self.model)

    def forward(self, input_ids, imgs):
        return self.model(input_ids, imgs=imgs, use_cache=True, middle_training=True)

    def training_step(self, batch, batch_idx):
        input_ids, images = batch
        target = input_ids[:, 1:].contiguous()
        
        num_iterations = BASE_ITERATIONS
        outputs, confidence, vit_loss = self(input_ids[:, :-1], imgs=images, num_iterations=num_iterations)
        
        loss = self.criterion(outputs.view(-1, VOCAB_SIZE), target.view(-1))
        total_loss = loss + vit_loss
        
        confidence_target = torch.clamp(1 - (total_loss.detach() / LOSS_THRESHOLD), 0, 1)
        confidence_loss = self.confidence_criterion(confidence, confidence_target)
        
        total_loss += confidence_loss

        while confidence.mean().item() < CONFIDENCE_THRESHOLD and num_iterations < MAX_ITERATIONS:
            num_iterations += 1
            outputs, confidence, vit_loss = self(input_ids[:, :-1], imgs=images, num_iterations=num_iterations)
            
            loss = self.criterion(outputs.view(-1, VOCAB_SIZE), target.view(-1))
            iter_total_loss = loss + vit_loss
            
            confidence_target = torch.clamp(1 - (iter_total_loss.detach() / LOSS_THRESHOLD), 0, 1)
            confidence_loss = self.confidence_criterion(confidence, confidence_target)
            
            iter_total_loss += confidence_loss
            total_loss += iter_total_loss

        self.log('train_loss', total_loss)
        self.log('confidence', confidence.mean())
        self.log('num_iterations', num_iterations)
        return total_loss

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)

# Main training function
def train_model():
    # Create the base model
    base_model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, CONTEXT_SIZE, IMG_SIZE, PATCH_SIZE, VIT_LAYERS, NUM_GROUPS)

    # Load pre-trained weights
    load_model_weights(base_model, "model_weights.safetensors")

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "main_transformer.layers.*.attention.query",
            "main_transformer.layers.*.attention.key",
            "main_transformer.layers.*.attention.value",
            "main_transformer.layers.*.attention.out",
            "vit.layers.*.attention.query",
            "vit.layers.*.attention.key",
            "vit.layers.*.attention.value",
            "vit.layers.*.attention.out",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Create Lightning module
    model = BitNetLightningModule(base_model, lora_config)

    # Sample data (replace with your dataset)
    data = [
        ("This is a sample text with an image [IMG]", "path/to/image1.jpg"),
        ("Another example of text and image [IMG] data.", "path/to/image2.jpg"),
        # Add more text-image pairs...
    ]

    # Create dataset and dataloader
    dataset = TextImageDataset(data, tokenizer, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Define DeepSpeed config
    deepspeed_config = {
        "train_batch_size": BATCH_SIZE,
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        }
    }

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='bitnet-model-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        monitor='train_loss'
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback],
        strategy=DeepSpeedStrategy(config=deepspeed_config),
        precision=16,  # Use mixed precision
        accelerator="gpu",
        devices=1  # Adjust based on your GPU setup
    )

    # Train the model
    trainer.fit(model, dataloader)

    # Save the fine-tuned BitNet weights
    model.model.save_pretrained("bitnet_weights")

    print("Fine-tuning completed. BitNet weights saved.")

    # Merge the LoRA weights with the base model (if needed)
    merged_model = model.model.merge_and_unload()

    # Save the merged model
    save_model_weights(merged_model, "merged_bitnet_weights.safetensors")

    print("BitNet weights merged with base model and saved.")

if __name__ == "__main__":
    train_model()
