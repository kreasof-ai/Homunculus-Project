# inference.py

import torch
from main import TransformerModel
from tokenizer import Tokenizer
from saveModel import load_model_weights

def main():
    # Define constants as per training
    VOCAB_SIZE = 128000
    EMBED_SIZE = 8192
    NUM_HEADS = 64
    NUM_LAYERS = 80
    CONTEXT_SIZE = 128000
    IMG_SIZE = 1024
    PATCH_SIZE = 16
    VIT_LAYERS = 16
    NUM_GROUPS = 8
    USE_FLASH_ATTENTION = False

    # Initialize model
    model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, CONTEXT_SIZE, IMG_SIZE, PATCH_SIZE, VIT_LAYERS, NUM_GROUPS, USE_FLASH_ATTENTION)
    load_model_weights(model, "merged_bitnet_weights", num_files=4)
    model.eval()

    # Load tokenizer
    tokenizer = Tokenizer.from_file("../output/bpe_tokenizer_autoregressive.json")

    # Example input
    text = "Generate an image of a sunset [IMG] ... [/IMG] and play the calming sounds [SPEECH] ... [/SPEECH]"
    input_ids = torch.tensor([tokenizer.encode(text).ids])

    # Example image and speech inputs
    image = torch.randn(1, 3, 224, 224)  # Placeholder image tensor
    speech = torch.randn(1, 16000)  # Placeholder speech tensor (1 sec at 16kHz)

    with torch.no_grad():
        output, confidence, past_outputs, img_pos, end_img_pos, speech_pos, end_speech_pos, vit_loss, speech_loss, attention_weights = model(
            input_ids, 
            imgs=[image], 
            speech=[speech], 
            num_iterations=1, 
            use_cache=False, 
            middle_training=False
        )

    # Detokenize images
    reconstructed_images = model.detokenize_images(output)
    print("Reconstructed Images Shape:", reconstructed_images.shape)

    # Extract influential tokens
    influential_tokens = model.get_influential_tokens(attention_weights, top_k=1)
    print("Influential Tokens per Layer:", influential_tokens)

if __name__ == "__main__":
    main()