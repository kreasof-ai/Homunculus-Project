# Experimental Custom Transformer Architecture
By [Habibullah Akbar](https://chavyv.vercel.app).

Key features:
- Seamless integration with vision encoder. Along with selective RoPE for each image and text embedding sequence.
- Internal iteration, making deeper abstraction while keeping the same parameter count.
- GeGLU activation function, inspired by [Gemma 2 models](https://blog.google/technology/developers/google-gemma-2/).
- Custom KV-caching, making sure each internal iterations have an independent KV-cache.
- Grouped Query Attention.
- PyTorch Lightning implementation.
- DeepSpeed and ZeRO-3 integration. Automatically offload the memory overflow into CPU and NVMe.
- Finetuning scripts example with LoRA adapters, with and without quantization.
- Add BitNet implementation.
- Flash Attention implementation.

![Internal latent loop (9)](https://github.com/user-attachments/assets/fe74e8b8-2f74-4b20-9f36-6f61c6946f2a)

The iterable Transformer model, where the model can *rethink* its internal cognitive process with an internal confidence score as a guide. Akin of slow thinking mechanism.
So this is the simple explanation of how it works:
- We put an adjustable parameter to handle internal looping, the default value is 1.
- If the loss value is high, this iteration is triggered, with max iterations set to 10.
- We train an independent layer to output a confidence score, trained by loss value from the main training process.
- When inference, both the next token and confidence scores are outputted and can determine how many iterations are needed for the current inference.
- ~~No sophisticated tokenization or attention layer, just a pure simple transformer for learning purposes.~~
- I'm adding GeGLU activation function, BPE tokenizer, selective 1D & 2D RoPE, safetensors, custom KV-caching, a simple vision encoder, grouped-query attention (GQA), RMS Norm, PyTorch Lightning, DeepSpeed, and ZeRO-3.

> Notes: ~~I dunno why I'm impulsively adding unnecessary parts like ViT 🙃~~ I decided to put all of my ideas into this project, so this is probably not a simple learning project anymore 😅

YouTube progress documentation playlist:
- First short brief (27 July 2024): [https://youtu.be/NjK1BJyhrlI](https://youtu.be/NjK1BJyhrlI)

Soon:
- [Infini-attention](https://arxiv.org/abs/2404.07143) integration.
- Speech Encoder integration, possibly Whisper-like architecture.
- 3D RoPE for continuous vision input (video).
- ~~Flash Attention integration.~~ ✔️
- Diffusion Transformer (DiT) integration for image detokenization.
- Speech generation integration.
- Influential token extraction.
- [Discrete Latent Representation](https://arxiv.org/abs/2312.01203)
