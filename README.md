![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

[![Follow me on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-md.svg)](https://huggingface.co/ChavyvAkvar)

# Homunculus Project - Experimental Custom Transformer Architecture
By [Habibullah Akbar](https://chavyv.vercel.app).

Key features:
- Seamless integration with vision encoder. Along with selective RoPE for each image and text embedding sequence.
- Internal iteration, making deeper abstraction while keeping the same parameter count.
- GeGLU activation function, inspired by [Gemma 2 models](https://blog.google/technology/developers/google-gemma-2/).
- Custom KV-caching, making sure each internal iterations have an independent KV-cache.
- BPE tokenizer based on KBBI.
- Grouped Query Attention.
- PyTorch Lightning implementation.
- DeepSpeed and ZeRO-3 integration. Automatically offload the memory overflow into CPU and NVMe.
- Finetuning scripts example with LoRA adapters, with and without quantization.
- Add BitNet implementation.
- Flash Attention implementation.
- Jupyter notebook example, both for training and finetuning.

![Internal latent loop (9)](https://github.com/user-attachments/assets/fe74e8b8-2f74-4b20-9f36-6f61c6946f2a)

The iterable Transformer model, where the model can *rethink* its internal cognitive process with an internal confidence score as a guide. Akin of slow thinking mechanism.
So this is the simple explanation of how it works:
- We put an adjustable parameter to handle internal looping, the default value is 1.
- If the loss value is high, this iteration is triggered, with max iterations set to 10.
- We train an independent layer to output a confidence score, trained by loss value from the main training process.
- When inference, both the next token and confidence scores are outputted and can determine how many iterations are needed for the current inference.

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
- [Discrete Latent Representation](https://arxiv.org/abs/2312.01203).
- HuggingFace Hub integration (dataset and upload models).
- xFormers.
- Mamba2 block (?).
- Kolmogorov Arnold Network (KAN).
- Mixture of Experts block.
- Fast object detection integration, possibly YOLO or RT-DETR.
- OCR model integration.
- [MIinference](https://github.com/microsoft/MInference).
- Pre-train model integration, possibly Gemma 2 since it uses the same activation function.

> UPDATE LICENSE:
***This software is dual-licensed under the terms of the GNU Affero General Public License (AGPL) and a commercial license. For commercial use, please contact Habibullah Akbar at akbar2habibullah.gmail to obtain a commercial license. Commercial use is defined as any use of the software for financial gain, including but not limited to, selling, licensing, or distributing the software as part of a product or service.***

