# Internal Latent Loop Transformer
(Multiple internal inference loop transformer model)
By [Habibullah Akbar](https://chavyv.vercel.app)

![Internal latent loop (4)](https://github.com/user-attachments/assets/1756d087-8659-4709-bdad-532633c12a5f)
(The image above is a bit misleading because Rotary Positional Encoding happens at each block, not only at the first gate)

This is a simple implementation of the iterable Transformer model, where the model can *rethink* its internal cognitive process with an internal confidence score as a guide. Akin of slow thinking mechanism.
So this is the simple explanation of how it works:
- We put an adjustable parameter to handle internal looping, the default value is 1.
- If the loss value is high, this iteration is triggered, with max iterations set to 10.
- We train an independent layer to output a confidence score, trained by loss value from the main training process.
- When inference, both the next token and confidence scores are outputted and can determine how many iterations are needed for the current inference.
- ~~No sophisticated tokenization or attention layer, just a pure simple transformer for learning purposes.~~ I'm adding BPE, RoPE, safetensors, custom KV-caching, and a simple vision encoder.
