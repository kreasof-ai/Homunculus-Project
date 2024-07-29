from tokenizers import Tokenizer, models, pre_tokenizers, trainers

"""
This is the code to generate the custom tokenizer. Using Byte Pair Encoding
"""

def train_bpe_tokenizer(files, vocab_size=128000):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[IMG]", "[/IMG]"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train(files, trainer)
    return tokenizer

# Train and save the tokenizer
tokenizer = train_bpe_tokenizer(["../data/tokenizer_text.txt"])
tokenizer.save("../output/bpe_tokenizer_autoregressive.json")