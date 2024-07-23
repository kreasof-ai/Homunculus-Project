
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