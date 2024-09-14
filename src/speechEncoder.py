# speechEncoder.py

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class SpeechEncoder(nn.Module):
    def __init__(self, pretrained_model='facebook/wav2vec2-base-960h', embed_size=768):
        super(SpeechEncoder, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model)
        self.proj = nn.Linear(self.wav2vec.config.hidden_size, embed_size)

    def forward(self, speech_input, attention_mask=None):
        """
        Args:
            speech_input: Tensor of shape (batch_size, num_audio_samples)
            attention_mask: Optional Tensor for masking
        Returns:
            Tensor of shape (batch_size, seq_len, embed_size)
        """
        outputs = self.wav2vec(speech_input, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        projected = self.proj(hidden_states)  # (batch_size, seq_len, embed_size)
        return projected
