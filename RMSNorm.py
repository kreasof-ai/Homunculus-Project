import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, embed_size, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.embed_size = embed_size
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_size))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms * self.scale
        return x
